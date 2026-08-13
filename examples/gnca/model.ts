"use tsover"

// Ported from ~/code/graph-cellular-automata/src/gnca/model.py.

import { cat, Linear, Module, Tensor, uniform } from "../../index.ts"
import type { DimAdd, DimMul, Shape } from "../../src/shape.ts"
import { fromFlat } from "../../src/tensor.ts"
import { type Edges, inDegrees } from "./graphs.ts"

/**
 * Channels the rule perceives: its own state, the neighbour mean, the
 * gated difference mean, and one degree scalar.
 */
export type Percept<C extends number> = DimAdd<
  DimAdd<DimAdd<C, C>, C>,
  1
>

export type GateIn<C extends number> = DimMul<3, C>

export interface GraphTensors<
  N extends number,
  E extends number = number,
> {
  readonly src: Tensor<[E]>
  readonly dst: Tensor<[E]>
  readonly nodes: N
  readonly invDegree: Tensor<[N, 1]>
  readonly logDegree: Tensor<[N, 1]>
}

export function graphTensors<
  N extends number,
  E extends number,
>(edges: Edges<E>, nodes: N): GraphTensors<N, E> {
  const degree = inDegrees(edges, nodes)
  const invDegree = new Float32Array(nodes)
  const logDegree = new Float32Array(nodes)
  for (let i = 0; i < nodes; i++) {
    invDegree[i] = 1 / Math.max(degree[i]!, 1)
    logDegree[i] = Math.log1p(degree[i]!)
  }
  return {
    src: fromData<[E]>(edges.src, [edges.count]),
    dst: fromData<[E]>(edges.dst, [edges.count]),
    nodes,
    invDegree: fromData<[N, 1]>(invDegree, [nodes, 1]),
    logDegree: fromData<[N, 1]>(logDegree, [nodes, 1]),
  }
}

/**
 * A tensor of the given shape holding `data`.
 *
 * The shape is asserted because it comes from runtime values — an edge
 * count, a node count — that the type system cannot read off a
 * `number[]`. Everything downstream of here is inferred.
 */
function fromData<S extends Shape>(
  data: Float32Array,
  shape: number[],
): Tensor<S> {
  return fromFlat(data, shape) as unknown as Tensor<S>
}

export class GraphNCA<
  C extends number = 16,
  H extends number = 128,
> extends Module {
  readonly channels: C
  readonly hidden: H
  readonly inner: Linear<Percept<C>, H>
  readonly outer: Linear<H, C>
  readonly gate: Linear<GateIn<C>, C>

  constructor(channels: C = 16 as C, hidden: H = 128 as H) {
    super()
    this.channels = channels
    this.hidden = hidden
    // The widths are right by construction, but TypeScript cannot do
    // arithmetic on a runtime expression, so the derived input dims are
    // stated. Percept<C> and GateIn<C> are that same arithmetic in types.
    this.inner = new Linear(
      3 * channels + 1,
      hidden,
    ) as Linear<Percept<C>, H>
    this.outer = new Linear(hidden, channels, {
      bias: false,
    }) // Zero the last layer: the initial rule is the identity, so growth
     // starts from a standing seed rather than from noise.
    ;(this.outer.weight.data as Float32Array).fill(0)
    this.gate = new Linear(
      3 * channels,
      channels,
    ) as Linear<GateIn<C>, C> // Zero the gate too. 2·sigmoid(0) = 1 exactly, so at init the gate is
     // plain identity diffusion and a warm start is exact.
    ;(this.gate.weight.data as Float32Array).fill(0)
    ;(this.gate.bias!.data as Float32Array).fill(0)
  }

  forward<N extends number, E extends number>(
    x: Tensor<[N, C]>,
    // NoInfer pins N to the state, so the edge list is *checked* against
    // it rather than being a second place N could come from. Handing a
    // batched state the un-batched graph is the mistake this catches, and
    // it is a silent wrong answer at runtime.
    graph: GraphTensors<NoInfer<N>, E>,
    updateRate = 0.5,
  ): Tensor<[N, C]> {
    const c = this.channels
    const { src, dst, nodes, invDegree } = graph

    const fromNode = x.indexSelect(src)
    const toNode = x.indexSelect(dst)
    const difference = fromNode.sub(toNode)

    const aggregate = (
      messages: Tensor<[E, C]>,
    ): Tensor<[N, C]> => messages.scatterAdd(dst, nodes).mul(invDegree)

    const weight = this.gate.weight
    const endpoints = x.matmul(
      cat(weight.narrow(0, 0, c), weight.narrow(0, c, c), 1),
    )
    // The only shape written down in this method, and it is a check
    // rather than a cast: broadcasting the bias over the edge tensor leaves
    // TypeScript holding `Broadcast<..., BroadcastDim<C, C>>`, which is
    // equal to [E, C] and assignable to it, but is not *syntactically* it —
    // so it no longer unifies with a generic parameter downstream. Naming
    // the shape here re-anchors it. Everything else in this method is
    // inferred.
    const gate: Tensor<[E, C]> = endpoints
      .narrow(1, 0, c)
      .indexSelect(src)
      .add(endpoints.narrow(1, c, c).indexSelect(dst))
      .add(
        difference
          .abs()
          .matmul(weight.narrow(0, 2 * c, c))
          .add(this.gate.bias!),
      )
      .sigmoid()
      .mul(2)

    // log1p(degree) goes LAST, so warm-starting from a checkpoint whose
    // perception lacked it is a plain zero-column pad of the first layer.
    const perception = cat(
      cat(
        cat(x, aggregate(fromNode), 1),
        aggregate(gate.mul(difference)),
        1,
      ),
      graph.logDegree,
      1,
    )
    const increment = this.outer.forward(
      this.inner.forward(perception).relu(),
    )

    // Stochastic per-node update: the classic NCA trick for robustness to
    // an asynchronous update order. uniform() is a graph node, so a
    // compiled step redraws this mask on every call.
    const mask = uniform([nodes, 1]).lt(updateRate)
    return x.add(increment.mul(mask))
  }
}

/**
 * A node lives if it or any neighbour has alpha above `threshold` — the
 * graph equivalent of the growing NCA's 3×3 alive mask.
 */
export function aliveMask<
  N extends number,
  E extends number,
  C extends number,
>(
  x: Tensor<[N, C]>,
  graph: GraphTensors<NoInfer<N>, E>,
  threshold = 0.1,
): Tensor<[N, 1]> {
  const live = x.narrow(1, 3, 1).gt(threshold)
  const liveNeighbours = live
    .indexSelect(graph.src)
    .scatterAdd(graph.dst, graph.nodes)
  return live.add(liveNeighbours).gt(0)
}

/**
 * All-empty node states except one seed node, whose alpha and hidden
 * channels start at 1. Flat [batch · nodes, channels], the layout the rule
 * takes.
 */
export function seedState(
  batch: number,
  nodes: number,
  channels: number,
  center: number,
): Float32Array {
  const data = new Float32Array(batch * nodes * channels)
  for (let b = 0; b < batch; b++) {
    // channels 0-2 are RGB, 3 is alpha, 4 and up are hidden
    const at = (b * nodes + center) * channels
    for (let c = 3; c < channels; c++) data[at + c] = 1
  }
  return data
}
