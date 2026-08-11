"use tsover"

// The update rule: one shared MLP applied to every node, Distill-NCA
// style. Ported from ~/code/graph-cellular-automata/src/gnca/model.py.
//
// Perception (message passing): each node sees
//     [ own state, mean of neighbour states, mean of (neighbour - own),
//       log(1 + degree) ]
// The degree scalar restores the neighbour COUNT that mean aggregation
// erases. The difference term is gated per edge and per channel
// (Perona-Malik style): the rule can turn diffusion OFF across pattern
// boundaries instead of smearing them.
//
// Update: a tiny MLP produces a residual increment, applied to a random
// subset of nodes each step.

import {
  Linear,
  Module,
  Tensor,
  cat,
  uniform
} from "../../index.ts"
import type {
  DimAdd,
  DimMul,
  Shape
} from "../../src/shape.ts"
import { type Edges, inDegrees } from "./graphs.ts"

/**
 * Channels the rule perceives: its own state, the neighbour mean, the
 * gated difference mean, and one degree scalar.
 */
export type Percept<C extends number> = DimAdd<
  DimMul<3, C>,
  1
>

/** The three channel blocks the gate is linear in. */
export type GateIn<C extends number> = DimMul<3, C>

/**
 * Assert a shape the algebra will not derive.
 *
 * Every dim asserted below is right, and the runtime checks it. But
 * composing broadcasts over an *unresolved generic* dim leaves TypeScript
 * holding expressions like `BroadcastDim<C, C>` that it will not simplify
 * back to `C` — so a chain of ops on a `Tensor<[N, C]>` stops matching
 * `[N, C]` even though that is exactly what it is. (Each dim rule resolves
 * on its own; it is the composition that does not.)
 *
 * This appears only inside the generic kernel below. Every public
 * signature in this file is exact, so every *call site* is fully checked —
 * which is the part that catches mistakes.
 */
function shaped<S extends Shape>(
  t: Tensor<any, any>
): Tensor<S> {
  return t as Tensor<S>
}

/**
 * Everything about the graph the rule needs, as tensors. The graph never
 * changes during a run, so the degree terms are computed once here rather
 * than per step inside the rollout.
 *
 * `N` is the node count and `E` the edge count, both carried in the type,
 * so a rule cannot be handed a different graph's edge list and the per-edge
 * tensors inside `forward` stay tied to the state they came from.
 */
export interface GraphTensors<
  N extends number,
  E extends number = number
> {
  /** Source node of each edge. */
  readonly src: Tensor<[E]>
  /** Destination node of each edge. */
  readonly dst: Tensor<[E]>
  /** Node count these indices address. */
  readonly nodes: N
  /** 1 / max(in-degree, 1). */
  readonly invDegree: Tensor<[N, 1]>
  /** log(1 + in-degree). */
  readonly logDegree: Tensor<[N, 1]>
}

/** Build the rule's view of an edge list. */
export function graphTensors<
  N extends number,
  E extends number
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
    logDegree: fromData<[N, 1]>(logDegree, [nodes, 1])
  }
}

/** A tensor of the given shape holding `data`. */
function fromData<S extends Shape>(
  data: Float32Array,
  shape: number[]
): Tensor<S> {
  const t = Tensor.zeros(shape)
  ;(t.data as Float32Array).set(data)
  return shaped<S>(t)
}

/**
 * The update rule, generic over its channel count `C` and hidden width
 * `H`. Construct it with literals — `new GraphNCA(16, 128)` — and the
 * perception width, the gate's three blocks and the state shape are all
 * derived rather than assumed: handing `forward` a state with the wrong
 * channel count, or a different graph's edge list, is a compile error.
 */
export class GraphNCA<
  C extends number = 16,
  H extends number = 128
> extends Module {
  readonly channels: C
  readonly hidden: H
  /** Perception to hidden. Input is three channel blocks plus the degree. */
  readonly inner: Linear<Percept<C>, H>
  /** Hidden to the state increment. Zero-init, so the rule starts as identity. */
  readonly outer: Linear<H, C>
  /** Per-edge, per-channel diffusion conductivity. */
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
      hidden
    ) as Linear<Percept<C>, H>
    this.outer = new Linear(hidden, channels, {
      bias: false
    })
    // Zero the last layer: the initial rule is the identity, so growth
    // starts from a standing seed rather than from noise.
    ;(this.outer.weight.data as Float32Array).fill(0)
    this.gate = new Linear(
      3 * channels,
      channels
    ) as Linear<GateIn<C>, C>
    // Zero the gate too. 2·sigmoid(0) = 1 exactly, so at init the gate is
    // plain identity diffusion and a warm start is exact.
    ;(this.gate.weight.data as Float32Array).fill(0)
    ;(this.gate.bias!.data as Float32Array).fill(0)
  }

  /**
   * One step of the automaton: `[N, C]` state in, `[N, C]` state out.
   *
   * `N` is generic because a batch is B copies of the graph stacked into
   * the node dimension — the same rule runs on one graph or eight side by
   * side, and `graph` has to be the edge list for whichever it is.
   */
  forward<N extends number, E extends number>(
    x: Tensor<[N, C]>,
    // NoInfer pins N to the state, so the edge list is *checked* against
    // it rather than being a second place N could come from. Handing a
    // batched state the un-batched graph is the mistake this catches, and
    // it is a silent wrong answer at runtime.
    graph: GraphTensors<NoInfer<N>, E>,
    updateRate = 0.5
  ): Tensor<[N, C]> {
    const c = this.channels
    const { src, dst, nodes, invDegree } = graph

    // Gather each edge's endpoints once; both terms below need them.
    const fromNode: Tensor<[E, C]> = x.indexSelect(src)
    const toNode: Tensor<[E, C]> = x.indexSelect(dst)
    const difference: Tensor<[E, C]> = fromNode.sub(toNode)

    /** Sum per-edge messages onto their destination node, and average. */
    const aggregate = (
      messages: Tensor<[E, C]>
    ): Tensor<[N, C]> =>
      shaped(messages.scatterAdd(dst, nodes).mul(invDegree))

    // The gate is linear in [x_src, x_dst, |x_src - x_dst|], so its two
    // endpoint terms are one matmul over nodes, then gathered per edge. As
    // a single (E, 3C) matmul over edges it would cost several times as
    // much — there are many more edges than nodes.
    const weight = this.gate.weight
    const endpoints = x.matmul(
      cat(weight.narrow(0, 0, c), weight.narrow(0, c, c), 1)
    )
    const gate: Tensor<[E, C]> = shaped(
      endpoints
        .narrow(1, 0, c)
        .indexSelect(src)
        .add(endpoints.narrow(1, c, c).indexSelect(dst))
        .add(
          difference
            .abs()
            .matmul(weight.narrow(0, 2 * c, c))
            .add(this.gate.bias!)
        )
        .sigmoid()
        .mul(2)
    )

    // log1p(degree) goes LAST, so warm-starting from a checkpoint whose
    // perception lacked it is a plain zero-column pad of the first layer.
    const perception: Tensor<[N, Percept<C>]> = shaped(
      cat(
        cat(
          cat(x, aggregate(fromNode), 1),
          aggregate(shaped<[E, C]>(gate.mul(difference))),
          1
        ),
        graph.logDegree,
        1
      )
    )
    const increment: Tensor<[N, C]> = this.outer.forward(
      this.inner.forward(perception).relu()
    )

    // Stochastic per-node update: the classic NCA trick for robustness to
    // an asynchronous update order. uniform() is a graph node, so a
    // compiled step redraws this mask on every call.
    const mask: Tensor<[N, 1]> = shaped(
      uniform([nodes, 1]).lt(updateRate)
    )
    return shaped(x.add(increment.mul(mask)))
  }
}

/**
 * A node lives if it or any neighbour has alpha above `threshold` — the
 * graph equivalent of the growing NCA's 3×3 alive mask.
 *
 * The reference takes a max of alpha over neighbours and compares. This
 * counts live neighbours instead: the same predicate ("any neighbour above
 * threshold") through a scatter-add rather than a scatter-max.
 */
export function aliveMask<
  N extends number,
  E extends number,
  C extends number
>(
  x: Tensor<[N, C]>,
  graph: GraphTensors<NoInfer<N>, E>,
  threshold = 0.1
): Tensor<[N, 1]> {
  const live: Tensor<[N, 1]> = x
    .narrow(1, 3, 1)
    .gt(threshold)
  const liveNeighbours = live
    .indexSelect(graph.src)
    .scatterAdd(graph.dst, graph.nodes)
  return shaped(live.add(liveNeighbours).gt(0))
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
  center: number
): Float32Array {
  const data = new Float32Array(batch * nodes * channels)
  for (let b = 0; b < batch; b++) {
    // channels 0-2 are RGB, 3 is alpha, 4 and up are hidden
    const at = (b * nodes + center) * channels
    for (let c = 3; c < channels; c++) data[at + c] = 1
  }
  return data
}
