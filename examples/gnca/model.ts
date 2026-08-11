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
import { type Edges, inDegrees } from "./graphs.ts"

type AnyTensor = Tensor<any, any>

/**
 * Everything about the graph the rule needs, as tensors. The graph never
 * changes during a run, so the degree terms are computed once here
 * rather than per step inside the rollout.
 */
export interface GraphTensors {
  /** Source node of each edge, shape [E]. */
  readonly src: AnyTensor
  /** Destination node of each edge, shape [E]. */
  readonly dst: AnyTensor
  /** Node count these indices address. */
  readonly nodes: number
  /** 1 / max(in-degree, 1), shape [nodes, 1]. */
  readonly invDegree: AnyTensor
  /** log(1 + in-degree), shape [nodes, 1]. */
  readonly logDegree: AnyTensor
}

/** Build the rule's view of an edge list. */
export function graphTensors(
  edges: Edges,
  nodes: number
): GraphTensors {
  const degree = inDegrees(edges, nodes)
  const invDegree = new Float32Array(nodes)
  const logDegree = new Float32Array(nodes)
  for (let i = 0; i < nodes; i++) {
    invDegree[i] = 1 / Math.max(degree[i]!, 1)
    logDegree[i] = Math.log1p(degree[i]!)
  }
  return {
    src: fromData(edges.src, [edges.count]),
    dst: fromData(edges.dst, [edges.count]),
    nodes,
    invDegree: fromData(invDegree, [nodes, 1]),
    logDegree: fromData(logDegree, [nodes, 1])
  }
}

function fromData(
  data: Float32Array,
  shape: number[]
): AnyTensor {
  const t = Tensor.zeros(shape) as AnyTensor
  ;(t.data as Float32Array).set(data)
  return t
}

/** Concatenate several tensors along `dim`; typenet's cat is binary. */
function concat(
  parts: AnyTensor[],
  dim: number
): AnyTensor {
  return parts.reduce(
    (a, b) => cat(a, b, dim as any) as AnyTensor
  )
}

export class GraphNCA extends Module {
  readonly channels: number
  readonly hidden: number
  /** Perception to hidden. Input is three channel blocks plus the degree. */
  readonly inner: Linear<number, number>
  /** Hidden to the state increment. Zero-init, so the rule starts as identity. */
  readonly outer: Linear<number, number>
  /** Per-edge, per-channel diffusion conductivity. */
  readonly gate: Linear<number, number>

  constructor(channels = 16, hidden = 128) {
    super()
    this.channels = channels
    this.hidden = hidden
    this.inner = new Linear(3 * channels + 1, hidden)
    this.outer = new Linear(hidden, channels, {
      bias: false
    })
    // Zero the last layer: the initial rule is the identity, so growth
    // starts from a standing seed rather than from noise.
    ;(this.outer.weight.data as Float32Array).fill(0)
    this.gate = new Linear(3 * channels, channels)
    // Zero the gate too. 2·sigmoid(0) = 1 exactly, so at init the gate
    // is plain identity diffusion and a warm start is exact.
    ;(this.gate.weight.data as Float32Array).fill(0)
    ;(this.gate.bias!.data as Float32Array).fill(0)
  }

  /**
   * One step of the automaton. `x` is [nodes, channels]; for a batch the
   * copies are stacked into the node dimension and `graph` carries the
   * correspondingly offset edge list.
   */
  forward(
    x: AnyTensor,
    graph: GraphTensors,
    updateRate = 0.5
  ): AnyTensor {
    const c = this.channels
    const { src, dst, nodes, invDegree } = graph

    // Gather each edge's endpoints once; both terms below need them.
    const fromNode = x.indexSelect(src)
    const toNode = x.indexSelect(dst)
    const meanNeighbour = fromNode
      .scatterAdd(dst, nodes)
      .mul(invDegree)
    const difference = fromNode.sub(toNode)

    // The gate is linear in [x_src, x_dst, |x_src - x_dst|], so its two
    // endpoint terms are one matmul over nodes, then gathered per edge.
    // As a single (E, 3C) matmul over edges it would cost several times
    // as much — there are many more edges than nodes.
    const weight = this.gate.weight as AnyTensor
    const endpoints = x.matmul(
      cat(weight.narrow(0, 0, c), weight.narrow(0, c, c), 1)
    )
    const gate = endpoints
      .narrow(1, 0, c)
      .indexSelect(src)
      .add(endpoints.narrow(1, c, c).indexSelect(dst))
      .add(
        difference
          .abs()
          .matmul(weight.narrow(0, 2 * c, c))
          .add(this.gate.bias as AnyTensor)
      )
      .sigmoid()
      .mul(2)
    const meanDifference = gate
      .mul(difference)
      .scatterAdd(dst, nodes)
      .mul(invDegree)

    // log1p(degree) goes LAST, so warm-starting from a checkpoint whose
    // perception lacked it is a plain zero-column pad of the first layer.
    const perception = concat(
      [x, meanNeighbour, meanDifference, graph.logDegree],
      1
    )
    const increment = this.outer.forward(
      this.inner.forward(perception).relu() as any
    ) as AnyTensor

    // Stochastic per-node update: the classic NCA trick for robustness
    // to an asynchronous update order. uniform() is a graph node, so a
    // compiled step redraws this mask on every call.
    const mask = uniform([nodes, 1] as [number, number]).lt(
      updateRate
    ) as AnyTensor
    return x.add(increment.mul(mask))
  }
}

/**
 * A node lives if it or any neighbour has alpha above `threshold` — the
 * graph equivalent of the growing NCA's 3×3 alive mask.
 *
 * The reference takes a max of alpha over neighbours and compares. This
 * counts live neighbours instead: the same predicate ("any neighbour
 * above threshold") through a scatter-add rather than a scatter-max.
 */
export function aliveMask(
  x: AnyTensor,
  graph: GraphTensors,
  threshold = 0.1
): AnyTensor {
  const live = x.narrow(1, 3, 1).gt(threshold)
  const liveNeighbours = live
    .indexSelect(graph.src)
    .scatterAdd(graph.dst, graph.nodes)
  return live.add(liveNeighbours).gt(0)
}

/**
 * All-empty node states except one seed node, whose alpha and hidden
 * channels start at 1. Flat [batch · nodes, channels], the layout the
 * rule takes.
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
