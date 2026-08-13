// Ported from ~/code/graph-cellular-automata/src/gnca/graphs.py.

import { rng } from "./rng.ts"

/** Node positions as a flat row-major (n × dim) buffer. */
export interface Points {
  readonly data: Float32Array
  readonly n: number
  readonly dim: number
}

/**
 * A directed edge list, both directions present. `src[i] -> dst[i]`.
 * Held as float32 because that is what typenet index tensors are.
 */
export interface Edges<E extends number = number> {
  readonly src: Float32Array
  readonly dst: Float32Array
  readonly count: E
}

export function point(
  pos: Points,
  i: number,
  d: number,
): number {
  return pos.data[i * pos.dim + d]!
}

function squaredDistance(
  pos: Points,
  i: number,
  j: number,
): number {
  let total = 0
  for (let d = 0; d < pos.dim; d++) {
    const delta = point(pos, i, d) - point(pos, j, d)
    total += delta * delta
  }
  return total
}

export function knnGraph(pos: Points, k = 8): Edges {
  const n = pos.n
  const neighbors = Math.min(k, n - 1)
  const pairs = new Set<number>()
  const best = new Float64Array(neighbors)
  const bestAt = new Int32Array(neighbors)
  for (let i = 0; i < n; i++) {
    best.fill(Infinity)
    bestAt.fill(-1)
    for (let j = 0; j < n; j++) {
      if (j === i) continue
      const d2 = squaredDistance(pos, i, j)
      if (d2 >= best[neighbors - 1]!) continue
      let slot = neighbors - 1
      while (slot > 0 && best[slot - 1]! > d2) {
        best[slot] = best[slot - 1]!
        bestAt[slot] = bestAt[slot - 1]!
        slot--
      }
      best[slot] = d2
      bestAt[slot] = j
    }
    for (const j of bestAt) {
      if (j < 0) continue
      pairs.add(Math.min(i, j) * n + Math.max(i, j))
    }
  }
  return fromPairs(pairs, n)
}

/**
 * Expand undirected pairs into a bidirectional edge list, ordered by
 * source then destination. Order does not change the maths — aggregation
 * is a scatter-add — but a canonical one makes two builds of the same
 * graph comparable, including against the reference implementation.
 */
function fromPairs(pairs: Set<number>, n: number): Edges {
  const both: number[] = []
  for (const key of pairs) {
    const a = Math.floor(key / n)
    const b = key % n
    both.push(a * n + b, b * n + a)
  }
  both.sort((x, y) => x - y)
  const src = new Float32Array(both.length)
  const dst = new Float32Array(both.length)
  both.forEach((key, i) => {
    src[i] = Math.floor(key / n)
    dst[i] = key % n
  })
  return { src, dst, count: both.length }
}

export function randomGeometricGraph(options: {
  nodes?: number
  k?: number
  seed?: number
  dim?: number
}): { pos: Points; edges: Edges } {
  const { nodes = 1024, k = 8, seed = 0, dim = 2 } = options
  const random = rng(seed)
  const data = new Float32Array(nodes * dim)
  for (let i = 0; i < data.length; i++) {
    data[i] = random.next()
  }
  const pos: Points = { data, n: nodes, dim }
  return { pos, edges: knnGraph(pos, k) }
}

export function wattsStrogatzGraph(options: {
  nodes?: number
  k?: number
  beta?: number
  seed?: number
}): { pos: Points; edges: Edges } {
  const {
    nodes = 1024,
    k = 8,
    beta = 0.05,
    seed = 0,
  } = options
  if (k % 2 !== 0) {
    throw new Error(
      `wattsStrogatzGraph: k must be even, got ${k}`,
    )
  }
  const random = rng(seed)
  const pairs = new Set<number>()
  for (let i = 0; i < nodes; i++) {
    for (let d = 1; d <= k / 2; d++) {
      let j = (i + d) % nodes
      if (random.next() < beta) {
        j = random.int(nodes - 1)
        if (j >= i) j++
      }
      if (i === j) continue
      pairs.add(Math.min(i, j) * nodes + Math.max(i, j))
    }
  }
  const data = new Float32Array(nodes * 2)
  for (let i = 0; i < nodes; i++) {
    const theta = (i / nodes) * 2 * Math.PI
    data[i * 2] = (Math.cos(theta) + 1) / 2
    data[i * 2 + 1] = (Math.sin(theta) + 1) / 2
  }
  return {
    pos: { data, n: nodes, dim: 2 },
    edges: fromPairs(pairs, nodes),
  }
}

export function batchEdges(
  edges: Edges,
  batch: number,
  nodes: number,
): Edges {
  const count = edges.count * batch
  const src = new Float32Array(count)
  const dst = new Float32Array(count)
  for (let b = 0; b < batch; b++) {
    const offset = b * nodes
    const at = b * edges.count
    for (let i = 0; i < edges.count; i++) {
      src[at + i] = edges.src[i]! + offset
      dst[at + i] = edges.dst[i]! + offset
    }
  }
  return { src, dst, count }
}

export function inDegrees(
  edges: Edges,
  nodes: number,
): Float32Array {
  const degrees = new Float32Array(nodes)
  for (let i = 0; i < edges.count; i++) {
    degrees[edges.dst[i]!]!++
  }
  return degrees
}

export function nearestNode(
  pos: Points,
  at: readonly number[],
): number {
  let best = 0
  let bestD2 = Infinity
  for (let i = 0; i < pos.n; i++) {
    let d2 = 0
    for (let d = 0; d < pos.dim; d++) {
      const delta = point(pos, i, d) - (at[d] ?? 0)
      d2 += delta * delta
    }
    if (d2 < bestD2) {
      bestD2 = d2
      best = i
    }
  }
  return best
}
