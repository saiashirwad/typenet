// Ways to break a grown pattern, shared by training and evaluation. Each
// function returns the node indices to wipe (or an edge keep-mask), and
// nothing else decides what "a ball" or "a band" means — otherwise the
// evaluation only measures what training optimized by luck.
//
// Ported from ~/code/graph-cellular-automata/src/gnca/damage.py.

import { type Edges, type Points, point } from "./graphs.ts"
import { type Rng, quantile, rng } from "./rng.ts"
import type { Target } from "./targets.ts"

/**
 * Indices of the nodes the target actually paints. Damage outside the
 * pattern is a no-op the rule already handles, so every shape below is
 * defined relative to these.
 */
export function patternNodes(
  target: Target,
  threshold = 0.5
): Int32Array {
  const found: number[] = []
  for (let i = 0; i * 4 + 3 < target.length; i++)
    if (target[i * 4 + 3]! > threshold) found.push(i)
  return Int32Array.from(found)
}

/**
 * A blob around one node covering `frac` of the pattern. The radius is a
 * quantile of the squared distances to pattern nodes, so `frac` means
 * the same thing however the nodes happen to be spread out.
 */
export function ball(
  pos: Points,
  pattern: Int32Array,
  options: { frac?: number; center?: number; random?: Rng } = {}
): Int32Array {
  const { frac = 0.25, random = rng(0) } = options
  const center =
    options.center ?? pattern[random.int(pattern.length)]!
  const d2 = new Float64Array(pos.n)
  for (let i = 0; i < pos.n; i++) {
    let total = 0
    for (let d = 0; d < pos.dim; d++) {
      const delta = point(pos, i, d) - point(pos, center, d)
      total += delta * delta
    }
    d2[i] = total
  }
  const radius = quantile(
    Array.from(pattern, i => d2[i]!),
    frac
  )
  const hit: number[] = []
  for (let i = 0; i < pos.n; i++)
    if (d2[i]! <= radius) hit.push(i)
  return Int32Array.from(hit)
}

/** A random fraction of the pattern nodes: damage with no shape at all. */
export function scatter(
  pattern: Int32Array,
  options: { frac?: number; random?: Rng } = {}
): Int32Array {
  const { frac = 0.25, random = rng(0) } = options
  const count = Math.max(1, Math.floor(frac * pattern.length))
  return Int32Array.from(random.sample(pattern, count))
}

/** A horizontal slice through the middle of the pattern. */
export function band(
  pos: Points,
  pattern: Int32Array,
  height = 0.16
): Int32Array {
  let sum = 0
  for (const i of pattern) sum += point(pos, i, 1)
  const cy = sum / pattern.length
  const hit: number[] = []
  for (let i = 0; i < pos.n; i++)
    if (Math.abs(point(pos, i, 1) - cy) < height / 2)
      hit.push(i)
  return Int32Array.from(hit)
}

/** Everything on the far side of the pattern's centre; axis 0 is right. */
export function half(
  pos: Points,
  pattern: Int32Array,
  axis = 0
): Int32Array {
  let sum = 0
  for (const i of pattern) sum += point(pos, i, axis)
  const c = sum / pattern.length
  return Int32Array.from(
    Array.from(pattern).filter(i => point(pos, i, axis) > c)
  )
}

// Cutting edges instead of nodes: the graph-native damage a grid cannot
// have. Both return a keep-mask over the edge list.

/** Drop every edge that crosses the horizontal line `y`. */
export function cutAcross(
  pos: Points,
  edges: Edges,
  y: number
): Uint8Array {
  const keep = new Uint8Array(edges.count)
  for (let i = 0; i < edges.count; i++)
    keep[i] =
      point(pos, edges.src[i]!, 1) < y ===
      point(pos, edges.dst[i]!, 1) < y
        ? 1
        : 0
  return keep
}

/** Drop a random fraction of all edges. */
export function cutRandom(
  edges: Edges,
  frac: number,
  random: Rng = rng(0)
): Uint8Array {
  const keep = new Uint8Array(edges.count)
  for (let i = 0; i < edges.count; i++)
    keep[i] = random.next() >= frac ? 1 : 0
  return keep
}

/** Apply an edge keep-mask, producing a smaller edge list. */
export function keepEdges(
  edges: Edges,
  keep: Uint8Array
): Edges {
  const kept: number[] = []
  for (let i = 0; i < edges.count; i++)
    if (keep[i]) kept.push(i)
  const src = new Float32Array(kept.length)
  const dst = new Float32Array(kept.length)
  kept.forEach((e, i) => {
    src[i] = edges.src[e]!
    dst[i] = edges.dst[e]!
  })
  return { src, dst, count: kept.length }
}
