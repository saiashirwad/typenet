/** mulberry32: small, fast, good enough for sampling, and seedable. */
export function rng(seed: number): Rng {
  let a = seed >>> 0
  const next = (): number => {
    a = (a + 0x6d2b79f5) >>> 0
    let t = a
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
  return {
    next,
    int: (n: number) => Math.floor(next() * n),
    range: (lo: number, hi: number) => lo + next() * (hi - lo),
    intRange: (lo: number, hi: number) => lo + Math.floor(next() * (hi - lo + 1)),
    sample: (values: ArrayLike<number>, count: number) => {
      const pool = Array.from(values)
      const take = Math.min(count, pool.length)
      for (let i = 0; i < take; i++) {
        const j = i + Math.floor(next() * (pool.length - i))
        ;[pool[i], pool[j]] = [pool[j]!, pool[i]!]
      }
      return pool.slice(0, take)
    },
  }
}

export interface Rng {
  next(): number
  int(n: number): number
  range(lo: number, hi: number): number
  intRange(lo: number, hi: number): number
  sample(values: ArrayLike<number>, count: number): number[]
}

/** The `q`th quantile of `values`, linearly interpolated (numpy's default). */
export function quantile(
  values: ArrayLike<number>,
  q: number,
): number {
  const sorted = Array.from(values).sort((a, b) => a - b)
  if (sorted.length === 0) return NaN
  const pos = q * (sorted.length - 1)
  const lo = Math.floor(pos)
  const hi = Math.min(lo + 1, sorted.length - 1)
  return (
    sorted[lo]! + (pos - lo) * (sorted[hi]! - sorted[lo]!)
  )
}
