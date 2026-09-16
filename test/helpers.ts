import { expect } from "vitest"
import { disableNative, isNativeAvailable, useNative } from "../src/backends/native.ts"
import { configure } from "../src/lazy.ts"
import { type AnyTensor, fromFlat } from "../src/tensor.ts"

export function bothWays<T>(fn: () => T): {
  eager: T
  lazy: T
} {
  configure({ lazy: false })
  const eager = fn()
  configure({ lazy: true })
  const lazy = fn()
  configure({ lazy: false })
  return { eager, lazy }
}

export function expectClose(
  a: AnyTensor,
  b: AnyTensor,
  tol = 1e-4,
): void {
  expect(b.shape).toEqual(a.shape)
  const ad = a.data
  const bd = b.data
  expect(bd.length).toBe(ad.length)
  for (let i = 0; i < ad.length; i++) {
    expect(Math.abs(ad[i]! - bd[i]!)).toBeLessThan(tol)
  }
}

export function allPaths(fn: () => AnyTensor): {
  eager: AnyTensor
  lazy: AnyTensor
  native: AnyTensor | null
} {
  configure({ lazy: false })
  const eager = fn()
  configure({ lazy: true })
  const lazy = fn()
  let native: AnyTensor | null = null
  if (isNativeAvailable()) {
    useNative()
    native = fn()
    native.data // force before disabling
    disableNative()
  }
  configure({ lazy: false })
  return { eager, lazy, native }
}

/** Agreement of the lazy and (when available) native paths against eager. Skips the native leg when the addon is not built. */
export function expectAgree(
  fn: () => AnyTensor,
  tolerance = 1e-5,
): void {
  const { eager, lazy, native } = allPaths(fn)
  for (
    const [label, other] of [
      ["lazy", lazy],
      ["native", native],
    ] as const
  ) {
    if (!other) continue
    expect(other.shape, `${label} shape`).toEqual(
      eager.shape,
    )
    const a = eager.data
    const b = other.data
    for (let i = 0; i < a.length; i++) {
      expect(
        Math.abs(a[i]! - b[i]!),
        `${label} element ${i}: ${b[i]} vs eager ${a[i]}`,
      ).toBeLessThan(tolerance)
    }
  }
}

/** Same as expectAgree, but fails when the native addon is unavailable instead of skipping it. */
export function expectAgreeStrict(
  fn: () => AnyTensor,
  tolerance = 1e-5,
): void {
  const { eager, lazy, native } = allPaths(fn)
  for (
    const [label, other] of [
      ["lazy", lazy],
      ["native", native],
    ] as const
  ) {
    expect(other, `${label} result`).not.toBeNull()
    expect(other!.shape, `${label} shape`).toEqual(
      eager.shape,
    )
    const a = eager.data
    const b = other!.data
    for (let i = 0; i < a.length; i++) {
      expect(
        Math.abs(a[i]! - b[i]!),
        `${label} element ${i}: ${b[i]} vs eager ${a[i]}`,
      ).toBeLessThan(tolerance)
    }
  }
}

export function expectExact(a: AnyTensor, b: AnyTensor): void {
  expect(b.shape).toEqual(a.shape)
  expect(Array.from(b.data)).toEqual(Array.from(a.data))
}

export function mulberry32(seed: number): () => number {
  let a = seed >>> 0
  return () => {
    a = (a + 0x6d2b79f5) >>> 0
    let t = a
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

/** Deterministic spread of values in roughly [-1.6, 1.6]. */
export const sample = (n: number, shape: number[]): AnyTensor =>
  fromFlat(
    Float32Array.from(
      { length: n },
      (_, i) => Math.sin(i * 1.7 + 0.3) * 1.6,
    ),
    shape,
  ) as AnyTensor

/** Exact type equality, for `type _x = Expect<Equal<A, B>>` assertions. */
export type Equal<A, B> = (<T>() => T extends A ? 1 : 2) extends (<T>() => T extends B ? 1 : 2) ? true : false
export type Expect<T extends true> = T
