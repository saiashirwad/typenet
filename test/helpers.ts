import { expect } from "vitest"
import { disableNative, isNativeAvailable, useNative } from "../src/backends/native.ts"
import { configure } from "../src/lazy.ts"
import { Tensor } from "../src/tensor.ts"

type AnyTensor = Tensor<any>

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
