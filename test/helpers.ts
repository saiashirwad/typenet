import { expect } from "vitest"
import { disableNative, isNativeAvailable, useNative } from "../src/backends/native.ts"
import { configure } from "../src/lazy.ts"
import { Tensor } from "../src/tensor.ts"

type AnyTensor = Tensor<any>

/** Runs `fn` once eagerly (lazy off) and once in lazy mode, resetting
 * the global flag afterwards. */
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

/** Elementwise closeness: `|a[i] - b[i]| < tol` with matching shapes. */
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

/** Runs `fn` eagerly, in lazy (JS) mode, and — when the native addon is
 * built — through the native backend, resetting global flags afterwards.
 * `native` is `null` when the addon is not available. */
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

/** Elementwise agreement of the lazy and (when available) native paths
 * against the eager path. Silently skips the native leg when the addon
 * is not built — see `expectAgreeStrict` for a version that fails instead. */
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

/** Same as `expectAgree`, but fails when an expected target (e.g. the
 * native backend on a machine where `pnpm build:native` was run) is
 * unavailable instead of silently skipping it. */
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
