import { expect } from "vitest"
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
