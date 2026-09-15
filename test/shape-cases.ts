// Shared shape tables: `types.test-d.ts` asserts the positive tables
// through the type algebra while `shape.test.ts` runs every table through
// the runtime value functions in `src/shape.ts`, so the two worlds cannot
// drift. The `as [2, 3]` casts make each entry a *mutable* literal tuple,
// which is what the type-level operators (constrained to `Shape = number[]`)
// accept.

export const BROADCAST_CASES = [
  { a: [2, 3] as [2, 3], b: [3] as [3], out: [2, 3] as [2, 3] },
  {
    a: [8, 1, 6, 1] as [8, 1, 6, 1],
    b: [7, 1, 5] as [7, 1, 5],
    out: [8, 7, 6, 5] as [8, 7, 6, 5],
  },
  { a: [2, 1] as [2, 1], b: [1, 3] as [1, 3], out: [2, 3] as [2, 3] },
  { a: [] as [], b: [2] as [2], out: [2] as [2] },
] as const

export const BROADCAST_FAIL_CASES = [
  { a: [2, 3] as [2, 3], b: [4] as [4] },
] as const

export const MATMUL_CASES = [
  { a: [2, 3] as [2, 3], b: [3, 4] as [3, 4], out: [2, 4] as [2, 4] },
  {
    a: [10, 2, 3] as [10, 2, 3],
    b: [3, 4] as [3, 4],
    out: [10, 2, 4] as [10, 2, 4],
  },
] as const

export const MATMUL_FAIL_CASES = [
  { a: [2, 3] as [2, 3], b: [4, 7] as [4, 7] },
] as const

export const VIEW_CASES = [
  { s: [4, 6] as [4, 6], v: [2, -1, 3] as [2, -1, 3], out: [2, 4, 3] as [2, 4, 3] },
  { s: [2, 3] as [2, 3], v: [6] as [6], out: [6] as [6] },
] as const

export const VIEW_FAIL_CASES = [
  { s: [2, 3] as [2, 3], v: [7, 2] as [7, 2] },
] as const

export const CAT_CASES = [
  { a: [2, 3] as [2, 3], b: [4, 3] as [4, 3], dim: 0 as const, out: [6, 3] as [6, 3] },
  { a: [2, 3] as [2, 3], b: [2, 5] as [2, 5], dim: 1 as const, out: [2, 8] as [2, 8] },
] as const

export const CAT_FAIL_CASES = [
  { a: [2, 3] as [2, 3], b: [4, 4] as [4, 4], dim: 0 as const },
] as const

export const RESIZE_CASES = [
  { s: [8, 16] as [8, 16], dim: 1 as const, length: 4 as const, out: [8, 4] as [8, 4] },
] as const

export const SLICE_CASES = [
  { s: [4, 5] as [4, 5], spec: [2, 3] as [2, 3], out: [2, 3] as [2, 3] },
  { s: [4, 5] as [4, 5], spec: [null, [1, 3]] as [null, [1, 3]], out: [4, 2] as [4, 2] },
  { s: [3, 7, 2] as [3, 7, 2], spec: [undefined, [2, 5], 1] as [undefined, [2, 5], 1], out: [3, 3, 1] as [3, 3, 1] },
] as const

export const SLICE_FAIL_CASES = [
  // an end index past the axis
  { s: [4, 5] as [4, 5], spec: [7, 2] as [7, 2] },
  // a window past the axis
  { s: [4, 5] as [4, 5], spec: [2, [1, 9]] as [2, [1, 9]] },
  // a window that ends before it starts
  { s: [4, 5] as [4, 5], spec: [2, [3, 1]] as [2, [3, 1]] },
] as const

export const FLATTEN_CASES = [
  { s: [2, 3, 4] as [2, 3, 4], from: 0 as const, to: 1 as const, out: [6, 4] as [6, 4] },
  { s: [2, 3, 4] as [2, 3, 4], from: 1 as const, to: 2 as const, out: [2, 12] as [2, 12] },
  // a one-axis window is the identity
  { s: [2, 3, 4] as [2, 3, 4], from: 1 as const, to: 1 as const, out: [2, 3, 4] as [2, 3, 4] },
  { s: [2, 3, 4] as [2, 3, 4], from: 0 as const, to: 2 as const, out: [24] as [24] },
] as const

export const FLATTEN_FAIL_CASES = [
  { s: [2, 3, 4] as [2, 3, 4], from: 1 as const, to: 3 as const },
  { s: [2, 3, 4] as [2, 3, 4], from: 2 as const, to: 1 as const },
  { s: [2, 3, 4] as [2, 3, 4], from: -1 as const, to: 2 as const },
] as const

export const UNFLATTEN_CASES = [
  { s: [2, 3, 4] as [2, 3, 4], dim: 2 as const, sizes: [2, 2] as [2, 2], out: [2, 3, 2, 2] as [2, 3, 2, 2] },
  { s: [2, 3, 4] as [2, 3, 4], dim: 0 as const, sizes: [1, 2] as [1, 2], out: [1, 2, 3, 4] as [1, 2, 3, 4] },
  { s: [6] as [6], dim: 0 as const, sizes: [2, 3] as [2, 3], out: [2, 3] as [2, 3] },
] as const

export const UNFLATTEN_FAIL_CASES = [
  { s: [2, 3, 4] as [2, 3, 4], dim: 5 as const, sizes: [2, 2] as [2, 2] },
  { s: [2, 3, 4] as [2, 3, 4], dim: 2 as const, sizes: [2, 3] as [2, 3] },
  { s: [2, 3, 4] as [2, 3, 4], dim: 0 as const, sizes: [] as [] },
] as const

export const DIM_DIV_CASES = [
  { a: 384 as const, b: 6 as const, out: 64 as const },
  { a: 12 as const, b: 4 as const, out: 3 as const },
  { a: 384 as const, b: 1 as const, out: 384 as const },
  // truncation toward zero, in both worlds
  { a: 7 as const, b: 2 as const, out: 3 as const },
] as const

export const PERMUTE_CASES = [
  {
    s: [2, 3, 4] as [2, 3, 4],
    order: [2, 0, 1] as [2, 0, 1],
    out: [4, 2, 3] as [4, 2, 3],
  },
] as const

export const REDUCE_CASES = [
  { s: [2, 3, 4] as [2, 3, 4], dim: 1 as const, keepdim: false, out: [2, 4] as [2, 4] },
  { s: [2, 3, 4] as [2, 3, 4], dim: 2 as const, keepdim: true, out: [2, 3, 1] as [2, 3, 1] },
] as const

export const BROADCAST_TO_CASES = [
  { from: [3] as [3], to: [2, 3] as [2, 3] },
  { from: [8, 1] as [8, 1], to: [8, 5] as [8, 5] },
] as const

export const BROADCAST_TO_FAIL_CASES = [
  // mutually broadcastable, but not expand-only
  { from: [2, 3] as [2, 3], to: [3] as [3] },
] as const

/**
 * Type-only companion to `BROADCAST_TO_FAIL_CASES`: the "cannot broadcast
 * at all" branch of `BroadcastToCheck` throws a different runtime message
 * than the expand-only row there, so it gets its own table. Exercised only
 * by `test/polarity.test-d.ts`.
 */
export const BROADCAST_TO_TYPE_FAIL_CASES = [
  { from: [2, 3] as [2, 3], to: [4] as [4] },
] as const

/**
 * Conv / pool spatial arithmetic, driven by `types.test-d.ts` (the
 * `ConvOut`/`PoolOut`/`FlattenFrom` types) and by `shape.test.ts` (the
 * value twins), so a drift between the two worlds is a test failure.
 */
export const CONV_CASES = [
  { h: 28 as const, k: 3 as const, s: 1 as const, p: 0 as const, out: 26 as const },
  { h: 13 as const, k: 3 as const, s: 1 as const, p: 0 as const, out: 11 as const },
  // "same" padding
  { h: 32 as const, k: 3 as const, s: 1 as const, p: 1 as const, out: 32 as const },
  { h: 7 as const, k: 3 as const, s: 2 as const, p: 1 as const, out: 4 as const },
  // the kernel exactly fills the input: a 1-wide output is legal
  { h: 3 as const, k: 3 as const, s: 1 as const, p: 0 as const, out: 1 as const },
] as const

export const POOL_CASES = [
  { h: 26 as const, k: 2 as const, s: 2 as const, out: 13 as const },
  { h: 11 as const, k: 2 as const, s: 2 as const, out: 5 as const },
  // a stride that does not divide the extent drops the ragged tail
  { h: 5 as const, k: 2 as const, s: 2 as const, out: 2 as const },
] as const

/**
 * Kernels that do not fit. `span` is `h + 2p - k`, the quantity `ConvCheck`
 * actually tests; `out` is what `ConvOut` reports anyway, because
 * `Numbers.Div` and `Math.trunc` both truncate toward zero rather than
 * flooring. A check written on the quotient instead of the span reads the
 * last rows as a legal 1-wide output and lets a 5-wide kernel onto a
 * 4-wide input; `test/conv-shapes.test-d.ts` pins that regression.
 */
export const CONV_FIT_FAIL_CASES = [
  { h: 2 as const, k: 5 as const, s: 1 as const, p: 0 as const, span: -3 as const, out: -2 as const },
  { h: 2 as const, k: 5 as const, s: 2 as const, p: 0 as const, span: -3 as const, out: 0 as const },
  // the trap: trunc((4 - 5) / 2) + 1 == 1, while floor(-0.5) + 1 == 0
  { h: 4 as const, k: 5 as const, s: 2 as const, p: 0 as const, span: -1 as const, out: 1 as const },
  // the same trap on the pooling path: a 4-wide window on a 2-wide input
  { h: 2 as const, k: 4 as const, s: 4 as const, p: 0 as const, span: -2 as const, out: 1 as const },
] as const

/**
 * The classifier head's flatten. The rank-1 row folds an empty tail to
 * `1`, and the rank-0 row is its own flatten.
 */
export const FLATTEN_FROM_CASES = [
  { s: [64, 16, 5, 5] as [64, 16, 5, 5], out: [64, 400] as [64, 400] },
  { s: [2, 3, 4] as [2, 3, 4], out: [2, 12] as [2, 12] },
  { s: [7] as [7], out: [7, 1] as [7, 1] },
  { s: [] as [], out: [] as [] },
] as const
