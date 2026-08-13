/**
 * One table of shape cases. Most tables are run twice: `types.test-d.ts`
 * asserts the eight positive tables through the type algebra, while
 * `shape.test.ts` runs every table through the runtime value functions
 * in `src/shape.ts`. The negative `*_FAIL_CASES` and the
 * `BROADCAST_TO_*` tables have no type-level twin and are only exercised
 * by `shape.test.ts`. A positive case added here is checked in both
 * worlds, which is what keeps them from drifting.
 *
 * The `as [2, 3]` casts matter: they make each entry a *mutable* literal
 * tuple, which is what the type-level operators (constrained to
 * `Shape = number[]`) accept.
 */

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
