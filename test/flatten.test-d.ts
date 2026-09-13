/**
 * `FlattenShape` / `UnflattenShape` and their checks, against the real
 * exports. Checked-in version of `scratchpad/probe/flatten.ts`.
 *
 * These are the generic-dim reshape path (D21): `view()` needs the
 * element count to reduce to a literal, so `[B, T, D] -> [B*T, D]` is
 * unavailable to it while `B` and `T` are generics. The attention body
 * (`gpt-adopted.test-d.ts`) is built out of exactly these two.
 *
 * The `flatten` / `unflatten` *methods* land on `Tensor` in W1.8; the
 * free functions below are local stand-ins with the same signatures, so
 * this file exercises the type algebra without pre-announcing an API
 * that does not exist yet.
 */
import { DimMul } from "../src/shape.ts"
import type { FlattenCheck, FlattenShape, Shape, UnflattenCheck, UnflattenShape } from "../src/shape.ts"
import type { Tensor } from "../src/tensor.ts"

type Equal<A, B> = (<T>() => T extends A ? 1 : 2) extends (<T>() => T extends B ? 1 : 2) ? true : false
type Expect<T extends true> = T

declare function flatten<S extends Shape, const F extends number, const T extends number>(
  t: Tensor<S>,
  from: F & FlattenCheck<S, F, T>,
  to: T,
): Tensor<FlattenShape<S, F, T>>

declare function unflatten<S extends Shape, const D extends number, const Sizes extends number[]>(
  t: Tensor<S>,
  dim: D & UnflattenCheck<S, D, Sizes>,
  sizes: Sizes,
): Tensor<UnflattenShape<S, D, Sizes>>

// ---- literal shapes ---------------------------------------------------------
declare const lit234: Tensor<[2, 3, 4]>

const _f1 = flatten(lit234, 0, 1)
type _tf1 = Expect<Equal<typeof _f1.shape, [6, 4]>>

const _f2 = flatten(lit234, 1, 2)
type _tf2 = Expect<Equal<typeof _f2.shape, [2, 12]>>

// a one-axis window is the identity
const _f3 = flatten(lit234, 1, 1)
type _tf3 = Expect<Equal<typeof _f3.shape, [2, 3, 4]>>

const _u1 = unflatten(lit234, 2, [2, 2])
type _tu1 = Expect<Equal<typeof _u1.shape, [2, 3, 2, 2]>>

const _u2 = unflatten(lit234, 0, [1, 2])
type _tu2 = Expect<Equal<typeof _u2.shape, [1, 2, 3, 4]>>

// @ts-expect-error dim 3 is out of range for a rank-3 shape
const _fOutOfRange = flatten(lit234, 1, 3)
// @ts-expect-error the start dim is after the end dim
const _fReversed = flatten(lit234, 2, 1)
// @ts-expect-error negative dims are not accepted
const _fNegative = flatten(lit234, -1, 2)
// @ts-expect-error dim 5 is out of range for a rank-3 shape
const _uOutOfRange = unflatten(lit234, 5, [2, 2])
// @ts-expect-error 2 * 3 is 6, and axis 2 is 4
const _uWrongProduct = unflatten(lit234, 2, [2, 3])

// ---- generic dims: the point of the exercise --------------------------------
function _generic<B extends number, T extends number, D extends number, H extends number, Dh extends number>(
  x: Tensor<[B, T, D]>,
  idx: Tensor<[B, T]>,
  ctx: Tensor<[B, T, H, Dh]>,
  b: B,
  t: T,
  h: H,
  dh: Dh,
) {
  const f1 = flatten(idx, 0, 1)
  type _1 = Expect<Equal<typeof f1.shape, [DimMul<B, T>]>>

  const u1 = unflatten(f1, 0, [b, t])
  type _2 = Expect<Equal<typeof u1.shape, [B, T]>>

  const q4 = unflatten(x, 2, [h, dh])
  type _3 = Expect<Equal<typeof q4.shape, [B, T, H, Dh]>>

  const m1 = flatten(ctx, 2, 3)
  type _4 = Expect<Equal<typeof m1.shape, [B, T, DimMul<H, Dh>]>>

  const l1 = flatten(x, 0, 1)
  type _5 = Expect<Equal<typeof l1.shape, [DimMul<B, T>, D]>>

  return [f1, u1, q4, m1, l1] as const
}

// A naked generic shape decides nothing and is therefore accepted (law 1);
// the result is the residual, which re-fires at instantiation.
//
// This is the regression for the distribution triggers in `shape.ts`: a
// check whose error branch is still *reachable* while `S` is unresolved
// rejects every generic caller, because a deferred conditional in a
// parameter position is satisfied only by a value assignable to all of
// its branches.
function _naked<S extends Shape>(x: Tensor<S>) {
  return flatten(x, 0, 1)
}

function _nakedUnflatten<S extends Shape>(x: Tensor<S>, n: number) {
  return unflatten(x, 0, [1, n])
}

// ...and so is the dynamic shape.
function _dynamic(x: Tensor<number[]>) {
  const f = flatten(x, 0, 1)
  type _1 = Expect<Equal<typeof f.shape, number[]>>
  return f
}

export { _dynamic, _f1, _f2, _f3, _fNegative, _fOutOfRange, _fReversed, _generic, _naked, _nakedUnflatten, _u1, _u2, _uOutOfRange, _uWrongProduct }
