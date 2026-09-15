/**
 * Check-polarity survey: every exported `*Check` in `src/shape.ts`,
 * instantiated inside a generic function body against a naked generic
 * operand, asserting it does not error. A `*Check` must resolve to
 * `unknown` while its arguments are still type parameters; nothing stops
 * a future edit from flipping a polarity and turning a working check
 * fail-closed with `pnpm typecheck` still green, which is the regression
 * this file catches. Nothing here runs (vitest only collects `*.test.ts`).
 *
 * What "naked" means splits per check. The `IsExact`-guarded checks
 * (`BroadcastCheck`, `BroadcastToCheck`, `LastDimCheck`, `FlattenCheck`,
 * `SliceCheck`) stay open for a fully unconstrained `S extends Shape`,
 * because their guards route every comparison through helpers TypeScript
 * can still decide under a naked type parameter. The rank/dim-indexing
 * checks (`DimCheck`, `TransposeCheck`, `PermuteCheck`, `CatCheck`,
 * `MatMulCheck`, ...) are built on raw structural tests TypeScript cannot
 * decide against a fully free-floating `S`, so they are verified the way
 * this codebase actually uses them: a known-rank shape whose elements are
 * naked generics (`Tensor<[A, B, C]>`, never `Tensor<S>`). That is still a
 * real polarity test: every case must be accepted whatever `A`, `B`, `C`
 * turn out to be, so an inverted check fails the assertion.
 */
import { DimDiv } from "../src/shape.ts"
import type {
  BroadcastCheck,
  BroadcastToCheck,
  CatCheck,
  CatNCheck,
  ConvCheck,
  ConvOut,
  DimAdd,
  DimCheck,
  DimDivCheck,
  DimMul,
  DimSub,
  ErrorMessage,
  FlattenCheck,
  IndexCheck,
  IndexTensor,
  LastDimCheck,
  MatMulCheck,
  PermuteCheck,
  Rank1Check,
  Shape,
  Slice,
  SliceCheck,
  SqueezeDimCheck,
  TransposeCheck,
  UnflattenCheck,
  UnsqueezeCheck,
  ViewCheck,
} from "../src/shape.ts"
import type { AnyTensor, Tensor } from "../src/tensor.ts"
import { BROADCAST_TO_CASES, BROADCAST_TO_FAIL_CASES, BROADCAST_TO_TYPE_FAIL_CASES } from "./shape-cases.ts"

// Checks that stay open for a fully naked `S extends Shape`, paired with
// an equally free-floating, unrelated dim parameter.

// BroadcastCheck<A, B>
declare function _broadcastCheck<A extends Shape, B extends Shape>(b: Tensor<B> & BroadcastCheck<A, B>): Tensor<B>
function _broadcastCheckOpen<A extends Shape, B extends Shape>(b: Tensor<B>) {
  return _broadcastCheck<A, B>(b)
}

// BroadcastToCheck<S, V>
declare function _broadcastToCheck<S extends Shape, V extends Shape>(v: V & BroadcastToCheck<S, V>): V
function _broadcastToCheckOpen<S extends Shape, V extends Shape>(v: V) {
  return _broadcastToCheck<S, V>(v)
}

// LastDimCheck<S, D>
declare function _lastDimCheck<S extends Shape, D extends number>(x: Tensor<S> & LastDimCheck<S, D>): Tensor<S>
function _lastDimCheckOpen<S extends Shape, D extends number>(x: Tensor<S>) {
  return _lastDimCheck<S, D>(x)
}

// FlattenCheck<S, F, T>
declare function _flattenCheck<S extends Shape, F extends number, T extends number>(f: F & FlattenCheck<S, F, T>): F
function _flattenCheckOpen<S extends Shape, F extends number, T extends number>(f: F) {
  return _flattenCheck<S, F, T>(f)
}

// SliceCheck<S, Spec>
declare function _sliceCheck<S extends Shape, Spec extends readonly Slice[]>(spec: Spec & SliceCheck<S, Spec>): Spec
function _sliceCheckOpen<S extends Shape, Spec extends readonly Slice[]>(spec: Spec) {
  return _sliceCheck<S, Spec>(spec)
}

// Rank-sensitive checks: known rank, naked elements. Every `A`, `B`, `C`,
// `M`, `N`, `K` below is a fully generic, unresolved type parameter; only
// the tuple's length is fixed.

// DimCheck<S, D>
declare function _dimCheck<S extends Shape, D extends number>(d: D & DimCheck<S, D>): D
function _dimCheckOpen<A extends number, B extends number, C extends number>() {
  return _dimCheck<[A, B, C], 0>(0)
}

// MatMulCheck<A, B>
declare function _matMulCheck<A extends Shape, B extends Shape>(b: Tensor<B> & MatMulCheck<A, B>): Tensor<B>
function _matMulCheckOpen<M extends number, K extends number, N extends number>(b: Tensor<[K, N]>) {
  return _matMulCheck<[M, K], [K, N]>(b)
}

// ViewCheck<S, V>: the guard that stays open under genericity is the
// `IsDynamic<S>` wildcard escape (`number extends Prod<S>`), not a
// known-rank tuple: `view()` needs a literal element count either way, so
// reshaping a generic-dim shape is `flatten`/`unflatten`'s job and the
// honest naked-generic case for `view` is the truly dynamic `number[]`.
declare function _viewCheck<S extends Shape, V extends number[]>(v: V & ViewCheck<S, V>): V
function _viewCheckOpen(shape: number[]) {
  return _viewCheck<number[], [2, 3]>([2, 3])
}

// UnflattenCheck<S, D, Sizes>
declare function _unflattenCheck<S extends Shape, D extends number, Sizes extends Shape>(
  d: D & UnflattenCheck<S, D, Sizes>,
): D
function _unflattenCheckOpen<A extends number, B extends number>() {
  return _unflattenCheck<[A, B], 0, [A]>(0)
}

// TransposeCheck<S, D0, D1>
declare function _transposeCheck<S extends Shape, D0 extends number, D1 extends number>(
  d0: D0 & TransposeCheck<S, D0, D1>,
): D0
function _transposeCheckOpen<A extends number, B extends number, C extends number>() {
  return _transposeCheck<[A, B, C], 0, 1>(0)
}

// PermuteCheck<S, Order>
declare function _permuteCheck<S extends Shape, Order extends number[]>(order: Order & PermuteCheck<S, Order>): Order
function _permuteCheckOpen<A extends number, B extends number, C extends number>() {
  return _permuteCheck<[A, B, C], [2, 0, 1]>([2, 0, 1])
}

// SqueezeDimCheck<S, D>
declare function _squeezeDimCheck<S extends Shape, D extends number>(d: D & SqueezeDimCheck<S, D>): D
function _squeezeDimCheckOpen<A extends number, B extends number>() {
  return _squeezeDimCheck<[A, 1, B], 1>(1)
}

// UnsqueezeCheck<S, D>
declare function _unsqueezeCheck<S extends Shape, D extends number>(d: D & UnsqueezeCheck<S, D>): D
function _unsqueezeCheckOpen<A extends number, B extends number>() {
  return _unsqueezeCheck<[A, B], 0>(0)
}

// Rank1Check<S>: the rank itself (1) is the known part; the one element
// is a naked generic.
declare function _rank1Check<S extends Shape>(x: Tensor<S> & Rank1Check<S>): Tensor<S>
function _rank1CheckOpen<A extends number>(x: Tensor<[A]>) {
  return _rank1Check(x)
}

// CatCheck<A, B, D>
declare function _catCheck<A extends Shape, B extends Shape, D extends number>(
  b: Tensor<B> & CatCheck<A, B, D>,
): Tensor<B>
function _catCheckOpen<M extends number, N extends number, K extends number>(b: Tensor<[N, K]>) {
  return _catCheck<[M, K], [N, K], 0>(b)
}

// CatNCheck<T, D>
declare function _catNCheck<T extends readonly [AnyTensor, ...AnyTensor[]], D extends number>(
  t: T & CatNCheck<T, D>,
): T
function _catNCheckOpen<M extends number, N extends number, K extends number>(t: [Tensor<[M, K]>, Tensor<[N, K]>]) {
  return _catNCheck<[Tensor<[M, K]>, Tensor<[N, K]>], 0>(t)
}

// ConvCheck<H, K, S, P>: the one check in the survey whose fail-openness
// needs more than the right polarity. Its deferral bottoms out in
// `ConvOut`'s `number extends H` rather than in a distributive
// `X extends Y ? true : false`, so `ConvFits` supplies an escape of its
// own: the four `IsExact<_, number>` guards, which TypeScript decides
// eagerly even for a naked type parameter. The companion negatives are the
// scratch variants in the measured table below.
declare function _convCheck<H extends number, K extends number, St extends number, P extends number>(
  x: Tensor<[H]> & ConvCheck<H, K, St, P>,
): Tensor<[H]>
// the real shape of a `Conv2d(1, 8, 3)` call: literal kernel/stride/padding,
// naked generic spatial extent
function _convCheckOpen<H extends number>(x: Tensor<[H]>) {
  return _convCheck<H, 3, 1, 0>(x)
}
// ...and with every operand generic, which the `IsExact` guards answer first
function _convCheckOpenAllGeneric<H extends number, K extends number, St extends number, P extends number>(
  x: Tensor<[H]>,
) {
  return _convCheck<H, K, St, P>(x)
}

// IndexCheck<T>: not shape/dim-generic like the rest; the value is a
// genuinely branded `IndexTensor<S>` for a naked `S` (what every real
// caller of an index-typed signature has, via `t.toIndex()` or
// `Tensor.indices()`).
declare function _indexCheck<S extends Shape>(t: IndexTensor<S> & IndexCheck<IndexTensor<S>>): IndexTensor<S>
function _indexCheckOpen<S extends Shape>(t: IndexTensor<S>) {
  return _indexCheck(t)
}

export {
  _broadcastCheckOpen,
  _broadcastToCheckOpen,
  _catCheckOpen,
  _catNCheckOpen,
  _convCheckOpen,
  _convCheckOpenAllGeneric,
  _dimCheckOpen,
  _flattenCheckOpen,
  _indexCheckOpen,
  _lastDimCheckOpen,
  _matMulCheckOpen,
  _permuteCheckOpen,
  _rank1CheckOpen,
  _sliceCheckOpen,
  _squeezeDimCheckOpen,
  _transposeCheckOpen,
  _unflattenCheckOpen,
  _unsqueezeCheckOpen,
  _viewCheckOpen,
}

// DimDivCheck forwarding discipline.
//
// The check itself is fail-open for the shape it supports (proven in
// `dimdiv.test-d.ts`); this section is the sharper claim that a check
// carried correctly at one layer stays proven only if every layer between
// the proof and its use repeats it. `MHA` is the innermost consumer,
// `Block` the one intermediate layer.

// The disciplined chain: every constructor that needs the precondition
// carries its own `H & DimDivCheck<D, H>`, and every forwarding site names
// its type arguments explicitly. This must compile, generically and at a
// literal call site.
declare class MHA_Disciplined<D extends number, H extends number> {
  constructor(d: D, h: H & DimDivCheck<D, H>)
  readonly headDim: DimDiv<D, H>
}
class Block_Disciplined<D extends number, H extends number> {
  readonly attn: MHA_Disciplined<D, H>
  constructor(d: D, h: H & DimDivCheck<D, H>) {
    this.attn = new MHA_Disciplined<D, H>(d, h)
  }
}
function _forwardsDisciplined<D extends number, H extends number>(d: D, h: H & DimDivCheck<D, H>) {
  return new Block_Disciplined<D, H>(d, h)
}
const _disciplinedOk = new Block_Disciplined(384, 6)

// Failure mode 1: the intermediate constructor's own parameter drops the
// check (`h: H` instead of `h: H & DimDivCheck<D, H>`). `Block`'s own
// signature still catches a bad literal at its boundary...
declare class MHA_NoCheck<D extends number, H extends number> {
  constructor(d: D, h: H)
}
class Block_InnerDropsCheck<D extends number, H extends number> {
  readonly attn: MHA_NoCheck<D, H>
  constructor(d: D, h: H & DimDivCheck<D, H>) {
    this.attn = new MHA_NoCheck<D, H>(d, h)
  }
}
// @ts-expect-error 384 is not divisible by 5 (Block's own param has the check)
const _innerDropsCheckCaughtAtBlock = new Block_InnerDropsCheck(384, 5)
// ...but nothing stops a caller who reaches `MHA_NoCheck` directly: its own
// parameter never carried the check, so the identical bad literal compiles.
// No `@ts-expect-error` on this line: the absence of an error is the bug
// this failure mode names.
const _innerDropsCheckMissedDirectly = new MHA_NoCheck(384, 5)

// Failure mode 2: a forwarding site that omits explicit type arguments.
// This one is fail-closed outright: inference re-derives the
// intermediate's own type argument from the caller's already-checked `h`,
// so `DimDivCheck` is applied a second time and the two branded types are
// mutually unassignable for every `D, H`, not only a bad literal.
declare class MHA_ForForwarding<D extends number, H extends number> {
  constructor(d: D, h: H & DimDivCheck<D, H>)
}
class Block_ForgotTypeArgs<D extends number, H extends number> {
  readonly attn: MHA_ForForwarding<D, H>
  constructor(d: D, h: H & DimDivCheck<D, H>) {
    // @ts-expect-error inference re-derives H from the already-checked h,
    // double-applying DimDivCheck
    this.attn = new MHA_ForForwarding(d, h)
  }
}

export { _disciplinedOk, _forwardsDisciplined, Block_Disciplined, Block_ForgotTypeArgs, Block_InnerDropsCheck }

// `broadcastTo`'s fail table, given a type-level twin.
//
// `BroadcastToCheck`'s two error branches are genuinely different bugs: a
// target that cannot broadcast with the source at all
// (`BROADCAST_TO_TYPE_FAIL_CASES`), and a target that can, via the
// symmetric `CanBroadcast`, but only by shrinking the source, which
// `broadcastTo` must reject even though plain broadcasting would not
// (`BROADCAST_TO_FAIL_CASES`). The second is the case a polarity slip in
// `BroadcastToCheck` would silently let through.

type BroadcastToOk = (typeof BROADCAST_TO_CASES)[number]

function _broadcastToOk<C extends BroadcastToOk>(t: Tensor<C["from"]>, to: C["to"]) {
  return t.broadcastTo(to)
}
declare const _bcastToOk0: Tensor<BroadcastToOk["from"]>
_broadcastToOk(_bcastToOk0, BROADCAST_TO_CASES[0].to)
_broadcastToOk(_bcastToOk0, BROADCAST_TO_CASES[1].to)

declare const _bcastFrom0: Tensor<(typeof BROADCAST_TO_TYPE_FAIL_CASES)[0]["from"]>
// @ts-expect-error [2, 3] and [4] cannot broadcast at all
_bcastFrom0.broadcastTo(BROADCAST_TO_TYPE_FAIL_CASES[0].to)

declare const _bcastFrom1: Tensor<(typeof BROADCAST_TO_FAIL_CASES)[0]["from"]>
// @ts-expect-error [2, 3] broadcasts against [3], but not down to it
_bcastFrom1.broadcastTo(BROADCAST_TO_FAIL_CASES[0].to)

// `ConvCheck`'s fail-openness, pinned by its negatives.
//
// The survey row above proves the adopted `ConvCheck` is open under a naked
// generic spatial dim. This section proves why, by keeping scratch copies
// of `ConvFits` with one mechanism removed at a time and nothing else
// changed (same span test, same polarity, same carrier, same call). The
// measured result, with the project's own tsc:
//
//   IsExact guards + `H extends H ?`  (the adopted ConvFits)  -> OPEN
//   IsExact guards alone                                      -> OPEN
//   `H extends H ?` alone                                     -> CLOSED
//   neither                                                   -> CLOSED
//
// So the load-bearing escape is the `IsExact` guard chain: `IsExact<H,
// number>` is a function-type identity comparison, which TypeScript decides
// immediately even for a naked type parameter, and deciding it is what gives
// the rest of the chain somewhere to land. The `H extends H ?` distribution
// trigger is kept in `src/shape.ts` as the documented reading of why the
// deferral resolves, but on its own it is not sufficient; see
// `_convCheckTriggerAloneIsClosed` below. The quotient-based spellings at
// the end are the rejected alternatives: both fail-closed, which is why
// `ConvFits` tests the span.

type _IsExact<X, Y> = (<T>() => T extends X ? 1 : 2) extends <T>() => T extends Y ? 1 : 2 ? true : false
type _ConvSpan<H extends number, K extends number, P extends number> = DimSub<DimAdd<H, DimMul<2, P>>, K>
type _Err<F> = F extends false ? ErrorMessage<"conv: the kernel does not fit"> : unknown

// `src/shape.ts`'s `ConvFits` with the `IsExact` guard chain deleted and the
// distribution trigger kept.
type _FitsTriggerAlone<H extends number, K extends number, P extends number> = H extends H
  ? (`${_ConvSpan<H, K, P>}` extends `-${string}` ? false : true)
  : true
declare function _convTriggerAlone<H extends number>(x: Tensor<[H]> & _Err<_FitsTriggerAlone<H, 3, 0>>): Tensor<[H]>
function _convCheckTriggerAloneIsClosed<H extends number>(x: Tensor<[H]>) {
  // @ts-expect-error the distribution trigger without the IsExact guard
  // chain is fail-closed: the identical call the survey row above accepts
  // is rejected for every generic H
  return _convTriggerAlone<H>(x)
}

// ...and with both mechanisms gone, which is the naive spelling.
type _FitsNeither<H extends number, K extends number, P extends number> = `${_ConvSpan<H, K, P>}` extends `-${string}` ? false : true
declare function _convNeither<H extends number>(x: Tensor<[H]> & _Err<_FitsNeither<H, 3, 0>>): Tensor<[H]>
function _convCheckNeitherIsClosed<H extends number>(x: Tensor<[H]>) {
  // @ts-expect-error no escape at all: `${DimSub<H, 3>}` defers and the
  // ErrorMessage branch stays reachable
  return _convNeither<H>(x)
}

// The quotient-based spellings (why they would not even have worked, on
// top of trap (a) being why they are wrong).
type _IsNonPositive<D extends number> =
    number extends D ? false
  : `${D}` extends `-${string}` ? true
  : [D] extends [0] ? true
  : false
type _CheckOnQuotientNaive<H extends number, K extends number, St extends number, P extends number> = _IsNonPositive<ConvOut<H, K, St, P>> extends
  true ? ErrorMessage<"conv: quotient"> : unknown
type _CheckOnQuotientLaw1<H extends number, K extends number, St extends number, P extends number> = _IsNonPositive<ConvOut<H, K, St, P>> extends
  false ? unknown : ErrorMessage<"conv: quotient">

declare function _convQuotientNaive<H extends number>(
  x: Tensor<[H]> & _CheckOnQuotientNaive<H, 3, 1, 0>,
): Tensor<[H]>
declare function _convQuotientLaw1<H extends number>(x: Tensor<[H]> & _CheckOnQuotientLaw1<H, 3, 1, 0>): Tensor<[H]>

function _convCheckOnQuotientIsClosed<H extends number>(x: Tensor<[H]>) {
  // @ts-expect-error deciding on ConvOut defers at its `number extends H`
  // guard, and a deferred conditional with an ErrorMessage branch is
  // fail-closed whichever way the polarity is written
  const naive = _convQuotientNaive<H>(x)
  // @ts-expect-error ...law-1 polarity does not rescue it either
  const law1 = _convQuotientLaw1<H>(x)
  return { law1, naive }
}

// The escape is the whole difference: every spelling above agrees with the
// real `ConvCheck` on literals, so a green literal test suite is not
// evidence that any of them is safe to adopt.
type _Eq<A, B> = (<T>() => T extends A ? 1 : 2) extends <T>() => T extends B ? 1 : 2 ? true : false
type _Ex<T extends true> = T
type _sameOnLiterals0 = _Ex<_Eq<_Err<_FitsNeither<28, 3, 0>>, ConvCheck<28, 3, 1, 0>>>
type _sameOnLiterals1 = _Ex<_Eq<_Err<_FitsTriggerAlone<2, 5, 0>>, _Err<_FitsNeither<2, 5, 0>>>>

export { _convCheckNeitherIsClosed, _convCheckOnQuotientIsClosed, _convCheckTriggerAloneIsClosed }
