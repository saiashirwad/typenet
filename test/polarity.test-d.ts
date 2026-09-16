// Check-polarity survey: every exported `*Check` in `src/shape.ts` must stay open while its
// arguments are still type parameters, or a working check silently turns fail-closed.
// The `IsExact`-guarded checks stay open for a fully naked `S extends Shape`; the rank and
// dim-indexing checks are probed with a known-rank shape whose elements are naked generics.
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
  NarrowCheck,
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
import type { Equal, Expect } from "./helpers.ts"
import { BROADCAST_TO_CASES, BROADCAST_TO_FAIL_CASES, BROADCAST_TO_TYPE_FAIL_CASES } from "./shape-cases.ts"

// Checks that stay open for a fully naked `S extends Shape`.

declare function _broadcastCheck<A extends Shape, B extends Shape>(b: Tensor<B> & BroadcastCheck<A, B>): Tensor<B>
function _broadcastCheckOpen<A extends Shape, B extends Shape>(b: Tensor<B>) {
  return _broadcastCheck<A, B>(b)
}

declare function _broadcastToCheck<S extends Shape, V extends Shape>(v: V & BroadcastToCheck<S, V>): V
function _broadcastToCheckOpen<S extends Shape, V extends Shape>(v: V) {
  return _broadcastToCheck<S, V>(v)
}

declare function _lastDimCheck<S extends Shape, D extends number>(x: Tensor<S> & LastDimCheck<S, D>): Tensor<S>
function _lastDimCheckOpen<S extends Shape, D extends number>(x: Tensor<S>) {
  return _lastDimCheck<S, D>(x)
}

declare function _flattenCheck<S extends Shape, F extends number, T extends number>(f: F & FlattenCheck<S, F, T>): F
function _flattenCheckOpen<S extends Shape, F extends number, T extends number>(f: F) {
  return _flattenCheck<S, F, T>(f)
}

declare function _sliceCheck<S extends Shape, Spec extends readonly Slice[]>(spec: Spec & SliceCheck<S, Spec>): Spec
function _sliceCheckOpen<S extends Shape, Spec extends readonly Slice[]>(spec: Spec) {
  return _sliceCheck<S, Spec>(spec)
}

declare function _narrowCheck<S extends Shape, D extends number, Start extends number, L extends number>(
  dim: D & NarrowCheck<S, D, Start, L>,
  start: Start,
  length: L,
): D
// A fully naked shape with a literal window, which is what a generic caller of `narrow` has.
function _narrowCheckOpen<S extends Shape>() {
  return _narrowCheck<S, 2, 1, 2>(2, 1, 2)
}
// The `number`-typed window `MultiHeadAttention` passes, which the `IsExact` guards have to answer.
function _narrowCheckOpenDynamicWindow<S extends Shape>(off: number, len: number) {
  return _narrowCheck<S, 2, number, number>(2, off, len)
}
// Known rank, naked elements: a literal window over dims that stay generic.
function _narrowCheckOpenGenericDims<A extends number, B extends number, C extends number>() {
  return _narrowCheck<[A, B, C], 1, 1, 2>(1, 1, 2)
}

// Rank-sensitive checks: known rank, naked elements. Only the tuple length is fixed.

declare function _dimCheck<S extends Shape, D extends number>(d: D & DimCheck<S, D>): D
function _dimCheckOpen<A extends number, B extends number, C extends number>() {
  return _dimCheck<[A, B, C], 0>(0)
}

declare function _matMulCheck<A extends Shape, B extends Shape>(b: Tensor<B> & MatMulCheck<A, B>): Tensor<B>
function _matMulCheckOpen<M extends number, K extends number, N extends number>(b: Tensor<[K, N]>) {
  return _matMulCheck<[M, K], [K, N]>(b)
}

// ViewCheck<S, V>: the escape that stays open under genericity is the `IsDynamic<S>` wildcard,
// not a known-rank tuple, so the honest naked case for `view` is the truly dynamic `number[]`.
declare function _viewCheck<S extends Shape, V extends number[]>(v: V & ViewCheck<S, V>): V
function _viewCheckOpen() {
  return _viewCheck<number[], [2, 3]>([2, 3])
}

declare function _unflattenCheck<S extends Shape, D extends number, Sizes extends Shape>(
  d: D & UnflattenCheck<S, D, Sizes>,
): D
function _unflattenCheckOpen<A extends number, B extends number>() {
  return _unflattenCheck<[A, B], 0, [A]>(0)
}

declare function _transposeCheck<S extends Shape, D0 extends number, D1 extends number>(
  d0: D0 & TransposeCheck<S, D0, D1>,
): D0
function _transposeCheckOpen<A extends number, B extends number, C extends number>() {
  return _transposeCheck<[A, B, C], 0, 1>(0)
}

declare function _permuteCheck<S extends Shape, Order extends number[]>(order: Order & PermuteCheck<S, Order>): Order
function _permuteCheckOpen<A extends number, B extends number, C extends number>() {
  return _permuteCheck<[A, B, C], [2, 0, 1]>([2, 0, 1])
}

declare function _squeezeDimCheck<S extends Shape, D extends number>(d: D & SqueezeDimCheck<S, D>): D
function _squeezeDimCheckOpen<A extends number, B extends number>() {
  return _squeezeDimCheck<[A, 1, B], 1>(1)
}

declare function _unsqueezeCheck<S extends Shape, D extends number>(d: D & UnsqueezeCheck<S, D>): D
function _unsqueezeCheckOpen<A extends number, B extends number>() {
  return _unsqueezeCheck<[A, B], 0>(0)
}

// Rank1Check<S>: the rank 1 is the known part, and the single element stays a naked generic.
declare function _rank1Check<S extends Shape>(x: Tensor<S> & Rank1Check<S>): Tensor<S>
function _rank1CheckOpen<A extends number>(x: Tensor<[A]>) {
  return _rank1Check(x)
}

declare function _catCheck<A extends Shape, B extends Shape, D extends number>(
  b: Tensor<B> & CatCheck<A, B, D>,
): Tensor<B>
function _catCheckOpen<M extends number, N extends number, K extends number>(b: Tensor<[N, K]>) {
  return _catCheck<[M, K], [N, K], 0>(b)
}

declare function _catNCheck<T extends readonly [AnyTensor, ...AnyTensor[]], D extends number>(
  t: T & CatNCheck<T, D>,
): T
function _catNCheckOpen<M extends number, N extends number, K extends number>(t: [Tensor<[M, K]>, Tensor<[N, K]>]) {
  return _catNCheck<[Tensor<[M, K]>, Tensor<[N, K]>], 0>(t)
}

// ConvCheck<H, K, S, P> is the one check whose fail-openness needs more than the right polarity:
// `ConvFits` adds four `IsExact<_, number>` guards, which TypeScript decides eagerly.
declare function _convCheck<H extends number, K extends number, St extends number, P extends number>(
  x: Tensor<[H]> & ConvCheck<H, K, St, P>,
): Tensor<[H]>
// A Conv2d(1, 8, 3) call: literal kernel, stride and padding over a naked generic spatial extent.
function _convCheckOpen<H extends number>(x: Tensor<[H]>) {
  return _convCheck<H, 3, 1, 0>(x)
}
// Every operand generic, which the `IsExact` guards answer first.
function _convCheckOpenAllGeneric<H extends number, K extends number, St extends number, P extends number>(
  x: Tensor<[H]>,
) {
  return _convCheck<H, K, St, P>(x)
}

// IndexCheck<T> takes a branded `IndexTensor<S>` for a naked `S`, which is what a real caller
// of an index-typed signature has.
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
  _narrowCheckOpen,
  _narrowCheckOpenDynamicWindow,
  _narrowCheckOpenGenericDims,
  _permuteCheckOpen,
  _rank1CheckOpen,
  _sliceCheckOpen,
  _squeezeDimCheckOpen,
  _transposeCheckOpen,
  _unflattenCheckOpen,
  _unsqueezeCheckOpen,
  _viewCheckOpen,
}

// DimDivCheck forwarding discipline: the check stays proven only if every layer between the
// proof and its use carries it, and every forwarding site names its type arguments.

// The disciplined chain: every constructor carries its own `H & DimDivCheck<D, H>`.
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

// Failure mode 1: the intermediate constructor drops the check from its own parameter.
// `Block`'s signature still catches a bad literal at its boundary.
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
// A caller who reaches `MHA_NoCheck` directly is not stopped: its parameter never carried the
// check, so the identical bad literal compiles and the missing error is the point.
const _innerDropsCheckMissedDirectly = new MHA_NoCheck(384, 5)

// Failure mode 2: a forwarding site that omits its type arguments. Inference re-derives `H`
// from the already-checked `h`, applying `DimDivCheck` twice to mutually unassignable types.
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

// `broadcastTo`'s fail table. A target that cannot broadcast at all and a target that can only
// by shrinking the source are different errors, and a polarity slip would let the second through.

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

// `ConvCheck`'s fail-openness, pinned by scratch copies of `ConvFits` with one mechanism
// removed at a time: the `IsExact` guards are load-bearing, the distribution trigger alone is not.

type _ConvSpan<H extends number, K extends number, P extends number> = DimSub<DimAdd<H, DimMul<2, P>>, K>
type _Err<F> = F extends false ? ErrorMessage<"conv: the kernel does not fit"> : unknown

// `ConvFits` with the `IsExact` guard chain deleted and the distribution trigger kept.
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

// Both mechanisms gone, the naive spelling.
type _FitsNeither<H extends number, K extends number, P extends number> = `${_ConvSpan<H, K, P>}` extends `-${string}` ? false : true
declare function _convNeither<H extends number>(x: Tensor<[H]> & _Err<_FitsNeither<H, 3, 0>>): Tensor<[H]>
function _convCheckNeitherIsClosed<H extends number>(x: Tensor<[H]>) {
  // @ts-expect-error no escape at all: `${DimSub<H, 3>}` defers and the
  // ErrorMessage branch stays reachable
  return _convNeither<H>(x)
}

// The quotient-based spellings, kept as the rejected alternatives.
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

// Every spelling above agrees with the real `ConvCheck` on literals, so a green literal
// suite is not evidence that any of them is safe to adopt.
type _sameOnLiterals0 = Expect<Equal<_Err<_FitsNeither<28, 3, 0>>, ConvCheck<28, 3, 1, 0>>>
type _sameOnLiterals1 = Expect<Equal<_Err<_FitsTriggerAlone<2, 5, 0>>, _Err<_FitsNeither<2, 5, 0>>>>

export { _convCheckNeitherIsClosed, _convCheckOnQuotientIsClosed, _convCheckTriggerAloneIsClosed }
