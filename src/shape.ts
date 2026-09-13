import type { Call, Numbers } from "hotscript"
import { prod, showShape } from "./storage.ts"
// Type-only, and it stays that way: `verbatimModuleSyntax` erases the
// import, so the tensor.ts -> shape.ts edge remains the only runtime one.
import type { AnyTensor, Tensor } from "./tensor.ts"

export const zeroWidthSpace = "​"
type ZeroWidthSpace = typeof zeroWidthSpace

export type ErrorMessage<message extends string = string> = `${message}${ZeroWidthSpace}`

export type Shape = number[]

export type IsDynamic<S extends Shape> = number[] extends S ? true : false

/**
 * The first guard that mentions a naked generic defers the WHOLE chain.
 *
 * Deferral is not a failure — it is the residual. TS re-fires the
 * constructor when the generic is instantiated, so `DimAdd<C, 1>` hovers as
 * `DimAdd<C, 1>` inside a generic body and becomes `4` the moment `C` is
 * `3`. The rewrite rules exist for what must reduce NOW, so their guards
 * mention only one operand at a time, right operand first: `DimAdd<C, 0>`
 * is `C` even while `C` is unresolved, whereas `DimAdd<0, C>` defers (and
 * resolves identically at instantiation).
 *
 * A dim that is the wide `number` — not a literal, not a generic — remains
 * a wildcard that checks nothing rather than an error.
 */
export type DimAdd<A extends number, B extends number> =
    IsExact<B, 0> extends true ? A
  : IsExact<A, 0> extends true ? B
  : number extends A ? number
  : number extends B ? number
  : Call<Numbers.Add<A, B>> extends infer R extends number ? R
  : number

export function DimAdd<const A extends number, const B extends number>(a: A, b: B): DimAdd<A, B> {
  return (a + b) as any
}

export type DimMul<A extends number, B extends number> =
    IsExact<B, 0> extends true ? 0
  : IsExact<B, 1> extends true ? A
  : IsExact<A, 0> extends true ? 0
  : IsExact<A, 1> extends true ? B
  : number extends A ? number
  : number extends B ? number
  : Call<Numbers.Mul<A, B>> extends infer R extends number ? R
  : number

export function DimMul<const A extends number, const B extends number>(a: A, b: B): DimMul<A, B> {
  return (a * b) as any
}

export type DimSub<A extends number, B extends number> =
    IsExact<B, 0> extends true ? A
  : number extends A ? number
  : number extends B ? number
  : Call<Numbers.Sub<A, B>> extends infer R extends number ? R
  : number

export function DimSub<const A extends number, const B extends number>(a: A, b: B): DimSub<A, B> {
  return (a - b) as any
}

/**
 * Integer division, same guard discipline as {@link DimAdd}: the `/ 1`
 * identity first so it reduces while `A` is still a generic, then the
 * wildcards, then the arithmetic.
 *
 * hotscript's `Numbers.Div` truncates toward zero rather than flooring —
 * for shapes (non-negative) the two agree, and the value twin uses
 * `Math.trunc` so the runtime cannot drift from the type. A caller that
 * needs "does this divide evenly" must ask {@link DimDivCheck}, which
 * looks at the remainder; the quotient alone cannot tell you.
 */
export type DimDiv<A extends number, B extends number> =
    IsExact<B, 1> extends true ? A
  : number extends A ? number
  : number extends B ? number
  : Call<Numbers.Div<A, B>> extends infer R extends number ? R
  : number

export function DimDiv<const A extends number, const B extends number>(a: A, b: B): DimDiv<A, B> {
  return Math.trunc(a / b) as any
}

type Mod<A extends number, B extends number> = Call<Numbers.Mod<A, B>> extends infer R extends number ? R : number

/**
 * Fail-open divisibility (law 1). Errors ONLY when the remainder provably
 * reduces to a nonzero literal.
 *
 * The two-sided escape is what keeps it open: `[M] extends [0]` is the
 * definite pass, and `[0] extends [M]` catches everything that is still
 * deferred or wide (a deferred `Mod` has `number` for a constraint, and
 * `0` is assignable to that), so a generic `d_model` never trips the
 * error branch. `[V: final-verify/dimdiv.ts]` — and the two ways to lose
 * it are in `final-verify/dimdiv-bad.ts`: an intermediate constructor
 * that drops the check from its own parameter, or a forwarding site
 * without explicit type arguments.
 */
export type DimDivCheck<A extends number, B extends number> =
    Mod<A, B> extends infer M ?
      [M] extends [0] ? unknown
    : [0] extends [M] ? unknown
    : ErrorMessage<`attention: ${A} is not divisible by ${B}`>
  : unknown

type Reverse<T extends any[], Acc extends any[] = []> = T extends [infer H, ...infer R] ? Reverse<R, [H, ...Acc]> : Acc

/** Every axis but the last. `never` for a rank-0 shape. */
export type Init<S extends Shape> = S extends [...infer R extends number[], any] ? R : never

/** The last axis. `never` for a rank-0 shape. */
export type Last<S extends Shape> = S extends [...any[], infer L extends number] ? L : never

/**
 * The batch axes of a shape: everything the final (feature / class /
 * vocab) axis is not. Spelled as its own name because that is what it
 * means at every call site that uses it — `crossEntropy` targets,
 * per-row reductions — and `Init` reads like tuple surgery.
 */
export type BatchPrefix<S extends Shape> = Init<S>

/** The first `N` axes. Short tuples only; a non-tuple `Shape` yields `[]`. */
export type Take<S extends Shape, N extends number, Acc extends Shape = []> =
    Acc["length"] extends N ? Acc
  : S extends [infer X extends number, ...infer R extends number[]] ? Take<R, N, [...Acc, X]>
  : Acc

/** Everything after the first `N` axes. */
export type Drop<S extends Shape, N extends number, I extends 1[] = []> =
    I["length"] extends N ? S
  : S extends [any, ...infer R extends number[]] ? Drop<R, N, [...I, 1]>
  : []

/**
 * Counting tuple for the rank arithmetic below. Ranks past 12 fall off
 * the end of it — `Inc` and `Span` then return a wrong (never a
 * catastrophic) answer, and every consumer is guarded by a `*Check` that
 * fails open. No tensor in this library is rank 13.
 */
type Ones = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

/** `N + 1` for a rank index, by tuple length rather than hotscript. */
type Inc<N extends number> = [...Take<Ones, N>, 1]["length"] & number

/** `A - B + 1`: the number of axes in the inclusive window `B..A`. `never` when `A < B`. */
type Span<A extends number, B extends number> = [...Take<Ones, A>, 1] extends [...Take<Ones, B>, ...infer R] ? R["length"] & number : never

type HasDup<T extends number[], Seen extends number = never> =
    T extends [
      infer X extends number,
      ...infer Xs extends number[],
    ] ?
      [X] extends [Seen] ? true
    : HasDup<Xs, Seen | X>
  : false

type ShowShape<S extends Shape, Acc extends string = ""> =
    IsDynamic<S> extends true ? `[...]`
  : S extends [infer X extends number, ...infer Xs extends number[]] ? ShowShape<Xs, Acc extends "" ? `${X}` : `${Acc}, ${X}`>
  : `[${Acc}]`

type FoldMul<S extends Shape, Acc extends number> = S extends [infer X extends number, ...infer Xs extends number[]] ? FoldMul<Xs, DimMul<Acc, X>>
  : Acc

/**
 * The number of elements in a shape.
 *
 * The fold is SEEDED WITH `S[0]`, not with `1`, and that is load-bearing
 * (D22): `DimMul` tests `IsExact<B, 0>` before it can use the `1 * x`
 * identity, so `DimMul<1, B>` *defers* for a generic `B` and a seed of
 * `1` would make `Prod<[B, T]>` the residual `DimMul<DimMul<1, B>, T>` —
 * mutually unassignable with the `DimMul<B, T>` a caller writes by hand.
 * Reseeding makes the two the same type. The empty tuple still folds to
 * `1`.
 */
export type Prod<S extends Shape> =
    IsDynamic<S> extends true ? number
  : S extends [infer X extends number, ...infer Xs extends number[]] ? FoldMul<Xs, X>
  : 1

type IsNegative<D extends number> = `${D}` extends `-${string}` ? true : false

export type NormalizeDim<S extends Shape, D extends number> =
    number extends D ? number
  : IsNegative<D> extends true ? DimAdd<S["length"], D>
  : D

export type IsValidDim<S extends Shape, D extends number> =
    IsDynamic<S> extends true ? true
  : number extends D ? true
  : `${NormalizeDim<S, D>}` extends keyof S ? true
  : false

export type DimCheck<S extends Shape, D extends number> = IsValidDim<S, D> extends false
  ? ErrorMessage<`Dimension ${D} is out of range for shape ${ShowShape<S>}`>
  : unknown

type ReplaceAt<S extends Shape, I extends number, V extends number> = {
  [K in keyof S]: K extends `${I}` ? V : S[K]
}

type RemoveAt<S extends Shape, I extends number, Acc extends Shape = []> =
    S extends [
      infer X extends number,
      ...infer Xs extends number[],
    ] ?
      Acc["length"] extends I ? [...Acc, ...Xs]
    : RemoveAt<Xs, I, [...Acc, X]>
  : Acc

type InsertAt<
  S extends Shape,
  I extends number,
  V extends number,
  Acc extends Shape = [],
> =
    Acc["length"] extends I ? [...Acc, V, ...S]
  : S extends [infer X extends number, ...infer Xs extends number[]] ? InsertAt<Xs, I, V, [...Acc, X]>
  : [...Acc, V]

type IsExact<X, Y> = (<T>() => T extends X ? 1 : 2) extends <T>() => T extends Y ? 1 : 2 ? true : false

/**
 * Guard order obeys the law on `DimAdd`: the size-1 cases come first
 * because each guard mentions only ONE operand, so `BroadcastDim<C, 1>`
 * reduces to `C` even while `C` is an unresolved generic — the case that
 * keeps `Tensor<[N, C]>.mul(column)` typed as `Tensor<[N, C]>` once
 * instantiated. The price: `BroadcastDim<N, N>` defers behind the first
 * guard, but every branch of that deferred chain yields `N`, so
 * assignability still goes through. The same-shape suffix rule on
 * `Broadcast` catches the common `[E, C] + [C]` case before it ever
 * reaches this per-dim walk.
 */
type BroadcastDim<X extends number, Y extends number> =
    IsExact<Y, 1> extends true ? X
  : IsExact<X, 1> extends true ? Y
  : IsExact<X, number> extends true ? Y
  : IsExact<Y, number> extends true ? X
  : IsExact<X, Y> extends true ? X
  : never

type CanBroadcastRev<A extends Shape, B extends Shape> =
    A extends [
      infer X extends number,
      ...infer Xs extends number[],
    ] ?
      B extends [infer Y extends number, ...infer Ys extends number[]] ?
        IsExact<X, Y> extends true ? CanBroadcastRev<Xs, Ys>
      : [BroadcastDim<X, Y>] extends [never] ? false
      : CanBroadcastRev<Xs, Ys>
    : true
  : true

type BroadcastRev<A extends Shape, B extends Shape, Acc extends Shape = []> =
    A extends [
      infer X extends number,
      ...infer Xs extends number[],
    ] ?
      B extends [infer Y extends number, ...infer Ys extends number[]] ? BroadcastRev<Xs, Ys, [...Acc, BroadcastDim<X, Y>]>
    : BroadcastRev<Xs, [], [...Acc, X]>
  : B extends [infer Y extends number, ...infer Ys extends number[]] ? BroadcastRev<[], Ys, [...Acc, Y]>
  : Reverse<Acc>

/**
 * The suffix rules make `Broadcast<[E, C], [C]>` *syntactically*
 * `[E, C]` while `C` is an unresolved generic. Tuple-arity matching is
 * decidable whatever the elements instantiate to, and a strictly-longer
 * prefix (the nonempty `[number, ...number[]]` bound) keeps the rule
 * from even being attempted on same-rank pairs like `[N, C] × [N, 1]`,
 * whose element-level comparison would defer and poison the chain. This
 * is what lets a generic layer broadcast its bias without re-anchoring
 * the shape by hand.
 *
 * No structural rule for the generic outer product lives here, and that
 * is load-bearing: `[N, 1] × [1, N]` cannot be separated from
 * `[N, C] × [N, 1]` by any conditional — deciding "is this axis the
 * literal 1 or a generic that might be 1" defers, and the deferred
 * residual's branches differ, which poisons whichever case falls
 * through. The outer product is instead handled by the `this`-typed
 * overloads on `add`/`sub`/`mul`/`div` (and the operators): overload
 * resolution picks the first *applicable* signature with no branch
 * merging, which is exactly the routing a conditional type cannot do.
 */
/**
 * Rank-2 against a column: the answer is the left shape. No row-dim
 * comparison — `NoInfer<N>` vs `N` would defer it — because rejecting a
 * genuine row mismatch is `BroadcastCheck`'s job, not this type's.
 */
type ColumnBroadcast<A extends Shape, B extends Shape> =
    [A, B] extends [
      [infer _X extends number, infer _Y extends number],
      [infer _R extends number, infer C1 extends number],
    ] ?
      [C1] extends [1] ? A
    : never
  : never

export type Broadcast<A extends Shape, B extends Shape> = IsExact<A, B> extends true ? A
  : IsDynamic<A> extends true ? number[] : IsDynamic<B> extends true ? number[]
  : A extends [...infer _ extends [number, ...number[]], ...B] ? A
  : B extends [...infer _ extends [number, ...number[]], ...A] ? B
  // Rank-2 column broadcast, `[X, Y] × [X, 1]`: the literal 1 and the
  // identical first dim are both decidable, so multiplying by a
  // per-row column (an inverse-degree, a stochastic mask) keeps the
  // matrix's exact shape instead of a `BroadcastDim<X, X>` residual.
  // Only the column-on-the-right orientation is special-cased: the
  // mirrored rule would have to test "is A's second dim the literal
  // 1", which defers on a generic and poisons this very case.
  : [ColumnBroadcast<A, B>] extends [never] ? BroadcastRev<Reverse<A>, Reverse<B>>
  : ColumnBroadcast<A, B>

export type CanBroadcast<A extends Shape, B extends Shape> =
    IsExact<A, B> extends true ? true
  : IsDynamic<A> extends true ? true
  : IsDynamic<B> extends true ? true
  : CanBroadcastRev<Reverse<A>, Reverse<B>>

/**
 * Note the direction: `extends false`, not `extends true`.
 *
 * Inside a generic function body every predicate on a naked type parameter
 * *defers* — `IsExact<N, 1>` and `N extends 1` reduce to neither `true` nor
 * `false`. A check written `CanBroadcast<A,B> extends true ? unknown : Error`
 * therefore takes the error branch for all generic code, rejecting valid
 * programs like `[N,1] + [1,N]`.
 *
 * Testing for `false` inverts that: a deferred result is not `false` either,
 * so it falls through to `unknown` and the call is allowed. Concrete
 * mismatches still reduce to a definite `false` and are caught. The tradeoff
 * is deliberate — checks are permissive exactly where the compiler has no
 * information, and strict everywhere it does.
 */
export type BroadcastCheck<A extends Shape, B extends Shape> = CanBroadcast<A, B> extends false
  ? ErrorMessage<`Cannot broadcast ${ShowShape<A>} with ${ShowShape<B>}`>
  : unknown

/**
 * Public `broadcastTo` needs an *expand-only* check: `CanBroadcast` is
 * symmetric, so `CanBroadcast<[2, 3], [3]>` is true and would let a
 * `[2, 3]` tensor broadcast *down* to `[3]`. The target must be exactly
 * what broadcasting the source against it yields.
 */
export type BroadcastToCheck<S extends Shape, V extends Shape> =
    CanBroadcast<S, V> extends false ? ErrorMessage<`Cannot broadcast ${ShowShape<S>} to ${ShowShape<V>}`>
  : IsExact<Broadcast<S, V>, V> extends false ? ErrorMessage<`broadcastTo target ${ShowShape<V>} is not a broadcast of ${ShowShape<S>}`>
  : unknown

export type DimEq<X extends number, Y extends number> =
    IsExact<X, Y> extends true ? true
  : IsExact<X, number> extends true ? true
  : IsExact<Y, number> extends true ? true
  : X extends Y ? true
  : false

/**
 * "This layer owns the last axis": LayerNorm, RMSNorm, anything whose
 * parameters are per-channel.
 *
 * `[V: final-verify/lastdim.ts]` — fail-open for a naked generic `S`, for
 * `[B, T, D]` with generic dims, and for a generic `D`, while still
 * rejecting a literal mismatch. Do NOT rewrite this as
 * `S extends [...number[], infer L] ? … : ErrorMessage<…>`: that spelling
 * defers on a naked `S` *with an `ErrorMessage` in the fallthrough* and so
 * rejects every generic caller (`final-verify/failopen.ts`). Here the only
 * error branch is behind a definite `false`, which a deferred `DimEq`
 * never produces.
 */
export type LastDimCheck<S extends Shape, D extends number> =
    IsDynamic<S> extends true ? unknown
  : DimEq<Last<S>, D> extends false ? ErrorMessage<`expects ${D} features on the last axis, got ${ShowShape<S>}`>
  : unknown

type InnerA<A extends Shape> = A["length"] extends 1 ? A[0] : Last<A>

type InnerB<B extends Shape> = B["length"] extends 1 ? B[0] : Last<Init<B>>

type BatchDims<S extends Shape> = Init<Init<S>>

export type MatMul<A extends Shape, B extends Shape> =
    IsDynamic<A> extends true ? number[]
  : IsDynamic<B> extends true ? number[]
  : A extends [] ? never
  : B extends [] ? never
  : A["length"] extends 1 ?
      B["length"] extends 1 ? []
    : [...BatchDims<B>, Last<B>]
  : B["length"] extends 1 ? Init<A>
  : [...Broadcast<BatchDims<A>, BatchDims<B>>, Last<Init<A>>, Last<B>]

export type MatMulCheck<A extends Shape, B extends Shape> =
    IsDynamic<A> extends true ? unknown
  : IsDynamic<B> extends true ? unknown
  : A extends [] ? ErrorMessage<`matmul requires operands of rank >= 1`>
  : B extends [] ? ErrorMessage<`matmul requires operands of rank >= 1`>
  : DimEq<InnerA<A>, InnerB<B>> extends false ? ErrorMessage<`matmul: inner dimensions do not match (${ShowShape<A>} @ ${ShowShape<B>})`>
  : A["length"] extends 1 ? unknown
  : B["length"] extends 1 ? unknown
  : CanBroadcast<BatchDims<A>, BatchDims<B>> extends false ? ErrorMessage<
    `matmul: cannot broadcast batch dims of ${ShowShape<A>} with ${ShowShape<B>}`
  >
  : unknown

type ProductSkipNegOne<V extends number[], Acc extends number = 1> =
    V extends [
      infer X extends number,
      ...infer Xs extends number[],
    ] ?
      X extends -1 ? ProductSkipNegOne<Xs, Acc>
    : ProductSkipNegOne<Xs, DimMul<Acc, X>>
  : Acc

type CountNegOnes<V extends number[], Acc extends 1[] = []> =
    V extends [
      infer X,
      ...infer Xs extends number[],
    ] ?
      X extends -1 ? CountNegOnes<Xs, [...Acc, 1]>
    : CountNegOnes<Xs, Acc>
  : Acc["length"]

export type ResolveView<S extends Shape, V extends number[]> = {
  [K in keyof V]: V[K] extends -1 ? DimDiv<Prod<S>, ProductSkipNegOne<V>> : V[K]
}

export type ViewCheck<S extends Shape, V extends number[]> =
    CountNegOnes<V> extends 0 | 1 ?
      number extends Prod<S> ? unknown
    : number extends ProductSkipNegOne<V> ? unknown
    : CountNegOnes<V> extends 0 ?
        Prod<V> extends Prod<S> ? unknown
      : ErrorMessage<`Cannot view tensor of shape ${ShowShape<S>} as ${ShowShape<V>} (${Prod<S>} vs ${Prod<V>} elements)`>
    : DimMul<
      ProductSkipNegOne<V>,
      DimDiv<Prod<S>, ProductSkipNegOne<V>>
    > extends Prod<S> ? unknown
    : ErrorMessage<`Cannot infer -1 dim: ${Prod<S>} elements do not divide evenly into ${ShowShape<S>} -> ${ShowShape<V>}`>
  : ErrorMessage<`Only one -1 dim is allowed in view()`>

// ---------------------------------------------------------------------------
// Fail-open predicates, and the one rule that makes them work.
//
// A `*Check` that is still a *deferred* conditional when it lands in a
// parameter position is satisfied only by a value assignable to every
// branch TS has not ruled out — so a check that can still reach an
// `ErrorMessage` while its shape is a naked generic rejects every generic
// caller. Law 1 lost, silently, and `pnpm typecheck` green.
//
// TS rules the error branch out when the check is *definitely false*, and
// it decides that by resolving the predicate under a PERMISSIVE
// instantiation: every type parameter becomes a wildcard, and a wildcard
// is `IsExact` to everything. So a predicate that opens with
//
//     IsExact<S, Shape> extends true ? <the open answer>
//
// answers "open" under that instantiation whatever `S` turns out to be,
// the error branch is skipped, and the check resolves to `unknown`. The
// same guard is what makes the dynamic `number[]` shape a wildcard, so it
// is not a trick bolted on for the compiler's benefit — it is the same
// statement twice.
//
// The guards must come FIRST and they must be `IsExact`: `IsDynamic<S>`
// (which is `number[] extends S`) defers instead of resolving, and a
// structural test like `` `${F}` extends keyof S `` leaves the error on
// the undecidable side. Both spellings were tried; both are fail-closed
// for a naked `S` (`scratchpad/w013-dbg*.ts`). Same trap, same shape of
// fix, as `ConvFits` in D35.
//
// `_naked` in `test/flatten.test-d.ts` and `_w013FailOpen` in
// `test/types.test-d.ts` are the regressions: weaken a guard and they
// stop compiling.
// ---------------------------------------------------------------------------

/** {@link IsValidDim} that answers `true` rather than deferring under a naked `S`. */
type DimInRange<S extends Shape, D extends number> =
    IsExact<S, Shape> extends true ? true
  : IsExact<D, number> extends true ? true
  : IsValidDim<S, D> extends false ? false
  : true

/** {@link IsNegative} that answers `false` rather than deferring under a naked `D`. */
type IsNegativeDim<D extends number> =
    IsExact<D, number> extends true ? false
  : `${D}` extends `-${string}` ? true
  : false

/**
 * `flatten(from, to)` — the axes `from..to` (inclusive) collapse into one
 * whose extent is their product.
 *
 * This is the generic-dim reshape path (D21): `view()` needs the element
 * count to reduce to a literal, so `[B, T, D] -> [B*T, D]` is unavailable
 * to it while `B` and `T` are generics, whereas the fold here is exactly
 * the `DimMul<B, T>` a caller writes by hand — see D22 on why `Prod` is
 * reseeded.
 */
export type FlattenShape<S extends Shape, F extends number, T extends number> =
    IsDynamic<S> extends true ? number[]
  : number extends F ? number[]
  : number extends T ? number[]
  : [...Take<S, F>, Prod<Take<Drop<S, F>, Span<T, F>>>, ...Drop<S, Inc<T>>]

/**
 * Range and ordering, in that order. Every error branch hangs off one of
 * the fail-open predicates above rather than off the undecidable side of
 * a structural test — see the note on those predicates for why that is
 * the whole ballgame, and `_naked` in `test/flatten.test-d.ts` for the
 * regression.
 */
export type FlattenCheck<S extends Shape, F extends number, T extends number> =
    IsDynamic<S> extends true ? unknown
  : number extends F ? unknown
  : number extends T ? unknown
  : IsNegativeDim<F> extends true ? ErrorMessage<`flatten() takes non-negative dims, got ${F}`>
  : IsNegativeDim<T> extends true ? ErrorMessage<`flatten() takes non-negative dims, got ${T}`>
  : DimInRange<S, F> extends false ? ErrorMessage<`Dimension ${F} is out of range for shape ${ShowShape<S>}`>
  : DimInRange<S, T> extends false ? ErrorMessage<`Dimension ${T} is out of range for shape ${ShowShape<S>}`>
  : IsNegativeDim<DimSub<T, F>> extends true ? ErrorMessage<`flatten(${F}, ${T}): start dim is after end dim`>
  : unknown

/** `unflatten(dim, sizes)` — one axis splits into `sizes`, the inverse of {@link FlattenShape}. */
export type UnflattenShape<S extends Shape, D extends number, Sizes extends Shape> =
    IsDynamic<S> extends true ? number[]
  : number extends D ? number[]
  : [...Take<S, D>, ...Sizes, ...Drop<S, Inc<D>>]

/**
 * `Sizes` multiply to the extent of axis `D`, or the question is
 * undecidable and the answer is `true`.
 *
 * The `Prod<Sizes>` guard against `0` is not about zero-sized tensors
 * (those are a runtime concern): it is the second half of the
 * fail-open mechanism. TS decides a check is "not definitely false" by
 * resolving it under a *permissive* instantiation, where every type
 * parameter becomes a wildcard — and under that instantiation `DimMul`
 * takes its `IsExact<B, 0>` branch and a product of generics collapses to
 * `0`. Without this guard the permissive answer is a definite `false`,
 * the `ErrorMessage` branch stays reachable, and `unflatten(x, 2, [h, dh])`
 * — the attention split, the whole point of the op — is rejected.
 * A literal `0` size, the only thing this skips in real code, is caught by
 * {@link unflattenShape} at runtime.
 */
type SizesFillDim<S extends Shape, D extends number, Sizes extends Shape> =
    IsExact<S, Shape> extends true ? true
  : IsExact<Prod<Sizes>, 0> extends true ? true
  : DimEq<Prod<Sizes>, DimAt<S, D>> extends false ? false
  : true

export type UnflattenCheck<S extends Shape, D extends number, Sizes extends Shape> =
    IsDynamic<S> extends true ? unknown
  : number extends D ? unknown
  : IsNegativeDim<D> extends true ? ErrorMessage<`unflatten() takes a non-negative dim, got ${D}`>
  : DimInRange<S, D> extends false ? ErrorMessage<`Dimension ${D} is out of range for shape ${ShowShape<S>}`>
  : Sizes["length"] extends 0 ? ErrorMessage<`unflatten(${D}, []) needs at least one size`>
  : SizesFillDim<S, D, Sizes> extends false ? ErrorMessage<
    `unflatten(${D}, ${ShowShape<Sizes>}): sizes multiply to ${Prod<Sizes>}, not ${DimAt<S, D>}`
  >
  : unknown

type SwapDims<S extends Shape, I extends number, J extends number> =
    number extends I ? number[]
  : number extends J ? number[]
  : {
    [K in keyof S]: K extends `${I}` ? S[J] : K extends `${J}` ? S[I] : S[K]
  }

export type Transpose<S extends Shape, D0 extends number, D1 extends number> = IsDynamic<S> extends true ? number[]
  : SwapDims<S, NormalizeDim<S, D0>, NormalizeDim<S, D1>>

export type TransposeCheck<S extends Shape, D0 extends number, D1 extends number> =
    IsValidDim<S, D0> extends false ? ErrorMessage<`Dimension ${D0} is out of range for shape ${ShowShape<S>}`>
  : IsValidDim<S, D1> extends false ? ErrorMessage<`Dimension ${D1} is out of range for shape ${ShowShape<S>}`>
  : unknown

export type Permute<S extends Shape, Order extends number[]> = IsDynamic<S> extends true ? number[] : {
  [K in keyof Order]: S[NormalizeDim<S, Order[K]> extends infer I extends number ? I : never]
}

type AllValidDims<S extends Shape, Ds extends number[]> =
    Ds extends [
      infer X extends number,
      ...infer Xs extends number[],
    ] ?
      IsValidDim<S, X> extends true ? AllValidDims<S, Xs>
    : false
  : true

type NormalizeDims<S extends Shape, Ds extends number[]> = {
  [K in keyof Ds]: NormalizeDim<S, Ds[K]>
}

export type PermuteCheck<S extends Shape, Order extends number[]> =
    IsDynamic<S> extends true ? unknown
  : Order["length"] extends S["length"] ?
      AllValidDims<S, Order> extends false ? ErrorMessage<`permute(${ShowShape<Order>}) has a dim out of range for ${ShowShape<S>}`>
    : HasDup<NormalizeDims<S, Order> extends infer N extends number[] ? N : never> extends true ? ErrorMessage<
      `permute(${ShowShape<Order>}) repeats a dimension`
    >
    : unknown
  : ErrorMessage<`permute() expects ${S["length"]} dims, got ${Order["length"]}`>

export type Squeeze<S extends Shape, Acc extends Shape = []> =
    IsDynamic<S> extends true ? number[]
  : S extends [infer X extends number, ...infer Xs extends number[]] ?
      X extends 1 ? Squeeze<Xs, Acc>
    : Squeeze<Xs, [...Acc, X]>
  : Acc

export type SqueezeDim<S extends Shape, D extends number> =
    IsDynamic<S> extends true ? number[]
  : NormalizeDim<S, D> extends infer I extends number ?
      number extends I ? number[]
    : RemoveAt<S, I>
  : never

export type SqueezeDimCheck<S extends Shape, D extends number> =
    IsValidDim<S, D> extends false ? ErrorMessage<`Dimension ${D} is out of range for shape ${ShowShape<S>}`>
  : IsDynamic<S> extends true ? unknown
  : NormalizeDim<S, D> extends infer I extends number ?
      number extends I ? unknown
    : S[I & keyof S] extends 1 ? unknown
    : ErrorMessage<`Cannot squeeze dim ${D} of ${ShowShape<S>}: size is not 1`>
  : never

export type Unsqueeze<S extends Shape, D extends number> =
    IsDynamic<S> extends true ? number[]
  : NormalizeUnsqueezeDim<S, D> extends infer I extends number ?
      number extends I ? number[]
    : InsertAt<S, I, 1>
  : never

type NormalizeUnsqueezeDim<S extends Shape, D extends number> =
    number extends D ? number
  : IsNegative<D> extends true ? DimAdd<DimAdd<S["length"], 1>, D>
  : D

export type UnsqueezeCheck<S extends Shape, D extends number> =
    IsDynamic<S> extends true ? unknown
  : NormalizeUnsqueezeDim<S, D> extends infer I extends number ?
      number extends I ? unknown
    : `${I}` extends keyof S | `${S["length"]}` ? unknown
    : `${I}` extends `${number}` ? ErrorMessage<`Dimension ${D} is out of range for unsqueeze on ${ShowShape<S>}`>
    : unknown
  : never

export type ReduceDim<S extends Shape, D extends number, Keep extends boolean = false> =
    IsDynamic<S> extends true ? number[]
  : NormalizeDim<S, D> extends infer I extends number ?
      number extends I ? number[]
    : Keep extends true ? ReplaceAt<S, I, 1>
    : RemoveAt<S, I>
  : never

type ReduceDimsWalk<
  S extends Shape,
  Ns extends number,
  Keep extends boolean,
  I extends 1[] = [],
  Acc extends Shape = [],
> = S extends [
  infer X extends number,
  ...infer Xs extends number[],
] ? ReduceDimsWalk<
    Xs,
    Ns,
    Keep,
    [...I, 1],
    I["length"] extends Ns ? (Keep extends true ? [...Acc, 1] : Acc) : [...Acc, X]
  >
  : Acc

/**
 * Multi-axis reduce, the `ReduceDim` of a `sum([0, 2])`. Negative dims are
 * normalized first, exactly the way `Permute` normalizes its order, so the
 * walk below only ever compares non-negative positions.
 *
 * A dim that does not reduce to a literal makes the whole result the
 * dynamic `number[]`: which axis disappears is then unknowable, and a
 * wildcard shape is the honest answer (the same escape `ReduceDim` takes).
 */
export type ReduceDims<S extends Shape, Ds extends number[], Keep extends boolean = false> =
    IsDynamic<S> extends true ? number[]
  : NormalizeDims<S, Ds> extends infer Ns extends number[] ?
      number extends Ns[number] ? number[]
    : ReduceDimsWalk<S, Ns[number], Keep>
  : never

/** The size of dim `D` of `S`, with negative dims normalized. */
export type DimAt<S extends Shape, D extends number> = S[NormalizeDim<S, D> & keyof S] & number

/**
 * Rank-1 check on `this`, fail-open like every other check here: the
 * arity of a tuple is known even when its elements are generic, so
 * `[B]` passes decidably while a written `number[]` stays a wildcard.
 */
export type Rank1Check<S extends Shape> =
    IsDynamic<S> extends true ? unknown
  : S["length"] extends 1 ? unknown
  : ErrorMessage<"oneHot() requires a rank-1 tensor">

export type ResizeDim<S extends Shape, D extends number, L extends number> =
    IsDynamic<S> extends true ? number[]
  : NormalizeDim<S, D> extends infer I extends number ?
      number extends I ? number[]
    : ReplaceAt<S, I, L>
  : never

/** One axis of a `slice` spec: an end index (from 0), a `[start, end]` window, or keep-the-dim. */
export type Slice = number | readonly [number, number] | null | undefined

type SliceSize<C, D extends number> =
    C extends null | undefined ? D
  : C extends number ? C
  : C extends readonly [infer Start extends number, infer End extends number] ? DimSub<End, Start>
  : never

/**
 * `slice` folds {@link ResizeDim} over the axes: a `number` is an end
 * index (the resulting length is that number), `[start, end]` is a
 * window of length `end - start`, and `null` / `undefined` leave the
 * dim unchanged.
 */
export type SliceShape<S extends Shape, Spec extends readonly Slice[]> = {
  [K in keyof S]: SliceSize<Spec[K & keyof Spec], S[K] & number>
}

/**
 * `End <= D`, by the sign of the difference rather than by a comparison:
 * a literal `DimSub` either reduces to a number whose string form starts
 * with a minus or it does not, and a residual does neither.
 *
 * The wildcard guards come first for the usual two reasons at once — a
 * wide `number` axis checks nothing, and that is also the answer the
 * permissive instantiation needs in order to skip the error branch for a
 * generic axis (see the note on the fail-open predicates above).
 */
type FitsWithin<End extends number, D extends number> =
    IsExact<End, number> extends true ? true
  : IsExact<D, number> extends true ? true
  : `${DimSub<D, End>}` extends `-${string}` ? false
  : true

type SliceAxisCheck<C, D extends number> =
    C extends null | undefined ? unknown
  : [C] extends [number] ?
      IsExact<C, number> extends true ? unknown
    : IsNegativeDim<C> extends true ? ErrorMessage<`slice: negative end index ${C}`>
    : FitsWithin<C, D> extends false ? ErrorMessage<`slice: end index ${C} is past the axis extent ${D}`>
    : unknown
  : C extends readonly [
    infer St extends number,
    infer En extends number,
  ] ?
      IsNegativeDim<St> extends true ? ErrorMessage<`slice: negative start index ${St}`>
    : FitsWithin<St, En> extends false ? ErrorMessage<`slice: window [${St}, ${En}] ends before it starts`>
    : FitsWithin<En, D> extends false ? ErrorMessage<`slice: window [${St}, ${En}] is past the axis extent ${D}`>
    : unknown
  : unknown

type SliceAxes<S extends Shape, Spec extends readonly Slice[]> =
    S extends [
      infer X extends number,
      ...infer Xs extends number[],
    ] ?
      Spec extends readonly [
        infer C,
        ...infer Cs extends readonly Slice[],
      ] ?
        [SliceAxisCheck<C, X>] extends [ErrorMessage<string>] ? SliceAxisCheck<C, X>
      : SliceAxes<Xs, Cs>
    : unknown
  : unknown

/**
 * The one op family that had a result type and no check (`SliceShape`
 * happily produced a negative extent). Arity, then per axis: a window
 * that ends before it starts, and an end index past the axis.
 *
 * Literal windows only. A spec entry is a literal at every call site
 * (`slice` takes it `const`), while the axis it is measured against is
 * routinely a generic — so {@link FitsWithin} decides nothing there and
 * the axis passes, which is law 1 working as intended. `narrow()` still
 * throws at runtime, and {@link sliceShape} throws the same strings these
 * messages carry.
 *
 * Known limit, and the reason this check is not on `Tensor.slice` yet: a
 * spec entry that is itself a *generic* (`slice(t, [n, 2])` inside
 * `<N extends number>(…, n: N)`) leaves the per-axis walk deferred with
 * an `ErrorMessage` still in reach, and the call is rejected. Widening
 * the index to `number` or calling `narrow()` is the workaround; every
 * spelling that fixes it properly was tried against this file and none
 * resolved (see the note on the fail-open predicates).
 */
export type SliceCheck<S extends Shape, Spec extends readonly Slice[]> =
    IsDynamic<S> extends true ? unknown
  : SliceArityOk<S, Spec> extends false ? ErrorMessage<`slice() expects ${S["length"]} entries, got ${Spec["length"]}`>
  : SliceAxesOk<S, Spec> extends false ? SliceAxes<S, Spec>
  : unknown

/** One entry per axis — or a naked `S`, whose arity is nobody's business yet. */
type SliceArityOk<S extends Shape, Spec extends readonly Slice[]> =
    IsExact<S, Shape> extends true ? true
  : Spec["length"] extends S["length"] ? true
  : false

/** Any axis provably out of range. The message is recomputed by {@link SliceAxes} only in the error branch. */
type SliceAxesOk<S extends Shape, Spec extends readonly Slice[]> =
    IsExact<S, Shape> extends true ? true
  : [SliceAxes<S, Spec>] extends [ErrorMessage<string>] ? false
  : true

export type Stack<S extends Shape, N extends number, D extends number> =
    IsDynamic<S> extends true ? number[]
  : NormalizeUnsqueezeDim<S, D> extends infer I extends number ?
      number extends I ? number[]
    : InsertAt<S, I, N>
  : never

type CatDim<A extends Shape, B extends Shape, I extends number> = ReplaceAt<
  A,
  I,
  DimAdd<A[I & keyof A] & number, B[I & keyof B] & number>
>

export type Cat<A extends Shape, B extends Shape, D extends number> =
    IsDynamic<A> extends true ? number[]
  : IsDynamic<B> extends true ? number[]
  : NormalizeDim<A, D> extends infer I extends number ?
      number extends I ? number[]
    : CatDim<A, B, I> extends infer R extends number[] ? R
    : never
  : never

type EqualExceptAt<
  A extends Shape,
  B extends Shape,
  I extends number,
  Pos extends 1[] = [],
> =
    A extends [infer X extends number, ...infer Xs extends number[]] ?
      B extends [infer Y extends number, ...infer Ys extends number[]] ?
        Pos["length"] extends I ? EqualExceptAt<Xs, Ys, I, [...Pos, 1]>
      : DimEq<X, Y> extends true ? EqualExceptAt<Xs, Ys, I, [...Pos, 1]>
      : false
    : false
  : B extends [] ? true
  : false

export type CatCheck<A extends Shape, B extends Shape, D extends number> =
    IsDynamic<A> extends true ? unknown
  : IsDynamic<B> extends true ? unknown
  : A["length"] extends B["length"] ?
      IsValidDim<A, D> extends false ? ErrorMessage<`Dimension ${D} is out of range for shape ${ShowShape<A>}`>
    : NormalizeDim<A, D> extends infer I extends number ?
        number extends I ? unknown
      : EqualExceptAt<A, B, I> extends false ? ErrorMessage<`cat: shapes ${ShowShape<A>} and ${ShowShape<B>} differ outside dim ${D}`>
      : unknown
    : never
  : ErrorMessage<`cat: tensors must have the same rank (${ShowShape<A>} vs ${ShowShape<B>})`>

/** Shapes of a tuple of tensors, read structurally to avoid a cycle with tensor.ts. */
export type ShapesOf<T extends readonly unknown[]> = {
  [K in keyof T]: T[K] extends { readonly shape: infer S extends Shape } ? S : never
}

type CatNShapes<Ss extends readonly Shape[], D extends number> =
    Ss extends readonly [infer A extends Shape] ? A
  : Ss extends readonly [
    infer A extends Shape,
    infer B extends Shape,
    ...infer Rest extends Shape[],
  ] ? CatNShapes<[Cat<A, B, D>, ...Rest], D>
  : never

/** The shape of concatenating a tuple of tensors along `D`, as a fold of pairwise `Cat`. */
export type CatN<T extends readonly unknown[], D extends number> = CatNShapes<ShapesOf<T>, D>

type CatNCheckShapes<Ss extends readonly Shape[], D extends number> =
    Ss extends readonly [
      infer A extends Shape,
      infer B extends Shape,
      ...infer Rest extends Shape[],
    ] ?
      CatCheck<A, B, D> extends ErrorMessage<string> ? CatCheck<A, B, D>
    : CatNCheckShapes<[Cat<A, B, D>, ...Rest], D>
  : unknown

export type CatNCheck<T extends readonly unknown[], D extends number> = CatNCheckShapes<ShapesOf<T>, D>

export type InferShape<T, Depth extends 1[] = []> =
    Depth["length"] extends 12 ? number[]
  : T extends number ? []
  : T extends readonly any[] ? [T["length"], ...InferShape<T[0], [...Depth, 1]>]
  : never

export type NestedArray<S extends Shape> =
    IsDynamic<S> extends true ? any
  : S extends [] ? number
  : S extends [any, ...infer Rest extends number[]] ? NestedArray<Rest>[]
  : never

// ---------------------------------------------------------------------------
// Index dtype, as a phantom brand (D25).
//
// Full `Tensor<S, DType>` would put a second type parameter on ~40
// signatures to catch one bug class that shows up at exactly four:
// indexSelect, scatterAdd, Embedding.forward, and crossEntropy's targets.
// The brand is carried by the *type* only — nothing at runtime reads it —
// so it costs one intersection at those four places and nothing anywhere
// else. `Tensor.indices()` and `t.toIndex()` are the only ways to make
// one, and both check integrality at runtime.
// ---------------------------------------------------------------------------

declare const INDEX: unique symbol

export type IndexTensor<S extends Shape> = Tensor<S> & { readonly [INDEX]: true }

export type IndexCheck<T> = T extends { readonly [INDEX]: true } ? unknown
  : ErrorMessage<"index tensors must be int32/int64 — use t.toIndex() or Tensor.indices()">

/**
 * The documented cast, named once.
 *
 * Some shapes are true but not derivable: `narrow(2, 0, d)` on a
 * `[B, T, 3*D]` is a `[B, T, D]` because `d` is the runtime width the
 * slice was cut to, and no amount of type-level arithmetic recovers that
 * from `DimMul<3, D>`. Rather than let each such site grow its own
 * `as any`, they all call this, which says in one place: *the shape is an
 * assertion, and the runtime op below is what enforces it*.
 *
 * Legitimate uses are exactly those: a narrow/permute/reshape whose
 * result shape the author can state and the algebra cannot infer. It is
 * NOT a way past a `*Check` — a check that fires is a real mismatch, and
 * casting over it moves the failure to a kernel. `C` carries an extra
 * intersection for the rare site that also needs a brand back.
 */
export function assertChecked<S2 extends Shape, C = unknown>(t: AnyTensor): Tensor<S2> & C {
  return t as Tensor<S2> & C
}

// ---------------------------------------------------------------------------
// Runtime shape math. One value function per type above, throwing the
// same error strings the types put in ErrorMessage — the shared case
// table in test/shape-cases.ts runs both and keeps them from drifting.
// ---------------------------------------------------------------------------

/** Value twin of {@link Broadcast} / {@link CanBroadcast}. */
export function broadcastShapes(
  a: readonly number[],
  b: readonly number[],
): number[] {
  const rank = Math.max(a.length, b.length)
  const out = new Array<number>(rank)
  for (let i = 0; i < rank; i++) {
    const da = a[a.length - 1 - i] ?? 1
    const db = b[b.length - 1 - i] ?? 1
    if (da !== db && da !== 1 && db !== 1) {
      throw new Error(
        `Cannot broadcast ${showShape(a)} with ${showShape(b)}`,
      )
    }
    out[rank - 1 - i] = Math.max(da, db)
  }
  return out
}

/** Value twin of {@link ResolveView} / {@link ViewCheck}. */
export function resolveView(
  shape: readonly number[],
  view: readonly number[],
): number[] {
  const negOnes = view.filter(v => v === -1).length
  if (negOnes > 1) {
    throw new Error("Only one -1 dim is allowed in view()")
  }
  const total = prod(shape)
  if (negOnes === 1) {
    const rest = prod(view.filter(v => v !== -1))
    if (rest === 0 || total % rest !== 0) {
      throw new Error(
        `Cannot view tensor of shape ${showShape(shape)} as ${showShape(view)}`,
      )
    }
    return view.map(v => (v === -1 ? total / rest : v))
  }
  if (prod(view) !== total) {
    throw new Error(
      `Cannot view tensor of shape ${showShape(shape)} as ${showShape(view)} (${total} vs ${prod(view)} elements)`,
    )
  }
  return [...view]
}

/** Value twin of {@link MatMul} / {@link MatMulCheck} for rank >= 2 operands. */
export function matmulShape(
  a: readonly number[],
  b: readonly number[],
): number[] {
  const k = a[a.length - 1]!
  const k2 = b[b.length - 2]!
  if (k !== k2) {
    throw new Error(
      `matmul: inner dimensions do not match (${showShape(a)} @ ${showShape(b)})`,
    )
  }
  return [
    ...broadcastShapes(a.slice(0, -2), b.slice(0, -2)),
    a[a.length - 2]!,
    b[b.length - 1]!,
  ]
}

/** Value twin of {@link ReduceDim}. `dim` must already be normalized. */
export function reduceShape(
  shape: readonly number[],
  dim: number,
  keepdim: boolean,
): number[] {
  return keepdim
    ? shape.map((s, i) => (i === dim ? 1 : s))
    : shape.filter((_, i) => i !== dim)
}

/** Value twin of {@link Cat} / {@link CatCheck}. `dim` must already be normalized. */
export function catShape(
  a: readonly number[],
  b: readonly number[],
  dim: number,
): number[] {
  if (a.length !== b.length) {
    throw new Error(
      `cat: tensors must have the same rank (${showShape(a)} vs ${showShape(b)})`,
    )
  }
  for (let i = 0; i < a.length; i++) {
    if (i !== dim && a[i] !== b[i]) {
      throw new Error(
        `cat: shapes ${showShape(a)} and ${showShape(b)} differ outside dim ${dim}`,
      )
    }
  }
  return a.map((s, i) => (i === dim ? s + b[i]! : s))
}

/** Value twin of {@link ResizeDim}. `dim` must already be normalized. */
export function resizeDim(
  shape: readonly number[],
  dim: number,
  length: number,
): number[] {
  return shape.map((s, i) => (i === dim ? length : s))
}

/** Value twin of {@link SliceShape} / {@link SliceCheck}, throwing the check's strings. */
export function sliceShape(
  shape: readonly number[],
  spec: readonly Slice[],
): number[] {
  if (spec.length !== shape.length) {
    throw new Error(
      `slice() expects ${shape.length} entries, got ${spec.length}`,
    )
  }
  return shape.map((s, i) => {
    const c = spec[i]
    if (c == null) return s
    if (typeof c === "number") {
      if (c < 0) throw new Error(`slice: negative end index ${c}`)
      if (c > s) {
        throw new Error(
          `slice: end index ${c} is past the axis extent ${s}`,
        )
      }
      return c
    }
    const [start, end] = c
    if (start < 0) throw new Error(`slice: negative start index ${start}`)
    if (end < start) {
      throw new Error(
        `slice: window [${start}, ${end}] ends before it starts`,
      )
    }
    if (end > s) {
      throw new Error(
        `slice: window [${start}, ${end}] is past the axis extent ${s}`,
      )
    }
    return end - start
  })
}

/** Value twin of {@link FlattenShape} / {@link FlattenCheck}. */
export function flattenShape(
  shape: readonly number[],
  from: number,
  to: number,
): number[] {
  if (from < 0 || to < 0) {
    throw new Error(
      `flatten() takes non-negative dims, got ${from < 0 ? from : to}`,
    )
  }
  if (from >= shape.length) {
    throw new Error(
      `Dimension ${from} is out of range for shape ${showShape(shape)}`,
    )
  }
  if (to >= shape.length) {
    throw new Error(
      `Dimension ${to} is out of range for shape ${showShape(shape)}`,
    )
  }
  if (to < from) {
    throw new Error(`flatten(${from}, ${to}): start dim is after end dim`)
  }
  return [
    ...shape.slice(0, from),
    prod(shape.slice(from, to + 1)),
    ...shape.slice(to + 1),
  ]
}

/** Value twin of {@link UnflattenShape} / {@link UnflattenCheck}. */
export function unflattenShape(
  shape: readonly number[],
  dim: number,
  sizes: readonly number[],
): number[] {
  if (dim < 0) {
    throw new Error(`unflatten() takes a non-negative dim, got ${dim}`)
  }
  if (dim >= shape.length) {
    throw new Error(
      `Dimension ${dim} is out of range for shape ${showShape(shape)}`,
    )
  }
  if (sizes.length === 0) {
    throw new Error(`unflatten(${dim}, []) needs at least one size`)
  }
  const total = prod(sizes)
  if (total !== shape[dim]) {
    throw new Error(
      `unflatten(${dim}, ${showShape(sizes)}): sizes multiply to ${total}, not ${shape[dim]}`,
    )
  }
  return [...shape.slice(0, dim), ...sizes, ...shape.slice(dim + 1)]
}

/** Value twin of {@link Permute}. `order` must already be normalized. */
export function permuteShape(
  shape: readonly number[],
  order: readonly number[],
): number[] {
  return order.map(i => shape[i]!)
}

/**
 * Expand-only broadcast: `to` must be exactly what broadcasting `from`
 * against it yields, so `[2, 3] -> [3]` is rejected even though the two
 * shapes are mutually broadcastable.
 */
export function broadcastToShape(
  from: readonly number[],
  to: readonly number[],
): number[] {
  const out = broadcastShapes(from, to)
  if (
    out.length !== to.length
    || out.some((s, i) => s !== to[i])
  ) {
    throw new Error(
      `broadcastTo target ${showShape(to)} is not a broadcast of ${showShape(from)}`,
    )
  }
  return [...to]
}

// ---------------------------------------------------------------------------
// Conv / pooling spatial arithmetic (D35).
//
// Two traps live here, both found by compiling probes against this file
// with the project's own tsc, and both pinned by tests
// (`test/conv-shapes.test-d.ts`, `test/polarity.test-d.ts`) so neither can
// be refactored away:
//
//   (a) hotscript's `Numbers.Div` TRUNCATES TOWARD ZERO; it does not floor.
//       Measured on literals: `27/2 -> 13`, `26/3 -> 8`, `1/2 -> 0`,
//       `5/2 -> 2`, `7/3 -> 2`, and `-5/2 -> -2` where floor is `-3`. For a
//       non-negative numerator truncation and floor agree, so {@link ConvOut}
//       is correct wherever the kernel fits — but a check written on the
//       QUOTIENT inherits the divergence: `ConvOut<4, 5, 2, 0>` computes
//       `trunc(-1/2) + 1 = 1` while the true `floor(-0.5) + 1` is `0`, so a
//       5-wide kernel on a 4-wide input would type as a legal 1x1 output.
//       {@link ConvCheck} therefore tests the SPAN, never the quotient.
//
//   (b) Polarity alone is not enough for a check built on this algebra.
//       Both `IsNonPositive<ConvOut<...>> extends true ? Error : unknown` and
//       the law-1 spelling `... extends false ? Error : unknown` are
//       fail-CLOSED on a naked generic spatial dim: `ConvOut`'s first guard
//       is `number extends H`, which defers, and a source is not assignable
//       to a deferred conditional whose other branch is an `ErrorMessage`.
//       So is a span test with no escape at all. What buys the open branch
//       back is a guard TypeScript can decide EAGERLY even for a naked type
//       parameter, and {@link ConvFits}'s four `IsExact<_, number>` tests are
//       exactly that (`IsExact` is a function-type identity comparison, not a
//       conditional on the operand). Measured, same carrier, same call, one
//       mechanism at a time:
//
//         IsExact guards + `H extends H ?`   (as written below)  -> OPEN
//         IsExact guards alone                                   -> OPEN
//         `H extends H ?` alone                                  -> CLOSED
//         neither                                                -> CLOSED
//
//       The `H extends H ?` distribution trigger is kept because D35
//       specifies it and because it is the documented account of why the
//       residual resolves; the table above is the honest record that it is
//       belt-and-braces here rather than the load-bearing part. The OPEN row
//       as written, and both CLOSED rows, are pinned in
//       `test/polarity.test-d.ts`, so deleting the `IsExact` guard chain is a
//       compile error rather than a silent fail-closed check.
// ---------------------------------------------------------------------------

/**
 * `H + 2P - K` — the numerator of the conv output formula.
 *
 * Non-negative exactly when the kernel fits, which is the only question
 * {@link ConvCheck} asks. Deliberately not exported: it is an
 * implementation detail of the two things below, and exporting it would
 * invite a caller to test the quotient instead (trap (a)).
 */
type ConvSpan<H extends number, K extends number, P extends number> = DimSub<DimAdd<H, DimMul<2, P>>, K>

/**
 * `floor((H - K + 2P) / S) + 1` — the extent of one spatial axis after a
 * convolution with kernel `K`, stride `S` and padding `P`.
 *
 * Literal when every operand is literal, a plain `number` wildcard the
 * moment any operand is the wide `number`, and deferred-then-resolved for
 * generic dims (so a `Conv2d` inside a generic `<B, H, W>` body hovers as
 * the residual and becomes `26` when `H` is `28`).
 *
 * `floor` and `trunc` agree here because {@link ConvCheck} has already
 * ruled out a negative span wherever it is applied — see trap (a) above
 * for what happens if you trust this number without that check.
 */
export type ConvOut<H extends number, K extends number, S extends number, P extends number> =
    number extends H ? number
  : number extends K ? number
  : number extends S ? number
  : number extends P ? number
  : DimAdd<DimDiv<ConvSpan<H, K, P>, S>, 1>

/** Pooling is a convolution with no padding. */
export type PoolOut<H extends number, K extends number, S extends number> = ConvOut<H, K, S, 0>

/**
 * Does the kernel fit? FAIL-OPEN (law 1) and DISTRIBUTIVE (trap (b)).
 *
 * The four `IsExact<_, number>` guards are the escape: TypeScript decides a
 * function-type identity comparison immediately, even for a naked type
 * parameter, so the chain always reaches a branch instead of deferring with
 * an `ErrorMessage` still reachable in {@link ConvCheck}. `H extends H ?` is
 * D35's distribution trigger, kept as specified — the reading is that the
 * distributed conditional takes its default constraint `true | false` and
 * `boolean extends false` resolves to the open branch — but measured on its
 * own it is NOT sufficient (see the table in this section's header). Neither
 * is decoration to delete casually: `test/polarity.test-d.ts` pins the
 * positive and all three fail-closed variants.
 *
 * It tests the SPAN, never the quotient (trap (a)). With `S >= 1` a
 * non-negative span makes `trunc == floor` and the output `>= 1` by
 * construction.
 */
type ConvFits<H extends number, K extends number, S extends number, P extends number> =
    IsExact<H, number> extends true ? true
  : IsExact<K, number> extends true ? true
  : IsExact<S, number> extends true ? true
  : IsExact<P, number> extends true ? true
  : H extends H ? (`${ConvSpan<H, K, P>}` extends `-${string}` ? false : true)
  : true

/**
 * Errors ONLY when the kernel provably does not fit a literal extent.
 * `Axis` names the axis in the message (`"height"`, `"width"`, …) so a
 * rank-4 layer can say which one it was.
 */
export type ConvCheck<
  H extends number,
  K extends number,
  S extends number,
  P extends number,
  Axis extends string = "spatial",
> = ConvFits<H, K, S, P> extends false ? ErrorMessage<`conv: kernel ${K} with padding ${P} does not fit a ${Axis} extent of ${H}`> : unknown

/**
 * `[B, C, H, W] -> [B, C*H*W]` — the classifier head's flatten, keeping the
 * batch axis and folding everything else.
 *
 * The fold is {@link Prod}, reseeded with its first element (D22), which is
 * what makes `FlattenFrom<[B, C, H, W]>` mutually assignable with the
 * `[B, DimMul<DimMul<C, H>, W>]` a caller writes by hand. Conv is the first
 * caller that depends on that property; `test/conv-shapes.test-d.ts`
 * asserts it inside a generic body.
 */
export type FlattenFrom<S extends Shape> = S extends [infer B extends number, ...infer R extends number[]] ? [B, Prod<R>] : S

/**
 * Value twin of {@link ConvOut} (law 4).
 *
 * `Math.trunc`, not `Math.floor`, so the runtime cannot drift from the
 * type — see trap (a). Callers get the honest floor answer only where
 * {@link ConvCheck} holds, which is every call site in the library.
 */
export function ConvOut<
  const H extends number,
  const K extends number,
  const S extends number,
  const P extends number,
>(h: H, k: K, s: S, p: P): ConvOut<H, K, S, P> {
  return (Math.trunc((h + 2 * p - k) / s) + 1) as any
}

/** Value twin of {@link PoolOut} (law 4). */
export function PoolOut<const H extends number, const K extends number, const S extends number>(
  h: H,
  k: K,
  s: S,
): PoolOut<H, K, S> {
  return ConvOut(h, k, s, 0) as any
}

/** Value twin of {@link FlattenFrom} (law 4). A rank-0 shape is its own flatten. */
export function flattenFrom<const S extends Shape>(s: S): FlattenFrom<S> {
  if (s.length === 0) return [...s] as any
  return [s[0]!, prod(s.slice(1))] as any
}
