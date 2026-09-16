import type { Call, Numbers } from "hotscript"
import { prod, showShape } from "./storage.ts"
// Type-only: `verbatimModuleSyntax` erases it, so tensor.ts -> shape.ts stays the only runtime edge.
import type { AnyTensor, Tensor } from "./tensor.ts"

const zeroWidthSpace = "​"
type ZeroWidthSpace = typeof zeroWidthSpace

export type ErrorMessage<message extends string = string> = `${message}${ZeroWidthSpace}`

export type Shape = number[]

export type IsDynamic<S extends Shape> = number[] extends S ? true : false

/** The algebra below destructures mutable tuples, so a readonly shape is copied at the boundary. */
type MutableShape<V extends readonly number[]> = { -readonly [K in keyof V]: V[K] }

/** Each guard mentions one operand at a time, right operand first, so the identity cases reduce even while a dim is an unresolved generic. */
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

/** hotscript's Numbers.Div truncates toward zero; the value twin uses Math.trunc to match. */
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

/** Errors only when the remainder provably reduces to a nonzero literal; the two-sided `extends` keeps generic dims on the pass branch. */
export type DimDivCheck<A extends number, B extends number> =
    Mod<A, B> extends infer M ?
      [M] extends [0] ? unknown
    : [0] extends [M] ? unknown
    : ErrorMessage<`attention: ${A} is not divisible by ${B}`>
  : unknown

type Reverse<T extends any[], Acc extends any[] = []> = T extends [infer H, ...infer R] ? Reverse<R, [H, ...Acc]> : Acc

export type Init<S extends Shape> = S extends [...infer R extends number[], any] ? R : never

export type Last<S extends Shape> = S extends [...any[], infer L extends number] ? L : never

export type BatchPrefix<S extends Shape> = Init<S>

export type Take<S extends Shape, N extends number, Acc extends Shape = []> =
    Acc["length"] extends N ? Acc
  : S extends [infer X extends number, ...infer R extends number[]] ? Take<R, N, [...Acc, X]>
  : Acc

export type Drop<S extends Shape, N extends number, I extends 1[] = []> =
    I["length"] extends N ? S
  : S extends [any, ...infer R extends number[]] ? Drop<R, N, [...I, 1]>
  : []

type Ones = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

type Inc<N extends number> = [...Take<Ones, N>, 1]["length"] & number

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

/** Seeded with S[0] so a generic first dim stays exact: DimMul<1, B> defers for a generic B. */
export type Prod<S extends Shape> =
    IsDynamic<S> extends true ? number
  : S extends [infer X extends number, ...infer Xs extends number[]] ? FoldMul<Xs, X>
  : 1

type IsNegative<D extends number> = `${D}` extends `-${string}` ? true : false

export type NormalizeDim<S extends Shape, D extends number> =
    number extends D ? number
  : IsNegative<D> extends true ? DimAdd<S["length"], D>
  : D

type IsValidDim<S extends Shape, D extends number> =
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

/** The size-1 cases come first: each guard mentions one operand, so `BroadcastDim<C, 1>` reduces to `C` even while `C` is a generic. */
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

/** The generic outer product is left to the this-typed overloads on add/sub/mul/div. */
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
  // Rank-2 column broadcast [X, Y] x [X, 1]: both operands are decidable, while
  // the mirrored rule would test A's second dim and defer on a generic.
  : [ColumnBroadcast<A, B>] extends [never] ? BroadcastRev<Reverse<A>, Reverse<B>>
  : ColumnBroadcast<A, B>

export type CanBroadcast<A extends Shape, B extends Shape> =
    IsExact<A, B> extends true ? true
  : IsDynamic<A> extends true ? true
  : IsDynamic<B> extends true ? true
  : CanBroadcastRev<Reverse<A>, Reverse<B>>

/** Tests extends false, not extends true: a naked generic defers, and a deferred result must fall through to unknown. */
export type BroadcastCheck<A extends Shape, B extends Shape> = CanBroadcast<A, B> extends false
  ? ErrorMessage<`Cannot broadcast ${ShowShape<A>} with ${ShowShape<B>}`>
  : unknown

/** Expand-only: `CanBroadcast` is symmetric, so the target must also be exactly what broadcasting the source against it yields. */
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

/** The error branch needs a definite false, which a deferred DimEq never produces. */
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

export type ResolveView<S extends Shape, V extends readonly number[]> = ResolveViewOf<S, MutableShape<V>>

type ResolveViewOf<S extends Shape, V extends number[]> = {
  [K in keyof V]: V[K] extends -1 ? DimDiv<Prod<S>, ProductSkipNegOne<V>> : V[K]
}

export type ViewCheck<S extends Shape, V extends readonly number[]> = ViewCheckOf<S, MutableShape<V>>

type ViewCheckOf<S extends Shape, V extends number[]> =
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

// Fail-open rule for the checks below: the IsExact guards answer TS's permissive instantiation of a naked generic, so their ErrorMessage branch is unreachable.

type DimInRange<S extends Shape, D extends number> =
    IsExact<S, Shape> extends true ? true
  : IsExact<D, number> extends true ? true
  : IsValidDim<S, D> extends false ? false
  : true

type IsNegativeDim<D extends number> =
    IsExact<D, number> extends true ? false
  : `${D}` extends `-${string}` ? true
  : false

export type FlattenShape<S extends Shape, F extends number, T extends number> =
    IsDynamic<S> extends true ? number[]
  : number extends F ? number[]
  : number extends T ? number[]
  : [...Take<S, F>, Prod<Take<Drop<S, F>, Span<T, F>>>, ...Drop<S, Inc<T>>]

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

export type UnflattenShape<S extends Shape, D extends number, Sizes extends readonly number[]> =
    IsDynamic<S> extends true ? number[]
  : number extends D ? number[]
  : [...Take<S, D>, ...MutableShape<Sizes>, ...Drop<S, Inc<D>>]

/** Fail-open: under TS's permissive instantiation a product of generics collapses to 0. */
type SizesFillDim<S extends Shape, D extends number, Sizes extends Shape> =
    IsExact<S, Shape> extends true ? true
  : IsExact<Prod<Sizes>, 0> extends true ? true
  : DimEq<Prod<Sizes>, DimAt<S, D>> extends false ? false
  : true

export type UnflattenCheck<S extends Shape, D extends number, Sizes extends readonly number[]> = UnflattenCheckOf<S, D, MutableShape<Sizes>>

type UnflattenCheckOf<S extends Shape, D extends number, Sizes extends Shape> =
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

export type Permute<S extends Shape, Order extends readonly number[]> = PermuteOf<S, MutableShape<Order>>

type PermuteOf<S extends Shape, Order extends number[]> = IsDynamic<S> extends true ? number[] : {
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

export type PermuteCheck<S extends Shape, Order extends readonly number[]> = PermuteCheckOf<S, MutableShape<Order>>

type PermuteCheckOf<S extends Shape, Order extends number[]> =
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

export type ReduceDims<S extends Shape, Ds extends number[], Keep extends boolean = false> =
    IsDynamic<S> extends true ? number[]
  : NormalizeDims<S, Ds> extends infer Ns extends number[] ?
      number extends Ns[number] ? number[]
    : ReduceDimsWalk<S, Ns[number], Keep>
  : never

export type DimAt<S extends Shape, D extends number> = S[NormalizeDim<S, D> & keyof S] & number

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

export type Slice = number | readonly [number, number] | null | undefined

type SliceSize<C, D extends number> =
    C extends null | undefined ? D
  : C extends number ? C
  : C extends readonly [infer Start extends number, infer End extends number] ? DimSub<End, Start>
  : never

export type SliceShape<S extends Shape, Spec extends readonly Slice[]> = {
  [K in keyof S]: SliceSize<Spec[K & keyof Spec], S[K] & number>
}

/** The IsExact wildcards double as the fail-open escape. */
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

/** A generic spec entry leaves the walk deferred; widen the index to number or call narrow() to get the error. */
export type SliceCheck<S extends Shape, Spec extends readonly Slice[]> =
    IsDynamic<S> extends true ? unknown
  : SliceArityOk<S, Spec> extends false ? ErrorMessage<`slice() expects ${S["length"]} entries, got ${Spec["length"]}`>
  : SliceAxesOk<S, Spec> extends false ? SliceAxes<S, Spec>
  : unknown

type SliceArityOk<S extends Shape, Spec extends readonly Slice[]> =
    IsExact<S, Shape> extends true ? true
  : Spec["length"] extends S["length"] ? true
  : false

type SliceAxesOk<S extends Shape, Spec extends readonly Slice[]> =
    IsExact<S, Shape> extends true ? true
  : [SliceAxes<S, Spec>] extends [ErrorMessage<string>] ? false
  : true

/** The value twin throws the same sentence, so a shape that compiles cannot be rejected at run time. */
export type NarrowCheck<S extends Shape, D extends number, Start extends number, L extends number> =
    IsDynamic<S> extends true ? unknown
  : IsExact<D, number> extends true ? unknown
  : IsExact<Start, number> extends true ? unknown
  : IsExact<L, number> extends true ? unknown
  : DimInRange<S, D> extends false ? ErrorMessage<`Dimension ${D} is out of range for shape ${ShowShape<S>}`>
  : IsNegativeDim<Start> extends true ? ErrorMessage<`narrow: start ${Start} is negative`>
  : IsNegativeDim<L> extends true ? ErrorMessage<`narrow: length ${L} is negative`>
  : FitsWithin<DimAdd<Start, L>, DimAt<S, D>> extends false ? ErrorMessage<`narrow(${D}, ${Start}, ${L}) is out of range for ${ShowShape<S>}`>
  : unknown

/** `select(dim, i)` is `[B, T, E] -> [B, E]`: the selected axis is gone, the others keep their order. */
export type SelectShape<S extends Shape, D extends number> =
    IsDynamic<S> extends true ? Shape
  : NormalizeDim<S, D> extends infer I extends number ?
      number extends I ? Shape
    : RemoveAt<S, I>
  : never

/** The value twin rejects the same index, so a selection that compiles cannot be out of range at run time. */
export type SelectCheck<S extends Shape, D extends number, I extends number> =
    IsDynamic<S> extends true ? unknown
  : IsExact<D, number> extends true ? unknown
  : IsExact<I, number> extends true ? unknown
  : DimInRange<S, D> extends false ? ErrorMessage<`Dimension ${D} is out of range for shape ${ShowShape<S>}`>
  : IsNegativeDim<I> extends true ? ErrorMessage<`select: index ${I} is negative`>
  : FitsWithin<I, DimAt<S, D>> extends false ? ErrorMessage<`select(${D}, ${I}) is out of range for ${ShowShape<S>}`>
  : unknown

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

type ShapesOf<T extends readonly unknown[]> = {
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

declare const INDEX: unique symbol

export type IndexTensor<S extends Shape> = Tensor<S> & { readonly [INDEX]: true }

export type IndexCheck<T> = T extends { readonly [INDEX]: true } ? unknown
  : ErrorMessage<"index tensors must be int32/int64, use t.toIndex() or Tensor.indices()">

/** For shapes that are true but not derivable; never a way past a `*Check` that fires. */
export function assertChecked<S2 extends Shape, C = unknown>(t: AnyTensor): Tensor<S2> & C {
  return t as Tensor<S2> & C
}

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

export function reduceShape(
  shape: readonly number[],
  dim: number,
  keepdim: boolean,
): number[] {
  return keepdim
    ? shape.map((s, i) => (i === dim ? 1 : s))
    : shape.filter((_, i) => i !== dim)
}

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

export function resizeDim(
  shape: readonly number[],
  dim: number,
  length: number,
): number[] {
  return shape.map((s, i) => (i === dim ? length : s))
}

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

export function permuteShape(
  shape: readonly number[],
  order: readonly number[],
): number[] {
  return order.map(i => shape[i]!)
}

/** Expand-only: `to` must be exactly what broadcasting `from` against it yields. */
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

type ConvSpan<H extends number, K extends number, P extends number> = DimSub<DimAdd<H, DimMul<2, P>>, K>

/** floor and trunc agree here only because ConvCheck rules out a negative span. */
export type ConvOut<H extends number, K extends number, S extends number, P extends number> =
    number extends H ? number
  : number extends K ? number
  : number extends S ? number
  : number extends P ? number
  : DimAdd<DimDiv<ConvSpan<H, K, P>, S>, 1>

export type PoolOut<H extends number, K extends number, S extends number> = ConvOut<H, K, S, 0>

/** Tests the span, never the quotient: Numbers.Div truncates toward zero, which would diverge from floor. */
type ConvFits<H extends number, K extends number, S extends number, P extends number> =
    IsExact<H, number> extends true ? true
  : IsExact<K, number> extends true ? true
  : IsExact<S, number> extends true ? true
  : IsExact<P, number> extends true ? true
  : H extends H ? (`${ConvSpan<H, K, P>}` extends `-${string}` ? false : true)
  : true

export type ConvCheck<
  H extends number,
  K extends number,
  S extends number,
  P extends number,
  Axis extends string = "spatial",
> = ConvFits<H, K, S, P> extends false ? ErrorMessage<`conv: kernel ${K} with padding ${P} does not fit a ${Axis} extent of ${H}`> : unknown

export type FlattenFrom<S extends Shape> = S extends [infer B extends number, ...infer R extends number[]] ? [B, Prod<R>] : S

export function ConvOut<
  const H extends number,
  const K extends number,
  const S extends number,
  const P extends number,
>(h: H, k: K, s: S, p: P): ConvOut<H, K, S, P> {
  return (Math.trunc((h + 2 * p - k) / s) + 1) as any
}

export function PoolOut<const H extends number, const K extends number, const S extends number>(
  h: H,
  k: K,
  s: S,
): PoolOut<H, K, S> {
  return ConvOut(h, k, s, 0) as any
}

export function flattenFrom<const S extends Shape>(s: S): FlattenFrom<S> {
  if (s.length === 0) return [...s] as any
  return [s[0]!, prod(s.slice(1))] as any
}
