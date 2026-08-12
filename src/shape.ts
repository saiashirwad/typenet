import type { Call, Numbers } from "hotscript"

export const zeroWidthSpace = "​"
type ZeroWidthSpace = typeof zeroWidthSpace

export type ErrorMessage<message extends string = string> = `${message}${ZeroWidthSpace}`

export type Shape = number[]

export type IsDynamic<S extends Shape> = number[] extends S ? true : false

/**
 * Dimension arithmetic, for shapes derived from a generic dim — a percept
 * of `3 * C + 1` channels, say.
 *
 * These are smart constructors: unconditional rewrite rules first, literal
 * folding last. One law of type-level evaluation drives the design:
 *
 *   The first guard that mentions a naked generic defers the WHOLE chain.
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
  : DimAddFold<A, B>

type DimAddFold<A extends number, B extends number> =
    number extends A ? number
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
  : DimMulFold<A, B>

type DimMulFold<A extends number, B extends number> =
    number extends A ? number
  : number extends B ? number
  : Call<Numbers.Mul<A, B>> extends infer R extends number ? R
  : number

export function DimMul<const A extends number, const B extends number>(a: A, b: B): DimMul<A, B> {
  return (a * b) as any
}

type NumDiv<A extends number, B extends number> =
    IsExact<B, 1> extends true ? A
  : number extends A ? number
  : number extends B ? number
  : Call<Numbers.Div<A, B>> extends infer R extends number ? R
  : number

export type Reverse<T extends any[], Acc extends any[] = []> = T extends [infer H, ...infer R] ? Reverse<R, [H, ...Acc]> : Acc

export type Init<S extends Shape> = S extends [...infer R extends number[], any] ? R : never

export type Last<S extends Shape> = S extends [...any[], infer L extends number] ? L : never

type HasDup<T extends number[], Seen extends number = never> =
    T extends [
      infer X extends number,
      ...infer Xs extends number[],
    ] ?
      [X] extends [Seen] ? true
    : HasDup<Xs, Seen | X>
  : false

export type ShowShape<S extends Shape, Acc extends string = ""> =
    IsDynamic<S> extends true ? `[...]`
  : S extends [infer X extends number, ...infer Xs extends number[]] ? ShowShape<Xs, Acc extends "" ? `${X}` : `${Acc}, ${X}`>
  : `[${Acc}]`

export type Product<S extends Shape, Acc extends number = 1> =
    IsDynamic<S> extends true ? number
  : S extends [infer X extends number, ...infer Xs extends number[]] ? Product<Xs, DimMul<Acc, X>>
  : Acc

export type NumEl<S extends Shape> = Product<S>

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
 * Broadcast one dim — the single rewrite system for broadcast
 * compatibility: `BroadcastRev` accumulates it into the result shape,
 * `CanBroadcastRev` detects its `never`. (There used to be a separate
 * `DimCompatible` chain to keep in sync by hand; a missing wildcard case in
 * one of the two was how `[number, 2] + [4, 2]` once became a silent
 * `[never, 2]`.)
 *
 * Guard order obeys the law on `DimAdd`: the size-1 cases come first
 * because each guard mentions only ONE operand, so `BroadcastDim<C, 1>`
 * reduces to `C` even while `C` is an unresolved generic — the case that
 * keeps `Tensor<[N, C]>.mul(column)` typed as `Tensor<[N, C]>` once
 * instantiated. The price: `BroadcastDim<N, N>` defers behind the first
 * guard, but every branch of that deferred chain yields `N`.
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

export type Broadcast<A extends Shape, B extends Shape> =
    IsExact<A, B> extends true ? A
  : IsDynamic<A> extends true ? number[]
  : IsDynamic<B> extends true ? number[]
  : BroadcastRev<Reverse<A>, Reverse<B>>

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

export type DimEq<X extends number, Y extends number> =
    IsExact<X, Y> extends true ? true
  : IsExact<X, number> extends true ? true
  : IsExact<Y, number> extends true ? true
  : X extends Y ? true
  : false

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
  [K in keyof V]: V[K] extends -1 ? NumDiv<Product<S>, ProductSkipNegOne<V>> : V[K]
}

export type ViewCheck<S extends Shape, V extends number[]> =
    CountNegOnes<V> extends 0 | 1 ?
      number extends Product<S> ? unknown
    : number extends ProductSkipNegOne<V> ? unknown
    : CountNegOnes<V> extends 0 ?
        Product<V> extends Product<S> ? unknown
      : ErrorMessage<`Cannot view tensor of shape ${ShowShape<S>} as ${ShowShape<V>} (${Product<S>} vs ${Product<V>} elements)`>
    : DimMul<
      ProductSkipNegOne<V>,
      NumDiv<Product<S>, ProductSkipNegOne<V>>
    > extends Product<S> ? unknown
    : ErrorMessage<`Cannot infer -1 dim: ${Product<S>} elements do not divide evenly into ${ShowShape<S>} -> ${ShowShape<V>}`>
  : ErrorMessage<`Only one -1 dim is allowed in view()`>

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

/**
 * Shape of the gather/scatter pair: dim `D` is resized to `L` and every
 * other dim survives. `indexSelect` sets `L` to the index length,
 * `scatterAdd` to the requested output length.
 */
export type ResizeDim<S extends Shape, D extends number, L extends number> =
    IsDynamic<S> extends true ? number[]
  : NormalizeDim<S, D> extends infer I extends number ?
      number extends I ? number[]
    : ReplaceAt<S, I, L>
  : never

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
