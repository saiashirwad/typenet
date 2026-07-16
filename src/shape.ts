import type { Call, Numbers } from "hotscript"

export const zeroWidthSpace = "​"
type ZeroWidthSpace = typeof zeroWidthSpace

export type ErrorMessage<message extends string = string> =
  `${message}${ZeroWidthSpace}`

export type Shape = number[]

export type IsDynamic<S extends Shape> =
  number[] extends S ? true : false

type NumAdd<A extends number, B extends number> =
  number extends A ? number
  : number extends B ? number
  : Call<Numbers.Add<A, B>> extends infer R extends number ?
    R
  : number

type NumMul<A extends number, B extends number> =
  number extends A ? number
  : number extends B ? number
  : Call<Numbers.Mul<A, B>> extends infer R extends number ?
    R
  : number

type NumDiv<A extends number, B extends number> =
  number extends A ? number
  : number extends B ? number
  : Call<Numbers.Div<A, B>> extends infer R extends number ?
    R
  : number

export type Reverse<
  T extends any[],
  Acc extends any[] = []
> =
  T extends [infer H, ...infer R] ? Reverse<R, [H, ...Acc]>
  : Acc

export type Init<S extends Shape> =
  S extends [...infer R extends number[], any] ? R : never

export type Last<S extends Shape> =
  S extends [...any[], infer L extends number] ? L : never

type HasDup<
  T extends number[],
  Seen extends number = never
> =
  T extends (
    [infer X extends number, ...infer Xs extends number[]]
  ) ?
    [X] extends [Seen] ?
      true
    : HasDup<Xs, Seen | X>
  : false

export type ShowShape<
  S extends Shape,
  Acc extends string = ""
> =
  IsDynamic<S> extends true ? `[...]`
  : S extends (
    [infer X extends number, ...infer Xs extends number[]]
  ) ?
    ShowShape<Xs, Acc extends "" ? `${X}` : `${Acc}, ${X}`>
  : `[${Acc}]`

export type Product<
  S extends Shape,
  Acc extends number = 1
> =
  IsDynamic<S> extends true ? number
  : S extends (
    [infer X extends number, ...infer Xs extends number[]]
  ) ?
    Product<Xs, NumMul<Acc, X>>
  : Acc

export type NumEl<S extends Shape> = Product<S>

type IsNegative<D extends number> =
  `${D}` extends `-${string}` ? true : false

export type NormalizeDim<
  S extends Shape,
  D extends number
> =
  number extends D ? number
  : IsNegative<D> extends true ? NumAdd<S["length"], D>
  : D

export type IsValidDim<S extends Shape, D extends number> =
  IsDynamic<S> extends true ? true
  : number extends D ? true
  : `${NormalizeDim<S, D>}` extends keyof S ? true
  : false

export type DimCheck<S extends Shape, D extends number> =
  IsValidDim<S, D> extends true ? unknown
  : ErrorMessage<`Dimension ${D} is out of range for shape ${ShowShape<S>}`>

type ReplaceAt<
  S extends Shape,
  I extends number,
  V extends number
> = {
  [K in keyof S]: K extends `${I}` ? V : S[K]
}

type RemoveAt<
  S extends Shape,
  I extends number,
  Acc extends Shape = []
> =
  S extends (
    [infer X extends number, ...infer Xs extends number[]]
  ) ?
    Acc["length"] extends I ?
      [...Acc, ...Xs]
    : RemoveAt<Xs, I, [...Acc, X]>
  : Acc

type InsertAt<
  S extends Shape,
  I extends number,
  V extends number,
  Acc extends Shape = []
> =
  Acc["length"] extends I ? [...Acc, V, ...S]
  : S extends (
    [infer X extends number, ...infer Xs extends number[]]
  ) ?
    InsertAt<Xs, I, V, [...Acc, X]>
  : [...Acc, V]

type IsExact<X, Y> =
  (<T>() => T extends X ? 1 : 2) extends (
    <T>() => T extends Y ? 1 : 2
  ) ?
    true
  : false

type BroadcastDim<X extends number, Y extends number> =
  IsExact<X, Y> extends true ? X
  : IsExact<X, 1> extends true ? Y
  : IsExact<Y, 1> extends true ? X
  : IsExact<X, number> extends true ? Y
  : IsExact<Y, number> extends true ? X
  : X extends Y ? X
  : never

type DimCompatible<X extends number, Y extends number> =
  IsExact<X, Y> extends true ? true
  : IsExact<X, 1> extends true ? true
  : IsExact<Y, 1> extends true ? true
  : IsExact<X, number> extends true ? true
  : IsExact<Y, number> extends true ? true
  : X extends Y ? true
  : false

type CanBroadcastRev<A extends Shape, B extends Shape> =
  A extends (
    [infer X extends number, ...infer Xs extends number[]]
  ) ?
    B extends (
      [infer Y extends number, ...infer Ys extends number[]]
    ) ?
      DimCompatible<X, Y> extends true ?
        CanBroadcastRev<Xs, Ys>
      : false
    : true
  : true

type BroadcastRev<
  A extends Shape,
  B extends Shape,
  Acc extends Shape = []
> =
  A extends (
    [infer X extends number, ...infer Xs extends number[]]
  ) ?
    B extends (
      [infer Y extends number, ...infer Ys extends number[]]
    ) ?
      BroadcastRev<Xs, Ys, [...Acc, BroadcastDim<X, Y>]>
    : BroadcastRev<Xs, [], [...Acc, X]>
  : B extends (
    [infer Y extends number, ...infer Ys extends number[]]
  ) ?
    BroadcastRev<[], Ys, [...Acc, Y]>
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

export type BroadcastCheck<
  A extends Shape,
  B extends Shape
> =
  CanBroadcast<A, B> extends true ? unknown
  : ErrorMessage<`Cannot broadcast ${ShowShape<A>} with ${ShowShape<B>}`>

export type DimEq<X extends number, Y extends number> =
  IsExact<X, Y> extends true ? true
  : IsExact<X, number> extends true ? true
  : IsExact<Y, number> extends true ? true
  : X extends Y ? true
  : false

type InnerA<A extends Shape> =
  A["length"] extends 1 ? A[0] : Last<A>

type InnerB<B extends Shape> =
  B["length"] extends 1 ? B[0] : Last<Init<B>>

type BatchDims<S extends Shape> = Init<Init<S>>

export type MatMul<A extends Shape, B extends Shape> =
  IsDynamic<A> extends true ? number[]
  : IsDynamic<B> extends true ? number[]
  : A extends [] ? never
  : B extends [] ? never
  : A["length"] extends 1 ?
    B["length"] extends 1 ?
      []
    : [...BatchDims<B>, Last<B>]
  : B["length"] extends 1 ? Init<A>
  : [
      ...Broadcast<BatchDims<A>, BatchDims<B>>,
      Last<Init<A>>,
      Last<B>
    ]

export type MatMulCheck<A extends Shape, B extends Shape> =
  IsDynamic<A> extends true ? unknown
  : IsDynamic<B> extends true ? unknown
  : A extends [] ?
    ErrorMessage<`matmul requires operands of rank >= 1`>
  : B extends [] ?
    ErrorMessage<`matmul requires operands of rank >= 1`>
  : DimEq<InnerA<A>, InnerB<B>> extends true ?
    A["length"] extends 1 ? unknown
    : B["length"] extends 1 ? unknown
    : CanBroadcast<BatchDims<A>, BatchDims<B>> extends (
      true
    ) ?
      unknown
    : ErrorMessage<`matmul: cannot broadcast batch dims of ${ShowShape<A>} with ${ShowShape<B>}`>
  : ErrorMessage<`matmul: inner dimensions do not match (${ShowShape<A>} @ ${ShowShape<B>})`>

type ProductSkipNegOne<
  V extends number[],
  Acc extends number = 1
> =
  V extends (
    [infer X extends number, ...infer Xs extends number[]]
  ) ?
    X extends -1 ?
      ProductSkipNegOne<Xs, Acc>
    : ProductSkipNegOne<Xs, NumMul<Acc, X>>
  : Acc

type CountNegOnes<
  V extends number[],
  Acc extends 1[] = []
> =
  V extends [infer X, ...infer Xs extends number[]] ?
    X extends -1 ?
      CountNegOnes<Xs, [...Acc, 1]>
    : CountNegOnes<Xs, Acc>
  : Acc["length"]

export type ResolveView<
  S extends Shape,
  V extends number[]
> = {
  [K in keyof V]: V[K] extends -1 ?
    NumDiv<Product<S>, ProductSkipNegOne<V>>
  : V[K]
}

export type ViewCheck<S extends Shape, V extends number[]> =
  CountNegOnes<V> extends 0 | 1 ?
    number extends Product<S> ? unknown
    : number extends ProductSkipNegOne<V> ? unknown
    : CountNegOnes<V> extends 0 ?
      Product<V> extends Product<S> ?
        unknown
      : ErrorMessage<`Cannot view tensor of shape ${ShowShape<S>} as ${ShowShape<V>} (${Product<S>} vs ${Product<V>} elements)`>
    : NumMul<
      ProductSkipNegOne<V>,
      NumDiv<Product<S>, ProductSkipNegOne<V>>
    > extends Product<S> ?
      unknown
    : ErrorMessage<`Cannot infer -1 dim: ${Product<S>} elements do not divide evenly into ${ShowShape<S>} -> ${ShowShape<V>}`>
  : ErrorMessage<`Only one -1 dim is allowed in view()`>

type SwapDims<
  S extends Shape,
  I extends number,
  J extends number
> =
  number extends I ? number[]
  : number extends J ? number[]
  : {
      [K in keyof S]: K extends `${I}` ? S[J]
      : K extends `${J}` ? S[I]
      : S[K]
    }

export type Transpose<
  S extends Shape,
  D0 extends number,
  D1 extends number
> =
  IsDynamic<S> extends true ? number[]
  : SwapDims<S, NormalizeDim<S, D0>, NormalizeDim<S, D1>>

export type TransposeCheck<
  S extends Shape,
  D0 extends number,
  D1 extends number
> =
  IsValidDim<S, D0> extends true ?
    IsValidDim<S, D1> extends true ?
      unknown
    : ErrorMessage<`Dimension ${D1} is out of range for shape ${ShowShape<S>}`>
  : ErrorMessage<`Dimension ${D0} is out of range for shape ${ShowShape<S>}`>

export type Permute<
  S extends Shape,
  Order extends number[]
> =
  IsDynamic<S> extends true ? number[]
  : {
      [K in keyof Order]: S[NormalizeDim<
        S,
        Order[K]
      > extends infer I extends number ?
        I
      : never]
    }

type AllValidDims<S extends Shape, Ds extends number[]> =
  Ds extends (
    [infer X extends number, ...infer Xs extends number[]]
  ) ?
    IsValidDim<S, X> extends true ?
      AllValidDims<S, Xs>
    : false
  : true

type NormalizeDims<S extends Shape, Ds extends number[]> = {
  [K in keyof Ds]: NormalizeDim<S, Ds[K]>
}

export type PermuteCheck<
  S extends Shape,
  Order extends number[]
> =
  IsDynamic<S> extends true ? unknown
  : Order["length"] extends S["length"] ?
    AllValidDims<S, Order> extends true ?
      HasDup<
        NormalizeDims<S, Order> extends (
          infer N extends number[]
        ) ?
          N
        : never
      > extends false ?
        unknown
      : ErrorMessage<`permute(${ShowShape<Order>}) repeats a dimension`>
    : ErrorMessage<`permute(${ShowShape<Order>}) has a dim out of range for ${ShowShape<S>}`>
  : ErrorMessage<`permute() expects ${S["length"]} dims, got ${Order["length"]}`>

export type Squeeze<
  S extends Shape,
  Acc extends Shape = []
> =
  IsDynamic<S> extends true ? number[]
  : S extends (
    [infer X extends number, ...infer Xs extends number[]]
  ) ?
    X extends 1 ?
      Squeeze<Xs, Acc>
    : Squeeze<Xs, [...Acc, X]>
  : Acc

export type SqueezeDim<S extends Shape, D extends number> =
  IsDynamic<S> extends true ? number[]
  : NormalizeDim<S, D> extends infer I extends number ?
    number extends I ?
      number[]
    : RemoveAt<S, I>
  : never

export type SqueezeDimCheck<
  S extends Shape,
  D extends number
> =
  IsValidDim<S, D> extends true ?
    IsDynamic<S> extends true ? unknown
    : NormalizeDim<S, D> extends infer I extends number ?
      number extends I ? unknown
      : S[I & keyof S] extends 1 ? unknown
      : ErrorMessage<`Cannot squeeze dim ${D} of ${ShowShape<S>}: size is not 1`>
    : never
  : ErrorMessage<`Dimension ${D} is out of range for shape ${ShowShape<S>}`>

export type Unsqueeze<S extends Shape, D extends number> =
  IsDynamic<S> extends true ? number[]
  : NormalizeUnsqueezeDim<S, D> extends (
    infer I extends number
  ) ?
    number extends I ?
      number[]
    : InsertAt<S, I, 1>
  : never

type NormalizeUnsqueezeDim<
  S extends Shape,
  D extends number
> =
  number extends D ? number
  : IsNegative<D> extends true ?
    NumAdd<NumAdd<S["length"], 1>, D>
  : D

export type UnsqueezeCheck<
  S extends Shape,
  D extends number
> =
  IsDynamic<S> extends true ? unknown
  : NormalizeUnsqueezeDim<S, D> extends (
    infer I extends number
  ) ?
    number extends I ? unknown
    : `${I}` extends keyof S | `${S["length"]}` ? unknown
    : ErrorMessage<`Dimension ${D} is out of range for unsqueeze on ${ShowShape<S>}`>
  : never

export type ReduceDim<
  S extends Shape,
  D extends number,
  Keep extends boolean = false
> =
  IsDynamic<S> extends true ? number[]
  : NormalizeDim<S, D> extends infer I extends number ?
    number extends I ? number[]
    : Keep extends true ? ReplaceAt<S, I, 1>
    : RemoveAt<S, I>
  : never

export type Stack<
  S extends Shape,
  N extends number,
  D extends number
> =
  IsDynamic<S> extends true ? number[]
  : NormalizeUnsqueezeDim<S, D> extends (
    infer I extends number
  ) ?
    number extends I ?
      number[]
    : InsertAt<S, I, N>
  : never

type CatDim<
  A extends Shape,
  B extends Shape,
  I extends number
> = ReplaceAt<
  A,
  I,
  NumAdd<A[I & keyof A] & number, B[I & keyof B] & number>
>

export type Cat<
  A extends Shape,
  B extends Shape,
  D extends number
> =
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
  Pos extends 1[] = []
> =
  A extends (
    [infer X extends number, ...infer Xs extends number[]]
  ) ?
    B extends (
      [infer Y extends number, ...infer Ys extends number[]]
    ) ?
      Pos["length"] extends I ?
        EqualExceptAt<Xs, Ys, I, [...Pos, 1]>
      : DimEq<X, Y> extends true ?
        EqualExceptAt<Xs, Ys, I, [...Pos, 1]>
      : false
    : false
  : B extends [] ? true
  : false

export type CatCheck<
  A extends Shape,
  B extends Shape,
  D extends number
> =
  IsDynamic<A> extends true ? unknown
  : IsDynamic<B> extends true ? unknown
  : A["length"] extends B["length"] ?
    IsValidDim<A, D> extends true ?
      NormalizeDim<A, D> extends infer I extends number ?
        number extends I ? unknown
        : EqualExceptAt<A, B, I> extends true ? unknown
        : ErrorMessage<`cat: shapes ${ShowShape<A>} and ${ShowShape<B>} differ outside dim ${D}`>
      : never
    : ErrorMessage<`Dimension ${D} is out of range for shape ${ShowShape<A>}`>
  : ErrorMessage<`cat: tensors must have the same rank (${ShowShape<A>} vs ${ShowShape<B>})`>

export type InferShape<T, Depth extends 1[] = []> =
  Depth["length"] extends 12 ? number[]
  : T extends number ? []
  : T extends readonly any[] ?
    [T["length"], ...InferShape<T[0], [...Depth, 1]>]
  : never

export type NestedArray<S extends Shape> =
  IsDynamic<S> extends true ? any
  : S extends [] ? number
  : S extends [any, ...infer Rest extends number[]] ?
    NestedArray<Rest>[]
  : never
