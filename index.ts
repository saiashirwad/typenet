import type { Numbers, Call } from "hotscript"

export const zeroWidthSpace = " "
type ZeroWidthSpace = typeof zeroWidthSpace

export type ErrorMessage<message extends string = string> =
  `${message}${ZeroWidthSpace}`

type Clean<T> = {
  [K in keyof T]: T[K]
} & unknown

namespace Math {
  export type Add<
    A extends number,
    B extends number
  > = Call<Numbers.Add<A, B>>

  export type Mul<
    A extends number,
    B extends number
  > = Call<Numbers.Mul<A, B>>

  export type MulTuple<
    A extends number[],
    acc extends number = 1
  > =
    A["length"] extends 0 ? acc
    : A extends (
      [infer X extends number, ...infer Xs extends number[]]
    ) ?
      MulTuple<Xs, Mul<acc, X>>
    : never
}

type Merge<A, B> = Clean<
  {
    [K in keyof A as K extends keyof B ? never : K]: A[K]
  } & B
>

type DType = "float32" | "float64"
type ShapeType = any[]
type DeviceType = "cpu" | "gpu"

type TensorParams = {
  requires_grad: boolean
  device: DeviceType
  dtype: DType
  dims?: string[]
}

type DefaultParams = {
  requires_grad: false
  device: "cpu"
  dtype: "float32"
}

type ViewShape<
  Shape extends number[],
  acc extends string = "["
> =
  Shape["length"] extends 0 ? `${acc}]`
  : Shape extends (
    [infer X extends number, ...infer Xs extends number[]]
  ) ?
    ViewShape<
      Xs,
      `${acc}${acc extends "[" ? "" : ", "}${X}`
    >
  : never

type ValidView<Shape extends any[], View extends number[]> =
  Math.MulTuple<Shape> extends Math.MulTuple<View> ? unknown
  : `Cannot convert tensor of shape ${ViewShape<Shape>} to ${ViewShape<View>}`

type TensorToShape<
  T extends any[],
  Shape extends number[] = []
> =
  T[0] extends readonly any[] ?
    TensorToShape<T[0], [...Shape, T["length"]]>
  : T[0] extends number ? [...Shape, T["length"]]
  : never

type SqueezeShape<
  Shape extends any[],
  Acc extends any[] = []
> =
  Shape["length"] extends 0 ? Acc
  : Shape extends [infer X, ...infer Xs] ?
    X extends 1 ?
      SqueezeShape<Xs, Acc>
    : SqueezeShape<Xs, [...Acc, X]>
  : never

declare class TypedTensor<
  const Shape extends ShapeType,
  const Params extends TensorParams = Clean<DefaultParams>
> {
  static ones<const Shape extends ShapeType>(
    shape: Shape
  ): TypedTensor<Shape>

  static zeros<const Shape extends ShapeType>(
    shape: Shape
  ): TypedTensor<Shape>

  static randn<const Shape extends ShapeType>(
    shape: Shape
  ): TypedTensor<Shape>

  static tensor<const TensorValue extends any[]>(
    tensor: TensorValue
  ): TypedTensor<TensorToShape<TensorValue>>

  gpu(): TypedTensor<
    Shape,
    Merge<Params, { device: "gpu" }>
  >

  cpu(): TypedTensor<
    Shape,
    Merge<Params, { device: "cpu" }>
  >

  dtype<const D extends DType>(
    dtype: D
  ): TypedTensor<Shape, Merge<Params, { dtype: D }>>

  requires_grad(): TypedTensor<
    Shape,
    Merge<Params, { requires_grad: true }>
  >

  no_grad(): TypedTensor<
    Shape,
    Merge<Params, { requires_grad: false }>
  >

  view<const View extends number[]>(
    view: View & ValidView<Shape, View>
  ): TypedTensor<View, Params>

  squeeze(): TypedTensor<SqueezeShape<Shape>, Params>

  static stack<
    const Tensors extends TypedTensor<any>[],
    const Dim extends number,
    const Shape extends any[] = Tensors[0] extends (
      TypedTensor<infer S>
    ) ?
      S
    : never
  >(tensor: Tensors, dim: Dim): Shape
}

const asdf = TypedTensor.ones([1, 2]).squeeze()
const zeros = TypedTensor.zeros([2, 1, 5])

const haha = TypedTensor.ones([4, 4]).view([2, 2, 2, 2])

const asdfsdf = TypedTensor.stack([haha, asdf], 2)
