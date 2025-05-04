import type { Numbers, Call } from "hotscript"

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

type DType = "int32" | "int64" | "float32" | "float64"
type ShapeType = any[]
type DeviceType = "cpu" | "gpu"

type TensorParams = {
  requires_grad: boolean
  device: DeviceType
  dtype: DType
  dims?: string[]
}

declare class Err<const message extends string> {}

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

// type SqueezeShape<Shape extends any[]> = {
//   [K in keyof Shape as K extends `${number}` ?
//     Shape[K] extends 1 ?
//       never
//     : K
//   : never]: Shape[K]
// }

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

type sdf = SqueezeShape<[2, 1, 2, 3]>

declare class TN<
  const Shape extends ShapeType,
  const Params extends TensorParams = {
    requires_grad: false
    device: "cpu"
    dtype: "float32"
  }
> {
  static ones<const Shape extends ShapeType>(
    shape: Shape
  ): TN<Shape>

  static zeros<const Shape extends ShapeType>(
    shape: Shape
  ): TN<Shape>

  static randn<const Shape extends ShapeType>(
    shape: Shape
  ): TN<Shape>

  static tensor<const TensorValue extends any[]>(
    tensor: TensorValue
  ): TN<TensorToShape<TensorValue>>

  gpu(): TN<Shape, Merge<Params, { device: "gpu" }>>

  cpu(): TN<Shape, Merge<Params, { device: "cpu" }>>

  dtype<const D extends DType>(
    dtype: D
  ): TN<Shape, Merge<Params, { dtype: D }>>

  requires_grad(): TN<
    Shape,
    Merge<Params, { requires_grad: true }>
  >

  no_grad(): TN<
    Shape,
    Merge<Params, { requires_grad: false }>
  >

  view<const View extends number[]>(
    view: View & ValidView<Shape, View>
  ): TN<View, Params>

  squeeze(): TN<SqueezeShape<Shape>, Params>
}

const asdf = TN.ones([1, 2]).squeeze()
const zeros = TN.zeros([2, 1, 5])

const haha = TN.ones([4, 4]).view([2, 2, 2, 2])
