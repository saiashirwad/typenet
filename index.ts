import type { Numbers, Call } from "hotscript"

type Clean<T> = { [K in keyof T]: T } & unknown

namespace Math {
  export type Add<
    A extends number,
    B extends number
  > = Call<Numbers.Add<A, B>>
}

type DType = "int32" | "int64" | "float32" | "float64"
type ShapeType = any[]
type DeviceType = "cpu" | "gpu"

type TensorParams = {
  requires_grad: boolean
  device: DeviceType
  dtype: DType
}

type TensorToShape<
  T extends any[],
  Shape extends number[] = []
> =
  T[0] extends readonly any[] ?
    TensorToShape<T[0], [...Shape, T["length"]]>
  : T[0] extends number ? [...Shape, T["length"]]
  : never

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

  gpu(): TN<
    Shape,
    {
      device: "gpu"
      requires_grad: Params["requires_grad"]
      dtype: "float32"
    }
  >

  cpu(): TN<
    Shape,
    {
      device: "cpu"
      requires_grad: Params["requires_grad"]
      dtype: "float32"
    }
  >

  requires_grad(): TN<
    Shape,
    {
      device: Params["device"]
      requires_grad: true
      dtype: "float32"
    }
  >

  no_grad(): TN<
    Shape,
    {
      device: Params["device"]
      requires_grad: false
      dtype: "float32"
    }
  >
}

const asdf = TN.ones([1, 2])
const zeros = TN.zeros([2, 1, 5])

const lol = TN.tensor([
  [1, 2, 3, 5],
  [3, 4, 5, 5]
]).gpu()
