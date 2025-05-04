import type { Numbers, Call } from "hotscript"

type Clean<T> = { [K in keyof T]: T } & unknown

namespace Math {
  export type Add<
    A extends number,
    B extends number
  > = Call<Numbers.Add<A, B>>
}

type ShapeType = any[]
type DeviceType = "cpu" | "gpu"

type TensorParams = {
  requires_grad: boolean
  device: DeviceType
}

declare class Tensor<
  const Shape extends ShapeType,
  const Params extends TensorParams = {
    requires_grad: false
    device: "cpu"
  }
> {
  static ones<const Shape extends ShapeType>(
    shape: Shape
  ): Tensor<Shape>

  static zeros<const Shape extends ShapeType>(
    shape: Shape
  ): Tensor<Shape>

  static randn<const Shape extends ShapeType>(
    shape: Shape
  ): Tensor<Shape>

  gpu(): Tensor<
    Shape,
    {
      device: "gpu"
      requires_grad: Params["requires_grad"]
    }
  >

  cpu(): Tensor<
    Shape,
    {
      device: "cpu"
      requires_grad: Params["requires_grad"]
    }
  >

  requires_grad(): Tensor<
    Shape,
    { device: Params["device"]; requires_grad: true }
  >

  no_grad(): Tensor<
    Shape,
    { device: Params["device"]; requires_grad: false }
  >
}

const asdf = Tensor.ones([1, 2])
const zeros = Tensor.zeros([2, 1, 5])

