import type { AnyTensor } from "./tensor.ts"

export type DType = "float32" | "float64"

export type UnaryOp =
  | "pow"
  | "neg"
  | "exp"
  | "log"
  | "sqrt"
  | "abs"
  | "relu"
  | "leakyRelu"
  | "sigmoid"
  | "tanh"
  | "scalePowGrad"

export type BinaryOp =
  | "add"
  | "sub"
  | "mul"
  | "div"
  | "maximum"
  | "minimum"
  | "gt"
  | "ge"
  | "lt"
  | "le"
  | "eq"
  | "negDiv"
  | "halfDiv"
  | "mulSign"
  | "reluGrad"
  | "leakyReluGrad"
  | "sigmoidGrad"
  | "tanhGrad"

export type ReduceOp = "sum" | "max" | "argmax"

export type RandomKind = "uniform" | "normal"

export type TypedArray = Float32Array | Float64Array

type CpuStorage = {
  readonly kind: "cpu"
  readonly data: TypedArray
}

type LazyNodeBody =
  | {
    op: "binary"
    kind: BinaryOp
    parameter: number
    a: AnyTensor
    b: AnyTensor
  }
  | {
    op: "unary"
    kind: UnaryOp
    parameter: number
    input: AnyTensor
  }
  | { op: "matmul"; a: AnyTensor; b: AnyTensor }
  | {
    op: "reduce"
    kind: ReduceOp
    dim: number
    keepdim: boolean
    input: AnyTensor
  }
  | {
    op: "reduceAll"
    kind: "sum" | "max"
    input: AnyTensor
  }
  | { op: "broadcastTo"; input: AnyTensor }
  | { op: "permute"; order: number[]; input: AnyTensor }
  | { op: "view"; input: AnyTensor }
  | {
    op: "narrow"
    dim: number
    start: number
    length: number
    input: AnyTensor
  }
  | { op: "cat"; a: AnyTensor; b: AnyTensor; dim: number }
  | { op: "oneHot"; classes: number; input: AnyTensor }
  | {
    op: "indexSelect"
    dim: number
    input: AnyTensor
    index: AnyTensor
  }
  | {
    op: "scatterAdd"
    dim: number
    length: number
    input: AnyTensor
    index: AnyTensor
  }
  | {
    op: "random"
    kind: RandomKind
    // Identifies this node's own stream, so two random nodes in one
    // graph never draw the same numbers. Fixed when the node is
    // built, which keeps a compiled graph's structure stable.
    stream: number
  }

type LazyNode = LazyNodeBody & {
  shape: number[]
  dtype: DType
}

type LazyStorage = {
  readonly kind: "lazy"
  readonly node: LazyNode
  cache: AnyTensor | null
}

type TensorStorage = CpuStorage | LazyStorage

export type { CpuStorage, LazyNode, LazyNodeBody, LazyStorage, TensorStorage }

function arrayCtor(dtype: DType) {
  return dtype === "float64" ? Float64Array : Float32Array
}

function prod(xs: readonly number[]): number {
  let p = 1
  for (const x of xs) p *= x
  return p
}

function shapesEqual(
  a: readonly number[],
  b: readonly number[],
): boolean {
  return (
    a.length === b.length && a.every((x, i) => x === b[i])
  )
}

function showShape(s: readonly number[]): string {
  return `[${s.join(", ")}]`
}

function contiguousStrides(
  shape: readonly number[],
): number[] {
  const strides = new Array<number>(shape.length)
  let acc = 1
  for (let i = shape.length - 1; i >= 0; i--) {
    strides[i] = acc
    acc *= shape[i]!
  }
  return strides
}

function broadcastStrides(
  from: readonly number[],
  to: readonly number[],
): number[] {
  const strides = contiguousStrides(from)
  const out = new Array<number>(to.length).fill(0)
  const offset = to.length - from.length
  for (let i = 0; i < from.length; i++) {
    out[offset + i] = from[i] === 1 ? 0 : strides[i]!
  }
  return out
}

function normalizeDim(
  dim: number,
  rank: number,
  extra = 0,
): number {
  const d = dim < 0 ? rank + extra + dim : dim
  if (d < 0 || d >= rank + extra) {
    throw new Error(
      `Dimension ${dim} out of range for rank ${rank}`,
    )
  }
  return d
}

export { arrayCtor, broadcastStrides, contiguousStrides, normalizeDim, prod, shapesEqual, showShape }
