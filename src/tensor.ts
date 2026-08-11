import { Operator } from "tsover-runtime"
import * as nativeBackend from "./backends/native.ts"
import type {
  Broadcast,
  BroadcastCheck,
  Cat,
  CatCheck,
  DimCheck,
  ErrorMessage,
  InferShape,
  MatMul,
  MatMulCheck,
  NestedArray,
  Permute,
  PermuteCheck,
  ReduceDim,
  ResizeDim,
  ResolveView,
  Shape,
  Squeeze,
  SqueezeDim,
  SqueezeDimCheck,
  Stack,
  Transpose,
  TransposeCheck,
  Unsqueeze,
  UnsqueezeCheck,
  ViewCheck
} from "./shape.ts"

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

export type TensorParams = {
  requires_grad: boolean
  dtype: DType
}

export type DefaultParams = {
  requires_grad: false
  dtype: "float32"
}

type Clean<T> = { [K in keyof T]: T[K] } & unknown

type Merge<A, B> = Clean<
  {
    [K in keyof A as K extends keyof B ? never : K]: A[K]
  } & B
>

export type ShapeOf<T> =
  T extends Tensor<infer S, any> ? S : never
export type ParamsOf<T> =
  T extends Tensor<any, infer P> ? P : never

type TypedArray = Float32Array | Float64Array

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

export type NestedNumbers =
  number | readonly NestedNumbers[]

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
  b: readonly number[]
): boolean {
  return (
    a.length === b.length && a.every((x, i) => x === b[i])
  )
}

function showShape(s: readonly number[]): string {
  return `[${s.join(", ")}]`
}

function contiguousStrides(
  shape: readonly number[]
): number[] {
  const strides = new Array<number>(shape.length)
  let acc = 1
  for (let i = shape.length - 1; i >= 0; i--) {
    strides[i] = acc
    acc *= shape[i]!
  }
  return strides
}

export function broadcastShapes(
  a: readonly number[],
  b: readonly number[]
): number[] {
  const rank = Math.max(a.length, b.length)
  const out = new Array<number>(rank)
  for (let i = 0; i < rank; i++) {
    const da = a[a.length - 1 - i] ?? 1
    const db = b[b.length - 1 - i] ?? 1
    if (da !== db && da !== 1 && db !== 1)
      throw new Error(
        `Cannot broadcast ${showShape(a)} with ${showShape(b)}`
      )
    out[rank - 1 - i] = Math.max(da, db)
  }
  return out
}

function broadcastStrides(
  from: readonly number[],
  to: readonly number[]
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
  extra = 0
): number {
  const d = dim < 0 ? rank + extra + dim : dim
  if (d < 0 || d >= rank + extra)
    throw new Error(
      `Dimension ${dim} out of range for rank ${rank}`
    )
  return d
}

type AnyTensor = Tensor<any, any>

interface GradNode {
  name: string
  inputs: AnyTensor[]

  backward: (grad: AnyTensor) => (AnyTensor | null)[]
}

let gradEnabled = true

export function noGrad<T>(fn: () => T): T {
  const prev = gradEnabled
  gradEnabled = false
  try {
    return fn()
  } finally {
    gradEnabled = prev
  }
}

let lazyMode = false

export function configure(options: {
  lazy?: boolean
  /**
   * Reseeds `uniform()` / `normal()`. A run replays identically given
   * the same seed and the same sequence of operations. Does not affect
   * `Tensor.rand` / `Tensor.randn`, which use `Math.random`.
   */
  seed?: number
}): void {
  if (options.lazy !== undefined) lazyMode = options.lazy
  if (options.seed !== undefined) {
    randomSeed = options.seed >>> 0
    streamCounter = 0
  }
}

export function isLazy(): boolean {
  return lazyMode
}

function eagerly<T>(fn: () => T): T {
  const prev = lazyMode
  lazyMode = false
  try {
    return fn()
  } finally {
    lazyMode = prev
  }
}

function sumTo(t: AnyTensor, shape: number[]): AnyTensor {
  if (shapesEqual(t.shape, shape)) return t
  let out = t

  while (out.shape.length > shape.length)
    out = rawSum(out, 0, false)

  for (let i = 0; i < shape.length; i++)
    if (shape[i] === 1 && out.shape[i] !== 1)
      out = rawSum(out, i, true)
  return out
}

function makeRaw(
  data: TypedArray,
  shape: readonly number[],
  dtype: DType
): AnyTensor {
  return makeStorage({ kind: "cpu", data }, shape, dtype)
}

function makeStorage(
  storage: TensorStorage,
  shape: readonly number[],
  dtype: DType
): AnyTensor {
  return new (Tensor as any)(
    storage,
    [...shape],
    dtype,
    INTERNAL
  )
}

function applyBinary(
  op: BinaryOp,
  x: number,
  y: number,
  parameter: number
): number {
  switch (op) {
    case "add":
      return x + y
    case "sub":
      return x - y
    case "mul":
      return x * y
    case "div":
      return x / y
    case "maximum":
      return Math.max(x, y)
    case "minimum":
      return Math.min(x, y)
    case "gt":
      return x > y ? 1 : 0
    case "ge":
      return x >= y ? 1 : 0
    case "lt":
      return x < y ? 1 : 0
    case "le":
      return x <= y ? 1 : 0
    case "eq":
      return x === y ? 1 : 0
    case "negDiv":
      return -x / y
    case "halfDiv":
      return (0.5 * x) / y
    case "mulSign":
      return x * Math.sign(y)
    case "reluGrad":
      return y > 0 ? x : 0
    case "leakyReluGrad":
      return y > 0 ? x : parameter * x
    case "sigmoidGrad":
      return x * y * (1 - y)
    case "tanhGrad":
      return x * (1 - y * y)
  }
}

function applyUnary(
  op: UnaryOp,
  x: number,
  parameter: number
): number {
  switch (op) {
    case "pow":
      return x ** parameter
    case "neg":
      return -x
    case "exp":
      return Math.exp(x)
    case "log":
      return Math.log(x)
    case "sqrt":
      return Math.sqrt(x)
    case "abs":
      return Math.abs(x)
    case "relu":
      return x > 0 ? x : 0
    case "leakyRelu":
      return x > 0 ? x : parameter * x
    case "sigmoid":
      return 1 / (1 + Math.exp(-x))
    case "tanh":
      return Math.tanh(x)
    case "scalePowGrad":
      return parameter * x ** (parameter - 1)
  }
}

function rawBinary(
  a: AnyTensor,
  b: AnyTensor,
  op: BinaryOp,
  parameter = 0
): AnyTensor {
  const outShape = broadcastShapes(a.shape, b.shape)
  const dtype: DType =
    a.dtype === "float64" || b.dtype === "float64" ?
      "float64"
    : "float32"
  const rank = outShape.length
  const sa = broadcastStrides(a.shape, outShape)
  const sb = broadcastStrides(b.shape, outShape)
  if (lazyMode)
    return makeLazy(
      { op: "binary", kind: op, parameter, a, b },
      outShape,
      dtype
    )
  const n = prod(outShape)
  const out = new (arrayCtor(dtype))(n)
  const idx = new Array(rank).fill(0)
  let offA = 0
  let offB = 0
  const ad = a.data
  const bd = b.data
  for (let i = 0; i < n; i++) {
    out[i] = applyBinary(
      op,
      ad[offA]!,
      bd[offB]!,
      parameter
    )
    for (let d = rank - 1; d >= 0; d--) {
      idx[d]++
      offA += sa[d]!
      offB += sb[d]!
      if (idx[d] < outShape[d]!) break
      idx[d] = 0
      offA -= sa[d]! * outShape[d]!
      offB -= sb[d]! * outShape[d]!
    }
  }
  return makeRaw(out, outShape, dtype)
}

function rawUnary(
  a: AnyTensor,
  op: UnaryOp,
  parameter = 0
): AnyTensor {
  if (lazyMode)
    return makeLazy(
      { op: "unary", kind: op, parameter, input: a },
      a.shape,
      a.dtype
    )
  const out = new (arrayCtor(a.dtype))(a.data.length)
  for (let i = 0; i < a.data.length; i++)
    out[i] = applyUnary(op, a.data[i]!, parameter)
  return makeRaw(out, a.shape, a.dtype)
}

function rawSum(
  a: AnyTensor,
  dim: number,
  keepdim: boolean
): AnyTensor {
  return rawReduce(a, dim, keepdim, "sum")
}

function rawReduce(
  a: AnyTensor,
  dim: number,
  keepdim: boolean,
  op: ReduceOp
): AnyTensor {
  const d = normalizeDim(dim, a.shape.length)
  const outShape = a.shape.filter(
    (_: number, i: number) => i !== d
  )
  const keepShape = a.shape.map((s: number, i: number) =>
    i === d ? 1 : s
  )
  if (lazyMode)
    return makeLazy(
      { op: "reduce", kind: op, dim: d, keepdim, input: a },
      keepdim ? keepShape : outShape,
      a.dtype
    )
  const n = prod(outShape)
  const strides = contiguousStrides(a.shape)
  const outer = prod(a.shape.slice(0, d))
  const dimSize = a.shape[d]!
  const inner = strides[d]!
  const init = op === "sum" ? 0 : -Infinity
  const out = new (arrayCtor(a.dtype))(n).fill(init)
  const ad = a.data
  let o = 0
  for (let i = 0; i < outer; i++) {
    for (let k = 0; k < inner; k++) {
      let acc = init
      const base = i * dimSize * inner + k
      let bestIdx = 0
      for (let j = 0; j < dimSize; j++) {
        const value = ad[base + j * inner]!
        if (op === "sum") acc += value
        else if (value > acc) {
          acc = value
          bestIdx = j
        }
      }
      out[o++] = op === "argmax" ? bestIdx : acc
    }
  }
  return makeRaw(
    out,
    keepdim ? keepShape : outShape,
    a.dtype
  )
}

function rawReduceAll(
  a: AnyTensor,
  op: "sum" | "max"
): AnyTensor {
  if (lazyMode)
    return makeLazy(
      { op: "reduceAll", kind: op, input: a },
      [],
      a.dtype
    )
  const init = op === "sum" ? 0 : -Infinity
  let acc = init
  for (let i = 0; i < a.data.length; i++)
    if (op === "sum") acc += a.data[i]!
    else if (a.data[i]! > acc) acc = a.data[i]!
  return makeRaw(arrayCtor(a.dtype).of(acc), [], a.dtype)
}

function rawBroadcastTo(
  a: AnyTensor,
  shape: number[]
): AnyTensor {
  if (shapesEqual(a.shape, shape)) return a
  if (lazyMode)
    return makeLazy(
      { op: "broadcastTo", input: a },
      shape,
      a.dtype
    )
  const n = prod(shape)
  const rank = shape.length
  const sa = broadcastStrides(a.shape, shape)
  const out = new (arrayCtor(a.dtype))(n)
  const idx = new Array(rank).fill(0)
  let off = 0
  const ad = a.data
  for (let i = 0; i < n; i++) {
    out[i] = ad[off]!
    for (let d = rank - 1; d >= 0; d--) {
      idx[d]++
      off += sa[d]!
      if (idx[d] < shape[d]!) break
      idx[d] = 0
      off -= sa[d]! * shape[d]!
    }
  }
  return makeRaw(out, shape, a.dtype)
}

function rawPermute(
  a: AnyTensor,
  order: number[]
): AnyTensor {
  const rank = a.shape.length
  const outShape = order.map(i => a.shape[i]!)
  if (lazyMode)
    return makeLazy(
      { op: "permute", order, input: a },
      outShape,
      a.dtype
    )
  const inStrides = contiguousStrides(a.shape)
  const readStrides = order.map(i => inStrides[i]!)
  const n = a.numel
  const out = new (arrayCtor(a.dtype))(n)
  const idx = new Array(rank).fill(0)
  let off = 0
  const ad = a.data
  for (let i = 0; i < n; i++) {
    out[i] = ad[off]!
    for (let d = rank - 1; d >= 0; d--) {
      idx[d]++
      off += readStrides[d]!
      if (idx[d] < outShape[d]!) break
      idx[d] = 0
      off -= readStrides[d]! * outShape[d]!
    }
  }
  return makeRaw(out, outShape, a.dtype)
}

function rawMatmul(a: AnyTensor, b: AnyTensor): AnyTensor {
  const ar = a.shape.length
  const br = b.shape.length
  const m = a.shape[ar - 2]!
  const k = a.shape[ar - 1]!
  const k2 = b.shape[br - 2]!
  const n = b.shape[br - 1]!
  if (k !== k2)
    throw new Error(
      `matmul: inner dimensions do not match (${showShape(a.shape)} @ ${showShape(b.shape)})`
    )
  const batchA = a.shape.slice(0, -2)
  const batchB = b.shape.slice(0, -2)
  const batch = broadcastShapes(batchA, batchB)
  const outShape = [...batch, m, n]
  const dtype: DType =
    a.dtype === "float64" || b.dtype === "float64" ?
      "float64"
    : "float32"
  const batchCount = prod(batch)
  const saBatch = broadcastStrides(batchA, batch)
  const sbBatch = broadcastStrides(batchB, batch)
  if (lazyMode)
    return makeLazy({ op: "matmul", a, b }, outShape, dtype)
  const out = new (arrayCtor(dtype))(batchCount * m * n)

  const aMat = m * k
  const bMat = k * n
  const rank = batch.length
  const idx = new Array(rank).fill(0)
  const ad = a.data
  const bd = b.data
  for (let bi = 0; bi < batchCount; bi++) {
    let cellA = 0
    let cellB = 0
    for (let d = 0; d < rank; d++) {
      cellA += idx[d]! * saBatch[d]!
      cellB += idx[d]! * sbBatch[d]!
    }
    const baseA = cellA * aMat
    const baseB = cellB * bMat
    const baseO = bi * m * n
    for (let i = 0; i < m; i++) {
      for (let j = 0; j < n; j++) {
        let acc = 0
        for (let p = 0; p < k; p++)
          acc +=
            ad[baseA + i * k + p]! * bd[baseB + p * n + j]!
        out[baseO + i * n + j] = acc
      }
    }
    for (let d = rank - 1; d >= 0; d--) {
      idx[d]++
      if (idx[d] < batch[d]!) break
      idx[d] = 0
    }
  }
  return makeRaw(out, outShape, dtype)
}

function rawNarrow(
  a: AnyTensor,
  dim: number,
  start: number,
  length: number
): AnyTensor {
  const d = normalizeDim(dim, a.shape.length)
  const outShape = a.shape.map((s: number, i: number) =>
    i === d ? length : s
  )
  if (lazyMode)
    return makeLazy(
      { op: "narrow", dim: d, start, length, input: a },
      outShape,
      a.dtype
    )
  const strides = contiguousStrides(a.shape)
  const outer = prod(a.shape.slice(0, d))
  const inner = strides[d]!
  const dimSize = a.shape[d]!
  const out = new (arrayCtor(a.dtype))(prod(outShape))
  const ad = a.data
  let o = 0
  for (let i = 0; i < outer; i++) {
    const base = i * dimSize * inner + start * inner
    for (let j = 0; j < length; j++) {
      for (let kk = 0; kk < inner; kk++) {
        out[o++] = ad[base + j * inner + kk]!
      }
    }
  }
  return makeRaw(out, outShape, a.dtype)
}

function rawOneHot(
  a: AnyTensor,
  classes: number
): AnyTensor {
  if (lazyMode)
    return makeLazy(
      { op: "oneHot", classes, input: a },
      [a.numel, classes],
      a.dtype
    )
  const out = new (arrayCtor(a.dtype))(a.numel * classes)
  for (let i = 0; i < a.numel; i++) {
    const target = a.data[i]!
    if (
      !Number.isInteger(target) ||
      target < 0 ||
      target >= classes
    )
      throw new Error(
        `oneHot: target ${target} out of range for ${classes} classes`
      )
    out[i * classes + target] = 1
  }
  return makeRaw(out, [a.numel, classes], a.dtype)
}

function rawCat(
  a: AnyTensor,
  b: AnyTensor,
  dim: number
): AnyTensor {
  const outShape = a.shape.map((s: number, i: number) =>
    i === dim ? s + b.shape[dim]! : s
  )
  const dtype: DType =
    a.dtype === "float64" || b.dtype === "float64" ?
      "float64"
    : "float32"
  if (lazyMode)
    return makeLazy(
      { op: "cat", a, b, dim },
      outShape,
      dtype
    )
  const strides = contiguousStrides(outShape)
  const outer = prod(outShape.slice(0, dim))
  const inner = strides[dim]!
  const lenA = a.shape[dim]!
  const lenB = b.shape[dim]!
  const out = new (arrayCtor(dtype))(prod(outShape))
  let o = 0
  for (let i = 0; i < outer; i++) {
    for (let j = 0; j < lenA; j++)
      for (let k = 0; k < inner; k++)
        out[o++] = a.data[(i * lenA + j) * inner + k]!
    for (let j = 0; j < lenB; j++)
      for (let k = 0; k < inner; k++)
        out[o++] = b.data[(i * lenB + j) * inner + k]!
  }
  return makeRaw(out, outShape, dtype)
}

// --- random numbers inside the graph ------------------------------
// A compiled training step is traced once and replayed thousands of
// times, so anything stochastic in it — a dropout-style update mask,
// input noise — has to be drawn per replay rather than baked in at
// trace time. That rules out feeding randomness in as data, which
// would mean generating and copying megabytes per step.
//
// So random values come from a graph node, and the generator is
// counter-based: element `i` of stream `s` under seed `k` is a pure
// hash of (k, s, i). No state to carry, every element independent,
// and the identical arithmetic on the Rust side means all three
// execution paths agree element for element.

/** murmur3's 32-bit finalizer, in its stronger (Stafford 13) variant. */
function hash32(x: number): number {
  x = (x ^ (x >>> 16)) >>> 0
  x = Math.imul(x, 0x7feb352d) >>> 0
  x = (x ^ (x >>> 15)) >>> 0
  x = Math.imul(x, 0x846ca68b) >>> 0
  return (x ^ (x >>> 16)) >>> 0
}

/** Uniform in [0, 1) from 24 mantissa bits of a hashed counter. */
function unitFloat(
  seed: number,
  stream: number,
  i: number
) {
  return (
    (hash32(
      (hash32(seed ^ Math.imul(stream, 0x9e3779b9)) ^ i) >>>
        0
    ) >>>
      8) *
    2 ** -24
  )
}

// The seed every evaluation draws from, advanced per evaluation so a
// replayed graph gets fresh numbers. configure({ seed }) resets it, and
// a run replays exactly given the same seed and the same sequence of
// operations.
let randomSeed = 0x2545f491
let streamCounter = 0
// Seed of the evaluation in progress, so a random node reached by the
// interpreter knows which draw it belongs to.
let activeSeed = 0

function nextSeed(): number {
  randomSeed = hash32(randomSeed + 0x9e3779b9)
  return randomSeed
}

function rawRandom(
  kind: RandomKind,
  shape: readonly number[],
  stream: number,
  dtype: DType
): AnyTensor {
  if (lazyMode)
    return makeLazy(
      { op: "random", kind, stream },
      shape,
      dtype
    )
  return makeRaw(
    randomData(
      kind,
      prod(shape),
      stream,
      activeSeed,
      dtype
    ),
    shape,
    dtype
  )
}

function randomData(
  kind: RandomKind,
  n: number,
  stream: number,
  seed: number,
  dtype: DType
): TypedArray {
  const out = new (arrayCtor(dtype))(n)
  if (kind === "uniform")
    for (let i = 0; i < n; i++)
      out[i] = unitFloat(seed, stream, i)
  // Box-Muller per element from two independent draws: stateless, so
  // element i does not depend on how many were drawn before it.
  else
    for (let i = 0; i < n; i++) {
      const u = 1 - unitFloat(seed, stream, 2 * i)
      const v = unitFloat(seed, stream, 2 * i + 1)
      out[i] =
        Math.sqrt(-2 * Math.log(u)) *
        Math.cos(2 * Math.PI * v)
    }
  return out
}

/**
 * Uniform values in [0, 1), redrawn on every evaluation.
 *
 * Unlike {@link Tensor.rand}, which draws once and hands back fixed
 * data, this is a node in the graph: a compiled function gets fresh
 * numbers on every call, which is what a stochastic update rule needs.
 * In eager mode there is no graph to defer to, so it draws immediately.
 *
 * Seeded by `configure({ seed })`, not `Math.random`.
 */
export function uniform<const Sh extends Shape>(
  shape: Sh
): Tensor<Sh, DefaultParams> {
  return rawRandom(
    "uniform",
    shape,
    streamCounter++,
    "float32"
  ) as any
}

/** Standard normal values, redrawn per evaluation. See {@link uniform}. */
export function normal<const Sh extends Shape>(
  shape: Sh
): Tensor<Sh, DefaultParams> {
  return rawRandom(
    "normal",
    shape,
    streamCounter++,
    "float32"
  ) as any
}

// --- gather / scatter ---------------------------------------------
// The two ops message passing on a graph is built from, and each
// other's gradient: indexSelect reads row index[j] of the input into
// row j of the output, scatterAdd sums row j of the input into row
// index[j] of the output. `index` holds integral values in a float
// tensor — typenet has no integer dtype, and a float32 mantissa
// addresses 16.7M rows exactly.

function checkIndex(
  value: number,
  limit: number,
  what: string
): number {
  if (
    !Number.isInteger(value) ||
    value < 0 ||
    value >= limit
  )
    throw new Error(
      `${what}: index ${value} out of range for ${limit} rows`
    )
  return value
}

function rawIndexSelect(
  a: AnyTensor,
  index: AnyTensor,
  dim: number
): AnyTensor {
  const length = index.numel
  const outShape = a.shape.map((s: number, i: number) =>
    i === dim ? length : s
  )
  if (lazyMode)
    return makeLazy(
      { op: "indexSelect", dim, input: a, index },
      outShape,
      a.dtype
    )
  const strides = contiguousStrides(a.shape)
  const outer = prod(a.shape.slice(0, dim))
  const inner = strides[dim]!
  const dimSize = a.shape[dim]!
  const out = new (arrayCtor(a.dtype))(prod(outShape))
  const ad = a.data
  const id = index.data
  let o = 0
  for (let i = 0; i < outer; i++)
    for (let j = 0; j < length; j++) {
      const base =
        (i * dimSize +
          checkIndex(id[j]!, dimSize, "indexSelect")) *
        inner
      for (let k = 0; k < inner; k++)
        out[o++] = ad[base + k]!
    }
  return makeRaw(out, outShape, a.dtype)
}

function rawScatterAdd(
  a: AnyTensor,
  index: AnyTensor,
  dim: number,
  length: number
): AnyTensor {
  const outShape = a.shape.map((s: number, i: number) =>
    i === dim ? length : s
  )
  if (lazyMode)
    return makeLazy(
      { op: "scatterAdd", dim, length, input: a, index },
      outShape,
      a.dtype
    )
  const strides = contiguousStrides(a.shape)
  const outer = prod(a.shape.slice(0, dim))
  const inner = strides[dim]!
  const srcLength = a.shape[dim]!
  const out = new (arrayCtor(a.dtype))(prod(outShape))
  const ad = a.data
  const id = index.data
  for (let i = 0; i < outer; i++)
    for (let j = 0; j < srcLength; j++) {
      const to =
        (i * length +
          checkIndex(id[j]!, length, "scatterAdd")) *
        inner
      const from = (i * srcLength + j) * inner
      for (let k = 0; k < inner; k++)
        out[to + k]! += ad[from + k]!
    }
  return makeRaw(out, outShape, a.dtype)
}

function makeLazy(
  body: LazyNodeBody,
  shape: readonly number[],
  dtype: DType
): AnyTensor {
  const node = {
    ...body,
    shape: [...shape],
    dtype
  } as LazyNode
  return makeStorage(
    { kind: "lazy", node, cache: null },
    shape,
    dtype
  )
}

function evalNode(node: LazyNode): AnyTensor {
  switch (node.op) {
    case "binary":
      return rawBinary(
        force(node.a),
        force(node.b),
        node.kind,
        node.parameter
      )
    case "unary":
      return rawUnary(
        force(node.input),
        node.kind,
        node.parameter
      )
    case "matmul":
      return rawMatmul(force(node.a), force(node.b))
    case "reduce":
      return rawReduce(
        force(node.input),
        node.dim,
        node.keepdim,
        node.kind
      )
    case "reduceAll":
      return rawReduceAll(force(node.input), node.kind)
    case "broadcastTo":
      return rawBroadcastTo(force(node.input), node.shape)
    case "permute":
      return rawPermute(force(node.input), node.order)
    case "view":
      return reshapeRaw(force(node.input), node.shape)
    case "narrow":
      return rawNarrow(
        force(node.input),
        node.dim,
        node.start,
        node.length
      )
    case "cat":
      return rawCat(force(node.a), force(node.b), node.dim)
    case "oneHot":
      return rawOneHot(force(node.input), node.classes)
    case "indexSelect":
      return rawIndexSelect(
        force(node.input),
        force(node.index),
        node.dim
      )
    case "scatterAdd":
      return rawScatterAdd(
        force(node.input),
        force(node.index),
        node.dim,
        node.length
      )
    case "random":
      return makeRaw(
        randomData(
          node.kind,
          prod(node.shape),
          node.stream,
          activeSeed,
          node.dtype
        ),
        node.shape,
        node.dtype
      )
  }
}

/**
 * Post-order traversal of the lazy graph reachable from `roots`: every
 * tensor appears after all of its inputs, each exactly once. Leaves
 * (non-lazy tensors) are included, with no inputs of their own.
 *
 * Iterative on purpose. A compiled training step for a cellular
 * automaton rolls the update rule out over dozens of time steps and
 * then differentiates it, which is a graph thousands of nodes deep —
 * far past what recursion survives.
 */
function topoOrder(
  roots: readonly AnyTensor[]
): AnyTensor[] {
  const order: AnyTensor[] = []
  const seen = new Set<AnyTensor>()
  const stack: {
    t: AnyTensor
    inputs: AnyTensor[]
    i: number
  }[] = []
  const push = (t: AnyTensor): void => {
    if (seen.has(t)) return
    seen.add(t)
    stack.push({
      t,
      inputs:
        t._storage.kind === "lazy" ?
          nodeInputs(t._storage.node)
        : [],
      i: 0
    })
  }
  for (const root of roots) {
    push(root)
    while (stack.length > 0) {
      const frame = stack[stack.length - 1]!
      if (frame.i < frame.inputs.length) {
        push(frame.inputs[frame.i++]!)
        continue
      }
      stack.pop()
      order.push(frame.t)
    }
  }
  return order
}

/**
 * Evaluate `roots` and everything they depend on with the TS
 * interpreter, deepest node first. Walking in topological order means
 * each `evalNode` call finds its inputs already materialized, so the
 * `force()` calls inside it never recurse.
 */
function evalInterpreted(roots: AnyTensor[]): void {
  // One seed per evaluation: every random node in this pass draws from
  // it, and the next pass over the same graph draws different numbers.
  activeSeed = nextSeed()
  for (const t of topoOrder(roots)) {
    const storage = t._storage
    if (storage.kind !== "lazy") continue
    if (!storage.cache)
      storage.cache = eagerly(() => evalNode(storage.node))
    ;(t as { _storage: TensorStorage })._storage =
      storage.cache._storage
  }
}

function force(t: AnyTensor): AnyTensor {
  const storage = t._storage
  if (storage.kind !== "lazy") return t
  if (!storage.cache && !evalNativeMany([t]))
    evalInterpreted([t])
  ;(t as { _storage: TensorStorage })._storage =
    storage.cache!._storage
  return t
}

// --- native (candle) backend -------------------------------------
// Serialize the whole lazy graph to JSON with leaves as indexed
// placeholders, evaluate it natively in one FFI hop, and wrap the
// result as a CPU tensor. Returns null when the graph is not
// supported natively (float64) so force() falls back to the
// interpreter.

type SerializedNode = Record<string, unknown> & {
  op: string
}

// Graphs touching at most this many elements (leaves + intermediate node
// outputs) go to the native fused loop evaluator, which pays no dispatch
// or BLAS setup cost. 65536 = one 256×256 matrix.
const LOOP_EVALUATOR_MAX_WORK = 65536

/**
 * Which native evaluator a graph of `work` elements should run on.
 *
 * Tiny graphs go to the loop evaluator. Everything else goes to candle
 * on the CPU device, which on macOS means Accelerate for matmul. That is
 * not the obvious default, so the numbers behind it (Apple M5, see
 * LAZY.md): CPU matches Metal on chained large matmuls, loses to it by
 * ~1.5x on purely elementwise graphs, and beats it by ~7x on the
 * gather/scatter graphs message passing produces — candle's Metal
 * index_select/index_add are slow and the graphs are made of many small
 * kernels. `useNative({ device: "gpu" })` opts back in.
 */
function pickTarget(work: number): "loops" | "cpu" | "gpu" {
  if (work <= LOOP_EVALUATOR_MAX_WORK) return "loops"
  return nativeBackend.nativeDeviceMode()
}

function serializeLazyGraph(roots: AnyTensor[]): {
  json: string
  leaves: Float32Array
  rootShapes: number[][]
  leafTensors: AnyTensor[]
  leafOffsets: number[]
  leafBytes: number
} | null {
  const order = topoOrder(roots)
  const nodes: SerializedNode[] = []
  const index = new Map<AnyTensor, number>()
  const leafData: Float32Array[] = []
  const leafTensors: AnyTensor[] = []
  const leafOffsets: number[] = []
  let leafBytes = 0
  // Rough work estimate (elements touched) — used to pin tiny graphs
  // to the candle CPU device, where per-kernel Metal dispatch and
  // readback overhead would dwarf the compute.
  let work = 0

  for (const t of order) {
    index.set(t, nodes.length)
    if (t._storage.kind !== "lazy") {
      // A leaf: its data becomes one slice of the concatenated leaf
      // buffer. Anything the native bridge cannot take (float64,
      // non-CPU storage) aborts serialization so the caller falls
      // back to the interpreter.
      if (
        t._storage.kind !== "cpu" ||
        t.dtype !== "float32"
      )
        return null
      const data = t._storage.data as Float32Array
      nodes.push({
        op: "leaf",
        leaf: leafTensors.length,
        offset: leafBytes,
        shape: [...t.shape]
      })
      leafData.push(data)
      leafTensors.push(t)
      leafOffsets.push(leafBytes)
      leafBytes += data.length
      work += data.length
      continue
    }
    const node = t._storage.node
    work += prod(node.shape)
    // Every input precedes `t` in topological order, so its index is
    // already assigned.
    const ref = (u: AnyTensor): number => index.get(u)!
    switch (node.op) {
      case "binary":
        nodes.push({
          op: "binary",
          kind: node.kind,
          parameter: node.parameter,
          a: ref(node.a),
          b: ref(node.b),
          shape: node.shape
        })
        break
      case "unary":
        nodes.push({
          op: "unary",
          kind: node.kind,
          parameter: node.parameter,
          input: ref(node.input)
        })
        break
      case "matmul":
        nodes.push({
          op: "matmul",
          a: ref(node.a),
          b: ref(node.b)
        })
        break
      case "reduce":
        nodes.push({
          op: "reduce",
          kind: node.kind,
          dim: node.dim,
          keepdim: node.keepdim,
          input: ref(node.input)
        })
        break
      case "reduceAll":
        nodes.push({
          op: "reduceAll",
          kind: node.kind,
          input: ref(node.input)
        })
        break
      case "broadcastTo":
        nodes.push({
          op: "broadcastTo",
          input: ref(node.input),
          shape: node.shape
        })
        break
      case "permute":
        nodes.push({
          op: "permute",
          order: node.order,
          input: ref(node.input)
        })
        break
      case "view":
        nodes.push({
          op: "view",
          input: ref(node.input),
          shape: node.shape
        })
        break
      case "narrow":
        nodes.push({
          op: "narrow",
          dim: node.dim,
          start: node.start,
          length: node.length,
          input: ref(node.input)
        })
        break
      case "cat":
        nodes.push({
          op: "cat",
          a: ref(node.a),
          b: ref(node.b),
          dim: node.dim
        })
        break
      case "oneHot":
        nodes.push({
          op: "oneHot",
          classes: node.classes,
          input: ref(node.input)
        })
        break
      case "indexSelect":
        nodes.push({
          op: "indexSelect",
          dim: node.dim,
          input: ref(node.input),
          index: ref(node.index)
        })
        break
      case "scatterAdd":
        nodes.push({
          op: "scatterAdd",
          dim: node.dim,
          length: node.length,
          input: ref(node.input),
          index: ref(node.index)
        })
        break
      case "random":
        // The stream id is part of the graph's structure; the seed is
        // not, so it travels as an argument of the eval call and a
        // replayed graph keeps hitting the same prepared plan.
        nodes.push({
          op: "random",
          kind: node.kind,
          stream: node.stream,
          shape: node.shape
        })
        break
    }
  }

  const rootIndices = roots.map(root => index.get(root)!)
  const rootShapes = roots.map(root => [...root.shape])
  const total = leafData.reduce((n, d) => n + d.length, 0)
  const leaves = new Float32Array(total)
  let offset = 0
  for (const d of leafData) {
    leaves.set(d, offset)
    offset += d.length
  }
  return {
    json: JSON.stringify({
      nodes,
      roots: rootIndices,
      device: pickTarget(work)
    }),
    leaves,
    rootShapes,
    leafTensors,
    leafOffsets,
    leafBytes
  }
}

// Evaluate every root in one FFI hop. The roots share a single
// serialized graph (memoized on tensor identity), so subexpressions
// shared across roots are evaluated once natively. On success each
// root's LazyStorage cache is filled with a CPU tensor viewing its
// slice of the returned buffer and true is returned; force()/forceMany()
// then do the usual in-place storage swap.
function evalNativeMany(roots: AnyTensor[]): boolean {
  if (!nativeBackend.isNativeEnabled()) return false
  for (const t of roots)
    if (t._storage.kind !== "lazy") return false
  const serialized = serializeLazyGraph(roots)
  if (!serialized) return false
  const data = nativeBackend.evalGraphNative(
    serialized.json,
    serialized.leaves,
    nextSeed()
  )
  let offset = 0
  serialized.rootShapes.forEach((shape, i) => {
    const n = prod(shape)
    const storage = roots[i]!._storage
    if (storage.kind === "lazy")
      storage.cache = makeRaw(
        data.subarray(offset, offset + n),
        shape,
        "float32"
      )
    offset += n
  })
  if (offset !== data.length)
    throw new Error(
      `native backend returned ${data.length} values, expected ${offset} for roots [${serialized.rootShapes.map(showShape).join(", ")}]`
    )
  return true
}

// Force several lazy tensors, preferring a single multi-root native
// eval. Falls back to forcing each tensor with the interpreter, whose
// per-LazyStorage memoization still evaluates shared subgraphs once.
// Exported for src/optim.ts (optimizer-in-graph forcing point).
export function forceMany(ts: AnyTensor[]): void {
  const pending = ts.filter(t => {
    const storage = t._storage
    return storage.kind === "lazy" && !storage.cache
  })
  if (pending.length > 0 && !evalNativeMany(pending))
    evalInterpreted(pending)
  // Everything is materialized by now; force() only swaps storages in.
  for (const t of ts) force(t)
}

// --- optimizer in the graph (phase B task 4) ----------------------
// In lazy mode an optimizer step builds its parameter/state updates
// as lazy expressions instead of looping over `.data`, forces them in
// one multi-root hop, and writes the results back into the leaf
// buffers. During a compile() trace the updates are collected here
// instead of forced, becoming extra graph roots that replay writes
// back into the same leaves — the whole training step (forward +
// backward + optimizer update) is then one graph. Optimizer state
// (momentum velocities) lives in ordinary CPU leaf tensors carried
// between steps (the typonet model).

type GraphUpdate = {
  target: AnyTensor
  expr: AnyTensor
}

type UpdateTrace = {
  updates: GraphUpdate[]
  // Grad tensors the step consumed — replay materializes them so
  // `.grad` reads after a compiled step see this step's values.
  materialize: AnyTensor[]
}

let updateTrace: UpdateTrace | null = null

// Internal hook for src/optim.ts — not part of the public API.
export function _activeUpdateTrace(): UpdateTrace | null {
  return updateTrace
}

// --- compile(): build once, replay many ---------------------------
// compile(fn) traces fn once in lazy mode with placeholder leaves,
// serializes the graph once, and replays it on each call: the caller's
// input data is copied into the placeholder leaf buffers and the whole
// graph is evaluated in one multi-root native hop (interpreter
// fallback when native is unavailable). Closure-captured tensors
// (e.g. module parameters) are graph leaves too, read live on every
// call, so in-place optimizer steps are visible to the compiled fn.

function lazily<T>(fn: () => T): T {
  const prev = lazyMode
  lazyMode = true
  try {
    return fn()
  } finally {
    lazyMode = prev
  }
}

function nodeInputs(node: LazyNode): AnyTensor[] {
  switch (node.op) {
    case "binary":
    case "matmul":
    case "cat":
      return [node.a, node.b]
    case "unary":
    case "reduce":
    case "reduceAll":
    case "broadcastTo":
    case "permute":
    case "view":
    case "narrow":
    case "oneHot":
      return [node.input]
    case "indexSelect":
    case "scatterAdd":
      return [node.input, node.index]
    case "random":
      return []
  }
}

type CompiledInput<T extends AnyTensor> =
  T extends Tensor<infer S, any> ?
    Tensor<S, any> | ArrayLike<number>
  : never

// --- debug names + graph printing (phase C task 5) ----------------
// Names are debug metadata only: a WeakMap from tensor object to
// label, never consulted by compute, autograd, or serialization.
// Forcing keeps a tensor's identity (its storage is swapped in
// place), so a name survives materialization; names do NOT cross
// detach()/clone()/compile() placeholders, which create fresh tensor
// objects.

const tensorNames = new WeakMap<AnyTensor, string>()

/**
 * Dump the lazy graph behind `root` (or several roots) as a readable
 * SSA-ish listing: one line per node, in topological order, with
 * inputs, output shape, and dtype. Named tensors (see `.named()`)
 * print under their name; unnamed nodes get `%0`, `%1`, ... in
 * traversal order. Shared subgraphs print once. An eager tensor has
 * no graph and prints as a single `leaf` line.
 */
export function printGraph(
  roots: AnyTensor | AnyTensor[]
): string {
  const rootList = Array.isArray(roots) ? roots : [roots]
  const ids = new Map<AnyTensor, number>()
  const entries = topoOrder(rootList).map(t => ({
    t,
    node:
      t._storage.kind === "lazy" ? t._storage.node : null
  }))
  entries.forEach(({ t }, i) => ids.set(t, i))
  const label = (t: AnyTensor): string =>
    tensorNames.get(t) ?? `%${ids.get(t)!}`
  const width = Math.max(
    1,
    ...entries.map(({ t }) => label(t).length)
  )
  const rootSet = new Set(rootList)
  return entries
    .map(({ t, node }) => {
      const lhs = label(t).padEnd(width)
      const shape = showShape(t.shape)
      const tail = `${shape} ${t.dtype}${
        rootSet.has(t) ? " ; root" : ""
      }`
      if (!node) return `${lhs} = leaf ${tail}`
      const arg = (u: AnyTensor) => label(u)
      const attrs = (pairs: [string, unknown][]): string =>
        pairs.length === 0 ?
          ""
        : ` {${pairs.map(([k, v]) => `${k}=${v}`).join(", ")}}`
      const param = (p: number): [string, unknown][] =>
        p === 0 ? [] : [["parameter", p]]
      switch (node.op) {
        case "binary":
          return `${lhs} = ${node.kind}(${arg(node.a)}, ${arg(node.b)})${attrs(param(node.parameter))} ${tail}`
        case "unary":
          return `${lhs} = ${node.kind}(${arg(node.input)})${attrs(param(node.parameter))} ${tail}`
        case "matmul":
          return `${lhs} = matmul(${arg(node.a)}, ${arg(node.b)}) ${tail}`
        case "reduce":
          return `${lhs} = reduce.${node.kind}(${arg(node.input)})${attrs(
            [
              ["dim", node.dim],
              ...(node.keepdim ?
                ([["keepdim", node.keepdim]] as [
                  string,
                  unknown
                ][])
              : [])
            ]
          )} ${tail}`
        case "reduceAll":
          return `${lhs} = reduceAll.${node.kind}(${arg(node.input)}) ${tail}`
        case "broadcastTo":
          return `${lhs} = broadcastTo(${arg(node.input)}) ${tail}`
        case "permute":
          return `${lhs} = permute(${arg(node.input)})${attrs(
            [["order", `[${node.order.join(", ")}]`]]
          )} ${tail}`
        case "view":
          return `${lhs} = view(${arg(node.input)}) ${tail}`
        case "narrow":
          return `${lhs} = narrow(${arg(node.input)})${attrs(
            [
              ["dim", node.dim],
              ["start", node.start],
              ["length", node.length]
            ]
          )} ${tail}`
        case "cat":
          return `${lhs} = cat(${arg(node.a)}, ${arg(node.b)})${attrs(
            [["dim", node.dim]]
          )} ${tail}`
        case "oneHot":
          return `${lhs} = oneHot(${arg(node.input)})${attrs(
            [["classes", node.classes]]
          )} ${tail}`
        case "indexSelect":
          return `${lhs} = indexSelect(${arg(node.input)}, ${arg(node.index)})${attrs(
            [["dim", node.dim]]
          )} ${tail}`
        case "scatterAdd":
          return `${lhs} = scatterAdd(${arg(node.input)}, ${arg(node.index)})${attrs(
            [
              ["dim", node.dim],
              ["length", node.length]
            ]
          )} ${tail}`
        case "random":
          return `${lhs} = random.${node.kind}()${attrs([
            ["stream", node.stream]
          ])} ${tail}`
      }
    })
    .join("\n")
}

/**
 * Trace `fn` once and replay the resulting graph on every call.
 *
 * The first call must pass CPU float32 tensors (their shapes/dtype
 * pin the traced graph); later calls may pass either tensors of the
 * same shape or flat `ArrayLike<number>` buffers of matching length.
 * Calling with a different argument count, shape, or dtype throws —
 * compiled graphs are shape-stable, recompile for a new shape.
 *
 * Tracing always happens under lazy semantics regardless of the
 * global `configure({ lazy })` flag, and the flag is restored
 * afterwards. `fn` must be a pure dataflow function of its inputs
 * with one exception: a full training step — `loss.backward()` plus
 * an optimizer `step()` — is traced into the graph, so a compiled
 * step evaluates forward, backward, and the parameter update in one
 * pass and writes the updated values back into the parameter (and
 * optimizer state) buffers on every call. Other forcing points
 * (`.data`, `.item()`, ...) inside `fn` remain unsupported.
 */
export function compile<
  Args extends AnyTensor[],
  R extends AnyTensor | AnyTensor[]
>(
  fn: (...args: Args) => R
): (
  ...inputs: { [K in keyof Args]: CompiledInput<Args[K]> }
) => R {
  type State = {
    placeholders: AnyTensor[]
    outputs: AnyTensor[]
    tuple: boolean
    shapes: number[][]
    // Optimizer updates traced inside fn: on every replay the update
    // roots are evaluated together with the outputs and written back
    // into the target leaf buffers (parameters, optimizer state).
    updates: GraphUpdate[]
    // Grad tensors to materialize per replay (lazy tensor + its
    // original storage, so the result can be swapped in like force()).
    materialize: { t: AnyTensor; storage: LazyStorage }[]
    // Native replay: serialized once at trace time and handed to the
    // backend once as a prepared plan, so a call ships only the leaf
    // buffer, rebuilt from the live leaf tensors.
    native: {
      json: string
      handle: number | null
      leafTensors: AnyTensor[]
      leafOffsets: number[]
      leafBytes: number
      rootShapes: number[][]
    } | null
    // Interpreter replay: every lazy tensor in the traced graph with
    // its original storage, so caches can be reset between calls.
    lazy: { t: AnyTensor; storage: LazyStorage }[]
  }
  let state: State | null = null

  const trace = (inputs: readonly unknown[]): State => {
    const placeholders = inputs.map((input, i) => {
      if (!(input instanceof Tensor))
        throw new Error(
          `compile() traces on the first call, so argument ${i} must be a Tensor (later calls may pass flat buffers)`
        )
      const t = force(input as AnyTensor)
      if (
        t._storage.kind !== "cpu" ||
        t.dtype !== "float32"
      )
        throw new Error(
          `compile() only supports CPU float32 inputs, argument ${i} is ${t.dtype}`
        )
      // Own copy of the data: replay mutates this buffer per call.
      return makeRaw(
        (t._storage.data as Float32Array).slice(),
        t.shape,
        "float32"
      )
    })
    const prevTrace = updateTrace
    const traced: UpdateTrace = {
      updates: [],
      materialize: []
    }
    updateTrace = traced
    let result: unknown
    try {
      result = lazily(() => fn(...(placeholders as Args)))
    } finally {
      updateTrace = prevTrace
    }
    const tuple = Array.isArray(result)
    const outputs = (
      tuple ? result : [result]) as AnyTensor[]
    outputs.forEach((out, i) => {
      if (!(out instanceof Tensor))
        throw new Error(
          `compile() expected fn to return a Tensor or Tensor[], got ${typeof out} at output ${i}`
        )
    })
    const updates = traced.updates
    const materialize = traced.materialize.map(t => {
      if (t._storage.kind !== "lazy")
        throw new Error(
          "compile(): an optimizer step produced a non-lazy gradient — compiled training steps need lazy gradients"
        )
      return { t, storage: t._storage }
    })
    // Root order: outputs, then update expressions, then grads.
    const roots = [
      ...outputs,
      ...updates.map(u => u.expr),
      ...materialize.map(m => m.t)
    ]
    const lazy: State["lazy"] = topoOrder(roots)
      .filter(t => t._storage.kind === "lazy")
      .map(t => ({ t, storage: t._storage as LazyStorage }))
    const serialized = serializeLazyGraph(roots)
    return {
      placeholders,
      outputs,
      tuple,
      shapes: placeholders.map(p => [...p.shape]),
      updates,
      materialize,
      native:
        serialized ?
          {
            json: serialized.json,
            handle: null,
            leafTensors: serialized.leafTensors,
            leafOffsets: serialized.leafOffsets,
            leafBytes: serialized.leafBytes,
            rootShapes: serialized.rootShapes
          }
        : null,
      lazy
    }
  }

  const swapInputs = (
    state: State,
    inputs: readonly unknown[]
  ): void => {
    if (inputs.length !== state.placeholders.length)
      throw new Error(
        `compiled function expected ${state.placeholders.length} arguments, got ${inputs.length}`
      )
    inputs.forEach((input, i) => {
      const storage = state.placeholders[i]!._storage
      const buffer = (storage as CpuStorage)
        .data as Float32Array
      if (input instanceof Tensor) {
        const t = force(input as AnyTensor)
        if (t._storage.kind !== "cpu")
          throw new Error(
            `compiled function argument ${i}: expected a CPU tensor`
          )
        if (t.dtype !== "float32")
          throw new Error(
            `compiled function argument ${i}: expected float32, got ${t.dtype}`
          )
        if (!shapesEqual(t.shape, state.shapes[i]!))
          throw new Error(
            `compiled function argument ${i}: expected shape ${showShape(state.shapes[i]!)}, got ${showShape(t.shape)} — compiled graphs are shape-stable, recompile for a new shape`
          )
        buffer.set(t._storage.data as Float32Array)
      } else if (
        input != null &&
        typeof (input as ArrayLike<number>).length ===
          "number"
      ) {
        if (
          (input as ArrayLike<number>).length !==
          buffer.length
        )
          throw new Error(
            `compiled function argument ${i}: expected ${buffer.length} values for shape ${showShape(state.shapes[i]!)}, got ${(input as ArrayLike<number>).length}`
          )
        buffer.set(
          Array.from(input as ArrayLike<number>, Number)
        )
      } else {
        throw new Error(
          `compiled function argument ${i}: expected a Tensor or flat ArrayLike<number>`
        )
      }
    })
  }

  // Write one evaluated update root back into its target leaf buffer.
  const applyUpdate = (
    u: GraphUpdate,
    values: Float32Array
  ): void => {
    const storage = u.target._storage
    if (storage.kind !== "cpu")
      throw new Error(
        "compiled function: an optimizer update target is not CPU storage — compiled graphs require parameters and optimizer state to stay put"
      )
    if (storage.data.length !== values.length)
      throw new Error(
        "compiled function: an optimizer update target changed size"
      )
    ;(storage.data as Float32Array).set(values)
  }

  const runNative = (state: State): AnyTensor[] => {
    const native = state.native!
    const leaves = new Float32Array(native.leafBytes)
    native.leafTensors.forEach((leaf, i) => {
      const storage = leaf._storage
      if (storage.kind !== "cpu")
        throw new Error(
          "compiled function: a captured tensor is not CPU storage — compiled graphs require captured leaves (e.g. parameters) to stay put"
        )
      leaves.set(
        storage.data as Float32Array,
        native.leafOffsets[i]!
      )
    })
    // Prepared once, then replayed by handle: the graph JSON never
    // crosses the boundary again.
    if (native.handle === null)
      native.handle = nativeBackend.prepareGraphNative(
        native.json
      )
    const data = nativeBackend.evalPreparedNative(
      native.handle,
      leaves,
      nextSeed()
    )
    let offset = 0
    const take = (shape: number[]): Float32Array => {
      const n = prod(shape)
      const view = data.subarray(offset, offset + n)
      offset += n
      return view
    }
    const outputs = native.rootShapes
      .slice(0, state.outputs.length)
      .map(shape => makeRaw(take(shape), shape, "float32"))
    // Root order: outputs, update expressions, grads (see trace()).
    for (const u of state.updates)
      applyUpdate(u, take([...u.expr.shape]))
    for (const m of state.materialize) {
      const values = take([...m.t.shape])
      m.storage.cache = makeRaw(
        values,
        [...m.t.shape],
        "float32"
      )
      ;(m.t as { _storage: TensorStorage })._storage =
        m.storage.cache._storage
    }
    return outputs
  }

  const runInterpreter = (state: State): AnyTensor[] => {
    // Reset every traced lazy tensor so the graph re-evaluates with
    // the swapped leaf data instead of serving stale caches.
    for (const { t, storage } of state.lazy) {
      storage.cache = null
      ;(t as { _storage: TensorStorage })._storage = storage
    }
    forceMany([
      ...state.outputs,
      ...state.updates.map(u => u.expr),
      ...state.materialize.map(m => m.t)
    ])
    for (const u of state.updates)
      applyUpdate(u, u.expr.data as Float32Array)
    // Fresh tensors sharing this call's result buffers — the next
    // call allocates new buffers, so callers can hold onto these.
    return state.outputs.map(out =>
      makeRaw(out.data, out.shape, out.dtype)
    )
  }

  return ((...inputs: readonly unknown[]) => {
    if (!state) state = trace(inputs)
    swapInputs(state, inputs)
    const outputs =
      state.native && nativeBackend.isNativeEnabled() ?
        runNative(state)
      : runInterpreter(state)
    return (state.tuple ? outputs : outputs[0]) as R
  }) as (
    ...inputs: { [K in keyof Args]: CompiledInput<Args[K]> }
  ) => R
}

/**
 * Post-order traversal of the autograd tape behind `root`: every
 * tensor appears after the inputs it was computed from, so walking the
 * result in reverse visits each node only once its own gradient is
 * complete. Iterative for the same reason as `topoOrder` — the tape
 * behind a long rollout is thousands of nodes deep.
 */
function tapeOrder(root: AnyTensor): AnyTensor[] {
  const topo: AnyTensor[] = []
  const seen = new Set<AnyTensor>()
  const stack: {
    t: AnyTensor
    inputs: AnyTensor[]
    i: number
  }[] = []
  const push = (t: AnyTensor): void => {
    if (seen.has(t)) return
    seen.add(t)
    stack.push({
      t,
      inputs: t.gradNode ? t.gradNode.inputs : [],
      i: 0
    })
  }
  push(root)
  while (stack.length > 0) {
    const frame = stack[stack.length - 1]!
    if (frame.i < frame.inputs.length) {
      push(frame.inputs[frame.i++]!)
      continue
    }
    stack.pop()
    topo.push(frame.t)
  }
  return topo
}

function withGrad(
  result: AnyTensor,
  name: string,
  inputs: AnyTensor[],
  backward: (grad: AnyTensor) => (AnyTensor | null)[]
): AnyTensor {
  if (gradEnabled && inputs.some(t => t.needsGrad)) {
    result.gradNode = { name, inputs, backward }
  }
  return result
}

const INTERNAL = Symbol("tensor-internal")

function flatten(
  value: NestedNumbers,
  out: number[],
  shape: number[],
  depth: number
): void {
  if (typeof value === "number") {
    if (depth !== shape.length)
      throw new Error(
        "Ragged nested array passed to tensor()"
      )
    out.push(value)
    return
  }
  if (depth === shape.length) shape.push(value.length)
  else if (shape[depth] !== value.length)
    throw new Error(
      "Ragged nested array passed to tensor()"
    )
  for (const v of value) flatten(v, out, shape, depth + 1)
}

export class Tensor<
  S extends Shape = number[],
  P extends TensorParams = DefaultParams
> {
  readonly _storage: TensorStorage
  readonly shape: S
  readonly dtype: DType

  grad: Tensor<S, P> | null = null

  requiresGrad = false

  gradNode: GradNode | null = null

  constructor(
    storage: TensorStorage,
    shape: number[],
    dtype: DType,
    internal: typeof INTERNAL
  ) {
    if (internal !== INTERNAL)
      throw new Error(
        "Use Tensor.of / zeros / ones / randn to create tensors"
      )
    const length =
      storage.kind === "cpu" ?
        storage.data.length
      : prod(storage.node.shape)
    if (length !== prod(shape))
      throw new Error(
        `Data length ${length} does not match shape ${showShape(shape)}`
      )
    this._storage = storage
    this.shape = shape as S
    this.dtype = dtype
  }

  get data(): TypedArray {
    if (this._storage.kind === "lazy")
      return force(this as AnyTensor).data
    return this._storage.data
  }

  get needsGrad(): boolean {
    return this.requiresGrad || this.gradNode !== null
  }

  get rank(): S["length"] {
    return this.shape.length
  }

  get numel(): number {
    return prod(this.shape)
  }

  static of<const V extends NestedNumbers>(
    value: V
  ): Tensor<InferShape<V>, DefaultParams> {
    const flat: number[] = []
    const shape: number[] = []
    flatten(value, flat, shape, 0)
    return makeRaw(
      Float32Array.from(flat),
      shape,
      "float32"
    ) as any
  }

  static full<const Sh extends Shape>(
    shape: Sh,
    value: number
  ): Tensor<Sh, DefaultParams> {
    const data = new Float32Array(prod(shape)).fill(value)
    return makeRaw(data, shape, "float32") as any
  }

  static zeros<const Sh extends Shape>(
    shape: Sh
  ): Tensor<Sh, DefaultParams> {
    return Tensor.full(shape, 0)
  }

  static ones<const Sh extends Shape>(
    shape: Sh
  ): Tensor<Sh, DefaultParams> {
    return Tensor.full(shape, 1)
  }

  static rand<const Sh extends Shape>(
    shape: Sh
  ): Tensor<Sh, DefaultParams> {
    const data = new Float32Array(prod(shape))
    for (let i = 0; i < data.length; i++)
      data[i] = Math.random()
    return makeRaw(data, shape, "float32") as any
  }

  static randn<const Sh extends Shape>(
    shape: Sh
  ): Tensor<Sh, DefaultParams> {
    const data = new Float32Array(prod(shape))
    for (let i = 0; i < data.length; i += 2) {
      const u = 1 - Math.random()
      const v = Math.random()
      const r = Math.sqrt(-2 * Math.log(u))
      data[i] = r * Math.cos(2 * Math.PI * v)
      if (i + 1 < data.length)
        data[i + 1] = r * Math.sin(2 * Math.PI * v)
    }
    return makeRaw(data, shape, "float32") as any
  }

  static eye<const N extends number>(
    n: N
  ): Tensor<[N, N], DefaultParams> {
    const data = new Float32Array(n * n)
    for (let i = 0; i < n; i++) data[i * n + i] = 1
    return makeRaw(data, [n, n], "float32") as any
  }

  static arange<const N extends number>(
    n: N
  ): Tensor<[N], DefaultParams> {
    const data = new Float32Array(n)
    for (let i = 0; i < n; i++) data[i] = i
    return makeRaw(data, [n], "float32") as any
  }

  static scalar(value: number): Tensor<[], DefaultParams> {
    return makeRaw(
      Float32Array.of(value),
      [],
      "float32"
    ) as any
  }

  requires_grad(): Tensor<
    S,
    Merge<P, { requires_grad: true }>
  > {
    this.requiresGrad = true
    return this as any
  }

  no_grad(): Tensor<S, Merge<P, { requires_grad: false }>> {
    this.requiresGrad = false
    return this as any
  }

  to<const D extends DType>(
    dtype: D
  ): Tensor<S, Merge<P, { dtype: D }>> {
    if (dtype === this.dtype) return this as any
    const data = arrayCtor(dtype).from(this.data)
    return makeMovedData(this, data, dtype) as any
  }

  item(): number {
    if (this.numel !== 1)
      throw new Error(
        `item() requires a one-element tensor, got shape ${showShape(this.shape)}`
      )
    return this.data[0]!
  }

  get(...indices: number[]): number {
    if (indices.length !== this.shape.length)
      throw new Error(
        `get() expects ${this.shape.length} indices, got ${indices.length}`
      )
    const strides = contiguousStrides(this.shape)
    let off = 0
    for (let i = 0; i < indices.length; i++) {
      const idx = normalizeDim(indices[i]!, this.shape[i]!)
      off += idx * strides[i]!
    }
    return this.data[off]!
  }

  toArray(): NestedArray<S> {
    const build = (dim: number, offset: number): any => {
      if (dim === this.shape.length)
        return this.data[offset]!
      const stride = contiguousStrides(this.shape)[dim]!
      const out = new Array(this.shape[dim]!)
      for (let i = 0; i < this.shape[dim]!; i++)
        out[i] = build(dim + 1, offset + i * stride)
      return out
    }
    return build(0, 0)
  }

  toString(): string {
    return `Tensor(shape=${showShape(this.shape)}, dtype=${this.dtype}, data=${JSON.stringify(this.toArray())})`
  }

  detach(): Tensor<S, Merge<P, { requires_grad: false }>> {
    return makeStorage(
      this._storage,
      this.shape,
      this.dtype
    ) as any
  }

  clone(): Tensor<S, P> {
    force(this as AnyTensor)
    const t = makeRaw(
      this.data.slice(),
      this.shape,
      this.dtype
    )
    return withGrad(t, "clone", [this], g => [g]) as any
  }

  // Debug label, shown by printGraph(). Metadata only — no effect on
  // computation, autograd, or graph semantics. Returns `this` so it
  // chains inside an expression: `x.matmul(w).named("h")`.
  named(name: string): this {
    tensorNames.set(this as AnyTensor, name)
    return this
  }

  backward(gradient?: Tensor<S, any>): void {
    if (!this.needsGrad)
      throw new Error(
        "backward() on a tensor that does not require grad"
      )
    let seed: AnyTensor
    // Lazy symbolic backward: when the output is a lazy graph node,
    // the whole backward pass is built as lazy expressions and forced
    // in one multi-root hop instead of materializing the forward and
    // running the GradNode engine eagerly.
    const lazyPath =
      lazyMode && this._storage.kind === "lazy"
    if (gradient) {
      if (!shapesEqual(gradient.shape, this.shape))
        throw new Error(
          `backward() gradient shape ${showShape(gradient.shape)} does not match ${showShape(this.shape)}`
        )
      seed =
        lazyPath ?
          (gradient as AnyTensor)
        : force(gradient as AnyTensor)
    } else {
      if (this.numel !== 1)
        throw new Error(
          "backward() without a gradient requires a scalar output"
        )
      seed = makeRaw(
        new (arrayCtor(this.dtype))(this.numel).fill(1),
        this.shape,
        this.dtype
      )
    }

    const topo = tapeOrder(this as AnyTensor)

    const grads = new Map<AnyTensor, AnyTensor>()
    grads.set(this, seed)

    const walk = () =>
      noGrad(() => {
        for (let i = topo.length - 1; i >= 0; i--) {
          const t = topo[i]!
          const g = grads.get(t)
          if (!g) continue
          if (t.gradNode) {
            const inputGrads = t.gradNode.backward(g)
            t.gradNode.inputs.forEach((input, j) => {
              const ig = inputGrads[j]
              if (!ig || !input.needsGrad) return
              const existing = grads.get(input)
              grads.set(
                input,
                existing ?
                  rawBinary(existing, ig, "add")
                : ig
              )
            })
          } else if (t.requiresGrad) {
            t.grad =
              t.grad ?
                (rawBinary(t.grad, g, "add") as any)
              : (g as any)
          }
        }
      })

    if (lazyPath) {
      // Build all gradient expressions as lazy nodes — the backward
      // rules compose public ops, which emit lazy nodes while lazy
      // mode is on, and capture the lazy forward tensors, so forward
      // values (e.g. tanh output) are shared subgraphs, not recomputed.
      // Then materialize every parameter grad in a single multi-root
      // forcing point (one native FFI hop when native is enabled).
      walk()
      // Materialize the whole forward topo plus every parameter grad
      // in a single multi-root forcing point (one native FFI hop when
      // native is enabled). The forward tensors are forced too so that
      // values read after an in-place optimizer step (which mutates
      // the leaf parameter data) still see pre-step values, matching
      // eager and phase-1 lazy semantics. During a compile() trace the
      // forcing is deferred: the lazy grads stay graph expressions and
      // the traced optimizer updates (plus the grads themselves) become
      // extra roots of the compiled graph, materialized on replay.
      if (!updateTrace)
        forceMany([
          ...topo,
          ...topo
            .filter(t => t.requiresGrad && t.grad)
            .map(t => t.grad as AnyTensor)
        ])
      return
    }

    // Eager path: materialize the forward graph; the GradNode engine
    // then runs eagerly on real storages.
    for (const t of topo) force(t)
    eagerly(walk)
  }

  zeroGrad(): void {
    this.grad = null
  }

  add(other: number): Tensor<S, P>
  add(
    this: Tensor<[Dim0<S>, 1], P>,
    other: Tensor<[1, Dim0<S>], any>
  ): Tensor<[Dim0<S>, Dim0<S>], P>
  add(
    this: Tensor<[1, Dim1<S>], P>,
    other: Tensor<[Dim1<S>, 1], any>
  ): Tensor<[Dim1<S>, Dim1<S>], P>
  add<S2 extends Shape>(
    other: Tensor<S2, any> & BroadcastCheck<S, S2>
  ): Tensor<Broadcast<S, S2>, P>
  add(other: AnyTensor | number): AnyTensor {
    const b = coerce(other, this)
    const out = rawBinary(this, b, "add")
    return withGrad(out, "add", [this, b], g => [
      sumTo(g, this.shape),
      sumTo(g, b.shape)
    ])
  }

  sub(other: number): Tensor<S, P>
  sub(
    this: Tensor<[Dim0<S>, 1], P>,
    other: Tensor<[1, Dim0<S>], any>
  ): Tensor<[Dim0<S>, Dim0<S>], P>
  sub(
    this: Tensor<[1, Dim1<S>], P>,
    other: Tensor<[Dim1<S>, 1], any>
  ): Tensor<[Dim1<S>, Dim1<S>], P>
  sub<S2 extends Shape>(
    other: Tensor<S2, any> & BroadcastCheck<S, S2>
  ): Tensor<Broadcast<S, S2>, P>
  sub(other: AnyTensor | number): AnyTensor {
    const b = coerce(other, this)
    const out = rawBinary(this, b, "sub")
    return withGrad(out, "sub", [this, b], g => [
      sumTo(g, this.shape),
      sumTo(rawUnary(g, "neg"), b.shape)
    ])
  }

  mul(other: number): Tensor<S, P>
  mul(
    this: Tensor<[Dim0<S>, 1], P>,
    other: Tensor<[1, Dim0<S>], any>
  ): Tensor<[Dim0<S>, Dim0<S>], P>
  mul(
    this: Tensor<[1, Dim1<S>], P>,
    other: Tensor<[Dim1<S>, 1], any>
  ): Tensor<[Dim1<S>, Dim1<S>], P>
  mul<S2 extends Shape>(
    other: Tensor<S2, any> & BroadcastCheck<S, S2>
  ): Tensor<Broadcast<S, S2>, P>
  mul(other: AnyTensor | number): AnyTensor {
    const b = coerce(other, this)
    const out = rawBinary(this, b, "mul")
    return withGrad(out, "mul", [this, b], g => [
      sumTo(rawBinary(g, b, "mul"), this.shape),
      sumTo(rawBinary(g, this, "mul"), b.shape)
    ])
  }

  div(other: number): Tensor<S, P>
  div(
    this: Tensor<[Dim0<S>, 1], P>,
    other: Tensor<[1, Dim0<S>], any>
  ): Tensor<[Dim0<S>, Dim0<S>], P>
  div(
    this: Tensor<[1, Dim1<S>], P>,
    other: Tensor<[Dim1<S>, 1], any>
  ): Tensor<[Dim1<S>, Dim1<S>], P>
  div<S2 extends Shape>(
    other: Tensor<S2, any> & BroadcastCheck<S, S2>
  ): Tensor<Broadcast<S, S2>, P>
  div(other: AnyTensor | number): AnyTensor {
    const b = coerce(other, this)
    const out = rawBinary(this, b, "div")
    return withGrad(out, "div", [this, b], g => [
      sumTo(rawBinary(g, b, "div"), this.shape),
      sumTo(
        rawBinary(rawBinary(g, out, "mul"), b, "negDiv"),
        b.shape
      )
    ])
  }

  /**
   * Elementwise maximum. Gradient goes wholly to whichever operand won;
   * ties go to the left one.
   */
  maximum(other: number): Tensor<S, P>
  maximum<S2 extends Shape>(
    other: Tensor<S2, any> & BroadcastCheck<S, S2>
  ): Tensor<Broadcast<S, S2>, P>
  maximum(other: AnyTensor | number): AnyTensor {
    const b = coerce(other, this)
    const out = rawBinary(this, b, "maximum")
    return withGrad(out, "maximum", [this, b], g => [
      sumTo(
        rawBinary(g, rawBinary(this, b, "ge"), "mul"),
        this.shape
      ),
      sumTo(
        rawBinary(g, rawBinary(this, b, "lt"), "mul"),
        b.shape
      )
    ])
  }

  /** Elementwise minimum; ties go to the left operand. */
  minimum(other: number): Tensor<S, P>
  minimum<S2 extends Shape>(
    other: Tensor<S2, any> & BroadcastCheck<S, S2>
  ): Tensor<Broadcast<S, S2>, P>
  minimum(other: AnyTensor | number): AnyTensor {
    const b = coerce(other, this)
    const out = rawBinary(this, b, "minimum")
    return withGrad(out, "minimum", [this, b], g => [
      sumTo(
        rawBinary(g, rawBinary(this, b, "le"), "mul"),
        this.shape
      ),
      sumTo(
        rawBinary(g, rawBinary(this, b, "gt"), "mul"),
        b.shape
      )
    ])
  }

  /**
   * Clamp into `[min, max]`; pass `null` for an open end. Gradient is 1
   * inside the range and 0 outside, since this is `maximum` composed
   * with `minimum`.
   */
  clamp(
    min: number | null,
    max: number | null = null
  ): Tensor<S, P> {
    let out = this as AnyTensor
    if (min !== null) out = out.maximum(min)
    if (max !== null) out = out.minimum(max)
    return out as any
  }

  // Comparisons produce 1.0 / 0.0 masks and stop gradients: a step
  // function has zero derivative wherever it is differentiable.
  gt(other: number): Tensor<S, P>
  gt<S2 extends Shape>(
    other: Tensor<S2, any> & BroadcastCheck<S, S2>
  ): Tensor<Broadcast<S, S2>, P>
  gt(other: AnyTensor | number): AnyTensor {
    return rawBinary(this, coerce(other, this), "gt")
  }

  ge(other: number): Tensor<S, P>
  ge<S2 extends Shape>(
    other: Tensor<S2, any> & BroadcastCheck<S, S2>
  ): Tensor<Broadcast<S, S2>, P>
  ge(other: AnyTensor | number): AnyTensor {
    return rawBinary(this, coerce(other, this), "ge")
  }

  lt(other: number): Tensor<S, P>
  lt<S2 extends Shape>(
    other: Tensor<S2, any> & BroadcastCheck<S, S2>
  ): Tensor<Broadcast<S, S2>, P>
  lt(other: AnyTensor | number): AnyTensor {
    return rawBinary(this, coerce(other, this), "lt")
  }

  le(other: number): Tensor<S, P>
  le<S2 extends Shape>(
    other: Tensor<S2, any> & BroadcastCheck<S, S2>
  ): Tensor<Broadcast<S, S2>, P>
  le(other: AnyTensor | number): AnyTensor {
    return rawBinary(this, coerce(other, this), "le")
  }

  eq(other: number): Tensor<S, P>
  eq<S2 extends Shape>(
    other: Tensor<S2, any> & BroadcastCheck<S, S2>
  ): Tensor<Broadcast<S, S2>, P>
  eq(other: AnyTensor | number): AnyTensor {
    return rawBinary(this, coerce(other, this), "eq")
  }

  pow(exponent: number): Tensor<S, P> {
    const out = rawUnary(this, "pow", exponent)
    return withGrad(out, "pow", [this], g => [
      rawBinary(
        g,
        rawUnary(this, "scalePowGrad", exponent),
        "mul"
      )
    ]) as any
  }

  neg(): Tensor<S, P> {
    const out = rawUnary(this, "neg")
    return withGrad(out, "neg", [this], g => [
      rawUnary(g, "neg")
    ]) as any
  }

  exp(): Tensor<S, P> {
    const out = rawUnary(this, "exp")
    return withGrad(out, "exp", [this], g => [
      rawBinary(g, out, "mul")
    ]) as any
  }

  log(): Tensor<S, P> {
    const out = rawUnary(this, "log")
    return withGrad(out, "log", [this], g => [
      rawBinary(g, this, "div")
    ]) as any
  }

  sqrt(): Tensor<S, P> {
    const out = rawUnary(this, "sqrt")
    return withGrad(out, "sqrt", [this], g => [
      rawBinary(g, out, "halfDiv")
    ]) as any
  }

  abs(): Tensor<S, P> {
    const out = rawUnary(this, "abs")
    return withGrad(out, "abs", [this], g => [
      rawBinary(g, this, "mulSign")
    ]) as any
  }

  relu(): Tensor<S, P> {
    const out = rawUnary(this, "relu")
    return withGrad(out, "relu", [this], g => [
      rawBinary(g, this, "reluGrad")
    ]) as any
  }

  leakyRelu(negativeSlope = 0.01): Tensor<S, P> {
    const out = rawUnary(this, "leakyRelu", negativeSlope)
    return withGrad(out, "leakyRelu", [this], g => [
      rawBinary(g, this, "leakyReluGrad", negativeSlope)
    ]) as any
  }

  sigmoid(): Tensor<S, P> {
    const out = rawUnary(this, "sigmoid")
    return withGrad(out, "sigmoid", [this], g => [
      rawBinary(g, out, "sigmoidGrad")
    ]) as any
  }

  tanh(): Tensor<S, P> {
    const out = rawUnary(this, "tanh")
    return withGrad(out, "tanh", [this], g => [
      rawBinary(g, out, "tanhGrad")
    ]) as any
  }

  softmax<D extends number>(
    dim: D & DimCheck<S, D>
  ): Tensor<S, P> {
    const shifted = this.sub(
      this.max(dim as number as any, true).detach() as any
    )
    const e = shifted.exp()
    return e.div(
      (e as AnyTensor).sum(dim as any, true) as any
    ) as any
  }

  logSoftmax<D extends number>(
    dim: D & DimCheck<S, D>
  ): Tensor<S, P> {
    const shifted = this.sub(
      this.max(dim as number as any, true).detach() as any
    ) as AnyTensor
    const logSumExp = shifted
      .exp()
      .sum(dim as any, true)
      .log()
    return shifted.sub(logSumExp as any) as any
  }

  matmul<S2 extends Shape>(
    other: Tensor<S2, any> & MatMulCheck<S, S2>
  ): Tensor<MatMul<S, S2>, P> {
    const self = this as AnyTensor
    const b = other as AnyTensor
    if (self.rank === 0 || b.rank === 0)
      throw new Error("matmul requires rank >= 1 operands")
    const A = self.rank === 1 ? self.unsqueeze(0) : self
    const B = b.rank === 1 ? b.unsqueeze(-1) : b
    let out = matmul2(A, B)
    if (b.rank === 1) out = out.squeezeDim(-1)
    if (self.rank === 1)
      out = out.squeezeDim((b.rank === 1 ? -1 : -2) as any)
    return out as any
  }

  dot<S2 extends Shape>(
    other: Tensor<S2, any> & MatMulCheck<S, S2>
  ): Tensor<MatMul<S, S2>, P> {
    return this.matmul(other)
  }

  sum(): Tensor<[], P>
  sum<D extends number>(
    dim: D & DimCheck<S, D>
  ): Tensor<ReduceDim<S, D>, P>
  sum<D extends number, const K extends boolean>(
    dim: D & DimCheck<S, D>,
    keepdim: K
  ): Tensor<ReduceDim<S, D, K>, P>
  sum(dim?: number, keepdim = false): AnyTensor {
    if (dim === undefined) {
      const out = rawReduceAll(this, "sum")
      return withGrad(out, "sum", [this], g => [
        rawBroadcastTo(g, [...this.shape])
      ])
    }
    const d = normalizeDim(dim, this.shape.length)
    const out = rawSum(this, d, keepdim)
    const keepShape = this.shape.map((s, i) =>
      i === d ? 1 : s
    )
    return withGrad(out, "sum", [this], g => {
      const gk = keepdim ? g : reshapeRaw(g, keepShape)
      return [rawBroadcastTo(gk, [...this.shape])]
    })
  }

  mean(): Tensor<[], P>
  mean<D extends number>(
    dim: D & DimCheck<S, D>
  ): Tensor<ReduceDim<S, D>, P>
  mean<D extends number, const K extends boolean>(
    dim: D & DimCheck<S, D>,
    keepdim: K
  ): Tensor<ReduceDim<S, D, K>, P>
  mean(dim?: number, keepdim = false): AnyTensor {
    if (dim === undefined)
      return (this.sum() as AnyTensor).div(this.numel)
    const d = normalizeDim(dim, this.shape.length)
    return (this as AnyTensor)
      .sum(d as any, keepdim as any)
      .div(this.shape[d]!)
  }

  max(): Tensor<[], P>
  max<D extends number>(
    dim: D & DimCheck<S, D>
  ): Tensor<ReduceDim<S, D>, P>
  max<D extends number, const K extends boolean>(
    dim: D & DimCheck<S, D>,
    keepdim: K
  ): Tensor<ReduceDim<S, D, K>, P>
  max(dim?: number, keepdim = false): AnyTensor {
    if (dim === undefined) return rawReduceAll(this, "max")
    return rawReduce(this, dim, keepdim, "max")
  }

  argmax<D extends number>(
    dim: D & DimCheck<S, D>
  ): Tensor<ReduceDim<S, D>, P> {
    const d = normalizeDim(dim, this.shape.length)
    return rawReduce(this, d, false, "argmax") as any
  }

  oneHot(classes: number): Tensor<[number, number], P> {
    if (this.rank !== 1)
      throw new Error("oneHot() requires a rank-1 tensor")
    if (!Number.isInteger(classes) || classes <= 0)
      throw new Error(
        `oneHot() requires a positive class count, got ${classes}`
      )
    return rawOneHot(this, classes) as any
  }

  /**
   * A contiguous slice of `length` entries along `dim`, starting at
   * `start` — the `x[:, a:b]` of tensor libraries. Used to read a block
   * of channels out of a state tensor, or to split a weight matrix into
   * the pieces of a fused layer.
   */
  narrow<L extends number>(
    dim: 0,
    start: number,
    length: L
  ): Tensor<ResizeDim<S, 0, L>, P>
  narrow<D extends number, L extends number>(
    dim: D & DimCheck<S, D>,
    start: number,
    length: L
  ): Tensor<ResizeDim<S, D, L>, P>
  narrow(
    dim: number,
    start: number,
    length: number
  ): AnyTensor {
    const d = normalizeDim(dim, this.shape.length)
    const size = this.shape[d]!
    if (
      !Number.isInteger(start) ||
      !Number.isInteger(length) ||
      start < 0 ||
      length < 0 ||
      start + length > size
    )
      throw new Error(
        `narrow(${dim}, ${start}, ${length}) is out of range for ${showShape(this.shape)}`
      )
    const out = rawNarrow(this, d, start, length)
    // The gradient is the incoming one placed back in the window and
    // zero everywhere else — a scatter over the window's indices, which
    // costs an index of `length` entries rather than zero-padding.
    return withGrad(out, "narrow", [this], g => {
      const window = makeRaw(
        Float32Array.from({ length }, (_, i) => start + i),
        [length],
        "float32"
      )
      return [rawScatterAdd(g, window, d, size)]
    })
  }

  /**
   * Gather rows along `dim`: row `j` of the result is row `index[j]` of
   * this tensor. `index` is a rank-1 tensor of integral values (typenet
   * has no integer dtype); its length becomes the size of `dim`.
   *
   * The workhorse of graph message passing: with `index` an edge list,
   * `x.indexSelect(src)` is "the state of each edge's source node".
   * Gradients flow to the gathered tensor, never to the index.
   */
  indexSelect<E extends number>(
    index: Tensor<[E], any>
  ): Tensor<ResizeDim<S, 0, E>, P>
  indexSelect<E extends number, D extends number>(
    index: Tensor<[E], any>,
    dim: D & DimCheck<S, D>
  ): Tensor<ResizeDim<S, D, E>, P>
  indexSelect(index: AnyTensor, dim = 0): AnyTensor {
    if (index.rank !== 1)
      throw new Error(
        `indexSelect() requires a rank-1 index, got ${showShape(index.shape)}`
      )
    const d = normalizeDim(dim, this.shape.length)
    const rows = this.shape[d]!
    const out = rawIndexSelect(this, index, d)
    return withGrad(out, "indexSelect", [this], g => [
      rawScatterAdd(g, index, d, rows)
    ])
  }

  /**
   * Scatter-add rows along `dim` into an output of `length` rows: row
   * `j` of this tensor is *added into* row `index[j]` of the result.
   * Rows no index points at stay zero. This is `index_add_` on a zero
   * tensor, and the exact reverse of {@link indexSelect}.
   *
   * The aggregation half of message passing: with `index` the
   * destination side of an edge list, `messages.scatterAdd(dst, n)`
   * sums each node's incoming messages.
   */
  scatterAdd<L extends number>(
    index: Tensor<[Dim0<S>], any>,
    length: L
  ): Tensor<ResizeDim<S, 0, L>, P>
  scatterAdd<L extends number, D extends number>(
    index: Tensor<[number], any>,
    length: L,
    dim: D & DimCheck<S, D>
  ): Tensor<ResizeDim<S, D, L>, P>
  scatterAdd(
    index: AnyTensor,
    length: number,
    dim = 0
  ): AnyTensor {
    if (index.rank !== 1)
      throw new Error(
        `scatterAdd() requires a rank-1 index, got ${showShape(index.shape)}`
      )
    if (!Number.isInteger(length) || length < 0)
      throw new Error(
        `scatterAdd() requires a non-negative integer length, got ${length}`
      )
    const d = normalizeDim(dim, this.shape.length)
    if (index.numel !== this.shape[d])
      throw new Error(
        `scatterAdd(): ${index.numel} indices for ${this.shape[d]} rows along dim ${d}`
      )
    const out = rawScatterAdd(this, index, d, length)
    return withGrad(out, "scatterAdd", [this], g => [
      rawIndexSelect(g, index, d)
    ])
  }

  view<const V extends number[]>(
    shape: V & ViewCheck<S, V>
  ): Tensor<ResolveView<S, V>, P> {
    const resolved = resolveViewRuntime(
      [...this.shape],
      shape as number[]
    )
    const out = reshapeRaw(this, resolved)
    return withGrad(out, "view", [this], g => [
      reshapeRaw(g, [...this.shape])
    ]) as any
  }

  reshape<const V extends number[]>(
    shape: V & ViewCheck<S, V>
  ): Tensor<ResolveView<S, V>, P> {
    return this.view(shape as any) as any
  }

  squeeze(): Tensor<Squeeze<S>, P> {
    const target = this.shape.filter(s => s !== 1)
    const out = reshapeRaw(this, target)
    return withGrad(out, "squeeze", [this], g => [
      reshapeRaw(g, [...this.shape])
    ]) as any
  }

  squeezeDim<D extends number>(
    dim: D & SqueezeDimCheck<S, D>
  ): Tensor<SqueezeDim<S, D>, P> {
    const d = normalizeDim(dim as number, this.shape.length)
    if (this.shape[d] !== 1)
      throw new Error(
        `Cannot squeeze dim ${dim} of ${showShape(this.shape)}: size is not 1`
      )
    const target = this.shape.filter((_, i) => i !== d)
    const out = reshapeRaw(this, target)
    return withGrad(out, "squeeze", [this], g => [
      reshapeRaw(g, [...this.shape])
    ]) as any
  }

  unsqueeze<D extends number>(
    dim: D & UnsqueezeCheck<S, D>
  ): Tensor<Unsqueeze<S, D>, P> {
    const d = normalizeDim(
      dim as number,
      this.shape.length,
      1
    )
    const target = [...this.shape]
    target.splice(d, 0, 1)
    const out = reshapeRaw(this, target)
    return withGrad(out, "unsqueeze", [this], g => [
      reshapeRaw(g, [...this.shape])
    ]) as any
  }

  transpose<D0 extends number, D1 extends number>(
    dim0: D0 & TransposeCheck<S, D0, D1>,
    dim1: D1
  ): Tensor<Transpose<S, D0, D1>, P> {
    const rank = this.shape.length
    const a = normalizeDim(dim0 as number, rank)
    const b = normalizeDim(dim1, rank)
    const order = [...Array(rank).keys()]
    ;[order[a], order[b]] = [order[b]!, order[a]!]
    return this.permuteRaw(order) as any
  }

  get T(): S["length"] extends 2 ?
    Tensor<Transpose<S, 0, 1>, P>
  : ErrorMessage<".T is only defined for rank-2 tensors — use transpose(d0, d1)"> {
    if (this.shape.length !== 2)
      throw new Error(
        ".T is only defined for rank-2 tensors"
      )
    return this.permuteRaw([1, 0]) as any
  }

  permute<const O extends number[]>(
    ...order: O & PermuteCheck<S, O>
  ): Tensor<Permute<S, O>, P> {
    const rank = this.shape.length
    const normalized = (order as number[]).map(d =>
      normalizeDim(d, rank)
    )
    if (
      normalized.length !== rank ||
      new Set(normalized).size !== rank
    )
      throw new Error(
        `permute(${(order as number[]).join(", ")}) is not a permutation of ${showShape(this.shape)}`
      )
    return this.permuteRaw(normalized) as any
  }

  private permuteRaw(order: number[]): AnyTensor {
    const out = rawPermute(this, order)
    const inverse = new Array<number>(order.length)
    order.forEach((d, i) => (inverse[d] = i))
    return withGrad(out, "permute", [this], g => [
      rawPermute(g, inverse)
    ])
  }

  static stack<
    const T extends readonly [AnyTensor, ...AnyTensor[]],
    const D extends number = 0
  >(
    tensors: T & StackCheck<T>,
    dim?: D
  ): Tensor<
    Stack<ShapeOf<T[0]>, T["length"], D>,
    ParamsOf<T[0]>
  > {
    const ts = tensors as readonly AnyTensor[]
    const first = ts[0]!
    for (const t of ts)
      if (!shapesEqual(t.shape, first.shape))
        throw new Error(
          `stack: all tensors must share a shape (${showShape(first.shape)} vs ${showShape(t.shape)})`
        )
    const unsqueezed = ts.map(t =>
      t.unsqueeze((dim ?? 0) as any)
    )
    let acc = unsqueezed[0]!
    for (let i = 1; i < unsqueezed.length; i++)
      acc = Tensor.cat(
        acc as any,
        unsqueezed[i]! as any,
        (dim ?? 0) as any
      ) as AnyTensor
    return acc as any
  }

  static cat<
    A extends Shape,
    B extends Shape,
    PA extends TensorParams,
    const D extends number = 0
  >(
    a: Tensor<A, PA>,
    b: Tensor<B, any> & CatCheck<A, B, D>,
    dim?: D
  ): Tensor<Cat<A, B, D>, PA> {
    const ta = a as AnyTensor
    const tb = b as AnyTensor
    const d = normalizeDim(dim ?? 0, ta.shape.length)
    if (ta.shape.length !== tb.shape.length)
      throw new Error(
        `cat: tensors must have the same rank (${showShape(ta.shape)} vs ${showShape(tb.shape)})`
      )
    for (let i = 0; i < ta.shape.length; i++)
      if (i !== d && ta.shape[i] !== tb.shape[i])
        throw new Error(
          `cat: shapes ${showShape(ta.shape)} and ${showShape(tb.shape)} differ outside dim ${d}`
        )
    const lenA = ta.shape[d]!
    const lenB = tb.shape[d]!
    const result = rawCat(ta, tb, d)
    return withGrad(result, "cat", [ta, tb], g => [
      rawNarrow(g, d, 0, lenA),
      rawNarrow(g, d, lenA, lenB)
    ]) as any
  }

  [Operator.plus](
    lhs: Tensor<S, P>,
    rhs: number
  ): Tensor<S, P>
  [Operator.plus](
    lhs: number,
    rhs: Tensor<S, P>
  ): Tensor<S, P>
  [Operator.plus](
    lhs: Tensor<[Dim0<S>, 1], P>,
    rhs: Tensor<[1, Dim0<S>], any>
  ): Tensor<[Dim0<S>, Dim0<S>], P>
  [Operator.plus](
    lhs: Tensor<[1, Dim1<S>], P>,
    rhs: Tensor<[Dim1<S>, 1], any>
  ): Tensor<[Dim1<S>, Dim1<S>], P>
  [Operator.plus]<S2 extends Shape>(
    lhs: Tensor<S, P>,
    rhs: Tensor<S2, any> & BroadcastCheck<S, S2>
  ): Tensor<Broadcast<S, S2>, P>
  [Operator.plus](lhs: any, rhs: any): any {
    return coerceLhs(lhs, rhs).add(rhs)
  }

  [Operator.minus](
    lhs: Tensor<S, P>,
    rhs: number
  ): Tensor<S, P>
  [Operator.minus](
    lhs: number,
    rhs: Tensor<S, P>
  ): Tensor<S, P>
  [Operator.minus](
    lhs: Tensor<[Dim0<S>, 1], P>,
    rhs: Tensor<[1, Dim0<S>], any>
  ): Tensor<[Dim0<S>, Dim0<S>], P>
  [Operator.minus](
    lhs: Tensor<[1, Dim1<S>], P>,
    rhs: Tensor<[Dim1<S>, 1], any>
  ): Tensor<[Dim1<S>, Dim1<S>], P>
  [Operator.minus]<S2 extends Shape>(
    lhs: Tensor<S, P>,
    rhs: Tensor<S2, any> & BroadcastCheck<S, S2>
  ): Tensor<Broadcast<S, S2>, P>
  [Operator.minus](lhs: any, rhs: any): any {
    return coerceLhs(lhs, rhs).sub(rhs)
  }

  [Operator.star](
    lhs: Tensor<S, P>,
    rhs: number
  ): Tensor<S, P>
  [Operator.star](
    lhs: number,
    rhs: Tensor<S, P>
  ): Tensor<S, P>
  [Operator.star](
    lhs: Tensor<[Dim0<S>, 1], P>,
    rhs: Tensor<[1, Dim0<S>], any>
  ): Tensor<[Dim0<S>, Dim0<S>], P>
  [Operator.star](
    lhs: Tensor<[1, Dim1<S>], P>,
    rhs: Tensor<[Dim1<S>, 1], any>
  ): Tensor<[Dim1<S>, Dim1<S>], P>
  [Operator.star]<S2 extends Shape>(
    lhs: Tensor<S, P>,
    rhs: Tensor<S2, any> & BroadcastCheck<S, S2>
  ): Tensor<Broadcast<S, S2>, P>
  [Operator.star](lhs: any, rhs: any): any {
    return coerceLhs(lhs, rhs).mul(rhs)
  }

  [Operator.slash](
    lhs: Tensor<S, P>,
    rhs: number
  ): Tensor<S, P>
  [Operator.slash](
    lhs: number,
    rhs: Tensor<S, P>
  ): Tensor<S, P>
  [Operator.slash](
    lhs: Tensor<[Dim0<S>, 1], P>,
    rhs: Tensor<[1, Dim0<S>], any>
  ): Tensor<[Dim0<S>, Dim0<S>], P>
  [Operator.slash](
    lhs: Tensor<[1, Dim1<S>], P>,
    rhs: Tensor<[Dim1<S>, 1], any>
  ): Tensor<[Dim1<S>, Dim1<S>], P>
  [Operator.slash]<S2 extends Shape>(
    lhs: Tensor<S, P>,
    rhs: Tensor<S2, any> & BroadcastCheck<S, S2>
  ): Tensor<Broadcast<S, S2>, P>
  [Operator.slash](lhs: any, rhs: any): any {
    return coerceLhs(lhs, rhs).div(rhs)
  }

  [Operator.starStar](
    lhs: Tensor<S, P>,
    rhs: number
  ): Tensor<S, P>
  [Operator.starStar](lhs: any, rhs: any): any {
    if (typeof rhs !== "number")
      throw new Error(
        "** on tensors requires a scalar exponent"
      )
    return (lhs as AnyTensor).pow(rhs)
  }
}

type Dim0<S extends Shape> =
  S extends [infer A extends number, ...any[]] ? A : never
type Dim1<S extends Shape> =
  S extends [any, infer B extends number, ...any[]] ? B
  : never

type StackCheck<T extends readonly AnyTensor[]> =
  T[number] extends Tensor<ShapeOf<T[0]>, any> ? unknown
  : ErrorMessage<"stack: all tensors must have the same shape">

function coerce(
  value: AnyTensor | number,
  like: AnyTensor
): AnyTensor {
  if (typeof value === "number")
    return makeRaw(
      arrayCtor(like.dtype).of(value),
      [],
      like.dtype
    )
  return value
}

function coerceLhs(lhs: any, rhs: any): AnyTensor {
  if (lhs instanceof Tensor) return lhs as AnyTensor
  return coerce(lhs, rhs as AnyTensor)
}

function makeMovedData(
  t: AnyTensor,
  data: TypedArray,
  dtype: DType
): AnyTensor {
  const out = makeRaw(data, t.shape, dtype)
  return withGrad(out, "to", [t], g => [g])
}

function resolveViewRuntime(
  shape: number[],
  view: number[]
): number[] {
  const negOnes = view.filter(v => v === -1).length
  if (negOnes > 1)
    throw new Error("Only one -1 dim is allowed in view()")
  const total = prod(shape)
  if (negOnes === 1) {
    const rest = prod(view.filter(v => v !== -1))
    if (rest === 0 || total % rest !== 0)
      throw new Error(
        `Cannot view tensor of shape ${showShape(shape)} as ${showShape(view)}`
      )
    return view.map(v => (v === -1 ? total / rest : v))
  }
  if (prod(view) !== total)
    throw new Error(
      `Cannot view tensor of shape ${showShape(shape)} as ${showShape(view)} (${total} vs ${prod(view)} elements)`
    )
  return [...view]
}

function reshapeRaw(
  t: AnyTensor,
  shape: number[]
): AnyTensor {
  if (t.numel !== prod(shape))
    throw new Error(
      `Cannot reshape ${showShape(t.shape)} to ${showShape(shape)}`
    )
  if (lazyMode)
    return makeLazy(
      { op: "view", input: t },
      shape,
      t.dtype
    )
  return makeStorage(t._storage, shape, t.dtype)
}

function matmul2(a: AnyTensor, b: AnyTensor): AnyTensor {
  const out = rawMatmul(a, b)
  return withGrad(out, "matmul", [a, b], g => {
    const bt = rawPermute(b, swapLastTwo(b.shape.length))
    const at = rawPermute(a, swapLastTwo(a.shape.length))
    const da = sumTo(rawMatmul(g, bt), [...a.shape])
    const db = sumTo(rawMatmul(at, g), [...b.shape])
    return [da, db]
  })
}

function swapLastTwo(rank: number): number[] {
  const order = [...Array(rank).keys()]
  ;[order[rank - 2], order[rank - 1]] = [
    order[rank - 1]!,
    order[rank - 2]!
  ]
  return order
}

export const tensor = Tensor.of
export const zeros = Tensor.zeros
export const ones = Tensor.ones
export const full = Tensor.full
export const rand = Tensor.rand
export const randn = Tensor.randn
export const eye = Tensor.eye
export const arange = Tensor.arange
export const scalar = Tensor.scalar
export const stack = Tensor.stack
export const cat = Tensor.cat
