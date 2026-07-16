import { d, std } from "typegpu"
import type { TgpuMutable, TgpuRoot } from "typegpu"
import type { F32, WgslArray } from "typegpu/data"

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
  | "negDiv"
  | "halfDiv"
  | "mulSign"
  | "reluGrad"
  | "leakyReluGrad"
  | "sigmoidGrad"
  | "tanhGrad"

export type ReduceOp = "sum" | "max" | "argmax"

type FloatArray = WgslArray<F32>

export interface TypeGPUStorage {
  readonly kind: "typegpu"
  readonly root: TgpuRoot
  readonly values: TgpuMutable<FloatArray>
  readonly length: number
  disposed: boolean
}

function assertLive(storage: TypeGPUStorage): void {
  if (storage.disposed || storage.values.buffer.destroyed)
    throw new Error("TypeGPU tensor has been disposed")
}

function assertCompatible(
  a: TypeGPUStorage,
  b: TypeGPUStorage
): void {
  assertLive(a)
  assertLive(b)
  if (a.root !== b.root)
    throw new Error(
      "TypeGPU tensors belong to different roots"
    )
}

function mutable(
  root: TgpuRoot,
  length: number,
  initial?: ArrayLike<number>
): TypeGPUStorage {
  const schema = d.arrayOf(d.f32, length)
  const values = root.createMutable(
    schema,
    initial ? Array.from(initial) : undefined
  )
  return {
    kind: "typegpu",
    root,
    values,
    length,
    disposed: false
  }
}

function readonlyU32(
  root: TgpuRoot,
  values: readonly number[]
) {
  const data = values.length === 0 ? [0] : [...values]
  return root.createReadonly(
    d.arrayOf(d.u32, data.length),
    data
  )
}

function opCode(op: UnaryOp | BinaryOp): number {
  switch (op) {
    case "pow":
      return 0
    case "neg":
      return 1
    case "exp":
      return 2
    case "log":
      return 3
    case "sqrt":
      return 4
    case "abs":
      return 5
    case "relu":
      return 6
    case "leakyRelu":
      return 7
    case "sigmoid":
      return 8
    case "tanh":
      return 9
    case "scalePowGrad":
      return 10
    case "add":
      return 20
    case "sub":
      return 21
    case "mul":
      return 22
    case "div":
      return 23
    case "negDiv":
      return 24
    case "halfDiv":
      return 25
    case "mulSign":
      return 26
    case "reluGrad":
      return 27
    case "leakyReluGrad":
      return 28
    case "sigmoidGrad":
      return 29
    case "tanhGrad":
      return 30
  }
}

export function upload(
  root: TgpuRoot,
  data: Float32Array
): TypeGPUStorage {
  return mutable(root, data.length, data)
}

export function fill(
  root: TgpuRoot,
  length: number,
  value: number
): TypeGPUStorage {
  return mutable(
    root,
    length,
    new Float32Array(length).fill(value)
  )
}

export async function read(
  storage: TypeGPUStorage
): Promise<Float32Array> {
  assertLive(storage)
  return Float32Array.from(await storage.values.read())
}

export function write(
  storage: TypeGPUStorage,
  data: ArrayLike<number>
): void {
  assertLive(storage)
  if (data.length !== storage.length)
    throw new Error(
      `write() expected ${storage.length} values, got ${data.length}`
    )
  storage.values.write(Array.from(data))
}

export function dispose(storage: TypeGPUStorage): void {
  if (storage.disposed) return
  storage.disposed = true
  storage.values.buffer.destroy()
}

export function clone(a: TypeGPUStorage): TypeGPUStorage {
  assertLive(a)
  const out = mutable(a.root, a.length)
  const input = a.values
  const output = out.values
  const pipeline = a.root.createGuardedComputePipeline(
    i => {
      "use gpu"
      output.$[i] = input.$[i]
    }
  )
  pipeline.dispatchThreads(a.length)
  return out
}

export function unary(
  a: TypeGPUStorage,
  op: UnaryOp,
  parameter = 0
): TypeGPUStorage {
  assertLive(a)
  const out = mutable(a.root, a.length)
  const input = a.values
  const output = out.values
  const code = opCode(op)
  const pipeline = a.root.createGuardedComputePipeline(
    i => {
      "use gpu"
      const x = input.$[i]
      let value = x
      if (code === 0) value = std.pow(x, parameter)
      else if (code === 1) value = -x
      else if (code === 2) value = std.exp(x)
      else if (code === 3) value = std.log(x)
      else if (code === 4) value = std.sqrt(x)
      else if (code === 5) value = std.abs(x)
      else if (code === 6) {
        value = 0
        if (x > 0) value = x
      } else if (code === 7) {
        value = parameter * x
        if (x > 0) value = x
      } else if (code === 8) value = 1 / (1 + std.exp(-x))
      else if (code === 9) value = std.tanh(x)
      else if (code === 10)
        value = parameter * std.pow(x, parameter - 1)
      output.$[i] = value
    }
  )
  pipeline.dispatchThreads(a.length)
  return out
}

export function binary(
  a: TypeGPUStorage,
  b: TypeGPUStorage,
  op: BinaryOp,
  outShape: readonly number[],
  aStrides: readonly number[],
  bStrides: readonly number[],
  parameter = 0
): TypeGPUStorage {
  assertCompatible(a, b)
  const n = outShape.reduce((x, y) => x * y, 1)
  const out = mutable(a.root, n)
  const shape = readonlyU32(a.root, outShape)
  const sa = readonlyU32(a.root, aStrides)
  const sb = readonlyU32(a.root, bStrides)
  const av = a.values
  const bv = b.values
  const output = out.values
  const rank = outShape.length
  const code = opCode(op)
  const pipeline = a.root.createGuardedComputePipeline(
    i => {
      "use gpu"
      let remaining = i
      let offsetA = d.u32(0)
      let offsetB = d.u32(0)
      for (let rd = 0; rd < rank; rd++) {
        const dim = rank - 1 - rd
        const coordinate = remaining % shape.$[dim]
        remaining = d.u32(remaining / shape.$[dim])
        offsetA += coordinate * sa.$[dim]
        offsetB += coordinate * sb.$[dim]
      }
      const x = av.$[offsetA]
      const y = bv.$[offsetB]
      let value = x + y
      if (code === 21) value = x - y
      else if (code === 22) value = x * y
      else if (code === 23) value = x / y
      else if (code === 24) value = -x / y
      else if (code === 25) value = (0.5 * x) / y
      else if (code === 26) value = x * std.sign(y)
      else if (code === 27) {
        value = 0
        if (y > 0) value = x
      } else if (code === 28) {
        value = parameter * x
        if (y > 0) value = x
      } else if (code === 29) value = x * y * (1 - y)
      else if (code === 30) value = x * (1 - y * y)
      output.$[i] = value
    }
  )
  pipeline.dispatchThreads(n)
  shape.buffer.destroy()
  sa.buffer.destroy()
  sb.buffer.destroy()
  return out
}

export function reduce(
  a: TypeGPUStorage,
  op: ReduceOp,
  outer: number,
  dimSize: number,
  inner: number
): TypeGPUStorage {
  assertLive(a)
  const out = mutable(a.root, outer * inner)
  const input = a.values
  const output = out.values
  const isMax = op !== "sum"
  const isArgmax = op === "argmax"
  const pipeline = a.root.createGuardedComputePipeline(
    i => {
      "use gpu"
      const innerSize = d.u32(inner)
      const reduceSize = d.u32(dimSize)
      const outerIndex = d.u32(i / innerSize)
      const innerIndex = i % innerSize
      const base =
        outerIndex * reduceSize * innerSize + innerIndex
      let best = d.f32(0)
      if (isMax) best = input.$[base]
      let bestIndex = d.u32(0)
      for (let j = d.u32(0); j < reduceSize; j++) {
        const value = input.$[base + j * innerSize]
        if (isMax) {
          if (value > best) {
            best = value
            bestIndex = j
          }
        } else best += value
      }
      output.$[i] = isArgmax ? d.f32(bestIndex) : best
    }
  )
  pipeline.dispatchThreads(outer * inner)
  return out
}

export function broadcast(
  a: TypeGPUStorage,
  outShape: readonly number[],
  readStrides: readonly number[]
): TypeGPUStorage {
  assertLive(a)
  const n = outShape.reduce((x, y) => x * y, 1)
  const out = mutable(a.root, n)
  const shape = readonlyU32(a.root, outShape)
  const strides = readonlyU32(a.root, readStrides)
  const input = a.values
  const output = out.values
  const rank = outShape.length
  const pipeline = a.root.createGuardedComputePipeline(
    i => {
      "use gpu"
      let remaining = i
      let offset = d.u32(0)
      for (let rd = 0; rd < rank; rd++) {
        const dim = rank - 1 - rd
        const coordinate = remaining % shape.$[dim]
        remaining = d.u32(remaining / shape.$[dim])
        offset += coordinate * strides.$[dim]
      }
      output.$[i] = input.$[offset]
    }
  )
  pipeline.dispatchThreads(n)
  shape.buffer.destroy()
  strides.buffer.destroy()
  return out
}

export function permute(
  a: TypeGPUStorage,
  outShape: readonly number[],
  readStrides: readonly number[]
): TypeGPUStorage {
  return broadcast(a, outShape, readStrides)
}

export function matmul(
  a: TypeGPUStorage,
  b: TypeGPUStorage,
  batchShape: readonly number[],
  aBatchStrides: readonly number[],
  bBatchStrides: readonly number[],
  m: number,
  k: number,
  n: number
): TypeGPUStorage {
  assertCompatible(a, b)
  const batchCount = batchShape.reduce((x, y) => x * y, 1)
  const out = mutable(a.root, batchCount * m * n)
  const shape = readonlyU32(a.root, batchShape)
  const sa = readonlyU32(a.root, aBatchStrides)
  const sb = readonlyU32(a.root, bBatchStrides)
  const av = a.values
  const bv = b.values
  const output = out.values
  const rank = batchShape.length
  const pipeline = a.root.createGuardedComputePipeline(
    index => {
      "use gpu"
      const rows = d.u32(m)
      const innerSize = d.u32(k)
      const columns = d.u32(n)
      const matrixSize = rows * columns
      const batchIndex = d.u32(index / matrixSize)
      const cell = index % matrixSize
      const row = d.u32(cell / columns)
      const column = cell % columns
      let remaining = batchIndex
      let cellA = d.u32(0)
      let cellB = d.u32(0)
      for (let rd = 0; rd < rank; rd++) {
        const dim = rank - 1 - rd
        const coordinate = remaining % shape.$[dim]
        remaining = d.u32(remaining / shape.$[dim])
        cellA += coordinate * sa.$[dim]
        cellB += coordinate * sb.$[dim]
      }
      const baseA = cellA * rows * innerSize
      const baseB = cellB * innerSize * columns
      let sum = d.f32(0)
      for (let p = d.u32(0); p < innerSize; p++)
        sum +=
          av.$[baseA + row * innerSize + p] *
          bv.$[baseB + p * columns + column]
      output.$[index] = sum
    }
  )
  pipeline.dispatchThreads(batchCount * m * n)
  shape.buffer.destroy()
  sa.buffer.destroy()
  sb.buffer.destroy()
  return out
}

export function narrow(
  a: TypeGPUStorage,
  outer: number,
  dimSize: number,
  inner: number,
  start: number,
  length: number
): TypeGPUStorage {
  assertLive(a)
  const out = mutable(a.root, outer * length * inner)
  const input = a.values
  const output = out.values
  const block = length * inner
  const pipeline = a.root.createGuardedComputePipeline(
    i => {
      "use gpu"
      const blockSize = d.u32(block)
      const outerIndex = d.u32(i / blockSize)
      const within = i % blockSize
      output.$[i] =
        input.$[
          outerIndex * d.u32(dimSize) * d.u32(inner) +
            d.u32(start * inner) +
            within
        ]
    }
  )
  pipeline.dispatchThreads(outer * length * inner)
  return out
}

export function concatenate(
  a: TypeGPUStorage,
  b: TypeGPUStorage,
  outer: number,
  lenA: number,
  lenB: number,
  inner: number
): TypeGPUStorage {
  assertCompatible(a, b)
  const blockA = lenA * inner
  const blockB = lenB * inner
  const blockOut = blockA + blockB
  const out = mutable(a.root, outer * blockOut)
  const av = a.values
  const bv = b.values
  const output = out.values
  const pipeline = a.root.createGuardedComputePipeline(
    i => {
      "use gpu"
      const outSize = d.u32(blockOut)
      const aSize = d.u32(blockA)
      const bSize = d.u32(blockB)
      const outerIndex = d.u32(i / outSize)
      const within = i % outSize
      if (within < aSize)
        output.$[i] = av.$[outerIndex * aSize + within]
      else
        output.$[i] =
          bv.$[outerIndex * bSize + within - aSize]
    }
  )
  pipeline.dispatchThreads(outer * blockOut)
  return out
}

export function oneHot(
  targets: TypeGPUStorage,
  batch: number,
  classes: number
): TypeGPUStorage {
  assertLive(targets)
  const out = mutable(targets.root, batch * classes)
  const input = targets.values
  const output = out.values
  const classCount = d.u32(classes)
  const pipeline =
    targets.root.createGuardedComputePipeline(i => {
      "use gpu"
      const row = d.u32(i / classCount)
      const column = i % classCount
      output.$[i] =
        column === d.u32(input.$[row]) ? d.f32(1) : d.f32(0)
    })
  pipeline.dispatchThreads(batch * classes)
  return out
}

export function sgdStep(
  parameter: TypeGPUStorage,
  gradient: TypeGPUStorage,
  velocity: TypeGPUStorage | null,
  lr: number,
  momentum: number,
  weightDecay: number
): void {
  assertCompatible(parameter, gradient)
  if (velocity) assertCompatible(parameter, velocity)
  const p = parameter.values
  const g = gradient.values
  if (velocity) {
    const v = velocity.values
    const pipeline =
      parameter.root.createGuardedComputePipeline(i => {
        "use gpu"
        let grad = g.$[i]
        if (weightDecay !== 0) grad += weightDecay * p.$[i]
        v.$[i] = momentum * v.$[i] + grad
        p.$[i] -= lr * v.$[i]
      })
    pipeline.dispatchThreads(parameter.length)
  } else {
    const pipeline =
      parameter.root.createGuardedComputePipeline(i => {
        "use gpu"
        let grad = g.$[i]
        if (weightDecay !== 0) grad += weightDecay * p.$[i]
        p.$[i] -= lr * grad
      })
    pipeline.dispatchThreads(parameter.length)
  }
}

export function adamStep(
  parameter: TypeGPUStorage,
  gradient: TypeGPUStorage,
  firstMoment: TypeGPUStorage,
  secondMoment: TypeGPUStorage,
  lr: number,
  beta1: number,
  beta2: number,
  eps: number,
  weightDecay: number,
  biasCorrection1: number,
  biasCorrection2: number
): void {
  assertCompatible(parameter, gradient)
  assertCompatible(parameter, firstMoment)
  assertCompatible(parameter, secondMoment)
  const p = parameter.values
  const g = gradient.values
  const m = firstMoment.values
  const v = secondMoment.values
  const pipeline =
    parameter.root.createGuardedComputePipeline(i => {
      "use gpu"
      let grad = g.$[i]
      if (weightDecay !== 0) grad += weightDecay * p.$[i]
      m.$[i] = beta1 * m.$[i] + (1 - beta1) * grad
      v.$[i] = beta2 * v.$[i] + (1 - beta2) * grad * grad
      const mHat = m.$[i] / biasCorrection1
      const vHat = v.$[i] / biasCorrection2
      p.$[i] -= (lr * mHat) / (std.sqrt(vHat) + eps)
    })
  pipeline.dispatchThreads(parameter.length)
}
