import { applyBinary, applyUnary, getActiveSeed, nextStream, randomData } from "./kernels.ts"
import { lazyMode, makeLazy } from "./lazy.ts"
import type { Shape } from "./shape.ts"
import {
  arrayCtor,
  type BinaryOp,
  broadcastShapes,
  broadcastStrides,
  contiguousStrides,
  type DType,
  normalizeDim,
  prod,
  type RandomKind,
  type ReduceOp,
  shapesEqual,
  showShape,
  type UnaryOp,
} from "./storage.ts"
import { type AnyTensor, type DefaultParams, makeRaw, makeStorage, type Tensor } from "./tensor.ts"

function forEachStrided(
  shape: readonly number[],
  strideSets: readonly (readonly number[])[],
  fn: (i: number, offsets: readonly number[]) => void,
): void {
  const n = prod(shape)
  const rank = shape.length
  const idx = new Array<number>(rank).fill(0)
  const offs = strideSets.map(() => 0)
  for (let i = 0; i < n; i++) {
    fn(i, offs)
    for (let d = rank - 1; d >= 0; d--) {
      idx[d]++
      for (let s = 0; s < strideSets.length; s++) {
        offs[s]! += strideSets[s]![d]!
      }
      if (idx[d]! < shape[d]!) break
      idx[d] = 0
      for (let s = 0; s < strideSets.length; s++) {
        offs[s]! -= strideSets[s]![d]! * shape[d]!
      }
    }
  }
}

function sumTo(t: AnyTensor, shape: number[]): AnyTensor {
  if (shapesEqual(t.shape, shape)) return t
  let out = t

  while (out.shape.length > shape.length) {
    out = rawSum(out, 0, false)
  }

  for (let i = 0; i < shape.length; i++) {
    if (shape[i] === 1 && out.shape[i] !== 1) {
      out = rawSum(out, i, true)
    }
  }
  return out
}

function rawBinary(
  a: AnyTensor,
  b: AnyTensor,
  op: BinaryOp,
  parameter = 0,
): AnyTensor {
  const outShape = broadcastShapes(a.shape, b.shape)
  const dtype: DType = a.dtype === "float64" || b.dtype === "float64"
    ? "float64"
    : "float32"
  const sa = broadcastStrides(a.shape, outShape)
  const sb = broadcastStrides(b.shape, outShape)
  if (lazyMode) {
    return makeLazy(
      { op: "binary", kind: op, parameter, a, b },
      outShape,
      dtype,
    )
  }
  const n = prod(outShape)
  const out = new (arrayCtor(dtype))(n)
  const ad = a.data
  const bd = b.data
  forEachStrided(outShape, [sa, sb], (i, offs) => {
    out[i] = applyBinary(
      op,
      ad[offs[0]!]!,
      bd[offs[1]!]!,
      parameter,
    )
  })
  return makeRaw(out, outShape, dtype)
}

function rawUnary(
  a: AnyTensor,
  op: UnaryOp,
  parameter = 0,
): AnyTensor {
  if (lazyMode) {
    return makeLazy(
      { op: "unary", kind: op, parameter, input: a },
      a.shape,
      a.dtype,
    )
  }
  const out = new (arrayCtor(a.dtype))(a.data.length)
  for (let i = 0; i < a.data.length; i++) {
    out[i] = applyUnary(op, a.data[i]!, parameter)
  }
  return makeRaw(out, a.shape, a.dtype)
}

function rawSum(
  a: AnyTensor,
  dim: number,
  keepdim: boolean,
): AnyTensor {
  return rawReduce(a, dim, keepdim, "sum")
}

function rawReduce(
  a: AnyTensor,
  dim: number,
  keepdim: boolean,
  op: ReduceOp,
): AnyTensor {
  const d = normalizeDim(dim, a.shape.length)
  const outShape = a.shape.filter(
    (_: number, i: number) => i !== d,
  )
  const keepShape = a.shape.map((s: number, i: number) => i === d ? 1 : s)
  if (lazyMode) {
    return makeLazy(
      { op: "reduce", kind: op, dim: d, keepdim, input: a },
      keepdim ? keepShape : outShape,
      a.dtype,
    )
  }
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
    a.dtype,
  )
}

function rawReduceAll(
  a: AnyTensor,
  op: "sum" | "max",
): AnyTensor {
  if (lazyMode) {
    return makeLazy(
      { op: "reduceAll", kind: op, input: a },
      [],
      a.dtype,
    )
  }
  const init = op === "sum" ? 0 : -Infinity
  let acc = init
  for (let i = 0; i < a.data.length; i++) {
    if (op === "sum") acc += a.data[i]!
    else if (a.data[i]! > acc) acc = a.data[i]!
  }
  return makeRaw(arrayCtor(a.dtype).of(acc), [], a.dtype)
}

function rawBroadcastTo(
  a: AnyTensor,
  shape: number[],
): AnyTensor {
  if (shapesEqual(a.shape, shape)) return a
  if (lazyMode) {
    return makeLazy(
      { op: "broadcastTo", input: a },
      shape,
      a.dtype,
    )
  }
  const n = prod(shape)
  const sa = broadcastStrides(a.shape, shape)
  const out = new (arrayCtor(a.dtype))(n)
  const ad = a.data
  forEachStrided(shape, [sa], (i, offs) => {
    out[i] = ad[offs[0]!]!
  })
  return makeRaw(out, shape, a.dtype)
}

function rawPermute(
  a: AnyTensor,
  order: number[],
): AnyTensor {
  const outShape = order.map(i => a.shape[i]!)
  if (lazyMode) {
    return makeLazy(
      { op: "permute", order, input: a },
      outShape,
      a.dtype,
    )
  }
  const inStrides = contiguousStrides(a.shape)
  const readStrides = order.map(i => inStrides[i]!)
  const out = new (arrayCtor(a.dtype))(a.numel)
  const ad = a.data
  forEachStrided(outShape, [readStrides], (i, offs) => {
    out[i] = ad[offs[0]!]!
  })
  return makeRaw(out, outShape, a.dtype)
}

function rawMatmul(a: AnyTensor, b: AnyTensor): AnyTensor {
  const ar = a.shape.length
  const br = b.shape.length
  const m = a.shape[ar - 2]!
  const k = a.shape[ar - 1]!
  const k2 = b.shape[br - 2]!
  const n = b.shape[br - 1]!
  if (k !== k2) {
    throw new Error(
      `matmul: inner dimensions do not match (${showShape(a.shape)} @ ${showShape(b.shape)})`,
    )
  }
  const batchA = a.shape.slice(0, -2)
  const batchB = b.shape.slice(0, -2)
  const batch = broadcastShapes(batchA, batchB)
  const outShape = [...batch, m, n]
  const dtype: DType = a.dtype === "float64" || b.dtype === "float64"
    ? "float64"
    : "float32"
  const batchCount = prod(batch)
  const saBatch = broadcastStrides(batchA, batch)
  const sbBatch = broadcastStrides(batchB, batch)
  if (lazyMode) {
    return makeLazy({ op: "matmul", a, b }, outShape, dtype)
  }
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
        for (let p = 0; p < k; p++) {
          acc += ad[baseA + i * k + p]! * bd[baseB + p * n + j]!
        }
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
  length: number,
): AnyTensor {
  const d = normalizeDim(dim, a.shape.length)
  const outShape = a.shape.map((s: number, i: number) => i === d ? length : s)
  if (lazyMode) {
    return makeLazy(
      { op: "narrow", dim: d, start, length, input: a },
      outShape,
      a.dtype,
    )
  }
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
  classes: number,
): AnyTensor {
  if (lazyMode) {
    return makeLazy(
      { op: "oneHot", classes, input: a },
      [a.numel, classes],
      a.dtype,
    )
  }
  const out = new (arrayCtor(a.dtype))(a.numel * classes)
  for (let i = 0; i < a.numel; i++) {
    const target = a.data[i]!
    if (
      !Number.isInteger(target)
      || target < 0
      || target >= classes
    ) {
      throw new Error(
        `oneHot: target ${target} out of range for ${classes} classes`,
      )
    }
    out[i * classes + target] = 1
  }
  return makeRaw(out, [a.numel, classes], a.dtype)
}

function rawCat(
  a: AnyTensor,
  b: AnyTensor,
  dim: number,
): AnyTensor {
  const outShape = a.shape.map((s: number, i: number) => i === dim ? s + b.shape[dim]! : s)
  const dtype: DType = a.dtype === "float64" || b.dtype === "float64"
    ? "float64"
    : "float32"
  if (lazyMode) {
    return makeLazy(
      { op: "cat", a, b, dim },
      outShape,
      dtype,
    )
  }
  const strides = contiguousStrides(outShape)
  const outer = prod(outShape.slice(0, dim))
  const inner = strides[dim]!
  const lenA = a.shape[dim]!
  const lenB = b.shape[dim]!
  const out = new (arrayCtor(dtype))(prod(outShape))
  let o = 0
  for (let i = 0; i < outer; i++) {
    for (let j = 0; j < lenA; j++) {
      for (let k = 0; k < inner; k++) {
        out[o++] = a.data[(i * lenA + j) * inner + k]!
      }
    }
    for (let j = 0; j < lenB; j++) {
      for (let k = 0; k < inner; k++) {
        out[o++] = b.data[(i * lenB + j) * inner + k]!
      }
    }
  }
  return makeRaw(out, outShape, dtype)
}

function rawRandom(
  kind: RandomKind,
  shape: readonly number[],
  stream: number,
  dtype: DType,
): AnyTensor {
  if (lazyMode) {
    return makeLazy(
      { op: "random", kind, stream },
      shape,
      dtype,
    )
  }
  return makeRaw(
    randomData(
      kind,
      prod(shape),
      stream,
      getActiveSeed(),
      dtype,
    ),
    shape,
    dtype,
  )
}

/**
 * Uniform values in [0, 1), redrawn on every evaluation.
 *
 * Seeded by `configure({ seed })`, not `Math.random`.
 */
export function uniform<const Sh extends Shape>(
  shape: Sh,
): Tensor<Sh, DefaultParams> {
  return rawRandom(
    "uniform",
    shape,
    nextStream(),
    "float32",
  ) as any
}

/** Standard normal values, redrawn per evaluation. See {@link uniform}. */
export function normal<const Sh extends Shape>(
  shape: Sh,
): Tensor<Sh, DefaultParams> {
  return rawRandom(
    "normal",
    shape,
    nextStream(),
    "float32",
  ) as any
}

// `index` holds integral values in a float
// tensor — typenet has no integer dtype, and a float32 mantissa
// addresses 16.7M rows exactly.

function checkIndex(
  value: number,
  limit: number,
  what: string,
): number {
  if (
    !Number.isInteger(value)
    || value < 0
    || value >= limit
  ) {
    throw new Error(
      `${what}: index ${value} out of range for ${limit} rows`,
    )
  }
  return value
}

function rawIndexSelect(
  a: AnyTensor,
  index: AnyTensor,
  dim: number,
): AnyTensor {
  const length = index.numel
  const outShape = a.shape.map((s: number, i: number) => i === dim ? length : s)
  if (lazyMode) {
    return makeLazy(
      { op: "indexSelect", dim, input: a, index },
      outShape,
      a.dtype,
    )
  }
  const strides = contiguousStrides(a.shape)
  const outer = prod(a.shape.slice(0, dim))
  const inner = strides[dim]!
  const dimSize = a.shape[dim]!
  const out = new (arrayCtor(a.dtype))(prod(outShape))
  const ad = a.data
  const id = index.data
  let o = 0
  for (let i = 0; i < outer; i++) {
    for (let j = 0; j < length; j++) {
      const base = (i * dimSize
        + checkIndex(id[j]!, dimSize, "indexSelect"))
        * inner
      for (let k = 0; k < inner; k++) {
        out[o++] = ad[base + k]!
      }
    }
  }
  return makeRaw(out, outShape, a.dtype)
}

function rawScatterAdd(
  a: AnyTensor,
  index: AnyTensor,
  dim: number,
  length: number,
): AnyTensor {
  const outShape = a.shape.map((s: number, i: number) => i === dim ? length : s)
  if (lazyMode) {
    return makeLazy(
      { op: "scatterAdd", dim, length, input: a, index },
      outShape,
      a.dtype,
    )
  }
  const strides = contiguousStrides(a.shape)
  const outer = prod(a.shape.slice(0, dim))
  const inner = strides[dim]!
  const srcLength = a.shape[dim]!
  const out = new (arrayCtor(a.dtype))(prod(outShape))
  const ad = a.data
  const id = index.data
  for (let i = 0; i < outer; i++) {
    for (let j = 0; j < srcLength; j++) {
      const to = (i * length
        + checkIndex(id[j]!, length, "scatterAdd"))
        * inner
      const from = (i * srcLength + j) * inner
      for (let k = 0; k < inner; k++) {
        out[to + k]! += ad[from + k]!
      }
    }
  }
  return makeRaw(out, outShape, a.dtype)
}

function reshapeRaw(
  t: AnyTensor,
  shape: number[],
): AnyTensor {
  if (t.numel !== prod(shape)) {
    throw new Error(
      `Cannot reshape ${showShape(t.shape)} to ${showShape(shape)}`,
    )
  }
  if (lazyMode) {
    return makeLazy(
      { op: "view", input: t },
      shape,
      t.dtype,
    )
  }
  return makeStorage(t._storage, shape, t.dtype)
}

export { sumTo }
export {
  rawBinary,
  rawBroadcastTo,
  rawCat,
  rawIndexSelect,
  rawMatmul,
  rawNarrow,
  rawOneHot,
  rawPermute,
  rawReduce,
  rawReduceAll,
  rawScatterAdd,
  rawSum,
  rawUnary,
  reshapeRaw,
}
