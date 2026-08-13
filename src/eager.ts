import { isNativeEnabled, sgemmNative } from "./backends/native.ts"
import { applyBinary, applyUnary, getActiveSeed, randomData } from "./kernels.ts"
import type { BinaryOp, RandomKind, ReduceOp, UnaryOp } from "./ops.ts"
import { broadcastShapes, catShape, matmulShape, reduceShape, resizeDim } from "./shape.ts"
import { arrayCtor, broadcastStrides, contiguousStrides, type DType, prod, shapesEqual, type TypedArray } from "./storage.ts"
import { type AnyTensor, makeRaw } from "./tensor.ts"

// ---------------------------------------------------------------------------
// The JS eager kernels: one per IR kind, values in, values out. These
// are the numeric spec — the lazy interpreter replays them per node and
// the native backend must match them. No kernel consults the lazy flag;
// dispatch lives in ir.ts.
// ---------------------------------------------------------------------------

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

export function evalBinaryEager(
  a: AnyTensor,
  b: AnyTensor,
  op: BinaryOp,
  parameter: number,
): AnyTensor {
  const outShape = broadcastShapes(a.shape, b.shape)
  const dtype: DType = a.dtype === "float64" || b.dtype === "float64"
    ? "float64"
    : "float32"
  const n = prod(outShape)
  const out = new (arrayCtor(dtype))(n)
  const ad = a.data
  const bd = b.data
  // Every path applies the same scalar kernel, so the fast paths are
  // bit-identical to the strided walk — they only skip the odometer.
  if (shapesEqual(a.shape, b.shape)) {
    for (let i = 0; i < n; i++) {
      out[i] = applyBinary(op, ad[i]!, bd[i]!, parameter)
    }
    return makeRaw(out, outShape, dtype)
  }
  if (b.numel === 1) {
    const s = bd[0]!
    for (let i = 0; i < n; i++) {
      out[i] = applyBinary(op, ad[i]!, s, parameter)
    }
    return makeRaw(out, outShape, dtype)
  }
  if (a.numel === 1) {
    const s = ad[0]!
    for (let i = 0; i < n; i++) {
      out[i] = applyBinary(op, s, bd[i]!, parameter)
    }
    return makeRaw(out, outShape, dtype)
  }
  const sa = broadcastStrides(a.shape, outShape)
  const sb = broadcastStrides(b.shape, outShape)
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

export function evalUnaryEager(
  a: AnyTensor,
  op: UnaryOp,
  parameter: number,
): AnyTensor {
  const out = new (arrayCtor(a.dtype))(a.data.length)
  for (let i = 0; i < a.data.length; i++) {
    out[i] = applyUnary(op, a.data[i]!, parameter)
  }
  return makeRaw(out, a.shape, a.dtype)
}

export function evalReduceEager(
  a: AnyTensor,
  d: number,
  keepdim: boolean,
  op: ReduceOp,
): AnyTensor {
  const outShape = reduceShape(a.shape, d, false)
  const keepShape = reduceShape(a.shape, d, true)
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

export function evalReduceAllEager(
  a: AnyTensor,
  op: "sum" | "max",
): AnyTensor {
  const init = op === "sum" ? 0 : -Infinity
  let acc = init
  for (let i = 0; i < a.data.length; i++) {
    if (op === "sum") acc += a.data[i]!
    else if (a.data[i]! > acc) acc = a.data[i]!
  }
  return makeRaw(arrayCtor(a.dtype).of(acc), [], a.dtype)
}

export function evalBroadcastToEager(
  a: AnyTensor,
  shape: readonly number[],
): AnyTensor {
  const n = prod(shape)
  const sa = broadcastStrides(a.shape, shape)
  const out = new (arrayCtor(a.dtype))(n)
  const ad = a.data
  forEachStrided(shape, [sa], (i, offs) => {
    out[i] = ad[offs[0]!]!
  })
  return makeRaw(out, shape, a.dtype)
}

export function evalPermuteEager(
  a: AnyTensor,
  order: readonly number[],
): AnyTensor {
  const outShape = order.map(i => a.shape[i]!)
  const inStrides = contiguousStrides(a.shape)
  const readStrides = order.map(i => inStrides[i]!)
  const out = new (arrayCtor(a.dtype))(a.numel)
  const ad = a.data
  forEachStrided(outShape, [readStrides], (i, offs) => {
    out[i] = ad[offs[0]!]!
  })
  return makeRaw(out, outShape, a.dtype)
}

export function evalMatmulEager(
  a: AnyTensor,
  b: AnyTensor,
): AnyTensor {
  const ar = a.shape.length
  const br = b.shape.length
  const m = a.shape[ar - 2]!
  const k = a.shape[ar - 1]!
  const n = b.shape[br - 1]!
  const outShape = matmulShape(a.shape, b.shape)
  const batchA = a.shape.slice(0, -2)
  const batchB = b.shape.slice(0, -2)
  const batch = outShape.slice(0, -2)
  const dtype: DType = a.dtype === "float64" || b.dtype === "float64"
    ? "float64"
    : "float32"
  const batchCount = prod(batch)
  // A large packed f32 GEMM goes to Accelerate when the native addon is
  // enabled — eager + useNative() is no longer "ignore native". Only
  // + and * are involved, in the same association BLAS uses row-major,
  // so tests that require bit-stability opt out with disableNative().
  if (
    dtype === "float32"
    && batchCount === 1
    && m * k * n > 65536
    && isNativeEnabled()
  ) {
    const data = sgemmNative(
      a.data as Float32Array,
      b.data as Float32Array,
      m,
      k,
      n,
    )
    if (data) return makeRaw(data, outShape, dtype)
  }
  const saBatch = broadcastStrides(batchA, batch)
  const sbBatch = broadcastStrides(batchB, batch)
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

export function evalNarrowEager(
  a: AnyTensor,
  d: number,
  start: number,
  length: number,
): AnyTensor {
  const outShape = resizeDim(a.shape, d, length)
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

export function evalOneHotEager(
  a: AnyTensor,
  classes: number,
): AnyTensor {
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

export function evalCatEager(
  a: AnyTensor,
  b: AnyTensor,
  dim: number,
): AnyTensor {
  const outShape = catShape(a.shape, b.shape, dim)
  const dtype: DType = a.dtype === "float64" || b.dtype === "float64"
    ? "float64"
    : "float32"
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

export function evalRandomEager(
  kind: RandomKind,
  shape: readonly number[],
  stream: number,
  dtype: DType,
): AnyTensor {
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

// `index` holds integral values. typenet's integer dtypes (`int32` /
// `int64`) store them directly; a float index (the pre-integer default)
// addresses 16.7M rows exactly, the f32 mantissa limit.

function checkIndex(
  value: number | bigint,
  limit: number,
  what: string,
): number {
  if (typeof value === "bigint") {
    if (value < 0n || value >= BigInt(limit)) {
      throw new Error(
        `${what}: index ${value} out of range for ${limit} rows`,
      )
    }
    return Number(value)
  }
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

export function evalIndexSelectEager(
  a: AnyTensor,
  index: AnyTensor,
  dim: number,
): AnyTensor {
  const length = index.numel
  const outShape = resizeDim(a.shape, dim, length)
  const strides = contiguousStrides(a.shape)
  const outer = prod(a.shape.slice(0, dim))
  const inner = strides[dim]!
  const dimSize = a.shape[dim]!
  const out = new (arrayCtor(a.dtype))(prod(outShape))
  const ad = a.data
  const id = index.data as TypedArray
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

export function evalScatterAddEager(
  a: AnyTensor,
  index: AnyTensor,
  dim: number,
  length: number,
): AnyTensor {
  const outShape = resizeDim(a.shape, dim, length)
  const strides = contiguousStrides(a.shape)
  const outer = prod(a.shape.slice(0, dim))
  const inner = strides[dim]!
  const srcLength = a.shape[dim]!
  const out = new (arrayCtor(a.dtype))(prod(outShape))
  const ad = a.data
  const id = index.data as TypedArray
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
