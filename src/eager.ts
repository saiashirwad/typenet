import { isNativeEnabled, sgemmNative } from "./backends/native.ts"
import { applyBinary, applyUnary, gelu, geluGrad, getActiveSeed, randomData, silu, siluGrad } from "./kernels.ts"
import type { BinaryOp, RandomKind, ReduceOp, UnaryOp } from "./ops.ts"
import { broadcastShapes, catShape, matmulShape, reduceShape, resizeDim } from "./shape.ts"
import { arrayCtor, broadcastStrides, contiguousStrides, type DType, prod, promoteBinaryDtype, shapesEqual, type TypedArray } from "./storage.ts"
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
  const dtype: DType = promoteBinaryDtype(a.dtype, b.dtype)
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
  const dtype: DType = promoteBinaryDtype(a.dtype, b.dtype)
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

  // W4.1 step 8: the `i-j-k` loop below reads `b` with stride `n`, which is
  // cache-hostile, and the item asked for a LOCALITY fix that keeps the
  // per-output `k` accumulation order bit-identical (8(a)) — an `i-k-j` swap
  // would reassociate the k-sum and move the last bits of every output,
  // invalidating W0.7's recorded reference curve and gate C3.
  //
  // Measured, on this machine, at four nanoGPT/MLP-shaped GEMMs, all
  // bit-identical to this loop:
  //
  //   (i, j) blocking, JB in {16,32,64,128} x IB in {8,32,128}
  //     256x784x256 1.01x   64x784x256 1.01x   256x256x1024 1.01x
  //     512x512x512 1.18x
  //   pre-transposing `b` into a scratch buffer (contiguous inner reads)
  //     256x784x256 1.25x   64x784x256 1.15x   256x256x1024 0.97x
  //     512x512x512 1.19x   64x784x10  1.03x
  //
  // The bar was >= 2x. Per step 8(c) the speedup is therefore DROPPED and
  // the reference curve is NOT re-baselined: the loop stays exactly as it
  // was, the op plumbing lands without it, and the gap is recorded here.
  // This is not the fast path — a large packed f32 GEMM goes to Accelerate
  // above — and trading the anchor of gates C3/C3R/W3.3/W4.6/W4.7/W6.2/W6.4
  // for 1.25x on the specification path is a bad trade at any speedup.
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
  const dtype: DType = promoteBinaryDtype(a.dtype, b.dtype)
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

// ---------------------------------------------------------------------------
// W4.1 semantic kernels (PLAN-V2 §2.3 / §5A.2a).
//
// Written the obvious way on purpose: this is the *specification*, not the
// fast path (W4.1 step 3). Two rules the native kernels of W4.2-W4.5 inherit:
//
//  * each kernel is the composition §5A.2a's lowering table names, in that
//    evaluation order — so A-L1's lowering is the same arithmetic, not an
//    approximation of it;
//  * a node with several outputs returns ONE flat `[total]` tensor holding
//    its outputs' elements back to back, and a `pick` node slices it. No
//    multi-output machinery exists anywhere else (W4.1 step 2).
// ---------------------------------------------------------------------------

/** Rows x last-axis width, the layout every "over the last axis" kernel uses. */
function lastAxisLayout(shape: readonly number[]): {
  rows: number
  width: number
} {
  const width = shape[shape.length - 1] ?? 1
  return { rows: width === 0 ? 0 : prod(shape) / width, width }
}

/** `outer x dimSize x inner`, the layout a reduction over `d` walks. */
function axisLayout(
  shape: readonly number[],
  d: number,
): { outer: number; dimSize: number; inner: number } {
  return {
    outer: prod(shape.slice(0, d)),
    dimSize: shape[d]!,
    inner: prod(shape.slice(d + 1)),
  }
}

function floatCtor(dtype: DType): Float32ArrayConstructor | Float64ArrayConstructor {
  if (dtype !== "float32" && dtype !== "float64") {
    throw new Error(
      `this op is float-only; got a ${dtype} operand`,
    )
  }
  return arrayCtor(dtype) as Float32ArrayConstructor | Float64ArrayConstructor
}

function mapUnary(
  a: AnyTensor,
  fn: (x: number) => number,
): AnyTensor {
  const out = new (floatCtor(a.dtype))(a.numel)
  const ad = a.data
  for (let i = 0; i < out.length; i++) out[i] = fn(ad[i]!)
  return makeRaw(out, a.shape, a.dtype)
}

function mapGrad(
  g: AnyTensor,
  a: AnyTensor,
  fn: (g: number, x: number) => number,
): AnyTensor {
  const out = new (floatCtor(a.dtype))(a.numel)
  const gd = g.data
  const ad = a.data
  for (let i = 0; i < out.length; i++) {
    out[i] = fn(gd[i]!, ad[i]!)
  }
  return makeRaw(out, a.shape, a.dtype)
}

export function evalGeluEager(a: AnyTensor): AnyTensor {
  return mapUnary(a, gelu)
}

export function evalGeluGradEager(
  g: AnyTensor,
  a: AnyTensor,
): AnyTensor {
  return mapGrad(g, a, geluGrad)
}

export function evalSiluEager(a: AnyTensor): AnyTensor {
  return mapUnary(a, silu)
}

export function evalSiluGradEager(
  g: AnyTensor,
  a: AnyTensor,
): AnyTensor {
  return mapGrad(g, a, siluGrad)
}

/**
 * `m = max(x, dim)` -> `e = exp(x - m)` -> `e / sum(e, dim)`, exactly the
 * three-step shift-and-normalise §5A.2a pins.
 *
 * `causal` folds the additive `[T, T]` mask of a decoder block into the same
 * pass: the row maximum is taken over the unmasked prefix only, and the
 * masked entries are `exp(-Infinity) = 0` *exactly*, so there is no `-1e9`
 * fudge factor and no NaN — the diagonal is never masked, so every row's
 * maximum is finite.
 */
export function evalSoftmaxEager(
  a: AnyTensor,
  d: number,
  causal: boolean,
): AnyTensor {
  const { outer, dimSize, inner } = axisLayout(a.shape, d)
  const out = new (floatCtor(a.dtype))(a.numel)
  const ad = a.data
  // Under `causal`, `d` is the last axis (checked by the dispatcher), so
  // `inner === 1` and `outer` runs over query rows; the query position
  // inside the last-but-one axis is what bounds the key range.
  const queries = causal ? a.shape[a.shape.length - 2]! : 0
  for (let i = 0; i < outer; i++) {
    for (let k = 0; k < inner; k++) {
      const base = i * dimSize * inner + k
      const limit = causal
        ? Math.min(dimSize, (i % queries) + 1)
        : dimSize
      let m = -Infinity
      for (let j = 0; j < limit; j++) {
        const v = ad[base + j * inner]!
        if (v > m) m = v
      }
      let sum = 0
      for (let j = 0; j < limit; j++) {
        const e = Math.exp(ad[base + j * inner]! - m)
        out[base + j * inner] = e
        sum += e
      }
      for (let j = 0; j < limit; j++) {
        out[base + j * inner] = out[base + j * inner]! / sum
      }
      // The masked tail stays at the zero the buffer was allocated with.
    }
  }
  return makeRaw(out, a.shape, a.dtype)
}

/** `dx_j = y_j * (g_j - sum_k g_k*y_k)`, the closed form over `y`. */
export function evalSoftmaxGradEager(
  g: AnyTensor,
  y: AnyTensor,
  d: number,
): AnyTensor {
  const { outer, dimSize, inner } = axisLayout(y.shape, d)
  const out = new (floatCtor(y.dtype))(y.numel)
  const gd = g.data
  const yd = y.data
  for (let i = 0; i < outer; i++) {
    for (let k = 0; k < inner; k++) {
      const base = i * dimSize * inner + k
      let dot = 0
      for (let j = 0; j < dimSize; j++) {
        const o = base + j * inner
        dot += gd[o]! * yd[o]!
      }
      for (let j = 0; j < dimSize; j++) {
        const o = base + j * inner
        out[o] = yd[o]! * (gd[o]! - dot)
      }
    }
  }
  return makeRaw(out, y.shape, y.dtype)
}

/**
 * `(y, mean, rstd)` flat-concatenated, normalising over the last axis:
 * `mean = sum(x)/D`, `var = sum((x-mean)^2)/D`, `rstd = (var+eps)^-0.5`,
 * `y = (x-mean)*rstd*gamma + beta`.
 */
export function evalLayerNormEager(
  x: AnyTensor,
  gamma: AnyTensor,
  beta: AnyTensor,
  eps: number,
): AnyTensor {
  const { rows, width } = lastAxisLayout(x.shape)
  const out = new (floatCtor(x.dtype))(x.numel + 2 * rows)
  const xd = x.data
  const gd = gamma.data
  const bd = beta.data
  for (let r = 0; r < rows; r++) {
    const base = r * width
    let sum = 0
    for (let j = 0; j < width; j++) sum += xd[base + j]!
    const mean = sum / width
    let acc = 0
    for (let j = 0; j < width; j++) {
      const c = xd[base + j]! - mean
      acc += c * c
    }
    const rstd = (acc / width + eps) ** -0.5
    for (let j = 0; j < width; j++) {
      out[base + j] = (xd[base + j]! - mean) * rstd * gd[j]! + bd[j]!
    }
    out[x.numel + r] = mean
    out[x.numel + rows + r] = rstd
  }
  return makeRaw(out, [out.length], x.dtype)
}

/**
 * `(dx, dgamma, dbeta)` flat-concatenated. With `a_j = g_j*gamma_j` and
 * `xhat_j = (x_j - mean)*rstd`:
 * `dx_j = rstd*(a_j - mean_k(a) - xhat_j*mean_k(a*xhat))`,
 * `dgamma_j = sum_rows g*xhat`, `dbeta_j = sum_rows g`.
 */
export function evalLayerNormGradEager(
  g: AnyTensor,
  x: AnyTensor,
  gamma: AnyTensor,
  mean: AnyTensor,
  rstd: AnyTensor,
): AnyTensor {
  const { rows, width } = lastAxisLayout(x.shape)
  const out = new (floatCtor(x.dtype))(x.numel + 2 * width)
  const gd = g.data
  const xd = x.data
  const wd = gamma.data
  const md = mean.data
  const rd = rstd.data
  const dGamma = out.subarray(x.numel, x.numel + width)
  const dBeta = out.subarray(x.numel + width)
  for (let r = 0; r < rows; r++) {
    const base = r * width
    const mu = md[r]!
    const rs = rd[r]!
    let sumA = 0
    let sumAxhat = 0
    for (let j = 0; j < width; j++) {
      const xhat = (xd[base + j]! - mu) * rs
      const a = gd[base + j]! * wd[j]!
      sumA += a
      sumAxhat += a * xhat
      dGamma[j]! += gd[base + j]! * xhat
      dBeta[j]! += gd[base + j]!
    }
    const meanA = sumA / width
    const meanAxhat = sumAxhat / width
    for (let j = 0; j < width; j++) {
      const xhat = (xd[base + j]! - mu) * rs
      const a = gd[base + j]! * wd[j]!
      out[base + j] = rs * (a - meanA - xhat * meanAxhat)
    }
  }
  return makeRaw(out, [out.length], x.dtype)
}

/** `(y, rstd)` flat-concatenated: layerNorm without the mean subtraction. */
export function evalRmsNormEager(
  x: AnyTensor,
  gamma: AnyTensor,
  eps: number,
): AnyTensor {
  const { rows, width } = lastAxisLayout(x.shape)
  const out = new (floatCtor(x.dtype))(x.numel + rows)
  const xd = x.data
  const gd = gamma.data
  for (let r = 0; r < rows; r++) {
    const base = r * width
    let acc = 0
    for (let j = 0; j < width; j++) {
      const v = xd[base + j]!
      acc += v * v
    }
    const rstd = (acc / width + eps) ** -0.5
    for (let j = 0; j < width; j++) {
      out[base + j] = xd[base + j]! * rstd * gd[j]!
    }
    out[x.numel + r] = rstd
  }
  return makeRaw(out, [out.length], x.dtype)
}

/**
 * `(dx, dgamma)` flat-concatenated. With `a_j = g_j*gamma_j` and
 * `r = rstd`: `dx_j = r*(a_j - r^2*x_j*sum_k(a_k*x_k)/D)`,
 * `dgamma_j = sum_rows g_j*x_j*r`.
 */
export function evalRmsNormGradEager(
  g: AnyTensor,
  x: AnyTensor,
  gamma: AnyTensor,
  rstd: AnyTensor,
): AnyTensor {
  const { rows, width } = lastAxisLayout(x.shape)
  const out = new (floatCtor(x.dtype))(x.numel + width)
  const gd = g.data
  const xd = x.data
  const wd = gamma.data
  const rd = rstd.data
  const dGamma = out.subarray(x.numel)
  for (let r = 0; r < rows; r++) {
    const base = r * width
    const rs = rd[r]!
    let dot = 0
    for (let j = 0; j < width; j++) {
      const a = gd[base + j]! * wd[j]!
      dot += a * xd[base + j]!
      dGamma[j]! += gd[base + j]! * xd[base + j]! * rs
    }
    const scale = (rs * rs * dot) / width
    for (let j = 0; j < width; j++) {
      const a = gd[base + j]! * wd[j]!
      out[base + j] = rs * (a - xd[base + j]! * scale)
    }
  }
  return makeRaw(out, [out.length], x.dtype)
}

/**
 * `(loss, dlogits)` flat-concatenated over `[N, C]` logits and `[N]` class
 * indices: `logp = x - logSumExp(x, -1)`, `loss = -sum(logp[target])/N`,
 * `dlogits = (softmax(x, -1) - onehot(target))/N`. The `[N, C]` one-hot
 * never exists — that is the whole point of the fused node.
 */
export function evalCrossEntropyEager(
  logits: AnyTensor,
  target: AnyTensor,
): AnyTensor {
  const [n, classes] = logits.shape as [number, number]
  const out = new (floatCtor(logits.dtype))(1 + n * classes)
  const ld = logits.data
  const td = target.data as TypedArray
  const dLogits = out.subarray(1)
  let loss = 0
  for (let i = 0; i < n; i++) {
    const base = i * classes
    const t = checkIndex(td[i]!, classes, "crossEntropy")
    let m = -Infinity
    for (let j = 0; j < classes; j++) {
      const v = ld[base + j]!
      if (v > m) m = v
    }
    let sum = 0
    for (let j = 0; j < classes; j++) {
      const e = Math.exp(ld[base + j]! - m)
      dLogits[base + j] = e
      sum += e
    }
    const lse = m + Math.log(sum)
    loss += lse - ld[base + t]!
    for (let j = 0; j < classes; j++) {
      dLogits[base + j] = (dLogits[base + j]! / sum - (j === t ? 1 : 0)) / n
    }
  }
  out[0] = loss / n
  return makeRaw(out, [out.length], logits.dtype)
}

/** `m + log(sum(exp(x - m), dim))` with `m = max(x, dim)`. */
export function evalLogSumExpEager(
  a: AnyTensor,
  d: number,
  keepdim: boolean,
): AnyTensor {
  const { outer, dimSize, inner } = axisLayout(a.shape, d)
  const outShape = reduceShape(a.shape, d, keepdim)
  const out = new (floatCtor(a.dtype))(outer * inner)
  const ad = a.data
  let o = 0
  for (let i = 0; i < outer; i++) {
    for (let k = 0; k < inner; k++) {
      const base = i * dimSize * inner + k
      let m = -Infinity
      for (let j = 0; j < dimSize; j++) {
        const v = ad[base + j * inner]!
        if (v > m) m = v
      }
      let sum = 0
      for (let j = 0; j < dimSize; j++) {
        sum += Math.exp(ad[base + j * inner]! - m)
      }
      out[o++] = m + Math.log(sum)
    }
  }
  return makeRaw(out, outShape, a.dtype)
}

/** Rows of `table` addressed by an index of any rank (`Embedding`). */
export function evalGatherRowsEager(
  table: AnyTensor,
  index: AnyTensor,
): AnyTensor {
  const rows = table.shape[0]!
  const width = table.numel / (rows || 1)
  const n = index.numel
  const out = new (arrayCtor(table.dtype))(n * width)
  const td = table.data
  const id = index.data as TypedArray
  for (let i = 0; i < n; i++) {
    const src = checkIndex(id[i]!, rows, "gatherRows") * width
    for (let j = 0; j < width; j++) {
      out[i * width + j] = td[src + j]!
    }
  }
  return makeRaw(
    out,
    [...index.shape, ...table.shape.slice(1)],
    table.dtype,
  )
}

/** The transpose of {@link evalGatherRowsEager}: accumulate into `rows` rows. */
export function evalScatterAddRowsEager(
  src: AnyTensor,
  index: AnyTensor,
  rows: number,
): AnyTensor {
  const n = index.numel
  const width = src.numel / (n || 1)
  const out = new (arrayCtor(src.dtype))(rows * width)
  const sd = src.data
  const id = index.data as TypedArray
  for (let i = 0; i < n; i++) {
    const to = checkIndex(id[i]!, rows, "scatterAddRows") * width
    for (let j = 0; j < width; j++) {
      out[to + j]! += sd[i * width + j]!
    }
  }
  return makeRaw(
    out,
    [rows, ...src.shape.slice(index.shape.length)],
    src.dtype,
  )
}

/**
 * `(y, mask)` flat-concatenated. `mask` already carries the `1/(1-p)`
 * inverted-dropout scale, so `y = x*mask` and `dx = g*mask` — one draw,
 * used by both passes (see the node's own comment in `storage.ts` for why
 * the mask is an output on this runtime).
 */
export function evalDropoutEager(
  x: AnyTensor,
  p: number,
  stream: number,
  seed: number,
): AnyTensor {
  const n = x.numel
  const out = new (floatCtor(x.dtype))(2 * n)
  const xd = x.data
  const scale = 1 / (1 - p)
  const mask = out.subarray(n)
  const u = randomData("uniform", n, stream, seed, x.dtype)
  for (let i = 0; i < n; i++) {
    const keep = (u[i] as number) >= p ? scale : 0
    mask[i] = keep
    out[i] = xd[i]! * keep
  }
  return makeRaw(out, [out.length], x.dtype)
}

/** One output of a multi-output producer, sliced out of its flat buffer. */
export function evalPickEager(
  flat: AnyTensor,
  offset: number,
  shape: readonly number[],
): AnyTensor {
  const n = prod(shape)
  return makeRaw(
    (flat.data as TypedArray).subarray(offset, offset + n),
    shape,
    flat.dtype,
  )
}

/**
 * Explicit materialisation. A no-op on a runtime that materialises every
 * node anyway; it exists so Phase B's `Layout` canonicalisation has a node
 * to insert, and so the IR can say "this must be contiguous here" today.
 */
export function evalContiguousEager(a: AnyTensor): AnyTensor {
  const out = new (arrayCtor(a.dtype))(a.numel)
  const ad = a.data
  for (let i = 0; i < out.length; i++) out[i] = ad[i]!
  return makeRaw(out, a.shape, a.dtype)
}

/**
 * A reduction over several axes: one single-axis pass per axis, ASCENDING,
 * against the progressively shrinking tensor. The order is normative and is
 * not §5A.2a's descending one *by design*: ascending is exactly the chain
 * the pre-W4.1 `sumTo` emitted (`sum(0)` for each surplus leading axis, then
 * each broadcast axis left to right), so the step-6 rewrite moves no f32
 * bits — gate C3 and the pinned nanoGPT curve compare equal. Whoever
 * un-parks A-L1 must emit the wire chain in this same order (and
 * `lower-native.ts` does).
 */
export function evalReduceDimsEager(
  a: AnyTensor,
  dims: readonly number[],
  keepdim: boolean,
  op: ReduceOp,
): AnyTensor {
  let acc = a
  for (const d of dims) acc = evalReduceEager(acc, d, true, op)
  if (keepdim) return acc
  const outShape = a.shape.filter(
    (_: number, i: number) => !dims.includes(i),
  )
  return makeRaw(acc.data as TypedArray, outShape, acc.dtype)
}
