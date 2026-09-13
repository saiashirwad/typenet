import type { AnyTensor } from "./tensor.ts"

export type DType = "float32" | "float64" | "int32" | "int64"

// Integer dtypes are index-only, so f32/f64 promotion is the only
// relevant binary case.
export function promoteBinaryDtype(a: DType, b: DType): DType {
  return a === "float64" || b === "float64" ? "float64" : "float32"
}

import type { BinaryOp, RandomKind, ReduceOp, UnaryOp } from "./ops.ts"
export type { BinaryOp, RandomKind, ReduceOp, UnaryOp }

export type TypedArray = Float32Array | Float64Array | Int32Array | BigInt64Array

/** Typed arrays that store JS numbers; int64 storage is {@link BigInt64Array}. */
export type NumericArray = Float32Array | Float64Array | Int32Array

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
    /**
     * Axes to reduce, ascending, normalised against `input`'s rank
     * (W4.1 step 6). `sumTo` emits ONE of these instead of one node per
     * axis; the accumulation order is "ascending axis, one axis at a
     * time", which is exactly the chain the old `sumTo` emitted, so the
     * rewrite moves no f32 bits (gate C3).
     */
    dims: number[]
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
  // --- W4.1 semantic ops (PLAN-V2 §2.3) ------------------------------------
  //
  // Tensor operands keep using the generic slot names `OP_DESC` knows
  // (`input`/`a`/`b`/`index`/`grad`/`gamma`/`beta`/`mean`/`rstd`/`target`),
  // so `nodeInputs`, `formatLazyOp` and `serializeNode` stay table-driven —
  // W4.1 step 5's "no three fifteen-arm switches".
  //
  // MULTI-OUTPUT IS A LOWERING CONCEPT, NOT AN IR CONCEPT (step 2). A node
  // with several outputs produces ONE tensor whose shape is `[total]` — the
  // outputs' elements concatenated in declaration order — and each real
  // output is a `pick` node that slices it. Nothing in this file grows
  // multi-output machinery.
  | { op: "gelu"; input: AnyTensor }
  | { op: "geluGrad"; grad: AnyTensor; input: AnyTensor }
  | { op: "silu"; input: AnyTensor }
  | { op: "siluGrad"; grad: AnyTensor; input: AnyTensor }
  | {
    op: "softmax"
    dim: number
    /** Additive causal mask over the last two axes, folded into the kernel. */
    causal: boolean
    input: AnyTensor
  }
  | {
    op: "softmaxGrad"
    dim: number
    grad: AnyTensor
    /** The softmax OUTPUT, not its input: the rule is closed over `y`. */
    input: AnyTensor
  }
  /** `(y, mean, rstd)` over the last axis. */
  | {
    op: "layerNorm"
    eps: number
    input: AnyTensor
    gamma: AnyTensor
    beta: AnyTensor
  }
  /** `(dx, dgamma, dbeta)`. */
  | {
    op: "layerNormGrad"
    grad: AnyTensor
    input: AnyTensor
    gamma: AnyTensor
    mean: AnyTensor
    rstd: AnyTensor
  }
  /** `(y, rstd)` over the last axis. */
  | {
    op: "rmsNorm"
    eps: number
    input: AnyTensor
    gamma: AnyTensor
  }
  /** `(dx, dgamma)`. */
  | {
    op: "rmsNormGrad"
    grad: AnyTensor
    input: AnyTensor
    gamma: AnyTensor
    rstd: AnyTensor
  }
  /** `(loss, dlogits)` over `[N, C]` logits and `[N]` class indices. */
  | {
    op: "crossEntropy"
    input: AnyTensor
    target: AnyTensor
  }
  | {
    op: "logSumExp"
    dim: number
    keepdim: boolean
    input: AnyTensor
  }
  /** Row gather on axis 0 with an index of any rank (`Embedding`). */
  | { op: "gatherRows"; input: AnyTensor; index: AnyTensor }
  /** The transpose of `gatherRows`: accumulate into `rows` rows. */
  | {
    op: "scatterAddRows"
    rows: number
    input: AnyTensor
    index: AnyTensor
  }
  /**
   * `(y, mask)`. Two outputs, deliberately, and this is a Phase A
   * deviation from §2.3's `outs: 1` with a named reason: §2.3 assumes a
   * kernel that can REDRAW the same mask in backward from `(seed, stream)`
   * inside one program. Today's runtime has no in-program RNG replay — the
   * eager path has no per-evaluation seed at all — so the only way to
   * guarantee the backward multiplies by the SAME mask the forward did is
   * to make the mask an output. W3.4/W4.2 restore `outs: 1`.
   */
  | { op: "dropout"; p: number; stream: number; input: AnyTensor }
  /**
   * Projection of a multi-output producer. `out` is which output;
   * `offset` is where it starts in the producer's flat buffer, computed
   * once at graph-build time so the kernel needs nothing but its operand.
   */
  | {
    op: "pick"
    out: number
    offset: number
    input: AnyTensor
  }
  /** Explicit materialisation; an identity on a runtime that copies anyway. */
  | { op: "contiguous"; input: AnyTensor }

type LazyNode = LazyNodeBody & {
  shape: number[]
  dtype: DType
}

type LazyStorage = {
  readonly kind: "lazy"
  readonly node: LazyNode
}

type TensorStorage = CpuStorage | LazyStorage

export type { CpuStorage, LazyNode, LazyNodeBody, LazyStorage, TensorStorage }

function arrayCtor(
  dtype: DType,
): Float32ArrayConstructor | Float64ArrayConstructor | Int32ArrayConstructor {
  switch (dtype) {
    case "float64":
      return Float64Array
    case "int32":
      return Int32Array
    case "int64":
      throw new Error(
        "arrayCtor: int64 storage is built by convertData()",
      )
    default:
      return Float32Array
  }
}

/**
 * Element data converted to `dtype`'s storage. BigInt boundaries need an
 * explicit map (`BigInt64Array.from` refuses plain numbers, and the
 * reverse refuses bigints); a non-integral value into int64 throws,
 * matching the integer-storage contract.
 */
function convertData(
  data: ArrayLike<number> | ArrayLike<bigint>,
  dtype: DType,
): TypedArray {
  if (dtype === "int64") {
    return BigInt64Array.from(
      data as ArrayLike<number | bigint>,
      v => BigInt(v),
    )
  }
  const ctor = arrayCtor(dtype)
  return data instanceof BigInt64Array
    ? ctor.from(Array.from(data, Number))
    : ctor.from(data as ArrayLike<number>)
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

export { arrayCtor, broadcastStrides, contiguousStrides, convertData, normalizeDim, prod, shapesEqual, showShape }
