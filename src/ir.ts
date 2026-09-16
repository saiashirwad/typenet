import {
  evalBinaryEager,
  evalBroadcastToEager,
  evalCatEager,
  evalCrossEntropyEager,
  evalDropoutEager,
  evalGatherRowsEager,
  evalGeluEager,
  evalGeluGradEager,
  evalIndexSelectEager,
  evalLayerNormEager,
  evalLayerNormGradEager,
  evalLogSumExpEager,
  evalMatmulEager,
  evalNarrowEager,
  evalOneHotEager,
  evalPermuteEager,
  evalPickEager,
  evalRandomEager,
  evalReduceAllEager,
  evalReduceDimsEager,
  evalReduceEager,
  evalRmsNormEager,
  evalRmsNormGradEager,
  evalScatterAddEager,
  evalScatterAddRowsEager,
  evalSiluEager,
  evalSiluGradEager,
  evalSoftmaxEager,
  evalSoftmaxGradEager,
  evalUnaryEager,
} from "./eager.ts"
import { nextSeed, nextStream } from "./kernels.ts"
import type { BinaryOp, ReduceOp, UnaryOp } from "./ops.ts"
import { broadcastShapes, broadcastToShape, catShape, matmulShape, permuteShape, reduceShape, resizeDim } from "./shape.ts"
import {
  type DType,
  type LazyNode,
  type LazyNodeBody,
  normalizeDim,
  prod,
  promoteBinaryDtype,
  type RandomKind,
  shapesEqual,
  showShape,
} from "./storage.ts"
import { _internal, type AnyTensor, makeStorage, Tensor } from "./tensor.ts"

let lazyMode = false

export function isLazyMode(): boolean {
  return lazyMode
}

export function setLazyMode(value: boolean): void {
  lazyMode = value
}

function makeNode(
  body: LazyNodeBody,
  shape: readonly number[],
  dtype: DType,
): AnyTensor {
  const node = {
    ...body,
    shape: [...shape],
    dtype,
  } as LazyNode
  return makeStorage(
    { kind: "lazy", node },
    shape,
    dtype,
  )
}

type TensorField =
  | "a"
  | "b"
  | "input"
  | "index"
  | "grad"
  | "gamma"
  | "beta"
  | "mean"
  | "rstd"
  | "target"
type JsonField =
  | TensorField
  | "kind"
  | "parameter"
  | "dim"
  | "dims"
  | "keepdim"
  | "order"
  | "start"
  | "length"
  | "classes"
  | "stream"
  | "shape"
  | "eps"
  | "causal"
  | "p"
  | "rows"
  | "out"
  | "offset"

type OpDesc = {
  tensors: readonly TensorField[]
  json: readonly JsonField[]
  printName: "op" | "kind" | "dotted"
  printAttrs: readonly {
    key: Exclude<JsonField, TensorField | "shape">
    skipIf?: unknown
    format?: "list"
  }[]
}

export const OP_DESC: Record<LazyNode["op"], OpDesc> = {
  binary: {
    tensors: ["a", "b"],
    json: ["kind", "parameter", "a", "b", "shape"],
    printName: "kind",
    printAttrs: [{ key: "parameter", skipIf: 0 }],
  },
  unary: {
    tensors: ["input"],
    json: ["kind", "parameter", "input", "shape"],
    printName: "kind",
    printAttrs: [{ key: "parameter", skipIf: 0 }],
  },
  matmul: {
    tensors: ["a", "b"],
    json: ["a", "b", "shape"],
    printName: "op",
    printAttrs: [],
  },
  reduce: {
    tensors: ["input"],
    json: ["kind", "dims", "keepdim", "input", "shape"],
    printName: "dotted",
    printAttrs: [
      { key: "dims", format: "list" },
      { key: "keepdim", skipIf: false },
    ],
  },
  reduceAll: {
    tensors: ["input"],
    json: ["kind", "input", "shape"],
    printName: "dotted",
    printAttrs: [],
  },
  broadcastTo: {
    tensors: ["input"],
    json: ["input", "shape"],
    printName: "op",
    printAttrs: [],
  },
  permute: {
    tensors: ["input"],
    json: ["order", "input", "shape"],
    printName: "op",
    printAttrs: [{ key: "order", format: "list" }],
  },
  view: {
    tensors: ["input"],
    json: ["input", "shape"],
    printName: "op",
    printAttrs: [],
  },
  narrow: {
    tensors: ["input"],
    json: ["dim", "start", "length", "input", "shape"],
    printName: "op",
    printAttrs: [
      { key: "dim" },
      { key: "start" },
      { key: "length" },
    ],
  },
  cat: {
    tensors: ["a", "b"],
    json: ["a", "b", "dim", "shape"],
    printName: "op",
    printAttrs: [{ key: "dim" }],
  },
  oneHot: {
    tensors: ["input"],
    json: ["classes", "input", "shape"],
    printName: "op",
    printAttrs: [{ key: "classes" }],
  },
  indexSelect: {
    tensors: ["input", "index"],
    json: ["dim", "input", "index", "shape"],
    printName: "op",
    printAttrs: [{ key: "dim" }],
  },
  scatterAdd: {
    tensors: ["input", "index"],
    json: ["dim", "length", "input", "index", "shape"],
    printName: "op",
    printAttrs: [{ key: "dim" }, { key: "length" }],
  },
  random: {
    tensors: [],
    json: ["kind", "stream", "shape"],
    printName: "dotted",
    printAttrs: [{ key: "stream" }],
  },

  gelu: {
    tensors: ["input"],
    json: ["input", "shape"],
    printName: "op",
    printAttrs: [],
  },
  geluGrad: {
    tensors: ["grad", "input"],
    json: ["grad", "input", "shape"],
    printName: "op",
    printAttrs: [],
  },
  silu: {
    tensors: ["input"],
    json: ["input", "shape"],
    printName: "op",
    printAttrs: [],
  },
  siluGrad: {
    tensors: ["grad", "input"],
    json: ["grad", "input", "shape"],
    printName: "op",
    printAttrs: [],
  },
  softmax: {
    tensors: ["input"],
    json: ["dim", "causal", "input", "shape"],
    printName: "op",
    printAttrs: [
      { key: "dim" },
      { key: "causal", skipIf: false },
    ],
  },
  softmaxGrad: {
    tensors: ["grad", "input"],
    json: ["dim", "grad", "input", "shape"],
    printName: "op",
    printAttrs: [{ key: "dim" }],
  },
  layerNorm: {
    tensors: ["input", "gamma", "beta"],
    json: ["eps", "input", "gamma", "beta", "shape"],
    printName: "op",
    printAttrs: [{ key: "eps" }],
  },
  layerNormGrad: {
    tensors: ["grad", "input", "gamma", "mean", "rstd"],
    json: ["grad", "input", "gamma", "mean", "rstd", "shape"],
    printName: "op",
    printAttrs: [],
  },
  rmsNorm: {
    tensors: ["input", "gamma"],
    json: ["eps", "input", "gamma", "shape"],
    printName: "op",
    printAttrs: [{ key: "eps" }],
  },
  rmsNormGrad: {
    tensors: ["grad", "input", "gamma", "rstd"],
    json: ["grad", "input", "gamma", "rstd", "shape"],
    printName: "op",
    printAttrs: [],
  },
  crossEntropy: {
    tensors: ["input", "target"],
    json: ["input", "target", "shape"],
    printName: "op",
    printAttrs: [],
  },
  logSumExp: {
    tensors: ["input"],
    json: ["dim", "keepdim", "input", "shape"],
    printName: "op",
    printAttrs: [
      { key: "dim" },
      { key: "keepdim", skipIf: false },
    ],
  },
  gatherRows: {
    tensors: ["input", "index"],
    json: ["input", "index", "shape"],
    printName: "op",
    printAttrs: [],
  },
  scatterAddRows: {
    tensors: ["input", "index"],
    json: ["rows", "input", "index", "shape"],
    printName: "op",
    printAttrs: [{ key: "rows" }],
  },
  dropout: {
    tensors: ["input"],
    json: ["p", "stream", "input", "shape"],
    printName: "op",
    printAttrs: [{ key: "p" }, { key: "stream" }],
  },
  pick: {
    tensors: ["input"],
    json: ["out", "offset", "input", "shape"],
    printName: "op",
    printAttrs: [{ key: "out" }],
  },
}

function nodeFields(
  node: LazyNode,
): LazyNode & Record<string, unknown> {
  return node as LazyNode & Record<string, unknown>
}

export function nodeInputs(node: LazyNode): AnyTensor[] {
  const rec = nodeFields(node)
  return OP_DESC[node.op].tensors.map(k => rec[k] as AnyTensor)
}

export function formatLazyOp(
  node: LazyNode,
  arg: (t: AnyTensor) => string,
): string {
  const desc = OP_DESC[node.op]
  const rec = nodeFields(node)
  const name = desc.printName === "kind"
    ? String(rec.kind)
    : desc.printName === "dotted"
    ? `${node.op}.${rec.kind}`
    : node.op
  const args = desc.tensors
    .map(k => arg(rec[k] as AnyTensor))
    .join(", ")
  const pairs: [string, unknown][] = []
  for (const attr of desc.printAttrs) {
    const value = rec[attr.key]
    if (attr.skipIf !== undefined && value === attr.skipIf) {
      continue
    }
    const shown = attr.format === "list"
      ? `[${(value as number[]).join(", ")}]`
      : value
    pairs.push([attr.key, shown])
  }
  const extra = pairs.length === 0
    ? ""
    : ` {${pairs.map(([k, v]) => `${k}=${v}`).join(", ")}}`
  return `${name}(${args})${extra}`
}

export type SerializedNode = Record<string, unknown> & {
  op: string
}

export function serializeNode(
  node: LazyNode,
  ref: (t: AnyTensor) => number,
): SerializedNode {
  const desc = OP_DESC[node.op]
  const rec = nodeFields(node)
  const out: SerializedNode = { op: node.op }
  for (const field of desc.json) {
    if (field === "shape") {
      out.shape = [...node.shape]
    } else if ((desc.tensors as readonly string[]).includes(field)) {
      out[field] = ref(rec[field] as AnyTensor)
    } else {
      out[field] = rec[field]
    }
  }
  return out
}

/** Iterative because real graphs run thousands of nodes deep. */
export function topoOrder(
  roots: readonly AnyTensor[],
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
    const source = _internal.sourceOf(t)
    stack.push({
      t,
      // A materialized lazy tensor is a leaf: its value is frozen.
      inputs: source.kind === "lazy" && !_internal.hasValue(t)
        ? nodeInputs(source.node)
        : [],
      i: 0,
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

export function rawBinary(
  a: AnyTensor,
  b: AnyTensor,
  op: BinaryOp,
  parameter = 0,
): AnyTensor {
  if (lazyMode) {
    const outShape = broadcastShapes(a.shape, b.shape)
    const dtype: DType = promoteBinaryDtype(a.dtype, b.dtype)
    return makeNode(
      { op: "binary", kind: op, parameter, a, b },
      outShape,
      dtype,
    )
  }
  return evalBinaryEager(a, b, op, parameter)
}

export function rawUnary(
  a: AnyTensor,
  op: UnaryOp,
  parameter = 0,
): AnyTensor {
  if (lazyMode) {
    return makeNode(
      { op: "unary", kind: op, parameter, input: a },
      a.shape,
      a.dtype,
    )
  }
  return evalUnaryEager(a, op, parameter)
}

export function rawSum(
  a: AnyTensor,
  dim: number,
  keepdim: boolean,
): AnyTensor {
  return rawReduce(a, dim, keepdim, "sum")
}

export function rawReduce(
  a: AnyTensor,
  dim: number,
  keepdim: boolean,
  op: ReduceOp,
): AnyTensor {
  return rawReduceDims(a, [dim], keepdim, op)
}

function reduceDimsShape(
  shape: readonly number[],
  dims: readonly number[],
  keepdim: boolean,
): number[] {
  return keepdim
    ? shape.map((s, i) => (dims.includes(i) ? 1 : s))
    : shape.filter((_, i) => !dims.includes(i))
}

/** Normalised, de-duplicated and ascending, so structurally identical graphs serialise identically. */
function rawReduceDims(
  a: AnyTensor,
  dims: readonly number[],
  keepdim: boolean,
  op: ReduceOp,
): AnyTensor {
  const ds = [
    ...new Set(dims.map(d => normalizeDim(d, a.shape.length))),
  ].sort((x, y) => x - y)
  if (ds.length > 1 && op === "argmax") {
    throw new Error(
      `reduce.argmax over ${ds.length} axes is not defined; argmax reduces exactly one axis of ${showShape(a.shape)}`,
    )
  }
  if (lazyMode) {
    return makeNode(
      { op: "reduce", kind: op, dims: ds, keepdim, input: a },
      reduceDimsShape(a.shape, ds, keepdim),
      a.dtype,
    )
  }
  return ds.length === 1
    ? evalReduceEager(a, ds[0]!, keepdim, op)
    : evalReduceDimsEager(a, ds, keepdim, op)
}

export function rawReduceAll(
  a: AnyTensor,
  op: "sum" | "max",
): AnyTensor {
  if (lazyMode) {
    return makeNode(
      { op: "reduceAll", kind: op, input: a },
      [],
      a.dtype,
    )
  }
  return evalReduceAllEager(a, op)
}

export function rawBroadcastTo(
  a: AnyTensor,
  shape: number[],
): AnyTensor {
  if (shapesEqual(a.shape, shape)) return a
  broadcastToShape(a.shape, shape)
  if (lazyMode) {
    return makeNode(
      { op: "broadcastTo", input: a },
      shape,
      a.dtype,
    )
  }
  return evalBroadcastToEager(a, shape)
}

export function rawPermute(
  a: AnyTensor,
  order: number[],
): AnyTensor {
  if (lazyMode) {
    return makeNode(
      { op: "permute", order, input: a },
      permuteShape(a.shape, order),
      a.dtype,
    )
  }
  return evalPermuteEager(a, order)
}

export function rawMatmul(a: AnyTensor, b: AnyTensor): AnyTensor {
  if (lazyMode) {
    const dtype: DType = promoteBinaryDtype(a.dtype, b.dtype)
    return makeNode(
      { op: "matmul", a, b },
      matmulShape(a.shape, b.shape),
      dtype,
    )
  }
  return evalMatmulEager(a, b)
}

export function rawNarrow(
  a: AnyTensor,
  dim: number,
  start: number,
  length: number,
): AnyTensor {
  const d = normalizeDim(dim, a.shape.length)
  if (lazyMode) {
    return makeNode(
      { op: "narrow", dim: d, start, length, input: a },
      resizeDim(a.shape, d, length),
      a.dtype,
    )
  }
  return evalNarrowEager(a, d, start, length)
}

/**
 * Stacks `tensors` along a new axis at `dim`, as a balanced tree of `cat` nodes over `unsqueeze`.
 *
 * `Tensor.stack` folds a left-to-right `cat`, which copies the accumulated tensor once per input
 * and so is quadratic in the number of inputs. A loop that stacks one output per timestep hits
 * that directly. Pairing the inputs up instead keeps the copy count linear and the graph shallow,
 * and it is built only from `unsqueeze` and `cat`, so it lowers to the native backend exactly as
 * those two do.
 */
export function rawStackList(
  tensors: readonly AnyTensor[],
  dim: number,
): AnyTensor {
  if (tensors.length === 0) {
    throw new Error("stack: needs at least one tensor")
  }
  const first = tensors[0]!
  for (const t of tensors) {
    if (!shapesEqual(t.shape, first.shape)) {
      throw new Error(
        `stack: all tensors must share a shape (${showShape(first.shape)} vs ${showShape(t.shape)})`,
      )
    }
  }
  const d = normalizeDim(dim, first.shape.length + 1)
  // Every input gains the new axis at `d`, which is what turns a stack into a concatenation.
  let level = tensors.map(t => t.unsqueeze(d))
  while (level.length > 1) {
    const next: AnyTensor[] = []
    for (let i = 0; i < level.length; i += 2) {
      const a = level[i]!
      const b = level[i + 1]
      // `Tensor.cat` on the class, not the raw form: the raw form omits the tape entry.
      next.push(b === undefined ? a : (Tensor.cat as (x: AnyTensor, y: AnyTensor, dim: number) => AnyTensor)(a, b, d))
    }
    level = next
  }
  return level[0]!
}

export function rawCat(
  a: AnyTensor,
  b: AnyTensor,
  dim: number,
): AnyTensor {
  if (lazyMode) {
    const dtype: DType = promoteBinaryDtype(a.dtype, b.dtype)
    return makeNode(
      { op: "cat", a, b, dim },
      catShape(a.shape, b.shape, dim),
      dtype,
    )
  }
  return evalCatEager(a, b, dim)
}

export function rawOneHot(
  a: AnyTensor,
  classes: number,
): AnyTensor {
  if (lazyMode) {
    return makeNode(
      { op: "oneHot", classes, input: a },
      [a.numel, classes],
      a.dtype,
    )
  }
  return evalOneHotEager(a, classes)
}

export function rawIndexSelect(
  a: AnyTensor,
  index: AnyTensor,
  dim: number,
): AnyTensor {
  if (lazyMode) {
    return makeNode(
      { op: "indexSelect", dim, input: a, index },
      resizeDim(a.shape, dim, index.numel),
      a.dtype,
    )
  }
  return evalIndexSelectEager(a, index, dim)
}

export function rawScatterAdd(
  a: AnyTensor,
  index: AnyTensor,
  dim: number,
  length: number,
): AnyTensor {
  if (lazyMode) {
    return makeNode(
      { op: "scatterAdd", dim, length, input: a, index },
      resizeDim(a.shape, dim, length),
      a.dtype,
    )
  }
  return evalScatterAddEager(a, index, dim, length)
}

export function rawRandom(
  kind: RandomKind,
  shape: readonly number[],
  stream: number,
  dtype: DType,
): AnyTensor {
  if (lazyMode) {
    return makeNode(
      { op: "random", kind, stream },
      shape,
      dtype,
    )
  }
  return evalRandomEager(kind, shape, stream, dtype)
}

export function reshapeRaw(
  t: AnyTensor,
  shape: number[],
): AnyTensor {
  if (t.numel !== prod(shape)) {
    throw new Error(
      `Cannot reshape ${showShape(t.shape)} to ${showShape(shape)}`,
    )
  }
  if (lazyMode) {
    return makeNode(
      { op: "view", input: t },
      shape,
      t.dtype,
    )
  }
  return _internal.makeView(t, shape)
}

export function sumTo(t: AnyTensor, shape: number[]): AnyTensor {
  if (shapesEqual(t.shape, shape)) return t
  const offset = t.shape.length - shape.length
  const dims: number[] = []
  for (let i = 0; i < offset; i++) dims.push(i)
  for (let i = 0; i < shape.length; i++) {
    if (shape[i] === 1 && t.shape[offset + i] !== 1) {
      dims.push(offset + i)
    }
  }
  if (dims.length === 0) return t
  if (shapesEqual(reduceDimsShape(t.shape, dims, false), shape)) {
    return rawReduceDims(t, dims, false, "sum")
  }
  const reduced = rawReduceDims(t, dims, true, "sum")
  return shapesEqual(reduced.shape, shape)
    ? reduced
    : reshapeRaw(reduced, shape)
}

// Multi-output ops are one flat producer plus a slice per output; lazy mode turns
// the slices into `pick` nodes, eager mode reads the kernel's buffer directly.

function splitFlat(
  flat: AnyTensor,
  outShapes: readonly (readonly number[])[],
): AnyTensor[] {
  let offset = 0
  return outShapes.map(shape => {
    const out = evalPickEager(flat, offset, shape)
    offset += prod(shape)
    return out
  })
}

function makeMultiNode(
  body: LazyNodeBody,
  outShapes: readonly (readonly number[])[],
  dtype: DType,
): AnyTensor[] {
  const total = outShapes.reduce((n, s) => n + prod(s), 0)
  const producer = makeNode(body, [total], dtype)
  let offset = 0
  return outShapes.map((shape, out) => {
    const pick = makeNode(
      { op: "pick", out, offset, input: producer },
      shape,
      dtype,
    )
    offset += prod(shape)
    return pick
  })
}

function requireFloat(t: AnyTensor, what: string): void {
  if (t.dtype !== "float32" && t.dtype !== "float64") {
    throw new Error(
      `${what}: expected a float operand, got ${t.dtype} of shape ${showShape(t.shape)}`,
    )
  }
}

export function rawGelu(a: AnyTensor): AnyTensor {
  requireFloat(a, "gelu")
  if (lazyMode) {
    return makeNode({ op: "gelu", input: a }, a.shape, a.dtype)
  }
  return evalGeluEager(a)
}

export function rawGeluGrad(
  g: AnyTensor,
  a: AnyTensor,
): AnyTensor {
  if (lazyMode) {
    return makeNode(
      { op: "geluGrad", grad: g, input: a },
      a.shape,
      a.dtype,
    )
  }
  return evalGeluGradEager(g, a)
}

export function rawSilu(a: AnyTensor): AnyTensor {
  requireFloat(a, "silu")
  if (lazyMode) {
    return makeNode({ op: "silu", input: a }, a.shape, a.dtype)
  }
  return evalSiluEager(a)
}

export function rawSiluGrad(
  g: AnyTensor,
  a: AnyTensor,
): AnyTensor {
  if (lazyMode) {
    return makeNode(
      { op: "siluGrad", grad: g, input: a },
      a.shape,
      a.dtype,
    )
  }
  return evalSiluGradEager(g, a)
}

export function rawSoftmax(
  a: AnyTensor,
  dim: number,
  causal = false,
): AnyTensor {
  requireFloat(a, "softmax")
  const d = normalizeDim(dim, a.shape.length)
  if (causal) {
    // The mask is folded into the kernel, not materialised as a [T, T] buffer.
    if (a.shape.length < 2 || d !== a.shape.length - 1) {
      throw new Error(
        `softmax{causal}: needs the last axis of a rank>=2 score matrix; got dim ${d} of ${showShape(a.shape)}`,
      )
    }
  }
  if (lazyMode) {
    return makeNode(
      { op: "softmax", dim: d, causal, input: a },
      a.shape,
      a.dtype,
    )
  }
  return evalSoftmaxEager(a, d, causal)
}

export function rawSoftmaxGrad(
  g: AnyTensor,
  y: AnyTensor,
  dim: number,
): AnyTensor {
  const d = normalizeDim(dim, y.shape.length)
  if (lazyMode) {
    return makeNode(
      { op: "softmaxGrad", dim: d, grad: g, input: y },
      y.shape,
      y.dtype,
    )
  }
  return evalSoftmaxGradEager(g, y, d)
}

/** mean and rstd are saved for the backward rule. */
export function rawLayerNorm(
  x: AnyTensor,
  gamma: AnyTensor,
  beta: AnyTensor,
  eps: number,
): AnyTensor[] {
  requireFloat(x, "layerNorm")
  const width = normWidth(x, "layerNorm")
  checkNormWeight(gamma, width, "layerNorm", "gamma")
  checkNormWeight(beta, width, "layerNorm", "beta")
  const stats = x.shape.slice(0, -1)
  const outShapes = [x.shape, stats, stats]
  if (lazyMode) {
    return makeMultiNode(
      { op: "layerNorm", eps, input: x, gamma, beta },
      outShapes,
      x.dtype,
    )
  }
  return splitFlat(
    evalLayerNormEager(x, gamma, beta, eps),
    outShapes,
  )
}

export function rawLayerNormGrad(
  g: AnyTensor,
  x: AnyTensor,
  gamma: AnyTensor,
  mean: AnyTensor,
  rstd: AnyTensor,
): AnyTensor[] {
  const width = x.shape[x.shape.length - 1]!
  const outShapes = [x.shape, [width], [width]]
  if (lazyMode) {
    return makeMultiNode(
      { op: "layerNormGrad", grad: g, input: x, gamma, mean, rstd },
      outShapes,
      x.dtype,
    )
  }
  return splitFlat(
    evalLayerNormGradEager(g, x, gamma, mean, rstd),
    outShapes,
  )
}

export function rawRmsNorm(
  x: AnyTensor,
  gamma: AnyTensor,
  eps: number,
): AnyTensor[] {
  requireFloat(x, "rmsNorm")
  const width = normWidth(x, "rmsNorm")
  checkNormWeight(gamma, width, "rmsNorm", "gamma")
  const outShapes = [x.shape, x.shape.slice(0, -1)]
  if (lazyMode) {
    return makeMultiNode(
      { op: "rmsNorm", eps, input: x, gamma },
      outShapes,
      x.dtype,
    )
  }
  return splitFlat(evalRmsNormEager(x, gamma, eps), outShapes)
}

export function rawRmsNormGrad(
  g: AnyTensor,
  x: AnyTensor,
  gamma: AnyTensor,
  rstd: AnyTensor,
): AnyTensor[] {
  const outShapes = [
    x.shape,
    [x.shape[x.shape.length - 1]!],
  ]
  if (lazyMode) {
    return makeMultiNode(
      { op: "rmsNormGrad", grad: g, input: x, gamma, rstd },
      outShapes,
      x.dtype,
    )
  }
  return splitFlat(
    evalRmsNormGradEager(g, x, gamma, rstd),
    outShapes,
  )
}

function normWidth(x: AnyTensor, what: string): number {
  if (x.shape.length === 0) {
    throw new Error(
      `${what}: needs a rank>=1 input to normalise over its last axis, got []`,
    )
  }
  return x.shape[x.shape.length - 1]!
}

function checkNormWeight(
  w: AnyTensor,
  width: number,
  what: string,
  slot: string,
): void {
  if (w.shape.length !== 1 || w.shape[0] !== width) {
    throw new Error(
      `${what}: ${slot} must be ${showShape([width])} to match the normalised axis, got ${showShape(w.shape)}`,
    )
  }
}

export function rawCrossEntropy(
  logits: AnyTensor,
  target: AnyTensor,
): AnyTensor[] {
  requireFloat(logits, "crossEntropy")
  if (logits.shape.length !== 2) {
    throw new Error(
      `crossEntropy: logits must be [N, C], got ${showShape(logits.shape)}`,
    )
  }
  const n = logits.shape[0]!
  if (target.numel !== n) {
    throw new Error(
      `crossEntropy: ${target.numel} targets ${showShape(target.shape)} for ${n} rows of ${showShape(logits.shape)}`,
    )
  }
  const outShapes = [[] as number[], logits.shape]
  if (lazyMode) {
    return makeMultiNode(
      { op: "crossEntropy", input: logits, target },
      outShapes,
      logits.dtype,
    )
  }
  return splitFlat(
    evalCrossEntropyEager(logits, target),
    outShapes,
  )
}

export function rawLogSumExp(
  a: AnyTensor,
  dim: number,
  keepdim: boolean,
): AnyTensor {
  requireFloat(a, "logSumExp")
  const d = normalizeDim(dim, a.shape.length)
  if (lazyMode) {
    return makeNode(
      { op: "logSumExp", dim: d, keepdim, input: a },
      reduceShape(a.shape, d, keepdim),
      a.dtype,
    )
  }
  return evalLogSumExpEager(a, d, keepdim)
}

export function rawGatherRows(
  table: AnyTensor,
  index: AnyTensor,
): AnyTensor {
  if (table.shape.length === 0) {
    throw new Error(
      "gatherRows: the table must have a row axis, got []",
    )
  }
  const shape = [...index.shape, ...table.shape.slice(1)]
  if (lazyMode) {
    return makeNode(
      { op: "gatherRows", input: table, index },
      shape,
      table.dtype,
    )
  }
  return evalGatherRowsEager(table, index)
}

export function rawScatterAddRows(
  src: AnyTensor,
  index: AnyTensor,
  rows: number,
): AnyTensor {
  const rank = index.shape.length
  if (
    src.shape.length < rank
    || !shapesEqual(src.shape.slice(0, rank), index.shape)
  ) {
    throw new Error(
      `scatterAddRows: source ${showShape(src.shape)} must start with the index shape ${showShape(index.shape)}`,
    )
  }
  const shape = [rows, ...src.shape.slice(rank)]
  if (lazyMode) {
    return makeNode(
      { op: "scatterAddRows", rows, input: src, index },
      shape,
      src.dtype,
    )
  }
  return evalScatterAddRowsEager(src, index, rows)
}

/** The stream is fixed at build time so the graph structure stays stable; the per-evaluation seed resamples the mask. */
export function rawDropout(
  x: AnyTensor,
  p: number,
): AnyTensor[] {
  requireFloat(x, "dropout")
  if (!(p >= 0) || p >= 1) {
    throw new Error(
      `dropout: p must be in [0, 1), got ${p}`,
    )
  }
  const stream = nextStream()
  const outShapes = [x.shape, x.shape]
  if (lazyMode) {
    return makeMultiNode(
      { op: "dropout", p, stream, input: x },
      outShapes,
      x.dtype,
    )
  }
  // Eager draws its own seed so two calls never share a mask.
  return splitFlat(
    evalDropoutEager(x, p, stream, nextSeed()),
    outShapes,
  )
}
