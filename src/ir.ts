import {
  evalBinaryEager,
  evalBroadcastToEager,
  evalCatEager,
  evalIndexSelectEager,
  evalMatmulEager,
  evalNarrowEager,
  evalOneHotEager,
  evalPermuteEager,
  evalRandomEager,
  evalReduceAllEager,
  evalReduceEager,
  evalScatterAddEager,
  evalUnaryEager,
} from "./eager.ts"
import { nextStream } from "./kernels.ts"
import type { BinaryOp, ReduceOp, UnaryOp } from "./ops.ts"
import { broadcastShapes, broadcastToShape, catShape, matmulShape, permuteShape, reduceShape, resizeDim, type Shape } from "./shape.ts"
import { type DType, type LazyNode, type LazyNodeBody, normalizeDim, prod, promoteBinaryDtype, type RandomKind, shapesEqual, showShape } from "./storage.ts"
import { _internal, type AnyTensor, makeStorage, type Tensor } from "./tensor.ts"

// ---------------------------------------------------------------------------
// The IR: one node constructor, one description table, and the raw*
// dispatchers every typed method calls. A dispatcher computes the
// output shape, then either records a node (lazy) or runs the eager
// kernel — the kernels never see the mode flag, and the lazy evaluator
// replays them through the same table.
// ---------------------------------------------------------------------------

let lazyMode = false

export function isLazyMode(): boolean {
  return lazyMode
}

export function setLazyMode(value: boolean): void {
  lazyMode = value
}

/** The only way to build a lazy tensor: every node goes through here. */
export function makeNode(
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

type TensorField = "a" | "b" | "input" | "index"
type JsonField =
  | TensorField
  | "kind"
  | "parameter"
  | "dim"
  | "keepdim"
  | "order"
  | "start"
  | "length"
  | "classes"
  | "stream"
  | "shape"

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
    json: ["kind", "dim", "keepdim", "input", "shape"],
    printName: "dotted",
    printAttrs: [
      { key: "dim" },
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

/**
 * Post-order traversal of the lazy graph reachable from `roots`: every
 * tensor appears after all of its inputs, each exactly once. Leaves
 * (non-lazy or already-materialized tensors) are included, with no
 * inputs of their own.
 *
 * Iterative on purpose. A compiled training step for a cellular
 * automaton rolls the update rule out over dozens of time steps and
 * then differentiates it, which is a graph thousands of nodes deep —
 * far past what recursion survives.
 */
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
      // A materialized lazy tensor is a leaf: its value is frozen, so
      // nothing below it is walked (or re-randomized) again.
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

// ---------------------------------------------------------------------------
// Dispatchers.
// ---------------------------------------------------------------------------

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
  const d = normalizeDim(dim, a.shape.length)
  if (lazyMode) {
    return makeNode(
      { op: "reduce", kind: op, dim: d, keepdim, input: a },
      reduceShape(a.shape, d, keepdim),
      a.dtype,
    )
  }
  return evalReduceEager(a, d, keepdim, op)
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

/**
 * Uniform values in [0, 1), redrawn on every evaluation.
 *
 * Seeded by `configure({ seed })`, not `Math.random`.
 */
export function uniform<const Sh extends Shape>(
  shape: Sh,
): Tensor<Sh> {
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
): Tensor<Sh> {
  return rawRandom(
    "normal",
    shape,
    nextStream(),
    "float32",
  ) as any
}
