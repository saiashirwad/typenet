import * as nativeBackend from "./backends/native.ts"
import {
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
  rawUnary,
  reshapeRaw,
} from "./eager.ts"
import { getActiveSeed, nextSeed, randomData, reseed, setActiveSeed } from "./kernels.ts"
import { type CpuStorage, type DType, type LazyNode, type LazyNodeBody, type LazyStorage, prod, showShape, type TensorStorage } from "./storage.ts"
import { _internal, type AnyTensor, makeRaw, makeStorage } from "./tensor.ts"

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
  if (options.seed !== undefined) reseed(options.seed)
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

function lazily<T>(fn: () => T): T {
  const prev = lazyMode
  lazyMode = true
  try {
    return fn()
  } finally {
    lazyMode = prev
  }
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

const OP_DESC: Record<LazyNode["op"], OpDesc> = {
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

function nodeInputs(node: LazyNode): AnyTensor[] {
  const rec = nodeFields(node)
  return OP_DESC[node.op].tensors.map(k => rec[k] as AnyTensor)
}

function formatLazyOp(
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

function serializeNode(
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

function makeLazy(
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

function evalNode(node: LazyNode): AnyTensor {
  switch (node.op) {
    case "binary":
      return rawBinary(
        force(node.a),
        force(node.b),
        node.kind,
        node.parameter,
      )
    case "unary":
      return rawUnary(
        force(node.input),
        node.kind,
        node.parameter,
      )
    case "matmul":
      return rawMatmul(force(node.a), force(node.b))
    case "reduce":
      return rawReduce(
        force(node.input),
        node.dim,
        node.keepdim,
        node.kind,
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
        node.length,
      )
    case "cat":
      return rawCat(force(node.a), force(node.b), node.dim)
    case "oneHot":
      return rawOneHot(force(node.input), node.classes)
    case "indexSelect":
      return rawIndexSelect(
        force(node.input),
        force(node.index),
        node.dim,
      )
    case "scatterAdd":
      return rawScatterAdd(
        force(node.input),
        force(node.index),
        node.dim,
        node.length,
      )
    case "random":
      return makeRaw(
        randomData(
          node.kind,
          prod(node.shape),
          node.stream,
          getActiveSeed(),
          node.dtype,
        ),
        node.shape,
        node.dtype,
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

function evalInterpreted(roots: AnyTensor[]): void {
  // One seed per evaluation: every random node in this pass draws from
  // it, and the next pass over the same graph draws different numbers.
  setActiveSeed(nextSeed())
  for (const t of topoOrder(roots)) {
    const source = _internal.sourceOf(t)
    if (source.kind !== "lazy" || _internal.hasValue(t)) {
      continue
    }
    const result = eagerly(() => evalNode(source.node))
    _internal.setCpu(t, result.data)
  }
}

function force(t: AnyTensor): AnyTensor {
  const source = _internal.sourceOf(t)
  if (source.kind !== "lazy" || _internal.hasValue(t)) {
    return t
  }
  if (!evalNativeMany([t])) {
    evalInterpreted([t])
  }
  return t
}

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
 * PLAN.md "History: measured numbers"): CPU matches Metal on chained
 * large matmuls, loses to it by
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
  let work = 0

  for (const t of order) {
    index.set(t, nodes.length)
    const source = _internal.sourceOf(t)
    if (source.kind !== "lazy" || _internal.hasValue(t)) {
      if (t.dtype !== "float32") {
        // Native is f32-on-CPU-leaves only. This used to fall back to
        // the JS interpreter silently; with native enabled that is a
        // performance surprise, so it is now an error.
        throw new Error(
          `native backend requires float32 CPU leaves; got a ${t.dtype} leaf. `
            + "Keep the graph in float32 or call disableNative().",
        )
      }
      const data = (_internal.cpuOf(t)
        ?? (source as CpuStorage).data) as Float32Array
      nodes.push({
        op: "leaf",
        leaf: leafTensors.length,
        offset: leafBytes,
        shape: [...t.shape],
      })
      leafData.push(data)
      leafTensors.push(t)
      leafOffsets.push(leafBytes)
      leafBytes += data.length
      work += data.length
      continue
    }
    const node = (source as LazyStorage).node
    work += prod(node.shape)
    // Every input precedes `t` in topological order, so its index is
    // already assigned.
    nodes.push(serializeNode(node, u => index.get(u)!))
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
      device: pickTarget(work),
    }),
    leaves,
    rootShapes,
    leafTensors,
    leafOffsets,
    leafBytes,
  }
}

function evalNativeMany(roots: AnyTensor[]): boolean {
  if (!nativeBackend.isNativeEnabled()) return false
  for (const t of roots) {
    if (
      _internal.sourceOf(t).kind !== "lazy"
      || _internal.hasValue(t)
    ) {
      return false
    }
  }
  const serialized = serializeLazyGraph(roots)
  if (!serialized) return false
  const data = nativeBackend.evalGraphNative(
    serialized.json,
    serialized.leaves,
    nextSeed(),
  )
  let offset = 0
  serialized.rootShapes.forEach((shape, i) => {
    const n = prod(shape)
    _internal.setCpu(
      roots[i]!,
      data.subarray(offset, offset + n),
    )
    offset += n
  })
  if (offset !== data.length) {
    throw new Error(
      `native backend returned ${data.length} values, expected ${offset} for roots [${serialized.rootShapes.map(showShape).join(", ")}]`,
    )
  }
  return true
}

export function forceMany(ts: AnyTensor[]): void {
  const pending = ts.filter(t =>
    _internal.sourceOf(t).kind === "lazy"
    && !_internal.hasValue(t)
  )
  if (pending.length > 0 && !evalNativeMany(pending)) {
    evalInterpreted(pending)
  }
  for (const t of ts) force(t)
}

export { eagerly, force, formatLazyOp, lazily, lazyMode, makeLazy, serializeLazyGraph, topoOrder }
export type { CpuStorage, LazyStorage, TensorStorage }
