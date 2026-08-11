// Lazy graph machinery: build nodes, walk them, and materialize them
// either with the TS interpreter or in one native FFI hop. Also owns
// the lazy-mode flag and the configure()/eagerly()/lazily() dispatch
// every raw* kernel consults.
//
// evalNode calls back into the eager raw* kernels to materialize a
// node, while those same kernels call makeLazy here to record one —
// the eager↔lazy cycle. It resolves at runtime because every cross
// call sits inside a function body.

import * as nativeBackend from "./backends/native.ts"
import {
  prod,
  showShape,
  type CpuStorage,
  type DType,
  type LazyNode,
  type LazyNodeBody,
  type LazyStorage,
  type TensorStorage
} from "./storage.ts"
import {
  getActiveSeed,
  nextSeed,
  randomData,
  reseed,
  setActiveSeed
} from "./kernels.ts"
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
  reshapeRaw
} from "./eager.ts"
import {
  makeRaw,
  makeStorage,
  type AnyTensor
} from "./tensor.ts"

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
          getActiveSeed(),
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
  setActiveSeed(nextSeed())
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

export { force, eagerly, lazily, topoOrder, serializeLazyGraph, makeLazy, lazyMode }
// CpuStorage/LazyStorage are re-exported so compile.ts can narrow
// storages without a second import hop back through tensor.ts.
export type { CpuStorage, LazyStorage, TensorStorage }
