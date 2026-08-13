import * as nativeBackend from "./backends/native.ts"
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
import { isLazyMode, nodeInputs, type SerializedNode, serializeNode, setLazyMode, topoOrder } from "./ir.ts"
import { nextSeed, reseed, setActiveSeed } from "./kernels.ts"
import { type CpuStorage, type LazyNode, type LazyStorage, prod, showShape } from "./storage.ts"
import { _internal, type AnyTensor } from "./tensor.ts"

export function configure(options: {
  lazy?: boolean
  /**
   * Reseeds `uniform()` / `normal()`. A run replays identically given
   * the same seed and the same sequence of operations. Does not affect
   * `Tensor.rand` / `Tensor.randn`, which use `Math.random`.
   */
  seed?: number
}): void {
  if (options.lazy !== undefined) setLazyMode(options.lazy)
  if (options.seed !== undefined) reseed(options.seed)
}

export function isLazy(): boolean {
  return isLazyMode()
}

function eagerly<T>(fn: () => T): T {
  const prev = isLazyMode()
  setLazyMode(false)
  try {
    return fn()
  } finally {
    setLazyMode(prev)
  }
}

function lazily<T>(fn: () => T): T {
  const prev = isLazyMode()
  setLazyMode(true)
  try {
    return fn()
  } finally {
    setLazyMode(prev)
  }
}

/**
 * One IR node, replayed through the eager kernels. Inputs are forced
 * first, so this never re-enters the dispatchers or flips the mode
 * flag — eager is the spec, and this is the interpreter reading it.
 */
function evalNode(node: LazyNode): AnyTensor {
  switch (node.op) {
    case "binary":
      return evalBinaryEager(
        force(node.a),
        force(node.b),
        node.kind,
        node.parameter,
      )
    case "unary":
      return evalUnaryEager(
        force(node.input),
        node.kind,
        node.parameter,
      )
    case "matmul":
      return evalMatmulEager(force(node.a), force(node.b))
    case "reduce":
      return evalReduceEager(
        force(node.input),
        node.dim,
        node.keepdim,
        node.kind,
      )
    case "reduceAll":
      return evalReduceAllEager(force(node.input), node.kind)
    case "broadcastTo":
      return evalBroadcastToEager(
        force(node.input),
        node.shape,
      )
    case "permute":
      return evalPermuteEager(force(node.input), node.order)
    case "view":
      return _internal.makeView(
        force(node.input),
        node.shape,
      )
    case "narrow":
      return evalNarrowEager(
        force(node.input),
        node.dim,
        node.start,
        node.length,
      )
    case "cat":
      return evalCatEager(
        force(node.a),
        force(node.b),
        node.dim,
      )
    case "oneHot":
      return evalOneHotEager(force(node.input), node.classes)
    case "indexSelect":
      return evalIndexSelectEager(
        force(node.input),
        force(node.index),
        node.dim,
      )
    case "scatterAdd":
      return evalScatterAddEager(
        force(node.input),
        force(node.index),
        node.dim,
        node.length,
      )
    case "random":
      return evalRandomEager(
        node.kind,
        node.shape,
        node.stream,
        node.dtype,
      )
  }
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
    const result = evalNode(source.node)
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

// Graphs touching at most this many elements (leaves + intermediate node
// outputs) go to the native fused loop evaluator, which pays no dispatch
// or BLAS setup cost. 65536 = one 256×256 matrix.
//
// The cutover was re-measured (2026-08) after the loop evaluator grew
// strided views, buffer drop, and a single-pass scatter: on the
// published GNCA training recipe (1024 nodes x 8, one [48, 80] bucket)
// loops run 0.56 steps/s against candle CPU's 1.50 — the loss is
// per-element dispatch in the fused passes, not copies — so candle
// stays the default above the cap and the race with torch.mps rides on
// a fused Metal step (PR17), not on lifting this constant.
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

type ForcePlan = {
  roots: AnyTensor[]
  handle: number
  leafTensors: AnyTensor[]
  leafOffsets: number[]
  leafBytes: number
  rootShapes: number[][]
}

// Prepare-once for the uncompiled path, keyed by the FIRST root tensor
// (a WeakMap cannot key on a fresh root array). A hit requires the same
// root list by identity; anything else re-serializes and replaces the
// plan. Every leaf on this path is an "input" — the tape has no
// placeholder/captured distinction — so all leaves are re-sent per
// eval; the win over evalGraph is skipping the JSON round-trip.
const forcePlans = new WeakMap<AnyTensor, ForcePlan>()

// When the first root is collected, the JS plan dies with it; without
// this, the native handle would leak (PLAN_HANDLES is a different,
// explicit-release map).
const planRegistry = new FinalizationRegistry<number>(
  handle => nativeBackend.releaseGraphNative(handle),
)

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
  const first = roots[0]!
  let plan = forcePlans.get(first)
  const hit = plan !== undefined
    && plan.roots.length === roots.length
    && plan.roots.every((r, i) => r === roots[i])
  if (!hit) {
    const serialized = serializeLazyGraph(roots)
    if (!serialized) return false
    const handle = nativeBackend.prepareGraphNative(
      serialized.json,
    )
    if (plan) {
      // Replace-on-miss: free the old handle now, and unregister so
      // collection of the old plan cannot double-free it.
      planRegistry.unregister(plan)
      nativeBackend.releaseGraphNative(plan.handle)
    }
    plan = {
      roots: [...roots],
      handle,
      leafTensors: serialized.leafTensors,
      leafOffsets: serialized.leafOffsets,
      leafBytes: serialized.leafBytes,
      rootShapes: serialized.rootShapes,
    }
    planRegistry.register(plan, handle, plan)
    forcePlans.set(first, plan)
  }
  const dirty = new Float32Array(plan!.leafBytes)
  plan!.leafTensors.forEach((t, i) => {
    dirty.set(
      _internal.cpuOf(t) as Float32Array,
      plan!.leafOffsets[i]!,
    )
  })
  const dirtyIndex = Uint32Array.from(
    plan!.leafTensors.keys(),
  )
  const data = nativeBackend.evalPreparedNative(
    plan!.handle,
    dirty,
    dirtyIndex,
    nextSeed(),
  )
  let offset = 0
  plan!.rootShapes.forEach((shape, i) => {
    const n = prod(shape)
    _internal.setCpu(
      roots[i]!,
      data.subarray(offset, offset + n),
    )
    offset += n
  })
  if (offset !== data.length) {
    throw new Error(
      `native backend returned ${data.length} values, expected ${offset} for roots [${plan!.rootShapes.map(showShape).join(", ")}]`,
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

export { eagerly, force, lazily, serializeLazyGraph }
