import * as nativeBackend from "./backends/native.ts"
import { noteFallback } from "./counters.ts"
import {
  evalBinaryEager,
  evalBroadcastToEager,
  evalCatEager,
  evalContiguousEager,
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
import { isLazyMode, nodeInputs, OP_DESC, type SerializedNode, setLazyMode, topoOrder } from "./ir.ts"
import { getActiveSeed, nextSeed, reseed, setActiveSeed } from "./kernels.ts"
import { encodeForWire, supportOf } from "./lower-native.ts"
import { type CpuStorage, type LazyNode, type LazyStorage, prod, showShape } from "./storage.ts"
import { _internal, type AnyTensor } from "./tensor.ts"

export function configure(options: {
  lazy?: boolean
  /**
   * Reseeds `rand()` / `randn()` (both `resample` modes) and
   * `Tensor.rand` / `Tensor.randn` — all draw through the seeded
   * generator; there is no `Math.random` in the RNG path. A run
   * replays identically given the same seed and the same sequence of
   * operations.
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
      return evalReduceDimsEager(
        force(node.input),
        node.dims,
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

    // --- W4.1 semantic ops ---------------------------------------------
    case "gelu":
      return evalGeluEager(force(node.input))
    case "geluGrad":
      return evalGeluGradEager(
        force(node.grad),
        force(node.input),
      )
    case "silu":
      return evalSiluEager(force(node.input))
    case "siluGrad":
      return evalSiluGradEager(
        force(node.grad),
        force(node.input),
      )
    case "softmax":
      return evalSoftmaxEager(
        force(node.input),
        node.dim,
        node.causal,
      )
    case "softmaxGrad":
      return evalSoftmaxGradEager(
        force(node.grad),
        force(node.input),
        node.dim,
      )
    case "layerNorm":
      return evalLayerNormEager(
        force(node.input),
        force(node.gamma),
        force(node.beta),
        node.eps,
      )
    case "layerNormGrad":
      return evalLayerNormGradEager(
        force(node.grad),
        force(node.input),
        force(node.gamma),
        force(node.mean),
        force(node.rstd),
      )
    case "rmsNorm":
      return evalRmsNormEager(
        force(node.input),
        force(node.gamma),
        node.eps,
      )
    case "rmsNormGrad":
      return evalRmsNormGradEager(
        force(node.grad),
        force(node.input),
        force(node.gamma),
        force(node.rstd),
      )
    case "crossEntropy":
      return evalCrossEntropyEager(
        force(node.input),
        force(node.target),
      )
    case "logSumExp":
      return evalLogSumExpEager(
        force(node.input),
        node.dim,
        node.keepdim,
      )
    case "gatherRows":
      return evalGatherRowsEager(
        force(node.input),
        force(node.index),
      )
    case "scatterAddRows":
      return evalScatterAddRowsEager(
        force(node.input),
        force(node.index),
        node.rows,
      )
    // One seed per evaluation pass (set by `evalInterpreted` below), so a
    // replay of the same graph redraws the mask and the forward and its
    // backward — which run in the SAME pass — share it.
    case "dropout":
      return evalDropoutEager(
        force(node.input),
        node.p,
        node.stream,
        getActiveSeed(),
      )
    case "pick":
      return evalPickEager(
        force(node.input),
        node.offset,
        node.shape,
      )
    case "contiguous":
      return evalContiguousEager(force(node.input))
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
// strided views, buffer drop, and a single-pass scatter: on a rolled-out
// graph training step the loop evaluator ran at about a third of candle
// CPU's rate — the loss is per-element dispatch in the fused passes, not
// copies — so candle stays the default above the cap.
const LOOP_EVALUATOR_MAX_WORK = 65536

/**
 * Which native evaluator a graph of `work` elements should run on.
 *
 * Tiny graphs go to the loop evaluator. Everything else goes to candle
 * on the CPU device, which on macOS means Accelerate for matmul. That is
 * not the obvious default, so the numbers behind it (Apple M5,
 * measured 2026-08): CPU matches Metal on chained
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
  leaves: Uint8Array
  rootShapes: number[][]
  leafTensors: AnyTensor[]
  leafOffsets: number[]
  leafBytes: number
} | null {
  const order = topoOrder(roots)
  const nodes: SerializedNode[] = []
  const index = new Map<AnyTensor, number>()
  const leafChunks: Uint8Array[] = []
  const leafTensors: AnyTensor[] = []
  const leafOffsets: number[] = []
  // Integer leaves may only feed gather/scatter `index` slots; compute
  // leaves stay f32. Kept as node indices for the post-pass below.
  const intLeaves = new Set<number>()
  let leafBytes = 0
  let work = 0

  for (const t of order) {
    const source = _internal.sourceOf(t)
    if (source.kind !== "lazy" || _internal.hasValue(t)) {
      if (t.dtype === "float64") {
        // Native compute is f32-only. This used to fall back to the JS
        // interpreter silently; with native enabled that is a
        // performance surprise, so it is now an error.
        throw new Error(
          `native backend requires float32 CPU leaves; got a ${t.dtype} leaf. `
            + "Keep the graph in float32 or call disableNative().",
        )
      }
      const data = _internal.cpuOf(t)
        ?? (source as CpuStorage).data
      const bytes = new Uint8Array(
        data.buffer,
        data.byteOffset,
        data.byteLength,
      )
      index.set(t, nodes.length)
      nodes.push({
        op: "leaf",
        leaf: leafTensors.length,
        offset: leafBytes,
        shape: [...t.shape],
        dtype: t.dtype,
      })
      if (t.dtype !== "float32") {
        intLeaves.add(nodes.length - 1)
      }
      leafChunks.push(bytes)
      leafTensors.push(t)
      leafOffsets.push(leafBytes)
      leafBytes += bytes.length
      work += data.length
      continue
    }
    const node = (source as LazyStorage).node
    const support = supportOf(node)
    if (support.kind === "unsupported") {
      // Loudly, once per reason, and countably: the whole graph goes to
      // the JS interpreter through the `null` route `compile()` and
      // `forceMany` already take. This is a CAPABILITY gap, so it is a
      // notice; the two errors below it are USER mistakes, and they stay
      // errors.
      noteFallback(node.op, support.reason)
      return null
    }
    work += prod(node.shape)
    // Every input precedes `t` in topological order, so its index is
    // already assigned. `encodeForWire` may append more than one wire
    // node (a multi-axis `reduce` is a chain the addon can parse); the
    // index it returns is the one this tensor's consumers reference.
    index.set(
      t,
      encodeForWire(
        node,
        u => index.get(u)!,
        body => {
          nodes.push(body)
          return nodes.length - 1
        },
      ),
    )
  }

  // Integer leaves are gather/scatter indices only. A compute op that
  // reads one would either error in candle or silently convert through
  // the loop evaluator, so reject it up front.
  for (const node of nodes) {
    if (node.op === "leaf") continue
    const op = node.op as keyof typeof OP_DESC
    for (const slot of OP_DESC[op].tensors) {
      if (
        (op === "indexSelect" || op === "scatterAdd")
        && slot === "index"
      ) {
        continue
      }
      const ref = node[slot] as number
      if (intLeaves.has(ref)) {
        throw new Error(
          `native backend: an integer leaf must feed a gather/scatter index, but one feeds ${op}`,
        )
      }
    }
  }

  const rootIndices = roots.map(root => index.get(root)!)
  const rootShapes = roots.map(root => [...root.shape])
  const leaves = new Uint8Array(leafBytes)
  let offset = 0
  for (const chunk of leafChunks) {
    leaves.set(chunk, offset)
    offset += chunk.length
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
  const dirty = new Uint8Array(plan!.leafBytes)
  plan!.leafTensors.forEach((t, i) => {
    const data = _internal.cpuOf(t)!
    dirty.set(
      new Uint8Array(
        data.buffer,
        data.byteOffset,
        data.byteLength,
      ),
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

export { eagerly, force, serializeLazyGraph }
