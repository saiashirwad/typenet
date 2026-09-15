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
  /** Reseeds every RNG path (rand/randn, Tensor.rand/r.randn); same seed, same replay. */
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

/** Replays one IR node through the eager kernels, forcing inputs first. */
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
    // One seed per pass: forward, backward, and any replay share it.
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
  // One seed per evaluation pass; a replay of the same graph draws different numbers.
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

// Graphs up to this many elements run on the native fused loop evaluator;
// above the cap, per-element dispatch makes candle the faster default.
const LOOP_EVALUATOR_MAX_WORK = 65536

/** Native target for a graph of `work` elements: loop evaluator below the cap, else the configured device. */
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
  // Integer leaves may only feed gather/scatter index slots.
  const intLeaves = new Set<number>()
  let leafBytes = 0
  let work = 0

  for (const t of order) {
    const source = _internal.sourceOf(t)
    if (source.kind !== "lazy" || _internal.hasValue(t)) {
      if (t.dtype === "float64") {
        // Native compute is f32-only.
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
      // Capability gaps fall back to the interpreter via `null` and are counted; user errors below throw.
      noteFallback(node.op, support.reason)
      return null
    }
    work += prod(node.shape)
    // Inputs precede `t` topologically, so their indices exist; `encodeForWire` may
    // append several wire nodes, and the index it returns is the one consumers reference.
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

  // A compute op reading an integer leaf would error in candle or silently convert, so reject it up front.
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

// Prepare-once cache keyed by the FIRST root tensor (a WeakMap cannot key a fresh root array);
// a hit requires the same root list by identity, otherwise the plan is replaced.
const forcePlans = new WeakMap<AnyTensor, ForcePlan>()

// Frees the native handle when the keyed root is collected (PLAN_HANDLES is the explicit-release map).
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
      // Free the old handle now; unregister so collection of the old plan cannot double-free.
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
