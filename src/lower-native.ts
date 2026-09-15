import { type SerializedNode, serializeNode } from "./ir.ts"
import { NODE_OPS, type NodeOp } from "./ops.ts"
import type { LazyNode } from "./storage.ts"
import type { AnyTensor } from "./tensor.ts"

/** What the native addon can run; an unsupported graph falls back to the JS interpreter. */
export type Support =
  | { kind: "native" }
  /** The whole graph falls back; `reason` is what the user is told. */
  | { kind: "unsupported"; reason: string }

/** The addon's own op set; tests cross-check this against the Rust enum in both directions. */
export const WIRE_OPS = [
  "leaf",
  "binary",
  "unary",
  "matmul",
  "reduce",
  "reduceAll",
  "broadcastTo",
  "permute",
  "view",
  "narrow",
  "cat",
  "oneHot",
  "indexSelect",
  "scatterAdd",
  "random",
] as const satisfies readonly NodeOp[]

const WIRE_SET: ReadonlySet<string> = new Set(WIRE_OPS)

const PHASE_A = (op: string): Support => ({
  kind: "unsupported",
  reason: `${op} has no native kernel on today's runtime (Phase A runs it on the JS interpreter; A-L1 lowers it, W4.2-W4.5 make it a kernel)`,
})

/** An unclassified new NODE_OPS member is a type error, not a test failure. */
export const SUPPORT: Record<NodeOp, Support> = {
  leaf: { kind: "native" },
  binary: { kind: "native" },
  unary: { kind: "native" },
  matmul: { kind: "native" },
  // The addon parses one axis per reduce; encodeForWire expands `dims`, so this stays native at every arity.
  reduce: { kind: "native" },
  reduceAll: { kind: "native" },
  broadcastTo: { kind: "native" },
  permute: { kind: "native" },
  view: { kind: "native" },
  narrow: { kind: "native" },
  cat: { kind: "native" },
  oneHot: { kind: "native" },
  indexSelect: { kind: "native" },
  scatterAdd: { kind: "native" },
  random: { kind: "native" },

  gelu: PHASE_A("gelu"),
  geluGrad: PHASE_A("geluGrad"),
  silu: PHASE_A("silu"),
  siluGrad: PHASE_A("siluGrad"),
  softmax: PHASE_A("softmax"),
  softmaxGrad: PHASE_A("softmaxGrad"),
  layerNorm: PHASE_A("layerNorm"),
  layerNormGrad: PHASE_A("layerNormGrad"),
  rmsNorm: PHASE_A("rmsNorm"),
  rmsNormGrad: PHASE_A("rmsNormGrad"),
  crossEntropy: PHASE_A("crossEntropy"),
  logSumExp: PHASE_A("logSumExp"),
  gatherRows: PHASE_A("gatherRows"),
  scatterAddRows: PHASE_A("scatterAddRows"),
  dropout: PHASE_A("dropout"),
  pick: PHASE_A("pick"),
  contiguous: PHASE_A("contiguous"),
}

/** Every node kind the wire cannot carry, for tests and diagnostics. */
export const NON_WIRE_OPS: readonly NodeOp[] = NODE_OPS.filter(
  op => !WIRE_SET.has(op),
)

export function supportOf(node: LazyNode): Support {
  return SUPPORT[node.op]
}

/** Wire encoding of a node: a multi-axis reduce expands to an ascending chain matching evalReduceDimsEager's order. */
export function encodeForWire(
  node: LazyNode,
  ref: (t: AnyTensor) => number,
  emit: (body: SerializedNode) => number,
): number {
  if (node.op !== "reduce") {
    return emit(serializeNode(node, ref))
  }
  const { dims, keepdim, kind, input } = node
  if (dims.length === 1) {
    return emit({
      op: "reduce",
      kind,
      dim: dims[0]!,
      keepdim,
      input: ref(input),
      shape: [...node.shape],
    })
  }
  let shape = [...input.shape]
  let cursor = ref(input)
  dims.forEach((d, i) => {
    // Each axis already removed shifts every later axis down by one.
    const axis = d - i
    shape = shape.filter((_, j) => j !== axis)
    cursor = emit({
      op: "reduce",
      kind,
      dim: axis,
      keepdim: false,
      input: cursor,
      shape: [...shape],
    })
  })
  // keepdim is shape metadata, never accumulation order, so restoring the axes as 1s is one view.
  if (!keepdim) return cursor
  return emit({
    op: "view",
    input: cursor,
    shape: [...node.shape],
  })
}
