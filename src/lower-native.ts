import { type SerializedNode, serializeNode } from "./ir.ts"
import { NODE_OPS, type NodeOp } from "./ops.ts"
import type { LazyNode } from "./storage.ts"
import type { AnyTensor } from "./tensor.ts"

type Support =
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

const noNativeKernel = (op: string): Support => ({
  kind: "unsupported",
  reason: `${op} has no native kernel on this runtime; the graph falls back to the JS interpreter`,
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

  gelu: noNativeKernel("gelu"),
  geluGrad: noNativeKernel("geluGrad"),
  silu: noNativeKernel("silu"),
  siluGrad: noNativeKernel("siluGrad"),
  softmax: noNativeKernel("softmax"),
  softmaxGrad: noNativeKernel("softmaxGrad"),
  layerNorm: noNativeKernel("layerNorm"),
  layerNormGrad: noNativeKernel("layerNormGrad"),
  rmsNorm: noNativeKernel("rmsNorm"),
  rmsNormGrad: noNativeKernel("rmsNormGrad"),
  crossEntropy: noNativeKernel("crossEntropy"),
  logSumExp: noNativeKernel("logSumExp"),
  gatherRows: noNativeKernel("gatherRows"),
  scatterAddRows: noNativeKernel("scatterAddRows"),
  dropout: noNativeKernel("dropout"),
  pick: noNativeKernel("pick"),
}

export const NON_WIRE_OPS: readonly NodeOp[] = NODE_OPS.filter(
  op => !WIRE_SET.has(op),
)

export function supportOf(node: LazyNode): Support {
  return SUPPORT[node.op]
}

/** A multi-axis reduce expands to an ascending chain matching evalReduceDimsEager's order. */
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
  // keepdim restores axis metadata only, so one trailing view is enough.
  if (!keepdim) return cursor
  return emit({
    op: "view",
    input: cursor,
    shape: [...node.shape],
  })
}
