import { type SerializedNode, serializeNode } from "./ir.ts"
import { NODE_OPS, type NodeOp } from "./ops.ts"
import type { LazyNode } from "./storage.ts"
import type { AnyTensor } from "./tensor.ts"

/**
 * What today's native addon can and cannot run (PLAN-V2 §5A.2a, in the
 * showcase cut's §5A.9 shape: the FALLBACK half only).
 *
 * W4.1 adds seventeen semantic node kinds that `native/src/lib.rs` has never
 * heard of, and `native/` is frozen for the whole of Phase A. Handing the
 * addon a node it cannot parse would be a JSON error at prepare, or worse a
 * silent misparse — so `serializeLazyGraph` asks this table first and
 * returns `null` for a graph it cannot express, which is the route
 * `compile()` and the uncompiled lazy path already take to the JS
 * interpreter (`compile.ts`'s `runInterpreter`, `lazy.ts`'s
 * `evalInterpreted`). No control flow is added anywhere.
 *
 * A-L1 turns the `unsupported` rows into `lower` rows — a decomposition into
 * primitives the addon *does* run — without changing this file's shape or
 * its callers. Nothing here is a lowering today.
 */
export type Support =
  | { kind: "native" }
  /** The whole graph falls back; `reason` is what the user is told. */
  | { kind: "unsupported"; reason: string }

/**
 * The addon's own op set: the fifteen `Node` arms of `native/src/lib.rs`.
 *
 * `test/ops.test.ts` compares the Rust enum against THIS list rather than
 * against `NODE_OPS`, and separately requires every `NODE_OPS \ WIRE_OPS`
 * member to be classified below — so an op added on one side only still
 * fails the build, in both directions, exactly as it did before the op set
 * grew past what the addon parses.
 */
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

/**
 * Exhaustive by construction: a new `NODE_OPS` member that is not
 * classified here is a TYPE ERROR, not a test failure.
 */
export const SUPPORT: Record<NodeOp, Support> = {
  leaf: { kind: "native" },
  binary: { kind: "native" },
  unary: { kind: "native" },
  matmul: { kind: "native" },
  // `reduce` grew a `dims` array in W4.1 and the addon still speaks one
  // axis per node; `encodeForWire` below expands it, so the op stays
  // native at every arity.
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

/**
 * A node's form ON THE WIRE, which is not always one node.
 *
 * This is the one place where the IR and the addon's JSON disagree, and the
 * disagreement has exactly one cause: W4.1 step 6 replaced `reduce{dim}`
 * with `reduce{dims}` so `sumTo` emits one node instead of N, and the addon
 * — frozen since W0.8 — still parses a single `dim`. Expanding the array
 * back into the addon's own chain here keeps every broadcast backward rule
 * on the native path, which is what gate GA-e's `nativeFallbacks === 0`
 * clause is actually about.
 *
 * The chain is ascending-axis and drops each axis as it goes, which is
 * *exactly* the sequence of `reduce` nodes the pre-W4.1 `sumTo` emitted and
 * the same order `evalReduceDimsEager` accumulates in — so eager,
 * interpreted and native all sum in one order and the rewrite moves no f32
 * bits. A single-axis `reduce` still serialises to precisely today's node,
 * byte for byte.
 *
 * This is the wire ENCODING of one structural op, not a lowering of a
 * semantic one: no `unsupported` kind is decomposed here, and A-L1 inherits
 * the function unchanged.
 *
 * `emit` appends one wire node and returns its index; the value returned is
 * the index this IR node's consumers must reference.
 */
export function encodeForWire(
  node: LazyNode,
  ref: (t: AnyTensor) => number,
  emit: (body: SerializedNode) => number,
): number {
  if (node.op !== "reduce") {
    return emit(serializeNode(node, ref))
  }
  const { dims, keepdim, kind, input } = node
  // The overwhelmingly common case, and the one every pre-W4.1 graph was
  // made of: one axis, encoded byte for byte as it always was.
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
  // `keepdim` is shape metadata, never accumulation order, so putting the
  // reduced axes back as 1s is one `view`.
  if (!keepdim) return cursor
  return emit({
    op: "view",
    input: cursor,
    shape: [...node.shape],
  })
}
