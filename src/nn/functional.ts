// nn.functional (W5.1-A): free functions with the exact semantics of the
// layer catalog above, so a hand-rolled block (no `Module` in sight) still
// gets the fused kernels the new IR ops (W4.1) provide — one node for
// `gelu`/`silu`/`softmax`/`layerNorm`/`rmsNorm`/`dropout`, not the
// composed spelling.
//
// `gelu`/`silu`/`softmax`/`layerNorm`/`rmsNorm`/`dropout` are re-exports,
// not wrappers: they ARE the functions `Embedding`/`LayerNorm`/`RMSNorm`/
// `Dropout`/`GELU`/`SiLU` above are built out of, so there is exactly one
// implementation of each op's semantics, not two that could drift.
import type { DimCheck, IndexTensor, Shape } from "../shape.ts"
import { showShape } from "../storage.ts"
import { type AnyTensor, dropout as dropoutOp, logSumExp, softmax as softmaxOp, Tensor } from "../tensor.ts"

export { dropout, gelu, layerNorm, rmsNorm, silu, softmax } from "../tensor.ts"

/**
 * `log(softmax(x, dim))`, over the fused `logSumExp` node (W4.1) —
 * `logp = x - logSumExp(x, dim)`, the same composition `crossEntropy`'s
 * eager kernel uses internally (`src/ir.ts`) — rather than
 * `Tensor.prototype.logSoftmax`'s five-node `max`/`sub`/`exp`/`sum`/`log`
 * composed spelling.
 */
export function logSoftmax<
  S extends Shape,
  const D extends number,
>(
  x: Tensor<S>,
  dim: D & DimCheck<S, D>,
): Tensor<S> {
  const a = x as AnyTensor
  const lse = logSumExp(a, dim as number, true) as AnyTensor
  return a.sub(lse) as Tensor<S>
}

/**
 * `IndexTensor<[N]>` holding `0, 1, ..., N-1` — the row indices
 * `Embedding`/`gatherRows` expect, e.g. for a position embedding over a
 * sequence of length `T`.
 */
export function arangeIndex<const N extends number>(n: N): IndexTensor<[N]> {
  const data = new Array<number>(n)
  for (let i = 0; i < n; i++) data[i] = i
  return Tensor.indices(data, [n])
}

/**
 * Scaled dot-product attention at rank 4 — the generic escape hatch under
 * {@link MultiHeadAttention} (W5.2-A step 3), for the block that wants to
 * own its own projections.
 *
 * `k` arrives ALREADY TRANSPOSED as `[B, H, K, T]` rather than as
 * `[B, H, T, K]` with a `transpose` inside. That is not a convenience: the
 * caller reaches rank 4 through `unflatten(...).permute(...)` anyway
 * (D21 — `view` cannot express a generic-dim head split), so it costs the
 * caller one different permutation order and it saves this function from
 * emitting a second `permute` node the caller's own permute already paid
 * for. It is also what makes the shape algebra decide: `q @ k` is
 * `[B,H,T,K] @ [B,H,K,T]` straight out of `MatMul`, with no
 * `Transpose<...>` for TS to reduce through.
 *
 * Causal masking is a property of the `softmax{causal}` NODE (W4.1), not a
 * materialised `[T, T]` buffer added to the scores: nothing here allocates
 * `T*T` floats, and `test/attention.test.ts` asserts that against the
 * serialised graph rather than trusting this comment.
 */
export function sdpa<
  B extends number,
  H extends number,
  T extends number,
  K extends number,
>(
  q: Tensor<[B, H, T, K]>,
  // `NoInfer` on every axis of `k` and `v`: all four dims are read off `q`
  // alone, so a head count or a head width that disagrees is a mismatch
  // against a KNOWN target rather than one more inference candidate TS
  // gets to reconcile. Without it, `sdpa(q, k, v)` with the wrong number
  // of heads in `k` silently unifies and the error surfaces, if at all, at
  // the kernel.
  k: Tensor<[NoInfer<B>, NoInfer<H>, NoInfer<K>, NoInfer<T>]>,
  v: Tensor<[NoInfer<B>, NoInfer<H>, NoInfer<T>, NoInfer<K>]>,
  options: { causal?: boolean; dropout?: number } = {},
): Tensor<[B, H, T, K]> {
  const qa = q as AnyTensor
  const ka = k as AnyTensor
  const va = v as AnyTensor
  if (qa.rank !== 4 || ka.rank !== 4 || va.rank !== 4) {
    throw new Error(
      `sdpa: expects rank-4 [B, H, T, K] operands (k transposed to [B, H, K, T]); `
        + `got q ${showShape(qa.shape)}, k ${showShape(ka.shape)}, v ${showShape(va.shape)}`,
    )
  }
  const headDim = qa.shape[3]!
  if (ka.shape[2] !== headDim || va.shape[3] !== headDim) {
    throw new Error(
      `sdpa: head dim disagrees — q ${showShape(qa.shape)}, k ${showShape(ka.shape)}, v ${showShape(va.shape)}`,
    )
  }
  // Scale the scores, not `q`: one `[B,H,T,T]` scalar multiply instead of
  // a `[B,H,T,K]` one is the cheaper of the two only when T < K, but it
  // is the one PyTorch's reference formula writes and the one the torch
  // fixtures (W5.8) are generated from, so it is the one that keeps the
  // cross-framework parity check comparing like with like.
  const scores = qa.matmul(ka).mul(1 / Math.sqrt(headDim))
  const weights = softmaxOp(scores, -1, { causal: options.causal ?? false })
  const p = options.dropout ?? 0
  // `p === 0` emits NO node at all, the same exactness rule `nn.Dropout`
  // follows: an inference-time attention graph is genuinely smaller, not
  // the training graph with a multiply-by-one left in.
  const dropped = p > 0 ? dropoutOp(weights, p) : weights
  return dropped.matmul(va) as Tensor<[B, H, T, K]>
}
