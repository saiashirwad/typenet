import type { DimCheck, IndexTensor, Shape } from "../shape.ts"
import { showShape } from "../storage.ts"
import { type AnyTensor, dropout as dropoutOp, logSumExp, softmax as softmaxOp, Tensor } from "../tensor.ts"

export { dropout, gelu, layerNorm, rmsNorm, silu, softmax } from "../tensor.ts"

/** `log(softmax(x, dim))` computed as `x - logSumExp(x, dim)` over the fused node. */
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

/** `IndexTensor<[N]>` holding `0, 1, ..., N-1`, e.g. for position embeddings. */
export function arangeIndex<const N extends number>(n: N): IndexTensor<[N]> {
  const data = new Array<number>(n)
  for (let i = 0; i < n; i++) data[i] = i
  return Tensor.indices(data, [n])
}

/** Scaled dot-product attention at rank 4. `k` arrives pre-transposed as `[B, H, K, T]`, and causal masking is a softmax option, not a materialised `[T, T]` buffer. */
export function sdpa<
  B extends number,
  H extends number,
  T extends number,
  K extends number,
>(
  q: Tensor<[B, H, T, K]>,
  // NoInfer on every axis of `k`/`v`: all dims are read off `q`, so a
  // mismatched head count fails against a known target instead of unifying.
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
  // Scale the scores, not `q`: the PyTorch reference formula's spelling.
  const scores = qa.matmul(ka).mul(1 / Math.sqrt(headDim))
  const weights = softmaxOp(scores, -1, { causal: options.causal ?? false })
  const p = options.dropout ?? 0
  // `p === 0` emits no node at all.
  const dropped = p > 0 ? dropoutOp(weights, p) : weights
  return dropped.matmul(va) as Tensor<[B, H, T, K]>
}
