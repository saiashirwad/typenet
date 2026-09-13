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
import { type AnyTensor, logSumExp, Tensor } from "../tensor.ts"

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
