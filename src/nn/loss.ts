import type { IndexTensor, Init, Shape } from "../shape.ts"
import { type AnyTensor, Tensor } from "../tensor.ts"

export function mseLoss<
  S extends Shape,
>(
  prediction: Tensor<S>,
  target: Tensor<NoInfer<S>>,
): Tensor<[]> {
  return (prediction as AnyTensor)
    .sub(target as AnyTensor)
    .pow(2)
    .mean() as any
}

/** Cross-entropy over the last axis of `logits`: `[B,C]` logits take a `[B]` target, `[B,T,V]` logits take a `[B,T]` target. */
export function crossEntropy<
  S extends Shape,
>(
  logits: Tensor<S>,
  targets: IndexTensor<Init<S>>,
  o: {
    /** Excluded from the loss and its gradient entirely, like PyTorch's `ignore_index`. */
    readonly ignoreIndex?: number
    /** Blend the one-hot target with a uniform distribution over classes by this fraction. */
    readonly labelSmoothing?: number
  } = {},
): Tensor<[]> {
  const l = logits as AnyTensor
  const t = targets as AnyTensor
  if (l.rank < 2) {
    throw new Error(
      `crossEntropy: logits must be at least rank 2 ([N, C]), got rank ${l.rank}`,
    )
  }
  const classes = l.shape[l.rank - 1]!
  // Everything but the class axis collapses into one batch axis, so
  // [B,T,V]/[B,T] reduces to [B,C]/[B].
  const flatLogits = (l.rank > 2 ? l.flatten(0, l.rank - 2) : l) as AnyTensor
  const flatTargets = (t.rank > 1 ? t.flatten() : t) as AnyTensor
  const batch = flatLogits.shape[0]!
  if (flatTargets.numel !== batch) {
    throw new Error(
      `crossEntropy: ${flatTargets.numel} targets for batch of ${batch}`,
    )
  }

  let keep: AnyTensor | null = null
  let denom = batch
  let oneHotSource = flatTargets
  if (o.ignoreIndex !== undefined) {
    const ii = o.ignoreIndex
    const raw = Array.from(flatTargets.data, v => Number(v))
    let kept = 0
    const keepData = new Array<number>(raw.length)
    const safeIds = new Array<number>(raw.length)
    for (let i = 0; i < raw.length; i++) {
      const isIgnored = raw[i] === ii
      keepData[i] = isIgnored ? 0 : 1
      // A sentinel like PyTorch's -100 is not a valid class id; use
      // class 0 here and zero its contribution below.
      safeIds[i] = isIgnored ? 0 : raw[i]!
      if (!isIgnored) kept++
    }
    keep = Tensor.of(keepData) as AnyTensor
    // Every row ignored is loss 0 with denominator 1, not a div-by-zero.
    denom = kept || 1
    oneHotSource = Tensor.indices(safeIds, [raw.length]) as AnyTensor
  }

  let mask = oneHotSource.oneHot(classes)
  if (o.labelSmoothing) {
    const eps = o.labelSmoothing
    mask = mask.mul(1 - eps).add(eps / classes)
  }

  let perRow = flatLogits.logSoftmax(1).mul(mask).sum(1).neg()
  if (keep) perRow = perRow.mul(keep)
  return perRow.sum().div(denom) as any
}

/** Fraction of rows whose argmax over the last axis matches `targets`; reads `.data` directly and never joins the autograd tape. */
export function accuracy<
  S extends Shape,
>(
  logits: Tensor<S>,
  targets: IndexTensor<Init<S>>,
): number {
  const l = logits as AnyTensor
  const t = targets as AnyTensor
  if (l.rank < 2) {
    throw new Error(
      `accuracy: logits must be at least rank 2 ([N, C]), got rank ${l.rank}`,
    )
  }
  const flatLogits = (l.rank > 2 ? l.flatten(0, l.rank - 2) : l) as AnyTensor
  const flatTargets = (t.rank > 1 ? t.flatten() : t) as AnyTensor
  const batch = flatLogits.shape[0]!
  if (flatTargets.numel !== batch) {
    throw new Error(
      `accuracy: ${flatTargets.numel} targets for batch of ${batch}`,
    )
  }
  const preds = flatLogits.argmax(1)
  const predData = preds.data
  const targetData = flatTargets.data
  let correct = 0
  for (let i = 0; i < batch; i++) {
    if (Number(predData[i]) === Number(targetData[i])) correct++
  }
  return correct / batch
}
