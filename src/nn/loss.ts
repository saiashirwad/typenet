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

/**
 * Cross-entropy over the LAST axis of `logits`, with `targets` addressing
 * that same axis one rank down: `[B,C]` logits take a `[B]` target,
 * `[B,T,V]` logits (a transformer's LM head, unreshaped) take a `[B,T]`
 * target. The flatten a caller used to have to do by hand before calling
 * this happens here instead, once — `Init<S>` (everything but the class
 * axis) IS the target shape, so there is nothing left for a caller to get
 * wrong about which axes to merge.
 *
 * `targets` is an {@link IndexTensor} (OWNER-5, D26): the untyped plain
 * numeric array spelling this used to also accept is gone, with no
 * overload and no deprecation window — build one with `Tensor.indices(ids,
 * shape)` or brand an existing tensor with `.toIndex()`.
 */
export function crossEntropy<
  S extends Shape,
>(
  logits: Tensor<S>,
  targets: IndexTensor<Init<S>>,
  o: {
    /** A target value to exclude from the mean and its gradient entirely,
     * the way PyTorch's `ignore_index` does — e.g. padding positions in a
     * batched sequence. */
    readonly ignoreIndex?: number
    /** Blend the one-hot target with a uniform distribution over classes,
     * by this fraction, before taking the log-probability dot product. */
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
  // Everything but the class axis collapses into one batch axis — the
  // [B,T,V]/[B,T] case is [B,C]/[B] with B := the product of the leading
  // dims, and `flatten` (not a reshape a caller wrote by hand) is what
  // makes that true without copying the backward rule too.
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
      // A sentinel like PyTorch's -100 is not a valid class id and would
      // make `oneHot` throw; swap it for class 0 and zero its row's
      // contribution below instead.
      safeIds[i] = isIgnored ? 0 : raw[i]!
      if (!isIgnored) kept++
    }
    keep = Tensor.of(keepData) as AnyTensor
    // Every row ignored is a degenerate call, not a div-by-zero: the loss
    // is then exactly 0 (nothing contributes) with a denominator of 1.
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

/**
 * Fraction of rows whose argmax over the last axis matches `targets` — the
 * same target typing {@link crossEntropy} uses, so a `[B,T,V]` LM head's
 * predictions compare against `[B,T]` targets with no reshape in between.
 * A plain number, not a `Tensor<[]>`: this reads `.data` directly and
 * never joins the autograd tape, so it costs nothing to compute inside a
 * training loop's periodic logging.
 */
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
