import type { IndexTensor, Init, Shape } from "../shape.ts"
import { type AnyTensor, Tensor } from "../tensor.ts"

/** Collapses everything but the class axis into one batch axis, so `[B,T,V]`/`[B,T]` reduces to `[B,C]`/`[B]`. */
function flattenBatch(l: AnyTensor, t: AnyTensor, who: string): { logits: AnyTensor; targets: AnyTensor; batch: number } {
  if (l.rank < 2) {
    throw new Error(
      `${who}: logits must be at least rank 2 ([N, C]), got rank ${l.rank}`,
    )
  }
  const logits = (l.rank > 2 ? l.flatten(0, l.rank - 2) : l) as AnyTensor
  const targets = (t.rank > 1 ? t.flatten() : t) as AnyTensor
  const batch = logits.shape[0]!
  if (targets.numel !== batch) {
    throw new Error(
      `${who}: ${targets.numel} targets for batch of ${batch}`,
    )
  }
  return { logits, targets, batch }
}

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

/** Cross-entropy over the last axis: `[B,C]` logits take a `[B]` target, `[B,T,V]` logits take a `[B,T]` target. */
export function crossEntropy<
  S extends Shape,
>(
  logits: Tensor<S>,
  targets: IndexTensor<Init<S>>,
  options: {
    readonly ignoreIndex?: number
    readonly labelSmoothing?: number
  } = {},
): Tensor<[]> {
  const l = logits as AnyTensor
  const classes = l.shape[l.rank - 1]!
  const { logits: flatLogits, targets: flatTargets, batch } = flattenBatch(l, targets as AnyTensor, "crossEntropy")

  let keep: AnyTensor | null = null
  let denom = batch
  let oneHotSource = flatTargets
  if (options.ignoreIndex !== undefined) {
    const ii = options.ignoreIndex
    const raw = Array.from(flatTargets.data, v => Number(v))
    let kept = 0
    const keepData = new Array<number>(raw.length)
    const safeIds = new Array<number>(raw.length)
    for (let i = 0; i < raw.length; i++) {
      const isIgnored = raw[i] === ii
      keepData[i] = isIgnored ? 0 : 1
      // A sentinel like PyTorch's -100 is not a valid class id, so use class 0 and zero its contribution below.
      safeIds[i] = isIgnored ? 0 : raw[i]!
      if (!isIgnored) kept++
    }
    keep = Tensor.of(keepData) as AnyTensor
    // Every row ignored is loss 0 with denominator 1, not a div-by-zero.
    denom = kept || 1
    oneHotSource = Tensor.indices(safeIds, [raw.length]) as AnyTensor
  }

  let mask = oneHotSource.oneHot(classes)
  if (options.labelSmoothing) {
    const eps = options.labelSmoothing
    mask = mask.mul(1 - eps).add(eps / classes)
  }

  let perRow = flatLogits.logSoftmax(1).mul(mask).sum(1).neg()
  if (keep) perRow = perRow.mul(keep)
  return perRow.sum().div(denom) as any
}

/** Fraction of rows whose argmax over the last axis matches `targets`. Reads `.data` directly, so it never joins the autograd tape. */
export function accuracy<
  S extends Shape,
>(
  logits: Tensor<S>,
  targets: IndexTensor<Init<S>>,
): number {
  const { logits: flatLogits, targets: flatTargets, batch } = flattenBatch(logits as AnyTensor, targets as AnyTensor, "accuracy")
  const predData = flatLogits.argmax(1).data
  const targetData = flatTargets.data
  let correct = 0
  for (let i = 0; i < batch; i++) {
    if (Number(predData[i]) === Number(targetData[i])) correct++
  }
  return correct / batch
}
