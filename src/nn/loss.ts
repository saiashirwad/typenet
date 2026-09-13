import type { Shape } from "../shape.ts"
import { type AnyTensor, fromFlat, Tensor } from "../tensor.ts"

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

export function crossEntropy<
  B extends number,
  C extends number,
>(
  logits: Tensor<[B, C]>,
  targets: readonly number[] | Tensor<[NoInfer<B>]>,
): Tensor<[]> {
  const l = logits as AnyTensor
  const [batch, classes] = l.shape as number[]
  let mask: AnyTensor
  if (targets instanceof Tensor) {
    if (targets.numel !== batch) {
      throw new Error(
        `crossEntropy: ${targets.numel} targets for batch of ${batch}`,
      )
    }
    mask = targets.oneHot(classes!)
  } else {
    if (targets.length !== batch) {
      throw new Error(
        `crossEntropy: ${targets.length} targets for batch of ${batch}`,
      )
    }
    const onehot = new Float32Array(batch! * classes!)
    for (let i = 0; i < batch!; i++) {
      const target = targets[i]!
      if (
        target < 0
        || target >= classes!
        || !Number.isInteger(target)
      ) {
        throw new Error(
          `crossEntropy: target ${target} out of range for ${classes} classes`,
        )
      }
      onehot[i * classes! + target] = 1
    }
    mask = fromFlat(onehot, [batch!, classes!])
  }
  return l
    .logSoftmax(1)
    .mul(mask)
    .sum()
    .neg()
    .div(batch!) as any
}
