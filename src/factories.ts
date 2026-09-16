import { rawRandom } from "./ir.ts"
import { nextStream } from "./kernels.ts"
import type { IndexTensor, Shape } from "./shape.ts"
import { showShape } from "./storage.ts"
import { type AnyTensor, Tensor } from "./tensor.ts"

export const tensor = Tensor.of
export const zeros = Tensor.zeros
export const ones = Tensor.ones
export const full = Tensor.full
export const eye = Tensor.eye
export const arange = Tensor.arange
export const scalar = Tensor.scalar
export const stack = Tensor.stack
export const cat = Tensor.cat

export type ResampleOptions = {
  /** `"once"` (default) draws a CPU leaf now; `"perCall"` is a graph node that redraws on every evaluation. */
  resample?: "once" | "perCall"
}

export function rand<const Sh extends Shape>(
  shape: Sh,
  options?: ResampleOptions,
): Tensor<Sh> {
  if (options?.resample === "perCall") {
    return rawRandom(
      "uniform",
      shape,
      nextStream(),
      "float32",
    ) as any
  }
  return Tensor.rand(shape)
}

export function randn<const Sh extends Shape>(
  shape: Sh,
  options?: ResampleOptions,
): Tensor<Sh> {
  if (options?.resample === "perCall") {
    return rawRandom(
      "normal",
      shape,
      nextStream(),
      "float32",
    ) as any
  }
  return Tensor.randn(shape)
}

export interface CategoricalOptions {
  /**
   * `1` (default) samples the row as given. Other values raise or lower the weights to the power
   * `1/temperature`, which is the same as scaling logits when the row is a softmax output.
   */
  temperature?: number
  /** A uniform draw in `[0, 1)` per row. Pass one to make a sample replay; the default draws. */
  rng?: () => number
}

/**
 * Draws one index per row of `probabilities` and returns them as an `IndexTensor<[N]>`, which
 * `Embedding` and `indexSelect` accept directly.
 *
 * `probabilities` is a rank-2 `[N, C]` of non-negative weights, which need not sum to one, so
 * sampling a softmax's output and sampling its logits with a temperature are the same call. The
 * draw is a cumulative sum, so it never builds an `[N, C]` uniform tensor to compare against, and
 * it reads values rather than joining the tape: a sample is not differentiable.
 */
export function categorical(
  probabilities: AnyTensor,
  options: CategoricalOptions = {},
): IndexTensor<[number]> {
  if (probabilities.rank !== 2) {
    throw new Error(
      `categorical: expected a rank-2 [N, C] tensor, got ${showShape(probabilities.shape)}`,
    )
  }
  const temperature = options.temperature ?? 1
  if (!(temperature > 0)) {
    throw new Error(`categorical: temperature must be positive, got ${temperature}`)
  }
  const rng = options.rng ?? Math.random
  const [rows, classes] = probabilities.shape as [number, number]
  const data = probabilities.data
  const picks = new Array<number>(rows)
  const power = 1 / temperature
  for (let n = 0; n < rows; n++) {
    const base = n * classes
    let total = 0
    for (let c = 0; c < classes; c++) total += Math.max(Number(data[base + c]!), 0) ** power
    const threshold = rng() * total
    // The last class is the fallback, so a draw of exactly `total` (or a rounding shortfall)
    // still lands on a valid index.
    let pick = classes - 1
    let cumulative = 0
    for (let c = 0; c < classes - 1; c++) {
      cumulative += Math.max(Number(data[base + c]!), 0) ** power
      if (threshold < cumulative) {
        pick = c
        break
      }
    }
    picks[n] = pick
  }
  return Tensor.indices(picks, [rows])
}
