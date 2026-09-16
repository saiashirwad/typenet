import { rawRandom } from "./ir.ts"
import { nextStream } from "./kernels.ts"
import type { Shape } from "./shape.ts"
import { Tensor } from "./tensor.ts"

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
