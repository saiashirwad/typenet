import { rawRandom } from "./ir.ts"
import { nextStream } from "./kernels.ts"
import type { Shape } from "./shape.ts"
import { Tensor } from "./tensor.ts"

/** Free-function aliases of the static factories. */
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

/** Uniform values in [0, 1), seeded by `configure({ seed })`, not `Math.random`. See {@link ResampleOptions}. */
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

/** Standard normal values. See {@link rand} for `resample`. */
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
