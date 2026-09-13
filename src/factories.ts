import { rawRandom } from "./ir.ts"
import { nextStream } from "./kernels.ts"
import type { Shape } from "./shape.ts"
import { Tensor } from "./tensor.ts"

/**
 * Free-function spellings of the static factories. `fromFlat` (typed
 * flat-buffer constructor) lives in tensor.ts with the class; these are
 * the ergonomic aliases the package surface exports.
 */
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
  /**
   * `"once"` (default) draws a plain CPU leaf immediately — fixed for
   * the life of the tensor, and baked into a compiled graph as a
   * constant. `"perCall"` is a graph node that redraws on every
   * evaluation: this is what the deleted separate `uniform`/`normal`
   * spellings always were (OWNER-5) — same node, same draws, same
   * bits, just reached through `rand`/`randn` now.
   */
  resample?: "once" | "perCall"
}

/**
 * Uniform values in [0, 1). See {@link ResampleOptions} for `resample`.
 * Seeded by `configure({ seed })`, not `Math.random`.
 */
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
