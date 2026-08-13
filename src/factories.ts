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
export const rand = Tensor.rand
export const randn = Tensor.randn
export const eye = Tensor.eye
export const arange = Tensor.arange
export const scalar = Tensor.scalar
export const stack = Tensor.stack
export const cat = Tensor.cat
