import type { Shape } from "../../shape.ts"
import { dropout as dropoutOp, Tensor } from "../../tensor.ts"
import { Module } from "../module.ts"
import { SHAPE_EFFECT } from "../sequential.ts"

/** Inverted dropout keyed off `this.training`; eval returns `x` itself, so an eval graph is genuinely smaller, never a multiply-by-one. */
export class Dropout extends Module {
  declare readonly [SHAPE_EFFECT]: "identity"

  readonly p: number

  constructor(p = 0.5) {
    super()
    if (!(p >= 0) || p >= 1) {
      throw new Error(`Dropout: p must be in [0, 1), got ${p}`)
    }
    this.p = p
  }

  forward<S extends Shape>(x: Tensor<S>): Tensor<S> {
    if (!this.training) return x
    return dropoutOp(x, this.p)
  }
}
