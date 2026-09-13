import type { Shape } from "../../shape.ts"
import { dropout as dropoutOp, Tensor } from "../../tensor.ts"
import { Module } from "../module.ts"
import { SHAPE_EFFECT } from "../sequential.ts"

/**
 * Inverted dropout (W5.1-A), reading `this.training` (`Module`) rather
 * than taking a mode flag: `net.eval()` turns every nested `Dropout` off
 * at once.
 *
 * `p` is a trace-time literal in Phase A (the A-delta on W5.1-A): training
 * and evaluation are two separate graphs rather than one program serving
 * both through a runtime scalar, which is already what a train/eval
 * mismatch under `compile()` must reject (D28) — W3.4 (Phase B) collapses
 * them into one program.
 *
 * `eval()` is the identity **exactly**, not statistically: `forward`
 * returns `x` itself — not a copy, not a node that happens to multiply by
 * 1 — so a compiled eval program is a genuinely smaller graph, never the
 * same graph with a no-op scaled in.
 */
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
