import type { LastDimCheck, Shape } from "../../shape.ts"
import { type AnyTensor, layerNorm, rmsNorm, Tensor } from "../../tensor.ts"
import { Module } from "../module.ts"
import { type Parameter, parameter } from "../parameter.ts"
import { SHAPE_EFFECT } from "../sequential.ts"

/**
 * LayerNorm over the last axis (W5.1-A), emitting the single `layerNorm`
 * node (W4.1) rather than the mean/var/sub/mul composed spelling — one
 * fused kernel, one fused gradient.
 *
 * Declares `["mapLast", D, D]`, not `"identity"` (W4.9's protocol): a norm
 * OWNS the last axis, so feeding a `LayerNorm(8)` from a `Linear(4, 4)`
 * inside `sequential(...)` must stay a width mismatch, not a shape the
 * chain quietly accepts because the effect looked identity-shaped.
 */
export class LayerNorm<D extends number> extends Module {
  declare readonly [SHAPE_EFFECT]: [effect: "mapLast", In: D, Out: D]

  readonly gamma: Parameter<[D]>
  readonly beta: Parameter<[D]>
  readonly dim: D
  readonly eps: number

  constructor(dim: D, options: { eps?: number } = {}) {
    super()
    this.dim = dim
    this.eps = options.eps ?? 1e-5
    // Explicit type arguments: inferring `Sh` from `[dim]` through a
    // second, outer generic call widens it to the bare `Shape` instead of
    // the tuple `const Sh` would otherwise pick up directly (the same TS
    // inference quirk `Linear`'s constructor documents).
    this.gamma = parameter(Tensor.ones<[D]>([dim]))
    this.beta = parameter(Tensor.zeros<[D]>([dim]))
  }

  /**
   * `LastDimCheck<S, D>` is the compile-time half of the shape-type law;
   * `layerNorm`'s own `checkNormWeight` (ir.ts) is the runtime twin — the
   * cast to `AnyTensor` below only bridges generic deferral (for a generic
   * `S`, `Last<S>` never reduces far enough for TS to see `gamma`/`beta`
   * as the right width), it does not skip either check.
   */
  forward<S extends Shape>(
    x: Tensor<S> & LastDimCheck<S, D>,
  ): Tensor<S> {
    const a = x as AnyTensor
    return layerNorm(a, this.gamma as AnyTensor, this.beta as AnyTensor, {
      eps: this.eps,
    }) as Tensor<S>
  }
}

/**
 * RMSNorm over the last axis (W5.1-A): LayerNorm without the mean
 * subtraction, one `rmsNorm` node, one learnable `gamma`.
 */
export class RMSNorm<D extends number> extends Module {
  declare readonly [SHAPE_EFFECT]: [effect: "mapLast", In: D, Out: D]

  readonly gamma: Parameter<[D]>
  readonly dim: D
  readonly eps: number

  constructor(dim: D, options: { eps?: number } = {}) {
    super()
    this.dim = dim
    this.eps = options.eps ?? 1e-5
    this.gamma = parameter(Tensor.ones<[D]>([dim]))
  }

  forward<S extends Shape>(
    x: Tensor<S> & LastDimCheck<S, D>,
  ): Tensor<S> {
    const a = x as AnyTensor
    return rmsNorm(a, this.gamma as AnyTensor, { eps: this.eps }) as Tensor<S>
  }
}
