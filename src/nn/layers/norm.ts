import type { LastDimCheck, Shape } from "../../shape.ts"
import { type AnyTensor, layerNorm, rmsNorm, Tensor } from "../../tensor.ts"
import { Module } from "../module.ts"
import { type Parameter, parameter } from "../parameter.ts"
import { SHAPE_EFFECT } from "../sequential.ts"

/** LayerNorm over the last axis, emitting the single fused `layerNorm` node. */
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
    // Explicit type arguments: inference through the outer generic call
    // widens `[dim]` to bare `Shape`.
    this.gamma = parameter(Tensor.ones<[D]>([dim]))
    this.beta = parameter(Tensor.zeros<[D]>([dim]))
  }

  /** The cast only bridges generic deferral; {@link LastDimCheck} here and the runtime check in `layerNorm` both still apply. */
  forward<S extends Shape>(
    x: Tensor<S> & LastDimCheck<S, D>,
  ): Tensor<S> {
    const a = x as AnyTensor
    return layerNorm(a, this.gamma as AnyTensor, this.beta as AnyTensor, {
      eps: this.eps,
    }) as Tensor<S>
  }
}

/** LayerNorm without the mean subtraction: one `rmsNorm` node, one learnable `gamma`. */
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
