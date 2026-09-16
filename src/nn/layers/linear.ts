import type { LastDimCheck, MatMul, MatMulCheck, Shape } from "../../shape.ts"
import { Tensor } from "../../tensor.ts"
import * as init from "../init.ts"
import { Module } from "../module.ts"
import type { Parameter } from "../parameter.ts"
import { SHAPE_EFFECT } from "../sequential.ts"
import { Embedding } from "./embedding.ts"

export class Linear<
  In extends number,
  Out extends number,
> extends Module {
  readonly weight: Tensor<[In, Out]>
  readonly bias: Tensor<[Out]> | null
  readonly inFeatures: In
  readonly outFeatures: Out

  constructor(
    inFeatures: In,
    outFeatures: Out,
    options: { bias?: boolean } = {},
  ) {
    super()
    this.inFeatures = inFeatures
    this.outFeatures = outFeatures
    // `weight` is `[In, Out]` (matmul order), the transpose of PyTorch's `[Out, In]`; fanMode "fanOut" makes the fan table land on fanIn = inFeatures.
    this.weight = init.kaimingUniform_(
      // Explicit type argument: inferring `Sh` through the outer generic call widens it to bare `Shape`.
      Tensor.zeros<[In, Out]>([inFeatures, outFeatures]),
      { fanMode: "fanOut" },
    )
      .detach()
      .requiresGrad()
    this.bias = options.bias === false
      ? null
      : Tensor.zeros([outFeatures]).requiresGrad()
  }

  forward<S extends Shape>(
    x: Tensor<S> & MatMulCheck<S, [In, Out]>,
  ): Tensor<MatMul<S, [In, Out]>> {
    // Both casts bridge generic deferral: for a generic `S`, TS cannot reduce MatMulCheck to `unknown` or Broadcast<..., [Out]> to `MatMul<...>`.
    const y = x.matmul(this.weight as Tensor<[In, Out]> & MatMulCheck<S, [In, Out]>)
    return (this.bias ? y.add(this.bias) : y) as unknown as Tensor<
      MatMul<S, [In, Out]>
    >
  }
}

/** An LM head that stores the embedding's own `Parameter` object, so tying is literal: one parameter, one accumulated `.grad`, one `stateDict` entry. `tie()` alone cannot express this, since it requires equal shapes. */
export class TiedLinear<D extends number, V extends number> extends Module {
  // `weight` is `[V, D]` but the layer maps `D -> V`: the stored weight is the transpose of the declared effect.
  declare readonly [SHAPE_EFFECT]: [effect: "mapLast", In: D, Out: V]

  readonly weight: Parameter<[V, D]>

  private constructor(weight: Parameter<[V, D]>) {
    super()
    this.weight = weight
  }

  static of<V extends number, D extends number>(embedding: Embedding<V, D>): TiedLinear<D, V> {
    return new TiedLinear<D, V>(embedding.weight)
  }

  forward<S extends Shape>(
    x: Tensor<S> & LastDimCheck<S, D>,
  ): Tensor<MatMul<S, [D, V]>> {
    const wt = this.weight.transpose(0, 1) as Tensor<[D, V]> & MatMulCheck<S, [D, V]>
    return x.matmul(wt)
  }
}
