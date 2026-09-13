import type { MatMul, MatMulCheck, Shape } from "../../shape.ts"
import { Tensor } from "../../tensor.ts"
import { Module } from "../module.ts"

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
    const k = 1 / Math.sqrt(inFeatures)
    this.weight = Tensor.rand([inFeatures, outFeatures])
      .mul(2 * k)
      .sub(k)
      .detach()
      .requiresGrad()
    this.bias = options.bias === false
      ? null
      : Tensor.zeros([outFeatures]).requiresGrad()
  }

  forward<S extends Shape>(
    x: Tensor<S> & MatMulCheck<S, [In, Out]>,
  ): Tensor<MatMul<S, [In, Out]>> {
    // `forward`'s precondition (`MatMulCheck<S, [In, Out]>` on `x`)
    // already proves the matmul is shape-valid; the cast below only bridges
    // generic-deferral: for a generic `S`, TS can't reduce `MatMulCheck` to
    // `unknown`, so the `other` argument must carry the check explicitly.
    const y = x.matmul(this.weight as Tensor<[In, Out]> & MatMulCheck<S, [In, Out]>)
    // Broadcasting a `[Out]` bias over a `[..., Out]` matrix is shape-
    // preserving at runtime; `Broadcast<MatMul<S,[In,Out]>, [Out]>` does not
    // reduce to `MatMul<S,[In,Out]>` for generic `S`, so the bias branch
    // needs the documented double cast rather than `as any`, which would
    // erase the declared return type entirely.
    return (this.bias ? y.add(this.bias) : y) as unknown as Tensor<
      MatMul<S, [In, Out]>
    >
  }
}
