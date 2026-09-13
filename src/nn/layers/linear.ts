import type { MatMul, MatMulCheck, Shape } from "../../shape.ts"
import { Tensor } from "../../tensor.ts"
import * as init from "../init.ts"
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
    // `weight` is `[In, Out]` (matmul order), transposed relative to
    // PyTorch's `[Out, In]` — `fanMode: "fanOut"` reads `init`'s
    // fan table (dim 0 = "fan-out") to land on `fanIn = inFeatures`,
    // reproducing this layer's old `1/sqrt(fanIn)` bound bit for bit
    // (init.ts's `kaimingBound`, W1.9's accept #2).
    this.weight = init.kaimingUniform_(
      // Explicit type argument: inferring `Sh` from `[inFeatures,
      // outFeatures]` through a second, outer generic call
      // (`kaimingUniform_<T extends AnyTensor>`) widens it to the bare
      // `Shape` instead of the tuple `const Sh` would otherwise pick up
      // directly — a TS inference quirk with nested generic calls, not
      // a runtime concern.
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
