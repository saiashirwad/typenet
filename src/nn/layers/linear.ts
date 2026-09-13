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

/**
 * The transposed LM head (§3.3, W5.10-A). Exists because plain `tie()`
 * cannot express this tying: an untied head is `Linear<D, V>` with a
 * `[D, V]` weight, an `Embedding<V, D>` owns a `[V, D]` weight, and
 * `tie<S>(a: Parameter<S>, b: Parameter<NoInfer<S>>)` can only relate two
 * parameters of the *same* shape — `[D, V]` is not `[V, D]` (verified
 * during review: `scratchpad/review/gpt-block.ts` rejects
 * `nn.tie(this.head.weight, this.wte.weight)` with a shape mismatch).
 *
 * `TiedLinear.of(embedding)` sidesteps the problem instead of solving it:
 * it stores **the embedding's own `Parameter<[V, D]>` object**, not a
 * second, independently-owned weight. Both uses are then literally the
 * same `Tensor`/same storage — `Module`'s reflection (module.ts) finds
 * `weight` again at this layer's own path and dedups it against the
 * embedding's path by storage identity, so a tied model has exactly one
 * `V*D`-sized parameter, one accumulated `.grad`, and one `stateDict`
 * entry, never two that can drift apart.
 *
 * `forward` emits `matmul(x, transpose(weight))`. On today's wire the
 * `transpose` is a `permute` node (a `WIRE_OPS` primitive, so this never
 * trips A-L1's native fallback), which candle materialises as one
 * `[V, D]` -> `[D, V]` copy per step — correctness-identical to a second,
 * independently-owned `[D, V]` weight. `views_to_layout` + `matmul_strided`
 * turning that into a zero-copy `transb = CblasTrans` GEMM is W4.6 (Phase
 * B, tracked in the deferred-acceptance register, PLAN-V2 §5A.5 #9);
 * nothing about the shape or the sharing changes when that lands.
 */
export class TiedLinear<D extends number, V extends number> extends Module {
  // The one layer whose stored weight is the TRANSPOSE of its declared
  // effect (W4.9's own reason the shape-effect protocol exists over the
  // structural probe): `weight` is `[V, D]`, but this layer maps `D -> V`.
  declare readonly [SHAPE_EFFECT]: [effect: "mapLast", In: D, Out: V]

  readonly weight: Parameter<[V, D]>

  private constructor(weight: Parameter<[V, D]>) {
    super()
    this.weight = weight
  }

  /** Ties the LM head to `embedding`'s own weight — no second parameter is created. */
  static of<V extends number, D extends number>(embedding: Embedding<V, D>): TiedLinear<D, V> {
    return new TiedLinear<D, V>(embedding.weight)
  }

  forward<S extends Shape>(
    x: Tensor<S> & LastDimCheck<S, D>,
  ): Tensor<MatMul<S, [D, V]>> {
    // Same generic-deferral bridge `Linear.forward` documents: the
    // precondition already proves the matmul is shape-valid for a
    // concrete `S`, but for a generic `S` TS cannot reduce `MatMulCheck`
    // to `unknown` on its own, so the transposed weight carries the
    // check explicitly.
    const wt = this.weight.transpose(0, 1) as Tensor<[D, V]> & MatMulCheck<S, [D, V]>
    return x.matmul(wt)
  }
}
