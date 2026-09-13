import type { DimEq, ErrorMessage, Shape } from "../shape.ts"
import { type AnyTensor, Tensor } from "../tensor.ts"
import { Linear } from "./layers/linear.ts"
import { Module } from "./module.ts"

/**
 * What one layer does to a shape, at the type level. `Linear` is
 * special-cased so a chain of Linears stays rank-generic: the last axis
 * is rewritten, any batch prefix rides along. Other layers are read off
 * their `forward` signature.
 */
type ApplyLayer<L, S extends Shape> =
    number[] extends S ? number[]
  : L extends Linear<infer In, infer Out> ?
      S extends [...infer Prefix extends number[], In] ? [...Prefix, Out]
    : never
  : L extends { forward(x: Tensor<S>): Tensor<infer R extends Shape> } ? R
  : S

type ChainShape<L extends readonly unknown[], S extends Shape> =
    L extends readonly [infer H, ...infer R] ?
      ApplyLayer<H, S> extends infer S2 extends Shape ? ChainShape<R, S2>
    : never
  : S

type ChainShapeCheck<L extends readonly unknown[], S extends Shape> = [ChainShape<L, S>] extends [never]
  ? ErrorMessage<`sequential: input shape does not fit the layer chain`>
  : unknown

/**
 * Constructed through {@link sequential} only. Typed as the tuple of its
 * layers, so `forward` composes their shapes: a `Sequential` of Linears
 * maps `Tensor<[B, T, 2]>` to `Tensor<[B, T, 3]>`.
 */
export class Sequential<
  const L extends readonly unknown[],
> extends Module {
  constructor(readonly layers: L) {
    super()
  }

  forward<S extends Shape>(
    x: Tensor<S> & ChainShapeCheck<L, S>,
  ): Tensor<ChainShape<L, S>> {
    let h: AnyTensor = x as AnyTensor
    for (const layer of this.layers) {
      h = (layer as { forward(t: AnyTensor): AnyTensor })
        .forward(h)
    }
    return h as Tensor<ChainShape<L, S>>
  }
}

type LayerIn<L> =
    L extends { readonly inFeatures?: infer I } ?
      NonNullable<I> extends number ? NonNullable<I>
    : undefined
  : undefined

type LayerOut<L> =
    L extends { readonly outFeatures?: infer O } ?
      NonNullable<O> extends number ? NonNullable<O>
    : undefined
  : undefined

type NextDim<H, Prev> = LayerOut<H> extends number ? LayerOut<H> : Prev

type ChainCheck<
  L extends readonly unknown[],
  Prev extends number | undefined = undefined,
> =
    L extends readonly [infer H, ...infer R] ?
      LayerIn<H> extends infer I ?
        I extends number ?
          Prev extends number ?
            DimEq<Prev, I> extends false ? ErrorMessage<`sequential: layer expects ${I} input features but the previous layer outputs ${Prev}`>
          : ChainCheck<R, NextDim<H, Prev>>
        : ChainCheck<R, NextDim<H, Prev>>
      : ChainCheck<R, NextDim<H, Prev>>
    : never
  : unknown

// L is deliberately unconstrained: Tensor is invariant in S, so bounding
// it to a rank-2 layer type would make `Linear<2, 16>` unassignable and
// reject every real call. ChainCheck does the real work.
export function sequential<
  const L extends readonly unknown[],
>(
  ...layers: L & ChainCheck<L>
): Sequential<L>
export function sequential(
  ...layers: readonly { readonly inFeatures?: number; readonly outFeatures?: number }[]
): Sequential<readonly unknown[]> {
  let prevOut: number | undefined
  layers.forEach((l, i) => {
    if (
      prevOut !== undefined
      && l.inFeatures !== undefined
      && l.inFeatures !== prevOut
    ) {
      throw new Error(
        `sequential: layer ${i} expects ${l.inFeatures} features but the previous layer outputs ${prevOut}`,
      )
    }
    if (l.outFeatures !== undefined) prevOut = l.outFeatures
    else if (l.inFeatures !== undefined) {
      prevOut = l.inFeatures
    }
  })
  return new Sequential(layers)
}
