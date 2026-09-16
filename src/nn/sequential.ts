import type { DimEq, ErrorMessage, Shape } from "../shape.ts"
import { type AnyTensor, Tensor } from "../tensor.ts"
import { Linear } from "./layers/linear.ts"
import { Module } from "./module.ts"

/** Declares a layer's shape effect so `sequential` need not infer it from `forward`. A runtime Symbol because computed property names cannot be spelled through `import type`. */
export const SHAPE_EFFECT: unique symbol = Symbol("typenet.nn.SHAPE_EFFECT")

export type ShapeEffect =
  | "identity"
  | [effect: "mapLast", In: number, Out: number]
  | [effect: "appendDim", D: number]

/** The declared half of {@link ApplyLayer}. The generic-S guard comes first: a fully generic shape must decide nothing. */
type ApplyEffect<E extends ShapeEffect, S extends Shape> =
    number[] extends S ? number[]
  : E extends "identity" ? S
  : E extends [effect: "mapLast", In: infer In extends number, Out: infer Out extends number] ?
      S extends [...infer Prefix extends number[], In] ? [...Prefix, Out]
    : never
  : E extends [effect: "appendDim", D: infer D extends number] ? [...S, D]
  : S

type ApplyLayer<L, S extends Shape> =
    L extends { readonly [SHAPE_EFFECT]: infer E extends ShapeEffect } ? ApplyEffect<E, S>
  : number[] extends S ? number[]
  : L extends Linear<infer In, infer Out> ?
      S extends [...infer Prefix extends number[], In] ? [...Prefix, Out]
    : never
  : L extends { forward(x: Tensor<S>): Tensor<infer R extends Shape> } ? R
  : L extends { forward: unknown } ? S
  : ErrorMessage<`sequential: every layer needs a forward method; this one has none`>

type ChainShape<L extends readonly unknown[], S extends Shape> =
    L extends readonly [infer H, ...infer R] ?
      ApplyLayer<H, S> extends infer S2 extends Shape ? ChainShape<R, S2>
    : never
  : S

type ChainShapeCheck<L extends readonly unknown[], S extends Shape> = [ChainShape<L, S>] extends [never]
  ? ErrorMessage<`sequential: input shape does not fit the layer chain`>
  : unknown

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

/** The width a layer demands of the axis it is handed, or `undefined`. Only `mapLast` contributes. */
type LayerIn<L> =
    L extends { readonly [SHAPE_EFFECT]: [effect: "mapLast", In: infer In extends number, Out: number] } ? In
  : L extends { readonly [SHAPE_EFFECT]: ShapeEffect } ? undefined
  : L extends { readonly inFeatures?: infer I } ?
      NonNullable<I> extends number ? NonNullable<I>
    : undefined
  : undefined

/** The width a layer leaves on the last axis, or `undefined` to carry the previous one through. */
type LayerOut<L> =
    L extends { readonly [SHAPE_EFFECT]: [effect: "mapLast", In: number, Out: infer Out extends number] } ? Out
  : L extends { readonly [SHAPE_EFFECT]: [effect: "appendDim", D: infer D extends number] } ? D
  : L extends { readonly [SHAPE_EFFECT]: ShapeEffect } ? undefined
  : L extends { readonly outFeatures?: infer O } ?
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

// L stays unconstrained: Tensor is invariant in S, so bounding it to a
// rank-2 layer type would reject every real call. ChainCheck does the work.
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
