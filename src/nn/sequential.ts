import type { DimEq, ErrorMessage, Shape } from "../shape.ts"
import { type AnyTensor, Tensor } from "../tensor.ts"
import { Linear } from "./layers/linear.ts"
import { Module } from "./module.ts"

/**
 * The `nn` shape-effect protocol (W4.9). A layer that declares
 *
 * ```ts
 * declare readonly [SHAPE_EFFECT]: ["mapLast", In, Out]
 * ```
 *
 * tells {@link ApplyLayer} what it does to a shape *directly*, instead of
 * having it inferred by instantiating the layer's `forward` at every shape
 * in the chain. Each of the three reasons that matters is a bug the
 * structural probe cannot fix:
 *
 * 1. **A `forward` that is not a plain `Tensor<S> -> Tensor<R>` does not
 *    probe at all.** `Embedding.forward` takes an `IndexTensor<S>` — a
 *    branded `Tensor`, so `Tensor<S>` is not assignable to it — and the
 *    probe falls through to its `S` fallback, making the layer behave like
 *    the identity inside `sequential`. That silent no-op is the bug this
 *    item exists to fix.
 * 2. **A layer whose stored weight is the transpose of its shape effect**
 *    — `TiedLinear` holds the embedding's `[V, D]` and maps `D -> V` —
 *    cannot be read off its fields, only off a declaration (§3.3).
 * 3. The probe re-instantiates a generic `forward` once per chain step;
 *    matching a declared tuple does not.
 *
 * It lives here, beside the type that consumes it, and not in
 * `src/shape.ts`: this is an `nn` protocol, not shape algebra, and
 * `src/shape.ts` must not gain an `nn` dependency.
 *
 * The symbol is a real runtime `Symbol` rather than the ambient
 * `declare const` a type-only marker would use, because
 * `verbatimModuleSyntax` keeps a layer's `import { SHAPE_EFFECT }` in the
 * emitted module graph — and a computed property name cannot be spelled
 * through `import type`. A binding that existed only in the type world
 * would therefore be a missing export at run time. Nothing ever reads it.
 */
export const SHAPE_EFFECT: unique symbol = Symbol("typenet.nn.SHAPE_EFFECT")

/**
 * What a layer does to a shape, declared rather than inferred.
 *
 * - `"identity"` — shape-preserving *and* width-agnostic: activations,
 *   `Dropout`, anything whose `forward` is `Tensor<S> -> Tensor<S>` for
 *   every `S`.
 * - `["mapLast", In, Out]` — owns the last axis and rewrites it, any batch
 *   prefix riding along: `Linear`, `TiedLinear`, and the norms. A norm
 *   declares `["mapLast", D, D]` and *not* `"identity"`, so that feeding a
 *   `LayerNorm(8)` from a `Linear(4, 4)` is still a width mismatch rather
 *   than a shape the chain quietly accepts.
 * - `["appendDim", D]` — grows the rank by one: `Embedding` turns `[...S]`
 *   indices into `[...S, D]` vectors.
 */
export type ShapeEffect =
  | "identity"
  | [effect: "mapLast", In: number, Out: number]
  | [effect: "appendDim", D: number]

/**
 * The declared half of {@link ApplyLayer}. The `number[] extends S` guard
 * is repeated here rather than left to the branch below it because law 1
 * (§2.1) is not negotiable: a fully generic shape must decide nothing.
 * Without it `["appendDim", D]` would turn `number[]` into the strictly
 * narrower `[...number[], D]`, and `["mapLast", In, Out]` would fail its
 * tuple match and collapse the whole chain to `never`.
 */
type ApplyEffect<E extends ShapeEffect, S extends Shape> =
    number[] extends S ? number[]
  : E extends "identity" ? S
  : E extends [effect: "mapLast", In: infer In extends number, Out: infer Out extends number] ?
      S extends [...infer Prefix extends number[], In] ? [...Prefix, Out]
    : never
  : E extends [effect: "appendDim", D: infer D extends number] ? [...S, D]
  : S

/**
 * What one layer does to a shape, at the type level.
 *
 * A declared {@link SHAPE_EFFECT} wins over everything else. Below it,
 * `Linear` keeps its special case so a chain of Linears stays rank-generic
 * (the last axis is rewritten, any batch prefix rides along), and the
 * structural probe keeps reading an undeclared layer off its `forward`
 * signature. Both are deliberately retained (D24): a third-party layer
 * that declares nothing must compose exactly as it does today, so the
 * protocol is an opt-in refinement and never a migration.
 *
 * Only the last fallback — a "layer" with no `forward` at all, which can
 * never be a layer — is an `ErrorMessage`. A layer that *has* a `forward`
 * the probe cannot read still falls through to `S`, unchanged.
 */
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

/**
 * The width a layer demands of the axis it is handed, or `undefined` when
 * it demands nothing. A declared {@link SHAPE_EFFECT} is read first — and
 * only its `mapLast` arm contributes: `"identity"` is width-agnostic by
 * definition, and `appendDim`'s operand is an index tensor whose last axis
 * is not a feature width. The `inFeatures` probe below it is unchanged, so
 * `Linear` and every undeclared third-party layer behave exactly as today.
 */
type LayerIn<L> =
    L extends { readonly [SHAPE_EFFECT]: [effect: "mapLast", In: infer In extends number, Out: number] } ? In
  : L extends { readonly [SHAPE_EFFECT]: ShapeEffect } ? undefined
  : L extends { readonly inFeatures?: infer I } ?
      NonNullable<I> extends number ? NonNullable<I>
    : undefined
  : undefined

/**
 * The width a layer leaves on the last axis, or `undefined` when it leaves
 * whatever it was given. `mapLast` reports its `Out`; `appendDim` reports
 * the axis it appends, which *is* the new last axis; `"identity"` reports
 * nothing so {@link NextDim} carries the previous width through.
 */
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
