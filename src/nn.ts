import type { DimCheck, DimEq, ErrorMessage, MatMul, MatMulCheck, Shape } from "./shape.ts"
import { fromFlat, Tensor } from "./tensor.ts"

type AnyTensor = Tensor<any>

export abstract class Module {
  parameters(): AnyTensor[] {
    const out: AnyTensor[] = []
    const visit = (value: unknown) => {
      if (value instanceof Tensor) {
        if (value.needsGrad) out.push(value)
      } else if (value instanceof Module) {
        out.push(...value.parameters())
      } else if (Array.isArray(value)) {
        for (const v of value) visit(v)
      }
    }
    for (const value of Object.values(this)) visit(value)
    return out
  }

  zeroGrad(): void {
    for (const p of this.parameters()) p.zeroGrad()
  }
}

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
    const y = (x as AnyTensor).matmul(this.weight)
    return (this.bias ? y.add(this.bias) : y) as any
  }
}

export interface Layer<
  In extends number,
  Out extends number,
> {
  readonly inFeatures?: In
  readonly outFeatures?: Out

  forward<B extends number>(
    x: Tensor<[B, NoInfer<In>]>,
  ): Tensor<[B, NoInfer<Out>]>
}

class Activation extends Module {
  constructor(
    private readonly apply: (x: AnyTensor) => AnyTensor,
  ) {
    super()
  }

  forward<S extends Shape>(
    x: Tensor<S>,
  ): Tensor<S> {
    return this.apply(x) as any
  }
}

export class ReLU extends Activation {
  constructor() {
    super(x => x.relu())
  }
}

export class LeakyReLU extends Activation {
  constructor(negativeSlope = 0.01) {
    super(x => x.leakyRelu(negativeSlope))
  }
}

export class Tanh extends Activation {
  constructor() {
    super(x => x.tanh())
  }
}

export class Sigmoid extends Activation {
  constructor() {
    super(x => x.sigmoid())
  }
}

/**
 * Generic in the dim so `DimCheck` sees a literal at the call, not the
 * wide `number` (`IsValidDim<S, number>` is always true).
 */
export class Softmax<const D extends number = -1> extends Module {
  constructor(readonly dim: D = -1 as D) {
    super()
  }

  forward<S extends Shape>(
    x: Tensor<S> & DimCheck<S, D>,
  ): Tensor<S> {
    return (x as AnyTensor).softmax(this.dim) as any
  }
}

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
    return h as any
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

// L is deliberately NOT bounded by `Layer<number, number>`: Tensor is
// invariant in S, so `Linear<2, 16>` would not be assignable to it and
// every real call would be rejected. ChainCheck does the real work.
export function sequential<
  const L extends readonly unknown[],
>(
  ...layers: L & ChainCheck<L>
): Sequential<L>
export function sequential(
  ...layers: Layer<any, any>[]
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

export function mseLoss<
  S extends Shape,
>(
  prediction: Tensor<S>,
  target: Tensor<NoInfer<S>>,
): Tensor<[]> {
  return (prediction as AnyTensor)
    .sub(target as AnyTensor)
    .pow(2)
    .mean() as any
}

export function crossEntropy<
  B extends number,
  C extends number,
>(
  logits: Tensor<[B, C]>,
  targets: readonly number[] | Tensor<[NoInfer<B>]>,
): Tensor<[]> {
  const l = logits as AnyTensor
  const [batch, classes] = l.shape as number[]
  let mask: AnyTensor
  if (targets instanceof Tensor) {
    if (targets.numel !== batch) {
      throw new Error(
        `crossEntropy: ${targets.numel} targets for batch of ${batch}`,
      )
    }
    mask = targets.oneHot(classes!)
  } else {
    if (targets.length !== batch) {
      throw new Error(
        `crossEntropy: ${targets.length} targets for batch of ${batch}`,
      )
    }
    const onehot = new Float32Array(batch! * classes!)
    for (let i = 0; i < batch!; i++) {
      const target = targets[i]!
      if (
        target < 0
        || target >= classes!
        || !Number.isInteger(target)
      ) {
        throw new Error(
          `crossEntropy: target ${target} out of range for ${classes} classes`,
        )
      }
      onehot[i * classes! + target] = 1
    }
    mask = fromFlat(onehot, [batch!, classes!])
  }
  return l
    .logSoftmax(1)
    .mul(mask)
    .sum()
    .neg()
    .div(batch!) as any
}
