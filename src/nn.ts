import type { DimEq, ErrorMessage, MatMul, MatMulCheck, Shape } from "./shape.ts"
import { fromFlat, Tensor } from "./tensor.ts"

type AnyTensor = Tensor<any>

export abstract class Module {
  parameters(): AnyTensor[] {
    const out: AnyTensor[] = []
    const visit = (value: unknown) => {
      if (value instanceof Tensor) {
        if (value.requiresGrad) out.push(value)
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
      .requires_grad()
    this.bias = options.bias === false
      ? null
      : Tensor.zeros([outFeatures]).requires_grad()
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

export class Softmax extends Activation {
  constructor() {
    super(x => x.softmax(-1 as any) as any)
  }
}

export class Sequential<
  In extends number,
  Out extends number,
> extends Module implements Layer<In, Out> {
  declare readonly inFeatures?: In
  declare readonly outFeatures?: Out

  constructor(readonly layers: Layer<any, any>[]) {
    super()
  }

  forward<B extends number>(
    x: Tensor<[B, In]>,
  ): Tensor<[B, Out]> {
    let h: AnyTensor = x
    for (const layer of this.layers) h = layer.forward(h)
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

type ChainIn<L extends readonly unknown[]> =
    L extends readonly [infer H, ...infer R] ?
      LayerIn<H> extends number ? LayerIn<H>
    : ChainIn<R>
  : number

type ChainOut<
  L extends readonly unknown[],
  Acc extends number = number,
> = L extends readonly [infer H, ...infer R] ? ChainOut<
    R,
    LayerOut<H> extends number ? LayerOut<H> : Acc
  >
  : Acc

export function sequential<
  const L extends readonly Layer<any, any>[],
>(
  ...layers: L & ChainCheck<L>
): Sequential<ChainIn<L>, ChainOut<L>>
export function sequential(
  ...layers: Layer<any, any>[]
): Sequential<any, any> {
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
