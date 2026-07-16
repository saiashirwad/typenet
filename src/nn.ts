import { Tensor } from "./tensor.ts";
import type { Device, TensorParams } from "./tensor.ts";
import type { DimEq, ErrorMessage, MatMul, MatMulCheck, Shape } from "./shape.ts";

type AnyTensor = Tensor<any, any>;

type GradParams = {
  requires_grad: true;
  device: Device;
  dtype: "float32";
};

export abstract class Module {
  parameters(): AnyTensor[] {
    const out: AnyTensor[] = [];
    const visit = (value: unknown) => {
      if (value instanceof Tensor) {
        if (value.requiresGrad) out.push(value);
      } else if (value instanceof Module) {
        out.push(...value.parameters());
      } else if (Array.isArray(value)) {
        for (const v of value) visit(v);
      }
    };
    for (const value of Object.values(this)) visit(value);
    return out;
  }

  zeroGrad(): void {
    for (const p of this.parameters()) p.zeroGrad();
  }

  gpu(): this {
    for (const [key, value] of Object.entries(this)) {
      if (value instanceof Tensor) (this as any)[key] = value.gpu();
      else if (value instanceof Module) value.gpu();
      else if (Array.isArray(value))
        (this as any)[key] = value.map((item) =>
          item instanceof Tensor ? item.gpu() : item instanceof Module ? item.gpu() : item,
        );
    }
    return this;
  }

  async toCPU(): Promise<this> {
    for (const [key, value] of Object.entries(this)) {
      if (value instanceof Tensor) (this as any)[key] = await value.toCPU();
      else if (value instanceof Module) await value.toCPU();
      else if (Array.isArray(value))
        (this as any)[key] = await Promise.all(
          value.map((item) => (item instanceof Tensor ? item.toCPU() : item instanceof Module ? item.toCPU() : item)),
        );
    }
    return this;
  }
}

export class Linear<In extends number, Out extends number> extends Module {
  readonly weight: Tensor<[In, Out], GradParams>;
  readonly bias: Tensor<[Out], GradParams> | null;
  readonly inFeatures: In;
  readonly outFeatures: Out;

  constructor(inFeatures: In, outFeatures: Out, options: { bias?: boolean } = {}) {
    super();
    this.inFeatures = inFeatures;
    this.outFeatures = outFeatures;
    const k = 1 / Math.sqrt(inFeatures);
    this.weight = Tensor.rand([inFeatures, outFeatures])
      .mul(2 * k)
      .sub(k)
      .detach()
      .requires_grad() as any;
    this.bias = options.bias === false ? null : (Tensor.zeros([outFeatures]).requires_grad() as any);
  }

  forward<S extends Shape, P extends TensorParams>(
    x: Tensor<S, P> & MatMulCheck<S, [In, Out]>,
  ): Tensor<MatMul<S, [In, Out]>, P> {
    const y = (x as AnyTensor).matmul(this.weight);
    return (this.bias ? y.add(this.bias) : y) as any;
  }
}

export interface Layer<In extends number, Out extends number> {
  readonly inFeatures?: In;
  readonly outFeatures?: Out;

  forward<B extends number, P extends TensorParams>(x: Tensor<[B, NoInfer<In>], P>): Tensor<[B, NoInfer<Out>], P>;
}

export class ReLU extends Module {
  forward<S extends Shape, P extends TensorParams>(x: Tensor<S, P>): Tensor<S, P> {
    return x.relu();
  }
}

export class LeakyReLU extends Module {
  constructor(private negativeSlope = 0.01) {
    super();
  }
  forward<S extends Shape, P extends TensorParams>(x: Tensor<S, P>): Tensor<S, P> {
    return x.leakyRelu(this.negativeSlope);
  }
}

export class Tanh extends Module {
  forward<S extends Shape, P extends TensorParams>(x: Tensor<S, P>): Tensor<S, P> {
    return x.tanh();
  }
}

export class Sigmoid extends Module {
  forward<S extends Shape, P extends TensorParams>(x: Tensor<S, P>): Tensor<S, P> {
    return x.sigmoid();
  }
}

export class Softmax extends Module {
  forward<S extends Shape, P extends TensorParams>(x: Tensor<S, P>): Tensor<S, P> {
    return x.softmax(-1 as any) as any;
  }
}

export class Sequential<In extends number, Out extends number> extends Module implements Layer<In, Out> {
  declare readonly inFeatures?: In;
  declare readonly outFeatures?: Out;

  constructor(readonly layers: Layer<any, any>[]) {
    super();
  }

  forward<B extends number, P extends TensorParams>(x: Tensor<[B, In], P>): Tensor<[B, Out], P> {
    let h: AnyTensor = x;
    for (const layer of this.layers) h = layer.forward(h);
    return h as any;
  }
}

type LayerIn<L> = L extends { readonly inFeatures?: infer I }
  ? NonNullable<I> extends number
    ? NonNullable<I>
    : undefined
  : undefined;

type LayerOut<L> = L extends { readonly outFeatures?: infer O }
  ? NonNullable<O> extends number
    ? NonNullable<O>
    : undefined
  : undefined;

type NextDim<H, Prev> = LayerOut<H> extends number ? LayerOut<H> : Prev;

type ChainCheck<L extends readonly unknown[], Prev extends number | undefined = undefined> = L extends readonly [
  infer H,
  ...infer R,
]
  ? LayerIn<H> extends infer I
    ? I extends number
      ? Prev extends number
        ? DimEq<Prev, I> extends true
          ? ChainCheck<R, NextDim<H, Prev>>
          : ErrorMessage<`sequential: layer expects ${I} input features but the previous layer outputs ${Prev}`>
        : ChainCheck<R, NextDim<H, Prev>>
      : ChainCheck<R, NextDim<H, Prev>>
    : never
  : unknown;

type ChainIn<L extends readonly unknown[]> = L extends readonly [infer H, ...infer R]
  ? LayerIn<H> extends number
    ? LayerIn<H>
    : ChainIn<R>
  : number;

type ChainOut<L extends readonly unknown[], Acc extends number = number> = L extends readonly [infer H, ...infer R]
  ? ChainOut<R, LayerOut<H> extends number ? LayerOut<H> : Acc>
  : Acc;

export function sequential<const L extends readonly Layer<any, any>[]>(
  ...layers: L & ChainCheck<L>
): Sequential<ChainIn<L>, ChainOut<L>>;
export function sequential(...layers: Layer<any, any>[]): Sequential<any, any> {
  let prevOut: number | undefined;
  layers.forEach((l, i) => {
    if (prevOut !== undefined && l.inFeatures !== undefined && l.inFeatures !== prevOut)
      throw new Error(
        `sequential: layer ${i} expects ${l.inFeatures} features but the previous layer outputs ${prevOut}`,
      );
    if (l.outFeatures !== undefined) prevOut = l.outFeatures;
    else if (l.inFeatures !== undefined) prevOut = l.inFeatures;
  });
  return new Sequential(layers);
}

export function mseLoss<S extends Shape, P extends TensorParams>(
  prediction: Tensor<S, P>,
  target: Tensor<S, any>,
): Tensor<[], P> {
  return (prediction as AnyTensor)
    .sub(target as AnyTensor)
    .pow(2)
    .mean() as any;
}

export function crossEntropy<B extends number, C extends number, P extends TensorParams>(
  logits: Tensor<[B, C], P>,
  targets: readonly number[] | Tensor<[B], any>,
): Tensor<[], P> {
  const l = logits as AnyTensor;
  const [batch, classes] = l.shape as number[];
  let mask: AnyTensor;
  if (targets instanceof Tensor) {
    if (targets.numel !== batch) throw new Error(`crossEntropy: ${targets.numel} targets for batch of ${batch}`);
    if (targets.device !== l.device)
      throw new Error(`crossEntropy: logits are on ${l.device} but targets are on ${targets.device}`);
    mask = targets.oneHot(classes!);
  } else {
    if (targets.length !== batch) throw new Error(`crossEntropy: ${targets.length} targets for batch of ${batch}`);
    const onehot = new Float32Array(batch! * classes!);
    for (let i = 0; i < batch!; i++) {
      const target = targets[i]!;
      if (target < 0 || target >= classes! || !Number.isInteger(target))
        throw new Error(`crossEntropy: target ${target} out of range for ${classes} classes`);
      onehot[i * classes! + target] = 1;
    }
    mask = fromFlat(onehot, [batch!, classes!]);
    if (l.device === "gpu") mask = mask.gpu();
  }
  return l.logSoftmax(1).mul(mask).sum().neg().div(batch!) as any;
}

function fromFlat(data: Float32Array, shape: number[]): AnyTensor {
  const t = Tensor.zeros(shape as [number, number]);
  t.write(data);
  return t as AnyTensor;
}
