
import type { Tensor } from "./tensor.ts";

type AnyTensor = Tensor<any, any>;

export abstract class Optimizer {
  constructor(protected params: AnyTensor[]) {
    for (const p of params)
      if (!p.requiresGrad) throw new Error("Optimizer received a tensor without requires_grad");
  }

  zeroGrad(): void {
    for (const p of this.params) p.zeroGrad();
  }

  abstract step(): void;
}

export interface SGDOptions {
  lr: number;
  momentum?: number;
  weightDecay?: number;
}

export class SGD extends Optimizer {
  private readonly lr: number;
  private readonly momentum: number;
  private readonly weightDecay: number;
  private velocities: Float64Array[] | null = null;

  constructor(params: AnyTensor[], options: SGDOptions) {
    super(params);
    this.lr = options.lr;
    this.momentum = options.momentum ?? 0;
    this.weightDecay = options.weightDecay ?? 0;
  }

  step(): void {
    if (this.momentum > 0 && !this.velocities)
      this.velocities = this.params.map((p) => new Float64Array(p.numel));
    this.params.forEach((p, pi) => {
      const g = p.grad;
      if (!g) return;
      const data = p.data;
      const gd = g.data;
      for (let i = 0; i < data.length; i++) {
        let grad = gd[i]!;
        if (this.weightDecay !== 0) grad += this.weightDecay * data[i]!;
        if (this.momentum > 0) {
          const v = this.velocities![pi]!;
          v[i] = this.momentum * v[i]! + grad;
          grad = v[i]!;
        }
        data[i]! -= this.lr * grad;
      }
    });
  }
}

export interface AdamOptions {
  lr?: number;
  betas?: [number, number];
  eps?: number;
  weightDecay?: number;
}

export class Adam extends Optimizer {
  private readonly lr: number;
  private readonly beta1: number;
  private readonly beta2: number;
  private readonly eps: number;
  private readonly weightDecay: number;
  private t = 0;
  private m: Float64Array[];
  private v: Float64Array[];

  constructor(params: AnyTensor[], options: AdamOptions = {}) {
    super(params);
    this.lr = options.lr ?? 0.001;
    [this.beta1, this.beta2] = options.betas ?? [0.9, 0.999];
    this.eps = options.eps ?? 1e-8;
    this.weightDecay = options.weightDecay ?? 0;
    this.m = params.map((p) => new Float64Array(p.numel));
    this.v = params.map((p) => new Float64Array(p.numel));
  }

  step(): void {
    this.t++;
    const bc1 = 1 - this.beta1 ** this.t;
    const bc2 = 1 - this.beta2 ** this.t;
    this.params.forEach((p, pi) => {
      const g = p.grad;
      if (!g) return;
      const data = p.data;
      const gd = g.data;
      const m = this.m[pi]!;
      const v = this.v[pi]!;
      for (let i = 0; i < data.length; i++) {
        let grad = gd[i]!;
        if (this.weightDecay !== 0) grad += this.weightDecay * data[i]!;
        m[i] = this.beta1 * m[i]! + (1 - this.beta1) * grad;
        v[i] = this.beta2 * v[i]! + (1 - this.beta2) * grad * grad;
        const mHat = m[i]! / bc1;
        const vHat = v[i]! / bc2;
        data[i]! -= (this.lr * mHat) / (Math.sqrt(vHat) + this.eps);
      }
    });
  }
}
