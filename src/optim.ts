import type { Tensor } from "./tensor.ts";
import * as typegpuBackend from "./backends/typegpu.ts";
import type { TypeGPUStorage } from "./backends/typegpu.ts";

type AnyTensor = Tensor<any, any>;

export abstract class Optimizer {
  constructor(protected params: AnyTensor[]) {
    for (const p of params) if (!p.requiresGrad) throw new Error("Optimizer received a tensor without requires_grad");
  }

  zeroGrad(): void {
    for (const p of this.params) p.zeroGrad();
  }

  dispose(): void {}

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
  private velocities: (Float64Array | TypeGPUStorage)[] | null = null;

  constructor(params: AnyTensor[], options: SGDOptions) {
    super(params);
    this.lr = options.lr;
    this.momentum = options.momentum ?? 0;
    this.weightDecay = options.weightDecay ?? 0;
  }

  step(): void {
    if (this.momentum > 0 && !this.velocities)
      this.velocities = this.params.map((p) =>
        p.device === "gpu"
          ? typegpuBackend.fill((p._storage as TypeGPUStorage).root, p.numel, 0)
          : new Float64Array(p.numel),
      );
    this.params.forEach((p, pi) => {
      const g = p.grad;
      if (!g) return;
      if (p.device === "gpu") {
        typegpuBackend.sgdStep(
          p._storage as TypeGPUStorage,
          g._storage as TypeGPUStorage,
          this.momentum > 0 ? (this.velocities![pi] as TypeGPUStorage) : null,
          this.lr,
          this.momentum,
          this.weightDecay,
        );
        return;
      }
      const data = p.data;
      const gd = g.data;
      for (let i = 0; i < data.length; i++) {
        let grad = gd[i]!;
        if (this.weightDecay !== 0) grad += this.weightDecay * data[i]!;
        if (this.momentum > 0) {
          const v = this.velocities![pi]! as Float64Array;
          v[i] = this.momentum * v[i]! + grad;
          grad = v[i]!;
        }
        data[i]! -= this.lr * grad;
      }
    });
  }

  override dispose(): void {
    for (const velocity of this.velocities ?? [])
      if (!(velocity instanceof Float64Array)) typegpuBackend.dispose(velocity);
    this.velocities = null;
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
  private m: (Float64Array | TypeGPUStorage)[];
  private v: (Float64Array | TypeGPUStorage)[];

  constructor(params: AnyTensor[], options: AdamOptions = {}) {
    super(params);
    this.lr = options.lr ?? 0.001;
    [this.beta1, this.beta2] = options.betas ?? [0.9, 0.999];
    this.eps = options.eps ?? 1e-8;
    this.weightDecay = options.weightDecay ?? 0;
    this.m = params.map((p) =>
      p.device === "gpu"
        ? typegpuBackend.fill((p._storage as TypeGPUStorage).root, p.numel, 0)
        : new Float64Array(p.numel),
    );
    this.v = params.map((p) =>
      p.device === "gpu"
        ? typegpuBackend.fill((p._storage as TypeGPUStorage).root, p.numel, 0)
        : new Float64Array(p.numel),
    );
  }

  step(): void {
    this.t++;
    const bc1 = 1 - this.beta1 ** this.t;
    const bc2 = 1 - this.beta2 ** this.t;
    this.params.forEach((p, pi) => {
      const g = p.grad;
      if (!g) return;
      if (p.device === "gpu") {
        typegpuBackend.adamStep(
          p._storage as TypeGPUStorage,
          g._storage as TypeGPUStorage,
          this.m[pi]! as TypeGPUStorage,
          this.v[pi]! as TypeGPUStorage,
          this.lr,
          this.beta1,
          this.beta2,
          this.eps,
          this.weightDecay,
          bc1,
          bc2,
        );
        return;
      }
      const data = p.data;
      const gd = g.data;
      const m = this.m[pi]! as Float64Array;
      const v = this.v[pi]! as Float64Array;
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

  override dispose(): void {
    for (const state of [...this.m, ...this.v]) if (!(state instanceof Float64Array)) typegpuBackend.dispose(state);
    this.m = [];
    this.v = [];
  }
}
