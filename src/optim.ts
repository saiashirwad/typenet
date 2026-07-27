import {
  Tensor,
  _activeUpdateTrace,
  forceMany,
  isLazy,
  noGrad
} from "./tensor.ts"
import type { Tensor as TensorType } from "./tensor.ts"
import * as typegpuBackend from "./backends/typegpu.ts"
import type { TypeGPUStorage } from "./backends/typegpu.ts"

type AnyTensor = TensorType<any, any>

// One in-graph update: `expr` evaluates to the new contents of the
// `target` leaf buffer (a parameter or an optimizer state tensor).
type GraphUpdate = { target: AnyTensor; expr: AnyTensor }

// Finish the graph updates collected during step(): inside a compile()
// trace they are handed to the tracer (becoming extra graph roots that
// replay writes back); otherwise they are forced in one multi-root hop
// and copied back into the leaf buffers in place. Grad tensors are
// registered for materialization so `.grad` reads after a compiled
// step see this step's values.
function finishGraphUpdates(
  updates: GraphUpdate[],
  grads: AnyTensor[]
): void {
  const trace = _activeUpdateTrace()
  if (trace) {
    trace.updates.push(...updates)
    trace.materialize.push(...grads)
    return
  }
  forceMany(updates.map(u => u.expr))
  for (const u of updates)
    (u.target.data as Float32Array).set(
      u.expr.data as Float32Array
    )
}

// The in-graph path covers CPU float32 parameters in lazy mode (or a
// compile() trace, which traces under lazy semantics); everything else
// keeps the original eager/GPU loops.
function useGraphStep(p: AnyTensor): boolean {
  return (
    (_activeUpdateTrace() !== null || isLazy()) &&
    p.device === "cpu" &&
    p.dtype === "float32"
  )
}

export abstract class Optimizer {
  constructor(protected params: AnyTensor[]) {
    for (const p of params)
      if (!p.requiresGrad)
        throw new Error(
          "Optimizer received a tensor without requires_grad"
        )
  }

  zeroGrad(): void {
    for (const p of this.params) p.zeroGrad()
  }

  dispose(): void {}

  abstract step(): void
}

export interface SGDOptions {
  lr: number
  momentum?: number
  weightDecay?: number
}

export class SGD extends Optimizer {
  private readonly lr: number
  private readonly momentum: number
  private readonly weightDecay: number
  private velocities:
    (Float64Array | TypeGPUStorage)[] | null = null
  // Momentum state for the in-graph path: plain CPU leaf tensors whose
  // buffers the update graph reads and rewrites on every step.
  private graphVelocities: AnyTensor[] | null = null

  constructor(params: AnyTensor[], options: SGDOptions) {
    super(params)
    this.lr = options.lr
    this.momentum = options.momentum ?? 0
    this.weightDecay = options.weightDecay ?? 0
  }

  step(): void {
    if (this.momentum > 0 && !this.velocities)
      this.velocities = this.params.map(p =>
        p.device === "gpu" ?
          typegpuBackend.fill(
            (p._storage as TypeGPUStorage).root,
            p.numel,
            0
          )
        : new Float64Array(p.numel)
      )
    const updates: GraphUpdate[] = []
    const grads: AnyTensor[] = []
    noGrad(() => {
      this.params.forEach((p, pi) => {
        const g = p.grad
        if (!g) return
        if (useGraphStep(p)) {
          // Build the update as graph expressions instead of a
          // CPU-side loop over `.data`; finishGraphUpdates forces
          // them (or, inside compile(), hands them to the tracer).
          let grad = g as AnyTensor
          if (this.weightDecay !== 0)
            grad = grad.add(p.mul(this.weightDecay))
          if (this.momentum > 0) {
            if (!this.graphVelocities)
              this.graphVelocities = this.params.map(
                q => Tensor.zeros(q.shape) as AnyTensor
              )
            const v = this.graphVelocities[pi]!
            const nextV = v.mul(this.momentum).add(grad)
            updates.push({ target: v, expr: nextV })
            grad = nextV
          }
          updates.push({
            target: p,
            expr: p.sub(grad.mul(this.lr))
          })
          grads.push(g)
          return
        }
        if (p.device === "gpu") {
          typegpuBackend.sgdStep(
            p._storage as TypeGPUStorage,
            g._storage as TypeGPUStorage,
            this.momentum > 0 ?
              (this.velocities![pi] as TypeGPUStorage)
            : null,
            this.lr,
            this.momentum,
            this.weightDecay
          )
          return
        }
        const data = p.data
        const gd = g.data
        for (let i = 0; i < data.length; i++) {
          let grad = gd[i]!
          if (this.weightDecay !== 0)
            grad += this.weightDecay * data[i]!
          if (this.momentum > 0) {
            const v = this.velocities![pi]! as Float64Array
            v[i] = this.momentum * v[i]! + grad
            grad = v[i]!
          }
          data[i]! -= this.lr * grad
        }
      })
    })
    if (updates.length > 0)
      finishGraphUpdates(updates, grads)
  }

  override dispose(): void {
    for (const velocity of this.velocities ?? [])
      if (!(velocity instanceof Float64Array))
        typegpuBackend.dispose(velocity)
    this.velocities = null
  }
}

export interface AdamOptions {
  lr?: number
  betas?: [number, number]
  eps?: number
  weightDecay?: number
}

export class Adam extends Optimizer {
  private readonly lr: number
  private readonly beta1: number
  private readonly beta2: number
  private readonly eps: number
  private readonly weightDecay: number
  private t = 0
  private m: (Float64Array | TypeGPUStorage)[]
  private v: (Float64Array | TypeGPUStorage)[]
  // First/second moments for the in-graph path, as CPU leaf tensors.
  private graphM: AnyTensor[] | null = null
  private graphV: AnyTensor[] | null = null

  constructor(
    params: AnyTensor[],
    options: AdamOptions = {}
  ) {
    super(params)
    this.lr = options.lr ?? 0.001
    ;[this.beta1, this.beta2] = options.betas ?? [
      0.9, 0.999
    ]
    this.eps = options.eps ?? 1e-8
    this.weightDecay = options.weightDecay ?? 0
    this.m = params.map(p =>
      p.device === "gpu" ?
        typegpuBackend.fill(
          (p._storage as TypeGPUStorage).root,
          p.numel,
          0
        )
      : new Float64Array(p.numel)
    )
    this.v = params.map(p =>
      p.device === "gpu" ?
        typegpuBackend.fill(
          (p._storage as TypeGPUStorage).root,
          p.numel,
          0
        )
      : new Float64Array(p.numel)
    )
  }

  step(): void {
    this.t++
    const bc1 = 1 - this.beta1 ** this.t
    const bc2 = 1 - this.beta2 ** this.t
    const updates: GraphUpdate[] = []
    const grads: AnyTensor[] = []
    noGrad(() => {
      this.params.forEach((p, pi) => {
        const g = p.grad
        if (!g) return
        if (useGraphStep(p)) {
          if (_activeUpdateTrace())
            throw new Error(
              "Adam.step() inside compile() is not supported: bias correction depends on the host-side step count, which a compiled graph cannot carry — use SGD in compiled training steps"
            )
          if (!this.graphM)
            this.graphM = this.params.map(
              q => Tensor.zeros(q.shape) as AnyTensor
            )
          if (!this.graphV)
            this.graphV = this.params.map(
              q => Tensor.zeros(q.shape) as AnyTensor
            )
          let grad = g as AnyTensor
          if (this.weightDecay !== 0)
            grad = grad.add(p.mul(this.weightDecay))
          const m = this.graphM[pi]!
          const v = this.graphV[pi]!
          const nextM = m
            .mul(this.beta1)
            .add(grad.mul(1 - this.beta1))
          const nextV = v
            .mul(this.beta2)
            .add(grad.mul(grad).mul(1 - this.beta2))
          const mHat = nextM.mul(1 / bc1)
          const vHat = nextV.mul(1 / bc2)
          updates.push({ target: m, expr: nextM })
          updates.push({ target: v, expr: nextV })
          updates.push({
            target: p,
            expr: p.sub(
              mHat
                .mul(this.lr)
                .div(vHat.sqrt().add(this.eps))
            )
          })
          grads.push(g)
          return
        }
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
            bc2
          )
          return
        }
        const data = p.data
        const gd = g.data
        const m = this.m[pi]! as Float64Array
        const v = this.v[pi]! as Float64Array
        for (let i = 0; i < data.length; i++) {
          let grad = gd[i]!
          if (this.weightDecay !== 0)
            grad += this.weightDecay * data[i]!
          m[i] =
            this.beta1 * m[i]! + (1 - this.beta1) * grad
          v[i] =
            this.beta2 * v[i]! +
            (1 - this.beta2) * grad * grad
          const mHat = m[i]! / bc1
          const vHat = v[i]! / bc2
          data[i]! -=
            (this.lr * mHat) / (Math.sqrt(vHat) + this.eps)
        }
      })
    })
    if (updates.length > 0)
      finishGraphUpdates(updates, grads)
  }

  override dispose(): void {
    for (const state of [...this.m, ...this.v])
      if (!(state instanceof Float64Array))
        typegpuBackend.dispose(state)
    this.m = []
    this.v = []
  }
}
