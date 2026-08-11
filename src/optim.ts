import {
  Tensor,
  _activeUpdateTrace,
  forceMany,
  isLazy,
  noGrad
} from "./tensor.ts"
import type { Tensor as TensorType } from "./tensor.ts"

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
// keeps the original eager loops.
function useGraphStep(p: AnyTensor): boolean {
  return (
    (_activeUpdateTrace() !== null || isLazy()) &&
    p.dtype === "float32"
  )
}

/**
 * Scale every gradient down so their combined L2 norm is at most
 * `maxNorm`, leaving them alone when it already is. The standard fix for
 * a training run that diverges on the occasional huge gradient.
 *
 * Call it between `backward()` and `step()`. Gradients are rewritten in
 * place, so a following `step()` sees the clipped values — and in
 * lazy/compiled mode the rewrite is a graph expression, so a compiled
 * training step clips with the rest of the step in one pass.
 *
 * Returns the pre-clipping norm in eager mode, where it is known
 * without forcing anything, and `null` otherwise.
 */
export function clipGradNorm(
  params: AnyTensor[],
  maxNorm: number
): number | null {
  if (!(maxNorm > 0))
    throw new Error(
      `clipGradNorm: maxNorm must be positive, got ${maxNorm}`
    )
  const withGrads = params.filter(p => p.grad)
  if (withGrads.length === 0) return null
  if (withGrads.every(useGraphStep))
    return noGrad(() => {
      let total = withGrads[0]!.grad!.pow(2).sum()
      for (const p of withGrads.slice(1))
        total = total.add(p.grad!.pow(2).sum())
      // maxNorm / (norm + 1e-6), never scaling up
      const scale = Tensor.scalar(maxNorm)
        .div(total.sqrt().add(1e-6))
        .minimum(1)
      for (const p of withGrads) p.grad = p.grad!.mul(scale)
      return null
    })
  let total = 0
  for (const p of withGrads)
    for (const g of p.grad!.data) total += g * g
  const norm = Math.sqrt(total)
  const scale = Math.min(maxNorm / (norm + 1e-6), 1)
  if (scale < 1)
    for (const p of withGrads) {
      const data = p.grad!.data
      for (let i = 0; i < data.length; i++)
        data[i]! *= scale
    }
  return norm
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
  private velocities: Float64Array[] | null = null
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
      this.velocities = this.params.map(
        p => new Float64Array(p.numel)
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
        const data = p.data
        const gd = g.data
        for (let i = 0; i < data.length; i++) {
          let grad = gd[i]!
          if (this.weightDecay !== 0)
            grad += this.weightDecay * data[i]!
          if (this.momentum > 0) {
            const v = this.velocities![pi]!
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
  private m: Float64Array[]
  private v: Float64Array[]
  // First/second moments for the in-graph path, as CPU leaf tensors.
  private graphM: AnyTensor[] | null = null
  private graphV: AnyTensor[] | null = null
  // Step count for the in-graph path. It has to be a graph leaf rather
  // than the host-side `t`: a compiled step is traced once, so a
  // trace-time constant would freeze the bias correction at t = 1
  // forever. As a leaf it is read and rewritten per step like any other
  // optimizer state, and the correction is computed in the graph.
  private graphT: AnyTensor | null = null

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
    this.m = params.map(p => new Float64Array(p.numel))
    this.v = params.map(p => new Float64Array(p.numel))
  }

  step(): void {
    this.t++
    const bc1 = 1 - this.beta1 ** this.t
    const bc2 = 1 - this.beta2 ** this.t
    const updates: GraphUpdate[] = []
    const grads: AnyTensor[] = []
    // Bias corrections as graph expressions of the step-count leaf.
    // beta^t becomes exp(t·ln beta), the only way to raise a constant to
    // a tensor power with the ops available. Built lazily, and only when
    // some parameter actually takes the graph path.
    let graphBc: { one: AnyTensor; two: AnyTensor } | null =
      null
    const corrections = () => {
      if (graphBc) return graphBc
      if (!this.graphT)
        this.graphT = Tensor.zeros([]) as AnyTensor
      const next = this.graphT.add(1)
      updates.push({ target: this.graphT, expr: next })
      const correct = (beta: number) =>
        next.mul(Math.log(beta)).exp().neg().add(1)
      graphBc = {
        one: correct(this.beta1),
        two: correct(this.beta2)
      }
      return graphBc
    }
    noGrad(() => {
      this.params.forEach((p, pi) => {
        const g = p.grad
        if (!g) return
        if (useGraphStep(p)) {
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
          const bc = corrections()
          const mHat = nextM.div(bc.one)
          const vHat = nextV.div(bc.two)
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
        const data = p.data
        const gd = g.data
        const m = this.m[pi]!
        const v = this.v[pi]!
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
    this.m = []
    this.v = []
    this.graphM = null
    this.graphV = null
    this.graphT = null
  }
}
