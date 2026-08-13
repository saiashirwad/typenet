import { noGrad } from "./autograd.ts"
import { _activeUpdateTrace } from "./compile.ts"
import { rawBinary, rawUnary } from "./ir.ts"
import { forceMany, isLazy } from "./lazy.ts"
import { Tensor } from "./tensor.ts"
import type { Tensor as TensorType } from "./tensor.ts"

type AnyTensor = TensorType<any>

type GraphUpdate = { target: AnyTensor; expr: AnyTensor }

function finishGraphUpdates(
  updates: GraphUpdate[],
  grads: AnyTensor[],
): void {
  const trace = _activeUpdateTrace()
  if (trace) {
    trace.updates.push(...updates)
    trace.materialize.push(...grads)
    return
  }
  forceMany(updates.map(u => u.expr))
  for (const u of updates) {
    ;(u.target.data as Float32Array).set(
      u.expr.data as Float32Array,
    )
  }
}

function useGraphStep(p: AnyTensor): boolean {
  return (
    (_activeUpdateTrace() !== null || isLazy())
    && p.dtype === "float32"
  )
}

type Algebra<T> = {
  of(n: number): T
  add(a: T, b: T | number): T
  sub(a: T, b: T): T
  mul(a: T, b: T | number): T
  div(a: T, b: T | number): T
  sqrt(a: T): T
  min1(a: T): T
}

const nums: Algebra<number> = {
  of: n => n,
  add: (a, b) => a + (b as number),
  sub: (a, b) => a - b,
  mul: (a, b) => a * (b as number),
  div: (a, b) => a / (b as number),
  sqrt: Math.sqrt,
  min1: a => Math.min(a, 1),
}

// A file-local untyped algebra over the raw dispatchers: optimizer
// formulas relate shapes the public BroadcastCheck cannot see, and the
// step runs under noGrad, so the tape-attaching typed methods buy
// nothing here.
const asTensor = (v: AnyTensor | number): AnyTensor => typeof v === "number" ? Tensor.scalar(v) as AnyTensor : v

const tensors: Algebra<AnyTensor> = {
  of: n => Tensor.scalar(n),
  add: (a, b) => rawBinary(a, asTensor(b), "add"),
  sub: (a, b) => rawBinary(a, b, "sub"),
  mul: (a, b) => rawBinary(a, asTensor(b), "mul"),
  div: (a, b) => rawBinary(a, asTensor(b), "div"),
  sqrt: a => rawUnary(a, "sqrt"),
  min1: a => rawBinary(a, asTensor(1), "minimum"),
}

function clipScale<T>(
  A: Algebra<T>,
  sumSq: T,
  maxNorm: number,
): T {
  return A.min1(
    A.div(A.of(maxNorm), A.add(A.sqrt(sumSq), 1e-6)),
  )
}

function sgdUpdate<T>(
  A: Algebra<T>,
  p: T,
  g: T,
  velocity: T | null,
  lr: number,
  momentum: number,
  weightDecay: number,
): { nextP: T; nextV: T | null } {
  let grad = g
  if (weightDecay !== 0) {
    grad = A.add(grad, A.mul(p, weightDecay))
  }
  let nextV: T | null = null
  if (momentum > 0 && velocity !== null) {
    nextV = A.add(A.mul(velocity, momentum), grad)
    grad = nextV
  }
  return { nextP: A.sub(p, A.mul(grad, lr)), nextV }
}

function adamUpdate<T>(
  A: Algebra<T>,
  p: T,
  g: T,
  m: T,
  v: T,
  bc1: T | number,
  bc2: T | number,
  lr: number,
  beta1: number,
  beta2: number,
  eps: number,
  weightDecay: number,
): { nextP: T; nextM: T; nextV: T } {
  let grad = g
  if (weightDecay !== 0) {
    grad = A.add(grad, A.mul(p, weightDecay))
  }
  const nextM = A.add(A.mul(m, beta1), A.mul(grad, 1 - beta1))
  const nextV = A.add(
    A.mul(v, beta2),
    A.mul(A.mul(grad, grad), 1 - beta2),
  )
  const mHat = A.div(nextM, bc1)
  const vHat = A.div(nextV, bc2)
  return {
    nextP: A.sub(
      p,
      A.div(A.mul(mHat, lr), A.add(A.sqrt(vHat), eps)),
    ),
    nextM,
    nextV,
  }
}

/**
 * Scale every gradient down so their combined L2 norm is at most
 * `maxNorm`, leaving them alone when it already is.
 *
 * Call it between `backward()` and `step()`. Gradients are rewritten in
 * place, so a following `step()` sees the clipped values — and in
 * lazy/compiled mode the rewrite is a graph expression, so a compiled
 * training step clips with the rest of the step in one pass.
 *
 * Returns the pre-clipping norm as a scalar tensor: a forced value in
 * eager mode, a graph node on the lazy/compile path — so a compiled
 * step clips in-graph and never forces a JS number. `.item()` reads it.
 */
export function clipGradNorm(
  params: AnyTensor[],
  maxNorm: number,
): Tensor<[]> {
  if (!(maxNorm > 0)) {
    throw new Error(
      `clipGradNorm: maxNorm must be positive, got ${maxNorm}`,
    )
  }
  const withGrads = params.filter(p => p.grad)
  if (withGrads.length === 0) return Tensor.scalar(0)
  if (withGrads.every(useGraphStep)) {
    return noGrad(() => {
      let total = withGrads[0]!.grad!.pow(2).sum()
      for (const p of withGrads.slice(1)) {
        total = total.add(p.grad!.pow(2).sum())
      }
      const scale = clipScale(tensors, total, maxNorm)
      for (const p of withGrads) {
        p.grad = tensors.mul(p.grad!, scale) as typeof p.grad
      }
      return total.sqrt() as Tensor<[]>
    })
  }
  let total = 0
  for (const p of withGrads) {
    for (const g of p.grad!.data) total += g * g
  }
  const scale = clipScale(nums, total, maxNorm)
  if (scale < 1) {
    for (const p of withGrads) {
      const data = p.grad!.data
      for (let i = 0; i < data.length; i++) {
        data[i]! *= scale
      }
    }
  }
  return Tensor.scalar(Math.sqrt(total))
}

export abstract class Optimizer {
  constructor(protected params: AnyTensor[]) {
    for (const p of params) {
      if (!p.needsGrad) {
        throw new Error(
          "Optimizer received a tensor without requiresGrad",
        )
      }
    }
  }

  zeroGrad(): void {
    for (const p of this.params) p.zeroGrad()
  }

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
  private graphVelocities: AnyTensor[] | null = null

  constructor(params: AnyTensor[], options: SGDOptions) {
    super(params)
    this.lr = options.lr
    this.momentum = options.momentum ?? 0
    this.weightDecay = options.weightDecay ?? 0
  }

  step(): void {
    if (this.momentum > 0 && !this.velocities) {
      this.velocities = this.params.map(
        p => new Float64Array(p.numel),
      )
    }
    const updates: GraphUpdate[] = []
    const grads: AnyTensor[] = []
    noGrad(() => {
      this.params.forEach((p, pi) => {
        const g = p.grad
        if (!g) return
        if (useGraphStep(p)) {
          let velocity: AnyTensor | null = null
          if (this.momentum > 0) {
            if (!this.graphVelocities) {
              this.graphVelocities = this.params.map(
                q => Tensor.zeros(q.shape) as AnyTensor,
              )
            }
            velocity = this.graphVelocities[pi]!
          }
          const next = sgdUpdate(
            tensors,
            p,
            g as AnyTensor,
            velocity,
            this.lr,
            this.momentum,
            this.weightDecay,
          )
          if (velocity && next.nextV) {
            updates.push({ target: velocity, expr: next.nextV })
          }
          updates.push({ target: p, expr: next.nextP })
          grads.push(g)
          return
        }
        const data = p.data
        const gd = g.data
        const vel = this.momentum > 0
          ? this.velocities![pi]!
          : null
        for (let i = 0; i < data.length; i++) {
          const next = sgdUpdate(
            nums,
            data[i]!,
            gd[i]!,
            vel ? vel[i]! : null,
            this.lr,
            this.momentum,
            this.weightDecay,
          )
          data[i] = next.nextP
          if (vel && next.nextV !== null) vel[i] = next.nextV
        }
      })
    })
    if (updates.length > 0) {
      finishGraphUpdates(updates, grads)
    }
  }

  dispose(): void {
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
    options: AdamOptions = {},
  ) {
    super(params)
    this.lr = options.lr ?? 0.001
    ;[this.beta1, this.beta2] = options.betas ?? [
      0.9,
      0.999,
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
    let graphBc: { one: AnyTensor; two: AnyTensor } | null = null
    const corrections = () => {
      if (graphBc) return graphBc
      if (!this.graphT) {
        this.graphT = Tensor.zeros([]) as AnyTensor
      }
      const next = this.graphT.add(1)
      updates.push({ target: this.graphT, expr: next })
      const correct = (beta: number) => next.mul(Math.log(beta)).exp().neg().add(1)
      graphBc = {
        one: correct(this.beta1),
        two: correct(this.beta2),
      }
      return graphBc
    }
    noGrad(() => {
      this.params.forEach((p, pi) => {
        const g = p.grad
        if (!g) return
        if (useGraphStep(p)) {
          if (!this.graphM) {
            this.graphM = this.params.map(
              q => Tensor.zeros(q.shape) as AnyTensor,
            )
          }
          if (!this.graphV) {
            this.graphV = this.params.map(
              q => Tensor.zeros(q.shape) as AnyTensor,
            )
          }
          const m = this.graphM[pi]!
          const v = this.graphV[pi]!
          const bc = corrections()
          const next = adamUpdate(
            tensors,
            p,
            g as AnyTensor,
            m,
            v,
            bc.one,
            bc.two,
            this.lr,
            this.beta1,
            this.beta2,
            this.eps,
            this.weightDecay,
          )
          updates.push({ target: m, expr: next.nextM })
          updates.push({ target: v, expr: next.nextV })
          updates.push({ target: p, expr: next.nextP })
          grads.push(g)
          return
        }
        const data = p.data
        const gd = g.data
        const m = this.m[pi]!
        const v = this.v[pi]!
        for (let i = 0; i < data.length; i++) {
          const next = adamUpdate(
            nums,
            data[i]!,
            gd[i]!,
            m[i]!,
            v[i]!,
            bc1,
            bc2,
            this.lr,
            this.beta1,
            this.beta2,
            this.eps,
            this.weightDecay,
          )
          m[i] = next.nextM
          v[i] = next.nextV
          data[i] = next.nextP
        }
      })
    })
    if (updates.length > 0) {
      finishGraphUpdates(updates, grads)
    }
  }

  dispose(): void {
    this.m = []
    this.v = []
    this.graphM = null
    this.graphV = null
    this.graphT = null
  }
}
