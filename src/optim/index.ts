import { noGrad } from "../autograd.ts"
import { _activeUpdateTrace, type GraphUpdate } from "../compile.ts"
import { isLazyMode, rawBinary, rawUnary } from "../ir.ts"
import { forceMany } from "../lazy.ts"
import { Module } from "../nn/module.ts"
import { type AnyTensor, Tensor } from "../tensor.ts"

export * from "./schedule.ts"

/** Optimizer state lives in a `Float64Array` (eager) or a graph leaf's `Float32Array` (compiled/lazy). */
type NumericLike = Float64Array | Float32Array

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
    ;(u.target.data as Float32Array).set(u.expr.data as Float32Array)
  }
}

function useGraphStep(p: AnyTensor): boolean {
  return (_activeUpdateTrace() !== null || isLazyMode()) && p.dtype === "float32"
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
  add: (a, b) => a + b,
  sub: (a, b) => a - b,
  mul: (a, b) => a * b,
  div: (a, b) => a / b,
  sqrt: Math.sqrt,
  min1: a => Math.min(a, 1),
}

// A file-local untyped algebra over the raw dispatchers: optimizer formulas relate shapes the public BroadcastCheck cannot see, and the step runs under noGrad.
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

function clipScale<T>(A: Algebra<T>, sumSq: T, maxNorm: number): T {
  return A.min1(A.div(A.of(maxNorm), A.add(A.sqrt(sumSq), 1e-6)))
}

function sgdUpdate<T>(
  A: Algebra<T>,
  p: T,
  g: T,
  velocity: T | null,
  lr: T,
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

/** Shared Adam arithmetic. `decoupled` selects coupled L2, decay folded into the gradient and carried through the moments, or AdamW's decay applied to `p` after the moment step. */
function adamUpdate<T>(
  A: Algebra<T>,
  p: T,
  g: T,
  m: T,
  v: T,
  bc1: T | number,
  bc2: T | number,
  lr: T,
  beta1: number,
  beta2: number,
  eps: number,
  weightDecay: number,
  decoupled: boolean,
): { nextP: T; nextM: T; nextV: T } {
  let grad = g
  if (weightDecay !== 0 && !decoupled) {
    grad = A.add(grad, A.mul(p, weightDecay))
  }
  const nextM = A.add(A.mul(m, beta1), A.mul(grad, 1 - beta1))
  const nextV = A.add(
    A.mul(v, beta2),
    A.mul(A.mul(grad, grad), 1 - beta2),
  )
  const mHat = A.div(nextM, bc1)
  const vHat = A.div(nextV, bc2)
  let nextP = A.sub(
    p,
    A.div(A.mul(mHat, lr), A.add(A.sqrt(vHat), eps)),
  )
  if (weightDecay !== 0 && decoupled) {
    nextP = A.sub(nextP, A.mul(p, A.mul(lr, weightDecay)))
  }
  return { nextP, nextM, nextV }
}

/** Scales gradients so their combined L2 norm is at most `maxNorm`; call it between `backward()` and `step()`. Returns the pre-clipping norm. */
export function clipGradNorm(
  params: AnyTensor[],
  maxNorm: number,
): Tensor<[]> {
  if (!(maxNorm > 0)) {
    throw new Error(`clipGradNorm: maxNorm must be positive, got ${maxNorm}`)
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

/** A flat parameter list, or one or more `Module`s (which adds name-keyed optimizer state and `parameterEpoch` drift detection). */
export type OptimizerSource = AnyTensor[] | Module | readonly Module[]

function isModuleArray(v: OptimizerSource): v is readonly Module[] {
  return Array.isArray(v) && v.every(x => x instanceof Module)
}

function toModules(source: OptimizerSource): readonly Module[] | null {
  if (source instanceof Module) return [source]
  if (isModuleArray(source)) return source
  return null
}

function paramNamesOf(
  modules: readonly Module[] | null,
  params: AnyTensor[],
): string[] {
  if (!modules) return params.map((_, i) => String(i))
  if (modules.length === 1) return [...modules[0]!.namedParameters().keys()]
  const names: string[] = []
  modules.forEach((m, mi) => {
    for (const name of m.namedParameters().keys()) names.push(`${mi}.${name}`)
  })
  return names
}

export interface OptimizerStateEntry {
  readonly velocity?: readonly number[]
  readonly m?: readonly number[]
  readonly v?: readonly number[]
}

/** Name-keyed optimizer state matching `Module.stateDict()`'s keying. `step` is Adam/AdamW's global step count, `undefined` for SGD. */
export interface OptimizerStateDict {
  readonly step?: number
  readonly state: Readonly<Record<string, OptimizerStateEntry>>
}

export abstract class Optimizer {
  protected params: AnyTensor[]
  protected readonly maxGradNorm: number | undefined
  protected readonly paramNames: string[]
  #modules: readonly Module[] | null
  #epochs: number[] | null

  constructor(source: OptimizerSource, maxGradNorm?: number) {
    const modules = toModules(source)
    this.params = modules
      ? modules.flatMap(m => m.parameters() as AnyTensor[])
      : (source as AnyTensor[])
    this.#modules = modules
    this.#epochs = modules ? modules.map(m => m.parameterEpoch) : null
    this.paramNames = paramNamesOf(modules, this.params)
    this.maxGradNorm = maxGradNorm
    for (const p of this.params) {
      if (!p.needsGrad) {
        throw new Error("Optimizer received a tensor without requiresGrad")
      }
      if (p.dtype === "int32" || p.dtype === "int64") {
        throw new Error(
          `Optimizer cannot use ${p.dtype} parameters, parameters must be float32 or float64, integer storage cannot hold gradient updates`,
        )
      }
    }
  }

  protected checkParameterEpoch(): void {
    if (!this.#modules) return
    this.#modules.forEach((m, i) => {
      if (m.parameterEpoch !== this.#epochs![i]) {
        throw new Error(
          `Optimizer: ${m.constructor.name}'s parameter set changed after this `
            + "optimizer was constructed, construct the optimizer only after every "
            + "submodule is registered, not before.",
        )
      }
    })
  }

  protected clipIfNeeded(): void {
    if (this.maxGradNorm !== undefined) {
      clipGradNorm(this.params, this.maxGradNorm)
    }
  }

  zeroGrad(): void {
    for (const p of this.params) p.zeroGrad()
  }

  abstract step(): void
}

/** `lr` is required to mirror PyTorch, where `SGD` has no default but `Adam` does; the asymmetry is intentional. */
export interface SGDOptions {
  lr: number
  momentum?: number
  weightDecay?: number
  maxGradNorm?: number
}

export class SGD extends Optimizer {
  /** Public and mutable; takes effect on the next eager `step()`. Under `compile()` the traced value is baked in at trace time. */
  lr: number
  private readonly momentum: number
  private readonly weightDecay: number
  private velocities: Float64Array[] | null = null
  private graphVelocities: AnyTensor[] | null = null

  constructor(source: OptimizerSource, options: SGDOptions) {
    super(source, options.maxGradNorm)
    this.lr = options.lr
    this.momentum = options.momentum ?? 0
    this.weightDecay = options.weightDecay ?? 0
  }

  step(): void {
    this.checkParameterEpoch()
    this.clipIfNeeded()
    if (this.momentum > 0 && !this.velocities) {
      this.velocities = this.params.map(p => new Float64Array(p.numel))
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
            this.graphVelocities ??= this.params.map(q => Tensor.zeros(q.shape) as AnyTensor)
            velocity = this.graphVelocities[pi]!
          }
          const next = sgdUpdate(
            tensors,
            p,
            g as AnyTensor,
            velocity,
            tensors.of(this.lr),
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
        const vel = this.momentum > 0 ? this.velocities![pi]! : null
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

  private velocityOf(i: number): NumericLike | null {
    if (this.velocities) return this.velocities[i]!
    if (this.graphVelocities) return this.graphVelocities[i]!.data as Float32Array
    return null
  }

  stateDict(): OptimizerStateDict {
    const state: Record<string, OptimizerStateEntry> = {}
    if (this.momentum > 0) {
      this.paramNames.forEach((name, i) => {
        const vel = this.velocityOf(i)
        if (vel) state[name] = { velocity: Array.from(vel) }
      })
    }
    return { state }
  }

  loadStateDict(d: OptimizerStateDict): void {
    if (this.momentum <= 0) return
    if (!this.velocities) {
      this.velocities = this.params.map(p => new Float64Array(p.numel))
    }
    this.paramNames.forEach((name, i) => {
      const entry = d.state[name]
      if (!entry?.velocity) return
      this.velocities![i]!.set(entry.velocity)
      if (this.graphVelocities) {
        ;(this.graphVelocities[i]!.data as Float32Array).set(entry.velocity)
      }
    })
  }

  dispose(): void {
    this.velocities = null
    this.graphVelocities = null
  }
}

/** `lr` is optional to mirror PyTorch, which requires one only for `SGD`; the asymmetry is intentional. */
export interface AdamOptions {
  lr?: number
  betas?: [number, number]
  eps?: number
  weightDecay?: number
  maxGradNorm?: number
}

export class Adam extends Optimizer {
  lr: number
  protected readonly beta1: number
  protected readonly beta2: number
  protected readonly eps: number
  protected readonly weightDecay: number
  /** `false` for `Adam` (coupled L2), `true` for `AdamW` (decoupled decay). */
  protected readonly decoupled: boolean
  private t = 0
  private m: Float64Array[] | null
  private v: Float64Array[] | null
  private graphM: AnyTensor[] | null = null
  private graphV: AnyTensor[] | null = null
  // The in-graph step count must be a leaf, not the host-side `t`: a traced constant would freeze the bias correction at t = 1.
  private graphT: AnyTensor | null = null

  constructor(
    source: OptimizerSource,
    options: AdamOptions = {},
    decoupled = false,
  ) {
    super(source, options.maxGradNorm)
    this.lr = options.lr ?? 0.001
    ;[this.beta1, this.beta2] = options.betas ?? [0.9, 0.999]
    this.eps = options.eps ?? 1e-8
    this.weightDecay = options.weightDecay ?? 0
    this.decoupled = decoupled
    this.m = this.params.map(p => new Float64Array(p.numel))
    this.v = this.params.map(p => new Float64Array(p.numel))
  }

  step(): void {
    this.checkParameterEpoch()
    this.clipIfNeeded()
    this.t++
    const ms = this.m ??= this.params.map(p => new Float64Array(p.numel))
    const vs = this.v ??= this.params.map(p => new Float64Array(p.numel))
    const bc1 = 1 - this.beta1 ** this.t
    const bc2 = 1 - this.beta2 ** this.t
    const updates: GraphUpdate[] = []
    const grads: AnyTensor[] = []
    // Bias corrections as graph expressions of the step-count leaf: beta^t as exp(t·ln beta), built lazily on first graph-path use.
    let graphBc: { one: AnyTensor; two: AnyTensor } | null = null
    const corrections = () => {
      if (graphBc) return graphBc
      this.graphT ??= Tensor.zeros([]) as AnyTensor
      const next = this.graphT.add(1)
      updates.push({ target: this.graphT, expr: next })
      const correct = (beta: number) => next.mul(Math.log(beta)).exp().neg().add(1)
      graphBc = { one: correct(this.beta1), two: correct(this.beta2) }
      return graphBc
    }
    noGrad(() => {
      this.params.forEach((p, pi) => {
        const g = p.grad
        if (!g) return
        if (useGraphStep(p)) {
          this.graphM ??= this.params.map(q => Tensor.zeros(q.shape) as AnyTensor)
          this.graphV ??= this.params.map(q => Tensor.zeros(q.shape) as AnyTensor)
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
            tensors.of(this.lr),
            this.beta1,
            this.beta2,
            this.eps,
            this.weightDecay,
            this.decoupled,
          )
          updates.push({ target: m, expr: next.nextM })
          updates.push({ target: v, expr: next.nextV })
          updates.push({ target: p, expr: next.nextP })
          grads.push(g)
          return
        }
        const data = p.data
        const gd = g.data
        const m = ms[pi]!
        const v = vs[pi]!
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
            this.decoupled,
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

  private mvOf(i: number): { m: NumericLike; v: NumericLike } | null {
    if (this.m && this.v) return { m: this.m[i]!, v: this.v[i]! }
    if (this.graphM && this.graphV) {
      return {
        m: this.graphM[i]!.data as Float32Array,
        v: this.graphV[i]!.data as Float32Array,
      }
    }
    return null
  }

  stateDict(): OptimizerStateDict {
    const state: Record<string, OptimizerStateEntry> = {}
    this.paramNames.forEach((name, i) => {
      const mv = this.mvOf(i)
      if (mv) state[name] = { m: Array.from(mv.m), v: Array.from(mv.v) }
    })
    return { step: this.t, state }
  }

  loadStateDict(d: OptimizerStateDict): void {
    if (d.step !== undefined) {
      this.t = d.step
      if (this.graphT) {
        ;(this.graphT.data as Float32Array)[0] = d.step
      }
    }
    if (!this.m) this.m = this.params.map(p => new Float64Array(p.numel))
    if (!this.v) this.v = this.params.map(p => new Float64Array(p.numel))
    this.paramNames.forEach((name, i) => {
      const entry = d.state[name]
      if (!entry) return
      if (entry.m) {
        this.m![i]!.set(entry.m)
        if (this.graphM) (this.graphM[i]!.data as Float32Array).set(entry.m)
      }
      if (entry.v) {
        this.v![i]!.set(entry.v)
        if (this.graphV) (this.graphV[i]!.data as Float32Array).set(entry.v)
      }
    })
  }

  dispose(): void {
    this.t = 0
    this.m = null
    this.v = null
    this.graphM = null
    this.graphV = null
    this.graphT = null
  }
}

export interface AdamWOptions {
  lr?: number
  betas?: [number, number]
  eps?: number
  /** Decoupled weight decay, applied to the parameter directly rather than folded into the gradient. Defaults to `0.01`, matching PyTorch's `AdamW`. */
  weightDecay?: number
  maxGradNorm?: number
}

export class AdamW extends Adam {
  constructor(source: OptimizerSource, options: AdamWOptions = {}) {
    super(source, { ...options, weightDecay: options.weightDecay ?? 0.01 }, true)
  }
}
