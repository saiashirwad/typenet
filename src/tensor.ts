import { Operator } from "tsover-runtime"
import { type GradNode, noGrad, tapeOrder, withGrad } from "./autograd.ts"
import { _activeUpdateTrace, tensorNames } from "./compile.ts"
import {
  rawBinary,
  rawBroadcastTo,
  rawCat,
  rawIndexSelect,
  rawMatmul,
  rawNarrow,
  rawOneHot,
  rawPermute,
  rawReduce,
  rawReduceAll,
  rawScatterAdd,
  rawSum,
  rawUnary,
  reshapeRaw,
  sumTo,
} from "./eager.ts"
import { eagerly, force, forceMany, lazyMode } from "./lazy.ts"
import type {
  Broadcast,
  BroadcastCheck,
  Cat,
  CatCheck,
  DimCheck,
  ErrorMessage,
  InferShape,
  MatMul,
  MatMulCheck,
  NestedArray,
  Permute,
  PermuteCheck,
  ReduceDim,
  ResizeDim,
  ResolveView,
  Shape,
  Squeeze,
  SqueezeDim,
  SqueezeDimCheck,
  Stack,
  Transpose,
  TransposeCheck,
  Unsqueeze,
  UnsqueezeCheck,
  ViewCheck,
} from "./shape.ts"
import {
  arrayCtor,
  contiguousStrides,
  type DType,
  normalizeDim,
  prod,
  shapesEqual,
  showShape,
  type TensorStorage,
  type TypedArray,
} from "./storage.ts"

export type TensorParams = {
  requires_grad: boolean
  dtype: DType
}

export type DefaultParams = {
  requires_grad: false
  dtype: "float32"
}

type Clean<T> = { [K in keyof T]: T[K] } & unknown

type Merge<A, B> = Clean<
  {
    [K in keyof A as K extends keyof B ? never : K]: A[K]
  } & B
>

export type ShapeOf<T> = T extends Tensor<infer S, any> ? S : never
export type ParamsOf<T> = T extends Tensor<any, infer P> ? P : never

export type NestedNumbers = number | readonly NestedNumbers[]

export type AnyTensor = Tensor<any, any>

const INTERNAL = Symbol("tensor-internal")

function flatten(
  value: NestedNumbers,
  out: number[],
  shape: number[],
  depth: number,
): void {
  if (typeof value === "number") {
    if (depth !== shape.length) {
      throw new Error(
        "Ragged nested array passed to tensor()",
      )
    }
    out.push(value)
    return
  }
  if (depth === shape.length) shape.push(value.length)
  else if (shape[depth] !== value.length) {
    throw new Error(
      "Ragged nested array passed to tensor()",
    )
  }
  for (const v of value) flatten(v, out, shape, depth + 1)
}

export function makeRaw(
  data: TypedArray,
  shape: readonly number[],
  dtype: DType,
): AnyTensor {
  return makeStorage({ kind: "cpu", data }, shape, dtype)
}

/** CPU tensor from a flat buffer, using the buffer as storage when it already matches `dtype`. */
export function fromFlat(
  data: ArrayLike<number>,
  shape: readonly number[],
  dtype: DType = "float32",
): AnyTensor {
  const ctor = arrayCtor(dtype)
  const buf = data instanceof ctor ? data : ctor.from(data)
  return makeRaw(buf, shape, dtype)
}

export function makeStorage(
  storage: TensorStorage,
  shape: readonly number[],
  dtype: DType,
): AnyTensor {
  return new (Tensor as any)(
    storage,
    [...shape],
    dtype,
    INTERNAL,
  )
}

type Dim0<S extends Shape> = S extends [infer A extends number, ...any[]] ? A : never
type Dim1<S extends Shape> = S extends [any, infer B extends number, ...any[]] ? B : never

type StackCheck<T extends readonly AnyTensor[]> = T[number] extends Tensor<ShapeOf<T[0]>, any> ? unknown
  : ErrorMessage<"stack: all tensors must have the same shape">

export class Tensor<
  S extends Shape = number[],
  P extends TensorParams = DefaultParams,
> {
  readonly _storage: TensorStorage
  readonly shape: S
  readonly dtype: DType

  grad: Tensor<S, P> | null = null

  requiresGrad = false

  gradNode: GradNode | null = null

  constructor(
    storage: TensorStorage,
    shape: number[],
    dtype: DType,
    internal: typeof INTERNAL,
  ) {
    if (internal !== INTERNAL) {
      throw new Error(
        "Use Tensor.of / zeros / ones / randn to create tensors",
      )
    }
    const length = storage.kind === "cpu"
      ? storage.data.length
      : prod(storage.node.shape)
    if (length !== prod(shape)) {
      throw new Error(
        `Data length ${length} does not match shape ${showShape(shape)}`,
      )
    }
    this._storage = storage
    this.shape = shape as S
    this.dtype = dtype
  }

  get data(): TypedArray {
    if (this._storage.kind === "lazy") {
      return force(this as AnyTensor).data
    }
    return this._storage.data
  }

  get needsGrad(): boolean {
    return this.requiresGrad || this.gradNode !== null
  }

  get rank(): S["length"] {
    return this.shape.length
  }

  get numel(): number {
    return prod(this.shape)
  }

  static of<const V extends NestedNumbers>(
    value: V,
  ): Tensor<InferShape<V>, DefaultParams> {
    const flat: number[] = []
    const shape: number[] = []
    flatten(value, flat, shape, 0)
    return makeRaw(
      Float32Array.from(flat),
      shape,
      "float32",
    ) as any
  }

  static full<const Sh extends Shape>(
    shape: Sh,
    value: number,
  ): Tensor<Sh, DefaultParams> {
    const data = new Float32Array(prod(shape)).fill(value)
    return makeRaw(data, shape, "float32") as any
  }

  static zeros<const Sh extends Shape>(
    shape: Sh,
  ): Tensor<Sh, DefaultParams> {
    return Tensor.full(shape, 0)
  }

  static ones<const Sh extends Shape>(
    shape: Sh,
  ): Tensor<Sh, DefaultParams> {
    return Tensor.full(shape, 1)
  }

  static rand<const Sh extends Shape>(
    shape: Sh,
  ): Tensor<Sh, DefaultParams> {
    const data = new Float32Array(prod(shape))
    for (let i = 0; i < data.length; i++) {
      data[i] = Math.random()
    }
    return makeRaw(data, shape, "float32") as any
  }

  static randn<const Sh extends Shape>(
    shape: Sh,
  ): Tensor<Sh, DefaultParams> {
    const data = new Float32Array(prod(shape))
    for (let i = 0; i < data.length; i += 2) {
      const u = 1 - Math.random()
      const v = Math.random()
      const r = Math.sqrt(-2 * Math.log(u))
      data[i] = r * Math.cos(2 * Math.PI * v)
      if (i + 1 < data.length) {
        data[i + 1] = r * Math.sin(2 * Math.PI * v)
      }
    }
    return makeRaw(data, shape, "float32") as any
  }

  static eye<const N extends number>(
    n: N,
  ): Tensor<[N, N], DefaultParams> {
    const data = new Float32Array(n * n)
    for (let i = 0; i < n; i++) data[i * n + i] = 1
    return makeRaw(data, [n, n], "float32") as any
  }

  static arange<const N extends number>(
    n: N,
  ): Tensor<[N], DefaultParams> {
    const data = new Float32Array(n)
    for (let i = 0; i < n; i++) data[i] = i
    return makeRaw(data, [n], "float32") as any
  }

  static scalar(value: number): Tensor<[], DefaultParams> {
    return makeRaw(
      Float32Array.of(value),
      [],
      "float32",
    ) as any
  }

  requires_grad(): Tensor<
    S,
    Merge<P, { requires_grad: true }>
  > {
    this.requiresGrad = true
    return this as any
  }

  no_grad(): Tensor<S, Merge<P, { requires_grad: false }>> {
    this.requiresGrad = false
    return this as any
  }

  to<const D extends DType>(
    dtype: D,
  ): Tensor<S, Merge<P, { dtype: D }>> {
    if (dtype === this.dtype) return this as any
    const data = arrayCtor(dtype).from(this.data)
    return makeMovedData(this, data, dtype) as any
  }

  item(): number {
    if (this.numel !== 1) {
      throw new Error(
        `item() requires a one-element tensor, got shape ${showShape(this.shape)}`,
      )
    }
    return this.data[0]!
  }

  get(...indices: number[]): number {
    if (indices.length !== this.shape.length) {
      throw new Error(
        `get() expects ${this.shape.length} indices, got ${indices.length}`,
      )
    }
    const strides = contiguousStrides(this.shape)
    let off = 0
    for (let i = 0; i < indices.length; i++) {
      const idx = normalizeDim(indices[i]!, this.shape[i]!)
      off += idx * strides[i]!
    }
    return this.data[off]!
  }

  toArray(): NestedArray<S> {
    const strides = contiguousStrides(this.shape)
    const build = (dim: number, offset: number): any => {
      if (dim === this.shape.length) {
        return this.data[offset]!
      }
      const stride = strides[dim]!
      const out = new Array(this.shape[dim]!)
      for (let i = 0; i < this.shape[dim]!; i++) {
        out[i] = build(dim + 1, offset + i * stride)
      }
      return out
    }
    return build(0, 0)
  }

  toString(): string {
    return `Tensor(shape=${showShape(this.shape)}, dtype=${this.dtype}, data=${JSON.stringify(this.toArray())})`
  }

  detach(): Tensor<S, Merge<P, { requires_grad: false }>> {
    return makeStorage(
      this._storage,
      this.shape,
      this.dtype,
    ) as any
  }

  clone(): Tensor<S, P> {
    force(this as AnyTensor)
    const t = makeRaw(
      this.data.slice(),
      this.shape,
      this.dtype,
    )
    return withGrad(t, "clone", [this], g => [g]) as any
  }

  // Debug label, shown by printGraph(). Metadata only — no effect on
  // computation, autograd, or graph semantics.
  named(name: string): this {
    tensorNames.set(this as AnyTensor, name)
    return this
  }

  backward(gradient?: Tensor<S, any>): void {
    if (!this.needsGrad) {
      throw new Error(
        "backward() on a tensor that does not require grad",
      )
    }
    let seed: AnyTensor
    const lazyPath = lazyMode && this._storage.kind === "lazy"
    if (gradient) {
      if (!shapesEqual(gradient.shape, this.shape)) {
        throw new Error(
          `backward() gradient shape ${showShape(gradient.shape)} does not match ${showShape(this.shape)}`,
        )
      }
      seed = lazyPath
        ? (gradient as AnyTensor)
        : force(gradient as AnyTensor)
    } else {
      if (this.numel !== 1) {
        throw new Error(
          "backward() without a gradient requires a scalar output",
        )
      }
      seed = makeRaw(
        new (arrayCtor(this.dtype))(this.numel).fill(1),
        this.shape,
        this.dtype,
      )
    }

    const topo = tapeOrder(this as AnyTensor)

    const grads = new Map<AnyTensor, AnyTensor>()
    grads.set(this, seed)

    const walk = () =>
      noGrad(() => {
        for (let i = topo.length - 1; i >= 0; i--) {
          const t = topo[i]!
          const g = grads.get(t)
          if (!g) continue
          if (t.gradNode) {
            const inputGrads = t.gradNode.backward(g)
            t.gradNode.inputs.forEach((input, j) => {
              const ig = inputGrads[j]
              if (!ig || !input.needsGrad) return
              const existing = grads.get(input)
              grads.set(
                input,
                existing
                  ? rawBinary(existing, ig, "add")
                  : ig,
              )
            })
          } else if (t.requiresGrad) {
            t.grad = t.grad
              ? (rawBinary(t.grad, g, "add") as any)
              : (g as any)
          }
        }
      })

    if (lazyPath) {
      walk()
      // Materialize the whole forward topo plus every parameter grad
      // in a single multi-root forcing point (one native FFI hop when
      // native is enabled). The forward tensors are forced too so that
      // values read after an in-place optimizer step (which mutates
      // the leaf parameter data) still see pre-step values, matching
      // eager and phase-1 lazy semantics. During a compile() trace the
      // forcing is deferred: the lazy grads stay graph expressions and
      // the traced optimizer updates (plus the grads themselves) become
      // extra roots of the compiled graph, materialized on replay.
      if (!_activeUpdateTrace()) {
        forceMany([
          ...topo,
          ...topo
            .filter(t => t.requiresGrad && t.grad)
            .map(t => t.grad as AnyTensor),
        ])
      }
      return
    }

    for (const t of topo) force(t)
    eagerly(walk)
  }

  zeroGrad(): void {
    this.grad = null
  }

  add(other: number): Tensor<S, P>
  add(
    this: Tensor<[Dim0<S>, 1], P>,
    other: Tensor<[1, Dim0<S>], any>,
  ): Tensor<[Dim0<S>, Dim0<S>], P>
  add(
    this: Tensor<[1, Dim1<S>], P>,
    other: Tensor<[Dim1<S>, 1], any>,
  ): Tensor<[Dim1<S>, Dim1<S>], P>
  add<S2 extends Shape>(
    other: Tensor<S2, any> & BroadcastCheck<S, S2>,
  ): Tensor<Broadcast<S, S2>, P>
  add(other: AnyTensor | number): AnyTensor {
    const b = coerce(other, this)
    const out = rawBinary(this, b, "add")
    return withGrad(out, "add", [this, b], g => [
      sumTo(g, this.shape),
      sumTo(g, b.shape),
    ])
  }

  sub(other: number): Tensor<S, P>
  sub(
    this: Tensor<[Dim0<S>, 1], P>,
    other: Tensor<[1, Dim0<S>], any>,
  ): Tensor<[Dim0<S>, Dim0<S>], P>
  sub(
    this: Tensor<[1, Dim1<S>], P>,
    other: Tensor<[Dim1<S>, 1], any>,
  ): Tensor<[Dim1<S>, Dim1<S>], P>
  sub<S2 extends Shape>(
    other: Tensor<S2, any> & BroadcastCheck<S, S2>,
  ): Tensor<Broadcast<S, S2>, P>
  sub(other: AnyTensor | number): AnyTensor {
    const b = coerce(other, this)
    const out = rawBinary(this, b, "sub")
    return withGrad(out, "sub", [this, b], g => [
      sumTo(g, this.shape),
      sumTo(rawUnary(g, "neg"), b.shape),
    ])
  }

  mul(other: number): Tensor<S, P>
  mul(
    this: Tensor<[Dim0<S>, 1], P>,
    other: Tensor<[1, Dim0<S>], any>,
  ): Tensor<[Dim0<S>, Dim0<S>], P>
  mul(
    this: Tensor<[1, Dim1<S>], P>,
    other: Tensor<[Dim1<S>, 1], any>,
  ): Tensor<[Dim1<S>, Dim1<S>], P>
  mul<S2 extends Shape>(
    other: Tensor<S2, any> & BroadcastCheck<S, S2>,
  ): Tensor<Broadcast<S, S2>, P>
  mul(other: AnyTensor | number): AnyTensor {
    const b = coerce(other, this)
    const out = rawBinary(this, b, "mul")
    return withGrad(out, "mul", [this, b], g => [
      sumTo(rawBinary(g, b, "mul"), this.shape),
      sumTo(rawBinary(g, this, "mul"), b.shape),
    ])
  }

  div(other: number): Tensor<S, P>
  div(
    this: Tensor<[Dim0<S>, 1], P>,
    other: Tensor<[1, Dim0<S>], any>,
  ): Tensor<[Dim0<S>, Dim0<S>], P>
  div(
    this: Tensor<[1, Dim1<S>], P>,
    other: Tensor<[Dim1<S>, 1], any>,
  ): Tensor<[Dim1<S>, Dim1<S>], P>
  div<S2 extends Shape>(
    other: Tensor<S2, any> & BroadcastCheck<S, S2>,
  ): Tensor<Broadcast<S, S2>, P>
  div(other: AnyTensor | number): AnyTensor {
    const b = coerce(other, this)
    const out = rawBinary(this, b, "div")
    return withGrad(out, "div", [this, b], g => [
      sumTo(rawBinary(g, b, "div"), this.shape),
      sumTo(
        rawBinary(rawBinary(g, out, "mul"), b, "negDiv"),
        b.shape,
      ),
    ])
  }

  /**
   * Gradient goes wholly to whichever operand won;
   * ties go to the left one.
   */
  maximum(other: number): Tensor<S, P>
  maximum<S2 extends Shape>(
    other: Tensor<S2, any> & BroadcastCheck<S, S2>,
  ): Tensor<Broadcast<S, S2>, P>
  maximum(other: AnyTensor | number): AnyTensor {
    const b = coerce(other, this)
    const out = rawBinary(this, b, "maximum")
    return withGrad(out, "maximum", [this, b], g => [
      sumTo(
        rawBinary(g, rawBinary(this, b, "ge"), "mul"),
        this.shape,
      ),
      sumTo(
        rawBinary(g, rawBinary(this, b, "lt"), "mul"),
        b.shape,
      ),
    ])
  }

  /** Ties go to the left operand. */
  minimum(other: number): Tensor<S, P>
  minimum<S2 extends Shape>(
    other: Tensor<S2, any> & BroadcastCheck<S, S2>,
  ): Tensor<Broadcast<S, S2>, P>
  minimum(other: AnyTensor | number): AnyTensor {
    const b = coerce(other, this)
    const out = rawBinary(this, b, "minimum")
    return withGrad(out, "minimum", [this, b], g => [
      sumTo(
        rawBinary(g, rawBinary(this, b, "le"), "mul"),
        this.shape,
      ),
      sumTo(
        rawBinary(g, rawBinary(this, b, "gt"), "mul"),
        b.shape,
      ),
    ])
  }

  /**
   * Clamp into `[min, max]`; pass `null` for an open end. Gradient is 1
   * inside the range and 0 outside.
   */
  clamp(
    min: number | null,
    max: number | null = null,
  ): Tensor<S, P> {
    let out = this as AnyTensor
    if (min !== null) out = out.maximum(min)
    if (max !== null) out = out.minimum(max)
    return out as any
  }

  // Comparisons produce 1.0 / 0.0 masks and stop gradients: a step
  // function has zero derivative wherever it is differentiable.
  gt(other: number): Tensor<S, P>
  gt<S2 extends Shape>(
    other: Tensor<S2, any> & BroadcastCheck<S, S2>,
  ): Tensor<Broadcast<S, S2>, P>
  gt(other: AnyTensor | number): AnyTensor {
    return this.compare(other, "gt")
  }

  ge(other: number): Tensor<S, P>
  ge<S2 extends Shape>(
    other: Tensor<S2, any> & BroadcastCheck<S, S2>,
  ): Tensor<Broadcast<S, S2>, P>
  ge(other: AnyTensor | number): AnyTensor {
    return this.compare(other, "ge")
  }

  lt(other: number): Tensor<S, P>
  lt<S2 extends Shape>(
    other: Tensor<S2, any> & BroadcastCheck<S, S2>,
  ): Tensor<Broadcast<S, S2>, P>
  lt(other: AnyTensor | number): AnyTensor {
    return this.compare(other, "lt")
  }

  le(other: number): Tensor<S, P>
  le<S2 extends Shape>(
    other: Tensor<S2, any> & BroadcastCheck<S, S2>,
  ): Tensor<Broadcast<S, S2>, P>
  le(other: AnyTensor | number): AnyTensor {
    return this.compare(other, "le")
  }

  eq(other: number): Tensor<S, P>
  eq<S2 extends Shape>(
    other: Tensor<S2, any> & BroadcastCheck<S, S2>,
  ): Tensor<Broadcast<S, S2>, P>
  eq(other: AnyTensor | number): AnyTensor {
    return this.compare(other, "eq")
  }

  private compare(
    other: AnyTensor | number,
    op: "gt" | "ge" | "lt" | "le" | "eq",
  ): AnyTensor {
    return rawBinary(this, coerce(other, this), op)
  }

  pow(exponent: number): Tensor<S, P> {
    const out = rawUnary(this, "pow", exponent)
    return withGrad(out, "pow", [this], g => [
      rawBinary(
        g,
        rawUnary(this, "scalePowGrad", exponent),
        "mul",
      ),
    ]) as any
  }

  neg(): Tensor<S, P> {
    const out = rawUnary(this, "neg")
    return withGrad(out, "neg", [this], g => [
      rawUnary(g, "neg"),
    ]) as any
  }

  exp(): Tensor<S, P> {
    const out = rawUnary(this, "exp")
    return withGrad(out, "exp", [this], g => [
      rawBinary(g, out, "mul"),
    ]) as any
  }

  log(): Tensor<S, P> {
    const out = rawUnary(this, "log")
    return withGrad(out, "log", [this], g => [
      rawBinary(g, this, "div"),
    ]) as any
  }

  sqrt(): Tensor<S, P> {
    const out = rawUnary(this, "sqrt")
    return withGrad(out, "sqrt", [this], g => [
      rawBinary(g, out, "halfDiv"),
    ]) as any
  }

  abs(): Tensor<S, P> {
    const out = rawUnary(this, "abs")
    return withGrad(out, "abs", [this], g => [
      rawBinary(g, this, "mulSign"),
    ]) as any
  }

  relu(): Tensor<S, P> {
    const out = rawUnary(this, "relu")
    return withGrad(out, "relu", [this], g => [
      rawBinary(g, this, "reluGrad"),
    ]) as any
  }

  leakyRelu(negativeSlope = 0.01): Tensor<S, P> {
    const out = rawUnary(this, "leakyRelu", negativeSlope)
    return withGrad(out, "leakyRelu", [this], g => [
      rawBinary(g, this, "leakyReluGrad", negativeSlope),
    ]) as any
  }

  sigmoid(): Tensor<S, P> {
    const out = rawUnary(this, "sigmoid")
    return withGrad(out, "sigmoid", [this], g => [
      rawBinary(g, out, "sigmoidGrad"),
    ]) as any
  }

  tanh(): Tensor<S, P> {
    const out = rawUnary(this, "tanh")
    return withGrad(out, "tanh", [this], g => [
      rawBinary(g, out, "tanhGrad"),
    ]) as any
  }

  softmax<D extends number>(
    dim: D & DimCheck<S, D>,
  ): Tensor<S, P> {
    const { e } = this.softmaxShift(dim as number)
    return e.div(e.sum(dim as any, true) as any) as any
  }

  logSoftmax<D extends number>(
    dim: D & DimCheck<S, D>,
  ): Tensor<S, P> {
    const { shifted, e } = this.softmaxShift(dim as number)
    return shifted.sub(
      e.sum(dim as any, true).log() as any,
    ) as any
  }

  private softmaxShift(
    dim: number,
  ): { shifted: AnyTensor; e: AnyTensor } {
    const shifted = this.sub(
      this.max(dim as any, true).detach() as any,
    ) as AnyTensor
    return { shifted, e: shifted.exp() }
  }

  matmul<S2 extends Shape>(
    other: Tensor<S2, any> & MatMulCheck<S, S2>,
  ): Tensor<MatMul<S, S2>, P> {
    const self = this as AnyTensor
    const b = other as AnyTensor
    if (self.rank === 0 || b.rank === 0) {
      throw new Error("matmul requires rank >= 1 operands")
    }
    const A = self.rank === 1 ? self.unsqueeze(0) : self
    const B = b.rank === 1 ? b.unsqueeze(-1) : b
    let out = matmul2(A, B)
    if (b.rank === 1) out = out.squeezeDim(-1)
    if (self.rank === 1) {
      out = out.squeezeDim((b.rank === 1 ? -1 : -2) as any)
    }
    return out as any
  }

  dot<S2 extends Shape>(
    other: Tensor<S2, any> & MatMulCheck<S, S2>,
  ): Tensor<MatMul<S, S2>, P> {
    return this.matmul(other)
  }

  sum(): Tensor<[], P>
  sum<D extends number>(
    dim: D & DimCheck<S, D>,
  ): Tensor<ReduceDim<S, D>, P>
  sum<D extends number, const K extends boolean>(
    dim: D & DimCheck<S, D>,
    keepdim: K,
  ): Tensor<ReduceDim<S, D, K>, P>
  sum(dim?: number, keepdim = false): AnyTensor {
    if (dim === undefined) {
      const out = rawReduceAll(this, "sum")
      return withGrad(out, "sum", [this], g => [
        rawBroadcastTo(g, [...this.shape]),
      ])
    }
    const d = normalizeDim(dim, this.shape.length)
    const out = rawSum(this, d, keepdim)
    const keepShape = this.shape.map((s, i) => i === d ? 1 : s)
    return withGrad(out, "sum", [this], g => {
      const gk = keepdim ? g : reshapeRaw(g, keepShape)
      return [rawBroadcastTo(gk, [...this.shape])]
    })
  }

  mean(): Tensor<[], P>
  mean<D extends number>(
    dim: D & DimCheck<S, D>,
  ): Tensor<ReduceDim<S, D>, P>
  mean<D extends number, const K extends boolean>(
    dim: D & DimCheck<S, D>,
    keepdim: K,
  ): Tensor<ReduceDim<S, D, K>, P>
  mean(dim?: number, keepdim = false): AnyTensor {
    if (dim === undefined) {
      return (this.sum() as AnyTensor).div(this.numel)
    }
    const d = normalizeDim(dim, this.shape.length)
    return (this as AnyTensor)
      .sum(d as any, keepdim as any)
      .div(this.shape[d]!)
  }

  max(): Tensor<[], P>
  max<D extends number>(
    dim: D & DimCheck<S, D>,
  ): Tensor<ReduceDim<S, D>, P>
  max<D extends number, const K extends boolean>(
    dim: D & DimCheck<S, D>,
    keepdim: K,
  ): Tensor<ReduceDim<S, D, K>, P>
  max(dim?: number, keepdim = false): AnyTensor {
    if (dim === undefined) return rawReduceAll(this, "max")
    return rawReduce(this, dim, keepdim, "max")
  }

  argmax<D extends number>(
    dim: D & DimCheck<S, D>,
  ): Tensor<ReduceDim<S, D>, P> {
    const d = normalizeDim(dim, this.shape.length)
    return rawReduce(this, d, false, "argmax") as any
  }

  oneHot(classes: number): Tensor<[number, number], P> {
    if (this.rank !== 1) {
      throw new Error("oneHot() requires a rank-1 tensor")
    }
    if (!Number.isInteger(classes) || classes <= 0) {
      throw new Error(
        `oneHot() requires a positive class count, got ${classes}`,
      )
    }
    return rawOneHot(this, classes) as any
  }

  narrow<L extends number>(
    dim: 0,
    start: number,
    length: L,
  ): Tensor<ResizeDim<S, 0, L>, P>
  narrow<D extends number, L extends number>(
    dim: D & DimCheck<S, D>,
    start: number,
    length: L,
  ): Tensor<ResizeDim<S, D, L>, P>
  narrow(
    dim: number,
    start: number,
    length: number,
  ): AnyTensor {
    const d = normalizeDim(dim, this.shape.length)
    const size = this.shape[d]!
    if (
      !Number.isInteger(start)
      || !Number.isInteger(length)
      || start < 0
      || length < 0
      || start + length > size
    ) {
      throw new Error(
        `narrow(${dim}, ${start}, ${length}) is out of range for ${showShape(this.shape)}`,
      )
    }
    const out = rawNarrow(this, d, start, length)
    return withGrad(out, "narrow", [this], g => {
      const window = makeRaw(
        Float32Array.from({ length }, (_, i) => start + i),
        [length],
        "float32",
      )
      return [rawScatterAdd(g, window, d, size)]
    })
  }

  /**
   * `index` is a rank-1 tensor of integral values (typenet
   * has no integer dtype); its length becomes the size of `dim`.
   * Gradients flow to the gathered tensor, never to the index.
   */
  indexSelect<E extends number>(
    index: Tensor<[E], any>,
  ): Tensor<ResizeDim<S, 0, E>, P>
  indexSelect<E extends number, D extends number>(
    index: Tensor<[E], any>,
    dim: D & DimCheck<S, D>,
  ): Tensor<ResizeDim<S, D, E>, P>
  indexSelect(index: AnyTensor, dim = 0): AnyTensor {
    if (index.rank !== 1) {
      throw new Error(
        `indexSelect() requires a rank-1 index, got ${showShape(index.shape)}`,
      )
    }
    const d = normalizeDim(dim, this.shape.length)
    const rows = this.shape[d]!
    const out = rawIndexSelect(this, index, d)
    return withGrad(out, "indexSelect", [this], g => [
      rawScatterAdd(g, index, d, rows),
    ])
  }

  /**
   * Scatter-add rows along `dim` into an output of `length` rows: row
   * `j` of this tensor is *added into* row `index[j]` of the result.
   * Rows no index points at stay zero. This is `index_add_` on a zero
   * tensor, and the exact reverse of {@link indexSelect}.
   */
  scatterAdd<L extends number>(
    index: Tensor<[Dim0<S>], any>,
    length: L,
  ): Tensor<ResizeDim<S, 0, L>, P>
  scatterAdd<L extends number, D extends number>(
    index: Tensor<[number], any>,
    length: L,
    dim: D & DimCheck<S, D>,
  ): Tensor<ResizeDim<S, D, L>, P>
  scatterAdd(
    index: AnyTensor,
    length: number,
    dim = 0,
  ): AnyTensor {
    if (index.rank !== 1) {
      throw new Error(
        `scatterAdd() requires a rank-1 index, got ${showShape(index.shape)}`,
      )
    }
    if (!Number.isInteger(length) || length < 0) {
      throw new Error(
        `scatterAdd() requires a non-negative integer length, got ${length}`,
      )
    }
    const d = normalizeDim(dim, this.shape.length)
    if (index.numel !== this.shape[d]) {
      throw new Error(
        `scatterAdd(): ${index.numel} indices for ${this.shape[d]} rows along dim ${d}`,
      )
    }
    const out = rawScatterAdd(this, index, d, length)
    return withGrad(out, "scatterAdd", [this], g => [
      rawIndexSelect(g, index, d),
    ])
  }

  view<const V extends number[]>(
    shape: V & ViewCheck<S, V>,
  ): Tensor<ResolveView<S, V>, P> {
    const resolved = resolveViewRuntime(
      [...this.shape],
      shape as number[],
    )
    const out = reshapeRaw(this, resolved)
    return withGrad(out, "view", [this], g => [
      reshapeRaw(g, [...this.shape]),
    ]) as any
  }

  reshape<const V extends number[]>(
    shape: V & ViewCheck<S, V>,
  ): Tensor<ResolveView<S, V>, P> {
    return this.view(shape as any) as any
  }

  squeeze(): Tensor<Squeeze<S>, P> {
    const target = this.shape.filter(s => s !== 1)
    const out = reshapeRaw(this, target)
    return withGrad(out, "squeeze", [this], g => [
      reshapeRaw(g, [...this.shape]),
    ]) as any
  }

  squeezeDim<D extends number>(
    dim: D & SqueezeDimCheck<S, D>,
  ): Tensor<SqueezeDim<S, D>, P> {
    const d = normalizeDim(dim as number, this.shape.length)
    if (this.shape[d] !== 1) {
      throw new Error(
        `Cannot squeeze dim ${dim} of ${showShape(this.shape)}: size is not 1`,
      )
    }
    const target = this.shape.filter((_, i) => i !== d)
    const out = reshapeRaw(this, target)
    return withGrad(out, "squeeze", [this], g => [
      reshapeRaw(g, [...this.shape]),
    ]) as any
  }

  unsqueeze<D extends number>(
    dim: D & UnsqueezeCheck<S, D>,
  ): Tensor<Unsqueeze<S, D>, P> {
    const d = normalizeDim(
      dim as number,
      this.shape.length,
      1,
    )
    const target = [...this.shape]
    target.splice(d, 0, 1)
    const out = reshapeRaw(this, target)
    return withGrad(out, "unsqueeze", [this], g => [
      reshapeRaw(g, [...this.shape]),
    ]) as any
  }

  transpose<D0 extends number, D1 extends number>(
    dim0: D0 & TransposeCheck<S, D0, D1>,
    dim1: D1,
  ): Tensor<Transpose<S, D0, D1>, P> {
    const rank = this.shape.length
    const a = normalizeDim(dim0 as number, rank)
    const b = normalizeDim(dim1, rank)
    const order = [...Array(rank).keys()]
    ;[order[a], order[b]] = [order[b]!, order[a]!]
    return this.permuteRaw(order) as any
  }

  get T(): S["length"] extends 2 ? Tensor<Transpose<S, 0, 1>, P> : ErrorMessage<".T is only defined for rank-2 tensors — use transpose(d0, d1)"> {
    if (this.shape.length !== 2) {
      throw new Error(
        ".T is only defined for rank-2 tensors",
      )
    }
    return this.permuteRaw([1, 0]) as any
  }

  permute<const O extends number[]>(
    ...order: O & PermuteCheck<S, O>
  ): Tensor<Permute<S, O>, P> {
    const rank = this.shape.length
    const normalized = (order as number[]).map(d => normalizeDim(d, rank))
    if (
      normalized.length !== rank
      || new Set(normalized).size !== rank
    ) {
      throw new Error(
        `permute(${(order as number[]).join(", ")}) is not a permutation of ${showShape(this.shape)}`,
      )
    }
    return this.permuteRaw(normalized) as any
  }

  private permuteRaw(order: number[]): AnyTensor {
    const out = rawPermute(this, order)
    const inverse = new Array<number>(order.length)
    order.forEach((d, i) => (inverse[d] = i))
    return withGrad(out, "permute", [this], g => [
      rawPermute(g, inverse),
    ])
  }

  static stack<
    const T extends readonly [AnyTensor, ...AnyTensor[]],
    const D extends number = 0,
  >(
    tensors: T & StackCheck<T>,
    dim?: D,
  ): Tensor<
    Stack<ShapeOf<T[0]>, T["length"], D>,
    ParamsOf<T[0]>
  > {
    const ts = tensors as readonly AnyTensor[]
    const first = ts[0]!
    for (const t of ts) {
      if (!shapesEqual(t.shape, first.shape)) {
        throw new Error(
          `stack: all tensors must share a shape (${showShape(first.shape)} vs ${showShape(t.shape)})`,
        )
      }
    }
    const unsqueezed = ts.map(t => t.unsqueeze((dim ?? 0) as any))
    let acc = unsqueezed[0]!
    for (let i = 1; i < unsqueezed.length; i++) {
      acc = Tensor.cat(
        acc as any,
        unsqueezed[i]! as any,
        (dim ?? 0) as any,
      ) as AnyTensor
    }
    return acc as any
  }

  static cat<
    A extends Shape,
    B extends Shape,
    PA extends TensorParams,
    const D extends number = 0,
  >(
    a: Tensor<A, PA>,
    b: Tensor<B, any> & CatCheck<A, B, D>,
    dim?: D,
  ): Tensor<Cat<A, B, D>, PA> {
    const ta = a as AnyTensor
    const tb = b as AnyTensor
    const d = normalizeDim(dim ?? 0, ta.shape.length)
    if (ta.shape.length !== tb.shape.length) {
      throw new Error(
        `cat: tensors must have the same rank (${showShape(ta.shape)} vs ${showShape(tb.shape)})`,
      )
    }
    for (let i = 0; i < ta.shape.length; i++) {
      if (i !== d && ta.shape[i] !== tb.shape[i]) {
        throw new Error(
          `cat: shapes ${showShape(ta.shape)} and ${showShape(tb.shape)} differ outside dim ${d}`,
        )
      }
    }
    const lenA = ta.shape[d]!
    const lenB = tb.shape[d]!
    const result = rawCat(ta, tb, d)
    return withGrad(result, "cat", [ta, tb], g => [
      rawNarrow(g, d, 0, lenA),
      rawNarrow(g, d, lenA, lenB),
    ]) as any
  }

  [Operator.plus](
    lhs: Tensor<S, P>,
    rhs: number,
  ): Tensor<S, P>
  [Operator.plus](
    lhs: number,
    rhs: Tensor<S, P>,
  ): Tensor<S, P>
  [Operator.plus](
    lhs: Tensor<[Dim0<S>, 1], P>,
    rhs: Tensor<[1, Dim0<S>], any>,
  ): Tensor<[Dim0<S>, Dim0<S>], P>
  [Operator.plus](
    lhs: Tensor<[1, Dim1<S>], P>,
    rhs: Tensor<[Dim1<S>, 1], any>,
  ): Tensor<[Dim1<S>, Dim1<S>], P>
  [Operator.plus]<S2 extends Shape>(
    lhs: Tensor<S, P>,
    rhs: Tensor<S2, any> & BroadcastCheck<S, S2>,
  ): Tensor<Broadcast<S, S2>, P>
  [Operator.plus](lhs: any, rhs: any): any {
    return coerceLhs(lhs, rhs).add(rhs)
  }

  [Operator.minus](
    lhs: Tensor<S, P>,
    rhs: number,
  ): Tensor<S, P>
  [Operator.minus](
    lhs: number,
    rhs: Tensor<S, P>,
  ): Tensor<S, P>
  [Operator.minus](
    lhs: Tensor<[Dim0<S>, 1], P>,
    rhs: Tensor<[1, Dim0<S>], any>,
  ): Tensor<[Dim0<S>, Dim0<S>], P>
  [Operator.minus](
    lhs: Tensor<[1, Dim1<S>], P>,
    rhs: Tensor<[Dim1<S>, 1], any>,
  ): Tensor<[Dim1<S>, Dim1<S>], P>
  [Operator.minus]<S2 extends Shape>(
    lhs: Tensor<S, P>,
    rhs: Tensor<S2, any> & BroadcastCheck<S, S2>,
  ): Tensor<Broadcast<S, S2>, P>
  [Operator.minus](lhs: any, rhs: any): any {
    return coerceLhs(lhs, rhs).sub(rhs)
  }

  [Operator.star](
    lhs: Tensor<S, P>,
    rhs: number,
  ): Tensor<S, P>
  [Operator.star](
    lhs: number,
    rhs: Tensor<S, P>,
  ): Tensor<S, P>
  [Operator.star](
    lhs: Tensor<[Dim0<S>, 1], P>,
    rhs: Tensor<[1, Dim0<S>], any>,
  ): Tensor<[Dim0<S>, Dim0<S>], P>
  [Operator.star](
    lhs: Tensor<[1, Dim1<S>], P>,
    rhs: Tensor<[Dim1<S>, 1], any>,
  ): Tensor<[Dim1<S>, Dim1<S>], P>
  [Operator.star]<S2 extends Shape>(
    lhs: Tensor<S, P>,
    rhs: Tensor<S2, any> & BroadcastCheck<S, S2>,
  ): Tensor<Broadcast<S, S2>, P>
  [Operator.star](lhs: any, rhs: any): any {
    return coerceLhs(lhs, rhs).mul(rhs)
  }

  [Operator.slash](
    lhs: Tensor<S, P>,
    rhs: number,
  ): Tensor<S, P>
  [Operator.slash](
    lhs: number,
    rhs: Tensor<S, P>,
  ): Tensor<S, P>
  [Operator.slash](
    lhs: Tensor<[Dim0<S>, 1], P>,
    rhs: Tensor<[1, Dim0<S>], any>,
  ): Tensor<[Dim0<S>, Dim0<S>], P>
  [Operator.slash](
    lhs: Tensor<[1, Dim1<S>], P>,
    rhs: Tensor<[Dim1<S>, 1], any>,
  ): Tensor<[Dim1<S>, Dim1<S>], P>
  [Operator.slash]<S2 extends Shape>(
    lhs: Tensor<S, P>,
    rhs: Tensor<S2, any> & BroadcastCheck<S, S2>,
  ): Tensor<Broadcast<S, S2>, P>
  [Operator.slash](lhs: any, rhs: any): any {
    return coerceLhs(lhs, rhs).div(rhs)
  }

  [Operator.starStar](
    lhs: Tensor<S, P>,
    rhs: number,
  ): Tensor<S, P>
  [Operator.starStar](lhs: any, rhs: any): any {
    if (typeof rhs !== "number") {
      throw new Error(
        "** on tensors requires a scalar exponent",
      )
    }
    return (lhs as AnyTensor).pow(rhs)
  }
}

function coerce(
  value: AnyTensor | number,
  like: AnyTensor,
): AnyTensor {
  if (typeof value === "number") {
    return makeRaw(
      arrayCtor(like.dtype).of(value),
      [],
      like.dtype,
    )
  }
  return value
}

function coerceLhs(lhs: any, rhs: any): AnyTensor {
  if (lhs instanceof Tensor) return lhs as AnyTensor
  return coerce(lhs, rhs as AnyTensor)
}

function makeMovedData(
  t: AnyTensor,
  data: TypedArray,
  dtype: DType,
): AnyTensor {
  const out = makeRaw(data, t.shape, dtype)
  return withGrad(out, "to", [t], g => [g])
}

function resolveViewRuntime(
  shape: number[],
  view: number[],
): number[] {
  const negOnes = view.filter(v => v === -1).length
  if (negOnes > 1) {
    throw new Error("Only one -1 dim is allowed in view()")
  }
  const total = prod(shape)
  if (negOnes === 1) {
    const rest = prod(view.filter(v => v !== -1))
    if (rest === 0 || total % rest !== 0) {
      throw new Error(
        `Cannot view tensor of shape ${showShape(shape)} as ${showShape(view)}`,
      )
    }
    return view.map(v => (v === -1 ? total / rest : v))
  }
  if (prod(view) !== total) {
    throw new Error(
      `Cannot view tensor of shape ${showShape(shape)} as ${showShape(view)} (${total} vs ${prod(view)} elements)`,
    )
  }
  return [...view]
}

function matmul2(a: AnyTensor, b: AnyTensor): AnyTensor {
  const out = rawMatmul(a, b)
  return withGrad(out, "matmul", [a, b], g => {
    const bt = rawPermute(b, swapLastTwo(b.shape.length))
    const at = rawPermute(a, swapLastTwo(a.shape.length))
    const da = sumTo(rawMatmul(g, bt), [...a.shape])
    const db = sumTo(rawMatmul(at, g), [...b.shape])
    return [da, db]
  })
}

function swapLastTwo(rank: number): number[] {
  const order = [...Array(rank).keys()]
  ;[order[rank - 2], order[rank - 1]] = [
    order[rank - 1]!,
    order[rank - 2]!,
  ]
  return order
}

export { forceMany, noGrad }
export { _activeUpdateTrace }
export { compile, printGraph } from "./compile.ts"
export type { CompiledFn } from "./compile.ts"
export { normal, uniform } from "./eager.ts"
export { configure, isLazy } from "./lazy.ts"
export { broadcastShapes } from "./storage.ts"
export type { DType, RandomKind } from "./storage.ts"

export const tensor = Tensor.of
export const zeros = Tensor.zeros
export const ones = Tensor.ones
export const full = Tensor.full
export const rand = Tensor.rand
export const randn = Tensor.randn
export const eye = Tensor.eye
export const arange = Tensor.arange
export const scalar = Tensor.scalar
export const stack = Tensor.stack
export const cat = Tensor.cat
