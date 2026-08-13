import { Operator } from "tsover-runtime"
import { type GradNode, noGrad, runBackward, withGrad } from "./autograd.ts"
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
import { resolveView } from "./shape.ts"
import type {
  Broadcast,
  BroadcastCheck,
  Cat,
  CatCheck,
  CatN,
  CatNCheck,
  DimAt,
  DimCheck,
  ErrorMessage,
  InferShape,
  IsDynamic,
  MatMul,
  MatMulCheck,
  NestedArray,
  Permute,
  PermuteCheck,
  Rank1Check,
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

export type ShapeOf<T> = T extends Tensor<infer S> ? S : never

export type NestedNumbers = number | readonly NestedNumbers[]

export type AnyTensor = Tensor<any>

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
export function fromFlat<const Sh extends Shape>(
  data: ArrayLike<number>,
  shape: Sh,
  dtype: DType = "float32",
): Tensor<Sh> {
  const ctor = arrayCtor(dtype)
  const buf = data instanceof ctor ? data : ctor.from(data)
  return makeRaw(buf, shape, dtype) as any
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

/**
 * Friend-module access to the private fields. Populated by a static
 * block inside `Tensor` (the only scope that can reach `#` fields), and
 * imported by the evaluator modules; tests go through `src/testing.ts`.
 */
export interface TensorInternal {
  sourceOf(t: AnyTensor): TensorStorage
  cpuOf(t: AnyTensor): TypedArray | null
  setCpu(t: AnyTensor, data: TypedArray): void
  /** Drop a lazy tensor's materialized value so a replay recomputes it. */
  resetCpu(t: AnyTensor): void
  hasValue(t: AnyTensor): boolean
  gradNodeOf(t: AnyTensor): GradNode | null
  setGradNode(t: AnyTensor, node: GradNode | null): void
  /** Same source and materialized buffer under a new shape (free view). */
  makeView(t: AnyTensor, shape: readonly number[]): AnyTensor
}

export let _internal!: TensorInternal

type StackCheck<T extends readonly AnyTensor[]> = T[number] extends Tensor<ShapeOf<T[0]>> ? unknown
  : ErrorMessage<"stack: all tensors must have the same shape">

export class Tensor<S extends Shape> {
  /**
   * Where the value comes from: a CPU buffer or a lazy graph node.
   * Immutable for the life of the tensor — forcing fills {@link #cpu}
   * instead of rewriting this field, so a lazy tensor stays lazy.
   */
  readonly #source: TensorStorage
  /** The materialized CPU buffer; for lazy sources, filled by force. */
  #cpu: TypedArray | null
  #gradNode: GradNode | null = null
  readonly shape: S
  readonly dtype: DType

  grad: Tensor<S> | null = null

  /** Internal grad-leaf flag; read {@link needsGrad}, set via {@link requiresGrad}. */
  _requiresGrad = false

  static {
    _internal = {
      sourceOf: t => t.#source,
      cpuOf: t => t.#cpu,
      setCpu: (t, data) => {
        t.#cpu = data
      },
      resetCpu: t => {
        if (t.#source.kind === "lazy") t.#cpu = null
      },
      hasValue: t => t.#cpu !== null,
      gradNodeOf: t => t.#gradNode,
      setGradNode: (t, node) => {
        t.#gradNode = node
      },
      makeView: (t, shape) => {
        const out = makeStorage(
          t.#source,
          shape,
          t.dtype,
        )
        out.#cpu = t.#cpu
        return out
      },
    }
  }

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
    this.#source = storage
    this.#cpu = storage.kind === "cpu" ? storage.data : null
    this.shape = shape as S
    this.dtype = dtype
  }

  get data(): TypedArray {
    if (this.#cpu === null) {
      force(this as AnyTensor)
    }
    return this.#cpu!
  }

  get needsGrad(): boolean {
    return this._requiresGrad || this.#gradNode !== null
  }

  /** Whether an autograd tape reaches this tensor. */
  get taped(): boolean {
    return this.#gradNode !== null
  }

  get rank(): S["length"] {
    return this.shape.length
  }

  get numel(): number {
    return prod(this.shape)
  }

  static of<const V extends NestedNumbers>(
    value: V,
  ): Tensor<InferShape<V>> {
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
  ): Tensor<Sh> {
    const data = new Float32Array(prod(shape)).fill(value)
    return makeRaw(data, shape, "float32") as any
  }

  static zeros<const Sh extends Shape>(
    shape: Sh,
  ): Tensor<Sh> {
    return Tensor.full(shape, 0)
  }

  static ones<const Sh extends Shape>(
    shape: Sh,
  ): Tensor<Sh> {
    return Tensor.full(shape, 1)
  }

  static rand<const Sh extends Shape>(
    shape: Sh,
  ): Tensor<Sh> {
    const data = new Float32Array(prod(shape))
    for (let i = 0; i < data.length; i++) {
      data[i] = Math.random()
    }
    return makeRaw(data, shape, "float32") as any
  }

  static randn<const Sh extends Shape>(
    shape: Sh,
  ): Tensor<Sh> {
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
  ): Tensor<[N, N]> {
    const data = new Float32Array(n * n)
    for (let i = 0; i < n; i++) data[i * n + i] = 1
    return makeRaw(data, [n, n], "float32") as any
  }

  static arange<const N extends number>(
    n: N,
  ): Tensor<[N]> {
    const data = new Float32Array(n)
    for (let i = 0; i < n; i++) data[i] = i
    return makeRaw(data, [n], "float32") as any
  }

  static scalar(value: number): Tensor<[]> {
    return makeRaw(
      Float32Array.of(value),
      [],
      "float32",
    ) as any
  }

  /**
   * A new leaf over this tensor's storage with gradients enabled — the
   * tape starts here. Like {@link detach}, the receiver is untouched:
   * rebind the result, do not rely on mutation.
   */
  requiresGrad(): Tensor<S> {
    const leaf = _internal.makeView(
      this as AnyTensor,
      this.shape,
    ) as Tensor<S>
    leaf._requiresGrad = true
    return leaf
  }

  to<const D extends DType>(
    dtype: D,
  ): Tensor<S> {
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

  get(...indices: { [K in keyof S]: number }): number {
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

  detach(): Tensor<S> {
    return _internal.makeView(
      this as AnyTensor,
      this.shape,
    ) as any
  }

  clone(): Tensor<S> {
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

  backward(gradient?: Tensor<S>): void {
    runBackward(
      this as AnyTensor,
      gradient as AnyTensor | undefined,
      _activeUpdateTrace,
    )
  }

  zeroGrad(): void {
    this.grad = null
  }

  add(other: number): Tensor<S>
  add(
    this: Tensor<[Dim0<S>, 1]>,
    other: Tensor<[1, Dim0<S>]>,
  ): Tensor<[Dim0<S>, Dim0<S>]>
  add(
    this: Tensor<[1, Dim1<S>]>,
    other: Tensor<[Dim1<S>, 1]>,
  ): Tensor<[Dim1<S>, Dim1<S>]>
  add<S2 extends Shape>(
    other: Tensor<S2> & BroadcastCheck<S, S2>,
  ): Tensor<Broadcast<S, S2>>
  add(other: AnyTensor | number): AnyTensor {
    const b = coerce(other, this)
    const out = rawBinary(this, b, "add")
    return withGrad(out, "add", [this, b], g => [
      sumTo(g, this.shape),
      sumTo(g, b.shape),
    ])
  }

  sub(other: number): Tensor<S>
  sub(
    this: Tensor<[Dim0<S>, 1]>,
    other: Tensor<[1, Dim0<S>]>,
  ): Tensor<[Dim0<S>, Dim0<S>]>
  sub(
    this: Tensor<[1, Dim1<S>]>,
    other: Tensor<[Dim1<S>, 1]>,
  ): Tensor<[Dim1<S>, Dim1<S>]>
  sub<S2 extends Shape>(
    other: Tensor<S2> & BroadcastCheck<S, S2>,
  ): Tensor<Broadcast<S, S2>>
  sub(other: AnyTensor | number): AnyTensor {
    const b = coerce(other, this)
    const out = rawBinary(this, b, "sub")
    return withGrad(out, "sub", [this, b], g => [
      sumTo(g, this.shape),
      sumTo(rawUnary(g, "neg"), b.shape),
    ])
  }

  mul(other: number): Tensor<S>
  mul(
    this: Tensor<[Dim0<S>, 1]>,
    other: Tensor<[1, Dim0<S>]>,
  ): Tensor<[Dim0<S>, Dim0<S>]>
  mul(
    this: Tensor<[1, Dim1<S>]>,
    other: Tensor<[Dim1<S>, 1]>,
  ): Tensor<[Dim1<S>, Dim1<S>]>
  mul<S2 extends Shape>(
    other: Tensor<S2> & BroadcastCheck<S, S2>,
  ): Tensor<Broadcast<S, S2>>
  mul(other: AnyTensor | number): AnyTensor {
    const b = coerce(other, this)
    const out = rawBinary(this, b, "mul")
    return withGrad(out, "mul", [this, b], g => [
      sumTo(rawBinary(g, b, "mul"), this.shape),
      sumTo(rawBinary(g, this, "mul"), b.shape),
    ])
  }

  div(other: number): Tensor<S>
  div(
    this: Tensor<[Dim0<S>, 1]>,
    other: Tensor<[1, Dim0<S>]>,
  ): Tensor<[Dim0<S>, Dim0<S>]>
  div(
    this: Tensor<[1, Dim1<S>]>,
    other: Tensor<[Dim1<S>, 1]>,
  ): Tensor<[Dim1<S>, Dim1<S>]>
  div<S2 extends Shape>(
    other: Tensor<S2> & BroadcastCheck<S, S2>,
  ): Tensor<Broadcast<S, S2>>
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
  maximum(other: number): Tensor<S>
  maximum<S2 extends Shape>(
    other: Tensor<S2> & BroadcastCheck<S, S2>,
  ): Tensor<Broadcast<S, S2>>
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
  minimum(other: number): Tensor<S>
  minimum<S2 extends Shape>(
    other: Tensor<S2> & BroadcastCheck<S, S2>,
  ): Tensor<Broadcast<S, S2>>
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
  ): Tensor<S> {
    let out = this as AnyTensor
    if (min !== null) out = out.maximum(min)
    if (max !== null) out = out.minimum(max)
    return out as any
  }

  // Comparisons produce 1.0 / 0.0 masks and stop gradients: a step
  // function has zero derivative wherever it is differentiable.
  gt(other: number): Tensor<S>
  gt<S2 extends Shape>(
    other: Tensor<S2> & BroadcastCheck<S, S2>,
  ): Tensor<Broadcast<S, S2>>
  gt(other: AnyTensor | number): AnyTensor {
    return this.compare(other, "gt")
  }

  ge(other: number): Tensor<S>
  ge<S2 extends Shape>(
    other: Tensor<S2> & BroadcastCheck<S, S2>,
  ): Tensor<Broadcast<S, S2>>
  ge(other: AnyTensor | number): AnyTensor {
    return this.compare(other, "ge")
  }

  lt(other: number): Tensor<S>
  lt<S2 extends Shape>(
    other: Tensor<S2> & BroadcastCheck<S, S2>,
  ): Tensor<Broadcast<S, S2>>
  lt(other: AnyTensor | number): AnyTensor {
    return this.compare(other, "lt")
  }

  le(other: number): Tensor<S>
  le<S2 extends Shape>(
    other: Tensor<S2> & BroadcastCheck<S, S2>,
  ): Tensor<Broadcast<S, S2>>
  le(other: AnyTensor | number): AnyTensor {
    return this.compare(other, "le")
  }

  eq(other: number): Tensor<S>
  eq<S2 extends Shape>(
    other: Tensor<S2> & BroadcastCheck<S, S2>,
  ): Tensor<Broadcast<S, S2>>
  eq(other: AnyTensor | number): AnyTensor {
    return this.compare(other, "eq")
  }

  private compare(
    other: AnyTensor | number,
    op: "gt" | "ge" | "lt" | "le" | "eq",
  ): AnyTensor {
    return rawBinary(this, coerce(other, this), op)
  }

  pow(exponent: number): Tensor<S> {
    const out = rawUnary(this, "pow", exponent)
    return withGrad(out, "pow", [this], g => [
      rawBinary(
        g,
        rawUnary(this, "scalePowGrad", exponent),
        "mul",
      ),
    ]) as any
  }

  neg(): Tensor<S> {
    const out = rawUnary(this, "neg")
    return withGrad(out, "neg", [this], g => [
      rawUnary(g, "neg"),
    ]) as any
  }

  exp(): Tensor<S> {
    const out = rawUnary(this, "exp")
    return withGrad(out, "exp", [this], g => [
      rawBinary(g, out, "mul"),
    ]) as any
  }

  log(): Tensor<S> {
    const out = rawUnary(this, "log")
    return withGrad(out, "log", [this], g => [
      rawBinary(g, this, "div"),
    ]) as any
  }

  sqrt(): Tensor<S> {
    const out = rawUnary(this, "sqrt")
    return withGrad(out, "sqrt", [this], g => [
      rawBinary(g, out, "halfDiv"),
    ]) as any
  }

  abs(): Tensor<S> {
    const out = rawUnary(this, "abs")
    return withGrad(out, "abs", [this], g => [
      rawBinary(g, this, "mulSign"),
    ]) as any
  }

  relu(): Tensor<S> {
    const out = rawUnary(this, "relu")
    return withGrad(out, "relu", [this], g => [
      rawBinary(g, this, "reluGrad"),
    ]) as any
  }

  leakyRelu(negativeSlope = 0.01): Tensor<S> {
    const out = rawUnary(this, "leakyRelu", negativeSlope)
    return withGrad(out, "leakyRelu", [this], g => [
      rawBinary(g, this, "leakyReluGrad", negativeSlope),
    ]) as any
  }

  sigmoid(): Tensor<S> {
    const out = rawUnary(this, "sigmoid")
    return withGrad(out, "sigmoid", [this], g => [
      rawBinary(g, out, "sigmoidGrad"),
    ]) as any
  }

  tanh(): Tensor<S> {
    const out = rawUnary(this, "tanh")
    return withGrad(out, "tanh", [this], g => [
      rawBinary(g, out, "tanhGrad"),
    ]) as any
  }

  softmax<D extends number>(
    dim: D & DimCheck<S, D>,
  ): Tensor<S> {
    const { e } = this.softmaxShift(dim as number)
    return e.div(e.sum(dim as any, true) as any) as any
  }

  logSoftmax<D extends number>(
    dim: D & DimCheck<S, D>,
  ): Tensor<S> {
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
    other: Tensor<S2> & MatMulCheck<S, S2>,
  ): Tensor<MatMul<S, S2>> {
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
    other: Tensor<S2> & MatMulCheck<S, S2>,
  ): Tensor<MatMul<S, S2>> {
    return this.matmul(other)
  }

  sum(): Tensor<[]>
  sum<D extends number>(
    dim: D & DimCheck<S, D>,
  ): Tensor<ReduceDim<S, D>>
  sum<D extends number, const K extends boolean>(
    dim: D & DimCheck<S, D>,
    keepdim: K,
  ): Tensor<ReduceDim<S, D, K>>
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

  mean(): Tensor<[]>
  mean<D extends number>(
    dim: D & DimCheck<S, D>,
  ): Tensor<ReduceDim<S, D>>
  mean<D extends number, const K extends boolean>(
    dim: D & DimCheck<S, D>,
    keepdim: K,
  ): Tensor<ReduceDim<S, D, K>>
  mean(dim?: number, keepdim = false): AnyTensor {
    if (dim === undefined) {
      return (this.sum() as AnyTensor).div(this.numel)
    }
    const d = normalizeDim(dim, this.shape.length)
    return (this as AnyTensor)
      .sum(d as any, keepdim as any)
      .div(this.shape[d]!)
  }

  max(): Tensor<[]>
  max<D extends number>(
    dim: D & DimCheck<S, D>,
  ): Tensor<ReduceDim<S, D>>
  max<D extends number, const K extends boolean>(
    dim: D & DimCheck<S, D>,
    keepdim: K,
  ): Tensor<ReduceDim<S, D, K>>
  max(dim?: number, keepdim = false): AnyTensor {
    if (dim === undefined) return rawReduceAll(this, "max")
    return rawReduce(this, dim, keepdim, "max")
  }

  argmax<D extends number>(
    dim: D & DimCheck<S, D>,
  ): Tensor<ReduceDim<S, D>> {
    const d = normalizeDim(dim, this.shape.length)
    return rawReduce(this, d, false, "argmax") as any
  }

  oneHot<const C extends number>(
    this: Tensor<S> & Rank1Check<S>,
    classes: C,
  ): S extends [infer N extends number] ? Tensor<[N, C]> : IsDynamic<S> extends true ? Tensor<[number, C]> : never
  oneHot(this: AnyTensor, classes: number): AnyTensor {
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
  ): Tensor<ResizeDim<S, 0, L>>
  narrow<D extends number, L extends number>(
    dim: D & DimCheck<S, D>,
    start: number,
    length: L,
  ): Tensor<ResizeDim<S, D, L>>
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
    index: Tensor<[E]>,
  ): Tensor<ResizeDim<S, 0, E>>
  indexSelect<E extends number, D extends number>(
    index: Tensor<[E]>,
    dim: D & DimCheck<S, D>,
  ): Tensor<ResizeDim<S, D, E>>
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
    index: Tensor<[Dim0<S>]>,
    length: L,
  ): Tensor<ResizeDim<S, 0, L>>
  scatterAdd<L extends number, D extends number>(
    index: Tensor<[DimAt<S, D>]>,
    length: L,
    dim: D & DimCheck<S, D>,
  ): Tensor<ResizeDim<S, D, L>>
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
  ): Tensor<ResolveView<S, V>> {
    const resolved = resolveView(
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
  ): Tensor<ResolveView<S, V>> {
    return this.view(shape as any) as any
  }

  squeeze(): Tensor<Squeeze<S>>
  squeeze<D extends number>(
    dim: D & SqueezeDimCheck<S, D>,
  ): Tensor<SqueezeDim<S, D>>
  squeeze(dim?: number): AnyTensor {
    let target: number[]
    if (dim === undefined) {
      target = this.shape.filter(s => s !== 1)
    } else {
      const d = normalizeDim(dim, this.shape.length)
      if (this.shape[d] !== 1) {
        throw new Error(
          `Cannot squeeze dim ${dim} of ${showShape(this.shape)}: size is not 1`,
        )
      }
      target = this.shape.filter((_, i) => i !== d)
    }
    const out = reshapeRaw(this, target)
    return withGrad(out, "squeeze", [this], g => [
      reshapeRaw(g, [...this.shape]),
    ])
  }

  /** @deprecated use `squeeze(dim)` */
  squeezeDim<D extends number>(
    dim: D & SqueezeDimCheck<S, D>,
  ): Tensor<SqueezeDim<S, D>> {
    return this.squeeze(dim as any) as any
  }

  unsqueeze<D extends number>(
    dim: D & UnsqueezeCheck<S, D>,
  ): Tensor<Unsqueeze<S, D>> {
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
  ): Tensor<Transpose<S, D0, D1>> {
    const rank = this.shape.length
    const a = normalizeDim(dim0 as number, rank)
    const b = normalizeDim(dim1, rank)
    const order = [...Array(rank).keys()]
    ;[order[a], order[b]] = [order[b]!, order[a]!]
    return this.permuteRaw(order) as any
  }

  get T(): S["length"] extends 2 ? Tensor<Transpose<S, 0, 1>> : ErrorMessage<".T is only defined for rank-2 tensors — use transpose(d0, d1)"> {
    if (this.shape.length !== 2) {
      throw new Error(
        ".T is only defined for rank-2 tensors",
      )
    }
    return this.permuteRaw([1, 0]) as any
  }

  permute<const O extends number[]>(
    ...order: O & PermuteCheck<S, O>
  ): Tensor<Permute<S, O>> {
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
    Stack<ShapeOf<T[0]>, T["length"], D>
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
    const D extends number = 0,
  >(
    a: Tensor<A>,
    b: Tensor<B> & CatCheck<A, B, D>,
    dim?: D,
  ): Tensor<Cat<A, B, D>>
  static cat<
    const T extends readonly [AnyTensor, ...AnyTensor[]],
    const D extends number = 0,
  >(
    tensors: T & CatNCheck<T, D>,
    dim?: D,
  ): Tensor<CatN<T, D>>
  static cat(
    a: any,
    b?: any,
    dim?: number,
  ): any {
    // The n-ary form is pairwise sugar: a fold over the same binary cat
    // nodes, so the IR, autograd rule and native lowering are untouched.
    if (Array.isArray(a)) {
      const d = (b as number | undefined) ?? 0
      let acc = a[0]! as AnyTensor
      for (let i = 1; i < a.length; i++) {
        acc = cat2(acc, a[i]! as AnyTensor, d)
      }
      return acc
    }
    return cat2(a as AnyTensor, b as AnyTensor, dim ?? 0)
  }

  [Operator.plus](
    lhs: Tensor<S>,
    rhs: number,
  ): Tensor<S>
  [Operator.plus](
    lhs: number,
    rhs: Tensor<S>,
  ): Tensor<S>
  [Operator.plus](
    lhs: Tensor<[Dim0<S>, 1]>,
    rhs: Tensor<[1, Dim0<S>]>,
  ): Tensor<[Dim0<S>, Dim0<S>]>
  [Operator.plus](
    lhs: Tensor<[1, Dim1<S>]>,
    rhs: Tensor<[Dim1<S>, 1]>,
  ): Tensor<[Dim1<S>, Dim1<S>]>
  [Operator.plus]<S2 extends Shape>(
    lhs: Tensor<S>,
    rhs: Tensor<S2> & BroadcastCheck<S, S2>,
  ): Tensor<Broadcast<S, S2>>
  [Operator.plus](lhs: any, rhs: any): any {
    return coerceLhs(lhs, rhs).add(rhs)
  }

  [Operator.minus](
    lhs: Tensor<S>,
    rhs: number,
  ): Tensor<S>
  [Operator.minus](
    lhs: number,
    rhs: Tensor<S>,
  ): Tensor<S>
  [Operator.minus](
    lhs: Tensor<[Dim0<S>, 1]>,
    rhs: Tensor<[1, Dim0<S>]>,
  ): Tensor<[Dim0<S>, Dim0<S>]>
  [Operator.minus](
    lhs: Tensor<[1, Dim1<S>]>,
    rhs: Tensor<[Dim1<S>, 1]>,
  ): Tensor<[Dim1<S>, Dim1<S>]>
  [Operator.minus]<S2 extends Shape>(
    lhs: Tensor<S>,
    rhs: Tensor<S2> & BroadcastCheck<S, S2>,
  ): Tensor<Broadcast<S, S2>>
  [Operator.minus](lhs: any, rhs: any): any {
    return coerceLhs(lhs, rhs).sub(rhs)
  }

  [Operator.star](
    lhs: Tensor<S>,
    rhs: number,
  ): Tensor<S>
  [Operator.star](
    lhs: number,
    rhs: Tensor<S>,
  ): Tensor<S>
  [Operator.star](
    lhs: Tensor<[Dim0<S>, 1]>,
    rhs: Tensor<[1, Dim0<S>]>,
  ): Tensor<[Dim0<S>, Dim0<S>]>
  [Operator.star](
    lhs: Tensor<[1, Dim1<S>]>,
    rhs: Tensor<[Dim1<S>, 1]>,
  ): Tensor<[Dim1<S>, Dim1<S>]>
  [Operator.star]<S2 extends Shape>(
    lhs: Tensor<S>,
    rhs: Tensor<S2> & BroadcastCheck<S, S2>,
  ): Tensor<Broadcast<S, S2>>
  [Operator.star](lhs: any, rhs: any): any {
    return coerceLhs(lhs, rhs).mul(rhs)
  }

  [Operator.slash](
    lhs: Tensor<S>,
    rhs: number,
  ): Tensor<S>
  [Operator.slash](
    lhs: number,
    rhs: Tensor<S>,
  ): Tensor<S>
  [Operator.slash](
    lhs: Tensor<[Dim0<S>, 1]>,
    rhs: Tensor<[1, Dim0<S>]>,
  ): Tensor<[Dim0<S>, Dim0<S>]>
  [Operator.slash](
    lhs: Tensor<[1, Dim1<S>]>,
    rhs: Tensor<[Dim1<S>, 1]>,
  ): Tensor<[Dim1<S>, Dim1<S>]>
  [Operator.slash]<S2 extends Shape>(
    lhs: Tensor<S>,
    rhs: Tensor<S2> & BroadcastCheck<S, S2>,
  ): Tensor<Broadcast<S, S2>>
  [Operator.slash](lhs: any, rhs: any): any {
    return coerceLhs(lhs, rhs).div(rhs)
  }

  [Operator.starStar](
    lhs: Tensor<S>,
    rhs: number,
  ): Tensor<S>
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

function cat2(
  ta: AnyTensor,
  tb: AnyTensor,
  dim: number,
): AnyTensor {
  const d = normalizeDim(dim, ta.shape.length)
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
  ])
}
