import { Operator } from "tsover-runtime"
import { type GradNode, runBackward, withGrad } from "./autograd.ts"
import { _activeUpdateTrace, tensorNames } from "./compile.ts"
import { isTracing } from "./context.ts"
import {
  rawBinary,
  rawBroadcastTo,
  rawCat,
  rawCrossEntropy,
  rawDropout,
  rawGatherRows,
  rawGelu,
  rawGeluGrad,
  rawIndexSelect,
  rawLayerNorm,
  rawLayerNormGrad,
  rawLogSumExp,
  rawMatmul,
  rawNarrow,
  rawOneHot,
  rawPermute,
  rawReduce,
  rawReduceAll,
  rawRmsNorm,
  rawRmsNormGrad,
  rawScatterAdd,
  rawScatterAddRows,
  rawSilu,
  rawSiluGrad,
  rawSoftmax,
  rawSoftmaxGrad,
  rawStackList,
  rawSum,
  rawUnary,
  reshapeRaw,
  sumTo,
} from "./ir.ts"
import { nextSeed, nextStream, randomData } from "./kernels.ts"
import { force } from "./lazy.ts"
import {
  type Broadcast,
  type BroadcastCheck,
  type BroadcastToCheck,
  type Cat,
  type CatCheck,
  type CatN,
  type CatNCheck,
  type DimAt,
  type DimCheck,
  type Drop,
  type ErrorMessage,
  type FlattenCheck,
  type FlattenShape,
  flattenShape,
  type IndexTensor,
  type InferShape,
  type IsDynamic,
  type Last,
  type MatMul,
  type MatMulCheck,
  type NarrowCheck,
  type NestedArray,
  type Permute,
  type PermuteCheck,
  type Prod,
  type Rank1Check,
  type ReduceDim,
  type ResizeDim,
  type ResolveView,
  resolveView,
  type SelectCheck,
  type SelectShape,
  type Shape,
  type Slice,
  type SliceCheck,
  type SliceShape,
  type Squeeze,
  type SqueezeDim,
  type SqueezeDimCheck,
  type Stack,
  type Transpose,
  type TransposeCheck,
  type UnflattenCheck,
  type UnflattenShape,
  unflattenShape,
  type Unsqueeze,
  type UnsqueezeCheck,
  type ViewCheck,
} from "./shape.ts"
import {
  contiguousStrides,
  convertData,
  type DType,
  normalizeDim,
  type NumericArray,
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

export function fromFlat<const Sh extends Shape>(
  data: ArrayLike<number> | ArrayLike<bigint>,
  shape: Sh,
  dtype: DType = "float32",
): Tensor<Sh> {
  return makeRaw(convertData(data, dtype), shape, dtype) as any
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

/** JSON.stringify throws on a bigint, so int64 leaves render as decimal strings. */
function bigintReplacer(_key: string, value: unknown): unknown {
  return typeof value === "bigint" ? value.toString() : value
}

type Dim0<S extends Shape> = S extends [infer A extends number, ...any[]] ? A : never
type Dim1<S extends Shape> = S extends [any, infer B extends number, ...any[]] ? B : never

/** Exact equality, no broadcasting. A fresh S2 keeps `other: Tensor<S>` from resolving S and breaking every `*Check` site. */
type IsSameType<A, B> = (<T>() => T extends A ? 1 : 2) extends (<T>() => T extends B ? 1 : 2) ? true : false

type SameShapeCheck<S extends Shape, S2 extends Shape> =
    IsDynamic<S> extends true ? unknown
  : IsDynamic<S2> extends true ? unknown
  : IsSameType<S, S2> extends true ? unknown
  : ErrorMessage<"shape does not match the receiver">

/** Populated by a static block inside `Tensor`, the only scope that can reach its `#` fields. */
interface TensorInternal {
  sourceOf(t: AnyTensor): TensorStorage
  cpuOf(t: AnyTensor): TypedArray | null
  setCpu(t: AnyTensor, data: TypedArray): void
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
  readonly #source: TensorStorage
  #cpu: TypedArray | null
  #gradNode: GradNode | null = null
  readonly shape: S
  readonly dtype: DType

  grad: Tensor<S> | null = null

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

  /** A live aliased view, not a copy; for `int64` the buffer is really a `BigInt64Array` despite the {@link NumericArray} return type. */
  get data(): NumericArray {
    if (isTracing()) {
      const label = tensorNames.get(this as AnyTensor)
      throw new Error(
        "compile() cannot read tensor values during tracing"
          + (label ? ` (reading "${label}")` : ""),
      )
    }
    if (this.#cpu === null) {
      force(this as AnyTensor)
    }
    return this.#cpu as NumericArray
  }

  get needsGrad(): boolean {
    return this._requiresGrad || this.#gradNode !== null
  }

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
    return makeRaw(
      randomData(
        "uniform",
        prod(shape),
        nextStream(),
        nextSeed(),
        "float32",
      ),
      shape,
      "float32",
    ) as any
  }

  static randn<const Sh extends Shape>(
    shape: Sh,
  ): Tensor<Sh> {
    return makeRaw(
      randomData(
        "normal",
        prod(shape),
        nextStream(),
        nextSeed(),
        "float32",
      ),
      shape,
      "float32",
    ) as any
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

  static indices<const Sh extends Shape>(
    data: ArrayLike<number>,
    shape: Sh,
  ): IndexTensor<Sh> {
    for (let i = 0; i < data.length; i++) {
      if (!Number.isInteger(data[i])) {
        throw new Error(
          `Tensor.indices(): element ${i} is ${data[i]}, not an integer`,
        )
      }
    }
    return fromFlat(data, shape, "int32") as unknown as IndexTensor<Sh>
  }

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
    const out = makeRaw(convertData(this.data, dtype), this.shape, dtype)
    return withGrad(out, "to", [this], g => [g]) as any
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
    return `Tensor(shape=${showShape(this.shape)}, dtype=${this.dtype}, data=${JSON.stringify(this.toArray(), bigintReplacer)})`
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

  private assertMutable(method: string): void {
    if (isTracing()) {
      const label = tensorNames.get(this as AnyTensor)
      throw new Error(
        `${method}() cannot mutate a tensor while compile() is tracing`
          + (label ? ` (mutating "${label}")` : "")
          + ", read snapshot() for a safe copy, or mutate outside the trace",
      )
    }
    if (this.taped) {
      throw new Error(
        `${method}() cannot mutate a tensor of shape ${showShape(this.shape)} `
          + "while an autograd tape is recording through it, call snapshot() "
          + "for a safe copy, or detach() the tensor first",
      )
    }
  }

  fill_(v: number): this {
    this.assertMutable("fill_")
    this.#cpu = filledData(this.numel, this.dtype, v)
    return this
  }

  zero_(): this {
    this.assertMutable("zero_")
    this.#cpu = filledData(this.numel, this.dtype, 0)
    return this
  }

  copy_<S2 extends Shape>(other: Tensor<S2> & SameShapeCheck<S, S2>): this
  copy_(other: AnyTensor): this {
    this.assertMutable("copy_")
    const src = other as AnyTensor
    if (!shapesEqual(this.shape, src.shape)) {
      throw new Error(
        `copy_(): source shape ${showShape(src.shape)} does not match `
          + `destination shape ${showShape(this.shape)}`,
      )
    }
    this.#cpu = src.dtype === this.dtype
      ? (src.data as TypedArray).slice()
      : convertData(src.data, this.dtype)
    return this
  }

  addScaled_<S2 extends Shape>(other: Tensor<S2> & SameShapeCheck<S, S2>, alpha: number): this
  addScaled_(other: AnyTensor, alpha: number): this {
    this.assertMutable("addScaled_")
    const src = other as AnyTensor
    if (!shapesEqual(this.shape, src.shape)) {
      throw new Error(
        `addScaled_(): operand shape ${showShape(src.shape)} does not match `
          + `receiver shape ${showShape(this.shape)}`,
      )
    }
    force(this as AnyTensor)
    const dst = this.#cpu!
    const rhs = src.data
    if (dst instanceof BigInt64Array) {
      for (let i = 0; i < dst.length; i++) {
        dst[i] = BigInt(Math.trunc(Number(dst[i]) + alpha * Number(rhs[i])))
      }
    } else {
      for (let i = 0; i < dst.length; i++) {
        dst[i] = dst[i]! + alpha * rhs[i]!
      }
    }
    return this
  }

  /** A copy that never aliases this tensor's storage, allowed during a `compile()` trace. */
  snapshot(): NumericArray {
    force(this as AnyTensor)
    return (this.#cpu as NumericArray).slice() as NumericArray
  }

  toIndex(): IndexTensor<S> {
    if (this.dtype !== "int32" && this.dtype !== "int64") {
      const data = this.data
      for (let i = 0; i < data.length; i++) {
        if (!Number.isInteger(data[i])) {
          throw new Error(
            `toIndex(): element ${i} is ${data[i]}, not an integer, `
              + "index tensors must be int32/int64 or an integral float",
          )
        }
      }
    }
    return this as unknown as IndexTensor<S>
  }

  // A debug label that printGraph() shows.
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

  /** Ties go to the left operand. */
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

  /** Gradient is 1 inside the clamp and 0 outside. */
  clamp(
    min: number | null,
    max: number | null = null,
  ): Tensor<S> {
    let out = this as AnyTensor
    if (min !== null) out = out.maximum(min)
    if (max !== null) out = out.minimum(max)
    return out as any
  }

  // Comparisons produce 1.0 / 0.0 masks and stop gradients: a step function has zero derivative.
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
    if (b.rank === 1) out = out.squeeze(-1)
    if (self.rank === 1) {
      out = out.squeeze((b.rank === 1 ? -1 : -2) as any)
    }
    return out as any
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

  broadcastTo<const V extends Shape>(
    shape: V & BroadcastToCheck<S, V>,
  ): Tensor<V> {
    if (shapesEqual(this.shape, shape as number[])) {
      return this as any
    }
    const out = rawBroadcastTo(
      this as AnyTensor,
      [...(shape as number[])],
    )
    return withGrad(out, "broadcastTo", [this], g => [
      sumTo(g, [...this.shape]),
    ]) as any
  }

  slice<const Spec extends readonly Slice[]>(
    spec: Spec & SliceCheck<S, Spec>,
  ): Tensor<SliceShape<S, Spec>> {
    if (spec.length !== this.shape.length) {
      throw new Error(
        `slice() expects ${this.shape.length} entries, got ${spec.length}`,
      )
    }
    let out: AnyTensor = this as AnyTensor
    for (let d = 0; d < this.shape.length; d++) {
      const c = spec[d]!
      if (c == null) continue
      const start = typeof c === "number" ? 0 : c[0]
      const length = typeof c === "number" ? c : c[1] - c[0]
      out = out.narrow(d, start, length) as AnyTensor
    }
    return out as any
  }

  narrow<Start extends number, L extends number>(
    dim: 0 & NarrowCheck<S, 0, Start, L>,
    start: Start,
    length: L,
  ): Tensor<ResizeDim<S, 0, L>>
  narrow<D extends number, Start extends number, L extends number>(
    dim: D & DimCheck<S, D> & NarrowCheck<S, D, Start, L>,
    start: Start,
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

  /** `narrow(dim, i, 1).squeeze(dim)`: `[B, T, E].select(1, t)` is `[B, E]`. */
  select<D extends number, I extends number>(
    dim: D & DimCheck<S, D> & SelectCheck<S, D, I>,
    index: I,
  ): Tensor<SelectShape<S, D>> {
    const d = normalizeDim(dim, this.shape.length)
    if (!Number.isInteger(index)) {
      throw new Error(`select() requires an integer index, got ${index}`)
    }
    // `narrow` takes a non-negative start only, so `select(0, -1)` resolves its own index first.
    const at = normalizeDim(index, this.shape[d]!)
    const window = (this as AnyTensor).narrow(d, at, 1) as AnyTensor
    return window.squeeze(d) as Tensor<SelectShape<S, D>>
  }

  /** Gradients flow to the gathered tensor, never to the index. */
  indexSelect<E extends number>(
    index: IndexTensor<[E]>,
  ): Tensor<ResizeDim<S, 0, E>>
  indexSelect<E extends number, D extends number>(
    index: IndexTensor<[E]>,
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

  scatterAdd<L extends number>(
    index: IndexTensor<[Dim0<S>]>,
    length: L,
  ): Tensor<ResizeDim<S, 0, L>>
  scatterAdd<L extends number, D extends number>(
    index: IndexTensor<[DimAt<S, D>]>,
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

  view<const V extends readonly number[]>(
    shape: V & ViewCheck<S, V>,
  ): Tensor<ResolveView<S, V>> {
    const resolved = resolveView(
      [...this.shape],
      shape,
    )
    const out = reshapeRaw(this, resolved)
    return withGrad(out, "view", [this], g => [
      reshapeRaw(g, [...this.shape]),
    ]) as any
  }

  flatten<const F extends number, const T extends number>(
    from: F & FlattenCheck<S, F, T>,
    to: T,
  ): Tensor<FlattenShape<S, F, T>>
  flatten(): Tensor<[Prod<S>]>
  flatten(from?: number, to?: number): AnyTensor {
    const target = from === undefined
      ? [this.numel]
      : flattenShape([...this.shape], from, to ?? from)
    const out = reshapeRaw(this, target)
    return withGrad(out, "flatten", [this], g => [
      reshapeRaw(g, [...this.shape]),
    ]) as any
  }

  unflatten<const D extends number, const Sizes extends readonly number[]>(
    dim: D & UnflattenCheck<S, D, Sizes>,
    sizes: Sizes,
  ): Tensor<UnflattenShape<S, D, Sizes>> {
    const target = unflattenShape([...this.shape], dim as number, sizes)
    const out = reshapeRaw(this, target)
    return withGrad(out, "unflatten", [this], g => [
      reshapeRaw(g, [...this.shape]),
    ]) as any
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

  get T(): S["length"] extends 2 ? Tensor<Transpose<S, 0, 1>> : ErrorMessage<".T is only defined for rank-2 tensors, use transpose(d0, d1)"> {
    if (this.shape.length !== 2) {
      throw new Error(
        ".T is only defined for rank-2 tensors",
      )
    }
    return this.permuteRaw([1, 0]) as any
  }

  permute<const O extends readonly number[]>(
    ...order: O & PermuteCheck<S, O>
  ): Tensor<Permute<S, O>> {
    const rank = this.shape.length
    const dims = order as readonly number[]
    const normalized = dims.map(d => normalizeDim(d, rank))
    if (normalized.length !== rank || new Set(normalized).size !== rank) {
      throw new Error(
        `permute(${dims.join(", ")}) is not a permutation of ${showShape(this.shape)}`,
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
    return rawStackList(tensors as readonly AnyTensor[], dim ?? 0) as any
  }

  /**
   * Stacks a list whose length is only known at run time, where `Tensor.stack` needs a tuple to
   * derive its result shape. The result is `Tensor<Shape>`: a runtime count cannot appear in the type.
   */
  static stackList(
    tensors: readonly AnyTensor[],
    dim = 0,
  ): Tensor<Shape> {
    return rawStackList(tensors, dim) as Tensor<Shape>
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

function filledData(numel: number, dtype: DType, v: number): TypedArray {
  return convertData(new Array(numel).fill(v), dtype)
}

function coerce(
  value: AnyTensor | number,
  like: AnyTensor,
): AnyTensor {
  if (typeof value === "number") {
    return makeRaw(
      convertData([value], like.dtype),
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

/** The tanh approximation of GELU. */
export function gelu<S extends Shape>(x: Tensor<S>): Tensor<S> {
  const a = x as AnyTensor
  const out = rawGelu(a)
  return withGrad(out, "gelu", [a], g => [
    rawGeluGrad(g, a),
  ]) as Tensor<S>
}

export function silu<S extends Shape>(x: Tensor<S>): Tensor<S> {
  const a = x as AnyTensor
  const out = rawSilu(a)
  return withGrad(out, "silu", [a], g => [
    rawSiluGrad(g, a),
  ]) as Tensor<S>
}

/** causal: true folds the decoder mask into the node and needs the last axis of a rank>=2 score matrix. */
export function softmax<
  S extends Shape,
  const D extends number,
>(
  x: Tensor<S>,
  dim: D & DimCheck<S, D>,
  options: { causal?: boolean } = {},
): Tensor<S> {
  const a = x as AnyTensor
  const d = normalizeDim(dim as number, a.shape.length)
  const out = rawSoftmax(a, d, options.causal ?? false)
  return withGrad(out, "softmax", [a], g => [
    rawSoftmaxGrad(g, out, d),
  ]) as Tensor<S>
}

export function layerNorm<S extends Shape>(
  x: Tensor<S>,
  gamma: Tensor<[Last<S>]>,
  beta: Tensor<[Last<S>]>,
  options: { eps?: number } = {},
): Tensor<S> {
  const a = x as AnyTensor
  const w = gamma as AnyTensor
  const b = beta as AnyTensor
  const [y, mean, rstd] = rawLayerNorm(
    a,
    w,
    b,
    options.eps ?? 1e-5,
  ) as [AnyTensor, AnyTensor, AnyTensor]
  return withGrad(y, "layerNorm", [a, w, b], g => rawLayerNormGrad(g, a, w, mean, rstd)) as Tensor<S>
}

export function rmsNorm<S extends Shape>(
  x: Tensor<S>,
  gamma: Tensor<[Last<S>]>,
  options: { eps?: number } = {},
): Tensor<S> {
  const a = x as AnyTensor
  const w = gamma as AnyTensor
  const [y, rstd] = rawRmsNorm(a, w, options.eps ?? 1e-5) as [
    AnyTensor,
    AnyTensor,
  ]
  return withGrad(y, "rmsNorm", [a, w], g => rawRmsNormGrad(g, a, w, rstd)) as Tensor<S>
}

export function crossEntropy<
  N extends number,
  C extends number,
>(
  logits: Tensor<[N, C]>,
  targets: Tensor<[NoInfer<N>]>,
): Tensor<[]> {
  const l = logits as AnyTensor
  const [loss, dlogits] = rawCrossEntropy(
    l,
    targets as AnyTensor,
  ) as [AnyTensor, AnyTensor]
  return withGrad(loss, "crossEntropy", [l], g => [
    rawBinary(dlogits, g, "mul"),
  ]) as Tensor<[]>
}

/** Shifted by the row maximum so it never overflows. */
export function logSumExp<
  S extends Shape,
  const D extends number,
>(
  x: Tensor<S>,
  dim: D & DimCheck<S, D>,
): Tensor<ReduceDim<S, D>>
export function logSumExp<
  S extends Shape,
  const D extends number,
  const K extends boolean,
>(
  x: Tensor<S>,
  dim: D & DimCheck<S, D>,
  keepdim: K,
): Tensor<ReduceDim<S, D, K>>
export function logSumExp(
  x: AnyTensor,
  dim: number,
  keepdim = false,
): AnyTensor {
  const d = normalizeDim(dim, x.shape.length)
  const out = rawLogSumExp(x, d, keepdim)
  const keepShape = x.shape.map(
    (s: number, i: number) => (i === d ? 1 : s),
  )
  return withGrad(out, "logSumExp", [x], g => {
    const gk = keepdim ? g : reshapeRaw(g, keepShape)
    // d/dx log-sum-exp is the softmax of the same axis.
    return [rawBinary(rawSoftmax(x, d, false), gk, "mul")]
  })
}

export function gatherRows<
  S extends Shape,
  I extends Shape,
>(
  table: Tensor<S>,
  index: IndexTensor<I>,
): Tensor<[...I, ...Drop<S, 1>]> {
  const t = table as AnyTensor
  const idx = index as AnyTensor
  const rows = t.shape[0]!
  const out = rawGatherRows(t, idx)
  return withGrad(out, "gatherRows", [t], g => [
    rawScatterAddRows(g, idx, rows),
  ]) as Tensor<[...I, ...Drop<S, 1>]>
}

/** Inverted dropout: survivors are scaled by `1/(1-p)` so the expectation is the identity. */
export function dropout<S extends Shape>(
  x: Tensor<S>,
  p: number,
): Tensor<S> {
  const a = x as AnyTensor
  const [y, mask] = rawDropout(a, p) as [AnyTensor, AnyTensor]
  return withGrad(y, "dropout", [a], g => [
    rawBinary(g, mask, "mul"),
  ]) as Tensor<S>
}
