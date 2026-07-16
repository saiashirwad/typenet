import { Operator } from "tsover-runtime";
import * as typegpuBackend from "./backends/typegpu.ts";
import type { BinaryOp, ReduceOp, TypeGPUStorage, UnaryOp } from "./backends/typegpu.ts";
import { getTypeGPURoot } from "./typegpu.ts";
import type {
  Broadcast,
  BroadcastCheck,
  BroadcastsInto,
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
} from "./shape.ts";

export type DType = "float32" | "float64";
export type Device = "cpu" | "gpu";

export type TensorParams = {
  requires_grad: boolean;
  device: Device;
  dtype: DType;
};

export type DefaultParams = {
  requires_grad: false;
  device: "cpu";
  dtype: "float32";
};

type Clean<T> = { [K in keyof T]: T[K] } & unknown;

type Merge<A, B> = Clean<
  {
    [K in keyof A as K extends keyof B ? never : K]: A[K];
  } & B
>;

export type ShapeOf<T> = T extends Tensor<infer S, any> ? S : never;
export type ParamsOf<T> = T extends Tensor<any, infer P> ? P : never;

type TypedArray = Float32Array | Float64Array;

type CpuStorage = {
  readonly kind: "cpu";
  readonly data: TypedArray;
};

type TensorStorage = CpuStorage | TypeGPUStorage;

export type NestedNumbers = number | readonly NestedNumbers[];

function arrayCtor(dtype: DType) {
  return dtype === "float64" ? Float64Array : Float32Array;
}

function prod(xs: readonly number[]): number {
  let p = 1;
  for (const x of xs) p *= x;
  return p;
}

function shapesEqual(a: readonly number[], b: readonly number[]): boolean {
  return a.length === b.length && a.every((x, i) => x === b[i]);
}

function showShape(s: readonly number[]): string {
  return `[${s.join(", ")}]`;
}

function contiguousStrides(shape: readonly number[]): number[] {
  const strides = new Array<number>(shape.length);
  let acc = 1;
  for (let i = shape.length - 1; i >= 0; i--) {
    strides[i] = acc;
    acc *= shape[i]!;
  }
  return strides;
}

export function broadcastShapes(a: readonly number[], b: readonly number[]): number[] {
  const rank = Math.max(a.length, b.length);
  const out = new Array<number>(rank);
  for (let i = 0; i < rank; i++) {
    const da = a[a.length - 1 - i] ?? 1;
    const db = b[b.length - 1 - i] ?? 1;
    if (da !== db && da !== 1 && db !== 1) throw new Error(`Cannot broadcast ${showShape(a)} with ${showShape(b)}`);
    out[rank - 1 - i] = Math.max(da, db);
  }
  return out;
}

function broadcastStrides(from: readonly number[], to: readonly number[]): number[] {
  const strides = contiguousStrides(from);
  const out = new Array<number>(to.length).fill(0);
  const offset = to.length - from.length;
  for (let i = 0; i < from.length; i++) {
    out[offset + i] = from[i] === 1 ? 0 : strides[i]!;
  }
  return out;
}

function normalizeDim(dim: number, rank: number, extra = 0): number {
  const d = dim < 0 ? rank + extra + dim : dim;
  if (d < 0 || d >= rank + extra) throw new Error(`Dimension ${dim} out of range for rank ${rank}`);
  return d;
}

type AnyTensor = Tensor<any, any>;

interface GradNode {
  name: string;
  inputs: AnyTensor[];

  backward: (grad: AnyTensor) => (AnyTensor | null)[];
}

let gradEnabled = true;

export function noGrad<T>(fn: () => T): T {
  const prev = gradEnabled;
  gradEnabled = false;
  try {
    return fn();
  } finally {
    gradEnabled = prev;
  }
}

function sumTo(t: AnyTensor, shape: number[]): AnyTensor {
  if (shapesEqual(t.shape, shape)) return t;
  let out = t;

  while (out.shape.length > shape.length) out = rawSum(out, 0, false);

  for (let i = 0; i < shape.length; i++) if (shape[i] === 1 && out.shape[i] !== 1) out = rawSum(out, i, true);
  return out;
}

function makeRaw(data: TypedArray, shape: readonly number[], dtype: DType, device: Device): AnyTensor {
  if (device === "gpu") {
    if (dtype === "float64") throw new Error("TypeGPU does not support float64 tensors");
    return makeStorage(typegpuBackend.upload(getTypeGPURoot(), data as Float32Array), shape, dtype);
  }
  return makeStorage({ kind: "cpu", data }, shape, dtype);
}

function makeStorage(storage: TensorStorage, shape: readonly number[], dtype: DType): AnyTensor {
  return new (Tensor as any)(storage, [...shape], dtype, INTERNAL);
}

function applyBinary(op: BinaryOp, x: number, y: number, parameter: number): number {
  switch (op) {
    case "add":
      return x + y;
    case "sub":
      return x - y;
    case "mul":
      return x * y;
    case "div":
      return x / y;
    case "negDiv":
      return -x / y;
    case "halfDiv":
      return (0.5 * x) / y;
    case "mulSign":
      return x * Math.sign(y);
    case "reluGrad":
      return y > 0 ? x : 0;
    case "leakyReluGrad":
      return y > 0 ? x : parameter * x;
    case "sigmoidGrad":
      return x * y * (1 - y);
    case "tanhGrad":
      return x * (1 - y * y);
  }
}

function applyUnary(op: UnaryOp, x: number, parameter: number): number {
  switch (op) {
    case "pow":
      return x ** parameter;
    case "neg":
      return -x;
    case "exp":
      return Math.exp(x);
    case "log":
      return Math.log(x);
    case "sqrt":
      return Math.sqrt(x);
    case "abs":
      return Math.abs(x);
    case "relu":
      return x > 0 ? x : 0;
    case "leakyRelu":
      return x > 0 ? x : parameter * x;
    case "sigmoid":
      return 1 / (1 + Math.exp(-x));
    case "tanh":
      return Math.tanh(x);
    case "scalePowGrad":
      return parameter * x ** (parameter - 1);
  }
}

function checkDevices(a: AnyTensor, b: AnyTensor): void {
  if (a.device !== b.device) throw new Error(`Mixed-device operation: ${a.device} tensor with ${b.device} tensor`);
}

function gpuStorage(t: AnyTensor): TypeGPUStorage {
  if (t._storage.kind !== "typegpu") throw new Error("Expected a TypeGPU tensor");
  return t._storage;
}

function rawBinary(a: AnyTensor, b: AnyTensor, op: BinaryOp, parameter = 0): AnyTensor {
  checkDevices(a, b);
  const outShape = broadcastShapes(a.shape, b.shape);
  const dtype: DType = a.dtype === "float64" || b.dtype === "float64" ? "float64" : "float32";
  const rank = outShape.length;
  const sa = broadcastStrides(a.shape, outShape);
  const sb = broadcastStrides(b.shape, outShape);
  if (a.device === "gpu") {
    if (dtype !== "float32") throw new Error("TypeGPU does not support float64 tensors");
    return makeStorage(
      typegpuBackend.binary(gpuStorage(a), gpuStorage(b), op, outShape, sa, sb, parameter),
      outShape,
      dtype,
    );
  }
  const n = prod(outShape);
  const out = new (arrayCtor(dtype))(n);
  const idx = new Array(rank).fill(0);
  let offA = 0;
  let offB = 0;
  const ad = a.data;
  const bd = b.data;
  for (let i = 0; i < n; i++) {
    out[i] = applyBinary(op, ad[offA]!, bd[offB]!, parameter);
    for (let d = rank - 1; d >= 0; d--) {
      idx[d]++;
      offA += sa[d]!;
      offB += sb[d]!;
      if (idx[d] < outShape[d]!) break;
      idx[d] = 0;
      offA -= sa[d]! * outShape[d]!;
      offB -= sb[d]! * outShape[d]!;
    }
  }
  return makeRaw(out, outShape, dtype, a.device);
}

function rawUnary(a: AnyTensor, op: UnaryOp, parameter = 0): AnyTensor {
  if (a.device === "gpu") return makeStorage(typegpuBackend.unary(gpuStorage(a), op, parameter), a.shape, a.dtype);
  const out = new (arrayCtor(a.dtype))(a.data.length);
  for (let i = 0; i < a.data.length; i++) out[i] = applyUnary(op, a.data[i]!, parameter);
  return makeRaw(out, a.shape, a.dtype, a.device);
}

function rawSum(a: AnyTensor, dim: number, keepdim: boolean): AnyTensor {
  return rawReduce(a, dim, keepdim, "sum");
}

function rawReduce(a: AnyTensor, dim: number, keepdim: boolean, op: ReduceOp): AnyTensor {
  const d = normalizeDim(dim, a.shape.length);
  const outShape = a.shape.filter((_: number, i: number) => i !== d);
  const keepShape = a.shape.map((s: number, i: number) => (i === d ? 1 : s));
  const n = prod(outShape);
  const strides = contiguousStrides(a.shape);
  const outer = prod(a.shape.slice(0, d));
  const dimSize = a.shape[d]!;
  const inner = strides[d]!;
  if (a.device === "gpu")
    return makeStorage(
      typegpuBackend.reduce(gpuStorage(a), op, outer, dimSize, inner),
      keepdim ? keepShape : outShape,
      a.dtype,
    );
  const init = op === "sum" ? 0 : -Infinity;
  const out = new (arrayCtor(a.dtype))(n).fill(init);
  const ad = a.data;
  let o = 0;
  for (let i = 0; i < outer; i++) {
    for (let k = 0; k < inner; k++) {
      let acc = init;
      const base = i * dimSize * inner + k;
      let bestIdx = 0;
      for (let j = 0; j < dimSize; j++) {
        const value = ad[base + j * inner]!;
        if (op === "sum") acc += value;
        else if (value > acc) {
          acc = value;
          bestIdx = j;
        }
      }
      out[o++] = op === "argmax" ? bestIdx : acc;
    }
  }
  return makeRaw(out, keepdim ? keepShape : outShape, a.dtype, a.device);
}

function rawBroadcastTo(a: AnyTensor, shape: number[]): AnyTensor {
  if (shapesEqual(a.shape, shape)) return a;
  const n = prod(shape);
  const rank = shape.length;
  const sa = broadcastStrides(a.shape, shape);
  if (a.device === "gpu") return makeStorage(typegpuBackend.broadcast(gpuStorage(a), shape, sa), shape, a.dtype);
  const out = new (arrayCtor(a.dtype))(n);
  const idx = new Array(rank).fill(0);
  let off = 0;
  const ad = a.data;
  for (let i = 0; i < n; i++) {
    out[i] = ad[off]!;
    for (let d = rank - 1; d >= 0; d--) {
      idx[d]++;
      off += sa[d]!;
      if (idx[d] < shape[d]!) break;
      idx[d] = 0;
      off -= sa[d]! * shape[d]!;
    }
  }
  return makeRaw(out, shape, a.dtype, a.device);
}

function rawPermute(a: AnyTensor, order: number[]): AnyTensor {
  const rank = a.shape.length;
  const outShape = order.map((i) => a.shape[i]!);
  const inStrides = contiguousStrides(a.shape);
  const readStrides = order.map((i) => inStrides[i]!);
  const n = a.numel;
  if (a.device === "gpu")
    return makeStorage(typegpuBackend.permute(gpuStorage(a), outShape, readStrides), outShape, a.dtype);
  const out = new (arrayCtor(a.dtype))(n);
  const idx = new Array(rank).fill(0);
  let off = 0;
  const ad = a.data;
  for (let i = 0; i < n; i++) {
    out[i] = ad[off]!;
    for (let d = rank - 1; d >= 0; d--) {
      idx[d]++;
      off += readStrides[d]!;
      if (idx[d] < outShape[d]!) break;
      idx[d] = 0;
      off -= readStrides[d]! * outShape[d]!;
    }
  }
  return makeRaw(out, outShape, a.dtype, a.device);
}

function rawMatmul(a: AnyTensor, b: AnyTensor): AnyTensor {
  checkDevices(a, b);
  const ar = a.shape.length;
  const br = b.shape.length;
  const m = a.shape[ar - 2]!;
  const k = a.shape[ar - 1]!;
  const k2 = b.shape[br - 2]!;
  const n = b.shape[br - 1]!;
  if (k !== k2)
    throw new Error(`matmul: inner dimensions do not match (${showShape(a.shape)} @ ${showShape(b.shape)})`);
  const batchA = a.shape.slice(0, -2);
  const batchB = b.shape.slice(0, -2);
  const batch = broadcastShapes(batchA, batchB);
  const outShape = [...batch, m, n];
  const dtype: DType = a.dtype === "float64" || b.dtype === "float64" ? "float64" : "float32";
  const batchCount = prod(batch);
  const saBatch = broadcastStrides(batchA, batch);
  const sbBatch = broadcastStrides(batchB, batch);
  if (a.device === "gpu") {
    if (dtype !== "float32") throw new Error("TypeGPU does not support float64 tensors");
    return makeStorage(
      typegpuBackend.matmul(gpuStorage(a), gpuStorage(b), batch, saBatch, sbBatch, m, k, n),
      outShape,
      dtype,
    );
  }
  const out = new (arrayCtor(dtype))(batchCount * m * n);

  const aMat = m * k;
  const bMat = k * n;
  const rank = batch.length;
  const idx = new Array(rank).fill(0);
  const ad = a.data;
  const bd = b.data;
  for (let bi = 0; bi < batchCount; bi++) {
    let cellA = 0;
    let cellB = 0;
    for (let d = 0; d < rank; d++) {
      cellA += idx[d]! * saBatch[d]!;
      cellB += idx[d]! * sbBatch[d]!;
    }
    const baseA = cellA * aMat;
    const baseB = cellB * bMat;
    const baseO = bi * m * n;
    for (let i = 0; i < m; i++) {
      for (let j = 0; j < n; j++) {
        let acc = 0;
        for (let p = 0; p < k; p++) acc += ad[baseA + i * k + p]! * bd[baseB + p * n + j]!;
        out[baseO + i * n + j] = acc;
      }
    }
    for (let d = rank - 1; d >= 0; d--) {
      idx[d]++;
      if (idx[d] < batch[d]!) break;
      idx[d] = 0;
    }
  }
  return makeRaw(out, outShape, dtype, a.device);
}

function rawNarrow(a: AnyTensor, dim: number, start: number, length: number): AnyTensor {
  const d = normalizeDim(dim, a.shape.length);
  const outShape = a.shape.map((s: number, i: number) => (i === d ? length : s));
  const strides = contiguousStrides(a.shape);
  const outer = prod(a.shape.slice(0, d));
  const inner = strides[d]!;
  const dimSize = a.shape[d]!;
  if (a.device === "gpu")
    return makeStorage(typegpuBackend.narrow(gpuStorage(a), outer, dimSize, inner, start, length), outShape, a.dtype);
  const out = new (arrayCtor(a.dtype))(prod(outShape));
  const ad = a.data;
  let o = 0;
  for (let i = 0; i < outer; i++) {
    const base = i * dimSize * inner + start * inner;
    for (let j = 0; j < length; j++) {
      for (let kk = 0; kk < inner; kk++) {
        out[o++] = ad[base + j * inner + kk]!;
      }
    }
  }
  return makeRaw(out, outShape, a.dtype, a.device);
}

function withGrad(
  result: AnyTensor,
  name: string,
  inputs: AnyTensor[],
  backward: (grad: AnyTensor) => (AnyTensor | null)[],
): AnyTensor {
  if (gradEnabled && inputs.some((t) => t.needsGrad)) {
    result.gradNode = { name, inputs, backward };
  }
  return result;
}

const INTERNAL = Symbol("tensor-internal");

function flatten(value: NestedNumbers, out: number[], shape: number[], depth: number): void {
  if (typeof value === "number") {
    if (depth !== shape.length) throw new Error("Ragged nested array passed to tensor()");
    out.push(value);
    return;
  }
  if (depth === shape.length) shape.push(value.length);
  else if (shape[depth] !== value.length) throw new Error("Ragged nested array passed to tensor()");
  for (const v of value) flatten(v, out, shape, depth + 1);
}

export class Tensor<S extends Shape = number[], P extends TensorParams = DefaultParams> {
  readonly _storage: TensorStorage;
  readonly shape: S;
  readonly dtype: DType;
  readonly device: Device;

  grad: Tensor<S, P> | null = null;

  requiresGrad = false;

  gradNode: GradNode | null = null;

  constructor(storage: TensorStorage, shape: number[], dtype: DType, internal: typeof INTERNAL) {
    if (internal !== INTERNAL) throw new Error("Use Tensor.of / zeros / ones / randn to create tensors");
    const length = storage.kind === "cpu" ? storage.data.length : storage.length;
    if (length !== prod(shape)) throw new Error(`Data length ${length} does not match shape ${showShape(shape)}`);
    this._storage = storage;
    this.shape = shape as S;
    this.dtype = dtype;
    this.device = storage.kind === "cpu" ? "cpu" : "gpu";
  }

  get data(): TypedArray {
    if (this._storage.kind !== "cpu")
      throw new Error("GPU tensor data is not host-accessible; use await tensor.read()");
    return this._storage.data;
  }

  get needsGrad(): boolean {
    return this.requiresGrad || this.gradNode !== null;
  }

  get rank(): S["length"] {
    return this.shape.length;
  }

  get numel(): number {
    return prod(this.shape);
  }

  static of<const V extends NestedNumbers>(value: V): Tensor<InferShape<V>, DefaultParams> {
    const flat: number[] = [];
    const shape: number[] = [];
    flatten(value, flat, shape, 0);
    return makeRaw(Float32Array.from(flat), shape, "float32", "cpu") as any;
  }

  static full<const Sh extends Shape>(shape: Sh, value: number): Tensor<Sh, DefaultParams> {
    const data = new Float32Array(prod(shape)).fill(value);
    return makeRaw(data, shape, "float32", "cpu") as any;
  }

  static zeros<const Sh extends Shape>(shape: Sh): Tensor<Sh, DefaultParams> {
    return Tensor.full(shape, 0);
  }

  static ones<const Sh extends Shape>(shape: Sh): Tensor<Sh, DefaultParams> {
    return Tensor.full(shape, 1);
  }

  static rand<const Sh extends Shape>(shape: Sh): Tensor<Sh, DefaultParams> {
    const data = new Float32Array(prod(shape));
    for (let i = 0; i < data.length; i++) data[i] = Math.random();
    return makeRaw(data, shape, "float32", "cpu") as any;
  }

  static randn<const Sh extends Shape>(shape: Sh): Tensor<Sh, DefaultParams> {
    const data = new Float32Array(prod(shape));
    for (let i = 0; i < data.length; i += 2) {
      const u = 1 - Math.random();
      const v = Math.random();
      const r = Math.sqrt(-2 * Math.log(u));
      data[i] = r * Math.cos(2 * Math.PI * v);
      if (i + 1 < data.length) data[i + 1] = r * Math.sin(2 * Math.PI * v);
    }
    return makeRaw(data, shape, "float32", "cpu") as any;
  }

  static eye<const N extends number>(n: N): Tensor<[N, N], DefaultParams> {
    const data = new Float32Array(n * n);
    for (let i = 0; i < n; i++) data[i * n + i] = 1;
    return makeRaw(data, [n, n], "float32", "cpu") as any;
  }

  static arange<const N extends number>(n: N): Tensor<[N], DefaultParams> {
    const data = new Float32Array(n);
    for (let i = 0; i < n; i++) data[i] = i;
    return makeRaw(data, [n], "float32", "cpu") as any;
  }

  static scalar(value: number): Tensor<[], DefaultParams> {
    return makeRaw(Float32Array.of(value), [], "float32", "cpu") as any;
  }

  requires_grad(): Tensor<S, Merge<P, { requires_grad: true }>> {
    this.requiresGrad = true;
    return this as any;
  }

  no_grad(): Tensor<S, Merge<P, { requires_grad: false }>> {
    this.requiresGrad = false;
    return this as any;
  }

  gpu(): Tensor<S, Merge<P, { device: "gpu" }>> {
    if (this.device === "gpu") return this as any;
    if (this.dtype === "float64") throw new Error("TypeGPU does not support float64 tensors");
    const out = makeStorage(typegpuBackend.upload(getTypeGPURoot(), this.data as Float32Array), this.shape, this.dtype);
    out.requiresGrad = this.requiresGrad;
    return out as any;
  }

  cpu(): Tensor<S, Merge<P, { device: "cpu" }>> {
    if (this.device === "cpu") return this as any;
    throw new Error("cpu() cannot synchronously read GPU data; use await tensor.toCPU()");
  }

  async toCPU(): Promise<Tensor<S, Merge<P, { device: "cpu" }>>> {
    if (this._storage.kind === "cpu") return this as any;
    const data = await typegpuBackend.read(this._storage);
    const out = makeRaw(data, this.shape, this.dtype, "cpu");
    out.requiresGrad = this.requiresGrad;
    return out as any;
  }

  async read(): Promise<TypedArray> {
    if (this._storage.kind === "cpu") return this._storage.data.slice();
    return typegpuBackend.read(this._storage);
  }

  write(values: ArrayLike<number>): this {
    if (values.length !== this.numel) throw new Error(`write() expected ${this.numel} values, got ${values.length}`);
    if (this._storage.kind === "cpu") this._storage.data.set(Array.from(values));
    else typegpuBackend.write(this._storage, values);
    return this;
  }

  dispose(): void {
    if (this._storage.kind === "typegpu") typegpuBackend.dispose(this._storage);
  }

  to<const D extends DType>(dtype: D): Tensor<S, Merge<P, { dtype: D }>> {
    if (dtype === this.dtype) return this as any;
    if (this.device === "gpu") throw new Error("GPU dtype conversion is unavailable; use await toCPU() first");
    const data = arrayCtor(dtype).from(this.data);
    return makeMovedData(this, data, dtype) as any;
  }

  item(): number {
    if (this.numel !== 1) throw new Error(`item() requires a one-element tensor, got shape ${showShape(this.shape)}`);
    return this.data[0]!;
  }

  get(...indices: number[]): number {
    if (indices.length !== this.shape.length)
      throw new Error(`get() expects ${this.shape.length} indices, got ${indices.length}`);
    const strides = contiguousStrides(this.shape);
    let off = 0;
    for (let i = 0; i < indices.length; i++) {
      const idx = normalizeDim(indices[i]!, this.shape[i]!);
      off += idx * strides[i]!;
    }
    return this.data[off]!;
  }

  toArray(): NestedArray<S> {
    const build = (dim: number, offset: number): any => {
      if (dim === this.shape.length) return this.data[offset]!;
      const stride = contiguousStrides(this.shape)[dim]!;
      const out = new Array(this.shape[dim]!);
      for (let i = 0; i < this.shape[dim]!; i++) out[i] = build(dim + 1, offset + i * stride);
      return out;
    };
    return build(0, 0);
  }

  toString(): string {
    if (this.device === "gpu") return `Tensor(shape=${showShape(this.shape)}, dtype=${this.dtype}, device=gpu)`;
    return `Tensor(shape=${showShape(this.shape)}, dtype=${this.dtype}, data=${JSON.stringify(this.toArray())})`;
  }

  detach(): Tensor<S, Merge<P, { requires_grad: false }>> {
    return makeStorage(this._storage, this.shape, this.dtype) as any;
  }

  clone(): Tensor<S, P> {
    const t =
      this._storage.kind === "cpu"
        ? makeRaw(this.data.slice(), this.shape, this.dtype, "cpu")
        : makeStorage(typegpuBackend.clone(this._storage), this.shape, this.dtype);
    return withGrad(t, "clone", [this], (g) => [g]) as any;
  }

  backward(gradient?: Tensor<S, any>): void {
    if (!this.needsGrad) throw new Error("backward() on a tensor that does not require grad");
    let seed: AnyTensor;
    if (gradient) {
      if (!shapesEqual(gradient.shape, this.shape))
        throw new Error(
          `backward() gradient shape ${showShape(gradient.shape)} does not match ${showShape(this.shape)}`,
        );
      if (gradient.device !== this.device)
        throw new Error(`backward() gradient is on ${gradient.device}, expected ${this.device}`);
      seed = gradient;
    } else {
      if (this.numel !== 1) throw new Error("backward() without a gradient requires a scalar output");
      seed =
        this.device === "cpu"
          ? makeRaw(new (arrayCtor(this.dtype))(this.numel).fill(1), this.shape, this.dtype, "cpu")
          : makeStorage(typegpuBackend.fill(getTypeGPURoot(), this.numel, 1), this.shape, this.dtype);
    }

    const topo: AnyTensor[] = [];
    const seen = new Set<AnyTensor>();
    const visit = (t: AnyTensor) => {
      if (seen.has(t)) return;
      seen.add(t);
      if (t.gradNode) for (const input of t.gradNode.inputs) visit(input);
      topo.push(t);
    };
    visit(this);

    const grads = new Map<AnyTensor, AnyTensor>();
    grads.set(this, seed);

    noGrad(() => {
      for (let i = topo.length - 1; i >= 0; i--) {
        const t = topo[i]!;
        const g = grads.get(t);
        if (!g) continue;
        if (t.gradNode) {
          const inputGrads = t.gradNode.backward(g);
          t.gradNode.inputs.forEach((input, j) => {
            const ig = inputGrads[j];
            if (!ig || !input.needsGrad) return;
            const existing = grads.get(input);
            grads.set(input, existing ? rawBinary(existing, ig, "add") : ig);
          });
        } else if (t.requiresGrad) {
          t.grad = t.grad ? (rawBinary(t.grad, g, "add") as any) : (g as any);
        }
      }
    });
  }

  zeroGrad(): void {
    this.grad = null;
  }

  add(other: number): Tensor<S, P>;
  add<S2 extends Shape>(other: Tensor<S2, any> & BroadcastCheck<S, S2>): Tensor<Broadcast<S, S2>, P>;
  add(other: AnyTensor | number): AnyTensor {
    const b = coerce(other, this);
    const out = rawBinary(this, b, "add");
    return withGrad(out, "add", [this, b], (g) => [sumTo(g, this.shape), sumTo(g, b.shape)]);
  }

  sub(other: number): Tensor<S, P>;
  sub<S2 extends Shape>(other: Tensor<S2, any> & BroadcastCheck<S, S2>): Tensor<Broadcast<S, S2>, P>;
  sub(other: AnyTensor | number): AnyTensor {
    const b = coerce(other, this);
    const out = rawBinary(this, b, "sub");
    return withGrad(out, "sub", [this, b], (g) => [sumTo(g, this.shape), sumTo(rawUnary(g, "neg"), b.shape)]);
  }

  mul(other: number): Tensor<S, P>;
  mul<S2 extends Shape>(other: Tensor<S2, any> & BroadcastCheck<S, S2>): Tensor<Broadcast<S, S2>, P>;
  mul(other: AnyTensor | number): AnyTensor {
    const b = coerce(other, this);
    const out = rawBinary(this, b, "mul");
    return withGrad(out, "mul", [this, b], (g) => [
      sumTo(rawBinary(g, b, "mul"), this.shape),
      sumTo(rawBinary(g, this, "mul"), b.shape),
    ]);
  }

  div(other: number): Tensor<S, P>;
  div<S2 extends Shape>(other: Tensor<S2, any> & BroadcastCheck<S, S2>): Tensor<Broadcast<S, S2>, P>;
  div(other: AnyTensor | number): AnyTensor {
    const b = coerce(other, this);
    const out = rawBinary(this, b, "div");
    return withGrad(out, "div", [this, b], (g) => [
      sumTo(rawBinary(g, b, "div"), this.shape),
      sumTo(rawBinary(rawBinary(g, out, "mul"), b, "negDiv"), b.shape),
    ]);
  }

  pow(exponent: number): Tensor<S, P> {
    const out = rawUnary(this, "pow", exponent);
    return withGrad(out, "pow", [this], (g) => [rawBinary(g, rawUnary(this, "scalePowGrad", exponent), "mul")]) as any;
  }

  neg(): Tensor<S, P> {
    const out = rawUnary(this, "neg");
    return withGrad(out, "neg", [this], (g) => [rawUnary(g, "neg")]) as any;
  }

  exp(): Tensor<S, P> {
    const out = rawUnary(this, "exp");
    return withGrad(out, "exp", [this], (g) => [rawBinary(g, out, "mul")]) as any;
  }

  log(): Tensor<S, P> {
    const out = rawUnary(this, "log");
    return withGrad(out, "log", [this], (g) => [rawBinary(g, this, "div")]) as any;
  }

  sqrt(): Tensor<S, P> {
    const out = rawUnary(this, "sqrt");
    return withGrad(out, "sqrt", [this], (g) => [rawBinary(g, out, "halfDiv")]) as any;
  }

  abs(): Tensor<S, P> {
    const out = rawUnary(this, "abs");
    return withGrad(out, "abs", [this], (g) => [rawBinary(g, this, "mulSign")]) as any;
  }

  relu(): Tensor<S, P> {
    const out = rawUnary(this, "relu");
    return withGrad(out, "relu", [this], (g) => [rawBinary(g, this, "reluGrad")]) as any;
  }

  leakyRelu(negativeSlope = 0.01): Tensor<S, P> {
    const out = rawUnary(this, "leakyRelu", negativeSlope);
    return withGrad(out, "leakyRelu", [this], (g) => [rawBinary(g, this, "leakyReluGrad", negativeSlope)]) as any;
  }

  sigmoid(): Tensor<S, P> {
    const out = rawUnary(this, "sigmoid");
    return withGrad(out, "sigmoid", [this], (g) => [rawBinary(g, out, "sigmoidGrad")]) as any;
  }

  tanh(): Tensor<S, P> {
    const out = rawUnary(this, "tanh");
    return withGrad(out, "tanh", [this], (g) => [rawBinary(g, out, "tanhGrad")]) as any;
  }

  softmax<D extends number>(dim: D & DimCheck<S, D>): Tensor<S, P> {
    const shifted = this.sub(this.max(dim as number as any, true).detach() as any);
    const e = shifted.exp();
    return e.div((e as AnyTensor).sum(dim as any, true) as any) as any;
  }

  logSoftmax<D extends number>(dim: D & DimCheck<S, D>): Tensor<S, P> {
    const shifted = this.sub(this.max(dim as number as any, true).detach() as any) as AnyTensor;
    const logSumExp = shifted
      .exp()
      .sum(dim as any, true)
      .log();
    return shifted.sub(logSumExp as any) as any;
  }

  matmul<S2 extends Shape>(other: Tensor<S2, any> & MatMulCheck<S, S2>): Tensor<MatMul<S, S2>, P> {
    const self = this as AnyTensor;
    const b = other as AnyTensor;
    if (self.rank === 0 || b.rank === 0) throw new Error("matmul requires rank >= 1 operands");
    const A = self.rank === 1 ? self.unsqueeze(0) : self;
    const B = b.rank === 1 ? b.unsqueeze(-1) : b;
    let out = matmul2(A, B);
    if (b.rank === 1) out = out.squeezeDim(-1);
    if (self.rank === 1) out = out.squeezeDim((b.rank === 1 ? -1 : -2) as any);
    return out as any;
  }

  dot<S2 extends Shape>(other: Tensor<S2, any> & MatMulCheck<S, S2>): Tensor<MatMul<S, S2>, P> {
    return this.matmul(other);
  }

  sum(): Tensor<[], P>;
  sum<D extends number>(dim: D & DimCheck<S, D>): Tensor<ReduceDim<S, D>, P>;
  sum<D extends number, const K extends boolean>(dim: D & DimCheck<S, D>, keepdim: K): Tensor<ReduceDim<S, D, K>, P>;
  sum(dim?: number, keepdim = false): AnyTensor {
    if (dim === undefined) {
      let out: AnyTensor;
      if (this.device === "gpu")
        out = makeStorage(typegpuBackend.reduce(gpuStorage(this), "sum", 1, this.numel, 1), [], this.dtype);
      else {
        let acc = 0;
        for (let i = 0; i < this.data.length; i++) acc += this.data[i]!;
        out = makeRaw(arrayCtor(this.dtype).of(acc), [], this.dtype, "cpu");
      }
      return withGrad(out, "sum", [this], (g) => [rawBroadcastTo(g, [...this.shape])]);
    }
    const d = normalizeDim(dim, this.shape.length);
    const out = rawSum(this, d, keepdim);
    const keepShape = this.shape.map((s, i) => (i === d ? 1 : s));
    return withGrad(out, "sum", [this], (g) => {
      const gk = keepdim ? g : reshapeRaw(g, keepShape);
      return [rawBroadcastTo(gk, [...this.shape])];
    });
  }

  mean(): Tensor<[], P>;
  mean<D extends number>(dim: D & DimCheck<S, D>): Tensor<ReduceDim<S, D>, P>;
  mean<D extends number, const K extends boolean>(dim: D & DimCheck<S, D>, keepdim: K): Tensor<ReduceDim<S, D, K>, P>;
  mean(dim?: number, keepdim = false): AnyTensor {
    if (dim === undefined) return (this.sum() as AnyTensor).div(this.numel);
    const d = normalizeDim(dim, this.shape.length);
    return (this as AnyTensor).sum(d as any, keepdim as any).div(this.shape[d]!);
  }

  max(): Tensor<[], P>;
  max<D extends number>(dim: D & DimCheck<S, D>): Tensor<ReduceDim<S, D>, P>;
  max<D extends number, const K extends boolean>(dim: D & DimCheck<S, D>, keepdim: K): Tensor<ReduceDim<S, D, K>, P>;
  max(dim?: number, keepdim = false): AnyTensor {
    if (dim === undefined) {
      if (this.device === "gpu")
        return makeStorage(typegpuBackend.reduce(gpuStorage(this), "max", 1, this.numel, 1), [], this.dtype);
      let m = -Infinity;
      for (let i = 0; i < this.data.length; i++) if (this.data[i]! > m) m = this.data[i]!;
      return makeRaw(arrayCtor(this.dtype).of(m), [], this.dtype, "cpu");
    }
    return rawReduce(this, dim, keepdim, "max");
  }

  argmax<D extends number>(dim: D & DimCheck<S, D>): Tensor<ReduceDim<S, D>, P> {
    const d = normalizeDim(dim, this.shape.length);
    return rawReduce(this, d, false, "argmax") as any;
  }

  oneHot(classes: number): Tensor<[number, number], P> {
    if (this.rank !== 1) throw new Error("oneHot() requires a rank-1 tensor");
    if (!Number.isInteger(classes) || classes <= 0)
      throw new Error(`oneHot() requires a positive class count, got ${classes}`);
    if (this.device === "gpu")
      return makeStorage(
        typegpuBackend.oneHot(gpuStorage(this), this.numel, classes),
        [this.numel, classes],
        this.dtype,
      ) as any;
    const out = new (arrayCtor(this.dtype))(this.numel * classes);
    for (let i = 0; i < this.numel; i++) {
      const target = this.data[i]!;
      if (!Number.isInteger(target) || target < 0 || target >= classes)
        throw new Error(`oneHot: target ${target} out of range for ${classes} classes`);
      out[i * classes + target] = 1;
    }
    return makeRaw(out, [this.numel, classes], this.dtype, "cpu") as any;
  }

  view<const V extends number[]>(shape: V & ViewCheck<S, V>): Tensor<ResolveView<S, V>, P> {
    const resolved = resolveViewRuntime([...this.shape], shape as number[]);
    const out = reshapeRaw(this, resolved);
    return withGrad(out, "view", [this], (g) => [reshapeRaw(g, [...this.shape])]) as any;
  }

  reshape<const V extends number[]>(shape: V & ViewCheck<S, V>): Tensor<ResolveView<S, V>, P> {
    return this.view(shape as any) as any;
  }

  squeeze(): Tensor<Squeeze<S>, P> {
    const target = this.shape.filter((s) => s !== 1);
    const out = reshapeRaw(this, target);
    return withGrad(out, "squeeze", [this], (g) => [reshapeRaw(g, [...this.shape])]) as any;
  }

  squeezeDim<D extends number>(dim: D & SqueezeDimCheck<S, D>): Tensor<SqueezeDim<S, D>, P> {
    const d = normalizeDim(dim as number, this.shape.length);
    if (this.shape[d] !== 1) throw new Error(`Cannot squeeze dim ${dim} of ${showShape(this.shape)}: size is not 1`);
    const target = this.shape.filter((_, i) => i !== d);
    const out = reshapeRaw(this, target);
    return withGrad(out, "squeeze", [this], (g) => [reshapeRaw(g, [...this.shape])]) as any;
  }

  unsqueeze<D extends number>(dim: D & UnsqueezeCheck<S, D>): Tensor<Unsqueeze<S, D>, P> {
    const d = normalizeDim(dim as number, this.shape.length, 1);
    const target = [...this.shape];
    target.splice(d, 0, 1);
    const out = reshapeRaw(this, target);
    return withGrad(out, "unsqueeze", [this], (g) => [reshapeRaw(g, [...this.shape])]) as any;
  }

  transpose<D0 extends number, D1 extends number>(
    dim0: D0 & TransposeCheck<S, D0, D1>,
    dim1: D1,
  ): Tensor<Transpose<S, D0, D1>, P> {
    const rank = this.shape.length;
    const a = normalizeDim(dim0 as number, rank);
    const b = normalizeDim(dim1, rank);
    const order = [...Array(rank).keys()];
    [order[a], order[b]] = [order[b]!, order[a]!];
    return this.permuteRaw(order) as any;
  }

  get T(): S["length"] extends 2
    ? Tensor<Transpose<S, 0, 1>, P>
    : ErrorMessage<".T is only defined for rank-2 tensors — use transpose(d0, d1)"> {
    if (this.shape.length !== 2) throw new Error(".T is only defined for rank-2 tensors");
    return this.permuteRaw([1, 0]) as any;
  }

  permute<const O extends number[]>(...order: O & PermuteCheck<S, O>): Tensor<Permute<S, O>, P> {
    const rank = this.shape.length;
    const normalized = (order as number[]).map((d) => normalizeDim(d, rank));
    if (normalized.length !== rank || new Set(normalized).size !== rank)
      throw new Error(`permute(${(order as number[]).join(", ")}) is not a permutation of ${showShape(this.shape)}`);
    return this.permuteRaw(normalized) as any;
  }

  private permuteRaw(order: number[]): AnyTensor {
    const out = rawPermute(this, order);
    const inverse = new Array<number>(order.length);
    order.forEach((d, i) => (inverse[d] = i));
    return withGrad(out, "permute", [this], (g) => [rawPermute(g, inverse)]);
  }

  static stack<const T extends readonly [AnyTensor, ...AnyTensor[]], const D extends number = 0>(
    tensors: T & StackCheck<T>,
    dim?: D,
  ): Tensor<Stack<ShapeOf<T[0]>, T["length"], D>, ParamsOf<T[0]>> {
    const ts = tensors as readonly AnyTensor[];
    const first = ts[0]!;
    for (const t of ts)
      if (!shapesEqual(t.shape, first.shape))
        throw new Error(`stack: all tensors must share a shape (${showShape(first.shape)} vs ${showShape(t.shape)})`);
    const unsqueezed = ts.map((t) => t.unsqueeze((dim ?? 0) as any));
    let acc = unsqueezed[0]!;
    for (let i = 1; i < unsqueezed.length; i++)
      acc = Tensor.cat(acc as any, unsqueezed[i]! as any, (dim ?? 0) as any) as AnyTensor;
    return acc as any;
  }

  static cat<A extends Shape, B extends Shape, PA extends TensorParams, const D extends number = 0>(
    a: Tensor<A, PA>,
    b: Tensor<B, any> & CatCheck<A, B, D>,
    dim?: D,
  ): Tensor<Cat<A, B, D>, PA> {
    const ta = a as AnyTensor;
    const tb = b as AnyTensor;
    const d = normalizeDim(dim ?? 0, ta.shape.length);
    if (ta.shape.length !== tb.shape.length)
      throw new Error(`cat: tensors must have the same rank (${showShape(ta.shape)} vs ${showShape(tb.shape)})`);
    for (let i = 0; i < ta.shape.length; i++)
      if (i !== d && ta.shape[i] !== tb.shape[i])
        throw new Error(`cat: shapes ${showShape(ta.shape)} and ${showShape(tb.shape)} differ outside dim ${d}`);
    const outShape = ta.shape.map((s: number, i: number) => (i === d ? s + tb.shape[d]! : s));
    const dtype: DType = ta.dtype === "float64" || tb.dtype === "float64" ? "float64" : "float32";
    const out = new (arrayCtor(dtype))(prod(outShape));
    const strides = contiguousStrides(outShape);
    const outer = prod(outShape.slice(0, d));
    const inner = strides[d]!;
    const lenA = ta.shape[d]!;
    const lenB = tb.shape[d]!;
    checkDevices(ta, tb);
    if (ta.device === "gpu") {
      if (dtype !== "float32") throw new Error("TypeGPU does not support float64 tensors");
      const result = makeStorage(
        typegpuBackend.concatenate(gpuStorage(ta), gpuStorage(tb), outer, lenA, lenB, inner),
        outShape,
        dtype,
      );
      return withGrad(result, "cat", [ta, tb], (g) => [rawNarrow(g, d, 0, lenA), rawNarrow(g, d, lenA, lenB)]) as any;
    }
    let o = 0;
    for (let i = 0; i < outer; i++) {
      for (let j = 0; j < lenA; j++) for (let k = 0; k < inner; k++) out[o++] = ta.data[(i * lenA + j) * inner + k]!;
      for (let j = 0; j < lenB; j++) for (let k = 0; k < inner; k++) out[o++] = tb.data[(i * lenB + j) * inner + k]!;
    }
    const result = makeRaw(out, outShape, dtype, ta.device);
    return withGrad(result, "cat", [ta, tb], (g) => [rawNarrow(g, d, 0, lenA), rawNarrow(g, d, lenA, lenB)]) as any;
  }

  [Operator.plus](lhs: Tensor<S, P>, rhs: number): Tensor<S, P>;
  [Operator.plus](lhs: number, rhs: Tensor<S, P>): Tensor<S, P>;

  [Operator.plus](lhs: Tensor<S, P>, rhs: Tensor<S, any>): Tensor<S, P>;

  [Operator.plus](lhs: Tensor<[Dim0<S>, 1], P>, rhs: Tensor<[1, Dim0<S>], any>): Tensor<[Dim0<S>, Dim0<S>], P>;
  [Operator.plus](lhs: Tensor<[1, Dim1<S>], P>, rhs: Tensor<[Dim1<S>, 1], any>): Tensor<[Dim1<S>, Dim1<S>], P>;
  [Operator.plus](lhs: Tensor<S, P>, rhs: Operand<S>): Tensor<S, P>;
  [Operator.plus](lhs: Operand<S>, rhs: Tensor<S, P>): Tensor<S, P>;
  [Operator.plus](lhs: Tensor<any, any>, rhs: Tensor<any, any>): typeof Operator.deferOperation;
  [Operator.plus](lhs: any, rhs: any): any {
    return coerceLhs(lhs, rhs).add(rhs);
  }

  [Operator.minus](lhs: Tensor<S, P>, rhs: number): Tensor<S, P>;
  [Operator.minus](lhs: number, rhs: Tensor<S, P>): Tensor<S, P>;

  [Operator.minus](lhs: Tensor<S, P>, rhs: Tensor<S, any>): Tensor<S, P>;

  [Operator.minus](lhs: Tensor<[Dim0<S>, 1], P>, rhs: Tensor<[1, Dim0<S>], any>): Tensor<[Dim0<S>, Dim0<S>], P>;
  [Operator.minus](lhs: Tensor<[1, Dim1<S>], P>, rhs: Tensor<[Dim1<S>, 1], any>): Tensor<[Dim1<S>, Dim1<S>], P>;
  [Operator.minus](lhs: Tensor<S, P>, rhs: Operand<S>): Tensor<S, P>;
  [Operator.minus](lhs: Operand<S>, rhs: Tensor<S, P>): Tensor<S, P>;
  [Operator.minus](lhs: Tensor<any, any>, rhs: Tensor<any, any>): typeof Operator.deferOperation;
  [Operator.minus](lhs: any, rhs: any): any {
    return coerceLhs(lhs, rhs).sub(rhs);
  }

  [Operator.star](lhs: Tensor<S, P>, rhs: number): Tensor<S, P>;
  [Operator.star](lhs: number, rhs: Tensor<S, P>): Tensor<S, P>;

  [Operator.star](lhs: Tensor<S, P>, rhs: Tensor<S, any>): Tensor<S, P>;

  [Operator.star](lhs: Tensor<[Dim0<S>, 1], P>, rhs: Tensor<[1, Dim0<S>], any>): Tensor<[Dim0<S>, Dim0<S>], P>;
  [Operator.star](lhs: Tensor<[1, Dim1<S>], P>, rhs: Tensor<[Dim1<S>, 1], any>): Tensor<[Dim1<S>, Dim1<S>], P>;
  [Operator.star](lhs: Tensor<S, P>, rhs: Operand<S>): Tensor<S, P>;
  [Operator.star](lhs: Operand<S>, rhs: Tensor<S, P>): Tensor<S, P>;
  [Operator.star](lhs: Tensor<any, any>, rhs: Tensor<any, any>): typeof Operator.deferOperation;
  [Operator.star](lhs: any, rhs: any): any {
    return coerceLhs(lhs, rhs).mul(rhs);
  }

  [Operator.slash](lhs: Tensor<S, P>, rhs: number): Tensor<S, P>;
  [Operator.slash](lhs: number, rhs: Tensor<S, P>): Tensor<S, P>;

  [Operator.slash](lhs: Tensor<S, P>, rhs: Tensor<S, any>): Tensor<S, P>;

  [Operator.slash](lhs: Tensor<[Dim0<S>, 1], P>, rhs: Tensor<[1, Dim0<S>], any>): Tensor<[Dim0<S>, Dim0<S>], P>;
  [Operator.slash](lhs: Tensor<[1, Dim1<S>], P>, rhs: Tensor<[Dim1<S>, 1], any>): Tensor<[Dim1<S>, Dim1<S>], P>;
  [Operator.slash](lhs: Tensor<S, P>, rhs: Operand<S>): Tensor<S, P>;
  [Operator.slash](lhs: Operand<S>, rhs: Tensor<S, P>): Tensor<S, P>;
  [Operator.slash](lhs: Tensor<any, any>, rhs: Tensor<any, any>): typeof Operator.deferOperation;
  [Operator.slash](lhs: any, rhs: any): any {
    return coerceLhs(lhs, rhs).div(rhs);
  }

  [Operator.starStar](lhs: Tensor<S, P>, rhs: number): Tensor<S, P>;
  [Operator.starStar](lhs: any, rhs: any): any {
    if (typeof rhs !== "number") throw new Error("** on tensors requires a scalar exponent");
    return (lhs as AnyTensor).pow(rhs);
  }
}

type Dim0<S extends Shape> = S extends [infer A extends number, ...any[]] ? A : never;
type Dim1<S extends Shape> = S extends [any, infer B extends number, ...any[]] ? B : never;

export type Operand<S extends Shape> =
  BroadcastsInto<S> extends infer R ? (R extends Shape ? Tensor<R, any> : never) : never;

type StackCheck<T extends readonly AnyTensor[]> =
  T[number] extends Tensor<ShapeOf<T[0]>, any> ? unknown : ErrorMessage<"stack: all tensors must have the same shape">;

function coerce(value: AnyTensor | number, like: AnyTensor): AnyTensor {
  if (typeof value === "number") return makeRaw(arrayCtor(like.dtype).of(value), [], like.dtype, like.device);
  return value;
}

function coerceLhs(lhs: any, rhs: any): AnyTensor {
  if (lhs instanceof Tensor) return lhs as AnyTensor;
  return coerce(lhs, rhs as AnyTensor);
}

function makeMovedData(t: AnyTensor, data: TypedArray, dtype: DType): AnyTensor {
  const out = makeRaw(data, t.shape, dtype, t.device);
  return withGrad(out, "to", [t], (g) => [g]);
}

function resolveViewRuntime(shape: number[], view: number[]): number[] {
  const negOnes = view.filter((v) => v === -1).length;
  if (negOnes > 1) throw new Error("Only one -1 dim is allowed in view()");
  const total = prod(shape);
  if (negOnes === 1) {
    const rest = prod(view.filter((v) => v !== -1));
    if (rest === 0 || total % rest !== 0)
      throw new Error(`Cannot view tensor of shape ${showShape(shape)} as ${showShape(view)}`);
    return view.map((v) => (v === -1 ? total / rest : v));
  }
  if (prod(view) !== total)
    throw new Error(
      `Cannot view tensor of shape ${showShape(shape)} as ${showShape(view)} (${total} vs ${prod(view)} elements)`,
    );
  return [...view];
}

function reshapeRaw(t: AnyTensor, shape: number[]): AnyTensor {
  if (t.numel !== prod(shape)) throw new Error(`Cannot reshape ${showShape(t.shape)} to ${showShape(shape)}`);
  return makeStorage(t._storage, shape, t.dtype);
}

function matmul2(a: AnyTensor, b: AnyTensor): AnyTensor {
  const out = rawMatmul(a, b);
  return withGrad(out, "matmul", [a, b], (g) => {
    const bt = rawPermute(b, swapLastTwo(b.shape.length));
    const at = rawPermute(a, swapLastTwo(a.shape.length));
    const da = sumTo(rawMatmul(g, bt), [...a.shape]);
    const db = sumTo(rawMatmul(at, g), [...b.shape]);
    return [da, db];
  });
}

function swapLastTwo(rank: number): number[] {
  const order = [...Array(rank).keys()];
  [order[rank - 2], order[rank - 1]] = [order[rank - 1]!, order[rank - 2]!];
  return order;
}

export const tensor = Tensor.of;
export const zeros = Tensor.zeros;
export const ones = Tensor.ones;
export const full = Tensor.full;
export const rand = Tensor.rand;
export const randn = Tensor.randn;
export const eye = Tensor.eye;
export const arange = Tensor.arange;
export const scalar = Tensor.scalar;
export const stack = Tensor.stack;
export const cat = Tensor.cat;
