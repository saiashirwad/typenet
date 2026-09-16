import { hash32, nextSeed, nextStream, randomData } from "../kernels.ts"
import type { Shape } from "../shape.ts"
import { prod, showShape } from "../storage.ts"
import { type AnyTensor, fromFlat, Tensor } from "../tensor.ts"

/** An explicit draw source, independent of the ambient seed counter: the same `Generator` gives identical draws in two calls. */
export interface Generator {
  readonly seed: number
  readonly stream: number
}

export function generator(seed: number): Generator {
  const s = seed >>> 0
  return { seed: hash32(s), stream: hash32(s ^ 0x9e3779b9) }
}

function drawTensor<const Sh extends Shape>(
  kind: "uniform" | "normal",
  shape: Sh,
  gen: Generator | undefined,
): Tensor<Sh> {
  if (!gen) return kind === "uniform" ? Tensor.rand(shape) : Tensor.randn(shape)
  return fromFlat(
    randomData(kind, prod(shape), gen.stream, gen.seed, "float32"),
    shape,
    "float32",
  )
}

function drawData(
  kind: "uniform" | "normal",
  n: number,
  gen: Generator | undefined,
): Float32Array {
  return randomData(
    kind,
    n,
    gen ? gen.stream : nextStream(),
    gen ? gen.seed : nextSeed(),
    "float32",
  ) as Float32Array
}

type FanMode = "fanIn" | "fanOut"
type Nonlinearity = "linear" | "sigmoid" | "tanh" | "relu" | "leakyRelu"

/** `{fanIn, fanOut}` by PyTorch's convention: dim 0 is fan-out, dim 1 is fan-in, trailing axes multiply into both. */
export function calculateFan(
  shape: readonly number[],
  who: string,
): { fanIn: number; fanOut: number } {
  if (shape.length < 2) {
    throw new Error(
      `${who}(): fan-based initialisers need a tensor of rank >= 2, got shape ${showShape(shape as number[])}`,
    )
  }
  let receptive = 1
  for (let i = 2; i < shape.length; i++) receptive *= shape[i]!
  return { fanIn: shape[1]! * receptive, fanOut: shape[0]! * receptive }
}

function pickFan(shape: readonly number[], mode: FanMode | undefined, who: string): number {
  const { fanIn, fanOut } = calculateFan(shape, who)
  return (mode ?? "fanIn") === "fanIn" ? fanIn : fanOut
}

function calculateGain(nonlinearity: Nonlinearity, negativeSlope = 0.01): number {
  switch (nonlinearity) {
    case "linear":
    case "sigmoid":
      return 1
    case "tanh":
      return 5 / 3
    case "relu":
      return Math.SQRT2
    case "leakyRelu":
      return Math.sqrt(2 / (1 + negativeSlope * negativeSlope))
  }
}

interface GainOptions {
  gain?: number
  nonlinearity?: Nonlinearity
  negativeSlope?: number
}

/** The Kaiming uniform half-width `gain·√(3/fan)`. With no gain or nonlinearity it is `1/√fan` in one rounding step, which is PyTorch's `a=√5` default. */
function kaimingBound(fan: number, o: GainOptions): number {
  if (o.gain === undefined && o.nonlinearity === undefined) {
    return 1 / Math.sqrt(fan)
  }
  const gain = o.gain ?? calculateGain(o.nonlinearity ?? "leakyRelu", o.negativeSlope)
  return gain * Math.sqrt(3 / fan)
}

/** The Kaiming normal std `gain/√fan`. With no gain or nonlinearity it is `√(1/(3·fan))`, which matches the variance of {@link kaimingBound}'s default uniform. */
function kaimingStd(fan: number, o: GainOptions): number {
  if (o.gain === undefined && o.nonlinearity === undefined) {
    return Math.sqrt(1 / (3 * fan))
  }
  const gain = o.gain ?? calculateGain(o.nonlinearity ?? "leakyRelu", o.negativeSlope)
  return gain / Math.sqrt(fan)
}

/** Abramowitz & Stegun 7.1.26, max error ~1.5e-7. */
function erf(x: number): number {
  const sign = x < 0 ? -1 : 1
  x = Math.abs(x)
  const a1 = 0.254829592, a2 = -0.284496736, a3 = 1.421413741, a4 = -1.453152027, a5 = 1.061405429, p = 0.3275911
  const t = 1 / (1 + p * x)
  const y = 1 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.exp(-x * x)
  return sign * y
}

/** Winitzki's approximation. */
function erfinv(x: number): number {
  const a = 0.147
  const ln1mx2 = Math.log(1 - x * x)
  const t1 = 2 / (Math.PI * a) + ln1mx2 / 2
  const t2 = ln1mx2 / a
  return Math.sign(x) * Math.sqrt(Math.sqrt(t1 * t1 - t2) - t1)
}

function normalCdf(x: number): number {
  return 0.5 * (1 + erf(x / Math.SQRT2))
}

function normalInvCdf(p: number): number {
  return Math.SQRT2 * erfinv(2 * p - 1)
}

export function zeros<const Sh extends Shape>(shape: Sh): Tensor<Sh> {
  return Tensor.zeros(shape)
}

export function ones<const Sh extends Shape>(shape: Sh): Tensor<Sh> {
  return Tensor.ones(shape)
}

export function constant<const Sh extends Shape>(shape: Sh, value: number): Tensor<Sh> {
  return Tensor.full(shape, value)
}

export interface UniformOptions {
  low?: number
  high?: number
  generator?: Generator | undefined
}

/** Uniform in `[low, high)` (default `[0, 1)`). */
export function uniform<const Sh extends Shape>(shape: Sh, o: UniformOptions = {}): Tensor<Sh> {
  const { low = 0, high = 1, generator: gen } = o
  const raw = drawTensor("uniform", shape, gen)
  if (low === 0 && high === 1) return raw
  const span = high - low
  return raw.mul(span).add(low) as unknown as Tensor<Sh>
}

export interface NormalOptions {
  mean?: number
  std?: number
  generator?: Generator | undefined
}

export function normal<const Sh extends Shape>(shape: Sh, o: NormalOptions = {}): Tensor<Sh> {
  const { mean = 0, std = 1, generator: gen } = o
  const raw = drawTensor("normal", shape, gen)
  if (mean === 0 && std === 1) return raw
  return raw.mul(std).add(mean) as unknown as Tensor<Sh>
}

export interface TruncNormalOptions {
  mean?: number
  std?: number
  a?: number
  b?: number
  generator?: Generator
}

/** `Normal(mean, std)` truncated to `[a, b]` (default `[-2, 2]`), via inverse-CDF sampling. */
export function truncNormal<const Sh extends Shape>(shape: Sh, o: TruncNormalOptions = {}): Tensor<Sh> {
  const { mean = 0, std = 1, a = -2, b = 2, generator: gen } = o
  if (a >= b) {
    throw new Error(`truncNormal(): lower bound a=${a} must be less than upper bound b=${b}`)
  }
  const lo = normalCdf((a - mean) / std)
  const hi = normalCdf((b - mean) / std)
  const n = prod(shape)
  const u = drawData("uniform", n, gen)
  const data = new Float32Array(n)
  for (let i = 0; i < n; i++) {
    // `normalInvCdf` blows up at exactly 0/1, which a boundary `u[i]` can hit.
    const p = Math.min(Math.max(lo + u[i]! * (hi - lo), 1e-7), 1 - 1e-7)
    // Far in a tail both CDF endpoints underflow to 0, putting the draw outside [a, b].
    data[i] = Math.min(Math.max(mean + std * normalInvCdf(p), a), b)
  }
  return fromFlat(data, shape, "float32")
}

export interface KaimingOptions extends GainOptions {
  fanMode?: FanMode
  generator?: Generator
}

export function kaimingUniform<const Sh extends Shape>(shape: Sh, o: KaimingOptions = {}): Tensor<Sh> {
  const fan = pickFan(shape, o.fanMode, "kaimingUniform")
  const bound = kaimingBound(fan, o)
  return uniform(shape, { low: -bound, high: bound, generator: o.generator })
}

export function kaimingNormal<const Sh extends Shape>(shape: Sh, o: KaimingOptions = {}): Tensor<Sh> {
  const fan = pickFan(shape, o.fanMode, "kaimingNormal")
  return normal(shape, { mean: 0, std: kaimingStd(fan, o), generator: o.generator })
}

export interface XavierOptions {
  gain?: number
  generator?: Generator
}

/** Xavier/Glorot uniform: `U(-bound, bound)`, `bound = gain·√(6/(fanIn+fanOut))`. */
export function xavierUniform<const Sh extends Shape>(shape: Sh, o: XavierOptions = {}): Tensor<Sh> {
  const { fanIn, fanOut } = calculateFan(shape, "xavierUniform")
  const bound = (o.gain ?? 1) * Math.sqrt(6 / (fanIn + fanOut))
  return uniform(shape, { low: -bound, high: bound, generator: o.generator })
}

/** Xavier/Glorot normal: `Normal(0, std)`, `std = gain·√(2/(fanIn+fanOut))`. */
export function xavierNormal<const Sh extends Shape>(shape: Sh, o: XavierOptions = {}): Tensor<Sh> {
  const { fanIn, fanOut } = calculateFan(shape, "xavierNormal")
  const std = (o.gain ?? 1) * Math.sqrt(2 / (fanIn + fanOut))
  return normal(shape, { mean: 0, std, generator: o.generator })
}

export function zeros_<T extends AnyTensor>(t: T): T {
  t.fill_(0)
  return t
}

export function ones_<T extends AnyTensor>(t: T): T {
  t.fill_(1)
  return t
}

export function constant_<T extends AnyTensor>(t: T, value: number): T {
  t.fill_(value)
  return t
}

export function uniform_<T extends AnyTensor>(t: T, o: UniformOptions = {}): T {
  t.copy_(uniform(t.shape, o))
  return t
}

export function normal_<T extends AnyTensor>(t: T, o: NormalOptions = {}): T {
  t.copy_(normal(t.shape, o))
  return t
}

export function truncNormal_<T extends AnyTensor>(t: T, o: TruncNormalOptions = {}): T {
  t.copy_(truncNormal(t.shape, o))
  return t
}

export function kaimingUniform_<T extends AnyTensor>(t: T, o: KaimingOptions = {}): T {
  t.copy_(kaimingUniform(t.shape, o))
  return t
}

export function kaimingNormal_<T extends AnyTensor>(t: T, o: KaimingOptions = {}): T {
  t.copy_(kaimingNormal(t.shape, o))
  return t
}

export function xavierUniform_<T extends AnyTensor>(t: T, o: XavierOptions = {}): T {
  t.copy_(xavierUniform(t.shape, o))
  return t
}

export function xavierNormal_<T extends AnyTensor>(t: T, o: XavierOptions = {}): T {
  t.copy_(xavierNormal(t.shape, o))
  return t
}
