// nn.init (§3.3, W1.9): parameter initialisers, in two forms each —
//
//   - a **pure constructor** (`kaimingUniform`, `xavierNormal`, ...) that
//     builds a fresh tensor of a given shape, and
//   - an **in-place** form (the same name with a trailing `_`, following
//     this codebase's mutation convention — see tensor.ts's "Safe mutation
//     primitives") that overwrites an existing tensor's bytes.
//
// The in-place forms are written as the pure form plus `copy_`/`fill_`, so
// every mutation guard — refusing to write through a tensor `compile()` is
// tracing, or that an autograd tape is recording through — lives in exactly
// one place (tensor.ts), not duplicated here.
//
// Every random form draws through `randomData` (kernels.ts): deterministic
// under `configure({ seed })`/`withContext({ seed })` by default (the same
// counter `Tensor.rand`/`Tensor.randn` already use), or, for a call site
// that wants draws independent of that ambient counter, an explicit
// `Generator` (see `generator()` below).
import { hash32, nextSeed, nextStream, randomData } from "../kernels.ts"
import type { Shape } from "../shape.ts"
import { prod, showShape } from "../storage.ts"
import { type AnyTensor, fromFlat, Tensor } from "../tensor.ts"

// ---------------------------------------------------------------------
// Generators — an explicit, independently-seeded draw source.
// ---------------------------------------------------------------------

/**
 * An explicit draw source, independent of the ambient
 * `configure({ seed })` counter (`nextSeed`/`nextStream` in kernels.ts):
 * pass the *same* `Generator` to two calls to get identical draws
 * regardless of what else ran in between — the opposite of the default,
 * where every call takes the next slice of the shared counter.
 */
export interface Generator {
  readonly seed: number
  readonly stream: number
}

/** A `Generator` deterministically derived from `seed` — same `seed` in, same `Generator` out. */
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

// ---------------------------------------------------------------------
// Fan and gain — the two quantities every fan-scaled initialiser needs.
// ---------------------------------------------------------------------

export type FanMode = "fanIn" | "fanOut"
export type Nonlinearity = "linear" | "sigmoid" | "tanh" | "relu" | "leakyRelu"

/**
 * `{fanIn, fanOut}` from a weight tensor's shape, PyTorch's convention:
 * dim 0 is the "fan-out" axis, dim 1 is "fan-in", and every axis past
 * that is a receptive field multiplied into both (e.g. `Conv2d`'s
 * `[COut, CIn, K, K]` gives `fanIn = CIn·K·K`, `fanOut = COut·K·K`).
 *
 * `Linear`'s weight is stored transposed relative to that convention
 * (`[In, Out]`, matmul order, not PyTorch's `[Out, In]`) — its
 * constructor deliberately reads `fanMode: "fanOut"` off this same
 * table to get `shape[0] = In`; see linear.ts.
 */
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

/** The recommended gain for `nonlinearity`, `torch.nn.init.calculate_gain`'s table. */
export function calculateGain(nonlinearity: Nonlinearity, negativeSlope = 0.01): number {
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

/**
 * The Kaiming uniform half-width, `gain·√(3/fan)`. When neither `gain`
 * nor `nonlinearity` is given, this returns `1/√fan` directly — computed
 * in that one step, not through the general formula — because that *is*
 * the general formula's value at PyTorch's own `Linear` default
 * (`kaiming_uniform_(..., a=√5)`, whose `gain = √(2/(1+5)) = √(1/3)`
 * makes `gain·√(3/fan)` reduce to `1/√fan` mathematically), and going
 * through two `Math.sqrt` calls and a multiply to reach the same value
 * would round twice instead of once. This is what lets `Linear`'s
 * constructor switch to this module with its seeded output unchanged
 * bit for bit (W1.9's accept #2).
 */
function kaimingBound(fan: number, o: GainOptions): number {
  if (o.gain === undefined && o.nonlinearity === undefined) {
    return 1 / Math.sqrt(fan)
  }
  const gain = o.gain ?? calculateGain(o.nonlinearity ?? "leakyRelu", o.negativeSlope)
  return gain * Math.sqrt(3 / fan)
}

/**
 * The Kaiming normal std, `gain/√fan`. Mirrors {@link kaimingBound}'s
 * no-`gain`-no-`nonlinearity` default: `√(1/(3·fan))`, the std whose
 * variance (`1/(3·fan)`) matches that default uniform bound's variance
 * exactly (`Var(U(-b,b)) = b²/3`).
 */
function kaimingStd(fan: number, o: GainOptions): number {
  if (o.gain === undefined && o.nonlinearity === undefined) {
    return Math.sqrt(1 / (3 * fan))
  }
  const gain = o.gain ?? calculateGain(o.nonlinearity ?? "leakyRelu", o.negativeSlope)
  return gain / Math.sqrt(fan)
}

// ---------------------------------------------------------------------
// erf / erfinv — closed-form truncated-normal sampling (no rejection
// loop, so it is exactly as deterministic under a seed as every other
// draw here: one uniform sample per element, always).
// ---------------------------------------------------------------------

/** Abramowitz & Stegun 7.1.26 — max error ~1.5e-7, plenty for an initialiser. */
function erf(x: number): number {
  const sign = x < 0 ? -1 : 1
  x = Math.abs(x)
  const a1 = 0.254829592, a2 = -0.284496736, a3 = 1.421413741, a4 = -1.453152027, a5 = 1.061405429, p = 0.3275911
  const t = 1 / (1 + p * x)
  const y = 1 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.exp(-x * x)
  return sign * y
}

/** Winitzki's approximation — accurate enough to place a truncated-normal sample. */
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

// ---------------------------------------------------------------------
// Pure constructors.
// ---------------------------------------------------------------------

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
  generator?: Generator
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
  generator?: Generator
}

/** `Normal(mean, std)` (default standard normal). */
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
    // Clamped away from the exact endpoints: `normalInvCdf` blows up at
    // 0/1, and floating-point `u[i]` can land there at the boundary.
    const p = Math.min(Math.max(lo + u[i]! * (hi - lo), 1e-7), 1 - 1e-7)
    data[i] = mean + std * normalInvCdf(p)
  }
  return fromFlat(data, shape, "float32")
}

export interface KaimingOptions extends GainOptions {
  fanMode?: FanMode
  generator?: Generator
}

/** Kaiming/He uniform: `U(-bound, bound)`, `bound = gain·√(3/fan)`. See {@link kaimingBound}. */
export function kaimingUniform<const Sh extends Shape>(shape: Sh, o: KaimingOptions = {}): Tensor<Sh> {
  const fan = pickFan(shape, o.fanMode, "kaimingUniform")
  const bound = kaimingBound(fan, o)
  return uniform(shape, { low: -bound, high: bound, generator: o.generator })
}

/** Kaiming/He normal: `Normal(0, std)`, `std = gain/√fan`. See {@link kaimingStd}. */
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

// ---------------------------------------------------------------------
// In-place forms — the pure form above, written into an existing
// tensor through `fill_`/`copy_` (tensor.ts owns the mutation guards).
// ---------------------------------------------------------------------

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
