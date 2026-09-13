import { arrayCtor, type BinaryOp, type DType, type RandomKind, type TypedArray, type UnaryOp } from "./storage.ts"

function applyBinary(
  op: BinaryOp,
  x: number,
  y: number,
  parameter: number,
): number {
  switch (op) {
    case "add":
      return x + y
    case "sub":
      return x - y
    case "mul":
      return x * y
    case "div":
      return x / y
    case "maximum":
      return Math.max(x, y)
    case "minimum":
      return Math.min(x, y)
    case "gt":
      return x > y ? 1 : 0
    case "ge":
      return x >= y ? 1 : 0
    case "lt":
      return x < y ? 1 : 0
    case "le":
      return x <= y ? 1 : 0
    case "eq":
      return x === y ? 1 : 0
    case "negDiv":
      return -x / y
    case "halfDiv":
      return (0.5 * x) / y
    case "mulSign":
      return x * Math.sign(y)
    case "reluGrad":
      return y > 0 ? x : 0
    case "leakyReluGrad":
      return y > 0 ? x : parameter * x
    case "sigmoidGrad":
      return x * y * (1 - y)
    case "tanhGrad":
      return x * (1 - y * y)
  }
}

function applyUnary(
  op: UnaryOp,
  x: number,
  parameter: number,
): number {
  switch (op) {
    case "pow":
      return x ** parameter
    case "neg":
      return -x
    case "exp":
      return Math.exp(x)
    case "log":
      return Math.log(x)
    case "sqrt":
      return Math.sqrt(x)
    case "abs":
      return Math.abs(x)
    case "relu":
      return x > 0 ? x : 0
    case "leakyRelu":
      return x > 0 ? x : parameter * x
    case "sigmoid":
      return 1 / (1 + Math.exp(-x))
    case "tanh":
      return Math.tanh(x)
    case "scalePowGrad":
      return parameter * x ** (parameter - 1)
  }
}

// ---------------------------------------------------------------------------
// W4.1 semantic scalar kernels. These are the *numeric specification* (D16):
// `src/eager.ts` maps them over a buffer, `src/lazy.ts`'s interpreter replays
// the same kernel per node, and a native kernel must reproduce them. Each is
// written as the composition PLAN-V2 §5A.2a's lowering table names, in that
// evaluation order, so when A-L1 lowers the node to primitives the two paths
// are the same arithmetic rather than two arithmetics within a tolerance.
// ---------------------------------------------------------------------------

/** `sqrt(2/pi)` — the constant of the tanh GELU approximation. */
const GELU_C = Math.sqrt(2 / Math.PI)
/** The cubic coefficient of the same approximation (Hendrycks & Gimpel). */
const GELU_A = 0.044715

/** `0.5*x*(1 + tanh(sqrt(2/pi)*(x + 0.044715*x^3)))`. */
function gelu(x: number): number {
  return 0.5 * x * (1 + Math.tanh(GELU_C * (x + GELU_A * x * x * x)))
}

/**
 * d/dx of {@link gelu}, times an upstream `g`. Written out rather than
 * differentiated numerically because the tanh approximation is the thing
 * being differentiated — the exact-erf GELU has a different derivative and
 * mixing the two is the classic silent 1e-3 error in a transformer.
 */
function geluGrad(g: number, x: number): number {
  const inner = GELU_C * (x + GELU_A * x * x * x)
  const t = Math.tanh(inner)
  const dInner = GELU_C * (1 + 3 * GELU_A * x * x)
  return g * (0.5 * (1 + t) + 0.5 * x * (1 - t * t) * dInner)
}

/** `x * sigmoid(x)`. */
function silu(x: number): number {
  return x / (1 + Math.exp(-x))
}

/** `g * (s + x*s*(1-s))` with `s = sigmoid(x)`. */
function siluGrad(g: number, x: number): number {
  const s = 1 / (1 + Math.exp(-x))
  return g * (s + x * s * (1 - s))
}

/** murmur3's 32-bit finalizer, in its stronger (Stafford 13) variant. */
function hash32(x: number): number {
  x = (x ^ (x >>> 16)) >>> 0
  x = Math.imul(x, 0x7feb352d) >>> 0
  x = (x ^ (x >>> 15)) >>> 0
  x = Math.imul(x, 0x846ca68b) >>> 0
  return (x ^ (x >>> 16)) >>> 0
}

/** Uniform in [0, 1) from 24 mantissa bits of a hashed counter. */
function unitFloat(
  seed: number,
  stream: number,
  i: number,
) {
  return (
    (hash32(
      (hash32(seed ^ Math.imul(stream, 0x9e3779b9)) ^ i)
        >>> 0,
    )
      >>> 8)
    * 2 ** -24
  )
}

let randomSeed = 0x2545f491
let streamCounter = 0
let activeSeed = 0

function nextSeed(): number {
  randomSeed = hash32(randomSeed + 0x9e3779b9)
  return randomSeed
}

function nextStream(): number {
  return streamCounter++
}

function getActiveSeed(): number {
  return activeSeed
}

function setActiveSeed(seed: number): void {
  activeSeed = seed
}

function reseed(seed: number): void {
  randomSeed = seed >>> 0
  streamCounter = 0
}

type RngState = { seed: number; stream: number; active: number }

function rngState(): RngState {
  return {
    seed: randomSeed,
    stream: streamCounter,
    active: activeSeed,
  }
}

function setRngState(state: RngState): void {
  randomSeed = state.seed
  streamCounter = state.stream
  activeSeed = state.active
}

function randomData(
  kind: RandomKind,
  n: number,
  stream: number,
  seed: number,
  dtype: DType,
): TypedArray {
  const out = new (arrayCtor(dtype))(n)
  if (kind === "uniform") {
    for (let i = 0; i < n; i++) {
      out[i] = unitFloat(seed, stream, i)
    }
  } // Box-Muller per element from two independent draws: stateless, so
  // element i does not depend on how many were drawn before it.
  else {
    for (let i = 0; i < n; i++) {
      const u = 1 - unitFloat(seed, stream, 2 * i)
      const v = unitFloat(seed, stream, 2 * i + 1)
      out[i] = Math.sqrt(-2 * Math.log(u))
        * Math.cos(2 * Math.PI * v)
    }
  }
  return out
}

export {
  applyBinary,
  applyUnary,
  gelu,
  GELU_A,
  GELU_C,
  geluGrad,
  getActiveSeed,
  hash32,
  nextSeed,
  nextStream,
  randomData,
  reseed,
  rngState,
  setActiveSeed,
  setRngState,
  silu,
  siluGrad,
  unitFloat,
}
