// Elementwise scalar kernels and the counter-based random generator.
// These never touch a Tensor: numbers in, numbers out (plus a typed
// array from the RNG). The generator's state lives here behind small
// accessors because both the eager path (draws immediately) and the
// lazy eval loop (advances the seed per pass) need to drive it, and an
// imported `let` binding cannot be reassigned from outside.

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

// --- random numbers inside the graph ------------------------------
// A compiled training step is traced once and replayed thousands of
// times, so anything stochastic in it — a dropout-style update mask,
// input noise — has to be drawn per replay rather than baked in at
// trace time. That rules out feeding randomness in as data, which
// would mean generating and copying megabytes per step.
//
// So random values come from a graph node, and the generator is
// counter-based: element `i` of stream `s` under seed `k` is a pure
// hash of (k, s, i). No state to carry, every element independent,
// and the identical arithmetic on the Rust side means all three
// execution paths agree element for element.

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

// The seed every evaluation draws from, advanced per evaluation so a
// replayed graph gets fresh numbers. configure({ seed }) resets it, and
// a run replays exactly given the same seed and the same sequence of
// operations.
let randomSeed = 0x2545f491
let streamCounter = 0
// Seed of the evaluation in progress, so a random node reached by the
// interpreter knows which draw it belongs to.
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

/** Reset the generator: `configure({ seed })` calls this. */
function reseed(seed: number): void {
  randomSeed = seed >>> 0
  streamCounter = 0
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

export { applyBinary, applyUnary, getActiveSeed, hash32, nextSeed, nextStream, randomData, reseed, setActiveSeed, unitFloat }
