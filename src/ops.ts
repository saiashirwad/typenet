/**
 * The single list of operation kinds. TypeScript unions are derived
 * from these arrays, and `test/ops.test.ts` checks the Rust addon's
 * `Node` tags and `Bin`/`Un` parse arms against the same lists — adding
 * an op means extending one array here and the matching Rust arm, and
 * the test fails until both moved.
 */

export const BINARY_OPS = [
  "add",
  "sub",
  "mul",
  "div",
  "maximum",
  "minimum",
  "gt",
  "ge",
  "lt",
  "le",
  "eq",
  "negDiv",
  "halfDiv",
  "mulSign",
  "reluGrad",
  "leakyReluGrad",
  "sigmoidGrad",
  "tanhGrad",
] as const

export type BinaryOp = (typeof BINARY_OPS)[number]

export const UNARY_OPS = [
  "pow",
  "neg",
  "exp",
  "log",
  "sqrt",
  "abs",
  "relu",
  "leakyRelu",
  "sigmoid",
  "tanh",
  "scalePowGrad",
] as const

export type UnaryOp = (typeof UNARY_OPS)[number]

export const REDUCE_OPS = ["sum", "max", "argmax"] as const

export type ReduceOp = (typeof REDUCE_OPS)[number]

export const RANDOM_KINDS = ["uniform", "normal"] as const

export type RandomKind = (typeof RANDOM_KINDS)[number]

/** Structural node kinds of the IR, `leaf` included (serialize-only). */
export const NODE_OPS = [
  "leaf",
  "binary",
  "unary",
  "matmul",
  "reduce",
  "reduceAll",
  "broadcastTo",
  "permute",
  "view",
  "narrow",
  "cat",
  "oneHot",
  "indexSelect",
  "scatterAdd",
  "random",

  // --- W4.1 semantic ops (PLAN-V2 §2.3) ------------------------------------
  // Every one of these is a NODE kind, never a new `unary` kind: each has
  // attributes, more than one output, or both. The fifteen kinds above are
  // exactly what the addon parses today (`WIRE_OPS` in `lower-native.ts`);
  // everything below is Phase A's growth area and falls back to the JS
  // interpreter until A-L1 lands a lowering and W4.2-W4.5 land kernels.
  "gelu",
  "geluGrad",
  "silu",
  "siluGrad",
  "softmax",
  "softmaxGrad",
  "layerNorm",
  "layerNormGrad",
  "rmsNorm",
  "rmsNormGrad",
  "crossEntropy",
  "logSumExp",
  "gatherRows",
  "scatterAddRows",
  "dropout",
  "pick",
  "contiguous",
] as const

/** Every structural node kind; the domain of `OP_DESC` and of `SUPPORT`. */
export type NodeOp = (typeof NODE_OPS)[number]
