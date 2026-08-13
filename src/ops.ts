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
] as const
