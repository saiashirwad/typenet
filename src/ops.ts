/** Adding an op means extending an array here and the matching Rust arm; `test/ops.test.ts` fails until both move. */

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

  // Kinds with attributes or multiple outputs, never new `unary` kinds; not yet
  // lowered natively.
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
] as const

export type NodeOp = (typeof NODE_OPS)[number]
