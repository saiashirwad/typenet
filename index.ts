export {
  Tensor,
  tensor,
  zeros,
  ones,
  full,
  rand,
  randn,
  eye,
  arange,
  scalar,
  stack,
  cat,
  noGrad,
  configure,
  isLazy,
  compile,
  broadcastShapes
} from "./src/tensor.ts"
export type {
  DType,
  Device,
  TensorParams,
  DefaultParams,
  ShapeOf,
  ParamsOf,
  NestedNumbers
} from "./src/tensor.ts"

export {
  Module,
  Linear,
  ReLU,
  LeakyReLU,
  Tanh,
  Sigmoid,
  Softmax,
  Sequential,
  sequential,
  mseLoss,
  crossEntropy
} from "./src/nn.ts"
export type { Layer } from "./src/nn.ts"
export { Optimizer, SGD, Adam } from "./src/optim.ts"
export type {
  SGDOptions,
  AdamOptions
} from "./src/optim.ts"
export {
  configureTypeGPU,
  initTypeGPU,
  clearTypeGPU
} from "./src/typegpu.ts"
export {
  useNative,
  disableNative,
  isNativeEnabled,
  isNativeAvailable,
  nativeDevice
} from "./src/backends/native.ts"
export type { TgpuRoot } from "typegpu"

export type {
  Shape,
  Broadcast,
  BroadcastCheck,
  CanBroadcast,
  MatMul,
  MatMulCheck,
  Transpose,
  Permute,
  Squeeze,
  Unsqueeze,
  ReduceDim,
  ResolveView,
  Stack,
  Cat,
  InferShape,
  NestedArray,
  ErrorMessage
} from "./src/shape.ts"
