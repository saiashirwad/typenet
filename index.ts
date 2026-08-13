export {
  arange,
  broadcastShapes,
  cat,
  compile,
  configure,
  eye,
  fromFlat,
  full,
  isLazy,
  noGrad,
  normal,
  ones,
  printGraph,
  rand,
  randn,
  scalar,
  stack,
  Tensor,
  tensor,
  uniform,
  zeros,
} from "./src/tensor.ts"
export type { CompiledFn, DType, NestedNumbers, RandomKind, ShapeOf } from "./src/tensor.ts"

export { disableNative, isNativeAvailable, isNativeEnabled, nativeDevice, nativeDeviceMode, useNative } from "./src/backends/native.ts"
export { crossEntropy, LeakyReLU, Linear, Module, mseLoss, ReLU, sequential, Sigmoid, Softmax, Tanh } from "./src/nn.ts"
// The Sequential *type* is public for annotations; construction goes
// through sequential(...) only.
export type { Layer, Sequential } from "./src/nn.ts"
export { Adam, clipGradNorm, Optimizer, SGD } from "./src/optim.ts"
export type { AdamOptions, SGDOptions } from "./src/optim.ts"

// DimAdd / DimMul are both a type and a value: the type does the
// arithmetic on literal dims, the function returns it at runtime, so a
// constructor width like `DimAdd(DimMul(3, channels), 1)` carries its
// derived type with no cast.
export { DimAdd, DimMul } from "./src/shape.ts"
export type {
  Broadcast,
  BroadcastCheck,
  CanBroadcast,
  Cat,
  ErrorMessage,
  InferShape,
  MatMul,
  MatMulCheck,
  NestedArray,
  Permute,
  ReduceDim,
  ResizeDim,
  ResolveView,
  Shape,
  Squeeze,
  Stack,
  Transpose,
  Unsqueeze,
} from "./src/shape.ts"
