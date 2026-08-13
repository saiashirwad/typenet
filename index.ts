export { noGrad } from "./src/autograd.ts"
export { compile, printGraph } from "./src/compile.ts"
export type { CompiledFn } from "./src/compile.ts"
export { context, withContext } from "./src/context.ts"
export type { RuntimeContext } from "./src/context.ts"
export { arange, cat, eye, full, ones, rand, randn, scalar, stack, tensor, zeros } from "./src/factories.ts"
export { normal, uniform } from "./src/ir.ts"
export { configure } from "./src/lazy.ts"
export { broadcastShapes } from "./src/shape.ts"
export type { DType, RandomKind } from "./src/storage.ts"
export { fromFlat, Tensor } from "./src/tensor.ts"
export type { NestedNumbers, ShapeOf } from "./src/tensor.ts"

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
export { DimAdd, DimMul, DimSub } from "./src/shape.ts"
export type {
  Broadcast,
  BroadcastCheck,
  BroadcastToCheck,
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
  Slice,
  SliceShape,
  Squeeze,
  Stack,
  Transpose,
  Unsqueeze,
} from "./src/shape.ts"
