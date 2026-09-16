export { noGrad } from "./src/autograd.ts"
export { compile, printGraph } from "./src/compile.ts"
export type { CompiledFn } from "./src/compile.ts"
export { context, eager, lazy, withContext } from "./src/context.ts"
export type { RuntimeContext } from "./src/context.ts"
export { arange, cat, categorical, eye, full, ones, rand, randn, scalar, stack, tensor, zeros } from "./src/factories.ts"
export type { CategoricalOptions, ResampleOptions } from "./src/factories.ts"
export { configure } from "./src/lazy.ts"
export { broadcastShapes } from "./src/shape.ts"
export type { DType, NumericArray, RandomKind, TypedArray } from "./src/storage.ts"
export { fromFlat, Tensor } from "./src/tensor.ts"
export type { AnyTensor, NestedNumbers, ShapeOf } from "./src/tensor.ts"

export {
  disableNative,
  isNativeAvailable,
  isNativeEnabled,
  nativeCounters,
  nativeDevice,
  nativeDeviceInfo,
  nativeDeviceMode,
  useNative,
} from "./src/backends/native.ts"
export { jsCounters, resetJsCounters } from "./src/counters.ts"
export * from "./src/nn/index.ts"
export { Adam, AdamW, clipGradNorm, Optimizer, SGD } from "./src/optim/index.ts"
export type { AdamOptions, AdamWOptions, OptimizerSource, OptimizerStateDict, OptimizerStateEntry, SGDOptions } from "./src/optim/index.ts"
export { constant, cosine, linearDecay, oneCycle, stepDecay, warmup, warmupCosine } from "./src/optim/schedule.ts"
export type { Schedule } from "./src/optim/schedule.ts"

// DimAdd, DimMul and the other shape helpers are both a type and a value, so a width like DimMul(4, d) keeps its derived type.
export { assertChecked, ConvOut, DimAdd, DimDiv, DimMul, DimSub, flattenFrom, flattenShape, PoolOut, unflattenShape } from "./src/shape.ts"
export type {
  BatchPrefix,
  Broadcast,
  BroadcastCheck,
  BroadcastToCheck,
  CanBroadcast,
  Cat,
  ConvCheck,
  DimDivCheck,
  Drop,
  ErrorMessage,
  FlattenCheck,
  FlattenFrom,
  FlattenShape,
  IndexCheck,
  IndexTensor,
  InferShape,
  Init,
  Last,
  LastDimCheck,
  MatMul,
  MatMulCheck,
  NestedArray,
  Permute,
  Prod,
  ReduceDim,
  ReduceDims,
  ResizeDim,
  ResolveView,
  Shape,
  Slice,
  SliceCheck,
  SliceShape,
  Squeeze,
  Stack,
  Take,
  Transpose,
  UnflattenCheck,
  UnflattenShape,
  Unsqueeze,
} from "./src/shape.ts"
