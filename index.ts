export { noGrad } from "./src/autograd.ts"
export { compile, printGraph } from "./src/compile.ts"
export type { CompiledFn } from "./src/compile.ts"
export { context, eager, lazy, withContext } from "./src/context.ts"
export type { RuntimeContext } from "./src/context.ts"
export { arange, cat, eye, full, ones, rand, randn, scalar, stack, tensor, zeros } from "./src/factories.ts"
export { configure } from "./src/lazy.ts"
export { broadcastShapes } from "./src/shape.ts"
export type { DType, RandomKind } from "./src/storage.ts"
export { fromFlat, Tensor } from "./src/tensor.ts"
export type { NestedNumbers, ShapeOf } from "./src/tensor.ts"

export {
  disableNative,
  isNativeAvailable,
  isNativeEnabled,
  nativeCounters,
  nativeDevice,
  nativeDeviceInfo,
  nativeDeviceMode,
  nativeProfile,
  useNative,
} from "./src/backends/native.ts"
export { jsCounters, resetJsCounters } from "./src/counters.ts"
export * from "./src/nn/index.ts"
export { Adam, AdamW, clipGradNorm, Optimizer, SGD } from "./src/optim.ts"
export type { AdamOptions, AdamWOptions, OptimizerSource, OptimizerStateDict, OptimizerStateEntry, SGDOptions } from "./src/optim.ts"
// Plain `(step: number) => number` schedules, decoupled from `Optimizer`;
// assign `opt.lr = schedule(step)` yourself.
export { constant, cosine, linearDecay, oneCycle, stepDecay, warmup, warmupCosine } from "./src/optim/schedule.ts"
export type { Schedule } from "./src/optim/schedule.ts"

// DimAdd / DimMul are both a type and a value: the type does the
// arithmetic on literal dims, the function returns it at runtime, so a
// constructor width like `DimAdd(DimMul(3, channels), 1)` carries its
// derived type with no cast.
export { DimAdd, DimDiv, DimMul, DimSub } from "./src/shape.ts"
// Same dual type/value shape for conv/pool spatial arithmetic: a head width
// like `DimMul(DimMul(16, PoolOut(11, 2, 2)), PoolOut(11, 2, 2))` carries its
// derived type with no cast.
export { assertChecked } from "./src/cast.ts"
export { ConvOut, flattenFrom, PoolOut } from "./src/shape.ts"
export { flattenShape, unflattenShape } from "./src/shape.ts"
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
