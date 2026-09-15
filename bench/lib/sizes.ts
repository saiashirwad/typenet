// Shared shape table every bench script imports from. Each case family has
// a `_SMOKE` counterpart picked via harness `isSmokeRun()` (true unless
// `--full`).

export interface MlpCase {
  id: string
  inputDim: number
  hiddenDim: number
  outputDim: number
  batch: number
}

// Frozen benchmark definition: the recorded native baselines and targets
// refer to this exact case. A new baseline needs a new case id, not an edit
// to this one.
export const MLP_LEGACY: readonly MlpCase[] = [
  { id: "mlp-legacy-b64", inputDim: 784, hiddenDim: 256, outputDim: 10, batch: 64 },
  { id: "mlp-legacy-b512", inputDim: 784, hiddenDim: 256, outputDim: 10, batch: 512 },
]

// Different model than MLP_LEGACY (GELU/Dropout/AdamW); never compare its
// numbers against an mlp-legacy baseline.
export const MLP_MODERN: readonly MlpCase[] = [
  { id: "mlp-modern-b64", inputDim: 784, hiddenDim: 256, outputDim: 10, batch: 64 },
  { id: "mlp-modern-b512", inputDim: 784, hiddenDim: 256, outputDim: 10, batch: 512 },
]

export const MLP_CASES: readonly MlpCase[] = [...MLP_LEGACY, ...MLP_MODERN]

// Smoke: the frozen MLP_LEGACY shape with only the batch shrunk.
export const MLP_SMOKE: readonly MlpCase[] = [
  { id: "mlp-legacy-smoke-b8", inputDim: 784, hiddenDim: 256, outputDim: 10, batch: 8 },
]

export interface NanoGptCase {
  id: "nanogpt-s" | "nanogpt-m" | "nanogpt-l" | "nanogpt-smoke"
  batch: number
  blockSize: number
  nEmbd: number
  nHead: number
  nLayer: number
  vocabSize: number
  approxParams: number
}

export const NANOGPT_SIZES: readonly NanoGptCase[] = [
  { id: "nanogpt-s", batch: 32, blockSize: 64, nEmbd: 128, nHead: 4, nLayer: 4, vocabSize: 65, approxParams: 0.81e6 },
  { id: "nanogpt-m", batch: 32, blockSize: 128, nEmbd: 256, nHead: 8, nLayer: 6, vocabSize: 65, approxParams: 4.79e6 },
  {
    id: "nanogpt-l",
    batch: 64,
    blockSize: 256,
    nEmbd: 384,
    nHead: 6,
    nLayer: 6,
    vocabSize: 65,
    approxParams: 10.77e6,
  },
]

// Tiny config so every `nanogpt-*` bench script still runs a real (if
// minuscule) transformer step; nHead 4 divides nEmbd 32 evenly.
export const NANOGPT_SMOKE: NanoGptCase = {
  id: "nanogpt-smoke",
  batch: 4,
  blockSize: 16,
  nEmbd: 32,
  nHead: 4,
  nLayer: 1,
  vocabSize: 65,
  approxParams: 0.02e6,
}

/** Shared by bench/macro-embedding.ts and bench/micro-gather-scatter.ts. */
export interface EmbeddingSizeConfig {
  vocabs: readonly number[]
  idsPerStep: number
  embedDim: number
}

export const EMBEDDING_FULL: EmbeddingSizeConfig = {
  vocabs: [65, 4096, 50257],
  idsPerStep: 16_384,
  embedDim: 128,
}

export const EMBEDDING_SMOKE: EmbeddingSizeConfig = {
  vocabs: [65],
  idsPerStep: 256,
  embedDim: 16,
}

export interface ElementwiseSizeConfig {
  sizes: readonly number[]
  chainLengths: readonly number[]
}

export const ELEMENTWISE_FULL: ElementwiseSizeConfig = {
  sizes: [4_000, 16_000, 64_000, 256_000, 1_000_000, 4_000_000],
  chainLengths: [1, 5, 12, 24],
}

export const ELEMENTWISE_SMOKE: ElementwiseSizeConfig = {
  sizes: [4_000, 16_000],
  chainLengths: [1, 5],
}

export interface MatmulSizeConfig {
  square: readonly number[]
  f64Square: readonly number[]
  accelerateDirectN: number
}

export const MATMUL_FULL: MatmulSizeConfig = {
  square: [256, 512, 1024, 2048],
  f64Square: [128, 256],
  accelerateDirectN: 4096,
}

export const MATMUL_SMOKE: MatmulSizeConfig = {
  square: [64, 128],
  f64Square: [32, 64],
  accelerateDirectN: 256,
}

export interface ReduceShape {
  label: string
  shape: readonly number[]
}

export const REDUCE_SHAPES_FULL: readonly ReduceShape[] = [
  { label: "small", shape: [1024, 64] },
  { label: "below-route", shape: [2048, 256] },
  { label: "gemv-route", shape: [8192, 256] },
  { label: "large", shape: [65536, 128] },
]

// Smallest two families only, shrunk further so every op x axis pair is
// instant.
export const REDUCE_SHAPES_SMOKE: readonly ReduceShape[] = [
  { label: "small", shape: [128, 32] },
  { label: "below-route", shape: [256, 64] },
]

export interface SoftmaxLnSizeConfig {
  rows: number
  widths: readonly number[]
  ceRows: number
  ceCols: number
}

export const SOFTMAX_LN_FULL: SoftmaxLnSizeConfig = {
  rows: 4096,
  widths: [64, 128, 384, 1024],
  ceRows: 16384,
  ceCols: 65,
}

export const SOFTMAX_LN_SMOKE: SoftmaxLnSizeConfig = {
  rows: 256,
  widths: [64, 128],
  ceRows: 512,
  ceCols: 65,
}

export interface ThreadingSizeConfig {
  threadCounts: readonly number[]
  sizes: readonly number[]
  reduceCols: number
}

export const THREADING_FULL: ThreadingSizeConfig = {
  threadCounts: Array.from({ length: 10 }, (_, i) => i + 1),
  sizes: [1_000, 4_000, 16_000, 64_000, 256_000, 1_000_000, 4_000_000],
  reduceCols: 100,
}

// One thread count because sweeping needs a fresh subprocess per value
// (native reads TYPENET_THREADS once behind a OnceLock) and per-process
// vite-node startup (~15-20s) blew the 30s smoke budget; --full still
// sweeps THREADING_FULL.threadCounts.
export const THREADING_SMOKE: ThreadingSizeConfig = {
  threadCounts: [1],
  sizes: [1_000, 4_000],
  reduceCols: 100,
}
