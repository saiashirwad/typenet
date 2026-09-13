// The shared config table every bench script reads shapes from (PLAN-V2
// §4.2, W0.1). No bench script hardcodes a shape — it imports one of these.
//
// Every case family below carries a `_SMOKE` counterpart: tiny sizes meant
// to prove a bench script still runs end to end in well under 30s, never to
// produce a number worth comparing against a `_FULL`/unsuffixed baseline
// (see bench/README.md). A script picks between them with
// `bench/lib/harness.ts`'s `isSmokeRun()`, which is true unless the CLI was
// given `--full`.

export interface MlpCase {
  id: string
  inputDim: number
  hiddenDim: number
  outputDim: number
  batch: number
}

// `mlp-legacy-*` is pinned to one definition for the life of the plan
// (PLAN-V2 §4.2): `Linear(784,256) -> relu -> Linear(256,10)`, `mseLoss`,
// `Adam(lr 1e-3)`. §0's 1.50 ms / 2.13 ms native numbers and every §6.2 row
// and G3.1 / G4.7 / G5.4 / G6.2 target refer to this exact case. Do not
// change its shape, batch sizes, loss or optimizer here — a new baseline
// needs a new case id, not an edit to this one. (`bench/models/mlp.ts`,
// created by W0.2, is what actually builds this model against today's API;
// this file only carries the shape data.)
export const MLP_LEGACY: readonly MlpCase[] = [
  { id: "mlp-legacy-b64", inputDim: 784, hiddenDim: 256, outputDim: 10, batch: 64 },
  { id: "mlp-legacy-b512", inputDim: 784, hiddenDim: 256, outputDim: 10, batch: 512 },
]

// `mlp-modern-*` arrives with its own baseline in W5.4 (`Linear(784,256) ->
// GELU -> Dropout(0.1) -> Linear(256,10)`, `crossEntropy`,
// `AdamW(lr 3e-4, wd 0.01)`). Defined here now, as data, so every later
// script imports the same shape. Never quote its numbers against
// `mlp-legacy`'s baseline — that mixes two different models.
export const MLP_MODERN: readonly MlpCase[] = [
  { id: "mlp-modern-b64", inputDim: 784, hiddenDim: 256, outputDim: 10, batch: 64 },
  { id: "mlp-modern-b512", inputDim: 784, hiddenDim: 256, outputDim: 10, batch: 512 },
]

export const MLP_CASES: readonly MlpCase[] = [...MLP_LEGACY, ...MLP_MODERN]

// Smoke: one batch-8 step of the frozen `mlp-legacy` shape (784/256/10 stay
// exactly as `MLP_LEGACY` defines them — only the batch shrinks) — enough
// to prove `bench/macro-mlp.ts` still runs a real forward+backward+Adam
// step, at owner-specified batch 8.
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
  /** Approximate parameter count, as measured in PLAN-V2 §0. */
  approxParams: number
}

// nanoGPT S / M / L exactly as PLAN-V2 §0 defines them:
//   S  B32 T64  d128 H4 L4 V65   0.81 M params
//   M  B32 T128 d256 H8 L6 V65   4.79 M params
//   L  B64 T256 d384 H6 L6 V65  10.77 M params
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

// Smoke: one deliberately tiny config — vocab 65 / block 16 / d 32 / 1
// layer / batch 4 (owner-specified) — so every `nanogpt-*` bench script
// still runs a real (if minuscule) transformer step. `nHead: 4` divides
// `nEmbd: 32` evenly (headDim 8).
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

// --- attention / embedding / gather-scatter shared sizes -------------------

/** `bench/macro-embedding.ts` and `bench/micro-gather-scatter.ts` share this
 * shape: vocab sizes swept, ids looked up per step, and the embedding
 * width. Full matches PLAN-V2 §4.2 (16 384 ids = batch×block, dim 128);
 * smoke keeps only the smallest vocab and shrinks the rest so a lookup /
 * scatter over even the "large" full vocab never has to run. */
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

// --- micro-elementwise -----------------------------------------------------

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

// --- micro-matmul ------------------------------------------------------------

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

// --- micro-reduce ------------------------------------------------------------

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

// Smallest two families only, shrunk further so every op × axis pair is
// instant.
export const REDUCE_SHAPES_SMOKE: readonly ReduceShape[] = [
  { label: "small", shape: [128, 32] },
  { label: "below-route", shape: [256, 64] },
]

// --- micro-softmax-ln --------------------------------------------------------

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

// --- micro-threading ---------------------------------------------------------

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

// Owner-specified target was 2 thread counts x 2 sizes, but each thread
// count here is a full subprocess spawn (native's `TYPENET_THREADS` is
// read once behind a `OnceLock`, so sweeping it needs a fresh process per
// value) and this project's per-process `vite-node`/tsover startup cost
// alone is already ~15-20s — two spawns blew the 30s smoke budget (~45s
// measured). One thread count keeps the subprocess-driver code path
// exercised (still a real spawn + worker run) while fitting the budget;
// `--full` still sweeps the full `THREADING_FULL.threadCounts`.
export const THREADING_SMOKE: ThreadingSizeConfig = {
  threadCounts: [1],
  sizes: [1_000, 4_000],
  reduceCols: 100,
}
