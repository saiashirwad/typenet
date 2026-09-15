// GEMM microbench: square/rect shapes a nanoGPT step touches, batched
// [B,H,T,K] (contiguous and permuted), f32 and f64, plus a direct-Accelerate
// case on the native path.

import { disableNative, nativeCounters, useNative } from "../index.ts"
import { sgemmNative } from "../src/backends/native.ts"
import { rand } from "../src/factories.ts"
import { configure } from "../src/lazy.ts"
import type { AnyTensor } from "../src/tensor.ts"
import { bench, type BenchCaseSpec, type Counters, isSmokeRun, type Mode } from "./lib/harness.ts"
import { MATMUL_FULL, MATMUL_SMOKE, NANOGPT_SIZES, NANOGPT_SMOKE } from "./lib/sizes.ts"

type F32Case = { kind: "f32"; m: number; k: number; n: number; batch?: number; permuted?: boolean }
type F64Case = { kind: "f64"; m: number; k: number; n: number }
interface MatmulCase extends BenchCaseSpec {
  spec: F32Case | F64Case
}

const SMOKE = isSmokeRun()
const SIZE_CONFIG = SMOKE ? MATMUL_SMOKE : MATMUL_FULL
const GPT_SIZES = SMOKE ? [NANOGPT_SMOKE] : NANOGPT_SIZES

const cases: MatmulCase[] = []

for (const s of SIZE_CONFIG.square) {
  cases.push({ id: `mm-square-${s}`, spec: { kind: "f32", m: s, k: s, n: s } })
}

// nanoGPT-shaped GEMMs: the QKV/output projection and the MLP
// up-projection at each size.
for (const size of GPT_SIZES) {
  const letter = size.id.split("-")[1]!
  const rows = size.batch * size.blockSize
  cases.push({
    id: `mm-proj-${letter}`,
    spec: { kind: "f32", m: rows, k: size.nEmbd, n: size.nEmbd },
  })
  cases.push({
    id: `mm-mlp-${letter}`,
    spec: { kind: "f32", m: rows, k: size.nEmbd, n: 4 * size.nEmbd },
  })
  // Batched attention scores, contiguous and permuted (the permuted
  // operand goes through a materializing transpose; see
  // bench/models/attention.ts for why).
  const headDim = size.nEmbd / size.nHead
  cases.push({
    id: `mm-batched-${letter}-contig`,
    spec: { kind: "f32", m: size.blockSize, k: headDim, n: size.blockSize, batch: size.batch * size.nHead },
  })
  cases.push({
    id: `mm-batched-${letter}-permuted`,
    spec: {
      kind: "f32",
      m: size.blockSize,
      k: headDim,
      n: size.blockSize,
      batch: size.batch * size.nHead,
      permuted: true,
    },
  })
}

// f64: native sgemm is f32-only, so restricted away from "native".
for (const s of SIZE_CONFIG.f64Square) {
  cases.push({ id: `mm-square-${s}-f64`, spec: { kind: "f64", m: s, k: s, n: s }, modes: ["eager", "interp"] })
}

// Under "native", `Tensor.matmul` routes through the lazy Program
// evaluator and candle, which is measurably slower at every GEMM shape
// than the raw Accelerate binding `sgemmNative` exposes. This case calls
// it directly to show what the linked Accelerate actually delivers,
// independent of candle's interim GEMM path.
const ACCELERATE_DIRECT_N = SIZE_CONFIG.accelerateDirectN
const ACCELERATE_DIRECT_ID = `mm-accelerate-direct-${ACCELERATE_DIRECT_N}`

function buildOperands(spec: F32Case | F64Case): { a: AnyTensor; b: AnyTensor } {
  if (spec.kind === "f64") {
    const a = (rand([spec.m, spec.k]) as AnyTensor).to("float64")
    const b = (rand([spec.k, spec.n]) as AnyTensor).to("float64")
    return { a, b }
  }
  const batch = spec.batch ?? 1
  const aShape = batch === 1 ? [spec.m, spec.k] : [batch, spec.m, spec.k]
  const bShape = batch === 1 ? [spec.k, spec.n] : [batch, spec.k, spec.n]
  let a = rand(aShape) as AnyTensor
  let b = rand(bShape) as AnyTensor
  if (spec.permuted) {
    // A genuinely permuted operand: build `b` with its last two axes
    // swapped, then permute it back, so the logical transpose is a
    // non-trivial stride rather than a relabel of a contiguous buffer.
    const swappedShape = batch === 1 ? [spec.n, spec.k] : [batch, spec.n, spec.k]
    const raw = rand(swappedShape) as AnyTensor
    b = batch === 1 ? raw.transpose(0, 1) : raw.transpose(1, 2)
  }
  return { a, b }
}

function flops(spec: F32Case | F64Case): number {
  const batch = spec.kind === "f32" ? (spec.batch ?? 1) : 1
  return 2 * batch * spec.m * spec.k * spec.n
}

// Above this, the naive eager fallback takes tens of seconds per timed
// sample (x13 samples x up to 2 modes), so such cases run native-only.
const HEAVY_FLOPS = 3e8
for (const c of cases) {
  if (c.modes === undefined && flops(c.spec) > HEAVY_FLOPS) {
    c.modes = ["native"]
  }
}

function setMode(mode: Mode): void {
  if (mode === "native") {
    configure({ lazy: true })
    useNative()
  } else if (mode === "interp") {
    disableNative()
    configure({ lazy: true })
  } else {
    disableNative()
    configure({ lazy: false })
  }
}

const tflopsByCase = new Map<string, Map<Mode, number>>()

const ACCELERATE_DIRECT_CASE: MatmulCase = {
  id: ACCELERATE_DIRECT_ID,
  spec: { kind: "f32", m: ACCELERATE_DIRECT_N, k: ACCELERATE_DIRECT_N, n: ACCELERATE_DIRECT_N },
  modes: ["native"],
}

async function main(): Promise<void> {
  const operands = new Map<string, { a: AnyTensor; b: AnyTensor }>()
  const n = ACCELERATE_DIRECT_N
  const accelA = new Float32Array(n * n).fill(0).map(() => Math.random())
  const accelB = new Float32Array(n * n).fill(0).map(() => Math.random())

  await bench("micro-matmul", [...cases, ACCELERATE_DIRECT_CASE], (kase, mode) => {
    if (kase.id === ACCELERATE_DIRECT_ID) {
      const t0 = performance.now()
      const out = sgemmNative(accelA, accelB, n, n, n)
      void out // already resolved eagerly; just marks it used
      const ms = performance.now() - t0
      const tflops = flops(kase.spec) / (ms / 1000) / 1e12
      if (!tflopsByCase.has(kase.id)) tflopsByCase.set(kase.id, new Map())
      tflopsByCase.get(kase.id)!.set(mode, tflops)
      return
    }

    setMode(mode)
    let pair = operands.get(kase.id)
    if (!pair) {
      pair = buildOperands(kase.spec)
      operands.set(kase.id, pair)
    }
    const t0 = performance.now()
    const out = pair.a.matmul(pair.b) as AnyTensor
    out.data // force materialization
    const ms = performance.now() - t0
    const tflops = flops(kase.spec) / (ms / 1000) / 1e12
    if (!tflopsByCase.has(kase.id)) tflopsByCase.set(kase.id, new Map())
    tflopsByCase.get(kase.id)!.set(mode, tflops)

    const counters: Partial<Counters> | undefined = mode === "native"
      ? (nativeCounters() as Partial<Counters>)
      : undefined
    return counters ? { counters } : undefined
  })

  configure({ lazy: false })
  disableNative()

  console.log("\nmicro-matmul: approx TFLOP/s (single-shot, not the harness median)")
  let best = 0
  for (const [id, byMode] of tflopsByCase) {
    for (const [mode, tflops] of byMode) {
      if (mode === "native") best = Math.max(best, tflops)
      console.log(`  ${id.padEnd(24)} ${mode.padEnd(8)} ${tflops.toFixed(3)} TFLOP/s`)
    }
  }
  console.log(`micro-matmul: best native TFLOP/s = ${best.toFixed(3)}`)
}

await main()
