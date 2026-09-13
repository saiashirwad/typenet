// Every square/rectangular GEMM shape a nanoGPT-shaped step touches, plus
// batched `[B,H,T,K]` (contiguous and permuted), f32 and f64 (PLAN-V2
// §4.2, W0.3). Accepts a `>= 1 TFLOP/s` case on the native path
// (Accelerate is linked at `native/src/lib.rs:1883-1942`).

import { disableNative, nativeCounters, useNative } from "../index.ts"
import { sgemmNative } from "../src/backends/native.ts"
import { rand } from "../src/factories.ts"
import { configure } from "../src/lazy.ts"
import type { AnyTensor } from "../src/tensor.ts"
import { bench, type BenchCaseSpec, isSmokeRun, type Counters, type Mode } from "./lib/harness.ts"
import { MATMUL_FULL, MATMUL_SMOKE, NANOGPT_SIZES, NANOGPT_SMOKE } from "./lib/sizes.ts"

type F32Case = { kind: "f32"; m: number; k: number; n: number; batch?: number; permuted?: boolean }
type F64Case = { kind: "f64"; m: number; k: number; n: number }
interface MatmulCase extends BenchCaseSpec {
  spec: F32Case | F64Case
}

const SMOKE = isSmokeRun()
const SIZE_CONFIG = SMOKE ? MATMUL_SMOKE : MATMUL_FULL
// Smoke keeps only the single owner-specified tiny nanoGPT config in place
// of the S/M/L sweep below.
const GPT_SIZES = SMOKE ? [NANOGPT_SMOKE] : NANOGPT_SIZES

const cases: MatmulCase[] = []

for (const s of SIZE_CONFIG.square) {
  cases.push({ id: `mm-square-${s}`, spec: { kind: "f32", m: s, k: s, n: s } })
}

// nanoGPT-shaped GEMMs: the QKV/output projection ([B*T, D] @ [D, D])
// and the MLP up-projection ([B*T, D] @ [D, 4D]) at each of S/M/L.
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
  // Batched attention scores: [B,H,T,Dh] @ [B,H,Dh,T], contiguous and
  // permuted (the permuted operand goes through an actual materializing
  // transpose — see `bench/models/attention.ts` for why a bare permute
  // into native `matmul` is not representative today).
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

// f64: naive fallback only (native sgemm is f32-only) — restricted away
// from "native" so this doesn't crash trying to route f64 through it.
for (const s of SIZE_CONFIG.f64Square) {
  cases.push({ id: `mm-square-${s}-f64`, spec: { kind: "f64", m: s, k: s, n: s }, modes: ["eager", "interp"] })
}

// Every `mm-*` case above measures GEMM the way the rest of this bench
// suite measures every op: through `Tensor.matmul` under whichever mode
// is active. Under "native" that means the *lazy Program* evaluator —
// which (pre-D1/W3.5) still dispatches matmul through candle, not
// straight to Accelerate, and candle is measurably slower at every GEMM
// shape (§0 fact 4) — so it alone tops out well under 1 TFLOP/s on this
// machine (candle Metal MLP step 3.56 ms vs candle CPU 1.50, per §0).
// `sgemmNative` is the *other* native entry point: the raw Accelerate
// binding `eager.ts` already calls opportunistically for a large,
// unbatched f32 `matmul` even outside lazy mode (`isNativeEnabled() &&
// batchCount === 1`, see its comment there). This case calls it directly
// to demonstrate what "Accelerate is already linked" (this item's own
// accept text) actually delivers, independent of candle's interim GEMM
// path — the number the ≥ 1 TFLOP/s criterion is about.
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
    // A genuinely permuted (transposed) operand: build `b` with its last
    // two axes swapped, then permute it back — a materialized array
    // whose logical transpose is a non-trivial stride, not a bare
    // relabel of a contiguous buffer.
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

// Above this, the naive triple-loop eager fallback (no batched sgemm
// fast path — `eager.ts`'s Accelerate shortcut only fires for
// `batchCount === 1`, and "eager" here means native explicitly disabled)
// takes tens of seconds to minutes *per timed sample*, ×13 samples ×
// (potentially) 2 modes. Above the threshold a case runs native-only —
// that is exactly the mode this script's TFLOP/s claim is about anyway.
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
      void out // force materialization (sgemmNative already returns a resolved Float32Array)
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
