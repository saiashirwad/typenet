// Sweeps `TYPENET_THREADS=1..10` x ops x sizes and writes the raw grid.
//
// `TYPENET_THREADS` is read once behind a Rust OnceLock, so a same-process
// env mutation between samples would silently never be observed. The
// driver instead spawns one subprocess per thread count, with the env var
// set before that process's native addon loads; each worker runs the full
// op x size grid and appends its own JSONL rows via the shared `bench()`
// harness (flags forwarded unchanged from the driver).

import { spawnSync } from "node:child_process"
import { disableNative, isNativeAvailable, useNative } from "../index.ts"
import { rand } from "../src/factories.ts"
import { configure } from "../src/lazy.ts"
import type { AnyTensor } from "../src/tensor.ts"
import { bench, type BenchCaseSpec, isSmokeRun } from "./lib/harness.ts"
import { THREADING_FULL, THREADING_SMOKE } from "./lib/sizes.ts"

// `--full` is forwarded to every spawned worker, which re-reads it.
const SIZE_CONFIG = isSmokeRun() ? THREADING_SMOKE : THREADING_FULL
const THREAD_COUNTS = SIZE_CONFIG.threadCounts
const SIZES = SIZE_CONFIG.sizes
const REDUCE_COLS = SIZE_CONFIG.reduceCols // divides every size above evenly

type Op = "add" | "mul-chain" | "gelu-composed" | "sum-last-axis"
const OPS: readonly Op[] = ["add", "mul-chain", "gelu-composed", "sum-last-axis"]

const WORKER_MARKER = "TN_THREADING_WORKER"

function gelu(x: AnyTensor): AnyTensor {
  // Tanh-approximate GELU, hand-composed (no fused kernel exists yet).
  const c = Math.sqrt(2 / Math.PI)
  const inner = x.add(x.pow(3).mul(0.044715)).mul(c)
  return x.mul(0.5).mul(inner.tanh().add(1))
}

function applyOp(x: AnyTensor, op: Op): AnyTensor {
  switch (op) {
    case "add":
      return x.add(1)
    case "mul-chain": {
      let h = x
      for (let i = 0; i < 5; i++) h = h.mul(1.0001)
      return h
    }
    case "gelu-composed":
      return gelu(x)
    case "sum-last-axis":
      return x.view([x.numel / REDUCE_COLS, REDUCE_COLS]).sum(-1)
  }
}

interface ThreadCase extends BenchCaseSpec {
  op: Op
  n: number
}

async function runWorker(threads: number): Promise<void> {
  configure({ lazy: true })
  useNative()

  const cases: ThreadCase[] = SIZES.flatMap(n => OPS.map(op => ({ id: `${op}-n${n}-t${threads}`, op, n, modes: ["native"] as const })))

  const inputs = new Map<number, AnyTensor>()
  await bench(
    "micro-threading",
    cases,
    kase => {
      let x = inputs.get(kase.n)
      if (!x) {
        x = rand([kase.n]) as AnyTensor
        inputs.set(kase.n, x)
      }
      const out = applyOp(x, kase.op)
      out.data // force materialization
    },
    // Halved from the harness default: 280 grid points already, and the
    // point is the shape of the threading curve, not tight percentiles.
    { warmup: 3, samples: 10 },
  )
}

async function runDriver(): Promise<void> {
  if (!isNativeAvailable()) {
    console.log(
      "micro-threading: @typenet/native is not built — TYPENET_THREADS only matters on the "
        + "native path, so this script has nothing to sweep. Skipping (green, no rows written).",
    )
    return
  }

  // With exactly one thread count there is nothing to isolate a subprocess
  // against: the OnceLock only freezes once a real native op runs, so
  // setting the env var and running in-process gives the identical result
  // for a fraction of the cost of a second vite-node startup. Sweeping
  // more than one count still needs one subprocess per value.
  if (THREAD_COUNTS.length === 1) {
    const threads = THREAD_COUNTS[0]!
    console.log(`micro-threading: TYPENET_THREADS=${threads} (in-process — only one count to run)`)
    process.env.TYPENET_THREADS = String(threads)
    await runWorker(threads)
    return
  }

  const forwardArgs = process.argv.slice(2)
  const thisFile = new URL(import.meta.url).pathname

  for (const threads of THREAD_COUNTS) {
    console.log(`micro-threading: TYPENET_THREADS=${threads}`)
    const result = spawnSync(
      "vite-node",
      [thisFile, ...forwardArgs],
      {
        stdio: "inherit",
        env: { ...process.env, TYPENET_THREADS: String(threads), [WORKER_MARKER]: "1" },
      },
    )
    if (result.error) throw result.error
    if (result.status !== 0) {
      process.exitCode = result.status ?? 1
      return
    }
  }
}

if (process.env[WORKER_MARKER]) {
  await runWorker(Number(process.env.TYPENET_THREADS))
} else {
  disableNative()
  await runDriver()
}
