// Sweeps `TYPENET_THREADS=1..10` × `{add, mul-chain, gelu-composed,
// sum-last-axis}` × `n ∈ {1k,4k,16k,64k,256k,1M,4M}` and writes the raw
// grid — the input to W6.7's measured constant table (PLAN-V2 §4.2,
// W0.3).
//
// `TYPENET_THREADS` is read once behind a Rust `OnceLock`
// (`native/src/lib.rs`'s `switches()`), so a same-process env mutation
// between samples would silently never be observed. This driver instead
// spawns one subprocess per thread count, with the env var set *before*
// that process's native addon ever loads; each subprocess (the "worker")
// runs the full op × size grid for its one fixed thread count and
// appends its own JSONL rows directly, via the same `bench()` harness
// (which reads `--only`/`--mode`/`--device`/`--tag` from its own argv,
// forwarded unchanged from the driver).

import { spawnSync } from "node:child_process"
import { disableNative, isNativeAvailable, useNative } from "../index.ts"
import { rand } from "../src/factories.ts"
import { configure } from "../src/lazy.ts"
import type { AnyTensor } from "../src/tensor.ts"
import { bench, type BenchCaseSpec, isSmokeRun } from "./lib/harness.ts"
import { THREADING_FULL, THREADING_SMOKE } from "./lib/sizes.ts"

// `--full` (forwarded to every spawned worker below) is what each
// subprocess re-reads to pick the same config the driver used.
const SIZE_CONFIG = isSmokeRun() ? THREADING_SMOKE : THREADING_FULL
const THREAD_COUNTS = SIZE_CONFIG.threadCounts
const SIZES = SIZE_CONFIG.sizes
const REDUCE_COLS = SIZE_CONFIG.reduceCols // divides every size above evenly

type Op = "add" | "mul-chain" | "gelu-composed" | "sum-last-axis"
const OPS: readonly Op[] = ["add", "mul-chain", "gelu-composed", "sum-last-axis"]

const WORKER_MARKER = "TN_THREADING_WORKER"

function gelu(x: AnyTensor): AnyTensor {
  // tanh-approximate GELU: 0.5x(1 + tanh(sqrt(2/pi)(x + 0.044715x^3))) —
  // the D10-weighted "8" transcendental op, hand-composed (no fused GELU
  // kernel exists yet).
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
    // Halved from the harness default (still floored at 3/10 there) —
    // 10 thread counts × 4 ops × 7 sizes already means 280 grid points;
    // the point of this script is the *shape* of the threading curve,
    // not tight percentiles on each cell.
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

  // `TYPENET_THREADS` is only ever *read* (behind that `OnceLock`) once an
  // actual native op runs — `isNativeAvailable()` above only `require()`s
  // the addon, it never touches switches() — so with exactly one thread
  // count to run (the smoke default; see `THREADING_SMOKE` in
  // bench/lib/sizes.ts) there is nothing to isolate a subprocess against:
  // setting the env var here and calling `runWorker` in-process gets the
  // identical result for a fraction of the cost of a second `vite-node`
  // startup, which is what pushed this script's smoke run over the 30s
  // budget. Sweeping more than one count still needs one subprocess per
  // value, since the first real op in *this* process would freeze the
  // OnceLock for the rest of its lifetime.
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
