// The phase ladder behind PLAN-V2 §0 fact 2: typenet's compiled MLP step
// is dominated by FFI parameter marshalling, not compute. Each phase adds
// one more thing that crosses the JS<->native boundary on every call --
// an empty compiled graph, a `[64,784]` input, a real forward pass, a
// forward+backward pass, +SGD, +Adam -- and this prints µs/step plus an
// implied marshalling rate in GB/s for each.
//
// Only `native` mode actually crosses the FFI boundary, so every phase
// runs there exclusively: `eager` and `interp` never leave the JS
// process, so there is nothing to marshal and no floor to measure.
//
// `pnpm vite-node bench/micro-ffi.ts` (no flags) is the accept-tested
// invocation; `--only <substr>` still filters phases when this is run
// through `pnpm bench:micro`.

import { Adam, compile, mseLoss, rand, SGD, Tensor } from "../index.ts"
import { parseCliArgs } from "./lib/cli.ts"
import { isSmokeRun } from "./lib/harness.ts"
import { appendResult, defaultCounters } from "./lib/report.ts"
import { mlpLegacyData, mlpLegacyNet, setMode } from "./models/mlp.ts"

type AnyTensor = Tensor<any>

const BATCH = 64
const BYTES_PER_F32 = 4

function bytesOf(t: AnyTensor | AnyTensor[]): number {
  const list = Array.isArray(t) ? t : [t]
  return list.reduce((sum, x) => sum + x.numel * BYTES_PER_F32, 0)
}

interface Phase {
  id: string
  /** One compiled call, returning whatever the compiled fn returns (for byte accounting). */
  call: () => AnyTensor | AnyTensor[]
  /** Bytes resent JS -> native on every call: placeholders plus every optimizer update target. */
  dirtyBytes: number
}

function buildPhases(): Phase[] {
  const phases: Phase[] = []

  // Phase 1: the FFI floor itself -- the smallest possible compiled call.
  {
    const x = rand([1]) as AnyTensor
    const step = compile((xIn: AnyTensor) => xIn.mul(2))
    phases.push({ id: "ffi-empty", call: () => step(x), dirtyBytes: bytesOf(x) })
  }

  // Phase 2: the same op, but on the `[64,784]` shape the MLP forward
  // pass reads -- isolates input marshalling from any actual compute.
  {
    const x = rand([BATCH, 784]) as AnyTensor
    const step = compile((xIn: AnyTensor) => xIn.mul(2))
    phases.push({ id: "ffi-input-64x784", call: () => step(x), dirtyBytes: bytesOf(x) })
  }

  // Phase 3: forward only.
  {
    const net = mlpLegacyNet()
    const { x } = mlpLegacyData(BATCH)
    const step = compile((xIn: AnyTensor) => net.forward(xIn))
    phases.push({ id: "ffi-forward", call: () => step(x), dirtyBytes: bytesOf(x) })
  }

  // Phase 4: forward + backward, no optimizer. Returning every
  // parameter's gradient as an extra output is what forces the native
  // backend to actually compute (and marshal back) the backward pass --
  // a bare `loss.backward()` whose grads nothing reads would let the
  // compiled graph prune the backward half away as dead code.
  {
    const net = mlpLegacyNet()
    const { x, y } = mlpLegacyData(BATCH)
    const step = compile((xIn: AnyTensor, yIn: AnyTensor) => {
      const loss = mseLoss(net.forward(xIn), yIn)
      loss.backward()
      return [loss, ...net.parameters().map(p => p.grad as AnyTensor)]
    })
    phases.push({ id: "ffi-forward-backward", call: () => step(x, y), dirtyBytes: bytesOf([x, y]) })
  }

  // Phase 5: + SGD. Every update target -- each parameter, plus its
  // momentum buffer -- is resent on every call, over the same 203 k
  // parameters as §0 fact 2.
  {
    const net = mlpLegacyNet()
    const { x, y } = mlpLegacyData(BATCH)
    const optim = new SGD(net.parameters(), { lr: 0.1, momentum: 0.9 })
    const step = compile((xIn: AnyTensor, yIn: AnyTensor) => {
      const loss = mseLoss(net.forward(xIn), yIn)
      optim.zeroGrad()
      loss.backward()
      optim.step()
      return loss
    })
    const paramBytes = bytesOf(net.parameters())
    phases.push({
      id: "ffi-sgd",
      call: () => step(x, y),
      // placeholders + {param, momentum} update targets per parameter.
      dirtyBytes: bytesOf([x, y]) + 2 * paramBytes,
    })
  }

  // Phase 6: + Adam. Adam resends two state buffers (m, v) per
  // parameter on top of the parameter itself -- this is where the
  // measured 8.5x-off-ceiling comes from.
  {
    const net = mlpLegacyNet()
    const { x, y } = mlpLegacyData(BATCH)
    const optim = new Adam(net.parameters(), { lr: 1e-3 })
    const step = compile((xIn: AnyTensor, yIn: AnyTensor) => {
      const loss = mseLoss(net.forward(xIn), yIn)
      optim.zeroGrad()
      loss.backward()
      optim.step()
      return loss
    })
    const paramBytes = bytesOf(net.parameters())
    phases.push({
      id: "ffi-adam",
      call: () => step(x, y),
      // placeholders + {param, m, v} update targets per parameter.
      dirtyBytes: bytesOf([x, y]) + 3 * paramBytes,
    })
  }

  return phases
}

function percentileOf(sorted: readonly number[], p: number): number {
  const idx = Math.min(sorted.length - 1, Math.max(0, Math.round((sorted.length - 1) * p)))
  return sorted[idx]!
}

interface BatchedTiming {
  medianMs: number
  p10Ms: number
  p90Ms: number
  lastOut: AnyTensor | AnyTensor[]
}

/**
 * Times `call` in batches sized so each batch takes at least
 * `targetBatchMs` -- at a floor of a few µs/call, timing one call at a
 * time makes `performance.now()`'s own overhead and V8's per-call
 * bookkeeping a large fraction of what gets measured. Doubling the inner
 * loop until a batch clears the target, then dividing batch time by
 * batch size, amortizes that overhead away.
 */
function timeBatched(
  call: () => AnyTensor | AnyTensor[],
  warmupBatches: number,
  samples: number,
  targetBatchMs = 2,
): BatchedTiming {
  let lastOut: AnyTensor | AnyTensor[] = call()

  let inner = 1
  for (;;) {
    const t0 = performance.now()
    for (let i = 0; i < inner; i++) lastOut = call()
    if (performance.now() - t0 >= targetBatchMs || inner >= 1 << 20) break
    inner *= 2
  }

  for (let b = 0; b < warmupBatches; b++) {
    for (let i = 0; i < inner; i++) lastOut = call()
  }

  const timingsMs: number[] = []
  for (let s = 0; s < samples; s++) {
    const t0 = performance.now()
    for (let i = 0; i < inner; i++) lastOut = call()
    timingsMs.push((performance.now() - t0) / inner)
  }

  const sorted = [...timingsMs].sort((a, b) => a - b)
  return {
    medianMs: percentileOf(sorted, 0.5),
    p10Ms: percentileOf(sorted, 0.10),
    p90Ms: percentileOf(sorted, 0.90),
    lastOut,
  }
}

async function main(): Promise<void> {
  const args = parseCliArgs()

  if (args.modes.length > 0 && !args.modes.includes("native")) {
    console.log(
      `bench micro-ffi: only the "native" mode crosses the FFI boundary; nothing to run for --mode ${args.modes.join(",")}`,
    )
    return
  }

  setMode("native")

  const phases = buildPhases().filter(p => args.only === undefined || p.id.includes(args.only))
  if (phases.length === 0) {
    console.log(`bench micro-ffi: --only "${args.only}" matched no phases; ran nothing.`)
    return
  }

  // Smoke (the default, no `--full`) shrinks this to the owner's fixed
  // shape (1 warm-up batch, 2 timed samples) the same way
  // `bench/lib/harness.ts`'s `bench()` does for every other script.
  const smoke = isSmokeRun(args)
  const warmupBatches = smoke ? 1 : 3
  const samples = smoke ? 2 : 15
  const results = new Map<string, { medianUs: number; gbPerSec: number }>()

  for (const phase of phases) {
    const { medianMs, p10Ms, p90Ms, lastOut } = timeBatched(phase.call, warmupBatches, samples)
    const medianUs = medianMs * 1000
    const totalBytes = phase.dirtyBytes + bytesOf(lastOut)
    const gbPerSec = totalBytes / (medianMs / 1000) / 1e9
    results.set(phase.id, { medianUs, gbPerSec })

    appendResult({
      mode: "native",
      script: "micro-ffi",
      case: phase.id,
      n: samples,
      median_ms: medianMs,
      p10_ms: p10Ms,
      p90_ms: p90Ms,
      counters: defaultCounters(),
      smoke,
      ...(args.tag === undefined ? {} : { tag: args.tag }),
    })
  }

  console.log("\nmicro-ffi (native mode only -- this is what crosses the FFI boundary)")
  const idWidth = Math.max(5, ...[...results.keys()].map(id => id.length))
  console.log(["phase".padEnd(idWidth), "us/step".padStart(10), "GB/s".padStart(8)].join(" "))
  for (const [id, r] of results) {
    console.log([id.padEnd(idWidth), r.medianUs.toFixed(2).padStart(10), r.gbPerSec.toFixed(2).padStart(8)].join(" "))
  }

  const floor = results.get("ffi-empty")?.medianUs
  const adam = results.get("ffi-adam")?.medianUs
  if (floor !== undefined) {
    console.log(`\nempty-call floor: ${floor.toFixed(2)} us/step`)
    if (adam !== undefined) {
      console.log(`+Adam step: ${adam.toFixed(2)} us/step (${(adam / floor).toFixed(1)}x the empty-call floor)`)
    }
  }
}

await main()
