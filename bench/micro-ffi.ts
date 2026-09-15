// FFI phase ladder: typenet's compiled MLP step is dominated by parameter
// marshalling, not compute. Each phase adds one more thing crossing the
// JS<->native boundary on every call, printing us/step plus an implied
// marshalling rate in GB/s. Only `native` mode crosses the boundary
// (eager/interp never leave the JS process), so every phase runs there
// exclusively.

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

  // The FFI floor: the smallest possible compiled call.
  {
    const x = rand([1]) as AnyTensor
    const step = compile((xIn: AnyTensor) => xIn.mul(2))
    phases.push({ id: "ffi-empty", call: () => step(x), dirtyBytes: bytesOf(x) })
  }

  // Same op at the [64,784] MLP input shape: isolates input marshalling
  // from any actual compute.
  {
    const x = rand([BATCH, 784]) as AnyTensor
    const step = compile((xIn: AnyTensor) => xIn.mul(2))
    phases.push({ id: "ffi-input-64x784", call: () => step(x), dirtyBytes: bytesOf(x) })
  }

  // Forward only.
  {
    const net = mlpLegacyNet()
    const { x } = mlpLegacyData(BATCH)
    const step = compile((xIn: AnyTensor) => net.forward(xIn))
    phases.push({ id: "ffi-forward", call: () => step(x), dirtyBytes: bytesOf(x) })
  }

  // Forward + backward, no optimizer. Returning every parameter's gradient
  // as an extra output is what forces the native backend to actually
  // compute (and marshal back) the backward pass: a bare `loss.backward()`
  // whose grads nothing reads would let the compiled graph prune the
  // backward half away as dead code.
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

  // + SGD: every parameter plus its momentum buffer is resent on every
  // call.
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

  // + Adam: two state buffers (m, v) resent per parameter on top of the
  // parameter itself -- the source of the measured 8.5x-off-ceiling.
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
 * `targetBatchMs`: at a few us/call, timing one call at a time makes
 * `performance.now()` overhead a large fraction of the measurement.
 * Doubling the inner loop until a batch clears the target amortizes it
 * away.
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
