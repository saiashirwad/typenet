// Chains of 1/5/12/24 elementwise ops at n ∈ {4k … 4M}, with and without
// a transcendental (PLAN-V2 §4.2, W0.3). Sweeps `TYPENET_BLK`,
// `TYPENET_PARALLEL_MIN`, `TYPENET_CHUNK` when those switches exist
// (W0.8); probes for them via `nativeDeviceInfo()` and prints a skip line
// instead of erroring when they are not declared yet — this script does
// not depend on W0.8 landing first.

import { disableNative, isNativeAvailable, nativeDeviceInfo, useNative } from "../index.ts"
import { rand } from "../src/factories.ts"
import { configure } from "../src/lazy.ts"
import type { AnyTensor } from "../src/tensor.ts"
import { bench, type BenchCaseSpec, isSmokeRun, type Mode } from "./lib/harness.ts"
import { ELEMENTWISE_FULL, ELEMENTWISE_SMOKE } from "./lib/sizes.ts"

const SIZE_CONFIG = isSmokeRun() ? ELEMENTWISE_SMOKE : ELEMENTWISE_FULL

interface ChainCase extends BenchCaseSpec {
  n: number
  len: number
  transcendental: boolean
}

const CASES: readonly ChainCase[] = SIZE_CONFIG.sizes.flatMap(n =>
  SIZE_CONFIG.chainLengths.flatMap(len => [
    { id: `chain-${len}-n${n}-plain`, n, len, transcendental: false },
    { id: `chain-${len}-n${n}-transcendental`, n, len, transcendental: true },
  ])
)

const inputs = new Map<number, AnyTensor>()
function inputFor(n: number): AnyTensor {
  let t = inputs.get(n)
  if (!t) {
    t = rand([n]) as AnyTensor
    inputs.set(n, t)
  }
  return t
}

function buildChain(x: AnyTensor, len: number, transcendental: boolean): AnyTensor {
  let h = x
  for (let i = 0; i < len; i++) {
    h = i % 2 === 0 ? h.mul(1.0001) : h.add(0.0001)
  }
  if (transcendental) h = h.tanh()
  return h
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

// §2.9's chunking/threading switches, probed rather than assumed — W0.8
// may not have landed yet, and this script must still run green.
const CHUNK_SWITCHES = ["TYPENET_BLK", "TYPENET_PARALLEL_MIN", "TYPENET_CHUNK"] as const
function declaredSwitches(): ReadonlySet<string> {
  if (!isNativeAvailable()) return new Set()
  const info = nativeDeviceInfo() as { switches?: Record<string, unknown> }
  return new Set(Object.keys(info.switches ?? {}))
}

async function main(): Promise<void> {
  const declared = declaredSwitches()
  const present = CHUNK_SWITCHES.filter(s => declared.has(s))
  if (present.length === 0) {
    console.log(
      "micro-elementwise: none of TYPENET_BLK/TYPENET_PARALLEL_MIN/TYPENET_CHUNK are declared "
        + "by the native addon yet (W0.8 not landed) — running the plain grid with no sweep.",
    )
  } else {
    // Each switch is read once behind a Rust `OnceLock` (native/src/lib.rs
    // `switches()`) and cached for the process's lifetime, so actually
    // sweeping a value needs one subprocess per value (the way
    // `micro-threading.ts` sweeps `TYPENET_THREADS`) — not a same-process
    // env mutation, which would silently no-op. That sweep is added the
    // wave that wires these switches; for now this is only the probe.
    console.log(
      `micro-elementwise: ${present.join(", ")} are declared but a same-process sweep would not `
        + `observe them (cached behind a OnceLock) — running the plain grid only.`,
    )
  }

  await bench("micro-elementwise", CASES, (kase, mode) => {
    setMode(mode)
    const x = inputFor(kase.n)
    const out = buildChain(x, kase.len, kase.transcendental)
    out.data // force materialization
  })

  configure({ lazy: false })
  disableNative()
}

await main()
