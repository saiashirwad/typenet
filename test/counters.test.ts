import { execFileSync } from "node:child_process"
import { mkdtempSync, rmSync, writeFileSync } from "node:fs"
import { createRequire } from "node:module"
import { tmpdir } from "node:os"
import { join } from "node:path"
import { afterEach, describe, expect, it } from "vitest"
import { disableNative, isNativeAvailable, nativeCounters, nativeDeviceInfo, useNative } from "../src/backends/native.ts"
import { compile } from "../src/compile.ts"
import { tensor } from "../src/factories.ts"
import { configure } from "../src/lazy.ts"
import { type AnyTensor } from "../src/tensor.ts"

const available = isNativeAvailable()

// The normative key list on `counters()`: no key may be invented,
// renamed or dropped.
const NORMATIVE_KEYS = [
  "prepares",
  "indexBuilds",
  "instrs",
  "fusedRegions",
  "gemmCalls",
  "rowwiseCalls",
  "csrBuilds",
  "arenaBytes",
  "peakLiveBytes",
  "allocationsDuringRun",
  "residentSlots",
  "candleDispatches",
  "programCacheHits",
  "programCacheMisses",
  "programCacheEvictions",
  "programs",
  "matchCounts",
  "phaseNs",
  "storeSlots",
].sort()

// Counters this runtime cannot measure yet; they must read -1, never 0.
const UNMEASURED_KEYS = [
  "csrBuilds",
  "arenaBytes",
  "peakLiveBytes",
  "allocationsDuringRun",
  "residentSlots",
  "storeSlots",
]

afterEach(() => {
  configure({ lazy: false })
  disableNative()
})

describe.skipIf(!available)("native counters", () => {
  it("counters() parses as JSON with the normative keys, in stable order across two calls", () => {
    const a = nativeCounters()
    const b = nativeCounters()
    expect(Object.keys(a).sort()).toEqual(NORMATIVE_KEYS)
    // Same process, no work in between: the two calls must render with
    // identical key order (not just identical key sets).
    expect(Object.keys(a)).toEqual(Object.keys(b))
  })

  it("reports -1, never 0, for counters this runtime cannot yet measure", () => {
    const c = nativeCounters()
    for (const key of UNMEASURED_KEYS) {
      expect(c[key]).toBe(-1)
    }
  })

  it("matchCounts and phaseNs are objects, not scalars", () => {
    const c = nativeCounters()
    expect(typeof c.matchCounts).toBe("object")
    expect(typeof c.phaseNs).toBe("object")
  })

  it("deviceInfo() reports every declared switch, wired or not", () => {
    const info = nativeDeviceInfo()
    const switches = info.switches as Record<string, { value: unknown; wired: boolean }>
    expect(switches.TYPENET_NO_FUSION!.wired).toBe(true)
    expect(switches.TYPENET_PARALLEL_MIN!.wired).toBe(true)
    expect(switches.TYPENET_CHUNK!.wired).toBe(true)
    // Declared-not-wired switches still show up, honestly labeled.
    for (const name of ["TYPENET_NO_ARENA", "TYPENET_NO_PEEPHOLE", "TYPENET_NO_SIMD", "TYPENET_NO_PARALLEL", "TYPENET_THREADS", "TYPENET_TRACE"]) {
      expect(switches[name]!.wired).toBe(false)
    }
  })

  it("prepares stays 1 across 100 calls of one compiled fn", () => {
    useNative()
    const fn = compile((x: AnyTensor) => x.mul(2).sum())
    const before = nativeCounters()
    fn(tensor([1, 2, 3, 4]))
    const afterFirst = nativeCounters()
    // Exactly one native prepareGraph call for this compiled fn, however
    // many times it is invoked afterwards.
    expect((afterFirst.prepares as number) - (before.prepares as number)).toBe(1)
    for (let i = 0; i < 99; i++) fn(tensor([1, 2, 3, 4]))
    const afterLoop = nativeCounters()
    expect(afterLoop.prepares).toBe(afterFirst.prepares)
    // No new plan was ever parsed/built for the 99 replays.
    expect(afterLoop.programCacheMisses).toBe(afterFirst.programCacheMisses)
    fn.dispose()
  })

  it("structural counters are identical across iterations of a steady loop", () => {
    useNative()
    const fn = compile((x: AnyTensor) => x.mul(2).sum())
    fn(tensor([1, 2, 3, 4])) // warm up: the one prepare happens here
    // Structural / plan counters (prepares, program-size, plan-cache
    // bookkeeping) must not move once warm; that is the whole point of
    // the plan cache. Per-eval work counters (candleDispatches, gemmCalls,
    // phaseNs.eval, ...) are expected to keep accumulating and are not
    // asserted here.
    const structuralKeys = [
      "prepares",
      "instrs",
      "fusedRegions",
      "programCacheHits",
      "programCacheMisses",
      "programCacheEvictions",
      "programs",
    ]
    const snapshots = Array.from({ length: 5 }, () => {
      fn(tensor([1, 2, 3, 4]))
      return nativeCounters()
    })
    for (const key of structuralKeys) {
      const values = snapshots.map(s => s[key])
      expect(new Set(values).size).toBe(1)
    }
    fn.dispose()
  })

  // TYPENET_NO_FUSION is read once into a Rust OnceLock at first use, so
  // exercising it needs a fresh process per value. Each side spawns one and
  // measures a bandwidth-bound elementwise chain (cheap ops: neg/relu)
  // through the loop evaluator directly, bypassing the TS compile layer.
  function runFusionProbe(env: NodeJS.ProcessEnv): { minNs: number; fusedRegions: number } {
    const addonPath = createRequire(import.meta.url).resolve("@typenet/native")
    const dir = mkdtempSync(join(tmpdir(), "typenet-fusion-probe-"))
    const scriptPath = join(dir, "probe.cjs")
    try {
      writeFileSync(
        scriptPath,
        `
        const native = require(${JSON.stringify(addonPath)})
        const N = 2_000_000
        const leaf = new Float32Array(N).fill(0.3)
        const leafBytes = new Uint8Array(leaf.buffer, leaf.byteOffset, leaf.byteLength)
        // A long chain of cheap (non-transcendental) ops is bandwidth-bound:
        // fused, it is one read + one write of the array; unfused, it is
        // 120 full read+write passes. Long enough that the gap clears
        // scheduling noise on a busy machine.
        const nodes = [{ op: "leaf", leaf: 0, offset: 0, shape: [N] }]
        for (let i = 0; i < 120; i++) {
          nodes.push({
            op: "unary",
            kind: i % 2 === 0 ? "neg" : "relu",
            parameter: 0,
            input: nodes.length - 1,
            shape: [N],
          })
        }
        const graph = JSON.stringify({ nodes, roots: [nodes.length - 1], device: "loops" })
        const handle = native.prepareGraph(graph)
        native.pinLeaf(handle, 0, leafBytes)
        const dirty = new Uint8Array(0)
        const dirtyIndex = new Uint32Array(0)
        native.evalPrepared(handle, dirty, dirtyIndex, 0) // warm up
        const iters = 21
        const times = []
        for (let i = 0; i < iters; i++) {
          const start = process.hrtime.bigint()
          native.evalPrepared(handle, dirty, dirtyIndex, 0)
          times.push(Number(process.hrtime.bigint() - start))
        }
        times.sort((a, b) => a - b)
        const counters = JSON.parse(native.counters())
        process.stdout.write(JSON.stringify({
          // Best-of-N, not mean: the floor a run can hit is what isolates
          // the switch's real cost from scheduler noise on a shared machine.
          minNs: times[0],
          fusedRegions: counters.fusedRegions,
        }))
        `,
      )
      const out = execFileSync(process.execPath, [scriptPath], { env, encoding: "utf8" })
      return JSON.parse(out)
    } finally {
      rmSync(dir, { recursive: true, force: true })
    }
  }

  it(
    "TYPENET_NO_FUSION=1 is wired: it disables the fusion pass",
    () => {
      const baseEnv = { ...process.env }
      delete baseEnv.TYPENET_NO_FUSION
      const unfusedEnv = { ...baseEnv, TYPENET_NO_FUSION: "1" }
      // The structural counter proves the switch reaches plan_fusion;
      // timing is flaky on shared machines and is not asserted.
      const fused = runFusionProbe(baseEnv)
      const unfused = runFusionProbe(unfusedEnv)
      expect(fused.fusedRegions).toBe(1)
      expect(unfused.fusedRegions).toBe(0)
    },
    60_000,
  )
})
