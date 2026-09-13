// W0.10 — appends the typecheck-budget numbers to bench/results/typecheck.jsonl
// on every bench run: `tsc --extendedDiagnostics` instantiations/types/check-time
// over the budget file set (tsconfig.budget.json — see scripts/typecheck-budget.mjs,
// the single source of truth for measuring and parsing that output).
//
// This script is intentionally self-contained rather than routed through
// `bench/lib/*` (a different item's Files): it is not a runtime-mode bench
// (there is no "eager/interp/native" for a `tsc` invocation), so it writes
// its own JSONL line following the general shape in PLAN-V2 §4.2
// (`{host, cores, git, dirty, mode, script, case, n, median_ms, p10_ms,
// p90_ms, counters}`), with `counters` carrying the normative structural
// keys (all -1 here — a `tsc` run measures none of them, per W0.8: report
// -1, never 0) plus the typecheck-specific numbers this script exists for.

import { execFileSync } from "node:child_process"
import { appendFileSync, mkdirSync } from "node:fs"
import { cpus, hostname } from "node:os"
import { dirname, resolve } from "node:path"
import { fileURLToPath } from "node:url"

import { measureBudget } from "../scripts/typecheck-budget.mjs"

// The structural-counters key list is normative (PLAN-V2 §2.9): no script
// may invent, rename or drop a key. A `tsc` run measures none of them.
const STRUCTURAL_COUNTER_KEYS = [
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
  "storeSlots",
] as const

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..")

function gitInfo(): { git: string; dirty: boolean } {
  try {
    const git = execFileSync("git", ["rev-parse", "--short", "HEAD"], { cwd: root, encoding: "utf8" }).trim()
    const dirty = execFileSync("git", ["status", "--porcelain"], { cwd: root, encoding: "utf8" }).trim().length > 0
    return { git, dirty }
  } catch {
    return { git: "unknown", dirty: false }
  }
}

function main(): void {
  let measured: ReturnType<typeof measureBudget>
  try {
    measured = measureBudget()
  } catch (err) {
    console.error(`bench/typecheck: ${(err as Error).message}\n${(err as { output?: string }).output ?? ""}`)
    process.exitCode = 1
    return
  }

  const { git, dirty } = gitInfo()
  const median_ms = measured.checkTimeSeconds !== undefined ? measured.checkTimeSeconds * 1000 : -1

  const counters: Record<string, number> = {}
  for (const key of STRUCTURAL_COUNTER_KEYS) counters[key] = -1
  counters.instantiations = measured.instantiations ?? -1
  counters.types = measured.types ?? -1
  counters.memoryUsedKB = measured.memoryUsedKB ?? -1
  counters.checkTimeSeconds = measured.checkTimeSeconds ?? -1
  counters.totalTimeSeconds = measured.totalTimeSeconds ?? -1

  const line = {
    ts: new Date().toISOString(),
    host: hostname(),
    cores: cpus().length,
    git,
    dirty,
    mode: "typecheck",
    script: "bench/typecheck.ts",
    case: "budget",
    n: 1,
    median_ms,
    p10_ms: median_ms,
    p90_ms: median_ms,
    counters,
  }

  const resultsDir = resolve(root, "bench", "results")
  mkdirSync(resultsDir, { recursive: true })
  appendFileSync(resolve(resultsDir, "typecheck.jsonl"), `${JSON.stringify(line)}\n`)
  console.log(JSON.stringify(line, null, 2))
}

main()
