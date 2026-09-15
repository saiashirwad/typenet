// Appends the typecheck-budget numbers to bench/results/typecheck.jsonl on
// every bench run. Measuring and parsing live in
// scripts/typecheck-budget.mjs.

import { execFileSync } from "node:child_process"
import { appendFileSync, mkdirSync } from "node:fs"
import { cpus, hostname } from "node:os"
import { dirname, resolve } from "node:path"
import { fileURLToPath } from "node:url"

import { measureBudget } from "../scripts/typecheck-budget.mjs"

// Fixed structural-counter keys; a tsc run measures none of them.
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
