// Appends one JSON line per (script, case, mode) to
// `bench/results/<script>.jsonl` (PLAN-V2 §4.2, §2.9, W0.1).
//
// Smoke runs (bench/README.md; the default unless `--full` is passed) write
// to `bench/results/smoke/<script>.jsonl` instead — a different directory,
// never the same file a full run appends to — and every line, smoke or
// full, carries `smoke: true | false` so a smoke number can never be
// mistaken for a real baseline even if a file is copied elsewhere.

import { execFileSync } from "node:child_process"
import { appendFileSync, mkdirSync } from "node:fs"
import { cpus, hostname } from "node:os"
import { dirname, join } from "node:path"

import type { Mode } from "./cli.ts"

// The structural-counters key list is normative (PLAN-V2 §2.9): no bench
// script or later work item may invent, rename or drop a key. A counter a
// given wave's runtime cannot yet measure reports `-1`, never `0` — W0.8
// is what starts wiring these up from the native side; until then every
// bench line carries this all-`-1` shape.
export const COUNTER_KEYS = [
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

export type CounterKey = (typeof COUNTER_KEYS)[number]

export type Counters =
  & Record<CounterKey, number>
  & {
    /** pass -> match count. */
    matchCounts: Record<string, number>
    /** phase -> nanoseconds. */
    phaseNs: Record<string, number>
  }

export function defaultCounters(): Counters {
  const counters = {} as Counters
  for (const key of COUNTER_KEYS) counters[key] = -1
  counters.matchCounts = {}
  counters.phaseNs = {}
  return counters
}

export interface BenchLine {
  ts: string
  host: string
  cores: number
  git: string
  dirty: boolean
  mode: Mode
  script: string
  case: string
  n: number
  median_ms: number
  p10_ms: number
  p90_ms: number
  counters: Counters
  /** Whether this line came from a smoke run (default) or a `--full` run. */
  smoke: boolean
  /** From `--tag`, when the run was given one. */
  tag?: string
}

export type BenchLineInput = Omit<BenchLine, "ts" | "host" | "cores" | "git" | "dirty">

let cachedGit: { git: string; dirty: boolean } | undefined

function gitInfo(): { git: string; dirty: boolean } {
  if (cachedGit) return cachedGit
  let git = "unknown"
  let dirty = false
  try {
    git = execFileSync("git", ["rev-parse", "--short", "HEAD"], { encoding: "utf8" }).trim()
  } catch {
    git = "unknown"
  }
  try {
    dirty = execFileSync("git", ["status", "--porcelain"], { encoding: "utf8" }).trim().length > 0
  } catch {
    dirty = false
  }
  cachedGit = { git, dirty }
  return cachedGit
}

/**
 * Smoke lines and full lines never share a file: smoke routes to
 * `bench/results/smoke/<script>.jsonl`, full to `bench/results/<script>.jsonl`.
 * The directory split — rather than a runtime check on an existing file's
 * content — is what makes "never in the same file" structurally true.
 */
export function resultsPath(script: string, smoke: boolean): string {
  return smoke
    ? join(process.cwd(), "bench", "results", "smoke", `${script}.jsonl`)
    : join(process.cwd(), "bench", "results", `${script}.jsonl`)
}

/**
 * Append one result line. Refuses to record — and prints a warning instead
 * — when `TYPENET_PROFILE` is set: a profiling run's timings are not a
 * throughput number and must never land in the same file as one.
 */
export function appendResult(input: BenchLineInput): void {
  if (process.env.TYPENET_PROFILE) {
    console.warn(
      `bench: TYPENET_PROFILE is set — refusing to record a throughput line for `
        + `${input.script}/${input.case}/${input.mode}`,
    )
    return
  }

  const { git, dirty } = gitInfo()
  const line: BenchLine = {
    ts: new Date().toISOString(),
    host: hostname(),
    cores: cpus().length,
    git,
    dirty,
    ...input,
  }

  const path = resultsPath(input.script, input.smoke)
  mkdirSync(dirname(path), { recursive: true })
  appendFileSync(path, `${JSON.stringify(line)}\n`)
}
