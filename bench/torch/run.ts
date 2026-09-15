// PyTorch bench driver: discovers every bench/torch/bench_*.py script, runs
// each once per available device (torch-cpu, and torch-mps when probed),
// and appends its JSONL lines to bench/results/torch-*.jsonl. Mirrors
// bench/lib/cli.ts's --only/--tag/--full flags with a local parse so the
// two drivers compose by flag shape rather than by import.

import { execFileSync } from "node:child_process"
import { appendFileSync, mkdirSync, readdirSync } from "node:fs"
import { cpus, hostname } from "node:os"
import { dirname, join } from "node:path"
import { fileURLToPath } from "node:url"

import { hasTorch, runTorch, torchPython } from "./lib/python.mjs"

type Device = "cpu" | "mps"

interface RunArgs {
  only?: string
  tag?: string
  full: boolean
}

const KNOWN_FLAGS = new Set(["--only", "--tag", "--full"])
const BOOLEAN_FLAGS = new Set(["--full"])

function parseArgs(argv: string[]): RunArgs {
  const args: RunArgs = { full: false }
  for (let i = 0; i < argv.length; i++) {
    const flag = argv[i]!
    if (!flag.startsWith("--")) {
      throw new Error(`bench/torch/run: unexpected positional argument "${flag}"`)
    }
    if (!KNOWN_FLAGS.has(flag)) {
      throw new Error(`bench/torch/run: unknown flag "${flag}"`)
    }
    if (BOOLEAN_FLAGS.has(flag)) {
      args.full = true
      continue
    }
    const value = argv[i + 1]
    if (value === undefined || value.startsWith("--")) {
      throw new Error(`bench/torch/run: flag "${flag}" requires a value`)
    }
    i++
    if (flag === "--only") args.only = value
    else if (flag === "--tag") args.tag = value
  }
  return args
}

// A torch run measures none of typenet's structural counters, so every key
// is -1. Duplicated from bench/lib/report.ts rather than imported.
const COUNTER_KEYS = [
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

function defaultCounters(): Record<string, unknown> {
  const counters: Record<string, unknown> = {}
  for (const key of COUNTER_KEYS) counters[key] = -1
  counters.matchCounts = {}
  counters.phaseNs = {}
  return counters
}

interface CaseLine {
  case: string
  n: number
  median_ms: number
  p10_ms: number
  p90_ms: number
}

function isCaseLine(x: unknown): x is CaseLine {
  return (
    typeof x === "object"
    && x !== null
    && typeof (x as CaseLine).case === "string"
    && typeof (x as CaseLine).n === "number"
    && typeof (x as CaseLine).median_ms === "number"
    && typeof (x as CaseLine).p10_ms === "number"
    && typeof (x as CaseLine).p90_ms === "number"
  )
}

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

const root = join(dirname(fileURLToPath(import.meta.url)), "..", "..")

/** Smoke and full results never share a file: smoke routes to bench/results/smoke/. */
function resultsPath(script: string, smoke: boolean): string {
  return smoke
    ? join(root, "bench", "results", "smoke", `torch-${script}.jsonl`)
    : join(root, "bench", "results", `torch-${script}.jsonl`)
}

function appendLine(script: string, device: Device, smoke: boolean, tag: string | undefined, line: CaseLine): void {
  const { git, dirty } = gitInfo()
  const full = {
    ts: new Date().toISOString(),
    host: hostname(),
    cores: cpus().length,
    git,
    dirty,
    mode: `torch-${device}`,
    script: `torch-${script}`,
    case: line.case,
    n: line.n,
    median_ms: line.median_ms,
    p10_ms: line.p10_ms,
    p90_ms: line.p90_ms,
    counters: defaultCounters(),
    smoke,
    ...(tag === undefined ? {} : { tag }),
  }
  const path = resultsPath(script, smoke)
  mkdirSync(dirname(path), { recursive: true })
  appendFileSync(path, `${JSON.stringify(full)}\n`)
}

function discoverScripts(dir: string): string[] {
  return readdirSync(dir)
    .filter(f => f.startsWith("bench_") && f.endsWith(".py"))
    .sort()
}

async function main(): Promise<void> {
  const args = parseArgs(process.argv.slice(2))
  const probe = hasTorch()

  if (!probe.ok) {
    console.log(
      `bench/torch: skipping — TORCH_PYTHON=${torchPython()} ${probe.reason}. `
        + `Set TORCH_PYTHON to a working torch interpreter (see bench/torch/README.md).`,
    )
    return
  }
  console.log(`bench/torch: using ${torchPython()} (torch ${probe.version}, mps=${probe.mps})`)

  const devices: Device[] = probe.mps ? ["cpu", "mps"] : ["cpu"]
  if (!probe.mps) {
    console.log(`bench/torch: MPS not available under this interpreter — running torch-cpu only.`)
  }

  const dir = dirname(fileURLToPath(import.meta.url))
  const scripts = discoverScripts(dir)
  if (scripts.length === 0) {
    console.log("bench/torch: no bench_*.py scripts exist yet; ran nothing.")
    return
  }

  let sawFailure = false

  for (const file of scripts) {
    const scriptName = file.replace(/^bench_/, "").replace(/\.py$/, "")
    if (args.only !== undefined && !file.includes(args.only) && !scriptName.includes(args.only)) continue

    for (const device of devices) {
      const pyArgs = ["--device", device]
      if (args.full) pyArgs.push("--full")
      if (args.only !== undefined) pyArgs.push("--only", args.only)

      const { status, stdout, stderr, error } = runTorch(join(dir, file), pyArgs)
      if (error || status !== 0) {
        console.error(
          `bench/torch/${file} --device ${device} failed (status ${status}): ${error?.message ?? stderr.trim()}`,
        )
        sawFailure = true
        continue
      }

      const rawLines = stdout.split("\n").map(l => l.trim()).filter(Boolean)
      let recorded = 0
      for (const raw of rawLines) {
        let parsed: unknown
        try {
          parsed = JSON.parse(raw)
        } catch {
          console.error(`bench/torch/${file} --device ${device}: could not parse output line: ${raw}`)
          continue
        }
        if (!isCaseLine(parsed)) {
          console.error(`bench/torch/${file} --device ${device}: malformed case line: ${raw}`)
          continue
        }
        appendLine(scriptName, device, !args.full, args.tag, parsed)
        recorded++
      }
      console.log(`bench/torch/${file} --device ${device}: recorded ${recorded} case line(s)`)
    }
  }

  if (sawFailure) process.exitCode = 1
}

main().catch(err => {
  console.error(err instanceof Error ? err.message : String(err))
  process.exitCode = 1
})
