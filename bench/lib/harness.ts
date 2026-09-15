// Per-script bench runner: `bench(name, cases, fn)` runs every case under
// every available mode, prints a mode x case table, and appends one JSONL
// line per (script, case, mode). Without `--full` this is a smoke run
// (1 warm-up, 2 timed samples, results routed to bench/results/smoke/).

import { isNativeAvailable } from "../../index.ts"
import { type CliArgs, type Mode, parseCliArgs } from "./cli.ts"
import { appendResult, type Counters, defaultCounters } from "./report.ts"

const ALL_MODES: readonly Mode[] = ["eager", "interp", "native"]

export interface BenchCaseSpec {
  id: string
  /** Modes this case supports; defaults to all three. "native" is skipped automatically when the native addon is not built. */
  modes?: readonly Mode[]
}

export interface BenchStats {
  median: number
  p10: number
  p90: number
  min: number
  max: number
  n: number
}

/** A case run may report partial structural counters; unset keys default to -1. */
export type BenchOutcome = void | { counters?: Partial<Counters> }

export type BenchFn<C extends BenchCaseSpec> = (kase: C, mode: Mode) => BenchOutcome | Promise<BenchOutcome>

export interface BenchConfig {
  warmup?: number
  samples?: number
}

function percentile(sorted: readonly number[], p: number): number {
  if (sorted.length === 1) return sorted[0]!
  const idx = (sorted.length - 1) * p
  const lo = Math.floor(idx)
  const hi = Math.ceil(idx)
  if (lo === hi) return sorted[lo]!
  return sorted[lo]! + (sorted[hi]! - sorted[lo]!) * (idx - lo)
}

function computeStats(samples: readonly number[]): BenchStats {
  const sorted = [...samples].sort((a, b) => a - b)
  return {
    median: percentile(sorted, 0.5),
    p10: percentile(sorted, 0.10),
    p90: percentile(sorted, 0.90),
    min: sorted[0]!,
    max: sorted[sorted.length - 1]!,
    n: sorted.length,
  }
}

export function isSmokeRun(args: CliArgs = parseCliArgs()): boolean {
  return !args.full
}

function modesForCase(kase: BenchCaseSpec, args: CliArgs): Mode[] {
  const declared = (kase.modes ?? ALL_MODES).filter(m => m !== "native" || isNativeAvailable())
  if (args.modes.length === 0) return [...declared]
  return declared.filter(m => args.modes.includes(m))
}

function printTable(scriptName: string, table: ReadonlyMap<string, Partial<Record<Mode, number>>>): void {
  if (table.size === 0) return
  console.log(`\n${scriptName}`)
  const idWidth = Math.max(4, ...[...table.keys()].map(id => id.length))
  const header = ["case".padEnd(idWidth), ...ALL_MODES.map(m => m.padStart(10))].join(" ")
  console.log(header)
  for (const [id, byMode] of table) {
    const cells = ALL_MODES.map(m => {
      const v = byMode[m]
      return (v === undefined ? "-" : v.toFixed(3)).padStart(10)
    })
    console.log([id.padEnd(idWidth), ...cells].join(" "))
  }
}

/**
 * Run `cases` under `fn` across every available mode. With `--only <substr>`,
 * a filter matching nothing prints a line and returns without error rather
 * than throwing.
 */
export async function bench<C extends BenchCaseSpec>(
  scriptName: string,
  cases: readonly C[],
  fn: BenchFn<C>,
  config: BenchConfig = {},
): Promise<void> {
  const args = parseCliArgs()
  const smoke = isSmokeRun(args)
  // Smoke runs ignore a script's own warmup/samples config.
  const warmup = smoke ? 1 : Math.max(3, config.warmup ?? 3)
  const samples = smoke ? 2 : Math.max(10, config.samples ?? 10)

  const filtered = args.only === undefined ? cases : cases.filter(c => c.id.includes(args.only!))
  if (filtered.length === 0) {
    console.log(`bench ${scriptName}: --only "${args.only}" matched no cases; ran nothing.`)
    return
  }

  console.log(`bench ${scriptName}: ${smoke ? "SMOKE" : "FULL"} run (warmup=${warmup}, samples=${samples})`)

  const table = new Map<string, Partial<Record<Mode, number>>>()

  for (const kase of filtered) {
    const modes = modesForCase(kase, args)
    const row: Partial<Record<Mode, number>> = {}
    table.set(kase.id, row)

    for (const mode of modes) {
      for (let i = 0; i < warmup; i++) await fn(kase, mode)

      const timings: number[] = []
      let counters: Partial<Counters> | undefined
      for (let i = 0; i < samples; i++) {
        const t0 = performance.now()
        const outcome = await fn(kase, mode)
        timings.push(performance.now() - t0)
        if (outcome?.counters) counters = outcome.counters
      }

      const stats = computeStats(timings)
      row[mode] = stats.median
      appendResult({
        mode,
        script: scriptName,
        case: kase.id,
        n: stats.n,
        median_ms: stats.median,
        p10_ms: stats.p10,
        p90_ms: stats.p90,
        counters: { ...defaultCounters(), ...counters },
        smoke,
        ...(args.tag === undefined ? {} : { tag: args.tag }),
      })
    }
  }

  printTable(scriptName, table)
}

export type { CliArgs, Mode } from "./cli.ts"
export type { Counters } from "./report.ts"
