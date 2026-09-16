#!/usr/bin/env node
// Measures Instantiations and Types from `tsc -p tsconfig.budget.json --extendedDiagnostics`
// against typecheck-budget.json and fails over the limits below. --reseed rewrites the baseline.

import { execFileSync } from "node:child_process"
import { existsSync, readFileSync, writeFileSync } from "node:fs"
import { dirname, resolve } from "node:path"
import { fileURLToPath } from "node:url"

const __dirname = dirname(fileURLToPath(import.meta.url))
const root = resolve(__dirname, "..")
const tsconfigRelPath = "tsconfig.budget.json"
const baselinePath = resolve(root, "typecheck-budget.json")

const INSTANTIATIONS_LIMIT = 1.5
const TYPES_LIMIT = 1.4

function parseArgs(argv) {
  const args = { reseed: false, reason: undefined }
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i]
    if (arg === "--reseed") {
      args.reseed = true
    } else if (arg === "--reason") {
      args.reason = argv[++i]
    } else if (arg.startsWith("--reason=")) {
      args.reason = arg.slice("--reason=".length)
    }
  }
  return args
}

function tscBinary() {
  const local = resolve(root, "node_modules", ".bin", process.platform === "win32" ? "tsc.cmd" : "tsc")
  return existsSync(local) ? local : "tsc"
}

export function parseExtendedDiagnostics(output) {
  const grab = (label) => {
    const re = new RegExp(`^${label}:\\s+([\\d.]+)(?:K|s)?\\s*$`, "m")
    const m = output.match(re)
    return m ? Number(m[1]) : undefined
  }
  return {
    instantiations: grab("Instantiations"),
    types: grab("Types"),
    memoryUsedKB: grab("Memory used"),
    checkTimeSeconds: grab("Check time"),
    totalTimeSeconds: grab("Total time"),
  }
}

// Throws with the raw tsc output attached as `.output` when tsc fails.
export function measureBudget() {
  const tsc = tscBinary()
  let output
  try {
    output = execFileSync(tsc, ["-p", tsconfigRelPath, "--noEmit", "--extendedDiagnostics"], {
      cwd: root,
      encoding: "utf8",
    })
  } catch (err) {
    output = (err.stdout ?? "") + (err.stderr ?? "")
    const parsed = parseExtendedDiagnostics(output)
    const wrapped = new Error(
      parsed?.instantiations !== undefined
        ? "the budget file set (tsconfig.budget.json) has type errors"
        : "tsc failed and produced no parseable --extendedDiagnostics output",
    )
    wrapped.output = output
    throw wrapped
  }
  const parsed = parseExtendedDiagnostics(output)
  if (parsed.instantiations === undefined || parsed.types === undefined) {
    const wrapped = new Error("could not parse tsc --extendedDiagnostics output")
    wrapped.output = output
    throw wrapped
  }
  return parsed
}

function loadBaseline() {
  if (!existsSync(baselinePath)) return undefined
  return JSON.parse(readFileSync(baselinePath, "utf8"))
}

function writeBaseline(measured, reason) {
  const baseline = {
    tsconfig: tsconfigRelPath,
    instantiations: measured.instantiations,
    types: measured.types,
    memoryUsedKB: measured.memoryUsedKB ?? -1,
    checkTimeSeconds: measured.checkTimeSeconds ?? -1,
    totalTimeSeconds: measured.totalTimeSeconds ?? -1,
    measuredAt: new Date().toISOString(),
    reason,
  }
  writeFileSync(baselinePath, JSON.stringify(baseline, null, 2) + "\n")
  return baseline
}

function ratio(current, base) {
  return current / base
}

function main() {
  const args = parseArgs(process.argv.slice(2))

  if (args.reseed) {
    if (!args.reason || !args.reason.trim()) {
      console.error(
        "typecheck-budget: --reseed requires a --reason \"<why the baseline is being rewritten>\".",
      )
      process.exit(1)
    }
    let measured
    try {
      measured = measureBudget()
    } catch (err) {
      console.error(`typecheck-budget: ${err.message}\n${err.output ?? ""}`)
      process.exit(1)
    }
    const baseline = writeBaseline(measured, args.reason.trim())
    console.log(`typecheck-budget: reseeded ${baselinePath}`)
    console.log(`  tsconfig:       ${baseline.tsconfig}`)
    console.log(`  instantiations: ${baseline.instantiations}`)
    console.log(`  types:          ${baseline.types}`)
    console.log(`  check time:     ${baseline.checkTimeSeconds}s`)
    console.log(`  total time:     ${baseline.totalTimeSeconds}s`)
    console.log(`  reason:         ${baseline.reason}`)
    process.exit(0)
  }

  const baseline = loadBaseline()
  if (!baseline) {
    console.error(
      `typecheck-budget: no baseline at ${baselinePath}. Seed one with --reseed --reason "<why>".`,
    )
    process.exit(1)
  }

  let measured
  try {
    measured = measureBudget()
  } catch (err) {
    console.error(`typecheck-budget: ${err.message}\n${err.output ?? ""}`)
    process.exit(1)
  }

  const instRatio = ratio(measured.instantiations, baseline.instantiations)
  const typesRatio = ratio(measured.types, baseline.types)

  const lines = [
    `typecheck-budget (${tsconfigRelPath}):`,
    `  instantiations: ${measured.instantiations} vs baseline ${baseline.instantiations}  -> ${
      instRatio.toFixed(3)
    }x (limit ${INSTANTIATIONS_LIMIT}x)`,
    `  types:          ${measured.types} vs baseline ${baseline.types}  -> ${typesRatio.toFixed(3)}x (limit ${TYPES_LIMIT}x)`,
  ]
  if (measured.checkTimeSeconds !== undefined && baseline.checkTimeSeconds !== undefined) {
    lines.push(
      `  check time:     ${measured.checkTimeSeconds}s vs baseline ${baseline.checkTimeSeconds}s  -> ${
        ratio(measured.checkTimeSeconds, baseline.checkTimeSeconds).toFixed(3)
      }x (not gated: wall clock is noise)`,
    )
  }
  if (measured.totalTimeSeconds !== undefined && baseline.totalTimeSeconds !== undefined) {
    lines.push(
      `  total time:     ${measured.totalTimeSeconds}s vs baseline ${baseline.totalTimeSeconds}s  -> ${
        ratio(measured.totalTimeSeconds, baseline.totalTimeSeconds).toFixed(3)
      }x (not gated)`,
    )
  }
  console.log(lines.join("\n"))

  const failures = []
  if (instRatio > INSTANTIATIONS_LIMIT) {
    failures.push(
      `instantiations ${measured.instantiations} is ${
        instRatio.toFixed(3)
      }x baseline ${baseline.instantiations}, over the ${INSTANTIATIONS_LIMIT}x budget`,
    )
  }
  if (typesRatio > TYPES_LIMIT) {
    failures.push(
      `types ${measured.types} is ${typesRatio.toFixed(3)}x baseline ${baseline.types}, over the ${TYPES_LIMIT}x budget`,
    )
  }

  if (failures.length > 0) {
    console.error("\ntypecheck-budget: FAIL")
    for (const f of failures) console.error(`  - ${f}`)
    process.exit(1)
  }

  console.log("\ntypecheck-budget: PASS")
}

const isMain = process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)
if (isMain) {
  main()
}
