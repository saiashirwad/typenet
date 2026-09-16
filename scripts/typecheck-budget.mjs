#!/usr/bin/env node
// Measures Instantiations and Types from `tsc -p tsconfig.budget.json --extendedDiagnostics`
// against typecheck-budget.json and fails over the limits below. --reseed rewrites the baseline.

import { execFileSync } from "node:child_process"
import { existsSync, readFileSync, writeFileSync } from "node:fs"
import { dirname, resolve } from "node:path"
import { fileURLToPath } from "node:url"

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..")
const tsconfigRelPath = "tsconfig.budget.json"
const baselinePath = resolve(root, "typecheck-budget.json")

const INSTANTIATIONS_LIMIT = 1.5
const TYPES_LIMIT = 1.4

function parseArgs(argv) {
  const args = { reseed: false, reason: undefined }
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i]
    if (arg === "--reseed") args.reseed = true
    else if (arg === "--reason") args.reason = argv[++i]
    else if (arg.startsWith("--reason=")) args.reason = arg.slice("--reason=".length)
  }
  return args
}

function fail(message, output = "") {
  console.error(`typecheck-budget: ${message}\n${output}`)
  process.exit(1)
}

function parseExtendedDiagnostics(output) {
  const grab = label => {
    const m = output.match(new RegExp(`^${label}:\\s+([\\d.]+)(?:K|s)?\\s*$`, "m"))
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

// Exits non-zero rather than returning when tsc fails or its output is unparseable.
function measureBudget() {
  const tsc = resolve(root, "node_modules", ".bin", process.platform === "win32" ? "tsc.cmd" : "tsc")
  let output
  try {
    output = execFileSync(existsSync(tsc) ? tsc : "tsc", ["-p", tsconfigRelPath, "--noEmit", "--extendedDiagnostics"], {
      cwd: root,
      encoding: "utf8",
    })
  } catch (err) {
    output = (err.stdout ?? "") + (err.stderr ?? "")
    fail(
      parseExtendedDiagnostics(output).instantiations !== undefined
        ? `the budget file set (${tsconfigRelPath}) has type errors`
        : "tsc failed and produced no parseable --extendedDiagnostics output",
      output,
    )
  }
  const parsed = parseExtendedDiagnostics(output)
  if (parsed.instantiations === undefined || parsed.types === undefined) {
    fail("could not parse tsc --extendedDiagnostics output", output)
  }
  return parsed
}

const args = parseArgs(process.argv.slice(2))

if (args.reseed) {
  if (!args.reason?.trim()) {
    fail("--reseed requires a --reason \"<why the baseline is being rewritten>\".")
  }
  const measured = measureBudget()
  const baseline = {
    tsconfig: tsconfigRelPath,
    instantiations: measured.instantiations,
    types: measured.types,
    memoryUsedKB: measured.memoryUsedKB ?? -1,
    checkTimeSeconds: measured.checkTimeSeconds ?? -1,
    totalTimeSeconds: measured.totalTimeSeconds ?? -1,
    measuredAt: new Date().toISOString(),
    reason: args.reason.trim(),
  }
  writeFileSync(baselinePath, JSON.stringify(baseline, null, 2) + "\n")
  console.log(`typecheck-budget: reseeded ${baselinePath}`)
  console.log(`  tsconfig:       ${baseline.tsconfig}`)
  console.log(`  instantiations: ${baseline.instantiations}`)
  console.log(`  types:          ${baseline.types}`)
  console.log(`  check time:     ${baseline.checkTimeSeconds}s`)
  console.log(`  total time:     ${baseline.totalTimeSeconds}s`)
  console.log(`  reason:         ${baseline.reason}`)
  process.exit(0)
}

if (!existsSync(baselinePath)) {
  fail(`no baseline at ${baselinePath}. Seed one with --reseed --reason "<why>".`)
}
const baseline = JSON.parse(readFileSync(baselinePath, "utf8"))
const measured = measureBudget()

const row = (label, current, base, note, unit = "") =>
  `  ${(label + ":").padEnd(16)}${current}${unit} vs baseline ${base}${unit}  -> ${(current / base).toFixed(3)}x ${note}`

const lines = [
  `typecheck-budget (${tsconfigRelPath}):`,
  row("instantiations", measured.instantiations, baseline.instantiations, `(limit ${INSTANTIATIONS_LIMIT}x)`),
  row("types", measured.types, baseline.types, `(limit ${TYPES_LIMIT}x)`),
]
if (measured.checkTimeSeconds !== undefined && baseline.checkTimeSeconds !== undefined) {
  lines.push(row("check time", measured.checkTimeSeconds, baseline.checkTimeSeconds, "(not gated: wall clock is noise)", "s"))
}
if (measured.totalTimeSeconds !== undefined && baseline.totalTimeSeconds !== undefined) {
  lines.push(row("total time", measured.totalTimeSeconds, baseline.totalTimeSeconds, "(not gated)", "s"))
}
console.log(lines.join("\n"))

const failures = []
const instRatio = measured.instantiations / baseline.instantiations
const typesRatio = measured.types / baseline.types
if (instRatio > INSTANTIATIONS_LIMIT) {
  failures.push(
    `instantiations ${measured.instantiations} is ${
      instRatio.toFixed(3)
    }x baseline ${baseline.instantiations}, over the ${INSTANTIATIONS_LIMIT}x budget`,
  )
}
if (typesRatio > TYPES_LIMIT) {
  failures.push(`types ${measured.types} is ${typesRatio.toFixed(3)}x baseline ${baseline.types}, over the ${TYPES_LIMIT}x budget`)
}

if (failures.length > 0) {
  console.error("\ntypecheck-budget: FAIL")
  for (const f of failures) console.error(`  - ${f}`)
  process.exit(1)
}

console.log("\ntypecheck-budget: PASS")
