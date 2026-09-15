// CLI flags shared by every bench script (see CliArgs below), plus the
// `bench:micro` / `bench:macro` entry point: invoked directly
// (`vite-node bench/lib/cli.ts micro -- --only foo`) this file discovers
// every `bench/<suite>-*.ts` script and runs each in turn, forwarding the
// same flags. Importing `parseCliArgs` from another module never triggers
// the runner.

import { spawnSync } from "node:child_process"
import { readdirSync } from "node:fs"
import { dirname, join } from "node:path"
import { fileURLToPath } from "node:url"

export type Mode = "eager" | "interp" | "native"

export interface CliArgs {
  /** Substring filter on case ids; undefined means "run every case". */
  only?: string
  /** Repeatable `--mode` flags; empty means "every mode available for the case". */
  modes: Mode[]
  device: "cpu" | "gpu"
  tag?: string
  /** `--full` opts into real sizes/iteration counts; false (the default) means a smoke run. */
  full: boolean
}

const MODES: readonly Mode[] = ["eager", "interp", "native"]
const KNOWN_FLAGS = new Set(["--only", "--mode", "--device", "--tag", "--full"])
/** Flags that take no value; everything else in KNOWN_FLAGS requires one. */
const BOOLEAN_FLAGS = new Set(["--full"])

export function parseCliArgs(argv: string[] = process.argv.slice(2)): CliArgs {
  const args: CliArgs = { modes: [], device: "cpu", full: false }
  for (let i = 0; i < argv.length; i++) {
    const flag = argv[i]
    if (!flag.startsWith("--")) {
      throw new Error(`bench cli: unexpected positional argument "${flag}"`)
    }
    if (!KNOWN_FLAGS.has(flag)) {
      throw new Error(`bench cli: unknown flag "${flag}"`)
    }
    if (BOOLEAN_FLAGS.has(flag)) {
      if (flag === "--full") args.full = true
      continue
    }
    const value = argv[i + 1]
    if (value === undefined || value.startsWith("--")) {
      throw new Error(`bench cli: flag "${flag}" requires a value`)
    }
    i++
    switch (flag) {
      case "--only":
        args.only = value
        break
      case "--mode":
        if (!(MODES as readonly string[]).includes(value)) {
          throw new Error(`bench cli: flag "--mode" got "${value}", expected one of ${MODES.join(", ")}`)
        }
        args.modes.push(value as Mode)
        break
      case "--device":
        if (value !== "cpu" && value !== "gpu") {
          throw new Error(`bench cli: flag "--device" got "${value}", expected "cpu" or "gpu"`)
        }
        args.device = value
        break
      case "--tag":
        args.tag = value
        break
    }
  }
  return args
}

const SUITES = ["micro", "macro"] as const
type Suite = (typeof SUITES)[number]

function isSuite(x: string | undefined): x is Suite {
  return x === "micro" || x === "macro"
}

function runSuite(suite: Suite, forwardArgs: string[]): void {
  // Validate forwarded flags up front so an unknown flag fails before any
  // bench script runs.
  parseCliArgs(forwardArgs)

  const libDir = dirname(fileURLToPath(import.meta.url)) // bench/lib
  const benchDir = join(libDir, "..") // bench/
  const files = readdirSync(benchDir)
    .filter(f => f.startsWith(`${suite}-`) && f.endsWith(".ts"))
    .sort()

  if (files.length === 0) {
    console.log(`bench:${suite}: no bench/${suite}-*.ts scripts exist yet; ran nothing.`)
    return
  }

  for (const file of files) {
    const result = spawnSync(
      "vite-node",
      [join(benchDir, file), "--", ...forwardArgs],
      { stdio: "inherit" },
    )
    if (result.error) throw result.error
    if (result.status !== 0) {
      process.exitCode = result.status ?? 1
      return
    }
  }
}

function main(): void {
  const [suite, ...rest] = process.argv.slice(2)
  if (!isSuite(suite)) {
    throw new Error(`bench cli: expected first argument to be "micro" or "macro", got ${JSON.stringify(suite)}`)
  }
  runSuite(suite, rest)
}

// vite-node strips the target script's own path from process.argv, so there
// is no portable "am I the entry module" check; instead the runner fires
// when argv[2] is literally "micro" or "macro", which is exactly what the
// package.json bench scripts pass and what no bench script's own flags
// (--only, --mode, --device, --tag) can collide with.
if (isSuite(process.argv[2])) {
  try {
    main()
  } catch (err) {
    console.error(err instanceof Error ? err.message : String(err))
    process.exitCode = 1
  }
}
