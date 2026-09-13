// The four CLI flags every bench script inherits, parsed once, in one
// place (PLAN-V2 §4.2, W0.1):
//
//   --only <substr>              filter cases by substring match on case id
//   --mode <eager|interp|native> repeatable; restricts which modes run
//   --device <cpu|gpu>           (W7 reads this; cpu is the only value today)
//   --tag <string>               written into the JSONL line
//   --full                       opt into real sizes/iteration counts; a
//                                 bench script defaults to a SMOKE run
//                                 (tiny sizes, 1 warm-up, 2 timed samples)
//                                 unless this flag is present — see
//                                 bench/README.md.
//
// Unknown flags throw, naming the flag, rather than being silently
// ignored — `pnpm bench:macro -- --nonsense` must fail loudly.
//
// This file doubles as the `bench:micro` / `bench:macro` entry point:
// invoked directly (`vite-node bench/lib/cli.ts micro -- --only foo`), it
// discovers every `bench/<suite>-*.ts` script and runs each one in turn,
// forwarding the same flags. Importing `parseCliArgs` from another module
// never triggers this runner — it only fires when this file is the
// process's entry module.

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
/** Flags that take no value — everything else in KNOWN_FLAGS requires one. */
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

// --- bench:micro / bench:macro runner -------------------------------------

const SUITES = ["micro", "macro"] as const
type Suite = (typeof SUITES)[number]

function isSuite(x: string | undefined): x is Suite {
  return x === "micro" || x === "macro"
}

function runSuite(suite: Suite, forwardArgs: string[]): void {
  // Validate the forwarded flags up front so an unknown flag fails loudly
  // even before any bench script exists to surface it itself.
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

// `vite-node` does not preserve the target script's own path in
// `process.argv` (it strips it, leaving only the args that followed it), so
// there is no portable "am I the entry module" check here the way there
// would be under plain `node`. Instead: this runner only fires when the
// process's first CLI argument is literally "micro" or "macro" — the
// package.json `bench:micro` / `bench:macro` scripts always pass exactly
// that as the first argument to this file, and no bench script's own flags
// (`--only`, `--mode`, `--device`, `--tag`) start with anything but `--`, so
// importing `parseCliArgs` from this module elsewhere never collides with it.
if (isSuite(process.argv[2])) {
  try {
    main()
  } catch (err) {
    console.error(err instanceof Error ? err.message : String(err))
    process.exitCode = 1
  }
}
