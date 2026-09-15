// Bit-identical loss-curve harness: `lossCurve()` trains the frozen
// `mlp-legacy` model for N compiled steps at a fixed seed and returns the
// per-step loss as raw f32 bits (a `Uint32Array`, never a widened
// `Float64Array`, because widening is the step that would hide a one-ulp
// difference). A `TYPENET_*` switch is read once at native-addon init, so a
// curve under a given switch set requires a fresh process; hence the
// `execFileSync` child runner. This file is both the parent-side API and,
// when invoked directly under `vite-node` with `--loss-curve-child`, the
// child runner that trains the model.

import { execFileSync } from "node:child_process"
import { existsSync } from "node:fs"
import { dirname, resolve } from "node:path"
import { fileURLToPath } from "node:url"
import { expect } from "vitest"
import { isNativeAvailable } from "../src/backends/native.ts"

const __dirname = dirname(fileURLToPath(import.meta.url))
const root = resolve(__dirname, "..")
const THIS_FILE = fileURLToPath(import.meta.url)
const CHILD_MARKER = "--loss-curve-child"
const OUTPUT_PREFIX = "LOSS_CURVE_JSON:"

/**
 * The env switches `lossCurve`'s `env` option may set. An allowlist rather
 * than a `TYPENET_` prefix check: a typo'd switch name (`TYPENET_NO_ARENAA`)
 * must fail loudly rather than silently produce a reference curve for the
 * default configuration.
 */
export const KNOWN_ENV_SWITCHES: ReadonlySet<string> = new Set([
  // per-pass kill switches
  "TYPENET_NO_FUSION",
  "TYPENET_NO_REDUCE_FUSION",
  "TYPENET_NO_MULTI_FUSION",
  "TYPENET_NO_ROWWISE",
  "TYPENET_NO_ARENA",
  "TYPENET_NO_VIEWS",
  "TYPENET_NO_STRIDED_GEMM",
  "TYPENET_NO_PEEPHOLE",
  "TYPENET_NO_SIMD",
  "TYPENET_NO_RESIDENT",
  "TYPENET_NO_ACCELERATE",
  "TYPENET_NO_PROGRAM_CACHE",
  "TYPENET_NO_PARALLEL",
  // traces
  "TYPENET_TRACE",
  "TYPENET_PROFILE",
  // budgets
  "TYPENET_THREADS",
  "TYPENET_ARENA_MB",
  "TYPENET_WORKSPACE_MB",
  "TYPENET_PARALLEL_MIN",
  "TYPENET_CHUNK",
  "TYPENET_BLK",
  // provenance
  "TYPENET_GIT_REVISION",
  "TYPENET_GIT_DIRTY",
])

/** The two modes that run a *compiled* step; `eager` never compiles, so it
 * cannot produce a loss curve here. */
export type LossCurveMode = "interp" | "native"

export interface LossCurveOptions {
  /** Number of compiled training steps to run and record. */
  steps?: number
  /** RNG seed for both weight init and the fixed training batch. */
  seed?: number
  /** Which compiled backend runs the steps. */
  mode?: LossCurveMode
  /** Extra env vars for the child process; keys must be in `KNOWN_ENV_SWITCHES`. */
  env?: Readonly<Record<string, string>>
}

/**
 * Throws unless every key of `env` is in `KNOWN_ENV_SWITCHES`. Split out
 * from `lossCurve` so its validation can be unit-tested without a
 * child-process spawn per switch name.
 */
export function assertKnownEnvSwitches(env: Readonly<Record<string, string>>): void {
  for (const key of Object.keys(env)) {
    if (!KNOWN_ENV_SWITCHES.has(key)) {
      throw new Error(
        `lossCurve(): unknown env switch "${key}" — not in the PLAN-V2 §2.9 switch set `
          + `(KNOWN_ENV_SWITCHES in test/loss-curve.ts). Refusing to silently run the `
          + `default configuration under a typo'd name; add it there if it's a real switch.`,
      )
    }
  }
}

function vitenodeBinary(): string {
  const local = resolve(root, "node_modules", ".bin", process.platform === "win32" ? "vite-node.cmd" : "vite-node")
  return existsSync(local) ? local : "vite-node"
}

/**
 * Trains `bench/models/mlp.ts`'s frozen `mlp-legacy` net for `steps`
 * compiled steps at a fixed seed and fixed batch, and returns the loss at
 * every step as raw f32 bits. Spawns a fresh `vite-node` child process so
 * `env` switches (read once at native-addon init) take effect; throws if
 * `env` names anything outside `KNOWN_ENV_SWITCHES`, or if `mode: "native"`
 * is requested but the native addon is not built.
 */
export function lossCurve(opts: LossCurveOptions = {}): Uint32Array {
  const steps = opts.steps ?? 200
  const seed = opts.seed ?? 1234
  const mode: LossCurveMode = opts.mode ?? "native"
  const env = opts.env ?? {}
  assertKnownEnvSwitches(env)
  if (mode === "native" && !isNativeAvailable()) {
    // Fail fast in the parent instead of after spawning a child that would
    // throw its own, uglier error.
    throw new Error(
      `lossCurve({ mode: "native" }): the native addon is not built — run \`pnpm build:native\` first, `
        + `or pass { mode: "interp" }.`,
    )
  }

  const args = [THIS_FILE, CHILD_MARKER, JSON.stringify({ steps, seed, mode })]
  let stdout: string
  try {
    stdout = execFileSync(vitenodeBinary(), args, {
      cwd: root,
      env: { ...process.env, ...env },
      encoding: "utf8",
      stdio: ["ignore", "pipe", "pipe"],
    })
  } catch (err) {
    const e = err as { stdout?: string; stderr?: string; message?: string }
    throw new Error(
      `lossCurve(): child process failed\n${e.stderr ?? e.message ?? String(err)}\n${e.stdout ?? ""}`,
    )
  }

  const line = stdout.split("\n").find(l => l.startsWith(OUTPUT_PREFIX))
  if (line === undefined) {
    throw new Error(`lossCurve(): child process produced no loss curve output:\n${stdout}`)
  }
  const bitsArray = JSON.parse(line.slice(OUTPUT_PREFIX.length)) as number[]
  if (bitsArray.length !== steps) {
    throw new Error(`lossCurve(): expected ${steps} loss values, child reported ${bitsArray.length}`)
  }
  return Uint32Array.from(bitsArray)
}

/**
 * Asserts two loss curves are bit-for-bit identical (`toBe`, never
 * `toBeCloseTo`: any tolerance would hide the one-ulp divergence this
 * harness exists to catch).
 */
export function expectIdenticalCurves(a: Uint32Array, b: Uint32Array, label: string): void {
  expect(a.length, `${label}: curve lengths differ`).toBe(b.length)
  for (let i = 0; i < a.length; i++) {
    expect(a[i], `${label}: loss bits differ at step ${i} (${a[i]} vs ${b[i]})`).toBe(b[i])
  }
}

// Child runner: only reached when this file is executed directly by
// `vite-node` with the `--loss-curve-child` marker, never when imported
// by a test, so importing this module never trains anything.

async function runChild(argJson: string): Promise<void> {
  const { steps, seed, mode } = JSON.parse(argJson) as { steps: number; seed: number; mode: LossCurveMode }

  const { compile } = await import("../src/compile.ts")
  const { configure } = await import("../src/lazy.ts")
  const { mseLoss } = await import("../src/nn.ts")
  const { mlpLegacyData, mlpLegacyNet, mlpLegacyOptim, setMode } = await import("../bench/models/mlp.ts")
  const { Tensor } = await import("../src/tensor.ts")
  type AnyTensor = InstanceType<typeof Tensor>

  // Seed before any rand() draw: weight init (inside mlpLegacyNet) and the
  // training batch (mlpLegacyData) consume the seeded generator in a fixed
  // order, so one up-front seed reproduces both.
  configure({ seed })
  setMode(mode)

  const net = mlpLegacyNet()
  const { x, y } = mlpLegacyData(64)
  const optim = mlpLegacyOptim(net)

  const step = compile((xIn: AnyTensor, yIn: AnyTensor) => {
    const loss = mseLoss(net.forward(xIn), yIn)
    optim.zeroGrad()
    loss.backward()
    optim.step()
    return loss
  })

  const bits = new Array<number>(steps)
  const f32 = new Float32Array(1)
  const asU32 = new Uint32Array(f32.buffer)
  for (let i = 0; i < steps; i++) {
    const loss = step(x, y)
    f32[0] = loss.item()
    bits[i] = asU32[0]!
  }
  step.dispose()

  process.stdout.write(`${OUTPUT_PREFIX}${JSON.stringify(bits)}\n`)
}

const markerIndex = process.argv.indexOf(CHILD_MARKER)
if (markerIndex !== -1) {
  runChild(process.argv[markerIndex + 1]!).catch(err => {
    console.error(err)
    process.exitCode = 1
  })
}
