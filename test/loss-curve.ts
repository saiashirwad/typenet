// Bit-identical loss-curve check. lossCurve() trains the frozen model in test/loss-curve-model.ts for N
// compiled steps and returns raw f32 bits; a switch set needs a fresh process, hence the child runner.

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
 * The env switches the addon and the runtime actually read. An allowlist rather than a TYPENET_ prefix
 * check, so a typo'd switch name fails loudly instead of running the default configuration.
 */
export const KNOWN_ENV_SWITCHES: ReadonlySet<string> = new Set([
  "TYPENET_NO_FUSION",
  "TYPENET_PARALLEL_MIN",
  "TYPENET_CHUNK",
  "TYPENET_PROFILE",
  "TYPENET_CHECK_SHAPES",
  "TYPENET_EVALUATOR",
  "TYPENET_STRICT_NATIVE",
])

/** The two modes that run a compiled step. Eager never compiles, so it cannot produce a loss curve. */
export type LossCurveMode = "interp" | "native"

export interface LossCurveOptions {
  steps?: number
  seed?: number
  mode?: LossCurveMode
  /** Extra env vars for the child process; keys must be in `KNOWN_ENV_SWITCHES`. */
  env?: Readonly<Record<string, string>>
}

/**
 * Throws unless every key of env is in KNOWN_ENV_SWITCHES. Split out of lossCurve so its validation
 * can be unit-tested without a child-process spawn per switch name.
 */
export function assertKnownEnvSwitches(env: Readonly<Record<string, string>>): void {
  for (const key of Object.keys(env)) {
    if (!KNOWN_ENV_SWITCHES.has(key)) {
      throw new Error(
        `lossCurve(): unknown env switch "${key}", not in the KNOWN_ENV_SWITCHES set `
          + `(test/loss-curve.ts). Refusing to silently run the `
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
 * Trains the frozen net for steps compiled steps at a fixed seed and batch, and returns the loss at every
 * step as raw f32 bits. Spawns a fresh vite-node child so env switches (read once at addon init) take effect.
 */
export function lossCurve(opts: LossCurveOptions = {}): Uint32Array {
  const steps = opts.steps ?? 200
  const seed = opts.seed ?? 1234
  const mode: LossCurveMode = opts.mode ?? "native"
  const env = opts.env ?? {}
  assertKnownEnvSwitches(env)
  if (mode === "native" && !isNativeAvailable()) {
    // Fail fast in the parent instead of after spawning a child that would throw its own uglier error.
    throw new Error(
      `lossCurve({ mode: "native" }): the native addon is not built, run \`pnpm build:native\` first, `
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
 * Asserts two loss curves are bit-for-bit identical (toBe, never toBeCloseTo: any tolerance would
 * hide the one-ulp divergence this check exists to catch).
 */
export function expectIdenticalCurves(a: Uint32Array, b: Uint32Array, label: string): void {
  expect(a.length, `${label}: curve lengths differ`).toBe(b.length)
  for (let i = 0; i < a.length; i++) {
    expect(a[i], `${label}: loss bits differ at step ${i} (${a[i]} vs ${b[i]})`).toBe(b[i])
  }
}

// Child runner: reached only when this file is executed directly with --loss-curve-child, never when
// imported by a test, so importing this module never trains anything.

async function runChild(argJson: string): Promise<void> {
  const { steps, seed, mode } = JSON.parse(argJson) as { steps: number; seed: number; mode: LossCurveMode }

  const { compile } = await import("../src/compile.ts")
  const { configure } = await import("../src/lazy.ts")
  const { mseLoss } = await import("../src/nn/index.ts")
  const { lossCurveData, lossCurveNet, lossCurveOptim, setMode } = await import("./loss-curve-model.ts")
  const { Tensor } = await import("../src/tensor.ts")
  type AnyTensor = InstanceType<typeof Tensor>

  // Seed before any rand() draw: weight init and the training batch consume the seeded generator
  // in a fixed order, so one up-front seed reproduces both.
  configure({ seed })
  setMode(mode)

  const net = lossCurveNet()
  const { x, y } = lossCurveData(64)
  const optim = lossCurveOptim(net)

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
