// nanoGPT S / M / L: forward, forward+backward, and full compiled step,
// across all three modes, plus a fixed-seed 50-step loss curve a rewritten
// model must reproduce to 1e-6.
//
// `nanogpt-l` runs on demand only: `--only nanogpt` covers S and M; opt
// into L with `--only nanogpt-l` or `TYPENET_BENCH_SLOW=1`.

import { mkdirSync, writeFileSync } from "node:fs"
import { dirname, join } from "node:path"
import { fileURLToPath } from "node:url"

import { Adam, compile, configure, disableNative, Tensor, useNative } from "../index.ts"
import type { CompiledFn } from "../index.ts"
import { parseCliArgs } from "./lib/cli.ts"
import { bench, type BenchCaseSpec, isSmokeRun, type Mode } from "./lib/harness.ts"
import { NANOGPT_SIZES, NANOGPT_SMOKE, type NanoGptCase } from "./lib/sizes.ts"
import { generateBatch, type NanoGptConfig, nanoGptCrossEntropy, NanoGptLegacy } from "./models/nanogpt-legacy.ts"

type AnyTensor = Tensor<any>

// "bwd" means forward+backward. Named so no case id is a string-prefix of
// another, since `--only` matches by substring.
type Variant = "fwd" | "bwd" | "step"
const VARIANTS: readonly Variant[] = ["fwd", "bwd", "step"]

interface NanoGptBenchCase extends BenchCaseSpec {
  size: NanoGptCase
  variant: Variant
}

function toConfig(size: NanoGptCase): NanoGptConfig {
  return {
    batch: size.batch,
    blockSize: size.blockSize,
    nEmbd: size.nEmbd,
    nHead: size.nHead,
    nLayer: size.nLayer,
    vocabSize: size.vocabSize,
  }
}

/**
 * Point global native config at one of the three bench modes. The lazy
 * flag stays false for every mode: compile() traces under lazy semantics
 * internally regardless of the ambient flag, and only isNativeEnabled()
 * decides interp from native at call time.
 */
function setMode(mode: Mode): void {
  configure({ lazy: false })
  if (mode === "native") useNative()
  else disableNative()
}

interface CaseState {
  model: NanoGptLegacy
  idx: AnyTensor
  targets: number[]
  optim: Adam | null
  /** Only built for interp/native; eager runs uncompiled every call. */
  compiled: CompiledFn<[AnyTensor], AnyTensor | AnyTensor[]> | null
}

const states = new Map<string, CaseState>()

/**
 * `fwd` returns the logits; `bwd` also runs backward and returns
 * `[loss, ...grads]` so every gradient is a traced root (compile() only
 * commits a training step's updates via `Optimizer.step()`, so a bare
 * `backward()` needs its grads returned to be recomputed on replay);
 * `step` is the full-training-step shape (forward, backward, `optim.step()`
 * traced together).
 *
 * Takes its pieces individually rather than a `CaseState` so `stateFor`
 * can pass this straight into `compile()` while `state` is still being
 * built.
 */
function runVariant(
  model: NanoGptLegacy,
  variant: Variant,
  targets: number[],
  optim: Adam | null,
  x: AnyTensor,
): AnyTensor | AnyTensor[] {
  if (variant === "fwd") return model.forward(x)
  const logits = model.forward(x)
  const loss = nanoGptCrossEntropy(logits, targets)
  model.zeroGrad()
  loss.backward()
  if (variant === "step") {
    optim!.step()
    return loss
  }
  const grads = model.parameters()
    .map(p => p.grad)
    .filter((g): g is AnyTensor => g != null)
  return [loss, ...grads]
}

function stateFor(kase: NanoGptBenchCase, mode: Mode): CaseState {
  const key = `${kase.id}:${mode}`
  const existing = states.get(key)
  if (existing) return existing

  setMode(mode)
  const cfg = toConfig(kase.size)
  const model = new NanoGptLegacy(cfg)
  const { idx, targets } = generateBatch(cfg)
  const optim = kase.variant === "step" ? new Adam(model.parameters(), { lr: 3e-4 }) : null
  const compiled = mode === "eager"
    ? null
    : compile((x: AnyTensor) => runVariant(model, kase.variant, targets, optim, x), [idx])
  const state: CaseState = { model, idx, targets, optim, compiled }
  states.set(key, state)
  return state
}

function runCase(state: CaseState, kase: NanoGptBenchCase): void {
  if (state.compiled) {
    state.compiled(state.idx)
    return
  }
  runVariant(state.model, kase.variant, state.targets, state.optim, state.idx)
}

function includeSlow(): boolean {
  if (process.env.TYPENET_BENCH_SLOW === "1") return true
  const cli = parseCliArgs()
  return cli.only !== undefined && cli.only.toLowerCase().includes("nanogpt-l")
}

// `bwd` and `step` are excluded from `native`: backward through the
// attention module's batched (rank-4) matmuls, whose operands are split
// into heads via view + permute, hits MatMulUnexpectedStriding once the
// traced graph also has to serve a backward consumer, even though the
// forward-only graph runs clean on native. Re-enable once strided GEMM
// lands.
const NATIVE_UNSUPPORTED_VARIANTS: ReadonlySet<Variant> = new Set(["bwd", "step"])

function modesFor(variant: Variant): readonly Mode[] | undefined {
  return NATIVE_UNSUPPORTED_VARIANTS.has(variant) ? ["eager", "interp"] : undefined
}

function buildCases(smoke: boolean): NanoGptBenchCase[] {
  const sizes = smoke ? [NANOGPT_SMOKE] : includeSlow() ? NANOGPT_SIZES : NANOGPT_SIZES.filter(s => s.id !== "nanogpt-l")
  return sizes.flatMap(size => VARIANTS.map(variant => ({ id: `${size.id}-${variant}`, size, variant, modes: modesFor(variant) })))
}

// `nanogpt-s` exactly, seed 1337, Adam(lr 3e-4), crossEntropy on the
// synthetic batch `generateBatch` builds. Reproducible RNG draw order:
// `configure({ seed })`, then `new NanoGptLegacy(cfg)` (wte, wpe, then per
// block: attn.qkv, attn.proj, mlp.fc, mlp.proj in field order; LayerNorm
// draws nothing), then `head`, then one `generateBatch` call.
const CURVE_SIZE = NANOGPT_SIZES.find(s => s.id === "nanogpt-s")!
const CURVE_SEED = 1337
const CURVE_STEPS = 50
const CURVE_LR = 3e-4
const CURVE_PATH = join(dirname(fileURLToPath(import.meta.url)), "results", "nanogpt-legacy-curve.json")

// The 50-step curve cannot be shrunk to smoke length without changing the
// config it is pinned to, so a smoke run skips it outright; run with
// `--full` to write it.
function writeLossCurve(smoke: boolean): void {
  if (smoke) {
    console.log(
      `bench/macro-nanogpt: skipping the 50-step ${CURVE_SIZE.id} loss curve in smoke mode `
        + `(its config is pinned, not smoke-sized, and native is disabled for it) — run with --full to write it.`,
    )
    return
  }

  configure({ lazy: false, seed: CURVE_SEED })
  disableNative()
  const cfg = toConfig(CURVE_SIZE)
  const model = new NanoGptLegacy(cfg)
  const optim = new Adam(model.parameters(), { lr: CURVE_LR })
  const { idx, targets } = generateBatch(cfg)

  const losses: number[] = []
  for (let step = 0; step < CURVE_STEPS; step++) {
    const logits = model.forward(idx)
    const loss = nanoGptCrossEntropy(logits, targets)
    model.zeroGrad()
    loss.backward()
    optim.step()
    losses.push(loss.item())
  }

  mkdirSync(dirname(CURVE_PATH), { recursive: true })
  writeFileSync(
    CURVE_PATH,
    JSON.stringify(
      {
        model: "nanogpt-legacy",
        case: CURVE_SIZE.id,
        config: cfg,
        seed: CURVE_SEED,
        steps: CURVE_STEPS,
        optimizer: "Adam",
        lr: CURVE_LR,
        losses,
      },
      null,
      2,
    ),
  )
  console.log(`bench/macro-nanogpt: wrote ${CURVE_STEPS}-step loss curve to ${CURVE_PATH}`)
}

async function main(): Promise<void> {
  const smoke = isSmokeRun()
  const cases = buildCases(smoke)

  await bench("macro-nanogpt", cases, (kase, mode) => {
    const state = stateFor(kase, mode)
    runCase(state, kase)
  })

  for (const state of states.values()) state.compiled?.dispose()

  writeLossCurve(smoke)

  configure({ lazy: false })
  disableNative()
}

await main()
