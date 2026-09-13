// The `mlp-legacy` macro benchmark (PLAN-V2 §4.2, W0.2): one full
// zeroGrad + forward + mseLoss + backward + Adam.step() training step, at
// batch 64 and 512, across all three modes. §0's baseline numbers —
// eager 31.26 ms, interpreter 49.47 ms, native 1.50 ms at batch 64; native
// 2.13 ms at batch 512 — are what `pnpm bench:macro` should reproduce
// within ±10%.
//
// `mlp-legacy-*`'s shape, loss, and optimizer are frozen (`bench/models/mlp.ts`);
// this file only drives it through the three modes and times one step.

import { compile, mseLoss, Tensor } from "../index.ts"
import type { CompiledFn } from "../index.ts"
import { bench, isSmokeRun, type Mode } from "./lib/harness.ts"
import { MLP_LEGACY, MLP_SMOKE, type MlpCase } from "./lib/sizes.ts"
import { mlpLegacyData, mlpLegacyNet, mlpLegacyOptim, setMode } from "./models/mlp.ts"

type AnyTensor = Tensor<any>

interface CaseState {
  net: ReturnType<typeof mlpLegacyNet>
  x: AnyTensor
  y: AnyTensor
  optim: ReturnType<typeof mlpLegacyOptim>
  /** Only built for `interp`/`native` — `eager` runs uncompiled every step. */
  step: (CompiledFn<[AnyTensor, AnyTensor], AnyTensor> & { dispose(): void }) | null
}

const states = new Map<string, CaseState>()

function stateFor(kase: MlpCase, mode: Mode): CaseState {
  const key = `${kase.id}:${mode}`
  const existing = states.get(key)
  if (existing) return existing

  setMode(mode)
  const net = mlpLegacyNet()
  const { x, y } = mlpLegacyData(kase.batch)
  const optim = mlpLegacyOptim(net)
  const step = mode === "eager" ? null : compile((xIn: AnyTensor, yIn: AnyTensor) => {
    const loss = mseLoss(net.forward(xIn), yIn)
    optim.zeroGrad()
    loss.backward()
    optim.step()
    return loss
  })
  const state: CaseState = { net, x, y, optim, step }
  states.set(key, state)
  return state
}

function trainStep(state: CaseState): void {
  if (state.step) {
    state.step(state.x, state.y)
    return
  }
  const loss = mseLoss(state.net.forward(state.x), state.y)
  state.optim.zeroGrad()
  loss.backward()
  state.optim.step()
}

const CASES = isSmokeRun() ? MLP_SMOKE : MLP_LEGACY

await bench("macro-mlp", CASES, (kase, mode) => {
  const state = stateFor(kase, mode)
  trainStep(state)
})
