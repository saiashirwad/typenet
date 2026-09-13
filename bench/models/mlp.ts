// `mlp-legacy`, frozen for the life of PLAN-V2 (§4.2, W0.2): a plain
// Linear(784,256) -> relu -> Linear(256,10) classifier trained with
// `mseLoss` and `Adam(lr 1e-3)`. Every §6.2 row and every G*.* MLP target
// refers to this exact case — the shape, the loss, and the optimizer here
// do not change. A differently-shaped or -trained baseline (the
// `mlp-modern` family, W5.4) gets its own file, not an edit to this one.
//
// Written against today's API (Linear/ReLU/sequential/mseLoss/Adam from
// the package root) on purpose: this is a baseline of what exists now,
// not a preview of what later waves build.

import { Adam, configure, disableNative, Linear, rand, ReLU, sequential, Tensor, useNative } from "../../index.ts"
import type { Mode } from "../lib/harness.ts"

type AnyTensor = Tensor<any>

/** `Linear(784,256) -> relu -> Linear(256,10)`, matching §4.2's `mlp-legacy`. */
export function mlpLegacyNet() {
  return sequential(
    new Linear(784, 256),
    new ReLU(),
    new Linear(256, 10),
  )
}

/** Random `[batch, 784]` input and `[batch, 10]` regression target. */
export function mlpLegacyData(batch: number): { x: AnyTensor; y: AnyTensor } {
  return {
    x: rand([batch, 784]) as AnyTensor,
    y: rand([batch, 10]) as AnyTensor,
  }
}

/** `Adam(lr 1e-3)` over a net's parameters, per §4.2's frozen `mlp-legacy` recipe. */
export function mlpLegacyOptim(net: { parameters(): AnyTensor[] }): Adam {
  return new Adam(net.parameters(), { lr: 1e-3 })
}

/**
 * Point global native config at one of the three bench modes (PLAN-V2
 * §4.2). The global lazy flag is deliberately left `false` for every
 * mode: `eager` needs it false so each op computes immediately, and
 * `compile()` (used for `interp`/`native`) always traces under lazy
 * semantics internally regardless of the ambient flag and never
 * consults it again on replay — only `isNativeEnabled()` decides
 * `interp` (interpreter) from `native` (native backend) at call time.
 * Leaving the ambient flag false also keeps model/optimizer
 * construction eager, so parameters and optimizer state land in real
 * CPU storage — compiled graphs require that of every leaf they mutate
 * in place (`compile.ts`'s `applyUpdate`).
 */
export function setMode(mode: Mode): void {
  configure({ lazy: false })
  if (mode === "native") useNative()
  else disableNative()
}
