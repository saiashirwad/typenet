// `mlp-legacy` baseline: Linear(784,256) -> relu -> Linear(256,10) trained
// with mseLoss and Adam(lr 1e-3). Frozen: a differently-shaped or -trained
// baseline gets its own file, not an edit to this one.

import { Adam, configure, disableNative, Linear, rand, ReLU, sequential, Tensor, useNative } from "../../index.ts"
import type { Mode } from "../lib/harness.ts"

type AnyTensor = Tensor<any>

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

export function mlpLegacyOptim(net: { parameters(): AnyTensor[] }): Adam {
  return new Adam(net.parameters(), { lr: 1e-3 })
}

/**
 * Point global native config at one of the three bench modes. The lazy
 * flag stays false for every mode: `compile()` (used for interp/native)
 * traces under lazy semantics internally regardless of the ambient flag,
 * and eager construction keeps parameters and optimizer state in real CPU
 * storage, which compiled graphs require of every leaf they mutate in
 * place.
 */
export function setMode(mode: Mode): void {
  configure({ lazy: false })
  if (mode === "native") useNative()
  else disableNative()
}
