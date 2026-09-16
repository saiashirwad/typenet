// The frozen model that test/loss-curve.ts's bit-identical loss-curve test depends on.
// Frozen: a differently-shaped or -trained variant gets its own file, not an edit to this one.

import { Adam, type AnyTensor, configure, disableNative, Linear, rand, ReLU, sequential, useNative } from "../index.ts"

/** The execution modes `setMode` selects; declared locally because this frozen model must depend on nothing outside `../index.ts`. */
type Mode = "eager" | "interp" | "native"

export function lossCurveNet() {
  return sequential(
    new Linear(784, 256),
    new ReLU(),
    new Linear(256, 10),
  )
}

export function lossCurveData(batch: number): { x: AnyTensor; y: AnyTensor } {
  return {
    x: rand([batch, 784]) as AnyTensor,
    y: rand([batch, 10]) as AnyTensor,
  }
}

export function lossCurveOptim(net: { parameters(): AnyTensor[] }): Adam {
  return new Adam(net.parameters(), { lr: 1e-3 })
}

/**
 * Point global native config at one of the three execution modes. The lazy flag stays false for every
 * mode: compile() traces under lazy semantics internally, and compiled graphs need real CPU leaves.
 */
export function setMode(mode: Mode): void {
  configure({ lazy: false })
  if (mode === "native") useNative()
  else disableNative()
}
