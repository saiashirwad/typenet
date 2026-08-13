import type { Tensor } from "../index.ts"

/**
 * Average wall-clock ms per iteration. `warmup` iterations run first and
 * are not timed.
 */
export function timeAvg(
  fn: () => void,
  { warmup = 1, iters }: { warmup?: number; iters: number },
): number {
  for (let i = 0; i < warmup; i++) fn()
  const t0 = performance.now()
  for (let i = 0; i < iters; i++) fn()
  return (performance.now() - t0) / iters
}

/**
 * Fraction of rows whose argmax class matches the targets, as a fixed-one
 * percentage string (e.g. "83.3").
 */
export function accuracy<N extends number>(
  logits: Tensor<[N, number]>,
  targets: readonly number[],
): string {
  const pred = logits.argmax(1)
  let correct = 0
  for (let i = 0; i < targets.length; i++) {
    if (pred.data[i] === targets[i]) correct++
  }
  return ((100 * correct) / targets.length).toFixed(1)
}
