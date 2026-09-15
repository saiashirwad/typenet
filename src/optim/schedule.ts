/** A learning-rate schedule: `(step: number) => number`, with `step` 0-indexed. */
export type Schedule = (step: number) => number

/** Clamps `step` into `[0, steps]`, so every schedule holds its boundary value past its horizon. */
function clampStep(step: number, steps: number): number {
  return Math.min(Math.max(step, 0), steps)
}

/** Cosine interpolation from `from` (t=0) to `to` (t=1). */
function cosineBetween(t: number, from: number, to: number): number {
  return to + (from - to) * 0.5 * (1 + Math.cos(Math.PI * t))
}

/** Always `lr`; useful as `inner` for {@link warmup}. */
export function constant(lr: number): Schedule {
  return () => lr
}

/** Cosine anneal from `base` at step 0 down to `min` (default 0) at `steps`, then hold. */
export function cosine(
  o: { base: number; steps: number; min?: number },
): Schedule {
  const min = o.min ?? 0
  return step => cosineBetween(clampStep(step, o.steps) / o.steps, o.base, min)
}

/** Straight-line decay from `base` at step 0 to `min` (default 0) at `steps`, then hold. */
export function linearDecay(
  o: { base: number; steps: number; min?: number },
): Schedule {
  const min = o.min ?? 0
  return step => {
    const t = clampStep(step, o.steps) / o.steps
    return o.base + (min - o.base) * t
  }
}

/** Multiplies `base` by `gamma` every `every` steps. */
export function stepDecay(
  o: { base: number; every: number; gamma: number },
): Schedule {
  if (o.every <= 0) {
    throw new Error(`stepDecay: every must be positive, got ${o.every}`)
  }
  return step => o.base * o.gamma ** Math.floor(Math.max(step, 0) / o.every)
}

/** Linear ramp from 0 to `inner(0)` over the first `steps` calls, then defers to `inner(step - steps)`. */
export function warmup(inner: Schedule, steps: number): Schedule {
  if (steps <= 0) return step => inner(step)
  const target = inner(0)
  return step => step < steps ? target * (step + 1) / steps : inner(step - steps)
}

/** `warmup` fused with `cosine`: ramp 0 to `base` over `warmupSteps`, then cosine-anneal `base` to `min` (default 0) over the rest. */
export function warmupCosine(
  o: { base: number; warmupSteps: number; totalSteps: number; min?: number },
): Schedule {
  const decaySteps = Math.max(1, o.totalSteps - o.warmupSteps)
  return warmup(cosine({ base: o.base, steps: decaySteps, min: o.min }), o.warmupSteps)
}

/** The 1-cycle policy (Smith 2018): cosine-anneal up from `base / divFactor` to `base`, then down to `base / (divFactor * finalDivFactor)`. Defaults match PyTorch's `OneCycleLR`. */
export function oneCycle(
  o: {
    base: number
    steps: number
    pctStart?: number
    divFactor?: number
    finalDivFactor?: number
  },
): Schedule {
  const pctStart = o.pctStart ?? 0.3
  const divFactor = o.divFactor ?? 25
  const finalDivFactor = o.finalDivFactor ?? 1e4
  const initLr = o.base / divFactor
  const minLr = initLr / finalDivFactor
  const upSteps = Math.max(1, Math.round(o.steps * pctStart))
  const downSteps = Math.max(1, o.steps - upSteps)
  return step => {
    if (step < upSteps) {
      return cosineBetween(step / upSteps, initLr, o.base)
    }
    const t = Math.min(step - upSteps, downSteps) / downSteps
    return cosineBetween(t, o.base, minLr)
  }
}
