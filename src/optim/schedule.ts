/** A learning-rate schedule: `(step: number) => number`, with `step` 0-indexed. */
export type Schedule = (step: number) => number

/** Clamps `step` into `[0, steps]` so every schedule holds its boundary value past its horizon. */
function clampStep(step: number, steps: number): number {
  return Math.min(Math.max(step, 0), steps)
}

function cosineBetween(t: number, from: number, to: number): number {
  return to + (from - to) * 0.5 * (1 + Math.cos(Math.PI * t))
}

export function constant(lr: number): Schedule {
  return () => lr
}

/** Cosine anneal from `base` at step 0 to `min` (default 0) at `steps`, then hold. */
export function cosine(
  options: { base: number; steps: number; min?: number | undefined },
): Schedule {
  const min = options.min ?? 0
  return step => cosineBetween(clampStep(step, options.steps) / options.steps, options.base, min)
}

/** Straight-line decay from `base` at step 0 to `min` (default 0) at `steps`, then hold. */
export function linearDecay(
  options: { base: number; steps: number; min?: number },
): Schedule {
  const min = options.min ?? 0
  return step => {
    const t = clampStep(step, options.steps) / options.steps
    return options.base + (min - options.base) * t
  }
}

export function stepDecay(
  options: { base: number; every: number; gamma: number },
): Schedule {
  if (options.every <= 0) {
    throw new Error(`stepDecay: every must be positive, got ${options.every}`)
  }
  return step => options.base * options.gamma ** Math.floor(Math.max(step, 0) / options.every)
}

/** Linear ramp from 0 to `inner(0)` over the first `steps` calls, then defers to `inner(step - steps)`. */
export function warmup(inner: Schedule, steps: number): Schedule {
  if (steps <= 0) return step => inner(step)
  const target = inner(0)
  return step => step < steps ? target * (step + 1) / steps : inner(step - steps)
}

/** `warmup` fused with `cosine`: ramp 0 to `base` over `warmupSteps`, then cosine-anneal `base` to `min` (default 0) over the rest. */
export function warmupCosine(
  options: { base: number; warmupSteps: number; totalSteps: number; min?: number },
): Schedule {
  const decaySteps = Math.max(1, options.totalSteps - options.warmupSteps)
  return warmup(cosine({ base: options.base, steps: decaySteps, min: options.min }), options.warmupSteps)
}

/** The 1-cycle policy (Smith 2018): cosine-anneal up to `base`, then down to `base / (divFactor * finalDivFactor)`. Defaults match PyTorch's `OneCycleLR`. */
export function oneCycle(
  options: {
    base: number
    steps: number
    pctStart?: number
    divFactor?: number
    finalDivFactor?: number
  },
): Schedule {
  const pctStart = options.pctStart ?? 0.3
  const divFactor = options.divFactor ?? 25
  const finalDivFactor = options.finalDivFactor ?? 1e4
  const initLr = options.base / divFactor
  const minLr = initLr / finalDivFactor
  const upSteps = Math.max(1, Math.round(options.steps * pctStart))
  const downSteps = Math.max(1, options.steps - upSteps)
  return step => {
    if (step < upSteps) {
      return cosineBetween(step / upSteps, initLr, options.base)
    }
    const t = Math.min(step - upSteps, downSteps) / downSteps
    return cosineBetween(t, options.base, minLr)
  }
}
