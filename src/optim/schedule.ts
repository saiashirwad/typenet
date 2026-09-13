/**
 * Learning-rate schedules (W1.10, §3.4 of PLAN-V2). Every schedule is a
 * plain `(step: number) => number` — no coupling to `Optimizer` at all,
 * so the call site is always the same:
 *
 * ```ts
 * const sched = cosine({ base: 3e-4, steps: 1000 })
 * for (let step = 0; step < 1000; step++) {
 *   opt.lr = sched(step)
 *   ...
 * }
 * ```
 *
 * `step` is 0-indexed (the value the caller is *about to* use), matching
 * the convention every call site above already follows for `opt.step()`
 * counters elsewhere in the optimizer.
 */
export type Schedule = (step: number) => number

/** Clamp `step` into `[0, steps]` — every schedule below holds its
 * boundary value forever outside its declared horizon rather than
 * extrapolating past it (a training loop that overruns its own step
 * budget should not silently get a negative or runaway lr). */
function clampStep(step: number, steps: number): number {
  return Math.min(Math.max(step, 0), steps)
}

/** Cosine interpolation from `from` (t=0) to `to` (t=1); `t` need not be
 * clamped by the caller — the two schedules below already clamp before
 * calling it. */
function cosineBetween(t: number, from: number, to: number): number {
  return to + (from - to) * 0.5 * (1 + Math.cos(Math.PI * t))
}

/** Always `lr`, every step. The trivial schedule, useful as `inner` for
 * {@link warmup} when only the ramp-up matters. */
export function constant(lr: number): Schedule {
  return () => lr
}

/**
 * Cosine anneal from `base` at step 0 down to `min` (default 0) at
 * `steps`, then hold at `min`.
 */
export function cosine(
  o: { base: number; steps: number; min?: number },
): Schedule {
  const min = o.min ?? 0
  return step => cosineBetween(clampStep(step, o.steps) / o.steps, o.base, min)
}

/**
 * Straight-line decay from `base` at step 0 to `min` (default 0) at
 * `steps`, then hold at `min`.
 */
export function linearDecay(
  o: { base: number; steps: number; min?: number },
): Schedule {
  const min = o.min ?? 0
  return step => {
    const t = clampStep(step, o.steps) / o.steps
    return o.base + (min - o.base) * t
  }
}

/**
 * Multiplies `base` by `gamma` every `every` steps: `base * gamma **
 * floor(step / every)`. The classic "drop the lr by 10x every N epochs"
 * schedule.
 */
export function stepDecay(
  o: { base: number; every: number; gamma: number },
): Schedule {
  if (o.every <= 0) {
    throw new Error(`stepDecay: every must be positive, got ${o.every}`)
  }
  return step => o.base * o.gamma ** Math.floor(Math.max(step, 0) / o.every)
}

/**
 * Wraps `inner` with a linear ramp: for the first `steps` calls, ramps
 * from 0 up to `inner(0)` (the value `inner` would have produced at its
 * own step 0); from `steps` on, defers to `inner(step - steps)` — so
 * `inner` runs its own horizon starting the moment warmup ends, and
 * `warmupCosine`/`warmup(linearDecay(...), n)` compose for free.
 */
export function warmup(inner: Schedule, steps: number): Schedule {
  if (steps <= 0) return step => inner(step)
  const target = inner(0)
  return step => step < steps ? target * (step + 1) / steps : inner(step - steps)
}

/**
 * `warmup` fused with `cosine`: ramps 0 → `base` over `warmupSteps`, then
 * cosine-anneals `base` → `min` (default 0) over the remaining
 * `totalSteps - warmupSteps`. Equivalent to
 * `warmup(cosine({ base, steps: totalSteps - warmupSteps, min }), warmupSteps)`,
 * spelled out because it is the single most common schedule for
 * transformer training (§3.7's GPT example, §4.2's nanoGPT bench).
 */
export function warmupCosine(
  o: { base: number; warmupSteps: number; totalSteps: number; min?: number },
): Schedule {
  const decaySteps = Math.max(1, o.totalSteps - o.warmupSteps)
  return warmup(cosine({ base: o.base, steps: decaySteps, min: o.min }), o.warmupSteps)
}

/**
 * The 1-cycle policy (Smith 2018): cosine-anneal *up* from `base /
 * divFactor` to `base` over the first `pctStart` fraction of `steps`,
 * then cosine-anneal *down* from `base` to `base / (divFactor *
 * finalDivFactor)` over the rest. Defaults match PyTorch's
 * `OneCycleLR(..., anneal_strategy="cos")`.
 */
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
