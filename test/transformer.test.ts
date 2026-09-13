// Whole hand-composed transformer block gradcheck (PLAN-V2 W4.10, gate
// C11): LayerNorm -> causal multi-head self-attention (via `sdpa`) ->
// residual add -> LayerNorm -> GELU MLP -> residual add, at
// B=2, T=4, D=8, H=2. Central differences (eps=1e-3, tol=1e-3 — the f32
// rationale is `gradcheck.test.ts`'s, unchanged) check every weight AND
// the input, and the analytic gradient is re-derived on every path
// typenet runs today:
//
//   - eager    — the plain JS kernels
//   - lazy     — the JS graph interpreter
//   - native   — candle, with `lower-native.ts`'s fallback half (PLAN-V2
//                §5A.9) sending softmax/layerNorm/gelu to the JS
//                interpreter automatically (they have no native kernel
//                yet); matmul/add still run through candle, so this path
//                genuinely exercises the fallback rather than assuming it
//   - compiled — `compile()`: forward + backward traced into one graph
//                once, then replayed
//
// Hand-composed on purpose: this is not `nn.TransformerBlock` (W5.2). It
// is the same arithmetic built directly out of the free functions
// (`layerNorm`, `sdpa`, `gelu`), so a sign error in one of W1.4/W4.1's
// hand-derived closed-form backward rules has nowhere to hide behind a
// layer's own tests — a finite difference is the only thing that catches
// it (Accept #2's mutation check, below, proves that it does).

import { describe, expect, it } from "vitest"
import { noGrad } from "../src/autograd.ts"
import { disableNative, isNativeAvailable, useNative } from "../src/backends/native.ts"
import { compile } from "../src/compile.ts"
import { configure } from "../src/lazy.ts"
import { sdpa } from "../src/nn/functional.ts"
import { _internal, type AnyTensor, fromFlat, gelu, layerNorm } from "../src/tensor.ts"

const EPS = 1e-3
const TOL = 1e-3
const SEED = 4242

// Fixed by the item. The MLP's hidden width is not specified there, so
// it is picked small (2x, not the usual 4x) purely to keep the ~600
// element finite-difference sweep fast — it changes nothing about what
// is being checked.
const B = 2
const T = 4
const D = 8
const H = 2
const K = D / H
const HID = 2 * D

// mulberry32 — small seeded PRNG so the sampled weights are
// deterministic and the test is never flaky (same generator as
// `gradcheck.test.ts`).
function mulberry32(seed: number): () => number {
  let a = seed >>> 0
  return () => {
    a = (a + 0x6d2b79f5) >>> 0
    let t = a
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

const PARAM_SHAPES = {
  x: [B, T, D],
  ln1g: [D],
  ln1b: [D],
  wq: [D, D],
  wk: [D, D],
  wv: [D, D],
  wo: [D, D],
  ln2g: [D],
  ln2b: [D],
  w1: [D, HID],
  b1: [HID],
  w2: [HID, D],
  b2: [D],
} as const satisfies Record<string, readonly number[]>

type Key = keyof typeof PARAM_SHAPES
const KEYS = Object.keys(PARAM_SHAPES) as Key[]

/** Deterministic starting values for every parameter, keyed the same
 * way `PARAM_SHAPES` is. LayerNorm gains start near 1 (their identity
 * value); everything else starts near 0 — both small, so GELU and
 * softmax stay away from the flat regions where a central difference
 * would be swamped by f32 noise rather than testing the block. */
function sampleAll(seed: number): Record<Key, Float32Array> {
  const out = {} as Record<Key, Float32Array>
  KEYS.forEach((k, i) => {
    const rand = mulberry32(seed + i * 97 + 1)
    const n = PARAM_SHAPES[k].reduce((a, b) => a * b, 1)
    const gain = k === "ln1g" || k === "ln2g" ? 1 : 0
    out[k] = Float32Array.from({ length: n }, () => gain + (rand() * 2 - 1) * 0.3)
  })
  return out
}

function makeParams(
  values: Record<Key, Float32Array>,
  grad: boolean,
): Record<Key, AnyTensor> {
  const out = {} as Record<Key, AnyTensor>
  for (const k of KEYS) {
    const t = fromFlat(
      values[k]!.slice(),
      PARAM_SHAPES[k] as unknown as number[],
    ) as AnyTensor
    out[k] = grad ? t.requiresGrad() : t
  }
  return out
}

/**
 * The mutation check (Accept #2): negate every gradient a node's
 * hand-derived backward reports, leaving its forward value untouched.
 * Applied to `sdpa`'s own output — a real closed-form rule (the final
 * `matmul`'s backward inside `sdpa`) turned into the deliberate sign
 * error a finite difference is supposed to catch.
 */
function flipBackward(t: AnyTensor): void {
  const node = _internal.gradNodeOf(t)
  if (!node) {
    throw new Error(
      "flipBackward: this tensor has no gradNode to mutate",
    )
  }
  _internal.setGradNode(t, {
    ...node,
    backward: g => node.backward(g).map(x => x ? x.neg() : x),
  })
}

/**
 * LayerNorm -> causal MHA (via `sdpa`) -> residual -> LayerNorm -> GELU
 * MLP -> residual. `opts.bug` runs the mutation check of Accept #2; it
 * is never set outside that one test.
 */
function block(
  p: Record<Key, AnyTensor>,
  opts: { bug?: boolean } = {},
): AnyTensor {
  const normed1 = layerNorm(p.x!, p.ln1g!, p.ln1b!)
  const q = normed1.matmul(p.wq!)
  const k = normed1.matmul(p.wk!)
  const v = normed1.matmul(p.wv!)
  // [B,T,D] -> [B,H,T,K] (q, v) or [B,H,K,T] (k, as `sdpa` expects it
  // pre-transposed — see `src/nn/functional.ts`'s `sdpa` doc comment).
  const toHeads = (t: AnyTensor): AnyTensor => t.unflatten(2, [H, K]).permute(0, 2, 1, 3)
  const qh = toHeads(q)
  const vh = toHeads(v)
  const kh = toHeads(k).permute(0, 1, 3, 2)
  const attnHeads = sdpa(qh as any, kh as any, vh as any, {
    causal: true,
  }) as AnyTensor
  if (opts.bug) flipBackward(attnHeads)
  const merged = attnHeads.permute(0, 2, 1, 3).flatten(2, 3)
  const attnOut = merged.matmul(p.wo!)
  const x2 = p.x!.add(attnOut)
  const normed2 = layerNorm(x2, p.ln2g!, p.ln2b!)
  const hidden = gelu(normed2.matmul(p.w1!).add(p.b1!))
  const mlpOut = hidden.matmul(p.w2!).add(p.b2!)
  return x2.add(mlpOut)
}

function lossOf(
  p: Record<Key, AnyTensor>,
  opts?: { bug?: boolean },
): AnyTensor {
  // `tanh` before `sum`, not `pow(2)`/`pow(3)`: a several-hundred-element
  // f32 sum over this many chained ops pushes the central difference's
  // own cancellation error above TOL, which would be a statement about
  // the test rather than about the block (same reasoning as
  // `gradcheck.test.ts`'s multi-axis `sumTo` case).
  return block(p, opts).tanh().sum() as AnyTensor
}

/** Central-difference reference gradient for every parameter, taken
 * eagerly and with grad disabled — the numeric spec every mode below is
 * checked against. */
function numericGrads(): Record<Key, Float32Array> {
  const values = sampleAll(SEED)
  const numeric = {} as Record<Key, Float32Array>
  for (const k of KEYS) {
    const n = values[k]!.length
    const g = new Float32Array(n)
    for (let j = 0; j < n; j++) {
      const original = values[k]![j]!
      values[k]![j] = original + EPS
      const up = noGrad(() => lossOf(makeParams(values, false)).item())
      values[k]![j] = original - EPS
      const down = noGrad(() => lossOf(makeParams(values, false)).item())
      values[k]![j] = original
      g[j] = (up - down) / (2 * EPS)
    }
    numeric[k] = g
  }
  return numeric
}

function extractGrads(
  p: Record<Key, AnyTensor>,
): Record<Key, Float32Array> {
  const out = {} as Record<Key, Float32Array>
  for (const k of KEYS) {
    const g = p[k]!.grad
    if (!g) {
      throw new Error(`extractGrads: ${String(k)} has no grad`)
    }
    out[k] = Float32Array.from(g.data as Float32Array)
  }
  return out
}

/** Analytic gradient under eager or lazy semantics, optionally native. */
function analyticDirect(
  mode: "eager" | "lazy" | "native",
  opts?: { bug?: boolean },
): Record<Key, Float32Array> {
  const values = sampleAll(SEED)
  const params = makeParams(values, true)
  configure({ lazy: mode !== "eager" })
  if (mode === "native") useNative()
  try {
    const loss = lossOf(params, opts)
    loss.backward()
    return extractGrads(params)
  } finally {
    disableNative()
    configure({ lazy: false })
  }
}

/** Analytic gradient through `compile()`: forward + backward traced once
 * and replayed. The gradients are returned as part of the compiled
 * function's own output tuple (`[loss, ...grads]`) rather than read back
 * off `params[k].grad` afterwards — a compiled graph's roots are exactly
 * its outputs plus its update targets, and a bare `.grad` read here would
 * be reading an unforced lazy expression from `trace()` time. */
function analyticCompiled(
  opts?: { bug?: boolean },
): Record<Key, Float32Array> {
  const values = sampleAll(SEED)
  const params = makeParams(values, true)
  const step = compile(() => {
    const loss = lossOf(params, opts)
    loss.backward()
    return [loss, ...KEYS.map(k => params[k]!.grad!)]
  })
  try {
    const out = step()
    const grads = out.slice(1)
    const result = {} as Record<Key, Float32Array>
    KEYS.forEach((k, i) => {
      result[k] = Float32Array.from(grads[i]!.data as Float32Array)
    })
    return result
  } finally {
    step.dispose()
    configure({ lazy: false })
  }
}

function maxRelError(
  analytic: Record<Key, Float32Array>,
  numeric: Record<Key, Float32Array>,
): number {
  let worst = 0
  for (const k of KEYS) {
    const a = analytic[k]!
    const n = numeric[k]!
    for (let j = 0; j < n.length; j++) {
      const diff = Math.abs(a[j]! - n[j]!)
      const scale = Math.max(1, Math.abs(a[j]!), Math.abs(n[j]!))
      worst = Math.max(worst, diff / scale)
    }
  }
  return worst
}

function expectAgreesWithNumeric(
  label: string,
  analytic: Record<Key, Float32Array>,
  numeric: Record<Key, Float32Array>,
): void {
  for (const k of KEYS) {
    const a = analytic[k]!
    const n = numeric[k]!
    for (let j = 0; j < n.length; j++) {
      const diff = Math.abs(a[j]! - n[j]!)
      const scale = Math.max(1, Math.abs(a[j]!), Math.abs(n[j]!))
      expect(
        diff / scale,
        `${label}: param ${String(k)} elem ${j}: numeric ${n[j]} vs autograd ${a[j]}`,
      ).toBeLessThan(TOL)
    }
  }
}

describe("transformer block gradcheck (gate C11)", () => {
  const numeric = numericGrads()

  it("eager matches finite differences", () => {
    expectAgreesWithNumeric("eager", analyticDirect("eager"), numeric)
  })

  it("lazy matches finite differences", () => {
    expectAgreesWithNumeric("lazy", analyticDirect("lazy"), numeric)
  })

  // Filtered rather than `.skip`ped when the addon is not built, the
  // same rule the gradcheck `describe.each` block follows.
  if (isNativeAvailable()) {
    it("native matches finite differences", () => {
      expectAgreesWithNumeric("native", analyticDirect("native"), numeric)
    })
  }

  it("compiled matches finite differences", () => {
    expectAgreesWithNumeric("compiled", analyticCompiled(), numeric)
  })

  it("a flipped sign in a hand-derived backward makes the check fail", () => {
    // Same seed, same block, the ONLY difference is `sdpa`'s backward
    // negated after the fact — proof that a real regression in a
    // closed-form rule is not silently accepted by this suite.
    const buggy = analyticDirect("eager", { bug: true })
    expect(
      maxRelError(buggy, numeric),
      "flipping sdpa's own backward sign should make at least one gradient disagree with the finite difference",
    ).toBeGreaterThanOrEqual(TOL)
  })
})
