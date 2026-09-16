// Hand-composed transformer block, gradchecked on eager, lazy, native and compiled paths against one
// eager finite-difference reference. Built from free functions so a sign error in sdpa has nowhere to hide.

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

// The MLP hidden width is 2x rather than the usual 4x, only to keep the ~600 element
// finite-difference sweep fast. It changes nothing about what is checked.
const B = 2
const T = 4
const D = 8
const H = 2
const K = D / H
const HID = 2 * D

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

/** Deterministic starting values, keyed like PARAM_SHAPES. LayerNorm gains start near 1 and
 *  everything else near 0, so GELU and softmax stay out of the flat regions where f32 noise swamps the check. */
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

/** Negates every gradient sdpa's hand-derived backward reports, leaving its forward value
 *  untouched, so the finite difference sees the deliberate sign error it is supposed to catch. */
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

function block(
  p: Record<Key, AnyTensor>,
  opts: { bug?: boolean } = {},
): AnyTensor {
  const normed1 = layerNorm(p.x!, p.ln1g!, p.ln1b!)
  const q = normed1.matmul(p.wq!)
  const k = normed1.matmul(p.wk!)
  const v = normed1.matmul(p.wv!)
  // k is [B,H,K,T], the pre-transposed layout sdpa expects.
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
  // tanh before sum, not pow(2)/pow(3): a several-hundred-element f32 sum over this many chained
  // ops pushes the central difference's own cancellation above TOL.
  return block(p, opts).tanh().sum() as AnyTensor
}

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

/** Analytic gradient through compile(). Gradients come back in the function's own output tuple;
 *  a bare .grad read would be an unforced lazy expression from trace() time. */
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

describe("transformer block gradcheck", () => {
  const numeric = numericGrads()

  it("eager matches finite differences", () => {
    expectAgreesWithNumeric("eager", analyticDirect("eager"), numeric)
  })

  it("lazy matches finite differences", () => {
    expectAgreesWithNumeric("lazy", analyticDirect("lazy"), numeric)
  })

  // Filtered rather than .skip ped, same as gradcheck.test.ts's describe.each block.
  if (isNativeAvailable()) {
    it("native matches finite differences", () => {
      expectAgreesWithNumeric("native", analyticDirect("native"), numeric)
    })
  }

  it("compiled matches finite differences", () => {
    expectAgreesWithNumeric("compiled", analyticCompiled(), numeric)
  })

  it("a flipped sign in a hand-derived backward makes the check fail", () => {
    // Same seed and same block; only sdpa's backward is negated after the fact, so a real
    // regression in a closed-form rule cannot be silently accepted.
    const buggy = analyticDirect("eager", { bug: true })
    expect(
      maxRelError(buggy, numeric),
      "flipping sdpa's own backward sign should make at least one gradient disagree with the finite difference",
    ).toBeGreaterThanOrEqual(TOL)
  })
})
