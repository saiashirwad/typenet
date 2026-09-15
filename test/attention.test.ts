// MultiHeadAttention, TransformerBlock, sdpa, ModuleList, Residual.
//
// The whole-block gradcheck lives at the bottom of this file: it gradchecks
// against an eager central-difference reference on all four paths (eager /
// lazy / native / compiled). "Native" here means the graph is handed to
// `serializeLazyGraph`, which returns `null` for any graph containing
// `layerNorm`/`softmax`/`gelu`/`dropout` (no lowering yet) and routes the
// whole thing to the JS interpreter. That is a real, distinct code path,
// asserted as a fallback through `jsCounters()`, not quietly run twice and
// called two paths.

import { afterEach, describe, expect, it } from "vitest"
import { noGrad } from "../src/autograd.ts"
import { disableNative, isNativeAvailable, nativeCounters, useNative } from "../src/backends/native.ts"
import { compile } from "../src/compile.ts"
import { jsCounters, resetJsCounters } from "../src/counters.ts"
import { topoOrder } from "../src/ir.ts"
import { configure } from "../src/lazy.ts"
import { functional, LayerNorm, Linear, Module, ModuleList, MultiHeadAttention, Residual, TransformerBlock } from "../src/nn/index.ts"
import { _internal, type AnyTensor, Tensor } from "../src/tensor.ts"
import { expectAgreeStrict } from "./helpers.ts"

const { sdpa } = functional

afterEach(() => {
  configure({ lazy: false })
  disableNative()
})

// mulberry32: the same small seeded PRNG the rest of the suite uses, so
// every tensor in this file is reproducible and nothing here is flaky.
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

/** Sampled away from zero, so no gradient sits on a kink or a cancellation. */
function awayFromZero(rand: () => number): number {
  return (rand() > 0.5 ? 1 : -1) * (0.3 + rand() * 0.6)
}

function filled(shape: number[], seed: number, grad = false): AnyTensor {
  const rand = mulberry32(seed)
  const t = Tensor.zeros(shape) as AnyTensor
  const d = t.data as Float32Array
  for (let i = 0; i < d.length; i++) d[i] = awayFromZero(rand)
  return grad ? (t.requiresGrad() as AnyTensor) : t
}

/** Overwrite every parameter of `m` with reproducible, away-from-zero values. */
function seedParameters(m: Module, seed: number): void {
  const rand = mulberry32(seed)
  for (const p of m.parameters()) {
    const d = (p as AnyTensor).data as Float32Array
    for (let i = 0; i < d.length; i++) d[i] = awayFromZero(rand)
  }
}

/** The lazy node ops and the leaf shapes of a graph, for structural assertions. */
function graphShape(root: AnyTensor): {
  ops: string[]
  leafShapes: number[][]
} {
  const ops: string[] = []
  const leafShapes: number[][] = []
  for (const t of topoOrder([root])) {
    const source = _internal.sourceOf(t)
    if (source.kind === "lazy" && !_internal.hasValue(t)) ops.push(source.node.op)
    else leafShapes.push([...t.shape])
  }
  return { ops, leafShapes }
}

function count(xs: readonly string[], op: string): number {
  return xs.filter(x => x === op).length
}

// An independent reference: plain JS loops over the layer's own weights,
// with no typenet op in sight. Comparing `MultiHeadAttention` against a
// composition of typenet ops would only prove the layer calls the ops it
// calls; this proves it computes attention.
function referenceAttention(
  x: Float32Array,
  dims: { B: number; T: number; D: number; H: number },
  wqkv: Float32Array,
  bqkv: Float32Array | null,
  wo: Float32Array,
  bo: Float32Array | null,
  causal: boolean,
): Float32Array {
  const { B, T, D, H } = dims
  const dh = D / H
  const qkv = new Float64Array(B * T * 3 * D)
  for (let b = 0; b < B; b++) {
    for (let t = 0; t < T; t++) {
      for (let o = 0; o < 3 * D; o++) {
        let s = bqkv ? bqkv[o]! : 0
        for (let i = 0; i < D; i++) s += x[(b * T + t) * D + i]! * wqkv[i * 3 * D + o]!
        qkv[(b * T + t) * 3 * D + o] = s
      }
    }
  }
  const merged = new Float64Array(B * T * D)
  const scale = 1 / Math.sqrt(dh)
  for (let b = 0; b < B; b++) {
    for (let head = 0; head < H; head++) {
      for (let t = 0; t < T; t++) {
        const limit = causal ? t + 1 : T
        const w = new Float64Array(T)
        let m = -Infinity
        for (let u = 0; u < limit; u++) {
          let s = 0
          for (let j = 0; j < dh; j++) {
            s += qkv[(b * T + t) * 3 * D + head * dh + j]!
              * qkv[(b * T + u) * 3 * D + D + head * dh + j]!
          }
          w[u] = s * scale
          if (w[u]! > m) m = w[u]!
        }
        let sum = 0
        for (let u = 0; u < limit; u++) {
          w[u] = Math.exp(w[u]! - m)
          sum += w[u]!
        }
        for (let j = 0; j < dh; j++) {
          let acc = 0
          for (let u = 0; u < limit; u++) {
            acc += (w[u]! / sum) * qkv[(b * T + u) * 3 * D + 2 * D + head * dh + j]!
          }
          merged[(b * T + t) * D + head * dh + j] = acc
        }
      }
    }
  }
  const out = new Float32Array(B * T * D)
  for (let b = 0; b < B; b++) {
    for (let t = 0; t < T; t++) {
      for (let o = 0; o < D; o++) {
        let s = bo ? bo[o]! : 0
        for (let i = 0; i < D; i++) s += merged[(b * T + t) * D + i]! * wo[i * D + o]!
        out[(b * T + t) * D + o] = s
      }
    }
  }
  return out
}

describe("MultiHeadAttention", () => {
  it("derives the head width and stores the BARE h, not the check intersection", () => {
    const mha = new MultiHeadAttention(384, 6)
    expect(mha.d).toBe(384)
    expect(mha.h).toBe(6)
    expect(mha.headDim).toBe(64)
    // The property carries a plain number at run time; the type-level half
    // (`h: H`, never `H & DimDivCheck<D, H>`) is asserted in
    // test/types.test-d.ts, where a poisoned property type would show up
    // as a downstream error rather than as a wrong value here.
    expect(typeof mha.h).toBe("number")
  })

  it("is the runtime twin of DimDivCheck: 5 heads do not divide 384", () => {
    expect(() => new MultiHeadAttention(384, 5 as unknown as 6)).toThrow(
      /5 heads do not divide a model width of 384/,
    )
  })

  it("names both widths when the forward is handed the wrong model width", () => {
    const mha = new MultiHeadAttention(8, 2)
    expect(() => mha.forward(filled([2, 3, 6], 1) as never)).toThrow(
      /expects a model width of 8, got \[2, 3, 6\]/,
    )
    expect(() => mha.forward(filled([2, 8], 1) as never)).toThrow(/rank-2 tensor/)
  })

  it.each([
    ["non-causal", false],
    ["causal", true],
  ])("matches a plain-JS reference (%s)", (_label, causal) => {
    const dims = { B: 2, T: 4, D: 6, H: 3 }
    const mha = new MultiHeadAttention(dims.D, dims.H, { causal })
    seedParameters(mha, 99)
    const x = filled([dims.B, dims.T, dims.D], 7)
    const y = mha.forward(x as never) as AnyTensor
    const want = referenceAttention(
      x.data as Float32Array,
      dims,
      mha.qkv!.weight.data as Float32Array,
      mha.qkv!.bias!.data as Float32Array,
      mha.proj.weight.data as Float32Array,
      mha.proj.bias!.data as Float32Array,
      causal,
    )
    expect(y.shape).toEqual([dims.B, dims.T, dims.D])
    const got = y.data as Float32Array
    for (let i = 0; i < want.length; i++) {
      expect(Math.abs(got[i]! - want[i]!), `element ${i}`).toBeLessThan(1e-5)
    }
  })

  it("causal attention cannot see the future, and non-causal can", () => {
    const dims = { B: 1, T: 5, D: 8, H: 2 }
    const run = (causal: boolean): { before: Float32Array; after: Float32Array } => {
      const mha = new MultiHeadAttention(dims.D, dims.H, { causal })
      seedParameters(mha, 4242)
      const x = filled([dims.B, dims.T, dims.D], 11)
      const before = Float32Array.from((mha.forward(x as never) as AnyTensor).data as Float32Array)
      // Rewrite the LAST token only.
      const d = x.data as Float32Array
      for (let i = (dims.T - 1) * dims.D; i < dims.T * dims.D; i++) d[i] = -d[i]! * 3 + 0.7
      const after = Float32Array.from((mha.forward(x as never) as AnyTensor).data as Float32Array)
      return { before, after }
    }

    const causal = run(true)
    // Every position before the last is bit-identical: `toBe`, not a
    // tolerance, because a causal mask that leaks is not a rounding
    // difference.
    for (let i = 0; i < (dims.T - 1) * dims.D; i++) {
      expect(causal.after[i], `causal position ${Math.floor(i / dims.D)}`).toBe(causal.before[i])
    }
    // ...and the last position did move, so the test is not vacuous.
    let moved = false
    for (let i = (dims.T - 1) * dims.D; i < dims.T * dims.D; i++) {
      if (causal.after[i] !== causal.before[i]) moved = true
    }
    expect(moved, "the last position must respond to its own input").toBe(true)

    const open = run(false)
    let leaked = false
    for (let i = 0; i < (dims.T - 1) * dims.D; i++) {
      if (open.after[i] !== open.before[i]) leaked = true
    }
    expect(leaked, "without causal masking every position sees the change").toBe(true)
  })

  it("fused and unfused qkv compute the same thing from the same weights", () => {
    const D = 6
    const H = 3
    const fused = new MultiHeadAttention(D, H, { causal: true })
    const split = new MultiHeadAttention(D, H, { causal: true, qkvFused: false })
    seedParameters(fused, 5)
    // Copy the fused `[D, 3D]` block into the three `[D, D]` projections
    // column-wise, which is exactly the cut `narrow(2, k*D, D)` makes.
    const w = fused.qkv!.weight.data as Float32Array
    const b = fused.qkv!.bias!.data as Float32Array
    const parts = [split.wq!, split.wk!, split.wv!]
    parts.forEach((lin, k) => {
      const lw = lin.weight.data as Float32Array
      const lb = lin.bias!.data as Float32Array
      for (let i = 0; i < D; i++) {
        for (let o = 0; o < D; o++) lw[i * D + o] = w[i * 3 * D + k * D + o]!
      }
      for (let o = 0; o < D; o++) lb[o] = b[k * D + o]!
    })
    ;(split.proj.weight.data as Float32Array).set(fused.proj.weight.data as Float32Array)
    ;(split.proj.bias!.data as Float32Array).set(fused.proj.bias!.data as Float32Array)

    const x = filled([2, 4, D], 13)
    const a = (fused.forward(x as never) as AnyTensor).data as Float32Array
    const c = (split.forward(x as never) as AnyTensor).data as Float32Array
    for (let i = 0; i < a.length; i++) {
      expect(Math.abs(a[i]! - c[i]!), `element ${i}`).toBeLessThan(1e-5)
    }
  })

  it("eager, lazy and native agree on the attention forward", () => {
    const mha = new MultiHeadAttention(8, 2, { causal: true })
    seedParameters(mha, 21)
    const x = filled([2, 5, 8], 31)
    expectAgreeStrict(() => mha.forward(x as never) as AnyTensor)
  })

  it("dropout is applied in train() and is gone in eval()", () => {
    const mha = new MultiHeadAttention(8, 2, { causal: true, dropout: 0.5 })
    seedParameters(mha, 6)
    const x = filled([1, 4, 8], 8)
    configure({ lazy: true })
    const train = graphShape(mha.forward(x as never) as AnyTensor)
    expect(count(train.ops, "dropout")).toBe(1)
    mha.eval()
    const evalOps = graphShape(mha.forward(x as never) as AnyTensor)
    // Not "a dropout node that happens to be the identity": no node at all.
    expect(count(evalOps.ops, "dropout")).toBe(0)
  })
})

// sdpa, the rank-4 escape hatch
describe("sdpa", () => {
  it("is softmax(q@k / sqrt(dh)) @ v", () => {
    const B = 2, H = 2, T = 3, K = 4
    const q = filled([B, H, T, K], 1)
    const kT = filled([B, H, K, T], 2)
    const v = filled([B, H, T, K], 3)
    const got = sdpa(q as never, kT as never, v as never) as AnyTensor
    // The fused `softmax` NODE, not `Tensor.prototype.softmax`'s composed
    // max/sub/exp/sum/div spelling: `sdpa` emits the node, and comparing
    // it against a different arithmetic would only be a tolerance check.
    const want = (functional.softmax(
      (q.matmul(kT as never) as AnyTensor).mul(1 / Math.sqrt(K)) as never,
      -1,
    ) as AnyTensor).matmul(v as never) as AnyTensor
    expect(got.shape).toEqual([B, H, T, K])
    const a = got.data as Float32Array
    const b = want.data as Float32Array
    for (let i = 0; i < a.length; i++) expect(a[i]).toBe(b[i])
  })

  it("causal: the first query attends to the first key alone, exactly", () => {
    const B = 1, H = 1, T = 4, K = 3
    const q = filled([B, H, T, K], 17)
    const kT = filled([B, H, K, T], 19)
    const v = filled([B, H, T, K], 23)
    const got = (sdpa(q as never, kT as never, v as never, { causal: true }) as AnyTensor).data as Float32Array
    const vd = v.data as Float32Array
    // Row 0's only unmasked weight is 1.0, so the context is v's first
    // row bit for bit: `exp(-Infinity)` is exactly 0 and the row
    // normalises to a single 1. No `-1e9` fudge could make this exact.
    for (let j = 0; j < K; j++) expect(got[j]).toBe(vd[j])
  })

  it("rejects operands that are not rank 4, naming all three shapes", () => {
    const q = filled([2, 3, 4], 1)
    expect(() => sdpa(q as never, q as never, q as never)).toThrow(
      /sdpa: expects rank-4 .* got q \[2, 3, 4\]/,
    )
  })

  it("emits no dropout node at p = 0 and one at p > 0", () => {
    const q = filled([1, 1, 3, 2], 1)
    const kT = filled([1, 1, 2, 3], 2)
    const v = filled([1, 1, 3, 2], 3)
    configure({ lazy: true })
    expect(count(graphShape(sdpa(q as never, kT as never, v as never) as AnyTensor).ops, "dropout")).toBe(0)
    expect(
      count(graphShape(sdpa(q as never, kT as never, v as never, { dropout: 0.25 }) as AnyTensor).ops, "dropout"),
    ).toBe(1)
  })
})

describe("the attention graph itself", () => {
  const build = (T: number): AnyTensor => {
    const mha = new MultiHeadAttention(8, 2, { causal: true })
    seedParameters(mha, 3)
    return mha.forward(filled([2, T, 8], T) as never) as AnyTensor
  }

  it("carries the causal mask on the softmax node and allocates no [T, T] buffer", () => {
    configure({ lazy: true })
    const T = 7
    const { ops, leafShapes } = graphShape(build(T))
    expect(count(ops, "softmax")).toBe(1)
    // Nothing in the graph is `[T, T]`-shaped, and nothing is `T`-sized at
    // all beyond the activations: every leaf is a weight, the input, or
    // the `1/sqrt(dh)` scalar.
    for (const shape of leafShapes) {
      expect(shape, `leaf of shape [${shape.join(", ")}] — a materialised mask?`).not.toEqual([T, T])
    }
    // Every leaf, enumerated: the input, the fused `[D, 3D]` projection
    // and its bias, the output projection and its bias, and the
    // `1/sqrt(dh)` scalar. Six, and not one of them depends on T.
    const show = (ss: number[][]): string[] => ss.map(s2 => `[${s2.join(", ")}]`).sort()
    expect(show(leafShapes)).toEqual(
      show([[2, T, 8], [8, 24], [24], [8, 8], [8], []]),
    )
  })

  it("costs exactly four permutes: three head splits and one merge", () => {
    configure({ lazy: true })
    const { ops } = graphShape(build(5))
    // Four, not five: `sdpa` takes `k` ALREADY transposed, so the score
    // matmul adds no permute of its own on top of the three the head
    // split pays for and the one that merges the heads back.
    expect(count(ops, "permute")).toBe(4)
    expect(count(ops, "matmul")).toBe(4) // qkv, scores, context, out proj
  })

  it("the graph is structurally identical at every sequence length", () => {
    configure({ lazy: true })
    const a = graphShape(build(4))
    const b = graphShape(build(64))
    expect(b.ops).toEqual(a.ops)
    expect(b.leafShapes.length).toBe(a.leafShapes.length)
  })
})

const DIMS = { B: 1, T: 3, D: 4, H: 2 }

function makeBlock(seed = 7): TransformerBlock<4, 2> {
  const blk = new TransformerBlock(DIMS.D as 4, DIMS.H as 2, { causal: true, dropout: 0 })
  seedParameters(blk, seed)
  return blk
}

describe("TransformerBlock: the attention block", () => {
  it("is pre-norm with a 4x GELU MLP and keeps its shape", () => {
    const blk = new TransformerBlock(16, 4)
    expect(blk.fc.outFeatures).toBe(64)
    expect(blk.proj.inFeatures).toBe(64)
    expect(blk.attn.headDim).toBe(4)
    const y = blk.forward(filled([2, 5, 16], 2) as never) as AnyTensor
    expect(y.shape).toEqual([2, 5, 16])
  })

  it("reports every parameter once, by dotted path, and round-trips a stateDict", () => {
    const blk = makeBlock()
    const names = [...blk.namedParameters().keys()]
    expect(names).toContain("attn.qkv.weight")
    expect(names).toContain("ln2.gamma")
    expect(names).toContain("fc.bias")
    expect(new Set(names).size).toBe(names.length)

    const snapshot = blk.stateDict()
    const fresh = new TransformerBlock(DIMS.D as 4, DIMS.H as 2, { causal: true, dropout: 0 })
    fresh.loadStateDict(snapshot)
    const x = filled([DIMS.B, DIMS.T, DIMS.D], 55)
    const a = (blk.forward(x as never) as AnyTensor).data as Float32Array
    const b = (fresh.forward(x as never) as AnyTensor).data as Float32Array
    for (let i = 0; i < a.length; i++) expect(b[i]).toBe(a[i])
  })

  it("eager, lazy and native agree on the whole block", () => {
    const blk = makeBlock(12)
    const x = filled([DIMS.B, DIMS.T, DIMS.D], 13)
    expectAgreeStrict(() => blk.forward(x as never) as AnyTensor)
  })

  it("falls back to the JS interpreter, loudly and countably, under the native backend", () => {
    // A block containing layerNorm/softmax/gelu/dropout cannot reach the
    // addon, and it must say so: a silent interpreter fallback on a
    // transformer is the 30x regression `jsCounters()` exists to catch.
    if (!isNativeAvailable()) return
    const blk = makeBlock(3)
    const x = filled([DIMS.B, DIMS.T, DIMS.D], 4)
    resetJsCounters()
    configure({ lazy: true })
    useNative()
    ;(blk.forward(x as never) as AnyTensor).data
    const c = jsCounters()
    expect(c.nativeFallbacks).toBeGreaterThan(0)
    // Whichever semantic op the topological order reaches first is the one
    // that reports; every one of them is a real reason.
    expect(Object.keys(c.fallbacksByOp).length).toBeGreaterThan(0)
    for (const op of Object.keys(c.fallbacksByOp)) {
      expect(["layerNorm", "softmax", "gelu", "dropout", "pick"]).toContain(op)
    }
  })
})

const EPS = 1e-3
const TOL = 3e-3

/** `tanh().sum()` rather than `.sum()`: a bare sum of a residual stream is
 *  nearly linear in every weight, which makes the central difference's own
 *  cancellation, not the gradient, the thing under test. */
function blockLoss(blk: TransformerBlock<4, 2>, x: AnyTensor): AnyTensor {
  return (blk.forward(x as never) as AnyTensor).tanh().sum() as AnyTensor
}

/**
 * Central differences over every scalar of `x` and every parameter,
 * computed EAGERLY once. Every mode is then compared against this one
 * reference: comparing a mode's numeric noise against its own analytic
 * gradient would hide exactly the regressions this gate is for.
 */
function numericGradients(blk: TransformerBlock<4, 2>, x: AnyTensor): Float32Array[] {
  const targets = [x, ...blk.parameters().map(p => p as AnyTensor)]
  configure({ lazy: false })
  return targets.map(t => {
    const buf = t.data as Float32Array
    const out = new Float32Array(buf.length)
    for (let j = 0; j < buf.length; j++) {
      const original = buf[j]!
      buf[j] = original + EPS
      const up = noGrad(() => blockLoss(blk, x).item())
      buf[j] = original - EPS
      const down = noGrad(() => blockLoss(blk, x).item())
      buf[j] = original
      out[j] = (up - down) / (2 * EPS)
    }
    return out
  })
}

function expectMatchesNumeric(
  label: string,
  analytic: readonly (Float32Array | null)[],
  numeric: readonly Float32Array[],
  names: readonly string[],
): void {
  analytic.forEach((g, i) => {
    expect(g, `${label}: no gradient for ${names[i]}`).not.toBeNull()
    const want = numeric[i]!
    expect(g!.length, `${label}: ${names[i]} length`).toBe(want.length)
    for (let j = 0; j < want.length; j++) {
      const scale = Math.max(1, Math.abs(want[j]!), Math.abs(g![j]!))
      expect(
        Math.abs(want[j]! - g![j]!) / scale,
        `${label}: ${names[i]} elem ${j}: numeric ${want[j]} vs autograd ${g![j]}`,
      ).toBeLessThan(TOL)
    }
  })
}

describe("the whole attention block gradchecks", () => {
  const blk = makeBlock(7)
  const x = filled([DIMS.B, DIMS.T, DIMS.D], 77, true)
  const paramNames = [...blk.namedParameters().keys()]
  const names = ["x", ...paramNames]
  const numeric = numericGradients(blk, x)

  const modes: { label: string; lazy: boolean; native: boolean }[] = [
    { label: "eager", lazy: false, native: false },
    { label: "lazy", lazy: true, native: false },
    ...(isNativeAvailable() ? [{ label: "native", lazy: true, native: true }] : []),
  ]

  it.each(modes.map(m => [m.label, m] as const))("%s", (label, mode) => {
    blk.zeroGrad()
    x.zeroGrad()
    if (mode.native) useNative()
    configure({ lazy: mode.lazy })
    blockLoss(blk, x).backward()
    const analytic = [x, ...blk.parameters().map(p => p as AnyTensor)].map(t => t.grad ? Float32Array.from(t.grad.data as Float32Array) : null)
    configure({ lazy: false })
    disableNative()
    expectMatchesNumeric(label, analytic, numeric, names)
  })

  it("compiled", () => {
    // `compile()` traces forward AND backward into one graph, so the
    // gradients it returns are graph outputs rather than a replay of the
    // eager tape. The input is a placeholder and therefore not a
    // differentiable leaf; parameter gradients are the whole of what a
    // compiled training step ever needs, and they are what is checked.
    blk.zeroGrad()
    const params = blk.parameters().map(p => p as AnyTensor)
    const step = compile((xIn: AnyTensor) => {
      blk.zeroGrad()
      const loss = blockLoss(blk, xIn)
      loss.backward()
      return [loss, ...params.map(p => p.grad!)]
    })
    try {
      const out = step(x) as AnyTensor[]
      const analytic = out.slice(1).map(g => Float32Array.from(g.data as Float32Array))
      expectMatchesNumeric("compiled", analytic, numeric.slice(1), names.slice(1))
    } finally {
      step.dispose()
      blk.zeroGrad()
      configure({ lazy: false })
    }
  })
})

describe("a compiled attention step traces once", () => {
  it("traces once over 1000 calls and never re-serialises", () => {
    const mha = new MultiHeadAttention(8, 2, { causal: true })
    seedParameters(mha, 2)
    const x = filled([2, 6, 8], 9)
    resetJsCounters()
    const preparesBefore = nativeCounters().prepares as number
    const step = compile((xIn: AnyTensor) => mha.forward(xIn as never) as AnyTensor)
    try {
      for (let i = 0; i < 1000; i++) step(x)
      // `serializeLazyGraph` runs exactly once, at trace time, and is the
      // only place `noteFallback` can fire, so a fallback count of one
      // after a thousand steps is "prepares === 1" for a graph that
      // cannot reach the addon yet.
      expect(jsCounters().nativeFallbacks).toBe(1)
      expect((nativeCounters().prepares as number) - preparesBefore).toBe(0)
    } finally {
      step.dispose()
    }
  })
})

describe("ModuleList (attention block containers)", () => {
  it("builds n independent modules and reports their parameters by path", () => {
    const stack = ModuleList.of(3, () => new TransformerBlock(8, 2))
    expect(stack.length).toBe(3)
    expect(stack.at(0)).not.toBe(stack.at(1))
    expect(stack.at(-1)).toBe(stack.at(2))
    const names = [...stack.namedParameters().keys()]
    expect(names).toContain("items.0.attn.qkv.weight")
    expect(names).toContain("items.2.ln1.gamma")
    // Three blocks, three independent parameter sets: `of` calls `make`
    // once per index rather than sharing one module.
    expect(names.filter(n => n.endsWith("attn.qkv.weight")).length).toBe(3)
  })

  it("composes in a plain loop, keeping the running shape", () => {
    const stack = ModuleList.of(2, () => new TransformerBlock(8, 2))
    let h = filled([2, 4, 8], 5) as AnyTensor
    for (const blk of stack) h = blk.forward(h as never) as AnyTensor
    expect(h.shape).toEqual([2, 4, 8])
  })

  it("bounds-checks, naming the index and the length", () => {
    const stack = ModuleList.of(2, () => new LayerNorm(4))
    expect(() => stack.at(2)).toThrow(/index 2 is out of range for a list of 2/)
    expect(() => ModuleList.of(-1, () => new LayerNorm(4))).toThrow(/non-negative integer/)
  })
})

describe("Residual (attention block containers)", () => {
  it("is x + inner(x)", () => {
    const inner = new LayerNorm(4)
    const res = new Residual(inner)
    const x = filled([2, 4], 3)
    const got = (res.forward(x as never) as AnyTensor).data as Float32Array
    const want = (x.add(inner.forward(x as never) as AnyTensor) as AnyTensor).data as Float32Array
    for (let i = 0; i < want.length; i++) expect(got[i]).toBe(want[i])
  })

  it("carries the inner module's parameters and its shape effect", () => {
    const res = new Residual(new LayerNorm(4))
    expect([...res.namedParameters().keys()]).toEqual(["inner.gamma", "inner.beta"])
  })

  it("is the runtime twin of ResidualCheck for an undeclared inner layer", () => {
    // `Linear` declares no SHAPE_EFFECT, so the type side is fail-open
    // here by design; the value side has to catch it, and name it.
    const res = new Residual(new Linear(4, 8))
    expect(() => res.forward(filled([2, 4], 1) as never)).toThrow(
      /Residual: Linear maps \[2, 4\] to \[2, 8\]/,
    )
  })
})
