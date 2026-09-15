// Central differences at eps = 1e-3: f32 loss noise (~1e-7) is amplified
// by 1/eps, so the stable relative tolerance is 1e-3, not f64-style 1e-4.

import { afterEach, describe, expect, it } from "vitest"
import { noGrad } from "../src/autograd.ts"
import {
  _nativeState,
  _setNativeState,
  disableNative,
  isNativeAvailable,
  isNativeEnabled,
  nativeCounters,
  useNative,
} from "../src/backends/native.ts"
import { configure, isLazy } from "../src/lazy.ts"
import { sdpa } from "../src/nn/functional.ts"
import {
  contiguous,
  crossEntropy,
  dropout,
  gatherRows,
  gelu,
  layerNorm,
  logSumExp,
  rmsNorm,
  scatterAddRows,
  silu,
  softmax,
  Tensor,
} from "../src/tensor.ts"
import { testing } from "../src/testing.ts"

type AnyTensor = Tensor<any>

// mulberry32: small seeded PRNG so the sampled inputs are deterministic
// and the test is never flaky.
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

const EPS = 1e-3
const TOL = 1e-3

interface Case {
  readonly name: string
  readonly shapes: readonly (readonly number[])[]
  readonly build: (xs: AnyTensor[]) => AnyTensor
  /** Sampler, so rules with kinks (relu, abs) can avoid the kink. */
  readonly sample?: (rand: () => number) => number
  /** Per-case tolerance override, when f32 cancellation in the
      finite difference sits just above the global 1e-3 floor. */
  readonly tol?: number
  /**
   * Marks a case the native block must filter: the op has no native
   * kernel or lowering yet, so a native run would land on the JS
   * interpreter and the "the native graph was prepared" assertion in
   * `checkCase` would fail for the wrong reason. The eager and lazy
   * blocks run every case.
   */
  readonly nativeFrom?: string
}

interface CheckOpts {
  /** dtype to run the case in; float64 checks gradient precision. */
  readonly dtype?: "float32" | "float64"
  readonly eps?: number
  readonly tol?: number
}

const defaultSample = (rand: () => number): number => (rand() * 2 - 1) * 1.5 + 0.6
const awayFromZero = (rand: () => number): number => (rand() > 0.5 ? 1 : -1) * (0.5 + rand())
const positive = (rand: () => number): number => 0.5 + rand() * 2
// |x| in [0.2, 0.7] or [1.5, 2.5]: never within EPS of ±1, so the
// corners of clamp/maximum/minimum stay outside the difference window
const awayFromUnit = (rand: () => number): number =>
  (rand() > 0.5 ? 1 : -1)
  * (rand() > 0.5 ? 0.2 + rand() * 0.5 : 1.5 + rand())

// Index tensors are exact integers, not sampled inputs, so they are
// built inside `build` rather than coming from `shapes`. `.toIndex()`
// brands them: `indexSelect`/`scatterAdd` take an `IndexTensor`, never a
// bare `Tensor<[E]>`.
//
// The static return is widened to `any` because `IndexTensor` is
// invariant in its shape: the cases below feed the same helper into
// parameters spelled `[number]`, `[any]` and `[never]`, and no single
// concrete spelling fits all three. The value is a real branded index
// tensor; only the type is loosened, and only inside this test.
const index = (values: number[]): any => Tensor.of(values).toIndex()

function checkCase(c: Case, seed: number, opts: CheckOpts = {}): void {
  const dtype = opts.dtype ?? "float32"
  const eps = opts.eps ?? EPS
  const tol = opts.tol ?? TOL

  const makeValues = (
    next: () => number,
    n: number,
  ): Float32Array | Float64Array =>
    dtype === "float64"
      ? Float64Array.from({ length: n }, () => next())
      : Float32Array.from({ length: n }, () => next())

  const values = c.shapes.map((shape, i) => {
    const rand = mulberry32(seed + i)
    const sample = c.sample ?? defaultSample
    const n = shape.reduce((a, b) => a * b, 1)
    return makeValues(() => sample(rand), n)
  })

  const make = (grad: boolean): AnyTensor[] =>
    c.shapes.map((shape, i) => {
      const t = Tensor.zeros(shape as number[]).to(dtype) as AnyTensor
      ;(t.data as Float32Array | Float64Array).set(values[i]!)
      return grad ? (t.requiresGrad() as AnyTensor) : t
    })

  const inputs = make(true)
  const loss = c.build(inputs)
  expect(
    loss.shape,
    `${c.name}: loss must be scalar, got [${loss.shape}]`,
  ).toEqual([])

  // Under native mode, the analytic gradient must actually run through
  // the native path: record the prepare count before backward() to
  // confirm it advanced, rather than silently falling back to JS.
  const nativeUnderTest = isNativeEnabled()
  const preparesBefore = nativeUnderTest
    ? (nativeCounters().prepares as number)
    : 0

  loss.backward()

  if (nativeUnderTest) {
    const preparesAfter = nativeCounters().prepares as number
    expect(
      preparesAfter,
      `${c.name}: expected the native graph to be prepared for the analytic gradient`,
    ).toBeGreaterThan(preparesBefore)
  }

  inputs.forEach((x, i) => {
    expect(
      x.grad,
      `${c.name}: grad for input ${i}`,
    ).not.toBeNull()
    if (nativeUnderTest) {
      expect(
        testing.storageOf(x.grad!),
        `${c.name}: grad for input ${i} should be a materialized native result`,
      ).toBe("materialized")
    }
  })

  // The finite-difference reference is always taken with the eager
  // evaluator, even when the analytic gradient above ran natively:
  // f32 accumulation on the native path is noisier, and central
  // differences amplify that noise by 1/eps, so comparing native noise
  // against native noise would hide real regressions. Eager and lazy
  // modes are unaffected; this is a no-op save/restore for them.
  const savedNativeState = _nativeState()
  const savedLazy = isLazy()
  if (nativeUnderTest) {
    disableNative()
    configure({ lazy: false })
  }
  try {
    inputs.forEach((x, i) => {
      const analytic = x.grad!.data as Float32Array | Float64Array
      const base = values[i]!
      for (let j = 0; j < base.length; j++) {
        const original = base[j]!

        base[j] = original + eps
        const up = noGrad(() => c.build(make(false)).item())
        base[j] = original - eps
        const down = noGrad(() => c.build(make(false)).item())
        base[j] = original

        const numeric = (up - down) / (2 * eps)
        const diff = Math.abs(numeric - analytic[j]!)
        const scale = Math.max(
          1,
          Math.abs(numeric),
          Math.abs(analytic[j]!),
        )
        expect(
          diff / scale,
          `${c.name}: input ${i} elem ${j}: numeric ${numeric} vs autograd ${analytic[j]}`,
        ).toBeLessThan(c.tol ?? tol)
      }
    })
  } finally {
    if (nativeUnderTest) {
      _setNativeState(savedNativeState)
      configure({ lazy: savedLazy })
    }
  }
}

const CASES: Case[] = [
  {
    name: "add",
    shapes: [[3], [3]],
    build: ([a, b]) => a!.add(b!).sum(),
  },
  {
    name: "sub",
    shapes: [[3], [3]],
    build: ([a, b]) => a!.sub(b!).sum(),
  },
  {
    name: "mul",
    shapes: [[3], [3]],
    build: ([a, b]) => a!.mul(b!).sum(),
  },
  {
    name: "div",
    shapes: [[3], [3]],
    build: ([a, b]) => a!.div(b!).sum(),
    sample: awayFromZero,
  },

  {
    name: "broadcast add [2,3]+[3]",
    shapes: [[2, 3], [3]],
    build: ([a, b]) => a!.add(b!).sum(),
  },
  {
    name: "broadcast mul [2,1]*[1,3]",
    shapes: [
      [2, 1],
      [1, 3],
    ],
    build: ([a, b]) => a!.mul(b!).sum(),
  },

  {
    name: "addScalar",
    shapes: [[4]],
    build: ([a]) => a!.add(2.5).sum(),
  },
  {
    name: "mulScalar",
    shapes: [[4]],
    build: ([a]) => a!.mul(-1.5).sum(),
  },
  {
    // typenet has no rsub; 1 - a as neg + addScalar
    name: "rsub (1 - a)",
    shapes: [[4]],
    build: ([a]) => a!.neg().add(1).sum(),
  },

  {
    name: "pow(3)",
    shapes: [[4]],
    build: ([a]) => a!.pow(3).sum(),
  },
  {
    name: "neg",
    shapes: [[4]],
    build: ([a]) => a!.neg().sum(),
  },
  {
    name: "exp",
    shapes: [[4]],
    build: ([a]) => a!.exp().sum(),
  },
  {
    name: "log",
    shapes: [[4]],
    build: ([a]) => a!.log().sum(),
    sample: positive,
  },
  {
    name: "sqrt",
    shapes: [[4]],
    build: ([a]) => a!.sqrt().sum(),
    sample: positive,
  },
  {
    name: "abs",
    shapes: [[4]],
    build: ([a]) => a!.abs().sum(),
    sample: awayFromZero,
  },
  {
    name: "relu",
    shapes: [[6]],
    build: ([a]) => a!.relu().sum(),
    sample: awayFromZero,
  },
  {
    name: "leakyRelu",
    shapes: [[6]],
    build: ([a]) => a!.leakyRelu(0.2).sum(),
    sample: awayFromZero,
  },
  {
    name: "sigmoid",
    shapes: [[4]],
    build: ([a]) => a!.sigmoid().sum(),
  },
  {
    name: "tanh",
    shapes: [[4]],
    build: ([a]) => a!.tanh().sum(),
  },

  {
    name: "softmax(1)",
    shapes: [[2, 3]],
    build: ([a]) => a!.softmax(1).mul(2).sum(),
  },
  {
    name: "logSoftmax(1)",
    shapes: [[2, 3]],
    build: ([a]) => a!.logSoftmax(1).mul(2).sum(),
  },

  {
    name: "matmul [2,3]@[3,4]",
    shapes: [
      [2, 3],
      [3, 4],
    ],
    build: ([a, b]) => a!.matmul(b!).sum(),
  },
  {
    name: "matmul batched [2,3,4]@[2,4,5]",
    shapes: [
      [2, 3, 4],
      [2, 4, 5],
    ],
    build: ([a, b]) => a!.matmul(b!).sum(),
    // grads near 0.05–0.15 against much larger intermediates:
    // f32 cancellation in the difference sits just above 1e-3
    tol: 2e-3,
  },
  {
    name: "matmul broadcast batch [1,3,4]@[2,4,5]",
    shapes: [
      [1, 3, 4],
      [2, 4, 5],
    ],
    build: ([a, b]) => a!.matmul(b!).sum(),
    tol: 2e-3,
  },

  {
    name: "sum(dim)",
    shapes: [[2, 3]],
    build: ([a]) => a!.sum(1).sum(),
  },
  {
    name: "sum(dim, keepdim)",
    shapes: [[2, 3]],
    build: ([a]) => a!.sum(1, true).sum(),
  },
  {
    name: "mean()",
    shapes: [[2, 3]],
    build: ([a]) => a!.mean(),
  },
  {
    name: "mean(dim)",
    shapes: [[2, 3]],
    build: ([a]) => a!.mean(0).sum(),
  },

  {
    name: "view",
    shapes: [[2, 3]],
    build: ([a]) => a!.view([3, 2]).sum(),
  },
  {
    name: "view(-1)",
    shapes: [[2, 3]],
    build: ([a]) => a!.view([-1]).sum(),
  },
  {
    name: "T",
    shapes: [[2, 3]],
    build: ([a]) => (a!.T as AnyTensor).mul(2).sum(),
  },
  {
    name: "permute",
    shapes: [[2, 3, 4]],
    build: ([a]) => a!.permute(2, 0, 1).mul(2).sum(),
  },
  {
    name: "unsqueeze/squeeze",
    shapes: [[2, 3]],
    build: ([a]) => a!.unsqueeze(1).squeeze().mul(2).sum(),
  },
  {
    name: "cat(dim 0)",
    shapes: [
      [2, 3],
      [4, 3],
    ],
    build: ([a, b]) => Tensor.cat(a!, b!, 0).mul(2).sum(),
  },
  {
    name: "cat(dim 1)",
    shapes: [
      [2, 3],
      [2, 5],
    ],
    build: ([a, b]) => Tensor.cat(a!, b!, 1).mul(2).sum(),
  },
  {
    name: "stack(dim 1)",
    shapes: [
      [2],
      [2],
    ],
    build: ([a, b]) => Tensor.stack([a!, b!] as any, 1 as any).pow(3).sum(),
  },

  {
    name: "mse-style ((a-b)^2).mean()",
    shapes: [
      [3, 2],
      [3, 2],
    ],
    build: ([a, b]) => a!.sub(b!).pow(2).mean(),
  },
  {
    name: "mlp: tanh(x@W).sigmoid().sum()",
    shapes: [
      [4, 2],
      [2, 8],
      [8, 1],
    ],
    build: ([x, w1, w2]) => x!.matmul(w1!).tanh().matmul(w2!).sigmoid().sum(),
  },
  {
    name: "attention core [1,2,3,4]",
    shapes: [
      [1, 2, 3, 4],
      [1, 2, 3, 4],
      [1, 2, 3, 4],
    ],
    // typenet's .T is rank-2 only; transpose the trailing axes
    build: ([q, k, v]) =>
      q!
        .matmul(k!.transpose(-1, -2))
        .mul(0.5)
        .softmax(-1)
        .matmul(v!)
        .mul(2)
        .sum(),
    tol: 2e-3,
  },

  {
    name: "indexSelect(dim 0), repeated indices",
    shapes: [[4, 3]],
    build: ([a]) =>
      a!
        .indexSelect(index([2, 0, 0, 3, 1]))
        .pow(3)
        .sum(),
  },
  {
    name: "indexSelect(dim 1)",
    shapes: [[2, 4]],
    build: ([a]) =>
      a!
        .indexSelect(index([3, 1, 1]), 1)
        .pow(3)
        .sum(),
  },
  {
    name: "indexSelect, some rows unused",
    shapes: [[5, 2]],
    build: ([a]) =>
      a!
        .indexSelect(index([1, 1, 4]))
        .pow(3)
        .sum(),
  },
  {
    name: "scatterAdd(dim 0), colliding indices",
    shapes: [[5, 3]],
    build: ([a]) =>
      a!
        .scatterAdd(index([2, 0, 0, 1, 1]), 3)
        .pow(3)
        .sum(),
  },
  {
    name: "scatterAdd(dim 1), empty output rows",
    shapes: [[2, 3]],
    build: ([a]) =>
      a!
        .scatterAdd(index([3, 0, 3]), 4, 1)
        .pow(3)
        .sum(),
  },
  {
    name: "message passing: gather, scale, scatter",
    shapes: [
      [4, 3],
      [4, 3],
    ],
    build: ([x, w]) => {
      const src = index([0, 1, 1, 2, 3, 0])
      const dst = index([1, 0, 2, 3, 0, 3])
      const messages = x!
        .indexSelect(src)
        .sub(x!.indexSelect(dst))
        .tanh()
      return messages.scatterAdd(dst, 4).mul(w!).sum()
    },
  },

  {
    name: "maximum(a, -1)",
    shapes: [[5]],
    build: ([a]) => a!.maximum(-1).pow(3).sum(),
    sample: awayFromUnit,
  },
  {
    name: "minimum(a, 1)",
    shapes: [[5]],
    build: ([a]) => a!.minimum(1).pow(3).sum(),
    sample: awayFromUnit,
  },
  {
    name: "maximum(a, b)",
    shapes: [[5], [5]],
    build: ([a, b]) => a!.maximum(b!).pow(3).sum(),
    sample: awayFromUnit,
  },
  {
    name: "minimum(a, b) broadcast [2,1] vs [1,3]",
    shapes: [
      [2, 1],
      [1, 3],
    ],
    build: ([a, b]) => a!.minimum(b!).pow(3).sum(),
    sample: awayFromUnit,
  },
  {
    name: "clamp(-1, 1)",
    shapes: [[6]],
    build: ([a]) => a!.clamp(-1, 1).pow(3).sum(),
    sample: awayFromUnit,
  },
  {
    name: "narrow(dim 1)",
    shapes: [[3, 5]],
    build: ([a]) => a!.narrow(1, 1, 3).pow(3).sum(),
  },
  {
    name: "narrow(dim 0), whole tensor",
    shapes: [[4, 2]],
    build: ([a]) => a!.narrow(0, 0, 4).pow(3).sum(),
  },
  {
    name: "narrow then cat back",
    shapes: [[2, 6]],
    build: ([a]) =>
      Tensor.cat(a!.narrow(1, 3, 3), a!.narrow(1, 0, 3), 1)
        .pow(3)
        .sum(),
  },
  {
    name: "overflow penalty (x - clamp(x)).abs().mean()",
    shapes: [[6]],
    build: ([a]) => a!.sub(a!.clamp(-1, 1)).abs().mean() as AnyTensor,
    sample: awayFromUnit,
  },

  // semantic ops: the addon cannot parse these yet, so the native block
  // filters them (`nativeFrom`) until a lowering exists
  {
    name: "gelu",
    shapes: [[6]],
    build: ([a]) => gelu(a!).mul(2).sum() as AnyTensor,
    nativeFrom: "A-L1",
  },
  {
    name: "silu",
    shapes: [[6]],
    build: ([a]) => silu(a!).mul(2).sum() as AnyTensor,
    nativeFrom: "A-L1",
  },
  {
    name: "softmax node(1)",
    shapes: [[2, 3]],
    build: ([a]) => softmax(a!, 1).mul(2).sum() as AnyTensor,
    nativeFrom: "A-L1",
  },
  {
    name: "softmax node(-1) causal",
    shapes: [[2, 3, 3]],
    build: ([a]) => softmax(a!, -1, { causal: true }).mul(2).sum() as AnyTensor,
    nativeFrom: "A-L1",
  },
  {
    name: "logSumExp(1)",
    shapes: [[2, 3]],
    build: ([a]) => logSumExp(a!, 1).mul(2).sum() as AnyTensor,
    nativeFrom: "A-L1",
  },
  {
    name: "logSumExp(1, keepdim)",
    shapes: [[2, 3]],
    build: ([a]) => logSumExp(a!, 1, true).mul(2).sum() as AnyTensor,
    nativeFrom: "A-L1",
  },
  {
    name: "layerNorm",
    shapes: [[3, 4], [4], [4]],
    build: ([x, g, b]) =>
      layerNorm(x as any, g as any, b as any)
        .pow(3)
        .sum() as AnyTensor,
    nativeFrom: "A-L1",
  },
  {
    name: "rmsNorm",
    shapes: [[3, 4], [4]],
    build: ([x, g]) =>
      rmsNorm(x as any, g as any)
        .pow(3)
        .sum() as AnyTensor,
    nativeFrom: "A-L1",
  },
  {
    name: "crossEntropy [4,3]",
    shapes: [[4, 3]],
    build: ([a]) => crossEntropy(a as any, index([2, 0, 1, 2]) as any) as AnyTensor,
    nativeFrom: "A-L1",
  },
  {
    // The [B,T,V] shape a transformer's LM head actually produces. The
    // explicit `flatten(0, 1)` is kept on purpose: it puts `flatten`'s
    // own backward under the same finite-difference check as the loss.
    // The no-reshape spelling is covered by `test/losses.test.ts`.
    name: "crossEntropy over [B,T,V]",
    shapes: [[2, 3, 4]],
    build: ([a]) =>
      crossEntropy(
        a!.flatten(0, 1) as any,
        index([3, 0, 1, 2, 3, 0]) as any,
      ) as AnyTensor,
    nativeFrom: "A-L1",
  },
  {
    // `sdpa` is not an IR node itself: it is matmul/mul (native) around
    // a softmax{causal} node (not yet native), so it carries the same
    // flag as the softmax node above.
    name: "sdpa (causal)",
    shapes: [
      [1, 2, 3, 2], // q [B,H,T,K]
      [1, 2, 2, 3], // k [B,H,K,T], already transposed (sdpa's own contract)
      [1, 2, 3, 2], // v [B,H,T,K]
    ],
    build: ([q, k, v]) =>
      sdpa(q as any, k as any, v as any, { causal: true })
        .pow(2)
        .sum() as AnyTensor,
    nativeFrom: "A-L1",
  },
  {
    name: "gatherRows, rank-1 index",
    shapes: [[4, 3]],
    build: ([a]) =>
      gatherRows(a as any, index([2, 0, 0, 3, 1]) as any)
        .pow(3)
        .sum() as AnyTensor,
    nativeFrom: "A-L1",
  },
  {
    name: "gatherRows, rank-2 index",
    shapes: [[4, 3]],
    build: ([a]) =>
      gatherRows(
        a as any,
        Tensor.of([[1, 0, 0], [3, 3, 2]]) as any,
      )
        .pow(3)
        .sum() as AnyTensor,
    nativeFrom: "A-L1",
  },
  {
    name: "scatterAddRows, colliding indices",
    shapes: [[3, 2]],
    build: ([a]) =>
      scatterAddRows(a as any, index([2, 0, 0]) as any, 4)
        .pow(3)
        .sum() as AnyTensor,
    nativeFrom: "A-L1",
  },
  {
    // p = 0 is the only deterministic dropout, and determinism is what a
    // finite difference needs: with p > 0 each perturbed evaluation would
    // draw its own mask. That the mask is shared between forward and
    // backward at p > 0 is asserted directly in `test/semantic-ops.test.ts`.
    name: "dropout(p=0)",
    shapes: [[6]],
    build: ([a]) => dropout(a!, 0).pow(3).sum() as AnyTensor,
    nativeFrom: "A-L1",
  },
  {
    name: "contiguous",
    shapes: [[2, 3]],
    build: ([a]) => contiguous(a!).pow(3).sum() as AnyTensor,
    nativeFrom: "A-L1",
  },
  {
    // The multi-axis `reduce{dims}`: a `[2,3,4] + [4]` bias backward
    // reduces axes 0 and 1 in one node, and `reduce` is a wire op at
    // every arity, so it is native-clean.
    name: "broadcast add [2,3,4]+[4] (multi-axis sumTo)",
    shapes: [[2, 3, 4], [4]],
    // `tanh` rather than `pow(3)`: a 24-term f32 sum of cubes puts the
    // central difference's own cancellation above the global floor, which
    // would be a statement about the test and not about `sumTo`.
    build: ([a, b]) => a!.add(b!).tanh().sum(),
  },
  {
    name: "broadcast mul [2,3,4]*[1,1,4] (keepdim sumTo)",
    shapes: [[2, 3, 4], [1, 1, 4]],
    build: ([a, b]) => a!.mul(b!).tanh().sum(),
  },
]

describe.each([
  { lazy: false, native: false, label: "eager" },
  { lazy: true, native: false, label: "lazy" },
  ...(isNativeAvailable() ? [{ lazy: true, native: true, label: "native" }] : []),
])("gradcheck ($label mode)", ({ lazy, native }) => {
  afterEach(() => {
    configure({ lazy: false })
    disableNative()
  })

  // Filtered, not skipped: a case whose op the addon cannot run is not a
  // native case yet, and `.skip`ing it would leave a permanently red-ish
  // suite that nobody reads.
  const cases = native
    ? CASES.filter(c => c.nativeFrom === undefined)
    : CASES

  it.each(cases.map(c => [c.name, c] as const))(
    "%s",
    (_name, c) => {
      if (native) useNative()
      configure({ lazy })
      checkCase(c, 1234)
    },
  )
})

// f32 finite differences cap out around 1e-3 relative error; a float64
// pass at a 10x tighter tolerance guards against precision regressions
// that f32 noise would hide. Runs eager-only: lazy graphs reject f64
// leaves (native compute is f32-only).
describe("gradcheck (float64 precision)", () => {
  afterEach(() => {
    configure({ lazy: false })
  })

  it("matches finite differences at float64 precision", () => {
    configure({ lazy: false })
    checkCase(
      {
        name: "mlp (float64): tanh(x@W).sigmoid().sum()",
        shapes: [
          [4, 2],
          [2, 8],
          [8, 1],
        ],
        build: ([x, w1, w2]) => x!.matmul(w1!).tanh().matmul(w2!).sigmoid().sum(),
      },
      5678,
      { dtype: "float64", eps: 1e-4, tol: 1e-4 },
    )
  })
})
