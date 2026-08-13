// Central differences at eps = 1e-3: f32 loss noise (~1e-7) is amplified
// by 1/eps, so the stable relative tolerance is 1e-3, not f64-style 1e-4.

import { afterEach, describe, expect, it } from "vitest"
import { noGrad } from "../src/autograd.ts"
import { configure } from "../src/lazy.ts"
import { Tensor } from "../src/tensor.ts"

type AnyTensor = Tensor<any>

// mulberry32 — small seeded PRNG so the sampled inputs are
// deterministic and the test is never flaky.
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
}

const defaultSample = (rand: () => number): number => (rand() * 2 - 1) * 1.5 + 0.6
const awayFromZero = (rand: () => number): number => (rand() > 0.5 ? 1 : -1) * (0.5 + rand())
const positive = (rand: () => number): number => 0.5 + rand() * 2
// |x| in [0.2, 0.7] or [1.5, 2.5] — never within EPS of ±1, so the
// corners of clamp/maximum/minimum stay outside the difference window
const awayFromUnit = (rand: () => number): number =>
  (rand() > 0.5 ? 1 : -1)
  * (rand() > 0.5 ? 0.2 + rand() * 0.5 : 1.5 + rand())

// Index tensors are exact integers, not sampled inputs, so they are
// built inside `build` rather than coming from `shapes`.
const index = (values: number[]): AnyTensor => Tensor.of(values) as AnyTensor

function checkCase(c: Case, seed: number): void {
  const values = c.shapes.map((shape, i) => {
    const rand = mulberry32(seed + i)
    const sample = c.sample ?? defaultSample
    const n = shape.reduce((a, b) => a * b, 1)
    return Float32Array.from({ length: n }, () => sample(rand))
  })

  const make = (grad: boolean): AnyTensor[] =>
    c.shapes.map((shape, i) => {
      const t = Tensor.zeros(shape as number[]) as AnyTensor
      ;(t.data as Float32Array).set(values[i]!)
      return grad ? (t.requiresGrad() as AnyTensor) : t
    })

  const inputs = make(true)
  const loss = c.build(inputs)
  expect(
    loss.shape,
    `${c.name}: loss must be scalar, got [${loss.shape}]`,
  ).toEqual([])
  loss.backward()

  inputs.forEach((x, i) => {
    expect(
      x.grad,
      `${c.name}: grad for input ${i}`,
    ).not.toBeNull()
    const analytic = x.grad!.data as Float32Array
    const base = values[i]!
    for (let j = 0; j < base.length; j++) {
      const original = base[j]!

      base[j] = original + EPS
      const up = noGrad(() => c.build(make(false)).item())
      base[j] = original - EPS
      const down = noGrad(() => c.build(make(false)).item())
      base[j] = original

      const numeric = (up - down) / (2 * EPS)
      const diff = Math.abs(numeric - analytic[j]!)
      const scale = Math.max(
        1,
        Math.abs(numeric),
        Math.abs(analytic[j]!),
      )
      expect(
        diff / scale,
        `${c.name}: input ${i} elem ${j}: numeric ${numeric} vs autograd ${analytic[j]}`,
      ).toBeLessThan(c.tol ?? TOL)
    }
  })
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
]

describe.each([
  { lazy: false, label: "eager" },
  { lazy: true, label: "lazy" },
])("gradcheck ($label mode)", ({ lazy }) => {
  afterEach(() => {
    configure({ lazy: false })
  })

  it.each(CASES.map(c => [c.name, c] as const))(
    "%s",
    (_name, c) => {
      configure({ lazy })
      checkCase(c, 1234)
    },
  )
})
