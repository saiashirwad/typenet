import { describe, expect, it } from "vitest"
import { withContext } from "../src/context.ts"
import { init, Linear } from "../src/nn/index.ts"
import { type AnyTensor, Tensor } from "../src/tensor.ts"

function stats(data: ArrayLike<number>): { mean: number; variance: number } {
  let sum = 0
  for (let i = 0; i < data.length; i++) sum += data[i]!
  const mean = sum / data.length
  let sq = 0
  for (let i = 0; i < data.length; i++) sq += (data[i]! - mean) ** 2
  return { mean, variance: sq / data.length }
}

/** ±10% relative tolerance against an analytic value — W1.9's accept #1. */
function expectVarianceNear(data: ArrayLike<number>, analytic: number) {
  const { variance } = stats(data)
  expect(variance).toBeGreaterThan(analytic * 0.9)
  expect(variance).toBeLessThan(analytic * 1.1)
}

/**
 * The analytic variance of `Normal(mean, std)` truncated to `[a, b]`,
 * by direct numeric quadrature of the truncated density — deliberately
 * independent of init.ts's own `erf`/`erfinv` approximations, so this
 * is an honest check of the implementation rather than a restatement
 * of it.
 */
function truncNormalVariance(mean: number, std: number, a: number, b: number): number {
  const steps = 20_000
  const dx = (b - a) / steps
  const phi = (x: number) => Math.exp(-0.5 * ((x - mean) / std) ** 2) / (std * Math.sqrt(2 * Math.PI))
  let z = 0, m1 = 0
  for (let i = 0; i <= steps; i++) {
    const x = a + i * dx
    const w = (i === 0 || i === steps) ? 1 : (i % 2 === 0 ? 2 : 4)
    z += w * phi(x)
    m1 += w * x * phi(x)
  }
  z *= dx / 3
  m1 *= dx / 3
  const meanT = m1 / z
  let m2 = 0
  for (let i = 0; i <= steps; i++) {
    const x = a + i * dx
    const w = (i === 0 || i === steps) ? 1 : (i % 2 === 0 ? 2 : 4)
    m2 += w * (x - meanT) ** 2 * phi(x)
  }
  m2 *= dx / 3
  return m2 / z
}

describe("in-place constant forms", () => {
  it("zeros_ / ones_ / constant_ overwrite every element", () => {
    const t = Tensor.full([8], 7) as AnyTensor
    expect(init.zeros_(t)).toBe(t)
    expect(Array.from(t.data)).toEqual(new Array(8).fill(0))
    init.ones_(t)
    expect(Array.from(t.data)).toEqual(new Array(8).fill(1))
    init.constant_(t, 3.5)
    expect(Array.from(t.data)).toEqual(new Array(8).fill(3.5))
  })

  it("pure zeros/ones/constant build a fresh tensor of the given shape", () => {
    expect(Array.from(init.zeros([4]).data)).toEqual([0, 0, 0, 0])
    expect(Array.from(init.ones([4]).data)).toEqual([1, 1, 1, 1])
    expect(Array.from(init.constant([4], 9).data)).toEqual([9, 9, 9, 9])
  })
})

describe("statistical accuracy (§ accept #1: within 10% of the analytic variance at [1024,1024])", () => {
  const BIG: [1024, 1024] = [1024, 1024]

  it("uniform_", () => {
    const t = init.uniform_(Tensor.zeros(BIG) as AnyTensor, { low: -3, high: 5 })
    expectVarianceNear(t.data, (5 - -3) ** 2 / 12)
  })

  it("normal_", () => {
    const t = init.normal_(Tensor.zeros(BIG) as AnyTensor, { mean: 2, std: 0.7 })
    expectVarianceNear(t.data, 0.7 ** 2)
  })

  it("kaimingUniform_ — default (fanMode: fanIn)", () => {
    const shape: [512, 2048] = [512, 2048]
    const fanIn = shape[1]
    const t = init.kaimingUniform_(Tensor.zeros(shape) as AnyTensor)
    expectVarianceNear(t.data, 1 / (3 * fanIn))
  })

  it("kaimingUniform_ — fanMode: fanOut", () => {
    const shape: [512, 2048] = [512, 2048]
    const fanOut = shape[0]
    const t = init.kaimingUniform_(Tensor.zeros(shape) as AnyTensor, { fanMode: "fanOut" })
    expectVarianceNear(t.data, 1 / (3 * fanOut))
  })

  it("kaimingUniform_ — nonlinearity: relu", () => {
    const shape: [512, 2048] = [512, 2048]
    const fanIn = shape[1]
    const t = init.kaimingUniform_(Tensor.zeros(shape) as AnyTensor, { nonlinearity: "relu" })
    const gain = Math.sqrt(2)
    expectVarianceNear(t.data, gain ** 2 / fanIn)
  })

  it("kaimingNormal_ — default", () => {
    const shape: [512, 2048] = [512, 2048]
    const fanIn = shape[1]
    const t = init.kaimingNormal_(Tensor.zeros(shape) as AnyTensor)
    expectVarianceNear(t.data, 1 / (3 * fanIn))
  })

  it("kaimingNormal_ — nonlinearity: relu", () => {
    const shape: [512, 2048] = [512, 2048]
    const fanIn = shape[1]
    const t = init.kaimingNormal_(Tensor.zeros(shape) as AnyTensor, { nonlinearity: "relu" })
    expectVarianceNear(t.data, 2 / fanIn)
  })

  it("xavierUniform_", () => {
    const shape: [512, 2048] = [512, 2048]
    const t = init.xavierUniform_(Tensor.zeros(shape) as AnyTensor)
    expectVarianceNear(t.data, 2 / (shape[0] + shape[1]))
  })

  it("xavierNormal_", () => {
    const shape: [512, 2048] = [512, 2048]
    const t = init.xavierNormal_(Tensor.zeros(shape) as AnyTensor)
    expectVarianceNear(t.data, 2 / (shape[0] + shape[1]))
  })

  it("truncNormal_ — bounds respected and variance matches the truncated-normal analytic value", () => {
    const t = init.truncNormal_(Tensor.zeros(BIG) as AnyTensor, { mean: 0, std: 1, a: -2, b: 2 })
    for (const x of t.data) {
      expect(x).toBeGreaterThanOrEqual(-2)
      expect(x).toBeLessThanOrEqual(2)
    }
    expectVarianceNear(t.data, truncNormalVariance(0, 1, -2, 2))
  }, 20_000)
})

describe("determinism under a seed", () => {
  it("uniform_ repeats exactly for a given seed", () => {
    const draw = () =>
      withContext(
        { seed: 41 },
        () => Array.from(init.uniform_(Tensor.zeros([64]) as AnyTensor).data),
      )
    expect(draw()).toEqual(draw())
  })

  it("kaimingNormal_ repeats exactly for a given seed", () => {
    const draw = () =>
      withContext(
        { seed: 41 },
        () => Array.from(init.kaimingNormal_(Tensor.zeros([16, 32]) as AnyTensor).data),
      )
    expect(draw()).toEqual(draw())
  })

  it("an explicit generator() reproduces identical draws independent of call order", () => {
    const g1 = init.generator(7)
    const a = Array.from(init.uniform([32] as const, { generator: g1 }).data)
    // Consume some of the ambient counter in between — an explicit
    // generator must not be affected by it.
    void init.uniform([32])
    const g2 = init.generator(7)
    const b = Array.from(init.uniform([32] as const, { generator: g2 }).data)
    expect(b).toEqual(a)
  })
})

describe("calculateFan", () => {
  it("throws, naming the shape, for rank < 2", () => {
    expect(() => init.calculateFan([8], "kaimingUniform")).toThrow(/rank >= 2/)
    expect(() => init.calculateFan([8], "kaimingUniform")).toThrow(/\[8\]/)
  })

  it("reads fanIn from dim 1 and fanOut from dim 0, receptive field beyond that", () => {
    expect(init.calculateFan([6, 5, 3, 3], "x")).toEqual({ fanIn: 5 * 9, fanOut: 6 * 9 })
  })
})

describe("Linear re-pointed at init.kaimingUniform_ (W1.9 accept #2)", () => {
  it("is bit-identical to the pre-refactor `1/sqrt(fanIn)` formula for a fixed seed", () => {
    const reference = () =>
      withContext({ seed: 12345 }, () => {
        const k = 1 / Math.sqrt(37)
        return Array.from(
          (Tensor.rand([37, 53]) as AnyTensor).mul(2 * k).sub(k).data,
        )
      })
    const actual = () => withContext({ seed: 12345 }, () => Array.from(new Linear(37, 53).weight.data))
    expect(actual()).toEqual(reference())
  })

  it("holds for a second, differently-shaped layer too", () => {
    const reference = () =>
      withContext({ seed: 999 }, () => {
        const k = 1 / Math.sqrt(128)
        return Array.from(
          (Tensor.rand([128, 4]) as AnyTensor).mul(2 * k).sub(k).data,
        )
      })
    const actual = () => withContext({ seed: 999 }, () => Array.from(new Linear(128, 4).weight.data))
    expect(actual()).toEqual(reference())
  })
})
