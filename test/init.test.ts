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

function expectVarianceNear(data: ArrayLike<number>, analytic: number) {
  const { variance } = stats(data)
  expect(variance).toBeGreaterThan(analytic * 0.9)
  expect(variance).toBeLessThan(analytic * 1.1)
}

/**
 * The analytic variance of Normal(mean, std) truncated to [a, b], by direct quadrature of the
 * truncated density. Independent of init.ts's own erf/erfinv approximations.
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

describe("statistical accuracy (within 10% of the analytic variance at [1024,1024])", () => {
  const BIG: [1024, 1024] = [1024, 1024]

  it("uniform_", () => {
    const t = init.uniform_(Tensor.zeros(BIG) as AnyTensor, { low: -3, high: 5 })
    expectVarianceNear(t.data, (5 - -3) ** 2 / 12)
  })

  it("normal_", () => {
    const t = init.normal_(Tensor.zeros(BIG) as AnyTensor, { mean: 2, std: 0.7 })
    expectVarianceNear(t.data, 0.7 ** 2)
  })

  it("kaimingUniform_: default (fanMode: fanIn)", () => {
    const shape: [512, 2048] = [512, 2048]
    const fanIn = shape[1]
    const t = init.kaimingUniform_(Tensor.zeros(shape) as AnyTensor)
    expectVarianceNear(t.data, 1 / (3 * fanIn))
  })

  it("kaimingUniform_: fanMode: fanOut", () => {
    const shape: [512, 2048] = [512, 2048]
    const fanOut = shape[0]
    const t = init.kaimingUniform_(Tensor.zeros(shape) as AnyTensor, { fanMode: "fanOut" })
    expectVarianceNear(t.data, 1 / (3 * fanOut))
  })

  it("kaimingUniform_: nonlinearity: relu", () => {
    const shape: [512, 2048] = [512, 2048]
    const fanIn = shape[1]
    const t = init.kaimingUniform_(Tensor.zeros(shape) as AnyTensor, { nonlinearity: "relu" })
    const gain = Math.sqrt(2)
    expectVarianceNear(t.data, gain ** 2 / fanIn)
  })

  it("kaimingNormal_: default", () => {
    const shape: [512, 2048] = [512, 2048]
    const fanIn = shape[1]
    const t = init.kaimingNormal_(Tensor.zeros(shape) as AnyTensor)
    expectVarianceNear(t.data, 1 / (3 * fanIn))
  })

  it("kaimingNormal_: nonlinearity: relu", () => {
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

  it("truncNormal_: bounds respected and variance matches the truncated-normal analytic value", () => {
    const t = init.truncNormal_(Tensor.zeros(BIG) as AnyTensor, { mean: 0, std: 1, a: -2, b: 2 })
    for (const x of t.data) {
      expect(x).toBeGreaterThanOrEqual(-2)
      expect(x).toBeLessThanOrEqual(2)
    }
    expectVarianceNear(t.data, truncNormalVariance(0, 1, -2, 2))
  }, 20_000)
})

describe("pure (non-mutating) initialisers", () => {
  it("truncNormal keeps values inside a shifted, scaled 2-sigma window", () => {
    const t = init.truncNormal([512, 512], {
      mean: 4,
      std: 0.5,
      a: 3,
      b: 5,
      generator: init.generator(11),
    })
    for (const x of t.data) {
      expect(x).toBeGreaterThanOrEqual(4 - 2 * 0.5)
      expect(x).toBeLessThanOrEqual(4 + 2 * 0.5)
    }
    // Symmetric bounds around mean, so the truncated mean is mean exactly and the variance
    // scales by std squared.
    expect(Math.abs(stats(t.data).mean - 4)).toBeLessThan(0.01)
    expectVarianceNear(t.data, 0.5 ** 2 * truncNormalVariance(0, 1, -2, 2))
  }, 20_000)

  it("truncNormal matches the truncated-normal mean and variance for asymmetric bounds", () => {
    const t = init.truncNormal([512, 512], { a: -1, b: 2, generator: init.generator(12) })
    // E[X | -1 < X < 2] = (phi(-1) - phi(2)) / (Phi(2) - Phi(-1)) for a standard normal.
    expect(Math.abs(stats(t.data).mean - 0.2296)).toBeLessThan(0.02)
    expectVarianceNear(t.data, truncNormalVariance(0, 1, -1, 2))
  }, 20_000)

  it("truncNormal rejects an empty interval", () => {
    expect(() => init.truncNormal([4], { a: 1, b: 1 })).toThrow(/must be less than/)
  })

  it("truncNormal respects the window when the whole window sits in the far tail", () => {
    // Mean 100 sigma above the default [-2, 2] window: both CDF endpoints underflow, so the
    // inverse-CDF map carries no trace of the window and only a value clamp can hold it.
    const t = init.truncNormal([256], { mean: 100, std: 1, generator: init.generator(23) })
    const values = Array.from(t.data)
    expect(Math.min(...values)).toBeGreaterThanOrEqual(-2)
    expect(Math.max(...values)).toBeLessThanOrEqual(2)
  })

  it("kaimingNormal: relu gain over fanIn, explicit gain over fanOut", () => {
    const shape: [512, 2048] = [512, 2048]
    const gain = Math.SQRT2
    expectVarianceNear(
      init.kaimingNormal(shape, { nonlinearity: "relu", generator: init.generator(13) }).data,
      gain ** 2 / shape[1],
    )
    expectVarianceNear(
      init.kaimingNormal(shape, { fanMode: "fanOut", gain: 3, generator: init.generator(13) }).data,
      3 ** 2 / shape[0],
    )
  })

  it("xavierUniform and xavierNormal carry gain·sqrt(2/(fanIn+fanOut)) into the variance", () => {
    const shape: [512, 2048] = [512, 2048]
    // U(-b, b) with b = gain*sqrt(6/(fanIn+fanOut)) has variance gain^2*2/(fanIn+fanOut), which
    // is also the exact variance of the Xavier normal with the same gain.
    const variance = 2 ** 2 * 2 / (shape[0] + shape[1])
    expectVarianceNear(init.xavierUniform(shape, { gain: 2, generator: init.generator(14) }).data, variance)
    expectVarianceNear(init.xavierNormal(shape, { gain: 2, generator: init.generator(14) }).data, variance)
  })

  it("an explicit generator reproduces every pure draw, independent of ambient state", () => {
    const shape: [16, 8] = [16, 8]
    const g = init.generator(17)
    const draw = () => [
      ...init.truncNormal(shape, { generator: g }).data,
      ...init.kaimingNormal(shape, { nonlinearity: "relu", generator: g }).data,
      ...init.xavierUniform(shape, { generator: g }).data,
      ...init.xavierNormal(shape, { generator: g }).data,
    ]
    const first = draw()
    // Burn ambient draws: an explicit generator must not consult the seed counter.
    void init.uniform(shape)
    void init.normal(shape)
    expect(draw()).toEqual(first)
  })
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
    // Burn some ambient draws: the explicit generator must not care.
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

describe("Linear re-pointed at init.kaimingUniform_", () => {
  it("is bit-identical to the `1/sqrt(fanIn)` formula for a fixed seed", () => {
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
