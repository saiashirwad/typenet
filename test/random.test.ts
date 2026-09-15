import { describe, expect, it } from "vitest"
import { isNativeAvailable } from "../src/backends/native.ts"
import { compile, printGraph } from "../src/compile.ts"
import { lazy, withContext } from "../src/context.ts"
import { rand, randn } from "../src/factories.ts"
import { Tensor } from "../src/tensor.ts"
import { testing } from "../src/testing.ts"

type AnyTensor = Tensor<any>

function stats(t: AnyTensor): { mean: number; sd: number } {
  const d = t.data
  let sum = 0
  for (const x of d) sum += x
  const mean = sum / d.length
  let variance = 0
  for (const x of d) variance += (x - mean) ** 2
  return { mean, sd: Math.sqrt(variance / d.length) }
}

describe("rand({ resample: \"perCall\" })", () => {
  it("fills the unit interval", () => {
    const u = rand([4096], { resample: "perCall" }) as AnyTensor
    const { mean, sd } = stats(u)
    expect(mean).toBeCloseTo(0.5, 1)
    // sd of U(0,1) is 1/sqrt(12) = 0.2887
    expect(sd).toBeCloseTo(0.2887, 2)
    for (const x of u.data) {
      expect(x).toBeGreaterThanOrEqual(0)
      expect(x).toBeLessThan(1)
    }
  })

  it("draws different values for different streams", () => {
    const a = rand([64], { resample: "perCall" }) as AnyTensor
    const b = rand([64], { resample: "perCall" }) as AnyTensor
    expect(Array.from(a.data)).not.toEqual(
      Array.from(b.data),
    )
  })

  it("repeats exactly for a given seed", () => {
    const draw = () =>
      withContext(
        { seed: 7 },
        () => Array.from((rand([32], { resample: "perCall" }) as AnyTensor).data),
      )
    expect(draw()).toEqual(draw())
  })

  it("holds one value per tensor once forced", () => {
    lazy(() => {
      const u = rand([16], { resample: "perCall" }) as AnyTensor
      const first = Array.from(u.data)
      expect(Array.from(u.data)).toEqual(first)
    })
  })

  it("matches a checked-in reference", () => {
    // Golden values (1024 draws, seed 99, lazy interpreter): if these
    // move, the RNG changed, not just a name.
    const data = withContext(
      { lazy: true, seed: 99 },
      () => Array.from((rand([1024], { resample: "perCall" }) as AnyTensor).data),
    )
    expect(data.slice(0, 8)).toEqual([
      0.7263181209564209,
      0.5634088516235352,
      0.1625751256942749,
      0.40070807933807373,
      0.8495884537696838,
      0.3222121000289917,
      0.25208210945129395,
      0.6284887790679932,
    ])
    expect(data.slice(-8)).toEqual([
      0.07920491695404053,
      0.7265327572822571,
      0.8621742129325867,
      0.5839124321937561,
      0.33841127157211304,
      0.7362820506095886,
      0.8269493579864502,
      0.007852375507354736,
    ])
    expect(data.reduce((a, b) => a + b, 0)).toBe(507.92976474761963)
  })
})

describe("randn({ resample: \"perCall\" })", () => {
  it("is standard normal", () => {
    const n = randn([8192], { resample: "perCall" }) as AnyTensor
    const { mean, sd } = stats(n)
    expect(Math.abs(mean)).toBeLessThan(0.05)
    expect(sd).toBeCloseTo(1, 1)
  })

  it("produces no NaNs (log(0) is excluded)", () => {
    for (const x of (randn([4096], { resample: "perCall" }) as AnyTensor).data) {
      expect(Number.isFinite(x)).toBe(true)
    }
  })
})

describe("random nodes in a graph", () => {
  it("print as sources with a stream id", () => {
    lazy(() => {
      const out = (rand([4], { resample: "perCall" }) as AnyTensor).add(1)
      expect(printGraph(out)).toMatch(
        /random\.uniform\(\) \{stream=\d+\}/,
      )
    })
  })

  it("stop gradients", () => {
    const u = rand([4], { resample: "perCall" }) as AnyTensor
    expect(u.needsGrad).toBe(false)
    expect(testing.gradNodeOf(u)).toBeNull()
  })

  it("redraw on every call of a compiled function", () => {
    const step = compile((x: Tensor<[64]>) => (x as AnyTensor).add(rand([64], { resample: "perCall" })))
    const zeros = Tensor.zeros([64])
    const first = Array.from(step(zeros).data)
    const second = Array.from(step(zeros).data)
    expect(second).not.toEqual(first)
    for (const x of second) {
      expect(x).toBeGreaterThanOrEqual(0)
      expect(x).toBeLessThan(1)
    }
  })

  it("keep one value per evaluation, however many times it is read", () => {
    // The mask is read twice in the same graph; both reads must see the
    // same draw, or the "gate" would not be a gate at all.
    const step = compile((x: Tensor<[256]>) => {
      const mask = (rand([256], { resample: "perCall" }) as AnyTensor).lt(0.5)
      return mask.sub(mask).abs().sum()
    })
    expect(step(Tensor.zeros([256])).item()).toBe(0)
  })

  it("gate roughly half the entries, differently each call", () => {
    const step = compile(() => (rand([4096], { resample: "perCall" }) as AnyTensor).lt(0.5).sum())
    const counts = [step(), step(), step()].map(t => t.item())
    for (const c of counts) {
      expect(c).toBeGreaterThan(1900)
      expect(c).toBeLessThan(2200)
    }
    expect(new Set(counts).size).toBeGreaterThan(1)
  })

  it("survives a graph reset in the interpreter path", () => {
    const draw = () =>
      withContext(
        { lazy: true, seed: 3 },
        () => Array.from((rand([32], { resample: "perCall" }) as AnyTensor).data),
      )
    expect(draw()).toEqual(draw())
  })
})

describe.skipIf(!isNativeAvailable())(
  "random nodes, native",
  () => {
    it("match the interpreter draw for draw", () => {
      // The uniform draw is pure integer mixing on both sides, so the
      // values are identical rather than merely similarly distributed.
      const interpreted = withContext(
        { lazy: true, seed: 99 },
        () => Array.from((rand([1024], { resample: "perCall" }) as AnyTensor).data),
      )
      const native = withContext(
        { lazy: true, seed: 99, native: true },
        () => Array.from((rand([1024], { resample: "perCall" }) as AnyTensor).data),
      )
      expect(native).toEqual(interpreted)
    })

    it("match for normal within f32 rounding", () => {
      const interpreted = withContext(
        { lazy: true, seed: 41 },
        () => (randn([1024], { resample: "perCall" }) as AnyTensor).data,
      )
      const native = withContext(
        { lazy: true, seed: 41, native: true },
        () => (randn([1024], { resample: "perCall" }) as AnyTensor).data,
      )
      let worst = 0
      for (let i = 0; i < interpreted.length; i++) {
        worst = Math.max(
          worst,
          Math.abs(interpreted[i]! - native[i]!),
        )
      }
      expect(worst).toBeLessThan(1e-5)
    })

    it("redraw per call in a compiled native step", () => {
      withContext({ native: true }, () => {
        const step = compile((x: Tensor<[4096]>) => (x as AnyTensor).add(rand([4096], { resample: "perCall" })).sum())
        const zeros = Tensor.zeros([4096])
        const first = step(zeros).item()
        const second = step(zeros).item()
        expect(first).not.toBe(second)
        expect(first / 4096).toBeCloseTo(0.5, 1)
        expect(second / 4096).toBeCloseTo(0.5, 1)
      })
    })
  },
)

describe("rand/randn are seeded", () => {
  it("replay identically under the same seed", () => {
    const { a, b } = withContext({ seed: 123 }, () => ({
      a: Tensor.rand([2, 3]).toArray(),
      b: Tensor.randn([4]).toArray(),
    }))
    withContext({ seed: 123 }, () => {
      expect(Tensor.rand([2, 3]).toArray()).toEqual(a)
      expect(Tensor.randn([4]).toArray()).toEqual(b)
    })
    withContext({ seed: 124 }, () => {
      expect(Tensor.rand([2, 3]).toArray()).not.toEqual(a)
    })
  })

  it("differ call to call under one seed", () => {
    withContext({ seed: 7 }, () => {
      expect(Tensor.rand([8]).toArray()).not.toEqual(
        Tensor.rand([8]).toArray(),
      )
    })
  })

  it("resample: \"once\" (default) draws a fixed leaf under compile(); \"perCall\" redraws", () => {
    // compile() traces once under lazy semantics regardless of ambient
    // mode: "once" is baked in at trace time, "perCall" is a graph node
    // re-evaluated on every call.
    const once = compile(() => randn([64]).add(0))
    expect(Array.from(once().data)).toEqual(Array.from(once().data))

    const perCall = compile(() => randn([64], { resample: "perCall" }).add(0))
    expect(Array.from(perCall().data)).not.toEqual(
      Array.from(perCall().data),
    )
  })
})
