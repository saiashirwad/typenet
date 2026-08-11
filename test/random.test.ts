// uniform() / normal() are graph nodes, not fixed data: a compiled
// function redraws them on every call. These tests pin the three
// properties that matters for: fresh values per evaluation, stable
// values within one evaluation, and reproducibility from a seed.

import { afterEach, describe, expect, it } from "vitest"
import {
  Tensor,
  compile,
  configure,
  normal,
  printGraph,
  tensor,
  uniform
} from "../src/tensor.ts"
import {
  disableNative,
  isNativeAvailable,
  useNative
} from "../src/backends/native.ts"

type AnyTensor = Tensor<any, any>

afterEach(() => {
  configure({ lazy: false })
  disableNative()
})

function stats(t: AnyTensor): { mean: number; sd: number } {
  const d = t.data
  let sum = 0
  for (const x of d) sum += x
  const mean = sum / d.length
  let variance = 0
  for (const x of d) variance += (x - mean) ** 2
  return { mean, sd: Math.sqrt(variance / d.length) }
}

describe("uniform", () => {
  it("fills the unit interval", () => {
    const u = uniform([4096]) as AnyTensor
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
    const a = uniform([64]) as AnyTensor
    const b = uniform([64]) as AnyTensor
    expect(Array.from(a.data)).not.toEqual(Array.from(b.data))
  })

  it("repeats exactly for a given seed", () => {
    configure({ seed: 7 })
    const first = Array.from((uniform([32]) as AnyTensor).data)
    configure({ seed: 7 })
    const second = Array.from((uniform([32]) as AnyTensor).data)
    expect(second).toEqual(first)
  })

  it("holds one value per tensor once forced", () => {
    configure({ lazy: true })
    const u = uniform([16]) as AnyTensor
    const first = Array.from(u.data)
    expect(Array.from(u.data)).toEqual(first)
  })
})

describe("normal", () => {
  it("is standard normal", () => {
    const n = normal([8192]) as AnyTensor
    const { mean, sd } = stats(n)
    expect(Math.abs(mean)).toBeLessThan(0.05)
    expect(sd).toBeCloseTo(1, 1)
  })

  it("produces no NaNs (log(0) is excluded)", () => {
    for (const x of (normal([4096]) as AnyTensor).data)
      expect(Number.isFinite(x)).toBe(true)
  })
})

describe("random nodes in a graph", () => {
  it("print as sources with a stream id", () => {
    configure({ lazy: true })
    const out = (uniform([4]) as AnyTensor).add(1)
    expect(printGraph(out)).toMatch(/random\.uniform\(\) \{stream=\d+\}/)
  })

  it("stop gradients", () => {
    const u = uniform([4]) as AnyTensor
    expect(u.needsGrad).toBe(false)
    expect(u.gradNode).toBeNull()
  })

  it("redraw on every call of a compiled function", () => {
    const step = compile((x: Tensor<[64]>) =>
      (x as AnyTensor).add(uniform([64]))
    )
    const zeros = Tensor.zeros([64])
    const first = Array.from(step(zeros).data)
    const second = Array.from(step(zeros).data)
    expect(second).not.toEqual(first)
    // still uniform, not garbage
    for (const x of second) {
      expect(x).toBeGreaterThanOrEqual(0)
      expect(x).toBeLessThan(1)
    }
  })

  it("keep one value per evaluation, however many times it is read", () => {
    // The mask is read twice in the same graph; both reads must see the
    // same draw, or the "gate" would not be a gate at all.
    const step = compile((x: Tensor<[256]>) => {
      const mask = (uniform([256]) as AnyTensor).lt(0.5)
      return mask.sub(mask).abs().sum()
    })
    expect(step(Tensor.zeros([256])).item()).toBe(0)
  })

  it("gate roughly half the entries, differently each call", () => {
    const step = compile(() =>
      (uniform([4096]) as AnyTensor).lt(0.5).sum()
    )
    const counts = [step(), step(), step()].map(t => t.item())
    for (const c of counts) {
      expect(c).toBeGreaterThan(1900)
      expect(c).toBeLessThan(2200)
    }
    expect(new Set(counts).size).toBeGreaterThan(1)
  })

  it("survives a graph reset in the interpreter path", () => {
    configure({ lazy: true, seed: 3 })
    const a = Array.from((uniform([32]) as AnyTensor).data)
    configure({ lazy: true, seed: 3 })
    const b = Array.from((uniform([32]) as AnyTensor).data)
    expect(b).toEqual(a)
  })
})

describe.skipIf(!isNativeAvailable())("random nodes, native", () => {
  it("match the interpreter draw for draw", () => {
    // uniform() is pure integer mixing on both sides, so the values are
    // identical rather than merely similarly distributed.
    configure({ lazy: true, seed: 99 })
    const interpreted = Array.from(
      (uniform([1024]) as AnyTensor).data
    )
    useNative()
    configure({ lazy: true, seed: 99 })
    const native = Array.from((uniform([1024]) as AnyTensor).data)
    expect(native).toEqual(interpreted)
  })

  it("match for normal within f32 rounding", () => {
    configure({ lazy: true, seed: 41 })
    const interpreted = (normal([1024]) as AnyTensor).data
    useNative()
    configure({ lazy: true, seed: 41 })
    const native = (normal([1024]) as AnyTensor).data
    let worst = 0
    for (let i = 0; i < interpreted.length; i++)
      worst = Math.max(
        worst,
        Math.abs(interpreted[i]! - native[i]!)
      )
    expect(worst).toBeLessThan(1e-5)
  })

  it("redraw per call in a compiled native step", () => {
    useNative()
    const step = compile((x: Tensor<[4096]>) =>
      (x as AnyTensor).add(uniform([4096])).sum()
    )
    const zeros = Tensor.zeros([4096])
    const first = step(zeros).item()
    const second = step(zeros).item()
    expect(first).not.toBe(second)
    // mean 0.5 over 4096 draws
    expect(first / 4096).toBeCloseTo(0.5, 1)
    expect(second / 4096).toBeCloseTo(0.5, 1)
  })
})
