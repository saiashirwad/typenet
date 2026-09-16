import { afterEach, describe, expect, it } from "vitest"
import { disableNative, isNativeAvailable, useNative } from "../src/backends/native.ts"
import { compile, printGraph } from "../src/compile.ts"
import { tensor } from "../src/factories.ts"
import { configure } from "../src/lazy.ts"
import { Tensor } from "../src/tensor.ts"
import { runOnSmallStack } from "./small-stack.ts"

type AnyTensor = Tensor<any>

const DEPTH = 20000

afterEach(() => {
  configure({ lazy: false })
  disableNative()
})

function chain(x: AnyTensor, depth: number): AnyTensor {
  let h = x
  for (let i = 0; i < depth; i++) h = h.mul(1.0001).add(0)
  return h
}

describe("deep graphs", () => {
  it("forces a chain far deeper than the JS stack", () => {
    configure({ lazy: true })
    const out = chain(tensor([1, 2]), DEPTH)
    const expected = 1.0001 ** DEPTH
    expect(out.get(0)).toBeCloseTo(expected, 2)
    expect(out.get(1)).toBeCloseTo(2 * expected, 2)
  })

  it("differentiates a deep chain", () => {
    configure({ lazy: true })
    const x = tensor([1, 2]).requiresGrad()
    chain(x as AnyTensor, DEPTH)
      .sum()
      .backward()
    const expected = 1.0001 ** DEPTH
    expect(x.grad!.get(0)).toBeCloseTo(expected, 2)
  })

  it("differentiates a deep chain eagerly too", () => {
    const x = tensor([1, 2]).requiresGrad()
    chain(x as AnyTensor, DEPTH)
      .sum()
      .backward()
    expect(x.grad!.get(0)).toBeCloseTo(1.0001 ** DEPTH, 2)
  })

  it("prints a deep graph", () => {
    configure({ lazy: true })
    const out = chain(tensor([1, 2]), DEPTH)
    const lines = printGraph(out).split("\n")
    expect(lines.length).toBeGreaterThan(DEPTH)
    expect(lines[lines.length - 1]).toContain("; root")
  })

  it("compiles and replays a deep chain", () => {
    const step = compile((x: Tensor<[2]>) => chain(x as AnyTensor, 2000).sum())
    const first = step(tensor([1, 2])).item()
    const second = step(tensor([1, 2])).item()
    expect(second).toBeCloseTo(first, 6)
    expect(first).toBeCloseTo(3 * 1.0001 ** 2000, 2)
  })
})

describe.skipIf(!isNativeAvailable())(
  "deep graphs, native",
  () => {
    it("evaluates a deep chain in one hop", () => {
      useNative()
      configure({ lazy: true })
      const out = chain(tensor([1, 2]), 4000)
      expect(out.get(0)).toBeCloseTo(1.0001 ** 4000, 3)
    })
  },
)

// These re-run the deep chains above as child processes with a tiny (256 KB) V8 stack, so graph
// construction, forcing and backward must stay iterative. A scenario asserts and exits non-zero.
describe("deep graphs, small stack", () => {
  // A cold vite-node process dominates the wall time, so give it headroom over vitest's 5s default.
  const SMALL_STACK_TIMEOUT = 60_000

  it("forces, and differentiates, the depth-20000 chain on a small stack", async () => {
    const { code, stdout, stderr } = await runOnSmallStack("deep-chain-20000")
    expect(code, `stdout:\n${stdout}\nstderr:\n${stderr}`).toBe(0)
  }, SMALL_STACK_TIMEOUT)

  it("forces, and differentiates, a depth-4000 chain of views and reductions on a small stack", async () => {
    const { code, stdout, stderr } = await runOnSmallStack("deep-chain-4000-mixed")
    expect(code, `stdout:\n${stdout}\nstderr:\n${stderr}`).toBe(0)
  }, SMALL_STACK_TIMEOUT)

  it("genuinely fails a recursive scenario", async () => {
    const { code } = await runOnSmallStack("recursive-overflow")
    expect(code).not.toBe(0)
  }, SMALL_STACK_TIMEOUT)
})
