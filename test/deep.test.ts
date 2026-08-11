import { afterEach, describe, expect, it } from "vitest"
import {
  Tensor,
  compile,
  configure,
  printGraph,
  tensor
} from "../src/tensor.ts"
import {
  isNativeAvailable,
  useNative,
  disableNative
} from "../src/backends/native.ts"

type AnyTensor = Tensor<any, any>

// Deep graphs are the shape a rolled-out cellular automaton takes: one
// long chain of ops, differentiated end to end. Every graph walker has
// to survive that depth, so these tests build chains far longer than
// recursion would tolerate.
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
    // 1.0001 applied DEPTH times, twice per iteration is one mul each.
    const expected = 1.0001 ** DEPTH
    expect(out.get(0)).toBeCloseTo(expected, 2)
    expect(out.get(1)).toBeCloseTo(2 * expected, 2)
  })

  it("differentiates a deep chain", () => {
    configure({ lazy: true })
    const x = tensor([1, 2]).requires_grad()
    chain(x as AnyTensor, DEPTH)
      .sum()
      .backward()
    const expected = 1.0001 ** DEPTH
    expect(x.grad!.get(0)).toBeCloseTo(expected, 2)
  })

  it("differentiates a deep chain eagerly too", () => {
    const x = tensor([1, 2]).requires_grad()
    chain(x as AnyTensor, DEPTH)
      .sum()
      .backward()
    expect(x.grad!.get(0)).toBeCloseTo(1.0001 ** DEPTH, 2)
  })

  it("prints a deep graph", () => {
    configure({ lazy: true })
    const out = chain(tensor([1, 2]), DEPTH)
    const lines = printGraph(out).split("\n")
    // one leaf, one scalar per mul/add, and one node per op
    expect(lines.length).toBeGreaterThan(DEPTH)
    expect(lines[lines.length - 1]).toContain("; root")
  })

  it("compiles and replays a deep chain", () => {
    const step = compile((x: Tensor<[2]>) =>
      chain(x as AnyTensor, 2000).sum()
    )
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
  }
)
