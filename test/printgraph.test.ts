import { afterEach, describe, expect, it } from "vitest"
import { configure, printGraph, Tensor } from "../src/tensor.ts"

afterEach(() => {
  configure({ lazy: false })
})

describe("named + printGraph", () => {
  it("prints a small chain with names, shapes, and dtypes", () => {
    configure({ lazy: true })
    const x = Tensor.rand([2, 3]).named("x")
    const w = Tensor.rand([3, 4]).named("w")
    const h = x.matmul(w).named("h")
    const loss = h.relu().sum().named("loss")
    const out = printGraph(loss)
    expect(out).toBe(
      [
        "x    = leaf [2, 3] float32",
        "w    = leaf [3, 4] float32",
        "h    = matmul(x, w) [2, 4] float32",
        "%3   = relu(h) [2, 4] float32",
        "loss = reduceAll.sum(%3) [] float32 ; root",
      ].join("\n"),
    )
  })

  it("gives unnamed nodes stable auto ids in traversal order", () => {
    configure({ lazy: true })
    const a = Tensor.rand([4])
    const b = a.add(1).mul(a)
    const out = printGraph(b).split("\n")
    expect(out[0]).toBe("%0 = leaf [4] float32")
    expect(out[1]).toBe("%1 = leaf [] float32")
    expect(out[2]).toBe("%2 = add(%0, %1) [4] float32")
    expect(out[3]).toBe(
      "%3 = mul(%2, %0) [4] float32 ; root",
    )
    expect(printGraph(b)).toBe(out.join("\n"))
  })

  it("shows reduce attributes", () => {
    configure({ lazy: true })
    const m = Tensor.rand([2, 3])
    const s = m.sum(1, true)
    expect(printGraph(s)).toBe(
      [
        "%0 = leaf [2, 3] float32",
        "%1 = reduce.sum(%0) {dim=1, keepdim=true} [2, 1] float32 ; root",
      ].join("\n"),
    )
  })

  it("prints multi-root graphs with shared subgraphs once", () => {
    configure({ lazy: true })
    const x = Tensor.rand([2, 2]).named("x")
    const shared = x.relu().named("shared")
    const p = shared.exp()
    const q = shared.neg()
    const out = printGraph([p, q])
    const lines = out.split("\n")
    expect(
      lines.filter(l => l.startsWith("x ")).length,
    ).toBe(1)
    expect(
      lines.filter(l => l.startsWith("shared ")).length,
    ).toBe(1)
    expect(lines).toHaveLength(4)
    expect(out).toContain("exp(shared) [2, 2] float32")
    expect(out).toContain("neg(shared) [2, 2] float32")
    expect(lines[2]).toContain("; root")
    expect(lines[3]).toContain("; root")
    expect(lines[1]).not.toContain("; root")
  })

  it("prints an eager tensor as a single leaf line", () => {
    const t = Tensor.rand([2, 2]).named("e")
    expect(printGraph(t)).toBe(
      "e = leaf [2, 2] float32 ; root",
    )
  })

  it("prints lazy backward results (grads are materialized leaves)", () => {
    configure({ lazy: true })
    const x = Tensor.rand([2, 2]).requiresGrad().named("x")
    const loss = x.mul(x).sum().named("loss")
    const fwd = printGraph(loss)
    expect(fwd).toContain("x    = leaf [2, 2] float32")
    expect(fwd).toContain("mul(x, x) [2, 2] float32")
    expect(fwd).toContain("loss = reduceAll.sum(")
    loss.backward()
    expect(printGraph(x.grad!)).toBe(
      "%0 = leaf [2, 2] float32 ; root",
    )
  })

  it("name survives forcing (tensor identity is kept)", () => {
    configure({ lazy: true })
    const h = Tensor.rand([2]).exp().named("h")
    h.data // force
    expect(printGraph(h)).toBe(
      "h = leaf [2] float32 ; root",
    )
  })
})
