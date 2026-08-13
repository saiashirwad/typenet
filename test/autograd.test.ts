import { describe, expect, it } from "vitest"
import { noGrad } from "../src/autograd.ts"
import { tensor } from "../src/factories.ts"
import { testing } from "../src/testing.ts"

// Closed-form (analytic) gradient values only. Finite-difference
// gradient checking lives in gradcheck.test.ts, which runs every op
// kind in both eager and lazy mode at float32 precision plus a
// float64-precision case.
describe("autograd", () => {
  it("simple chain: ((x*y)+x)^2", () => {
    const x = tensor([1, 2]).requiresGrad()
    const y = tensor([3, 4]).requiresGrad()
    x.mul(y).add(x).pow(2).sum().backward()
    expect(x.grad!.toArray()).toEqual([32, 100])
    expect(y.grad!.toArray()).toEqual([8, 40])
  })

  it("gradients accumulate until zeroGrad", () => {
    const x = tensor([1, 1]).requiresGrad()
    x.mul(2).sum().backward()
    x.mul(2).sum().backward()
    expect(x.grad!.toArray()).toEqual([4, 4])
    x.zeroGrad()
    expect(x.grad).toBeNull()
  })

  it("broadcast ops reduce gradients correctly", () => {
    const p = tensor([
      [1, 2],
      [3, 4],
    ]).requiresGrad()
    const q = tensor([10, 20]).requiresGrad()
    p.mul(q).sum().backward()
    expect(p.grad!.toArray()).toEqual([
      [10, 20],
      [10, 20],
    ])
    expect(q.grad!.toArray()).toEqual([4, 6])
  })

  it("noGrad suppresses graph building", () => {
    const x = tensor([1, 2]).requiresGrad()
    const y = noGrad(() => x.mul(3))
    expect(testing.gradNodeOf(y)).toBeNull()
  })

  it("detach cuts the graph", () => {
    const x = tensor([2]).requiresGrad()
    const y = x.mul(3).detach().mul(2)
    expect(() => y.sum().backward()).toThrow()
  })

  it("backward on non-scalar requires explicit gradient", () => {
    const x = tensor([1, 2]).requiresGrad()
    expect(() => x.mul(2).backward()).toThrow(/scalar/)
  })
})
