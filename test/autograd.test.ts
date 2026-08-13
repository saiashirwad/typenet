import { describe, expect, it } from "vitest"
import { noGrad } from "../src/autograd.ts"
import { tensor } from "../src/factories.ts"
import { Tensor } from "../src/tensor.ts"
import { testing } from "../src/testing.ts"

type AnyTensor = Tensor<any>

function gradCheck(
  f: (...inputs: AnyTensor[]) => AnyTensor,
  values: number[][],
  shapes: number[][],
  eps = 1e-4,
  tol = 1e-2,
): void {
  const make = () =>
    values.map((v, i) => {
      const t = Tensor.zeros(shapes[i]! as [number]).to(
        "float64",
      ) as AnyTensor
      ;(t.data as Float64Array).set(v)
      return t.requiresGrad()
    })

  const inputs = make()
  const out = f(...inputs).sum()
  out.backward()

  values.forEach((v, i) => {
    for (let j = 0; j < v.length; j++) {
      const plus = make()
      const minus = make()
      ;(plus[i]!.data as Float64Array)[j]! += eps
      ;(minus[i]!.data as Float64Array)[j]! -= eps
      const [up, down] = noGrad(() => [
        f(...plus)
          .sum()
          .item(),
        f(...minus)
          .sum()
          .item(),
      ])
      const numeric = (up! - down!) / (2 * eps)
      const analytic = inputs[i]!.grad!.data[j]!
      expect(
        Math.abs(numeric - analytic),
        `input ${i} elem ${j}: numeric ${numeric} vs autograd ${analytic}`,
      ).toBeLessThan(tol)
    }
  })
}

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

  it("matmul gradients (numerical)", () => {
    gradCheck(
      (a, b) => a.matmul(b),
      [
        [1, 2, 3, 4, 5, 6],
        [0.5, -1, 2, 1.5, 0, 1],
      ],
      [
        [2, 3],
        [3, 2],
      ],
    )
  })

  it("batched matmul gradients (numerical)", () => {
    gradCheck(
      (a, b) => a.matmul(b),
      [
        Array.from({ length: 12 }, (_, i) => i * 0.3 - 2),
        [0.5, -1, 2, 1.5],
      ],
      [
        [3, 2, 2],
        [2, 2],
      ],
    )
  })

  it("div / exp / log / sqrt gradients (numerical)", () => {
    gradCheck(
      (a, b) => a.div(b).exp().add(a.log()).add(a.sqrt()),
      [
        [1, 2, 3],
        [4, 5, 6],
      ],
      [[3], [3]],
    )
  })

  it("activations gradients (numerical)", () => {
    gradCheck(
      a =>
        a
          .relu()
          .add(a.sigmoid())
          .add(a.tanh())
          .add(a.leakyRelu(0.2)),
      [[-2, -0.5, 0.5, 2]],
      [[4]],
    )
  })

  it("softmax / logSoftmax gradients (numerical)", () => {
    gradCheck(
      a => a.view([2, 2]).softmax(1).pow(2),
      [[1, 2, 3, 4]],
      [[4]],
    )
    gradCheck(
      a => a.view([2, 2]).logSoftmax(1).mul(0.5),
      [[1, 2, 3, 4]],
      [[4]],
    )
  })

  it("reshape / transpose / cat / stack gradients (numerical)", () => {
    gradCheck(
      a =>
        a
          .view([2, 3])
          .transpose(0, 1)
          .mul(tensor([[1], [2], [3]]) as any),
      [[1, 2, 3, 4, 5, 6]],
      [[6]],
    )
    gradCheck(
      (a, b) =>
        Tensor.cat(
          a.view([1, 2]) as any,
          b.view([1, 2]) as any,
          0,
        ).pow(2),
      [
        [1, 2],
        [3, 4],
      ],
      [[2], [2]],
    )
    gradCheck(
      (a, b) => Tensor.stack([a, b] as any, 1 as any).pow(3),
      [
        [1, 2],
        [3, 4],
      ],
      [[2], [2]],
    )
  })

  it("mean over dim gradients (numerical)", () => {
    gradCheck(
      a => a.view([2, 3]).mean(1).pow(2),
      [[1, 2, 3, 4, 5, 6]],
      [[6]],
    )
  })
})
