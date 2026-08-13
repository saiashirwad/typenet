import { afterEach, describe, expect, it } from "vitest"
import { tensor } from "../src/factories.ts"
import { isLazy } from "../src/lazy.ts"
import { configure } from "../src/lazy.ts"
import { crossEntropy } from "../src/nn.ts"
import { SGD } from "../src/optim.ts"
import { Tensor } from "../src/tensor.ts"
import { testing } from "../src/testing.ts"
import { bothWays, expectClose } from "./helpers.ts"
import { makeXorNet } from "./xor-net.ts"

type AnyTensor = Tensor<any>

function expectSame(
  eager: AnyTensor,
  lazy: AnyTensor,
): void {
  expect(lazy.shape).toEqual(eager.shape)
  expect(lazy.toArray()).toEqual(eager.toArray())
}

function expectGradsClose(
  eager: AnyTensor[],
  lazy: AnyTensor[],
  tolerance = 1e-6,
): void {
  eager.forEach((p, i) => {
    expect(
      lazy[i]!.grad,
      `grad for param ${i}`,
    ).not.toBeNull()
    expectClose(p.grad!, lazy[i]!.grad!, tolerance)
  })
}

afterEach(() => {
  configure({ lazy: false })
})

describe("lazy mode", () => {
  it("is off by default and toggles via configure", () => {
    expect(isLazy()).toBe(false)
    configure({ lazy: true })
    expect(isLazy()).toBe(true)
    configure({ lazy: false })
    expect(isLazy()).toBe(false)
  })

  it("builds a graph without touching data", () => {
    configure({ lazy: true })
    const a = tensor([
      [1, 2, 3],
      [4, 5, 6],
    ])
    const b = a.add(tensor([10, 20, 30]))
    expect(testing.storageOf(b)).toBe("lazy")
    expect(b.shape).toEqual([2, 3])
    expect(b.toArray()).toEqual([
      [11, 22, 33],
      [14, 25, 36],
    ])
    expect(testing.storageOf(b)).toBe("materialized")
  })

  it("matches eager for binary broadcast", () => {
    const { eager, lazy } = bothWays(() => {
      const a = tensor([
        [1, 2, 3],
        [4, 5, 6],
      ])
      return a
        .mul(tensor([[10], [100]]))
        .add(tensor([1, 2, 3]))
        .sub(1)
        .div(2)
    })
    expectSame(eager, lazy)
  })

  it("matches eager for matmul", () => {
    const { eager, lazy } = bothWays(() =>
      tensor([
        [1, 2],
        [3, 4],
        [5, 6],
      ]).matmul(
        tensor([
          [1, 2, 3, 4],
          [5, 6, 7, 8],
        ]),
      )
    )
    expectSame(eager, lazy)
  })

  it("matches eager for reduces", () => {
    const { eager, lazy } = bothWays(() => {
      const a = tensor([
        [1, 2, 3],
        [4, 5, 6],
      ])
      return {
        dim: a.sum(0),
        keep: a.sum(1, true),
        all: a.sum(),
        mean: a.mean(1),
        max: a.max(0),
        maxAll: a.max(),
      }
    })
    expectSame(eager.dim, lazy.dim)
    expectSame(eager.keep, lazy.keep)
    expectSame(eager.all, lazy.all)
    expectSame(eager.mean, lazy.mean)
    expectSame(eager.max, lazy.max)
    expectSame(eager.maxAll, lazy.maxAll)
  })

  it("matches eager for a unary chain", () => {
    const { eager, lazy } = bothWays(() =>
      tensor([-1, 0.5, 2])
        .tanh()
        .exp()
        .log()
        .sigmoid()
        .sqrt()
        .abs()
        .pow(2)
        .neg()
        .relu()
    )
    expectSame(eager, lazy)
  })

  it("matches eager for view / permute", () => {
    const { eager, lazy } = bothWays(() => {
      const a = tensor([
        [1, 2, 3],
        [4, 5, 6],
      ])
      const v = a.view([3, 2]).transpose(0, 1)
      const p = a.unsqueeze(0).permute(1, 0, 2).squeeze()
      return { v, p }
    })
    expectSame(eager.v, lazy.v)
    expectSame(eager.p, lazy.p)
  })

  it("matches eager for cat", () => {
    const { eager, lazy } = bothWays(() =>
      Tensor.cat(
        tensor([
          [1, 2],
          [3, 4],
        ]),
        tensor([[5, 6]]),
        0,
      )
    )
    expectSame(eager, lazy)
  })

  it("matches eager for oneHot", () => {
    const { eager, lazy } = bothWays(() => tensor([0, 2, 1]).oneHot(3))
    expectSame(eager, lazy)
  })

  it("mixes lazy and eager tensors", () => {
    configure({ lazy: false })
    const eagerLeaf = tensor([1, 2, 3])
    configure({ lazy: true })
    const out = tensor([10, 20, 30]).add(eagerLeaf)
    expect(testing.storageOf(out)).toBe("lazy")
    expect(out.toArray()).toEqual([11, 22, 33])
  })

  it("forces at data and item()", () => {
    configure({ lazy: true })
    const a = tensor([2, 3]).mul(4)
    expect(a.data).toEqual(Float32Array.from([8, 12]))
    const s = tensor([5]).add(1)
    expect(s.item()).toBe(6)
  })

  it("matches eager for an XOR training step", () => {
    const step = () => {
      const { params, loss } = makeXorNet()
      const opt = new SGD(params, { lr: 0.5 })
      const l = loss()
      opt.zeroGrad()
      l.backward()
      opt.step()
      return { loss: l, params }
    }
    const { eager, lazy } = bothWays(step)
    expect(lazy.loss.item()).toBeCloseTo(
      eager.loss.item(),
      6,
    )
    eager.params.forEach((p, i) => expectSame(p, lazy.params[i]!))
  })

  it("matches eager gradients for a matmul/broadcast/reduce/unary chain", () => {
    const run = () => {
      const a = tensor([
        [0.5, -1, 2],
        [1.5, 0.25, -0.75],
      ]).requiresGrad()
      const b = tensor([
        [1, -2],
        [0.5, 0.5],
        [-1, 3],
      ]).requiresGrad()
      const c = tensor([2, -1]).requiresGrad()
      const loss = (a as AnyTensor)
        .matmul(b)
        .tanh()
        .mul(c)
        .add(0.5)
        .sigmoid()
        .sum(0)
        .log()
        .sum()
      loss.backward()
      return [a, b, c] as AnyTensor[]
    }
    const { eager, lazy } = bothWays(run)
    expectGradsClose(eager, lazy)
  })

  it("matches eager gradients through crossEntropy", () => {
    const run = () => {
      const logits = tensor([
        [2, 1, 0.1],
        [0.5, 1.5, -1],
        [-0.3, 0.8, 1.2],
        [1, -1, 0],
      ]).requiresGrad()
      const loss = crossEntropy(
        logits as any,
        tensor([0, 1, 2, 0]) as any,
      )
      loss.backward()
      return [logits] as AnyTensor[]
    }
    const { eager, lazy } = bothWays(run)
    expectGradsClose(eager, lazy)
    // array targets build the one-hot mask eagerly (a graph leaf)
    const withArrayTargets = () => {
      const logits = tensor([
        [2, 1, 0.1],
        [0.5, 1.5, -1],
      ]).requiresGrad()
      crossEntropy(logits as any, [0, 2]).backward()
      return [logits] as AnyTensor[]
    }
    const both = bothWays(withArrayTargets)
    expectGradsClose(both.eager, both.lazy)
  })

  it("matches eager gradients for an XOR training step", () => {
    const run = () => {
      const { params, loss } = makeXorNet()
      loss().backward()
      return params
    }
    const { eager, lazy } = bothWays(run)
    expectGradsClose(eager, lazy)
  })

  it("accumulates grads across repeated backward() calls", () => {
    const run = () => {
      const w = tensor([1, -2, 3]).requiresGrad()
      const x = tensor([2, 2, 2])
      const loss1 = (w as AnyTensor).mul(x).sum()
      loss1.backward()
      const loss2 = (w as AnyTensor).mul(3).sum()
      loss2.backward()
      return [w] as AnyTensor[]
    }
    const { eager, lazy } = bothWays(run)
    expectGradsClose(eager, lazy)
    expect(eager[0]!.grad!.toArray()).toEqual([5, 5, 5])
  })

  it("evaluates shared subexpressions once and materializes aliases", () => {
    configure({ lazy: true })
    const x = tensor([1, 2, 3]).requiresGrad()
    const z = (x as AnyTensor).mul(x)
    const y = z.add(z)
    const w = z.sum()
    const loss = y.sum().add(w)
    loss.backward()
    // y = 2x² + x² = 3x², dy/dx = 6x
    expect((x.grad as AnyTensor).toArray()).toEqual([
      6,
      12,
      18,
    ])
    // Aliasing: forcing the graph swapped the storage of every alias
    // of a shared node, so z (referenced by two parents plus sum) is
    // materialized exactly once and all aliases agree.
    expect(testing.storageOf(z)).toBe("materialized")
    expect(z.toArray()).toEqual([1, 4, 9])
    expect(y.toArray()).toEqual([2, 8, 18])
  })
})
