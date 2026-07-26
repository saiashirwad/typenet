import { afterEach, describe, expect, it } from "vitest"
import {
  Tensor,
  tensor,
  configure,
  isLazy
} from "../src/tensor.ts"
import { SGD } from "../src/optim.ts"

type AnyTensor = Tensor<any, any>

function bothWays<T>(fn: () => T): {
  eager: T
  lazy: T
} {
  configure({ lazy: false })
  const eager = fn()
  configure({ lazy: true })
  const lazy = fn()
  configure({ lazy: false })
  return { eager, lazy }
}

function expectSame(
  eager: AnyTensor,
  lazy: AnyTensor
): void {
  expect(lazy.shape).toEqual(eager.shape)
  expect(lazy.toArray()).toEqual(eager.toArray())
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
      [4, 5, 6]
    ])
    const b = a.add(tensor([10, 20, 30]))
    expect(b._storage.kind).toBe("lazy")
    expect(b.shape).toEqual([2, 3])
    expect(b.toArray()).toEqual([
      [11, 22, 33],
      [14, 25, 36]
    ])
    expect(b._storage.kind).toBe("cpu")
  })

  it("matches eager for binary broadcast", () => {
    const { eager, lazy } = bothWays(() => {
      const a = tensor([
        [1, 2, 3],
        [4, 5, 6]
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
        [5, 6]
      ]).matmul(
        tensor([
          [1, 2, 3, 4],
          [5, 6, 7, 8]
        ])
      )
    )
    expectSame(eager, lazy)
  })

  it("matches eager for reduces", () => {
    const { eager, lazy } = bothWays(() => {
      const a = tensor([
        [1, 2, 3],
        [4, 5, 6]
      ])
      return {
        dim: a.sum(0),
        keep: a.sum(1, true),
        all: a.sum(),
        mean: a.mean(1),
        max: a.max(0),
        maxAll: a.max()
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
        [4, 5, 6]
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
          [3, 4]
        ]),
        tensor([[5, 6]]),
        0
      )
    )
    expectSame(eager, lazy)
  })

  it("matches eager for oneHot", () => {
    const { eager, lazy } = bothWays(() =>
      tensor([0, 2, 1]).oneHot(3)
    )
    expectSame(eager, lazy)
  })

  it("mixes lazy and eager tensors", () => {
    configure({ lazy: false })
    const eagerLeaf = tensor([1, 2, 3])
    configure({ lazy: true })
    const out = tensor([10, 20, 30]).add(eagerLeaf)
    expect(out._storage.kind).toBe("lazy")
    expect(out.toArray()).toEqual([11, 22, 33])
  })

  it("forces at read() and item()", async () => {
    configure({ lazy: true })
    const a = tensor([2, 3]).mul(4)
    expect(await a.read()).toEqual(
      Float32Array.from([8, 12])
    )
    const s = tensor([5]).add(1)
    expect(s.item()).toBe(6)
  })

  it("matches eager for an XOR training step", () => {
    const step = () => {
      const x = tensor([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
      ])
      const y = tensor([[0], [1], [1], [0]])
      const w1 = tensor([
        [0.5, -0.5, 0.25, -0.25],
        [0.1, 0.2, -0.3, 0.4]
      ]).requires_grad()
      const b1 = tensor([
        0.1, -0.1, 0.05, -0.05
      ]).requires_grad()
      const w2 = tensor([
        [0.6],
        [-0.6],
        [0.3],
        [-0.3]
      ]).requires_grad()
      const b2 = tensor([0.2]).requires_grad()
      const params = [w1, b1, w2, b2] as AnyTensor[]
      const opt = new SGD(params, { lr: 0.5 })
      const h = x.matmul(w1).add(b1).tanh()
      const out = h.matmul(w2).add(b2).sigmoid()
      const loss = out.sub(y).pow(2).mean()
      opt.zeroGrad()
      loss.backward()
      opt.step()
      return { loss, params }
    }
    const { eager, lazy } = bothWays(step)
    expect(lazy.loss.item()).toBeCloseTo(
      eager.loss.item(),
      6
    )
    eager.params.forEach((p, i) =>
      expectSame(p, lazy.params[i]!)
    )
  })
})
