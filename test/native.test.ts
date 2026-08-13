import { afterEach, describe, expect, it } from "vitest"
import { disableNative, isNativeAvailable, nativeDevice, useNative } from "../src/backends/native.ts"
import { crossEntropy } from "../src/nn.ts"
import { SGD } from "../src/optim.ts"
import { configure, Tensor, tensor } from "../src/tensor.ts"

type AnyTensor = Tensor<any>

const available = isNativeAvailable()

function bothWays<T>(fn: () => T): { eager: T; native: T } {
  configure({ lazy: false })
  const eager = fn()
  configure({ lazy: true })
  const native = fn()
  configure({ lazy: false })
  return { eager, native }
}

function expectClose(
  eager: AnyTensor,
  native: AnyTensor,
): void {
  expect(native.shape).toEqual(eager.shape)
  const a = eager.data
  const b = native.data
  expect(b.length).toBe(a.length)
  for (let i = 0; i < a.length; i++) {
    expect(Math.abs(a[i]! - b[i]!)).toBeLessThan(1e-4)
  }
}

afterEach(() => {
  configure({ lazy: false })
  disableNative()
})

describe.skipIf(!available)("native backend", () => {
  it("matches eager for binary broadcast", () => {
    useNative()
    const { eager, native } = bothWays(() => {
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
    expectClose(eager, native)
  })

  it("matches eager for matmul (plain and batched)", () => {
    useNative()
    const { eager, native } = bothWays(() => {
      const plain = tensor([
        [1, 2],
        [3, 4],
        [5, 6],
      ]).matmul(
        tensor([
          [1, 2, 3, 4],
          [5, 6, 7, 8],
        ]),
      )
      const batched = Tensor.stack([
        tensor([
          [1, 0],
          [0, 1],
        ]),
        tensor([
          [2, 0],
          [0, 2],
        ]),
      ]).matmul(
        tensor([
          [1, 2],
          [3, 4],
        ]),
      )
      return { plain, batched }
    })
    expectClose(eager.plain, native.plain)
    expectClose(eager.batched, native.batched)
  })

  it("matches eager for reduces", () => {
    useNative()
    const { eager, native } = bothWays(() => {
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
        argmax: a.argmax(1),
      }
    })
    expectClose(eager.dim, native.dim)
    expectClose(eager.keep, native.keep)
    expectClose(eager.all, native.all)
    expectClose(eager.mean, native.mean)
    expectClose(eager.max, native.max)
    expectClose(eager.maxAll, native.maxAll)
    expectClose(eager.argmax, native.argmax)
  })

  it("matches eager for a unary chain", () => {
    useNative()
    const { eager, native } = bothWays(() =>
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
    expectClose(eager, native)
  })

  it("matches eager for view / permute / narrow / broadcastTo", () => {
    useNative()
    const { eager, native } = bothWays(() => {
      const a = tensor([
        [1, 2, 3],
        [4, 5, 6],
      ])
      const v = a.view([3, 2]).transpose(0, 1)
      const p = a.unsqueeze(0).permute(1, 0, 2).squeeze()
      const n = (a as AnyTensor)
        .transpose(0, 1)
        .view([3, 2])
      return { v, p, n }
    })
    expectClose(eager.v, native.v)
    expectClose(eager.p, native.p)
    expectClose(eager.n, native.n)
  })

  it("matches eager for cat", () => {
    useNative()
    const { eager, native } = bothWays(() =>
      Tensor.cat(
        tensor([
          [1, 2],
          [3, 4],
        ]),
        tensor([[5, 6]]),
        0,
      )
    )
    expectClose(eager, native)
  })

  it("matches eager for oneHot", () => {
    useNative()
    const { eager, native } = bothWays(() => tensor([0, 2, 1]).oneHot(3))
    expectClose(eager, native)
  })

  it("matches eager for an XOR training step", () => {
    useNative()
    const step = () => {
      const x = tensor([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
      ])
      const y = tensor([[0], [1], [1], [0]])
      const w1 = tensor([
        [0.5, -0.5, 0.25, -0.25],
        [0.1, 0.2, -0.3, 0.4],
      ]).requires_grad()
      const b1 = tensor([
        0.1,
        -0.1,
        0.05,
        -0.05,
      ]).requires_grad()
      const w2 = tensor([
        [0.6],
        [-0.6],
        [0.3],
        [-0.3],
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
    const { eager, native } = bothWays(step)
    expect(native.loss.item()).toBeCloseTo(
      eager.loss.item(),
      4,
    )
    eager.params.forEach((p, i) => expectClose(p, native.params[i]!))
  })

  it("throws a clear error when native eval fails", () => {
    useNative()
    configure({ lazy: true })
    const bad = tensor([0, 5, 1]).oneHot(3)
    expect(() => bad.toArray()).toThrow(/native backend/)
  })

  it("matches eager gradients for a matmul/broadcast/reduce/unary chain", () => {
    useNative()
    const run = () => {
      const a = tensor([
        [0.5, -1, 2],
        [1.5, 0.25, -0.75],
      ]).requires_grad()
      const b = tensor([
        [1, -2],
        [0.5, 0.5],
        [-1, 3],
      ]).requires_grad()
      const c = tensor([2, -1]).requires_grad()
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
    const { eager, native } = bothWays(run)
    eager.forEach((p, i) => expectClose(p.grad!, native[i]!.grad!))
  })

  it("matches eager gradients through crossEntropy", () => {
    useNative()
    const run = () => {
      const logits = tensor([
        [2, 1, 0.1],
        [0.5, 1.5, -1],
        [-0.3, 0.8, 1.2],
        [1, -1, 0],
      ]).requires_grad()
      const loss = crossEntropy(
        logits as any,
        tensor([0, 1, 2, 0]) as any,
      )
      loss.backward()
      return [logits] as AnyTensor[]
    }
    const { eager, native } = bothWays(run)
    expectClose(eager[0]!.grad!, native[0]!.grad!)
  })

  it("matches eager gradients for an XOR training step", () => {
    useNative()
    const run = () => {
      const x = tensor([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
      ])
      const y = tensor([[0], [1], [1], [0]])
      const w1 = tensor([
        [0.5, -0.5, 0.25, -0.25],
        [0.1, 0.2, -0.3, 0.4],
      ]).requires_grad()
      const b1 = tensor([
        0.1,
        -0.1,
        0.05,
        -0.05,
      ]).requires_grad()
      const w2 = tensor([
        [0.6],
        [-0.6],
        [0.3],
        [-0.3],
      ]).requires_grad()
      const b2 = tensor([0.2]).requires_grad()
      const h = x.matmul(w1).add(b1).tanh()
      const out = h.matmul(w2).add(b2).sigmoid()
      out.sub(y).pow(2).mean().backward()
      return [w1, b1, w2, b2] as AnyTensor[]
    }
    const { eager, native } = bothWays(run)
    eager.forEach((p, i) => expectClose(p.grad!, native[i]!.grad!))
  })

  it("evaluates shared subexpressions once across multiple roots", () => {
    useNative()
    configure({ lazy: true })
    const x = tensor([1, 2, 3]).requires_grad()
    const z = (x as AnyTensor).mul(x)
    const y = z.add(z)
    const w = z.sum()
    y.sum().add(w).backward()
    // loss = 3 * sum(x²) → d/dx = 6x; the forward x*x must be one
    // node in the serialized graph (dedupe) and every alias must see
    // the same materialized values.
    expectClose(
      tensor([6, 12, 18]) as AnyTensor,
      x.grad as AnyTensor,
    )
    expectClose(
      tensor([1, 4, 9]) as AnyTensor,
      z as AnyTensor,
    )
    expectClose(
      tensor([2, 8, 18]) as AnyTensor,
      y as AnyTensor,
    )
  })
})

describe("native backend availability", () => {
  it("reports availability without affecting lazy mode", () => {
    expect(isNativeAvailable()).toBe(available)
    if (available) {
      expect(["metal", "cpu"]).toContain(nativeDevice())
      useNative()
      configure({ lazy: true })
      expect(tensor([1, 2]).add(1).toArray()).toEqual([
        2,
        3,
      ])
    } else {
      expect(() => useNative()).toThrow(/build:native/)
    }
  })
})
