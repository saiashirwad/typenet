import { afterEach, describe, expect, it } from "vitest"
import { disableNative, isNativeAvailable, nativeDevice, preparedGraphCountNative, useNative } from "../src/backends/native.ts"
import { compile } from "../src/compile.ts"
import { tensor } from "../src/factories.ts"
import { configure } from "../src/lazy.ts"
import { crossEntropy } from "../src/nn.ts"
import { SGD } from "../src/optim.ts"
import { Tensor } from "../src/tensor.ts"
import { bothWays, expectAgree, expectClose } from "./helpers.ts"
import { makeXorNet } from "./xor-net.ts"

type AnyTensor = Tensor<any>

const available = isNativeAvailable()

afterEach(() => {
  configure({ lazy: false })
  disableNative()
})

describe.skipIf(!available)("native backend", () => {
  it("matches eager for binary broadcast", () => {
    expectAgree(() => {
      const a = tensor([
        [1, 2, 3],
        [4, 5, 6],
      ])
      return a
        .mul(tensor([[10], [100]]))
        .add(tensor([1, 2, 3]))
        .sub(1)
        .div(2)
    }, 1e-4)
  })

  it("matches eager for matmul (plain and batched)", () => {
    useNative()
    const { eager, lazy: native } = bothWays(() => {
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
    const { eager, lazy: native } = bothWays(() => {
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
    expectAgree(() =>
      tensor([-1, 0.5, 2])
        .tanh()
        .exp()
        .log()
        .sigmoid()
        .sqrt()
        .abs()
        .pow(2)
        .neg()
        .relu(), 1e-4)
  })

  it("matches eager for view / permute / transpose+view", () => {
    useNative()
    const { eager, lazy: native } = bothWays(() => {
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
    expectAgree(() =>
      Tensor.cat(
        tensor([
          [1, 2],
          [3, 4],
        ]),
        tensor([[5, 6]]),
        0,
      ), 1e-4)
  })

  it("matches eager for oneHot", () => {
    expectAgree(() => tensor([0, 2, 1]).oneHot(3), 1e-4)
  })

  it("matches eager for an XOR training step", () => {
    useNative()
    const step = () => {
      const { params, loss } = makeXorNet()
      const opt = new SGD(params, { lr: 0.5 })
      const l = loss()
      opt.zeroGrad()
      l.backward()
      opt.step()
      return { loss: l, params }
    }
    const { eager, lazy: native } = bothWays(step)
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
    const { eager, lazy: native } = bothWays(run)
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
      ]).requiresGrad()
      const loss = crossEntropy(
        logits as any,
        tensor([0, 1, 2, 0]) as any,
      )
      loss.backward()
      return [logits] as AnyTensor[]
    }
    const { eager, lazy: native } = bothWays(run)
    expectClose(eager[0]!.grad!, native[0]!.grad!)
  })

  it("matches eager gradients for an XOR training step", () => {
    useNative()
    const run = () => {
      const { params, loss } = makeXorNet()
      loss().backward()
      return params
    }
    const { eager, lazy: native } = bothWays(run)
    eager.forEach((p, i) => expectClose(p.grad!, native[i]!.grad!))
  })

  it("evaluates shared subexpressions once across multiple roots", () => {
    useNative()
    configure({ lazy: true })
    const x = tensor([1, 2, 3]).requiresGrad()
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

describe.skipIf(!available)("native f32 requirement", () => {
  it("throws on a float64 leaf instead of silently interpreting", () => {
    useNative()
    configure({ lazy: true })
    const wide = tensor([1, 2]).to("float64")
    const out = wide.add(1)
    expect(() => out.data).toThrow(
      /native backend requires float32 CPU leaves/,
    )
  })
})

describe.skipIf(!available)("eager native GEMM assist", () => {
  it("matches the JS matmul within f32 reassociation noise", () => {
    configure({ lazy: false })
    const a = Tensor.rand([80, 60]) as AnyTensor
    const b = Tensor.rand([60, 90]) as AnyTensor
    disableNative()
    const js = a.matmul(b)
    useNative()
    const accelerated = a.matmul(b)
    disableNative()
    const jd = js.data
    const ad = accelerated.data
    for (let i = 0; i < jd.length; i++) {
      expect(Math.abs(jd[i]! - ad[i]!)).toBeLessThan(1e-4)
    }
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

// Leak baselines: neither the eager-native GEMM assist nor the
// compile/dispose cycle should grow the native prepared-graph table.
// These don't move any gradient math, but a leak here would eventually
// exhaust native handles in any long-running process (e.g. a training
// loop), so they're asserted directly rather than left to intuition.
describe.skipIf(!available)("native backend leak baselines", () => {
  it("10,000 eager-native matmuls leave preparedGraphCount at baseline", () => {
    configure({ lazy: false })
    useNative()
    const before = preparedGraphCountNative()
    const a = Tensor.rand([8, 8]) as AnyTensor
    const b = Tensor.rand([8, 8]) as AnyTensor
    for (let i = 0; i < 10_000; i++) {
      // eager native GEMM assist path (src/eager.ts), never touches
      // prepareGraph/releaseGraph — this pins that invariant down.
      a.matmul(b).data
    }
    expect(preparedGraphCountNative()).toBe(before)
  })

  it("200 compile/dispose cycles return preparedGraphCount to baseline", () => {
    useNative()
    const before = preparedGraphCountNative()
    for (let i = 0; i < 200; i++) {
      // Distinct scale per cycle so each prepare allocates its own
      // handle rather than reusing one from a previous iteration.
      const scale = i + 1
      const fn = compile((x: AnyTensor) => x.mul(scale).sum())
      fn(tensor([1, 2, 3, 4]))
      expect(preparedGraphCountNative()).toBe(before + 1)
      fn.dispose()
      expect(preparedGraphCountNative()).toBe(before)
    }
    expect(preparedGraphCountNative()).toBe(before)
  })
})
