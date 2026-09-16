import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { disableNative, isNativeAvailable, nativeCounters, useNative } from "../src/backends/native.ts"
import { compile, printGraph } from "../src/compile.ts"
import { jsCounters, resetJsCounters } from "../src/counters.ts"
import { evalMatmulEager } from "../src/eager.ts"
import { sumTo } from "../src/ir.ts"
import { configure, serializeLazyGraph } from "../src/lazy.ts"
import { crossEntropy as composedCrossEntropy } from "../src/nn/index.ts"
import {
  type AnyTensor,
  crossEntropy,
  dropout,
  fromFlat,
  gatherRows,
  gelu,
  layerNorm,
  logSumExp,
  rmsNorm,
  silu,
  softmax,
  Tensor,
} from "../src/tensor.ts"
import { bothWays, expectClose, expectExact, sample } from "./helpers.ts"

afterEach(() => {
  configure({ lazy: false })
  disableNative()
})

// Each semantic kernel is checked against the composition it replaces, written independently here
// rather than against a magic constant nobody can re-derive.

describe("semantic ops match the composition they replace", () => {
  it("gelu == 0.5x(1 + tanh(sqrt(2/pi)(x + 0.044715 x^3)))", () => {
    const x = sample(24, [4, 6])
    const c = Math.sqrt(2 / Math.PI)
    const composed = x
      .mul(0.5)
      .mul(
        x
          .add(x.pow(3).mul(0.044715))
          .mul(c)
          .tanh()
          .add(1) as AnyTensor,
      )
    expectClose(gelu(x) as AnyTensor, composed, 1e-6)
  })

  it("silu == x * sigmoid(x)", () => {
    const x = sample(24, [4, 6])
    expectClose(
      silu(x) as AnyTensor,
      x.mul(x.sigmoid()) as AnyTensor,
      1e-6,
    )
  })

  it("softmax node == the shift-and-normalise composition", () => {
    const x = sample(24, [4, 6])
    expectClose(
      softmax(x, 1) as AnyTensor,
      x.softmax(1) as AnyTensor,
      1e-6,
    )
  })

  it("logSumExp == max + log(sum(exp(x - max)))", () => {
    const x = sample(24, [4, 6])
    const m = x.max(1, true) as AnyTensor
    const composed = x.sub(m).exp().sum(1, true).log().add(m)
    expectClose(
      logSumExp(x, 1, true) as AnyTensor,
      composed as AnyTensor,
      1e-6,
    )
  })

  it("layerNorm == (x - mean)/sqrt(var + eps) * gamma + beta", () => {
    const x = sample(24, [4, 6])
    const gamma = sample(6, [6]).abs().add(0.5) as AnyTensor
    const beta = sample(6, [6]).mul(0.25) as AnyTensor
    const mu = x.mean(-1, true) as AnyTensor
    const centered = x.sub(mu) as AnyTensor
    const variance = centered.mul(centered).mean(-1, true) as AnyTensor
    const composed = centered
      .div(variance.add(1e-5).sqrt() as AnyTensor)
      .mul(gamma)
      .add(beta)
    expectClose(
      layerNorm(x, gamma as any, beta as any) as AnyTensor,
      composed as AnyTensor,
      1e-5,
    )
  })

  it("rmsNorm == x/sqrt(mean(x^2) + eps) * gamma", () => {
    const x = sample(24, [4, 6])
    const gamma = sample(6, [6]).abs().add(0.5) as AnyTensor
    const ms = x.mul(x).mean(-1, true) as AnyTensor
    const composed = x
      .div(ms.add(1e-5).sqrt() as AnyTensor)
      .mul(gamma)
    expectClose(
      rmsNorm(x, gamma as any) as AnyTensor,
      composed as AnyTensor,
      1e-5,
    )
  })

  it("crossEntropy == the logSoftmax/one-hot composition", () => {
    const logits = sample(12, [4, 3])
    // nn.crossEntropy takes a branded IndexTensor, so the ids are built once and shared below.
    const targets = Tensor.indices([2, 0, 1, 2], [4])
    expectClose(
      crossEntropy(
        logits as any,
        targets as any,
      ) as AnyTensor,
      composedCrossEntropy(logits as any, targets as any) as AnyTensor,
      1e-6,
    )
  })

  it("gatherRows == indexSelect, with the index shape kept", () => {
    const table = sample(12, [4, 3])
    const flat = table.indexSelect(
      Tensor.of([1, 0, 3, 3, 2, 0]) as any,
    ) as AnyTensor
    const rows = gatherRows(
      table,
      Tensor.of([[1, 0, 3], [3, 2, 0]]) as any,
    ) as AnyTensor
    expect(rows.shape).toEqual([2, 3, 3])
    expectExact(rows.view([6, 3] as any) as AnyTensor, flat)
  })
})

// Eager and the lazy interpreter run the same kernel, so they agree bit for bit, not to a tolerance.

describe("eager and lazy agree bit for bit", () => {
  const bits = (t: AnyTensor): Uint32Array =>
    new Uint32Array(
      Float32Array.from(t.data as Float32Array).buffer,
    )

  const cases: [string, () => AnyTensor][] = [
    ["gelu", () => gelu(sample(24, [4, 6])) as AnyTensor],
    ["silu", () => silu(sample(24, [4, 6])) as AnyTensor],
    ["softmax", () => softmax(sample(24, [4, 6]), 1) as AnyTensor],
    [
      "softmax causal",
      () => softmax(sample(18, [2, 3, 3]), -1, { causal: true }) as AnyTensor,
    ],
    [
      "logSumExp",
      () => logSumExp(sample(24, [4, 6]), 1) as AnyTensor,
    ],
    [
      "layerNorm",
      () =>
        layerNorm(
          sample(24, [4, 6]),
          sample(6, [6]).abs().add(0.5) as any,
          sample(6, [6]).mul(0.25) as any,
        ) as AnyTensor,
    ],
    [
      "rmsNorm",
      () =>
        rmsNorm(
          sample(24, [4, 6]),
          sample(6, [6]).abs().add(0.5) as any,
        ) as AnyTensor,
    ],
    [
      "crossEntropy",
      () =>
        crossEntropy(
          sample(12, [4, 3]) as any,
          Tensor.of([2, 0, 1, 2]) as any,
        ) as AnyTensor,
    ],
    [
      "gatherRows",
      () =>
        gatherRows(
          sample(12, [4, 3]),
          Tensor.of([[1, 0, 3], [3, 2, 0]]) as any,
        ) as AnyTensor,
    ],
    [
      "multi-axis reduce",
      () => sumTo(sample(24, [2, 3, 4]), [4]),
    ],
  ]

  it.each(cases)("%s", (_name, build) => {
    const { eager, lazy } = bothWays(build)
    expect(Array.from(bits(lazy))).toEqual(
      Array.from(bits(eager)),
    )
  })
})

// Causal softmax: the mask belongs to the node, not to a buffer.
describe("softmax{causal}", () => {
  it("zeroes the strict upper triangle exactly and keeps rows summing to 1", () => {
    const scores = sample(18, [2, 3, 3])
    const y = softmax(scores, -1, { causal: true }) as AnyTensor
    const d = y.data as Float32Array
    for (let b = 0; b < 2; b++) {
      for (let i = 0; i < 3; i++) {
        let row = 0
        for (let j = 0; j < 3; j++) {
          const v = d[b * 9 + i * 3 + j]!
          if (j > i) expect(v).toBe(0)
          row += v
        }
        expect(row).toBeCloseTo(1, 6)
      }
    }
  })

  it("rejects a dim that is not the key axis, naming the shape", () => {
    expect(() => softmax(sample(18, [2, 3, 3]), 0 as any, { causal: true }))
      .toThrow(/softmax\{causal\}.*\[2, 3, 3\]/)
  })
})

// Dropout: one draw, shared by forward and backward.
describe("dropout", () => {
  it("p = 0 is exactly the identity", () => {
    const x = sample(16, [4, 4])
    expectExact(dropout(x, 0) as AnyTensor, x)
  })

  it("keeps survivors at 1/(1-p) and sends the same mask to backward", () => {
    const x = sample(16, [4, 4]).requiresGrad() as AnyTensor
    const y = dropout(x, 0.5) as AnyTensor
    const yd = Float32Array.from(y.data as Float32Array)
    const xd = x.data as Float32Array
    y.sum().backward()
    const g = x.grad!.data as Float32Array
    for (let i = 0; i < yd.length; i++) {
      // y = x*mask, and the gradient of `sum` is exactly that same mask.
      expect(g[i]).toBeOneOf([0, 2])
      expect(yd[i]).toBeCloseTo(xd[i]! * g[i]!, 6)
    }
    // A non-degenerate draw: some kept, some dropped.
    expect(g.some(v => v === 0)).toBe(true)
    expect(g.some(v => v === 2)).toBe(true)
  })

  it("rejects p outside [0, 1)", () => {
    expect(() => dropout(sample(4, [4]), 1)).toThrow(
      /dropout: p must be in \[0, 1\)/,
    )
  })
})

describe("sumTo emits one reduce, not one per axis", () => {
  const nodeLines = (t: AnyTensor): string[] => printGraph(t).split("\n")

  beforeEach(() => configure({ lazy: true }))

  it("[2,3,4] -> [4] is leaf + one reduce", () => {
    const lines = nodeLines(sumTo(sample(24, [2, 3, 4]), [4]))
    expect(lines).toHaveLength(2)
    expect(lines[1]).toContain("reduce.sum")
    expect(lines[1]).toContain("dims=[0, 1]")
  })

  it("[2,3,4] -> [] is leaf + one reduce", () => {
    const lines = nodeLines(sumTo(sample(24, [2, 3, 4]), []))
    expect(lines).toHaveLength(2)
    expect(lines[1]).toContain("dims=[0, 1, 2]")
  })

  it("[2,3,4] -> [1,1,4] is leaf + ONE keepdim reduce", () => {
    const lines = nodeLines(
      sumTo(sample(24, [2, 3, 4]), [1, 1, 4]),
    )
    expect(lines).toHaveLength(2)
    expect(lines[1]).toContain("keepdim=true")
  })

  it("[2,3,4] -> [1,4] needs the one extra view", () => {
    const lines = nodeLines(sumTo(sample(24, [2, 3, 4]), [1, 4]))
    expect(lines).toHaveLength(3)
    expect(lines[2]).toContain("view")
  })

  it("a single-axis reduce is unchanged", () => {
    const lines = nodeLines(sumTo(sample(6, [2, 3]), [3]))
    expect(lines).toHaveLength(2)
    expect(lines[1]).toContain("dims=[0]")
  })
})

describe("reduce{dims} on the wire", () => {
  beforeEach(() => configure({ lazy: true }))

  const wire = (t: AnyTensor): any[] => JSON.parse(serializeLazyGraph([t])!.json).nodes

  it("a single axis serialises to exactly today's node", () => {
    const nodes = wire(sumTo(sample(6, [2, 3]), [3]))
    expect(nodes).toHaveLength(2)
    expect(nodes[1]).toMatchObject({
      op: "reduce",
      kind: "sum",
      dim: 0,
      keepdim: false,
    })
    expect(nodes[1].dims).toBeUndefined()
  })

  it("several axes become the addon's own ascending chain", () => {
    const nodes = wire(sumTo(sample(24, [2, 3, 4]), [4]))
    expect(nodes.map((n: any) => n.op)).toEqual([
      "leaf",
      "reduce",
      "reduce",
    ])
    expect(nodes[1]).toMatchObject({ dim: 0, keepdim: false })
    expect(nodes[1].shape).toEqual([3, 4])
    expect(nodes[2]).toMatchObject({ dim: 0, keepdim: false })
    expect(nodes[2].shape).toEqual([4])
  })

  it("keepdim over several axes ends in one view", () => {
    const nodes = wire(
      sumTo(sample(24, [2, 3, 4]), [1, 1, 4]),
    )
    expect(nodes.map((n: any) => n.op)).toEqual([
      "leaf",
      "reduce",
      "reduce",
      "view",
    ])
    expect(nodes[3].shape).toEqual([1, 1, 4])
  })
})

// A verbatim copy of evalMatmulEager's inner loop: it is the loop the recorded loss curves were
// produced with, and any change to the per-output accumulation order invalidates them.
function referenceMatmul(
  a: Float32Array,
  b: Float32Array,
  m: number,
  k: number,
  n: number,
): Float32Array {
  const out = new Float32Array(m * n)
  for (let i = 0; i < m; i++) {
    for (let j = 0; j < n; j++) {
      let acc = 0
      for (let p = 0; p < k; p++) {
        acc += a[i * k + p]! * b[p * n + j]!
      }
      out[i * n + j] = acc
    }
  }
  return out
}

describe("eager matmul is bit-identical to the reference loop", () => {
  const SHAPES: [number, number, number][] = [
    [3, 5, 7],
    [16, 16, 16],
    [1, 64, 33],
    [64, 784, 17],
    [37, 29, 64],
  ]

  it.each(SHAPES)("[%i,%i] x [%i]", (m, k, n) => {
    disableNative()
    const rand = (seed: number, len: number): Float32Array => {
      const out = new Float32Array(len)
      let s = seed
      for (let i = 0; i < len; i++) {
        s = (s * 1103515245 + 12345) & 0x7fffffff
        out[i] = (s / 0x7fffffff) * 2 - 1
      }
      return out
    }
    const ad = rand(m * 7 + 1, m * k)
    const bd = rand(n * 13 + 3, k * n)
    const got = evalMatmulEager(
      fromFlat(ad, [m, k]) as AnyTensor,
      fromFlat(bd, [k, n]) as AnyTensor,
    )
    const want = referenceMatmul(ad, bd, m, k, n)
    expect(
      Array.from(
        new Uint32Array(
          Float32Array.from(got.data as Float32Array).buffer,
        ),
      ),
    ).toEqual(Array.from(new Uint32Array(want.buffer)))
  })
})

describe("graphs the addon cannot parse fall back, loudly", () => {
  beforeEach(() => {
    resetJsCounters()
    delete process.env.TYPENET_STRICT_NATIVE
  })
  afterEach(() => {
    delete process.env.TYPENET_STRICT_NATIVE
  })

  it("a compiled gelu graph runs on the interpreter and matches eager", () => {
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {})
    try {
      const x = sample(24, [4, 6])
      const eager = gelu(x) as AnyTensor
      const step = compile((t: AnyTensor) => gelu(t) as AnyTensor)
      const out = step(x)
      expectClose(out, eager, 1e-6)
      expect(jsCounters().nativeFallbacks).toBe(1)
      expect(jsCounters().fallbacksByOp.gelu).toBe(1)
      // Once per distinct reason per process, not once per call.
      step(x)
      expect(warn).toHaveBeenCalledTimes(1)
      expect(warn.mock.calls[0]![0]).toContain(
        "running this graph on the JS interpreter",
      )
      expect(warn.mock.calls[0]![0]).toContain("gelu")
    } finally {
      warn.mockRestore()
    }
  })

  it("a primitives-only graph stays native and never falls back", () => {
    const step = compile((t: AnyTensor) => t.matmul(t.T as AnyTensor).tanh().sum())
    const x = sample(24, [4, 6])
    if (isNativeAvailable()) {
      useNative()
      const before = nativeCounters().prepares as number
      step(x)
      step(x)
      expect(nativeCounters().prepares as number).toBe(
        before + 1,
      )
    } else {
      step(x)
    }
    expect(jsCounters().nativeFallbacks).toBe(0)
  })

  it("TYPENET_STRICT_NATIVE=1 turns the notice into a throw naming the op", () => {
    process.env.TYPENET_STRICT_NATIVE = "1"
    expect(() => compile((t: AnyTensor) => gelu(t) as AnyTensor)(sample(6, [2, 3])))
      .toThrow(/TYPENET_STRICT_NATIVE=1.*gelu/s)
  })
})
