import { describe, expect, it } from "vitest"
import { tensor } from "../src/factories.ts"
import { configure } from "../src/lazy.ts"
import { Dropout, Embedding, functional, GELU, LayerNorm, Linear, RMSNorm, sequential, SiLU, Softmax } from "../src/nn/index.ts"
import { type AnyTensor, fromFlat, Tensor } from "../src/tensor.ts"
import { expectAgreeStrict, expectClose } from "./helpers.ts"

type Equal<A, B> = (<T>() => T extends A ? 1 : 2) extends (
  <T>() => T extends B ? 1 : 2
) ? true
  : false
type Expect<T extends true> = T

/** Exact equality, element for element. */
function expectExact(a: AnyTensor, b: AnyTensor): void {
  expect(b.shape).toEqual(a.shape)
  expect(Array.from(b.data)).toEqual(Array.from(a.data))
}

const sample = (n: number, shape: number[]): AnyTensor =>
  fromFlat(
    Float32Array.from(
      { length: n },
      (_, i) => Math.sin(i * 1.7 + 0.3) * 1.6,
    ),
    shape,
  ) as AnyTensor

describe("layers", () => {
  describe("Embedding", () => {
    it("gathers rows and matches gatherRows across eager/lazy/native", () => {
      const table = new Embedding(6, 4)
      const ids = Tensor.indices([0, 2, 5, 1], [2, 2])
      expectAgreeStrict(() => table.forward(ids) as AnyTensor)
    })

    it("[B], [B,T] and [B,T,K] all typecheck and run", () => {
      const table = new Embedding(10, 3)

      const b = table.forward(Tensor.indices([0, 1, 2, 3], [4]))
      type _1 = Expect<Equal<typeof b.shape, [4, 3]>>
      expect(b.shape).toEqual([4, 3])

      const bt = table.forward(Tensor.indices([0, 1, 2, 3, 4, 5], [2, 3]))
      type _2 = Expect<Equal<typeof bt.shape, [2, 3, 3]>>
      expect(bt.shape).toEqual([2, 3, 3])

      const btk = table.forward(Tensor.indices([0, 1, 2, 3, 4, 5, 6, 7], [2, 2, 2]))
      type _3 = Expect<Equal<typeof btk.shape, [2, 2, 2, 3]>>
      expect(btk.shape).toEqual([2, 2, 2, 3])
    })

    it("returns the exact stored rows", () => {
      const table = new Embedding(3, 2)
      ;(table.weight.data as Float32Array).set([10, 11, 20, 21, 30, 31])
      const out = table.forward(Tensor.indices([2, 0], [2])) as AnyTensor
      expect(Array.from(out.data)).toEqual([30, 31, 10, 11])
    })

    it("a non-integral / non-index tensor is a compile error", () => {
      // Uncalled function: the line below is type-checked by `tsc`
      // but never runs.
      function _typeOnly(table: Embedding<10, 3>, x: Tensor<[4]>) {
        // @ts-expect-error a plain (non-index) Tensor is not an IndexTensor
        table.forward(x)
      }
      void _typeOnly
    })

    it("its shape effect (appendDim) composes with a mapLast layer inside sequential", () => {
      // Embedding first (a real IndexTensor input), then a norm that owns
      // the appended axis: appendDim followed by mapLast, in a chain that
      // actually runs.
      const net = sequential(new Embedding(8, 5), new LayerNorm(5))
      const out = net.forward(Tensor.indices([0, 1, 2, 3], [1, 4]))
      type _1 = Expect<Equal<typeof out.shape, [1, 4, 5]>>
      expect(out.shape).toEqual([1, 4, 5])
    })
  })

  describe("LayerNorm", () => {
    it("normalises the last axis to mean 0 / variance 1 before the affine, matches eager/lazy/native", () => {
      const norm = new LayerNorm(4)
      const x = sample(12, [3, 4])
      expectAgreeStrict(() => norm.forward(x) as AnyTensor)
    })

    it("matches a hand-written reference for a fixed row", () => {
      const norm = new LayerNorm(4, { eps: 1e-5 })
      ;(norm.gamma.data as Float32Array).set([1, 1, 1, 1])
      ;(norm.beta.data as Float32Array).set([0, 0, 0, 0])
      const row = [1, 2, 3, 4]
      const mean = row.reduce((a, b) => a + b, 0) / 4
      const variance = row.reduce((a, b) => a + (b - mean) ** 2, 0) / 4
      const expected = row.map(v => (v - mean) / Math.sqrt(variance + 1e-5))
      const out = norm.forward(tensor([row])) as AnyTensor
      const data = Array.from(out.data as Float32Array)
      for (let i = 0; i < expected.length; i++) {
        expect(data[i]).toBeCloseTo(expected[i]!, 4)
      }
    })

    it("preserves shape and composes as [mapLast, D, D] inside sequential", () => {
      const net = sequential(new Linear(4, 4), new Softmax(-1), new LayerNorm(4))
      const out = net.forward(tensor([[1, 2, 3, 4]]))
      type _1 = Expect<Equal<typeof out.shape, [1, 4]>>
      expect(out.shape).toEqual([1, 4])

      function _widthMismatch() {
        // @ts-expect-error the norm owns the last axis: 4 -> 8 is a width mismatch
        sequential(new Linear(4, 4), new Softmax(-1), new LayerNorm(8))
      }
      void _widthMismatch
    })
  })

  describe("RMSNorm", () => {
    it("matches eager/lazy/native", () => {
      const norm = new RMSNorm(4)
      const x = sample(12, [3, 4])
      expectAgreeStrict(() => norm.forward(x) as AnyTensor)
    })

    it("matches a hand-written reference (no mean subtraction)", () => {
      const norm = new RMSNorm(3, { eps: 1e-5 })
      ;(norm.gamma.data as Float32Array).set([1, 1, 1])
      const row = [2, 4, 4]
      const ms = row.reduce((a, b) => a + b * b, 0) / 3
      const expected = row.map(v => v / Math.sqrt(ms + 1e-5))
      const out = norm.forward(tensor([row])) as AnyTensor
      const data = Array.from(out.data as Float32Array)
      for (let i = 0; i < expected.length; i++) {
        expect(data[i]).toBeCloseTo(expected[i]!, 4)
      }
    })
  })

  describe("Dropout", () => {
    it("eval() is bit-identical to the identity: same object, no node at all", () => {
      const layer = new Dropout(0.5)
      layer.eval()
      const x = sample(16, [4, 4]).requiresGrad() as AnyTensor
      const y = layer.forward(x)
      // Not merely close: the very same tensor comes back.
      expect(y).toBe(x)
    })

    it("p = 0 is exactly the identity in training mode too", () => {
      const layer = new Dropout(0)
      const x = sample(16, [4, 4])
      expectExact(layer.forward(x) as AnyTensor, x)
    })

    it("rejects p outside [0, 1) at construction", () => {
      expect(() => new Dropout(1)).toThrow(/Dropout: p must be in \[0, 1\)/)
      expect(() => new Dropout(-0.1)).toThrow(/Dropout: p must be in \[0, 1\)/)
    })

    it("keeps survivors at 1/(1-p) in training mode", () => {
      const layer = new Dropout(0.5)
      const x = sample(64, [64]).requiresGrad() as AnyTensor
      const y = layer.forward(x) as AnyTensor
      const xd = x.data as Float32Array
      const yd = y.data as Float32Array
      for (let i = 0; i < yd.length; i++) {
        expect(yd[i]! === 0 || Math.abs(yd[i]! - xd[i]! * 2) < 1e-5).toBe(true)
      }
      expect(Array.from(yd).some(v => v === 0)).toBe(true)
      expect(Array.from(yd).some(v => v !== 0)).toBe(true)
    })

    it("two Dropout layers draw independent masks", () => {
      configure({ seed: 7 })
      const a = new Dropout(0.5)
      const b = new Dropout(0.5)
      const x = sample(64, [64]).requiresGrad() as AnyTensor
      const ya = a.forward(x) as AnyTensor
      const yb = b.forward(x) as AnyTensor
      const da = Array.from(ya.data as Float32Array)
      const db = Array.from(yb.data as Float32Array)
      expect(da).not.toEqual(db)
    })

    it("the same seed reproduces the same masks", () => {
      const run = () => {
        configure({ seed: 42 })
        const a = new Dropout(0.5)
        const b = new Dropout(0.5)
        const x = sample(64, [64]) as AnyTensor
        return [
          Array.from((a.forward(x) as AnyTensor).data as Float32Array),
          Array.from((b.forward(x) as AnyTensor).data as Float32Array),
        ]
      }
      const first = run()
      const second = run()
      expect(second).toEqual(first)
    })
  })

  describe("GELU", () => {
    it("matches the tanh approximation and eager/lazy/native agree", () => {
      const layer = new GELU()
      const x = sample(24, [4, 6])
      const c = Math.sqrt(2 / Math.PI)
      const composed = x
        .mul(0.5)
        .mul(x.add(x.pow(3).mul(0.044715)).mul(c).tanh().add(1) as AnyTensor)
      expectClose(layer.forward(x) as AnyTensor, composed, 1e-6)
      expectAgreeStrict(() => layer.forward(x) as AnyTensor)
    })
  })

  describe("SiLU", () => {
    it("matches x * sigmoid(x) and eager/lazy/native agree", () => {
      const layer = new SiLU()
      const x = sample(24, [4, 6])
      expectClose(layer.forward(x) as AnyTensor, x.mul(x.sigmoid()) as AnyTensor, 1e-6)
      expectAgreeStrict(() => layer.forward(x) as AnyTensor)
    })
  })

  // nn.functional: same semantics as the layers, so a hand-rolled block
  // gets the fused kernels too.
  describe("nn.functional", () => {
    it("gelu/silu are the exact functions GELU/SiLU are built out of", () => {
      const x = sample(24, [4, 6])
      expectExact(functional.gelu(x) as AnyTensor, new GELU().forward(x) as AnyTensor)
      expectExact(functional.silu(x) as AnyTensor, new SiLU().forward(x) as AnyTensor)
    })

    it("layerNorm/rmsNorm are the exact functions LayerNorm/RMSNorm are built out of", () => {
      const norm = new LayerNorm(4)
      const rms = new RMSNorm(4)
      const x = sample(12, [3, 4])
      expectExact(
        functional.layerNorm(x, norm.gamma, norm.beta, { eps: norm.eps }) as AnyTensor,
        norm.forward(x) as AnyTensor,
      )
      expectExact(
        functional.rmsNorm(x, rms.gamma, { eps: rms.eps }) as AnyTensor,
        rms.forward(x) as AnyTensor,
      )
    })

    it("dropout is the exact function Dropout is built out of (p=0 identity)", () => {
      const x = sample(16, [4, 4])
      expectExact(functional.dropout(x, 0) as AnyTensor, x)
    })

    it("softmax is the fused node — agrees with, but is not the composed spelling `Softmax` still uses", () => {
      // The `Softmax` layer keeps the composed `Tensor.prototype.softmax`
      // spelling so it stays on the native fast path, so this is
      // `expectClose` at 1e-6, not `expectExact`.
      const x = sample(24, [4, 6])
      expectClose(
        functional.softmax(x, -1) as AnyTensor,
        new Softmax(-1).forward(x) as AnyTensor,
        1e-6,
      )
    })

    it("logSoftmax matches x - logSumExp(x, dim) and the composed Tensor.logSoftmax", () => {
      const x = sample(24, [4, 6])
      const y = functional.logSoftmax(x, -1) as AnyTensor
      expectClose(y, x.logSoftmax(-1) as AnyTensor, 1e-5)
    })

    it("arangeIndex builds 0..N-1 as an IndexTensor usable by Embedding", () => {
      const idx = functional.arangeIndex(5)
      expect(Array.from(idx.data)).toEqual([0, 1, 2, 3, 4])
      const table = new Embedding(5, 2)
      const out = table.forward(idx)
      type _1 = Expect<Equal<typeof out.shape, [5, 2]>>
      expect(out.shape).toEqual([5, 2])
    })
  })
})
