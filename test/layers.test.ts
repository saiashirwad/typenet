import { describe, expect, it } from "vitest"
import { tensor } from "../src/factories.ts"
import { configure } from "../src/lazy.ts"
import {
  Dropout,
  Embedding,
  functional,
  GELU,
  LayerNorm,
  Linear,
  RMSNorm,
  Rnn,
  scan,
  Sequence,
  sequential,
  SiLU,
  Softmax,
  stepUnbatched,
} from "../src/nn/index.ts"
import { type AnyTensor, Tensor } from "../src/tensor.ts"
import { type Equal, type Expect, expectAgreeStrict, expectClose, expectExact, sample } from "./helpers.ts"

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
      // Uncalled function: type-checked by tsc, never run.
      function _typeOnly(table: Embedding<10, 3>, x: Tensor<[4]>) {
        // @ts-expect-error a plain (non-index) Tensor is not an IndexTensor
        table.forward(x)
      }
    })

    it("its shape effect (appendDim) composes with a mapLast layer inside sequential", () => {
      // Embedding first (a real IndexTensor input), then a norm that owns the appended axis:
      // appendDim followed by mapLast, in a chain that actually runs.
      const net = sequential(new Embedding(8, 5), new LayerNorm(5))
      const out = net.forward(Tensor.indices([0, 1, 2, 3], [1, 4]))
      type _1 = Expect<Equal<typeof out.shape, [1, 4, 5]>>
      expect(out.shape).toEqual([1, 4, 5])
    })
  })

  describe("LayerNorm", () => {
    it("matches eager/lazy/native", () => {
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

  describe("Rnn", () => {
    it("a step maps [B, In] and a [B, H] state to [B, H]", () => {
      const layer = new Rnn(4, 3)
      const x = sample(8, [2, 4])
      const state = sample(6, [2, 3])
      const out = layer.forward(x, { state })
      // H is inferred from the layer and lands in the type; B widens, since a generic `Tensor`
      // parameter is not something tsc will read a literal out of.
      type _1 = Expect<Equal<typeof out.shape[1], 3>>
      expect(out.shape).toEqual([2, 3])
      expectAgreeStrict(() => layer.forward(x, { state }) as AnyTensor)
    })

    it("{ batch } starts from the zero state, which is what it returns", () => {
      const layer = new Rnn(4, 3)
      const x = sample(8, [2, 4])
      const fromZero = layer.forward(x, { batch: 2 })
      expectExact(fromZero as AnyTensor, layer.forward(x, { state: layer.zeroState(2) }) as AnyTensor)
      // A biasless layer with zeroed weights has nothing but the state to contribute, so its
      // output is tanh(0) = 0 and the state really did start at zero.
      const biasless = new Rnn(4, 3, { bias: false })
      biasless.weightIH.zero_()
      biasless.weightHH.zero_()
      expectExact(biasless.forward(x, { batch: 2 }) as AnyTensor, Tensor.zeros([2, 3]) as AnyTensor)
    })

    it("is tanh(x @ weightIH + state @ weightHH + b_ih + b_hh), spelled out", () => {
      const layer = new Rnn(4, 3)
      const x = sample(8, [2, 4])
      const state = sample(6, [2, 3])
      const reference = (x as AnyTensor)
        .matmul(layer.weightIH as AnyTensor)
        .add((state as AnyTensor).matmul(layer.weightHH as AnyTensor))
        .add(layer.biasIH as AnyTensor)
        .add(layer.biasHH as AnyTensor)
        .tanh()
      expectClose(layer.forward(x, { state }) as AnyTensor, reference, 1e-6)
    })

    it("carries memory forward: the same input gives different outputs from different states", () => {
      const layer = new Rnn(3, 5)
      const x = (sample(3, [1, 3]) as AnyTensor).view([1, 3])
      const a = layer.forward(x, { state: layer.zeroState(1) })
      const b = layer.forward(x, { state: (a.mul(2) as AnyTensor).view([1, 5]) })
      expect(Array.from(a.data)).not.toEqual(Array.from(b.data))
    })

    it("stepUnbatched agrees with the batched step", () => {
      const layer = new Rnn(4, 3)
      const x = Tensor.zeros<[4]>([4])
      const state = Tensor.zeros<[3]>([3])
      const batched = layer.forward(Tensor.zeros<[1, 4]>([1, 4]), {
        state: Tensor.zeros<[1, 3]>([1, 3]),
      })
      expectClose(stepUnbatched(layer, x, state) as AnyTensor, (batched as AnyTensor).squeeze(0), 1e-6)
    })

    it("initialises every weight and bias in U(-1/sqrt(H), 1/sqrt(H))", () => {
      const layer = new Rnn(64, 16)
      const k = 1 / Math.sqrt(16)
      for (const p of layer.parameters()) {
        expect(p.shape.length).toBeGreaterThan(0)
        for (const v of p.data) {
          expect(Math.abs(Number(v))).toBeLessThanOrEqual(k + 1e-6)
        }
      }
      // The recurrent bound tracks H, not In: with In = 64 a fan-based draw would be far smaller.
      const values = Array.from(layer.weightHH.data, Number)
      expect(Math.max(...values.map(Math.abs))).toBeGreaterThan(0.05)
    })

    it("two projections are one parameter each, and stateDict names both", () => {
      const layer = new Rnn(4, 3)
      const names = [...layer.namedParameters().keys()].sort()
      expect(names).toEqual(["biasHH", "biasIH", "weightHH", "weightIH"])
    })

    it("a recurrent layer can be unrolled by hand and backprops through the chain", () => {
      const layer = new Rnn(4, 3)
      const x = sample(8, [2, 4])
      let state = layer.zeroState(2)
      let total = (state as AnyTensor).sum()
      // Two steps of the same input, so the second depends on the first through weightHH.
      for (let t = 0; t < 2; t++) {
        state = layer.forward(x, { state })
        total = (total as AnyTensor).add((state as AnyTensor).sum()) as typeof total
      }
      total.backward()
      const grad = layer.weightHH.grad
      expect(grad).not.toBeNull()
      expect(Array.from(grad!.data).some(v => v !== 0)).toBe(true)
    })
  })

  describe("scan", () => {
    it("collects every step's output and the final state", () => {
      const init = Tensor.zeros<[2, 3]>([2, 3])
      const run = scan<2, [2, 3], [2, 4]>(init, 5, (state, t) => ({
        output: Tensor.cat(state, Tensor.full<[2, 1]>([2, 1], t), 1),
        state: state.add(1),
      }))
      type _1 = Expect<Equal<typeof run.outputs.shape, [2, number, 4]>>
      expect(run.outputs.shape).toEqual([2, 5, 4])
      expect(run.state.shape).toEqual([2, 3])
      expectClose(run.state as AnyTensor, Tensor.full<[2, 3]>([2, 3], 5) as AnyTensor, 1e-6)
      // Step t's output carries t in its last column, so the order is the loop's own.
      for (let t = 0; t < 5; t++) {
        expect(run.outputs.select(1, t).get(0, 3)).toBe(t)
      }
    })

    it("the state that comes out is the one the last step returned", () => {
      const run = scan<1, [1, 2], [1, 2]>(Tensor.zeros<[1, 2]>([1, 2]), 3, state => ({
        output: state.mul(2),
        state: state.add(1),
      }))
      // state goes 0 -> 1 -> 2 -> 3 while the outputs are 0, 2, 4.
      expectClose(run.state as AnyTensor, Tensor.full<[1, 2]>([1, 2], 3) as AnyTensor, 1e-6)
      expect(run.outputs.select(1, 2).get(0, 0)).toBe(4)
    })

    it("backprops through the whole chain to the initial state", () => {
      const init = Tensor.zeros<[1, 2]>([1, 2]).requiresGrad()
      const run = scan<1, [1, 2], [1, 2]>(init, 4, state => ({
        output: state.mul(2),
        state: state.add(1),
      }))
      run.outputs.sum().backward()
      // Each of the four outputs is `2 * (x + t)`, so every one contributes a 2: 8 per element.
      expect(Array.from(init.grad!.data)).toEqual([8, 8])
    })

    it("rejects a step that changes the batch", () => {
      // The batch is generic, so a runtime one is the only kind that can disagree: for a literal
      // batch, tsc rejects the step outright.
      const mismatched = <B extends number>(state: Tensor<[B, 2]>): { output: Tensor<[B, 2]>; state: Tensor<[B, 2]> } => {
        const smaller = 0.5 * state.shape[0]!
        return { output: state, state: state.narrow(0, 0, smaller as B) }
      }
      expect(() => scan<4, [4, 2], [4, 2]>(Tensor.zeros<[4, 2]>([4, 2]), 2, mismatched)).toThrow(/batched 4/)
    })

    it("rejects a negative step count", () => {
      expect(() => scan<1, [1, 2], [1, 2]>(Tensor.zeros<[1, 2]>([1, 2]), -1, state => ({ output: state, state }))).toThrow(/non-negative integer/)
    })

    it("steps = 0 has no output, and says so", () => {
      expect(() => scan<1, [1, 2], [1, 2]>(Tensor.zeros<[1, 2]>([1, 2]), 0, state => ({ output: state, state }))).toThrow(/steps was 0/)
    })
  })

  describe("Sequence", () => {
    it("steps for an unbounded run and concatenates what it recorded", () => {
      const run = Sequence.of<2, [2, 3]>(Tensor.zeros<[2, 3]>([2, 3]))
      for (let i = 0; i < 4; i++) {
        run.step(Tensor.full<[2, 1]>([2, 1], i), Tensor.full<[2, 3]>([2, 3], i))
      }
      expect(run.length).toBe(4)
      expect(run.outputs<[2, 1]>().shape).toEqual([2, 4, 1])
      expect(run.state.get(0, 0)).toBe(3)
    })

    it("advance moves the state without recording anything", () => {
      const run = Sequence.of<1, [1, 2]>(Tensor.zeros<[1, 2]>([1, 2]))
      run.advance(Tensor.full<[1, 2]>([1, 2], 7))
      expect(run.length).toBe(0)
      expect(run.state.get(0, 0)).toBe(7)
      expect(() => run.outputs()).toThrow(/nothing has been stepped/)
    })

    it("a step that changes the batch is rejected", () => {
      const run = Sequence.of<2, [2, 3]>(Tensor.zeros<[2, 3]>([2, 3]))
      const halved = <B extends number>(state: Tensor<[B, 3]>): Tensor<[B, 3]> => state.narrow(0, 0, (0.5 * state.shape[0]!) as B)
      expect(() => run.step(Tensor.zeros<[2, 1]>([2, 1]), halved(run.state))).toThrow(/does not match the state's 2/)
    })
  })

  // nn.functional: the same semantics as the layers, so a hand-rolled block gets the fused kernels too.
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

    it("softmax agrees with the composed spelling `Softmax` still uses", () => {
      // The Softmax layer keeps the composed Tensor.prototype.softmax spelling to stay on the
      // native fast path, so this is expectClose at 1e-6, not expectExact.
      const x = sample(24, [4, 6])
      expectClose(
        functional.softmax(x, -1) as AnyTensor,
        new Softmax(-1).forward(x) as AnyTensor,
        1e-6,
      )
    })

    it("logSoftmax matches the composed Tensor.logSoftmax", () => {
      const x = sample(24, [4, 6])
      expectClose(functional.logSoftmax(x, -1) as AnyTensor, x.logSoftmax(-1) as AnyTensor, 1e-5)
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
