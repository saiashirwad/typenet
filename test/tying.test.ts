// TiedLinear.of(embedding) exists because plain tie() cannot relate an Embedding's [V, D] weight to
// a Linear<D, V>'s [D, V] weight: transposes, not the same shape. The tie is object identity.

import { describe, expect, it } from "vitest"
import { printGraph } from "../src/compile.ts"
import { configure } from "../src/lazy.ts"
import { Embedding, Linear, Module, TiedLinear } from "../src/nn/index.ts"
import { SGD } from "../src/optim/index.ts"
import type { IndexTensor } from "../src/shape.ts"
import { Tensor } from "../src/tensor.ts"
import { expectClose } from "./helpers.ts"

const V = 3
const D = 2

// Deterministic, so the two models below start bit-identical.
const initEmbedding = () =>
  Tensor.of([
    [1, 2],
    [3, 4],
    [5, 6],
  ])

class TiedNet extends Module {
  readonly wte: Embedding<typeof V, typeof D>
  readonly head: TiedLinear<typeof D, typeof V>

  constructor() {
    super()
    this.wte = new Embedding(V, D)
    this.wte.weight.copy_(initEmbedding())
    this.head = TiedLinear.of(this.wte)
  }

  forward(ids: IndexTensor<[2]>) {
    return this.head.forward(this.wte.forward(ids))
  }
}

/** Untied reference: two independent copies of the same initial values. */
class UntiedNet extends Module {
  readonly wte: Embedding<typeof V, typeof D>
  readonly head: Linear<typeof D, typeof V>

  constructor() {
    super()
    this.wte = new Embedding(V, D)
    this.wte.weight.copy_(initEmbedding())
    this.head = new Linear(D, V, { bias: false })
    this.head.weight.copy_(initEmbedding().transpose(0, 1))
  }

  forward(ids: IndexTensor<[2]>) {
    return this.head.forward(this.wte.forward(ids))
  }
}

describe("weight tying (TiedLinear)", () => {
  it("dedups with its Embedding: exactly one parameter, sized V*D not 2*V*D", () => {
    const tied = new TiedNet()
    const untied = new UntiedNet()

    expect(tied.parameters()).toHaveLength(1)
    expect([...tied.namedParameters().keys()]).toEqual(["wte.weight"])
    expect(untied.parameters()).toHaveLength(2)

    const totalElems = (m: Module) => m.parameters().reduce((n, p) => n + p.numel, 0)
    expect(totalElems(tied)).toBe(totalElems(untied) - V * D)
  })

  it("forward matches the untied reference", () => {
    const tied = new TiedNet()
    const untied = new UntiedNet()
    const ids = Tensor.indices([0, 2], [2])

    expectClose(tied.forward(ids), untied.forward(ids), 1e-6)
  })

  it("both uses accumulate into one .grad, matching the untied gradients summed by hand", () => {
    const tied = new TiedNet()
    const untied = new UntiedNet()
    const ids = Tensor.indices([0, 2], [2])

    tied.forward(ids).sum().backward()
    untied.forward(ids).sum().backward()

    // untied.head.weight is [D, V]; its gradient transposed back to [V, D] plus the gather
    // path's own gradient is what one shared leaf should have accumulated.
    const byHand = untied.wte.weight.grad!.add(untied.head.weight.grad!.transpose(0, 1))
    expectClose(tied.wte.weight.grad!, byHand, 1e-6)

    const before = tied.wte.weight.snapshot()
    const opt = new SGD(tied.parameters(), { lr: 0.1 })
    opt.step()
    const after = tied.wte.weight.data
    for (let i = 0; i < after.length; i++) {
      expect(after[i]).toBeCloseTo(before[i]! - 0.1 * byHand.data[i]!, 5)
    }
  })

  it("stateDict() writes one entry and loadStateDict restores both uses", () => {
    const tied = new TiedNet()
    const dict = tied.stateDict()
    expect(Object.keys(dict)).toEqual(["wte.weight"])

    const fresh = new TiedNet()
    fresh.wte.weight.fill_(0)
    expect(fresh.wte.weight.toArray()).not.toEqual(tied.wte.weight.toArray())

    fresh.loadStateDict(dict)
    expectClose(fresh.wte.weight, tied.wte.weight, 1e-6)
    const ids = Tensor.indices([0, 2], [2])
    expectClose(fresh.forward(ids), tied.forward(ids), 1e-6)
  })

  it("the shared weight is one leaf, reached through one permute", () => {
    configure({ lazy: true })
    try {
      const tied = new TiedNet()
      tied.wte.weight.named("wte")
      const ids = Tensor.indices([0, 2], [2])
      const logits = tied.forward(ids).named("logits")
      const lines = printGraph(logits).split("\n")

      const leafLines = lines.filter(l => /^wte\s+= leaf\b/.test(l))
      expect(leafLines).toHaveLength(1)

      // One permute of it, never a second buffer.
      const permuteLines = lines.filter(l => /= permute\(wte\)/.test(l))
      expect(permuteLines).toHaveLength(1)
    } finally {
      configure({ lazy: false })
    }
  })
})
