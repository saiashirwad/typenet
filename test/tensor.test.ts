import { describe, expect, it } from "vitest"
import { compile } from "../src/compile.ts"
import { arange, eye, ones, tensor, zeros } from "../src/factories.ts"
import { fromFlat, Tensor } from "../src/tensor.ts"
import type { Equal, Expect } from "./helpers.ts"

// `pnpm typecheck` checks this file too; vitest never runs the types.

describe("creation", () => {
  it("infers shape from nested arrays", () => {
    const t = tensor([
      [1, 2, 3],
      [4, 5, 6],
    ])
    expect(t.shape).toEqual([2, 3])
    expect(t.toArray()).toEqual([
      [1, 2, 3],
      [4, 5, 6],
    ])
  })

  it("rejects ragged arrays", () => {
    expect(() => Tensor.of([[1, 2], [3]] as any)).toThrow(
      /[Rr]agged/,
    )
  })

  it("creates scalars", () => {
    const s = Tensor.scalar(42)
    expect(s.shape).toEqual([])
    expect(s.item()).toBe(42)
  })

  it("zeros / ones / eye / arange", () => {
    expect(zeros([2, 2]).toArray()).toEqual([
      [0, 0],
      [0, 0],
    ])
    expect(ones([3]).toArray()).toEqual([1, 1, 1])
    expect(eye(2).toArray()).toEqual([
      [1, 0],
      [0, 1],
    ])
    expect(arange(4).toArray()).toEqual([0, 1, 2, 3])
  })
})

describe("elementwise + broadcasting", () => {
  const a = tensor([
    [1, 2, 3],
    [4, 5, 6],
  ])

  it("adds same-shape", () => {
    expect(a.add(a).toArray()).toEqual([
      [2, 4, 6],
      [8, 10, 12],
    ])
  })

  it("broadcasts a row", () => {
    expect(a.add(tensor([10, 20, 30])).toArray()).toEqual([
      [11, 22, 33],
      [14, 25, 36],
    ])
  })

  it("broadcasts a column", () => {
    expect(a.mul(tensor([[10], [100]])).toArray()).toEqual([
      [10, 20, 30],
      [400, 500, 600],
    ])
  })

  it("broadcasts across both operands", () => {
    const col = tensor([[1], [2]])
    const row = tensor([10, 20, 30])
    expect(col.add(row as any).shape).toEqual([2, 3])
    expect(col.add(row as any).toArray()).toEqual([
      [11, 21, 31],
      [12, 22, 32],
    ])
  })

  it("handles scalars", () => {
    expect(a.sub(1).toArray()).toEqual([
      [0, 1, 2],
      [3, 4, 5],
    ])
    expect(a.div(2).get(1, 2)).toBeCloseTo(3)
    expect(a.pow(2).get(1, 0)).toBe(16)
  })

  it("throws on incompatible shapes", () => {
    expect(() => a.add(tensor([1, 2, 3, 4]) as any)).toThrow(/broadcast/)
  })
})

describe("matmul", () => {
  it("2-D x 2-D", () => {
    const m1 = tensor([
      [1, 2],
      [3, 4],
    ])
    const m2 = tensor([
      [5, 6],
      [7, 8],
    ])
    expect(m1.matmul(m2).toArray()).toEqual([
      [19, 22],
      [43, 50],
    ])
  })

  it("batched with broadcast batch dims", () => {
    const b = Tensor.ones([2, 3, 4])
    const w = Tensor.ones([4, 5])
    const out = b.matmul(w)
    expect(out.shape).toEqual([2, 3, 5])
    expect(out.get(1, 2, 3)).toBe(4)
  })

  it("vector cases follow PyTorch semantics", () => {
    const v = tensor([1, 2, 3])
    expect(v.matmul(v).item()).toBe(14)
    const m = tensor([
      [1, 2],
      [3, 4],
    ])
    expect(m.matmul(tensor([1, 1])).toArray()).toEqual([
      3,
      7,
    ])
    expect(tensor([1, 1]).matmul(m).toArray()).toEqual([
      4,
      6,
    ])
  })

  it("throws on inner-dim mismatch", () => {
    expect(() => tensor([[1, 2]]).matmul(tensor([[1, 2]]) as any)).toThrow(/inner dimensions/)
  })
})

describe("shape manipulation", () => {
  const t = arange(24).view([2, 3, 4])

  it("view with -1 inference", () => {
    expect(t.view([-1]).shape).toEqual([24])
    expect(t.view([4, -1]).shape).toEqual([4, 6])
    expect(() => t.view([5, -1] as any)).toThrow()
  })

  it("squeeze / unsqueeze round trip", () => {
    const u = t.unsqueeze(0)
    expect(u.shape).toEqual([1, 2, 3, 4])
    expect(u.squeeze().shape).toEqual([2, 3, 4])
    expect(t.unsqueeze(-1).shape).toEqual([2, 3, 4, 1])
  })

  it("transpose moves data", () => {
    const tr = t.transpose(0, 2)
    expect(tr.shape).toEqual([4, 3, 2])
    expect(tr.get(3, 2, 1)).toBe(t.get(1, 2, 3))
  })

  it("permute", () => {
    const p = t.permute(2, 0, 1)
    expect(p.shape).toEqual([4, 2, 3])
    expect(p.get(3, 1, 2)).toBe(t.get(1, 2, 3))
  })

  it(".T transposes matrices", () => {
    const m = tensor([
      [1, 2, 3],
      [4, 5, 6],
    ])
    expect(m.T.toArray()).toEqual([
      [1, 4],
      [2, 5],
      [3, 6],
    ])
  })

  it("slice reads each axis: end index, window, null and undefined", () => {
    const g = tensor([
      [1, 2, 3, 4],
      [5, 6, 7, 8],
      [9, 10, 11, 12],
    ])
    // a plain number is an end index, so the window starts at 0
    expect(g.slice([2, null]).toArray()).toEqual([
      [1, 2, 3, 4],
      [5, 6, 7, 8],
    ])
    expect(g.slice([null, [1, 3]]).toArray()).toEqual([
      [2, 3],
      [6, 7],
      [10, 11],
    ])
    // undefined is the same keep-the-axis entry as null.
    expect(g.slice([3, undefined]).toArray()).toEqual(g.toArray())
  })

  it("slice walks a multi-axis spec over a rank-3 tensor", () => {
    const t3 = arange(24).view([2, 3, 4])
    const win = t3.slice([null, [1, 3], [2, 4]])
    expect(win.shape).toEqual([2, 2, 2])
    expect(win.toArray()).toEqual([
      [
        [6, 7],
        [10, 11],
      ],
      [
        [18, 19],
        [22, 23],
      ],
    ])
  })

  it("slice rejects the windows that are compile errors at a typed call site", () => {
    const g = tensor([
      [1, 2],
      [3, 4],
      [5, 6],
    ])
    // as any reaches the run-time floor: every spec below is a type error on the shipped
    // signature, and the throw comes from the per-axis narrow.
    expect(() => g.slice([[2, 0], null] as any)).toThrow(/out of range/)
    expect(() => g.slice([4, null] as any)).toThrow(/out of range/)
    expect(() => g.slice([-1, null] as any)).toThrow(/out of range/)
    expect(() => g.slice([null] as any)).toThrow(/expects 2 entries, got 1/)
  })
})

describe("reductions", () => {
  const a = tensor([
    [1, 2],
    [3, 4],
  ])

  it("sum / mean / max over all", () => {
    expect(a.sum().item()).toBe(10)
    expect(a.mean().item()).toBe(2.5)
    expect(a.max().item()).toBe(4)
  })

  it("sum over dims with keepdim", () => {
    expect(a.sum(0).toArray()).toEqual([4, 6])
    expect(a.sum(1).toArray()).toEqual([3, 7])
    expect(a.sum(-1, true).toArray()).toEqual([[3], [7]])
  })

  it("argmax", () => {
    expect(
      tensor([
        [1, 9, 2],
        [8, 3, 4],
      ])
        .argmax(1)
        .toArray(),
    ).toEqual([1, 0])
  })

  it("softmax rows sum to 1", () => {
    const sm = tensor([
      [1, 2, 3],
      [1, 1, 1],
    ]).softmax(1)
    const rows = sm.sum(1).toArray()
    expect(rows[0]).toBeCloseTo(1)
    expect(rows[1]).toBeCloseTo(1)
    expect(sm.get(1, 0)).toBeCloseTo(1 / 3)
  })
})

describe("stack / cat", () => {
  it("stack inserts a new dim", () => {
    const s = Tensor.stack(
      [tensor([1, 2]), tensor([3, 4]), tensor([5, 6])],
      0,
    )
    expect(s.shape).toEqual([3, 2])
    expect(s.toArray()).toEqual([
      [1, 2],
      [3, 4],
      [5, 6],
    ])
    const s1 = Tensor.stack(
      [tensor([1, 2]), tensor([3, 4])],
      1,
    )
    expect(s1.toArray()).toEqual([
      [1, 3],
      [2, 4],
    ])
  })

  it("cat joins along a dim", () => {
    const c = Tensor.cat(
      tensor([[1], [2]]),
      tensor([[3], [4]]),
      1,
    )
    expect(c.toArray()).toEqual([
      [1, 3],
      [2, 4],
    ])
  })
})

describe("dtype tags", () => {
  it("to(float64) converts storage", () => {
    const t = tensor([1.5]).to("float64")
    expect(t.data).toBeInstanceOf(Float64Array)
    expect(t.dtype).toBe("float64")
  })

  it("binary ops promote to float64", () => {
    const out = tensor([1])
      .to("float64")
      .add(tensor([2]))
    expect(out.dtype).toBe("float64")
  })

  it("toString renders float and int64 storage", () => {
    const f = tensor([
      [1, 2],
      [3, 4],
    ])
    expect(f.toString()).toBe(
      "Tensor(shape=[2, 2], dtype=float32, data=[[1,2],[3,4]])",
    )
    // int64 lives in a BigInt64Array, whose elements JSON.stringify refuses, so the int64
    // leaves render as decimal strings.
    const i = fromFlat([1n, 2n, 3n], [3], "int64")
    expect(i.dtype).toBe("int64")
    expect(i.toString()).toContain("dtype=int64")
    expect(i.toString()).toContain(String.raw`["1","2","3"]`)
  })
})

describe("flatten / unflatten", () => {
  it("flatten(from,to) agrees with view() on ranks 2-4", () => {
    const t3 = arange(24).view([2, 3, 4])
    expect(t3.flatten(0, 1).shape).toEqual([6, 4])
    expect(t3.flatten(0, 1).toArray()).toEqual(t3.view([6, 4]).toArray())
    expect(t3.flatten(1, 2).shape).toEqual([2, 12])
    expect(t3.flatten(1, 2).toArray()).toEqual(t3.view([2, 12]).toArray())

    const t4 = arange(48).view([2, 3, 4, 2])
    expect(t4.flatten(1, 2).shape).toEqual([2, 12, 2])
    expect(t4.flatten(1, 2).toArray()).toEqual(t4.view([2, 12, 2]).toArray())

    // A one-axis window is the identity, the same as view() with the same shape.
    expect(t3.flatten(1, 1).shape).toEqual([2, 3, 4])
  })

  it("flatten() with no args collapses every axis, agreeing with view()", () => {
    const t = arange(24).view([2, 3, 4])
    expect(t.flatten().shape).toEqual([24])
    expect(t.flatten().toArray()).toEqual(t.view([24]).toArray())
  })

  it("unflatten is flatten's inverse, ranks 2-4", () => {
    const t2 = arange(6).view([2, 3])
    const roundTrip2 = t2.flatten(0, 1).unflatten(0, [2, 3])
    expect(roundTrip2.shape).toEqual([2, 3])
    expect(roundTrip2.toArray()).toEqual(t2.toArray())

    const t3 = arange(24).view([2, 3, 4])
    const split = t3.unflatten(2, [2, 2])
    expect(split.shape).toEqual([2, 3, 2, 2])
    expect(split.flatten(2, 3).toArray()).toEqual(t3.toArray())

    const t4 = arange(48).view([2, 3, 4, 2])
    const split4 = t4.unflatten(1, [3, 1])
    expect(split4.shape).toEqual([2, 3, 1, 4, 2])
    expect(split4.flatten(1, 2).toArray()).toEqual(t4.toArray())
  })

  it("unflatten(dim, sizes).permute(...) builds the attention head split", () => {
    const B = 2, T = 3, H = 2, Dh = 2
    const q = arange(B * T * H * Dh).view([B, T, H * Dh])
    const heads = q.unflatten(2, [H, Dh]).permute(0, 2, 1, 3)
    expect(heads.shape).toEqual([B, H, T, Dh])
    type _headsType = Expect<Equal<(typeof heads)["shape"], [2, 2, 3, 2]>>
    // b=1, h=1, t=2, dh=1, so this is the same element as q[1, 2, 1*Dh + 1].
    expect(heads.get(1, 1, 2, 1)).toBe(q.get(1, 2, 3))
  })

  it("flatten/unflatten name both shapes on a bad range", () => {
    const t = arange(6).view([2, 3])
    expect(() => (t as any).flatten(0, 5)).toThrow(/out of range/)
    expect(() => (t as any).unflatten(0, [4, 5])).toThrow(/sizes multiply to 20, not 2/)
  })
})

describe("safe mutation primitives", () => {
  it("fill_ overwrites every element", () => {
    const t = zeros([3])
    t.fill_(7)
    expect(t.toArray()).toEqual([7, 7, 7])
  })

  it("zero_ clears every element", () => {
    const t = tensor([1, 2, 3])
    t.zero_()
    expect(t.toArray()).toEqual([0, 0, 0])
  })

  it("copy_ overwrites bytes, converting dtype", () => {
    const dst = zeros([3]).to("float64")
    dst.copy_(tensor([1, 2, 3]).to("float64"))
    expect(dst.dtype).toBe("float64")
    expect(dst.toArray()).toEqual([1, 2, 3])
  })

  it("copy_ throws on a shape mismatch, naming both shapes", () => {
    const dst = zeros([2, 3])
    const src = zeros([3, 2])
    expect(() => dst.copy_(src as any)).toThrow(/\[3, 2\]/)
    expect(() => dst.copy_(src as any)).toThrow(/\[2, 3\]/)
  })

  it("addScaled_ is a fused elementwise multiply-add in place", () => {
    const p = tensor([1, 2, 3])
    const g = tensor([1, 1, 1])
    p.addScaled_(g, -0.5)
    expect(p.toArray()).toEqual([0.5, 1.5, 2.5])
  })

  it("addScaled_ throws on a shape mismatch, naming both shapes", () => {
    const p = zeros([2])
    const g = zeros([3])
    expect(() => p.addScaled_(g as any, 1)).toThrow(/\[2\]/)
    expect(() => p.addScaled_(g as any, 1)).toThrow(/\[3\]/)
  })

  it("snapshot() returns a copy that does not alias", () => {
    const t = tensor([1, 2, 3])
    const snap = t.snapshot()
    t.fill_(0)
    expect(Array.from(snap)).toEqual([1, 2, 3])
    expect(t.toArray()).toEqual([0, 0, 0])
  })

  it("fill_ throws inside compile()'s trace, naming the method", () => {
    const compiled = compile((x: Tensor<[3]>) => {
      x.fill_(0)
      return x
    })
    expect(() => compiled(tensor([1, 2, 3]))).toThrow(/fill_.*tracing/)
  })

  it("fill_/copy_/zero_/addScaled_ throw on a tensor with an active autograd tape", () => {
    const x = tensor([1, -2, 3]).requiresGrad()
    const y = x.relu()
    expect(y.taped).toBe(true)
    expect(() => y.fill_(0)).toThrow(/fill_.*tape/)
    expect(() => y.zero_()).toThrow(/zero_.*tape/)
    expect(() => y.copy_(tensor([0, 0, 0]))).toThrow(/copy_.*tape/)
    expect(() => y.addScaled_(tensor([1, 1, 1]), 1)).toThrow(/addScaled_.*tape/)
  })
})

describe("index tensors", () => {
  it("toIndex brands an integral tensor in place, without copying", () => {
    const t = tensor([1, 2, 3])
    const idx = t.toIndex()
    expect(idx).toBe(t)
  })

  it("toIndex throws naming the offending element for a non-integral tensor", () => {
    const t = tensor([1, 2.5, 3])
    expect(() => t.toIndex()).toThrow(/element 1 is 2\.5, not an integer/)
  })

  it("Tensor.indices builds an int32 index leaf directly from data", () => {
    const idx = Tensor.indices([0, 2, 1], [3])
    expect(idx.dtype).toBe("int32")
    expect(idx.toArray()).toEqual([0, 2, 1])
  })

  it("Tensor.indices throws naming the offending element for non-integral data", () => {
    expect(() => Tensor.indices([0, 1.5], [2])).toThrow(/element 1 is 1\.5, not an integer/)
  })
})
