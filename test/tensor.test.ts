import { describe, expect, it } from "vitest";
import { Tensor, tensor, zeros, ones, arange, eye } from "../src/tensor.ts";

describe("creation", () => {
  it("infers shape from nested arrays", () => {
    const t = tensor([
      [1, 2, 3],
      [4, 5, 6],
    ]);
    expect(t.shape).toEqual([2, 3]);
    expect(t.toArray()).toEqual([
      [1, 2, 3],
      [4, 5, 6],
    ]);
  });

  it("rejects ragged arrays", () => {
    expect(() => Tensor.of([[1, 2], [3]] as any)).toThrow(/[Rr]agged/);
  });

  it("creates scalars", () => {
    const s = Tensor.scalar(42);
    expect(s.shape).toEqual([]);
    expect(s.item()).toBe(42);
  });

  it("zeros / ones / eye / arange", () => {
    expect(zeros([2, 2]).toArray()).toEqual([
      [0, 0],
      [0, 0],
    ]);
    expect(ones([3]).toArray()).toEqual([1, 1, 1]);
    expect(eye(2).toArray()).toEqual([
      [1, 0],
      [0, 1],
    ]);
    expect(arange(4).toArray()).toEqual([0, 1, 2, 3]);
  });
});

describe("elementwise + broadcasting", () => {
  const a = tensor([
    [1, 2, 3],
    [4, 5, 6],
  ]);

  it("adds same-shape", () => {
    expect(a.add(a).toArray()).toEqual([
      [2, 4, 6],
      [8, 10, 12],
    ]);
  });

  it("broadcasts a row", () => {
    expect(a.add(tensor([10, 20, 30])).toArray()).toEqual([
      [11, 22, 33],
      [14, 25, 36],
    ]);
  });

  it("broadcasts a column", () => {
    expect(a.mul(tensor([[10], [100]])).toArray()).toEqual([
      [10, 20, 30],
      [400, 500, 600],
    ]);
  });

  it("broadcasts across both operands", () => {
    const col = tensor([[1], [2]]);
    const row = tensor([10, 20, 30]);
    expect(col.add(row as any).shape).toEqual([2, 3]);
    expect(col.add(row as any).toArray()).toEqual([
      [11, 21, 31],
      [12, 22, 32],
    ]);
  });

  it("handles scalars", () => {
    expect(a.sub(1).toArray()).toEqual([
      [0, 1, 2],
      [3, 4, 5],
    ]);
    expect(a.div(2).get(1, 2)).toBeCloseTo(3);
    expect(a.pow(2).get(1, 0)).toBe(16);
  });

  it("throws on incompatible shapes", () => {
    expect(() => a.add(tensor([1, 2, 3, 4]) as any)).toThrow(/broadcast/);
  });
});

describe("matmul", () => {
  it("2-D x 2-D", () => {
    const m1 = tensor([
      [1, 2],
      [3, 4],
    ]);
    const m2 = tensor([
      [5, 6],
      [7, 8],
    ]);
    expect(m1.matmul(m2).toArray()).toEqual([
      [19, 22],
      [43, 50],
    ]);
  });

  it("batched with broadcast batch dims", () => {
    const b = Tensor.ones([2, 3, 4]);
    const w = Tensor.ones([4, 5]);
    const out = b.matmul(w);
    expect(out.shape).toEqual([2, 3, 5]);
    expect(out.get(1, 2, 3)).toBe(4);
  });

  it("vector cases follow PyTorch semantics", () => {
    const v = tensor([1, 2, 3]);
    expect(v.dot(v).item()).toBe(14);
    const m = tensor([
      [1, 2],
      [3, 4],
    ]);
    expect(m.matmul(tensor([1, 1])).toArray()).toEqual([3, 7]);
    expect(tensor([1, 1]).matmul(m).toArray()).toEqual([4, 6]);
  });

  it("throws on inner-dim mismatch", () => {
    expect(() => tensor([[1, 2]]).matmul(tensor([[1, 2]]) as any)).toThrow(/inner dimensions/);
  });
});

describe("shape manipulation", () => {
  const t = arange(24).view([2, 3, 4]);

  it("view with -1 inference", () => {
    expect(t.view([-1]).shape).toEqual([24]);
    expect(t.view([4, -1]).shape).toEqual([4, 6]);
    expect(() => t.view([5, -1] as any)).toThrow();
  });

  it("squeeze / unsqueeze round trip", () => {
    const u = t.unsqueeze(0);
    expect(u.shape).toEqual([1, 2, 3, 4]);
    expect(u.squeeze().shape).toEqual([2, 3, 4]);
    expect(t.unsqueeze(-1).shape).toEqual([2, 3, 4, 1]);
  });

  it("transpose moves data", () => {
    const tr = t.transpose(0, 2);
    expect(tr.shape).toEqual([4, 3, 2]);
    expect(tr.get(3, 2, 1)).toBe(t.get(1, 2, 3));
  });

  it("permute", () => {
    const p = t.permute(2, 0, 1);
    expect(p.shape).toEqual([4, 2, 3]);
    expect(p.get(3, 1, 2)).toBe(t.get(1, 2, 3));
  });

  it(".T transposes matrices", () => {
    const m = tensor([
      [1, 2, 3],
      [4, 5, 6],
    ]);
    expect(m.T.toArray()).toEqual([
      [1, 4],
      [2, 5],
      [3, 6],
    ]);
  });
});

describe("reductions", () => {
  const a = tensor([
    [1, 2],
    [3, 4],
  ]);

  it("sum / mean / max over all", () => {
    expect(a.sum().item()).toBe(10);
    expect(a.mean().item()).toBe(2.5);
    expect(a.max().item()).toBe(4);
  });

  it("sum over dims with keepdim", () => {
    expect(a.sum(0).toArray()).toEqual([4, 6]);
    expect(a.sum(1).toArray()).toEqual([3, 7]);
    expect(a.sum(-1, true).toArray()).toEqual([[3], [7]]);
  });

  it("argmax", () => {
    expect(
      tensor([
        [1, 9, 2],
        [8, 3, 4],
      ])
        .argmax(1)
        .toArray(),
    ).toEqual([1, 0]);
  });

  it("softmax rows sum to 1", () => {
    const sm = tensor([
      [1, 2, 3],
      [1, 1, 1],
    ]).softmax(1);
    const rows = sm.sum(1).toArray();
    expect(rows[0]).toBeCloseTo(1);
    expect(rows[1]).toBeCloseTo(1);
    expect(sm.get(1, 0)).toBeCloseTo(1 / 3);
  });
});

describe("stack / cat", () => {
  it("stack inserts a new dim", () => {
    const s = Tensor.stack([tensor([1, 2]), tensor([3, 4]), tensor([5, 6])], 0);
    expect(s.shape).toEqual([3, 2]);
    expect(s.toArray()).toEqual([
      [1, 2],
      [3, 4],
      [5, 6],
    ]);
    const s1 = Tensor.stack([tensor([1, 2]), tensor([3, 4])], 1);
    expect(s1.toArray()).toEqual([
      [1, 3],
      [2, 4],
    ]);
  });

  it("cat joins along a dim", () => {
    const c = Tensor.cat(tensor([[1], [2]]), tensor([[3], [4]]), 1);
    expect(c.toArray()).toEqual([
      [1, 3],
      [2, 4],
    ]);
  });
});

describe("dtype / device tags", () => {
  it("to(float64) converts storage", () => {
    const t = tensor([1.5]).to("float64");
    expect(t.data).toBeInstanceOf(Float64Array);
    expect(t.dtype).toBe("float64");
  });

  it("binary ops promote to float64", () => {
    const out = tensor([1])
      .to("float64")
      .add(tensor([2]));
    expect(out.dtype).toBe("float64");
  });

  it("supports backend-neutral writes and async CPU reads", async () => {
    const t = zeros([3]).write([1, 2, 3]);
    expect(Array.from(await t.read())).toEqual([1, 2, 3]);
  });

  it("requires TypeGPU configuration and rejects float64 upload", () => {
    expect(() => tensor([1]).gpu()).toThrow(/TypeGPU is not configured/);
    expect(() => tensor([1]).to("float64").gpu()).toThrow(/does not support float64/);
  });
});
