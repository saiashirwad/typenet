import { describe, expect, it } from "vitest";
import { Tensor, tensor } from "../src/tensor.ts";
import { Linear, Module, ReLU, crossEntropy, mseLoss, sequential } from "../src/nn.ts";
import { SGD, Adam } from "../src/optim.ts";

describe("Linear", () => {
  it("computes x @ W + b with typed dims", () => {
    const layer = new Linear(3, 2);
    (layer.weight.data as Float32Array).set([1, 0, 0, 1, 1, 1]);
    (layer.bias!.data as Float32Array).set([10, 20]);
    const out = layer.forward(tensor([[1, 2, 3]]));
    expect(out.shape).toEqual([1, 2]);

    expect(out.toArray()).toEqual([[14, 25]]);
  });

  it("collects parameters through nesting", () => {
    class Net extends Module {
      a = new Linear(4, 3);
      b = new Linear(3, 2);
      list = [new Linear(2, 2)];
    }
    const net = new Net();
    expect(net.parameters()).toHaveLength(6);
    const noBias = new Linear(4, 2, { bias: false });
    expect(noBias.parameters()).toHaveLength(1);
  });
});

describe("sequential", () => {
  it("runs the chain and matches manual composition", () => {
    const l1 = new Linear(2, 4);
    const l2 = new Linear(4, 3);
    const net = sequential(l1, new ReLU(), l2);
    const x = tensor([[0.5, -1]]);
    const manual = l2.forward(l1.forward(x).relu());
    expect(net.forward(x).toArray()).toEqual(manual.toArray());
    expect(net.parameters()).toHaveLength(4);
  });

  it("rejects mismatched chains at runtime too", () => {
    expect(() =>
      sequential(new Linear(2, 16), new Linear(17, 3) as any),
    ).toThrow(/expects 17 features/);
  });
});

describe("losses", () => {
  it("mseLoss", () => {
    const loss = mseLoss(tensor([1, 2, 3]), tensor([2, 2, 2]));
    expect(loss.item()).toBeCloseTo(2 / 3);
  });

  it("crossEntropy matches manual computation", () => {
    const logits = tensor([
      [Math.log(1), Math.log(3)],
      [Math.log(4), Math.log(4)],
    ]);
    const loss = crossEntropy(logits, [1, 0]);
    const expected = -(Math.log(3 / 4) + Math.log(0.5)) / 2;
    expect(loss.item()).toBeCloseTo(expected, 5);
  });

  it("crossEntropy validates targets", () => {
    expect(() => crossEntropy(tensor([[1, 2]]), [5])).toThrow(/out of range/);
  });
});

describe("training", () => {
  it("SGD fits y = 2x + 1", () => {
    const w = Tensor.scalar(0).requires_grad();
    const b = Tensor.scalar(0).requires_grad();
    const x = tensor([0, 1, 2, 3]);
    const y = tensor([1, 3, 5, 7]);
    const opt = new SGD([w, b], { lr: 0.05 });
    for (let i = 0; i < 500; i++) {
      const pred = x.mul(w as any).add(b as any);
      const loss = mseLoss(pred, y as any);
      opt.zeroGrad();
      loss.backward();
      opt.step();
    }
    expect(w.item()).toBeCloseTo(2, 1);
    expect(b.item()).toBeCloseTo(1, 1);
  });

  it("Adam + crossEntropy learns a toy classification", () => {

    const layer = new Linear(2, 2);
    const opt = new Adam(layer.parameters(), { lr: 0.05 });
    const xs: number[][] = [];
    const ys: number[] = [];
    for (let i = 0; i < 20; i++) {
      const cls = i % 2;
      const cx = cls === 0 ? -1 : 1;
      xs.push([cx + Math.sin(i) * 0.1, cx + Math.cos(i) * 0.1]);
      ys.push(cls);
    }
    const X = Tensor.zeros([20, 2]);
    (X.data as Float32Array).set(xs.flat());

    let loss = 0;
    for (let epoch = 0; epoch < 200; epoch++) {
      const out = layer.forward(X);
      const l = crossEntropy(out, ys);
      opt.zeroGrad();
      l.backward();
      opt.step();
      loss = l.item();
    }
    expect(loss).toBeLessThan(0.05);
    const preds = layer.forward(X).argmax(1).toArray();
    expect(preds).toEqual(ys);
  });
});
