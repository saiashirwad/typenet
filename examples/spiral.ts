"use tsover";

import { Tensor } from "../src/tensor.ts";
import { crossEntropy, Linear, ReLU, sequential } from "../src/nn.ts";
import { Adam } from "../src/optim.ts";

const CLASSES = 3;
const N = 180;

const net = sequential(
  new Linear(2, 16),
  new ReLU(),
  new Linear(16, 16),
  new ReLU(),
  new Linear(16, CLASSES),
);

const perClass = N / CLASSES;
const X = Tensor.zeros([N, 2]);
const targets: number[] = [];
for (let c = 0; c < CLASSES; c++) {
  for (let i = 0; i < perClass; i++) {
    const r = i / perClass;
    const t = (c * 2 * Math.PI) / CLASSES + r * 4 + (Math.random() - 0.5) * 0.35;
    const row = c * perClass + i;
    X.data[row * 2] = r * Math.cos(t);
    X.data[row * 2 + 1] = r * Math.sin(t);
    targets.push(c);
  }
}

const optim = new Adam(net.parameters(), { lr: 0.02 });

for (let epoch = 1; epoch <= 400; epoch++) {
  const logits = net.forward(X);
  const loss = crossEntropy(logits, targets);

  optim.zeroGrad();
  loss.backward();
  optim.step();

  if (epoch % 80 === 0) {
    const preds = net.forward(X).argmax(1);
    let correct = 0;
    for (let i = 0; i < N; i++) if (preds.data[i] === targets[i]) correct++;
    console.log(
      `epoch ${String(epoch).padStart(3)}  loss ${loss.item().toFixed(4)}  accuracy ${((100 * correct) / N).toFixed(1)}%`,
    );
  }
}
