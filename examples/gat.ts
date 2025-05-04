"use tsover";

import { Tensor, eye } from "../src/tensor.ts";
import type { TensorParams } from "../src/tensor.ts";
import { crossEntropy, Module } from "../src/nn.ts";
import { Adam } from "../src/optim.ts";

type GradParams = {
  requires_grad: true;
  device: "cpu";
  dtype: "float32";
};

class GATHead<FIn extends number, FOut extends number> extends Module {
  readonly W: Tensor<[FIn, FOut], GradParams>;
  readonly attSrc: Tensor<[FOut, 1], GradParams>;
  readonly attDst: Tensor<[FOut, 1], GradParams>;

  constructor(fin: FIn, fout: FOut) {
    super();
    const k = 1 / Math.sqrt(fin);
    this.W = (Tensor.rand([fin, fout]) * (2 * k) - k).requires_grad();
    this.attSrc = (Tensor.randn([fout, 1]) * 0.1).requires_grad();
    this.attDst = (Tensor.randn([fout, 1]) * 0.1).requires_grad();
  }

  forward<N extends number, P extends TensorParams>(
    h: Tensor<[N, FIn], P>,
    adj: Tensor<[N, N], any>,
  ): Tensor<[N, FOut], P> {
    const wh = h.matmul(this.W);
    const src = wh.matmul(this.attSrc);
    const dst = wh.matmul(this.attDst);

    const scores = (src + dst.T).leakyRelu(0.2);

    const masked = scores * adj + (1 - adj) * -1e9;

    const alpha = masked.softmax(1);
    return alpha.matmul(wh);
  }
}

class GAT<Features extends number, Classes extends number> extends Module {
  readonly head1: GATHead<Features, 8>;
  readonly head2: GATHead<Features, 8>;
  readonly out: GATHead<16, Classes>;

  constructor(
    readonly features: Features,
    readonly classes: Classes,
  ) {
    super();
    this.head1 = new GATHead(features, 8);
    this.head2 = new GATHead(features, 8);
    this.out = new GATHead(16, classes);
  }

  forward<N extends number, P extends TensorParams>(
    x: Tensor<[N, Features], P>,
    adj: Tensor<[N, N], any>,
  ): Tensor<[N, Classes], P> {
    const h1 = this.head1.forward(x, adj).leakyRelu(0.2);
    const h2 = this.head2.forward(x, adj).leakyRelu(0.2);

    const h = Tensor.cat(h1, h2, 1);
    return this.out.forward(h, adj);
  }
}

const N = 24;
const FEATURES = N;
const CLASSES = 2;

const adj = Tensor.eye(N);
const labels: number[] = [];
for (let i = 0; i < N; i++) labels.push(i < N / 2 ? 0 : 1);
for (let i = 0; i < N; i++) {
  for (let j = i + 1; j < N; j++) {
    const sameBlock = labels[i] === labels[j];
    const p = sameBlock ? 0.5 : 0.05;
    if (Math.random() < p) {
      adj.data[i * N + j] = 1;
      adj.data[j * N + i] = 1;
    }
  }
}

const X = eye(N);

const net = new GAT(FEATURES, CLASSES);
const optim = new Adam(net.parameters(), { lr: 0.01 });

for (let epoch = 1; epoch <= 300; epoch++) {
  const logits = net.forward(X, adj);
  const loss = crossEntropy(logits, labels);

  optim.zeroGrad();
  loss.backward();
  optim.step();

  if (epoch % 60 === 0) {
    const preds = net.forward(X, adj).argmax(1);
    let correct = 0;
    for (let i = 0; i < N; i++) if (preds.data[i] === labels[i]) correct++;
    console.log(
      `epoch ${String(epoch).padStart(3)}  loss ${loss
        .item()
        .toFixed(4)}  accuracy ${((100 * correct) / N).toFixed(1)}%`,
    );
  }
}
