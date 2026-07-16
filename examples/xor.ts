"use tsover"

import {
  Linear,
  Module,
  SGD,
  tensor,
  type Tensor,
  type TensorParams
} from "../index.ts"

class XorNet extends Module {
  hidden = new Linear(2, 8)
  out = new Linear(8, 1)

  forward<B extends number, P extends TensorParams>(
    x: Tensor<[B, 2], P>
  ): Tensor<[B, 1], P> {
    const h = this.hidden.forward(x).tanh()
    return this.out.forward(h).sigmoid()
  }
}

const X = tensor([
  [0, 0],
  [0, 1],
  [1, 0],
  [1, 1]
])

const Y = tensor([[0], [1], [1], [0]])

const net = new XorNet()
const optim = new SGD(net.parameters(), {
  lr: 0.5,
  momentum: 0.9
})

for (let epoch = 1; epoch <= 1500; epoch++) {
  const pred = net.forward(X)
  const loss = ((pred - Y) ** 2).mean()

  optim.zeroGrad()
  loss.backward()
  optim.step()

  if (epoch % 250 === 0)
    console.log(
      `epoch ${String(epoch).padStart(4)}  loss ${loss.item().toFixed(6)}`
    )
}

console.log("\npredictions:")
const final = net.forward(X)
for (let i = 0; i < 4; i++) {
  const a = X.get(i, 0)
  const b = X.get(i, 1)
  console.log(
    `  ${a} xor ${b} -> ${final.get(i, 0).toFixed(4)} (target ${Y.get(i, 0)})`
  )
}
