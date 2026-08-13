"use tsover"

import { SGD } from "../index.ts"
import { XorNet, XOR_X, XOR_Y } from "./xor-net.ts"

const net = new XorNet()
const optim = new SGD(net.parameters(), {
  lr: 0.5,
  momentum: 0.9,
})

for (let epoch = 1; epoch <= 1500; epoch++) {
  const pred = net.forward(XOR_X)
  const loss = ((pred - XOR_Y) ** 2).mean()

  optim.zeroGrad()
  loss.backward()
  optim.step()

  if (epoch % 250 === 0) {
    console.log(
      `epoch ${String(epoch).padStart(4)}  loss ${loss.item().toFixed(6)}`,
    )
  }
}

console.log("\npredictions:")
const final = net.forward(XOR_X)
for (let i = 0; i < 4; i++) {
  const a = XOR_X.get(i, 0)
  const b = XOR_X.get(i, 1)
  console.log(
    `  ${a} xor ${b} -> ${final.get(i, 0).toFixed(4)} (target ${XOR_Y.get(i, 0)})`,
  )
}
