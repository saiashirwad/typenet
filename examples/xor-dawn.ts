"use tsover"

import tgpu, { type TgpuRoot } from "typegpu"
import { create, globals } from "webgpu"

import {
  clearTypeGPU,
  configureTypeGPU,
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

const cases = [
  [0, 0],
  [0, 1],
  [1, 0],
  [1, 1]
] as const

Object.assign(globalThis, globals)
let gpu: GPU | undefined = create([])
let device: GPUDevice | undefined
let root: TgpuRoot | undefined
let optim: SGD | undefined

try {
  const adapter = await gpu.requestAdapter({
    powerPreference: "high-performance"
  })
  if (!adapter)
    throw new Error("Dawn could not find a WebGPU adapter")

  device = await adapter.requestDevice()
  root = tgpu.initFromDevice({ device })
  configureTypeGPU(root)

  const input = tensor(cases).gpu()
  const target = tensor([[0], [1], [1], [0]]).gpu()
  const net = new XorNet().gpu()
  optim = new SGD(net.parameters(), {
    lr: 0.5,
    momentum: 0.9
  })

  for (let epoch = 1; epoch <= 1500; epoch++) {
    const prediction = net.forward(input)
    const loss = ((prediction - target) ** 2).mean()

    optim.zeroGrad()
    loss.backward()
    optim.step()

    if (epoch % 250 === 0) {
      const [value] = await loss.read()
      console.log(
        `epoch ${String(epoch).padStart(4)}  loss ${value!.toFixed(6)}`
      )
    }
  }

  const prediction = Array.from(
    await net.forward(input).read()
  )
  console.log("\npredictions:")
  cases.forEach(([a, b], i) => {
    console.log(
      `  ${a} xor ${b} -> ${prediction[i]!.toFixed(4)} (target ${i === 1 || i === 2 ? 1 : 0})`
    )
  })
} finally {
  optim?.dispose()
  if (root) {
    clearTypeGPU(root)
    root.destroy()
  }
  device?.destroy()
  gpu = undefined
}
