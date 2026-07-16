"use tsover"

import tgpu, { type TgpuRoot } from "typegpu"
import { create, globals } from "webgpu"

import {
  clearTypeGPU,
  configureTypeGPU,
  tensor
} from "../index.ts"

// Dawn supplies the WebGPU implementation and the global WebGPU
// constants that TypeGPU expects, without creating a browser.
Object.assign(globalThis, globals)
let gpu: GPU | undefined = create([])

let device: GPUDevice | undefined
let root: TgpuRoot | undefined

try {
  const adapter = await gpu.requestAdapter({
    powerPreference: "high-performance"
  })
  if (!adapter)
    throw new Error("Dawn could not find a WebGPU adapter")

  device = await adapter.requestDevice()
  root = tgpu.initFromDevice({ device })
  configureTypeGPU(root)

  const input = tensor([
    [1, 2],
    [3, 4]
  ])
    .gpu()
    .requires_grad()
  const weight = tensor([[2], [-1]])
    .gpu()
    .requires_grad()

  const output = input.matmul(weight)
  const loss = (output ** 2).mean()
  loss.backward()

  console.log("output:", Array.from(await output.read()))
  console.log("loss:", Array.from(await loss.read()))
  console.log(
    "weight gradient:",
    Array.from(await weight.grad!.read())
  )
} finally {
  if (root) {
    clearTypeGPU(root)
    root.destroy()
  }
  device?.destroy()
  gpu = undefined
}
