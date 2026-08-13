"use tsover"

// The published-race gate: 1024 nodes, k=8, batch 8, C=16, H=128, ONE
// compiled bucket in [48, 80] (64), forward + backward + clipGradNorm +
// Adam, three-run median. This is the number the README's speed claims
// come from; `TYPENET_EVALUATOR=loops|cpu|gpu` forces an evaluator and
// `TYPENET_PROFILE=1` prints the per-op table afterwards.

import { Adam, configure, isNativeAvailable, nativeDevice, nativeDeviceMode, Tensor, useNative } from "../../index.ts"
import type { AnyTensor } from "../../src/tensor.ts"
import { batchEdges, randomGeometricGraph } from "./graphs.ts"
import { GraphNCA, graphTensors, seedState } from "./model.ts"
import { compiledTrainStep } from "./train-util.ts"

const nodes = 1024
const batch = 8
const C = 16
const steps = 64

if (!isNativeAvailable()) {
  throw new Error("native addon not available; run pnpm build:native")
}
useNative()
configure({ seed: 1 })

const { edges } = randomGeometricGraph({ nodes, dim: 2 })
const graph = graphTensors(
  batchEdges(edges, batch, nodes),
  batch * nodes,
)
const model = new GraphNCA(C)
const params = model.parameters()
const optimizer = new Adam(params, { lr: 5e-4 })
const target = Tensor.rand([batch * nodes, 4]) as AnyTensor
const x0 = Tensor.zeros([batch * nodes, C]) as AnyTensor
;(x0.data as Float32Array).set(seedState(batch, nodes, C, 0))

console.log(
  `backend: native, ${nativeDeviceMode()} device (accelerator: ${nativeDevice()}), `
    + `TYPENET_EVALUATOR=${process.env.TYPENET_EVALUATOR ?? "(unset)"}`,
)
console.log(
  `${nodes} nodes x ${batch} = ${batch * nodes}, `
    + `${edges.count * batch} edges, C=${C}, H=128, ${steps} rollout steps`,
)

const full = compiledTrainStep(
  model,
  optimizer,
  graph,
  target,
  x0,
  steps,
)

full(x0).item() // warmup (prepare + pin)

const runs: number[] = []
for (let i = 0; i < 3; i++) {
  const start = performance.now()
  full(x0).item()
  runs.push(performance.now() - start)
}
runs.sort((a, b) => a - b)
const median = runs[1]!
console.log(`runs (ms): ${runs.map(r => r.toFixed(0)).join(", ")}`)
console.log(
  `median full step: ${median.toFixed(1)} ms = ${(1000 / median).toFixed(2)} training steps/s`,
)

if (process.env.TYPENET_PROFILE === "1") {
  const native = await import("@typenet/native")
  console.log(native.takeProfile())
}
