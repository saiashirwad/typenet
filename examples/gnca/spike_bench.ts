"use tsover"

// Spike for issue #11: measure the loop evaluator at GNCA size.
// Published recipe: 1024 nodes, k=8, batch 8, C=16, H=128, one compiled
// bucket in [48, 80] (64 here), forward + backward + Adam.
// Run with TYPENET_EVALUATOR=loops (or cpu) to pick the evaluator.

import {
  Adam,
  clipGradNorm,
  compile,
  configure,
  isNativeAvailable,
  nativeDevice,
  nativeDeviceMode,
  Tensor,
  useNative,
} from "../../index.ts"
import { batchEdges, randomGeometricGraph } from "./graphs.ts"
import { aliveMask, GraphNCA, graphTensors, seedState } from "./model.ts"

type AnyTensor = Tensor<any, any>

const nodes = 1024
const batch = 8
const C = 16
const steps = 64

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

if (!isNativeAvailable()) {
  throw new Error("native addon not available; the spike needs it")
}
useNative()
configure({ seed: 1 })
console.log(
  `backend: native, ${nativeDeviceMode()} device (accelerator: ${nativeDevice()}), `
    + `TYPENET_EVALUATOR=${process.env.TYPENET_EVALUATOR ?? "(unset)"}`,
)
console.log(
  `${nodes} nodes x ${batch} = ${batch * nodes}, `
    + `${edges.count * batch} edges, C=${C}, H=128, ${steps} rollout steps`,
)

const rollout = (x: AnyTensor, n: number) => {
  for (let i = 0; i < n; i++) {
    x = model.forward(x, graph).mul(aliveMask(x, graph))
  }
  return x
}

const full = compile((input: AnyTensor) => {
  const loss = rollout(input, steps)
    .narrow(1, 0, 4)
    .sub(target)
    .pow(2)
    .mean()
  optimizer.zeroGrad()
  loss.backward()
  clipGradNorm(params, 1)
  optimizer.step()
  return loss
})

// warmup (includes trace + prepare)
full(x0).item()

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
console.log(
  `peak RSS: ${(process.memoryUsage().rss / 1e9).toFixed(2)} GB`,
)

if (process.env.TYPENET_PROFILE === "1") {
  const native = await import("@typenet/native")
  console.log(native.takeProfile())
}
