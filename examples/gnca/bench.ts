"use tsover"

// Where the time goes in a graph-CA training step. Times a compiled
// step at several rollout lengths and graph sizes, so the fixed cost per
// step separates from the cost per rolled-out time step.
//
//     pnpm vite-node examples/gnca/bench.ts

import {
  Adam,
  Tensor,
  clipGradNorm,
  compile,
  configure,
  disableNative,
  isNativeAvailable,
  nativeDevice,
  nativeDeviceMode,
  useNative
} from "../../index.ts"
import {
  batchEdges,
  randomGeometricGraph
} from "./graphs.ts"
import {
  GraphNCA,
  aliveMask,
  graphTensors,
  seedState
} from "./model.ts"

type AnyTensor = Tensor<any, any>

function time(label: string, runs: number, fn: () => void) {
  fn() // warm up (traces the graph)
  const start = performance.now()
  for (let i = 0; i < runs; i++) fn()
  const ms = (performance.now() - start) / runs
  console.log(`  ${label.padEnd(38)} ${ms.toFixed(1)} ms`)
  return ms
}

function bench(nodes: number, batch: number, horizons: number[]) {
  const C = 16
  const { pos, edges } = randomGeometricGraph({ nodes, dim: 2 })
  const graph = graphTensors(
    batchEdges(edges, batch, nodes),
    batch * nodes
  )
  const model = new GraphNCA(C)
  const params = model.parameters()
  const optimizer = new Adam(params, { lr: 5e-4 })
  const target = Tensor.rand([batch * nodes, 4]) as AnyTensor
  const x0 = Tensor.zeros([batch * nodes, C]) as AnyTensor
  ;(x0.data as Float32Array).set(
    seedState(batch, nodes, C, 0)
  )

  console.log(
    `\n${nodes} nodes x ${batch} = ${batch * nodes}, ` +
      `${edges.count * batch} edges`
  )
  const rollout = (x: AnyTensor, steps: number) => {
    for (let i = 0; i < steps; i++)
      x = model.forward(x, graph).mul(aliveMask(x, graph))
    return x
  }
  for (const steps of horizons) {
    const forward = compile((input: AnyTensor) =>
      rollout(input, steps).narrow(1, 0, 4).sub(target).pow(2).mean()
    )
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
    const f = time(`forward only, ${steps} steps`, 3, () =>
      forward(x0).item()
    )
    const t = time(`forward + backward + Adam, ${steps} steps`, 3, () =>
      full(x0).item()
    )
    console.log(
      `  ${"per rolled-out step".padEnd(38)} ` +
        `${(f / steps).toFixed(2)} / ${(t / steps).toFixed(2)} ms`
    )
  }
}

if (isNativeAvailable()) useNative()
const label =
  isNativeAvailable() ?
    `native, ${nativeDeviceMode()} device (accelerator: ${nativeDevice()})`
  : "interpreter"
configure({ seed: 1 })
console.log(`backend: ${label}`)
bench(256, 4, [8, 16, 32])
bench(1024, 8, [8, 16])

if (isNativeAvailable()) {
  disableNative()
  console.log("\nbackend: interpreter (same shapes, for comparison)")
  bench(256, 4, [8, 16])
}
