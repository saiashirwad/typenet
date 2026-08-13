import { readFileSync } from "node:fs"
import { afterEach, describe, expect, it } from "vitest"
import { batchEdges, knnGraph, nearestNode, randomGeometricGraph } from "../examples/gnca/graphs.ts"
import { aliveMask, GraphNCA, graphTensors, seedState } from "../examples/gnca/model.ts"
import { TARGETS } from "../examples/gnca/targets.ts"
import { Adam, clipGradNorm, compile, configure, disableNative, isNativeAvailable, normal, Tensor, useNative } from "../index.ts"
import { fromFlat } from "../src/tensor.ts"

type AnyTensor = Tensor<any>

const reference = JSON.parse(
  readFileSync(
    new URL(
      "./fixtures/gnca-reference.json",
      import.meta.url,
    ),
    "utf8",
  ),
) as {
  channels: number
  hidden: number
  nodes: number
  k: number
  steps: number
  batch: number
  pos: number[]
  edges: { src: number[]; dst: number[] }
  weights: Record<string, number[]>
  gradients: Record<string, number[]>
  x0: number[]
  target: number[]
  state: number[]
  mse: number
  loss: number
  xGrad: number[]
}

// PyTorch parameter names, in the order typenet's parameters() yields.
const PARAM_NAMES = [
  "net.0.weight",
  "net.0.bias",
  "net.2.weight",
  "gate.weight",
  "gate.bias",
]

afterEach(() => {
  configure({ lazy: false })
  disableNative()
})

function tensorFrom(
  values: ArrayLike<number>,
  shape: number[],
): AnyTensor {
  return fromFlat(values, shape)
}

function expectClose(
  actual: ArrayLike<number>,
  expected: ArrayLike<number>,
  tolerance: number,
  label: string,
): void {
  expect(actual.length, `${label}: length`).toBe(
    expected.length,
  )
  let worst = 0
  let at = -1
  for (let i = 0; i < expected.length; i++) {
    const diff = Math.abs(actual[i]! - expected[i]!)
    const scale = Math.max(1, Math.abs(expected[i]!))
    if (diff / scale > worst) {
      worst = diff / scale
      at = i
    }
  }
  expect(
    worst,
    `${label}: worst relative diff at ${at} (${actual[at]} vs ${expected[at]})`,
  ).toBeLessThan(tolerance)
}

const N = reference.nodes
const C = reference.channels
const B = reference.batch

const pos = {
  data: Float32Array.from(reference.pos),
  n: N,
  dim: 2,
}
const edges = knnGraph(pos, reference.k)

function buildModel(): {
  // The fixture's channel count is read from JSON, so it is `number`
  // rather than a literal — see the type tests for the literal case.
  model: GraphNCA<number, number>
  params: AnyTensor[]
} {
  const model = new GraphNCA(C, reference.hidden)
  const params = model.parameters()
  params.forEach((p, i) => {
    const values = reference.weights[PARAM_NAMES[i]!]!
    expect(p.numel, `${PARAM_NAMES[i]} element count`).toBe(
      values.length,
    )
    ;(p.data as Float32Array).set(values)
  })
  return { model, params }
}

function rollout(
  model: GraphNCA<number, number>,
  x0: AnyTensor,
  graph: ReturnType<typeof graphTensors>,
): { state: AnyTensor; loss: AnyTensor; mse: AnyTensor } {
  const target = tensorFrom(reference.target, [B * N, 4])
  let state = x0
  for (let i = 0; i < reference.steps; i++) {
    state = model
      .forward(state, graph, 1)
      .mul(aliveMask(state, graph))
  }
  const mse = state
    .narrow(1, 0, 4)
    .sub(target)
    .pow(2)
    .mean() as AnyTensor
  const loss = mse.add(
    state.sub(state.clamp(-1, 1)).abs().mean(),
  ) as AnyTensor
  return { state, loss, mse }
}

describe("graph cellular automaton, against PyTorch", () => {
  it("rebuilds the same k-NN graph from the same positions", () => {
    expect(edges.count).toBe(reference.edges.src.length)
    expect(Array.from(edges.src)).toEqual(
      reference.edges.src,
    )
    expect(Array.from(edges.dst)).toEqual(
      reference.edges.dst,
    )
  })

  it("has the parameter shapes PyTorch has, transposed", () => {
    const { params } = buildModel()
    expect(params.map(p => p.shape)).toEqual([
      [3 * C + 1, reference.hidden],
      [reference.hidden],
      [reference.hidden, C],
      [3 * C, C],
      [C],
    ])
  })

  it.each([
    { label: "eager", lazy: false, native: false },
    {
      label: "lazy interpreter",
      lazy: true,
      native: false,
    },
    { label: "native", lazy: true, native: true },
  ])(
    "matches the forward rollout and every gradient ($label)",
    ({ lazy, native }) => {
      if (native && !isNativeAvailable()) return
      const { model, params } = buildModel()
      const graph = graphTensors(
        batchEdges(edges, B, N),
        B * N,
      )
      const x0 = tensorFrom(reference.x0, [
        B * N,
        C,
      ]).requires_grad() as AnyTensor
      if (native) useNative()
      configure({ lazy })
      const { state, loss, mse } = rollout(model, x0, graph)
      loss.backward()
      // Agreement is limited only by f32 reassociation across three
      // rolled-out steps of matmuls, gathers and scatter-adds: the
      // measured worst case is 8e-6 relative, and the fixture itself is
      // rounded to seven digits.
      expectClose(
        [mse.item()],
        [reference.mse],
        1e-5,
        "mse",
      )
      expectClose(
        [loss.item()],
        [reference.loss],
        1e-5,
        "loss",
      )
      expectClose(
        state.data,
        reference.state,
        3e-5,
        "state",
      )
      expectClose(
        x0.grad!.data,
        reference.xGrad,
        3e-5,
        "grad of x0",
      )
      params.forEach((p, i) => {
        const name = PARAM_NAMES[i]!
        expect(p.grad, `${name}: gradient`).not.toBeNull()
        expectClose(
          p.grad!.data,
          reference.gradients[name]!,
          3e-5,
          `grad of ${name}`,
        )
      })
    },
  )

  it("matches through a compiled training step", () => {
    const { model, params } = buildModel()
    const graph = graphTensors(
      batchEdges(edges, B, N),
      B * N,
    )
    if (isNativeAvailable()) useNative()
    const step = compile((input: AnyTensor) => {
      const { loss } = rollout(
        model,
        input.requires_grad() as AnyTensor,
        graph,
      )
      loss.backward()
      return loss
    })
    const value = step(
      tensorFrom(reference.x0, [B * N, C]),
    ).item()
    expectClose(
      [value],
      [reference.loss],
      1e-5,
      "compiled loss",
    )
    params.forEach((p, i) =>
      expectClose(
        p.grad!.data,
        reference.gradients[PARAM_NAMES[i]!]!,
        3e-5,
        `compiled grad of ${PARAM_NAMES[i]}`,
      )
    )
  })

  it("trains: the loss falls over compiled steps", () => {
    const nodes = 96
    const batch = 2
    const channels = 8
    const steps = 6
    const built = randomGeometricGraph({
      nodes,
      dim: 2,
      seed: 3,
    })
    const targetData = TARGETS.heart!.build(built.pos)
    const center = nearestNode(
      built.pos,
      TARGETS.heart!.seedAt,
    )
    const graph = graphTensors(
      batchEdges(built.edges, batch, nodes),
      batch * nodes,
    )
    const rows = batch * nodes
    const tiled = new Float32Array(rows * 4)
    for (let b = 0; b < batch; b++) {
      tiled.set(targetData, b * nodes * 4)
    }
    const target = tensorFrom(tiled, [rows, 4])

    configure({ seed: 5 })
    if (isNativeAvailable()) useNative()
    const model = new GraphNCA(channels, 32)
    const params = model.parameters()
    const optimizer = new Adam(params, { lr: 3e-3 })
    const step = compile((input: AnyTensor) => {
      let x = input.add(
        (normal([rows, channels]) as AnyTensor).mul(0.02),
      )
      for (let i = 0; i < steps; i++) {
        x = model
          .forward(x, graph, 0.5)
          .mul(aliveMask(x, graph))
      }
      const loss = x
        .narrow(1, 0, 4)
        .sub(target)
        .pow(2)
        .mean()
        .add(x.sub(x.clamp(-1, 1)).abs().mean())
      optimizer.zeroGrad()
      loss.backward()
      clipGradNorm(params, 1)
      optimizer.step()
      return loss
    })

    const seed = tensorFrom(
      seedState(batch, nodes, channels, center),
      [rows, channels],
    )
    const losses: number[] = []
    for (let i = 0; i < 40; i++) {
      losses.push(step(seed).item())
    }
    const first = losses[0]!
    const last = losses[losses.length - 1]!
    expect(Number.isFinite(last)).toBe(true)
    expect(last, `${first} -> ${last}`).toBeLessThan(
      first * 0.7,
    )
    expect(
      Array.from(params[2]!.data).some(v => v !== 0),
    ).toBe(true)
  })

  it("redraws the update mask each step of a compiled rollout", () => {
    const { model } = buildModel()
    const graph = graphTensors(
      batchEdges(edges, B, N),
      B * N,
    )
    if (isNativeAvailable()) useNative()
    const step = compile((input: AnyTensor) => model.forward(input, graph, 0.5))
    const x0 = tensorFrom(reference.x0, [B * N, C])
    const first = Array.from(step(x0).data)
    const second = Array.from(step(x0).data)
    expect(second).not.toEqual(first)
  })
})
