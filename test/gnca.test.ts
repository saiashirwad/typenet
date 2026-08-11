// Parity against the PyTorch graph cellular automaton in
// ~/code/graph-cellular-automata: the model typenet is meant to be able
// to describe and train.
//
// The fixture is produced by test/fixtures/dump_gnca_reference.py, which
// records a k-NN graph, a set of weights, a three-step rollout, its loss
// and every gradient. This test rebuilds the graph from the same node
// positions, runs the same rollout, and checks the edge list, the final
// state, the loss and all five parameter gradients — in eager mode, the
// lazy interpreter, and natively.

import { readFileSync } from "node:fs"
import { afterEach, describe, expect, it } from "vitest"
import {
  Tensor,
  compile,
  configure,
  disableNative,
  isNativeAvailable,
  useNative
} from "../index.ts"
import { knnGraph, batchEdges } from "../examples/gnca/graphs.ts"
import {
  GraphNCA,
  aliveMask,
  graphTensors
} from "../examples/gnca/model.ts"

type AnyTensor = Tensor<any, any>

const reference = JSON.parse(
  readFileSync(
    new URL("./fixtures/gnca-reference.json", import.meta.url),
    "utf8"
  )
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
  "gate.bias"
]

afterEach(() => {
  configure({ lazy: false })
  disableNative()
})

function tensorFrom(
  values: ArrayLike<number>,
  shape: number[]
): AnyTensor {
  const t = Tensor.zeros(shape) as AnyTensor
  ;(t.data as Float32Array).set(values)
  return t
}

function expectClose(
  actual: ArrayLike<number>,
  expected: ArrayLike<number>,
  tolerance: number,
  label: string
): void {
  expect(actual.length, `${label}: length`).toBe(expected.length)
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
    `${label}: worst relative diff at ${at} (${actual[at]} vs ${expected[at]})`
  ).toBeLessThan(tolerance)
}

const N = reference.nodes
const C = reference.channels
const B = reference.batch

/** The k-NN graph rebuilt from the reference's node positions. */
const pos = {
  data: Float32Array.from(reference.pos),
  n: N,
  dim: 2
}
const edges = knnGraph(pos, reference.k)

function buildModel(): {
  model: GraphNCA
  params: AnyTensor[]
} {
  const model = new GraphNCA(C, reference.hidden)
  const params = model.parameters()
  params.forEach((p, i) => {
    const values = reference.weights[PARAM_NAMES[i]!]!
    expect(
      p.numel,
      `${PARAM_NAMES[i]} element count`
    ).toBe(values.length)
    ;(p.data as Float32Array).set(values)
  })
  return { model, params }
}

/** The rollout the fixture recorded: alive-mask, step, mask again. */
function rollout(
  model: GraphNCA,
  x0: AnyTensor,
  graph: ReturnType<typeof graphTensors>
): { state: AnyTensor; loss: AnyTensor; mse: AnyTensor } {
  const target = tensorFrom(reference.target, [B * N, 4])
  let state = x0
  for (let i = 0; i < reference.steps; i++)
    state = model
      .forward(state, graph, 1)
      .mul(aliveMask(state, graph))
  const mse = state
    .narrow(1, 0, 4)
    .sub(target)
    .pow(2)
    .mean() as AnyTensor
  const loss = mse.add(
    state.sub(state.clamp(-1, 1)).abs().mean()
  ) as AnyTensor
  return { state, loss, mse }
}

describe("graph cellular automaton, against PyTorch", () => {
  it("rebuilds the same k-NN graph from the same positions", () => {
    expect(edges.count).toBe(reference.edges.src.length)
    expect(Array.from(edges.src)).toEqual(reference.edges.src)
    expect(Array.from(edges.dst)).toEqual(reference.edges.dst)
  })

  it("has the parameter shapes PyTorch has, transposed", () => {
    const { params } = buildModel()
    expect(params.map(p => p.shape)).toEqual([
      [3 * C + 1, reference.hidden],
      [reference.hidden],
      [reference.hidden, C],
      [3 * C, C],
      [C]
    ])
  })

  it.each([
    { label: "eager", lazy: false, native: false },
    { label: "lazy interpreter", lazy: true, native: false },
    { label: "native", lazy: true, native: true }
  ])(
    "matches the forward rollout and every gradient ($label)",
    ({ lazy, native }) => {
      if (native && !isNativeAvailable()) return
      const { model, params } = buildModel()
      const graph = graphTensors(batchEdges(edges, B, N), B * N)
      const x0 = tensorFrom(reference.x0, [
        B * N,
        C
      ]).requires_grad() as AnyTensor
      if (native) useNative()
      configure({ lazy })
      const { state, loss, mse } = rollout(model, x0, graph)
      loss.backward()
      // Agreement is limited only by f32 reassociation across three
      // rolled-out steps of matmuls, gathers and scatter-adds: the
      // measured worst case is 8e-6 relative, and the fixture itself is
      // rounded to seven digits.
      expectClose([mse.item()], [reference.mse], 1e-5, "mse")
      expectClose([loss.item()], [reference.loss], 1e-5, "loss")
      expectClose(state.data, reference.state, 3e-5, "state")
      expectClose(x0.grad!.data, reference.xGrad, 3e-5, "grad of x0")
      params.forEach((p, i) => {
        const name = PARAM_NAMES[i]!
        expect(p.grad, `${name}: gradient`).not.toBeNull()
        expectClose(
          p.grad!.data,
          reference.gradients[name]!,
          3e-5,
          `grad of ${name}`
        )
      })
    }
  )

  it("matches through a compiled training step", () => {
    // compile() traces the same rollout and replays it, so the loss it
    // returns has to be the reference's on the first call.
    const { model, params } = buildModel()
    const graph = graphTensors(batchEdges(edges, B, N), B * N)
    if (isNativeAvailable()) useNative()
    const step = compile((input: AnyTensor) => {
      const { loss } = rollout(
        model,
        input.requires_grad() as AnyTensor,
        graph
      )
      loss.backward()
      return loss
    })
    const value = step(
      tensorFrom(reference.x0, [B * N, C])
    ).item()
    expectClose([value], [reference.loss], 1e-5, "compiled loss")
    params.forEach((p, i) =>
      expectClose(
        p.grad!.data,
        reference.gradients[PARAM_NAMES[i]!]!,
        3e-5,
        `compiled grad of ${PARAM_NAMES[i]}`
      )
    )
  })

  it("redraws the update mask each step of a compiled rollout", () => {
    // With updateRate 0.5 the mask gates about half the nodes, so two
    // calls of the same compiled step must not agree.
    const { model } = buildModel()
    const graph = graphTensors(batchEdges(edges, B, N), B * N)
    if (isNativeAvailable()) useNative()
    const step = compile((input: AnyTensor) =>
      model.forward(input, graph, 0.5)
    )
    const x0 = tensorFrom(reference.x0, [B * N, C])
    const first = Array.from(step(x0).data)
    const second = Array.from(step(x0).data)
    expect(second).not.toEqual(first)
  })
})
