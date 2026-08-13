import { existsSync, mkdtempSync } from "node:fs"
import { tmpdir } from "node:os"
import { join } from "node:path"
import { afterEach, describe, expect, it } from "vitest"
import { checkpointGraph, loadRule, readCheckpoint, saveCheckpoint } from "../examples/gnca/checkpoint.ts"
import { knnGraph, randomGeometricGraph } from "../examples/gnca/graphs.ts"
import { GraphNCA, graphTensors } from "../examples/gnca/model.ts"
import { loadCloud, POINTCLOUDS } from "../examples/gnca/pointclouds.ts"
import { renderSvg, visible } from "../examples/gnca/render.ts"
import { heart } from "../examples/gnca/targets.ts"
import { configure, disableNative, Tensor } from "../index.ts"

type AnyTensor = Tensor<any>

const scratch = mkdtempSync(join(tmpdir(), "typenet-gnca-"))

afterEach(() => {
  configure({ lazy: false })
  disableNative()
})

const graph = randomGeometricGraph({
  nodes: 60,
  dim: 2,
  seed: 2,
})
const target = heart(graph.pos)

function meta(step: number) {
  return {
    step,
    channels: 8,
    hidden: 16,
    target: "heart",
    center: 0,
    pos: Array.from(graph.pos.data),
    dim: 2,
    edges: {
      src: Array.from(graph.edges.src),
      dst: Array.from(graph.edges.dst),
    },
    targetRgba: Array.from(target),
  }
}

describe("checkpoints", () => {
  it("round-trips every weight exactly", () => {
    const saved = new GraphNCA(8, 16)
    // Non-zero everywhere, so the zero-initialised layers cannot hide a
    // parameter that never got written.
    saved.parameters().forEach((p, i) => {
      const data = p.data as Float32Array
      for (let j = 0; j < data.length; j++) {
        data[j] = Math.sin(j * 0.7 + i) * 0.4
      }
    })
    const path = join(scratch, "round-trip.json")
    saveCheckpoint(path, meta(120), saved.parameters())

    const loaded = new GraphNCA(8, 16)
    const padded = loadRule(
      loaded.parameters(),
      readCheckpoint(path),
    )
    expect(padded).toBe(0)
    loaded.parameters().forEach((p, i) => {
      expect(Array.from(p.data), `parameter ${i}`).toEqual(
        Array.from(saved.parameters()[i]!.data),
      )
    })
  })

  it("carries the graph it was trained on", () => {
    const path = join(scratch, "graph.json")
    saveCheckpoint(
      path,
      meta(7),
      new GraphNCA(8, 16).parameters(),
    )
    const checkpoint = readCheckpoint(path)
    expect(checkpoint.step).toBe(7)
    const rebuilt = checkpointGraph(checkpoint)
    expect(rebuilt.pos.n).toBe(graph.pos.n)
    expect(rebuilt.edges.count).toBe(graph.edges.count)
    expect(Array.from(rebuilt.edges.src)).toEqual(
      Array.from(graph.edges.src),
    )
    expect(
      Array.from(knnGraph(rebuilt.pos, 8).src),
    ).toEqual(Array.from(graph.edges.src))
  })

  it("warm-starts a wider percept without changing the rule", () => {
    const narrow = new GraphNCA(8, 16)
    narrow.parameters().forEach((p, i) => {
      const data = p.data as Float32Array
      for (let j = 0; j < data.length; j++) {
        data[j] = Math.cos(j * 0.3 + i) * 0.3
      }
    })
    const path = join(scratch, "narrow.json")
    saveCheckpoint(path, meta(1), narrow.parameters())

    const wide = new GraphNCA(8, 16)
    const first = wide.parameters()[0]!
    const grown = Tensor.zeros([
      first.shape[0]! + 3,
      first.shape[1]!,
    ]) as AnyTensor
    const params = [grown, ...wide.parameters().slice(1)]
    const padded = loadRule(params, readCheckpoint(path))
    expect(padded).toBe(3 * first.shape[1]!)

    const saved = narrow.parameters()[0]!.data
    const loaded = grown.data
    for (let i = 0; i < saved.length; i++) {
      expect(loaded[i]).toBe(saved[i])
    }
    for (let i = saved.length; i < loaded.length; i++) {
      expect(loaded[i]).toBe(0)
    }
  })

  it("rejects a checkpoint from a different model", () => {
    const path = join(scratch, "mismatch.json")
    saveCheckpoint(
      path,
      meta(1),
      new GraphNCA(8, 16).parameters(),
    )
    expect(() =>
      loadRule(
        new GraphNCA(16, 16).parameters(),
        readCheckpoint(path),
      )
    ).toThrow(/model wants/)
  })
})

describe("rendering", () => {
  it("draws one circle per live node, and skips the dead", () => {
    const nodes = graph.pos.n
    const svg = renderSvg(
      graph.pos.data,
      2,
      [visible(target, nodes, 4)],
      { nodes, labels: ["target"] },
    )
    const circles = svg.match(/<circle/g)?.length ?? 0
    const live = Array.from(
      { length: nodes },
      (_, i) => target[i * 4 + 3]!,
    ).filter(a => a >= 0.02).length
    expect(circles).toBe(live)
    expect(live).toBeGreaterThan(0)
    expect(svg).toContain("target")
    expect(svg.startsWith("<svg")).toBe(true)
  })

  it("puts three frames side by side", () => {
    const nodes = graph.pos.n
    const frame = visible(target, nodes, 4)
    const svg = renderSvg(
      graph.pos.data,
      2,
      [frame, frame, frame],
      {
        size: 100,
        nodes,
      },
    )
    // three panels of 100 with two 12px gaps
    expect(svg).toContain("width=\"324\"")
  })

  it("survives a rolled-out state with values outside [0, 1]", () => {
    const nodes = graph.pos.n
    const graphs = graphTensors(graph.edges, nodes)
    const model = new GraphNCA(8, 16)
    let x = Tensor.zeros([nodes, 8]) as AnyTensor
    ;(x.data as Float32Array).fill(5)
    configure({ lazy: true })
    x = model.forward(x, graphs, 1)
    configure({ lazy: false })
    const svg = renderSvg(
      graph.pos.data,
      2,
      [visible(x.data as Float32Array, nodes, 8)],
      { nodes },
    )
    expect(svg).not.toContain("NaN")
    expect(svg).toMatch(/fill="rgb\(255,255,255\)"/)
  })
})

// Surface point clouds — the targets the reference's own experiments use.
// The .npz reader has to handle ZIP64 (numpy writes it) and the values have
// to match the reference exactly, so the fixture records what Python
// produced for the same file.
const CLOUD_DIR = "../graph-cellular-automata/data/pointclouds"
const bunnyPath = `${CLOUD_DIR}/bunny.npz`
const haveBunny = existsSync(bunnyPath)

describe.skipIf(!haveBunny)("point clouds", () => {
  it("reads the bunny exactly as numpy does", () => {
    const cloud = loadCloud(bunnyPath)
    expect(cloud.pos.n).toBe(1536)
    expect(cloud.pos.dim).toBe(3)
    expect(cloud.seedAt).toEqual([0.5, 0.85, 0.55])

    // From gnca.pointclouds.load_cloud("bunny") in the reference repo.
    const firstPos = [0.114559, 0.579978, 0.507694]
    const firstRgba = [0.584566, 0.031965, 0, 1]
    firstPos.forEach((v, i) => expect(cloud.pos.data[i]).toBeCloseTo(v, 6))
    firstRgba.forEach((v, i) => expect(cloud.target[i]).toBeCloseTo(v, 6))

    let lo = Infinity
    let hi = -Infinity
    for (let i = 0; i < cloud.pos.n; i++) {
      lo = Math.min(lo, cloud.pos.data[i * 3]!)
      hi = Math.max(hi, cloud.pos.data[i * 3]!)
    }
    expect(lo).toBeCloseTo(0.08, 4)
    expect(hi).toBeCloseTo(0.92, 4)
    let alpha = 0
    for (let i = 0; i < cloud.pos.n; i++) {
      alpha += cloud.target[i * 4 + 3]!
    }
    expect(alpha).toBe(cloud.pos.n)
  })

  it("subsamples evenly, keeping the shape in the unit cube", () => {
    const cloud = loadCloud(bunnyPath, { nodes: 400 })
    expect(cloud.pos.n).toBe(400)
    for (let i = 0; i < cloud.pos.data.length; i++) {
      expect(cloud.pos.data[i]).toBeGreaterThanOrEqual(0)
      expect(cloud.pos.data[i]).toBeLessThanOrEqual(1)
    }
    const edges = knnGraph(cloud.pos, 12)
    expect(edges.count).toBeGreaterThan(400 * 12)
  })

  it("names the clouds the reference names", () => {
    expect(Object.keys(POINTCLOUDS).sort()).toEqual([
      "armadillo",
      "bunny",
      "spot",
      "teapot",
    ])
  })
})
