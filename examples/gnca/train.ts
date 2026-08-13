"use tsover"

// Ported from ~/code/graph-cellular-automata/scripts/train.py.

import { writeFileSync } from "node:fs"
import { Adam, clipGradNorm, compile, configure, isNativeAvailable, nativeDevice, nativeDeviceMode, normal, Tensor, useNative } from "../../index.ts"
import { fromFlat } from "../../src/tensor.ts"
import { loadRule, readCheckpoint, saveCheckpoint } from "./checkpoint.ts"
import * as damage from "./damage.ts"
import { batchEdges, type Edges, knnGraph, nearestNode, type Points, randomGeometricGraph, wattsStrogatzGraph } from "./graphs.ts"
import { aliveMask, GraphNCA, type GraphTensors, graphTensors, seedState } from "./model.ts"
import { loadCloud, POINTCLOUDS } from "./pointclouds.ts"
import { renderSvg, visible } from "./render.ts"
import { rng } from "./rng.ts"
import { type Target, TARGETS } from "./targets.ts"

type AnyTensor = Tensor<any, any>

const flags = parseFlags(process.argv.slice(2))
const options = {
  nodes: number("nodes", 1024),
  k: number("k", 8),
  channels: number("channels", 16),
  hidden: number("hidden", 128),
  steps: number("steps", 8000),
  pool: number("pool", 512),
  batch: number("batch", 8),
  damage: number("damage", 3),
  horizon: [
    number("horizon-min", 48),
    number("horizon-max", 80),
  ],
  /**
   * How many distinct rollout lengths to draw from. Each gets its own
   * compiled graph, since a graph traced once has a fixed depth.
   */
  buckets: number("buckets", 5),
  noise: number("noise", 0.02),
  /** Penalty on state magnitude outside [-1, 1]. */
  overflow: number("overflow", 1),
  lr: number("lr", 5e-4),
  updateRate: number("update-rate", 0.5),
  clip: number("clip", 1),
  target: text("target", "heart"),
  tag: text("tag", ""),
  initFrom: text("init-from", ""),
  graph: text("graph", "auto"),
  clouds: text(
    "clouds",
    "../graph-cellular-automata/data/pointclouds",
  ),
  beta: number("beta", 0.05),
  seed: number("seed", 0),
  report: number("report", 20),
  probe: number("probe", 500),
  native: !flags.has("no-native"),
} as const

function parseFlags(argv: string[]): Map<string, string> {
  const out = new Map<string, string>()
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i]!
    if (!arg.startsWith("--")) continue
    const name = arg.slice(2)
    const next = argv[i + 1]
    if (next !== undefined && !next.startsWith("--")) {
      out.set(name, next)
      i++
    } else out.set(name, "")
  }
  return out
}

function number(name: string, fallback: number): number {
  const raw = flags.get(name)
  if (raw === undefined || raw === "") return fallback
  const value = Number(raw)
  if (!Number.isFinite(value)) {
    throw new Error(
      `--${name} expects a number, got ${raw}`,
    )
  }
  return value
}

function text(name: string, fallback: string): string {
  return flags.get(name) || fallback
}

configure({ seed: options.seed })
if (options.native && isNativeAvailable()) useNative()

const isCloud = options.target in POINTCLOUDS
const useRing = !isCloud
  && (options.graph === "ws"
    || (options.graph === "auto" && options.target === "ring"))
if (!isCloud && !TARGETS[options.target]) {
  throw new Error(
    `unknown target ${options.target}; patterns: `
      + `${Object.keys(TARGETS).join(", ")}; surface clouds: `
      + `${Object.keys(POINTCLOUDS).join(", ")}`,
  )
}

let pos: Points
let edges: Edges
let targetData: Target
let seedAt: number[]
let kind: string

if (isCloud) {
  const path = `${options.clouds}/${options.target}.npz`
  let cloud
  try {
    cloud = loadCloud(path, { nodes: options.nodes })
  } catch (error) {
    throw new Error(
      `could not load the ${options.target} cloud from ${path}: `
        + `${error instanceof Error ? error.message : String(error)}\n`
        + `Point --clouds at a directory of .npz clouds; the reference `
        + `repo builds them with scripts/fetch_pointclouds.py.`,
    )
  }
  pos = cloud.pos
  targetData = cloud.target
  seedAt = cloud.seedAt
  // Clouds are denser than a random cube, so they want more neighbours.
  edges = knnGraph(pos, flags.has("k") ? options.k : 12)
  kind = "surface cloud"
} else {
  const spec = TARGETS[options.target]!
  seedAt = spec.seedAt
  const built = useRing
    ? wattsStrogatzGraph({
      nodes: options.nodes,
      k: options.k,
      beta: options.beta,
      seed: options.seed,
    })
    : randomGeometricGraph({
      nodes: options.nodes,
      k: options.k,
      seed: options.seed,
      dim: seedAt.length,
    })
  pos = built.pos
  edges = built.edges
  targetData = spec.build(pos)
  kind = useRing ? "ring" : "geometric"
}

const dim = pos.dim
const N = pos.n
const C = options.channels
const B = options.batch
const center = nearestNode(pos, seedAt)
const inPattern = damage.patternNodes(targetData)

console.log(
  `${options.target}: ${dim}-d ${kind}, `
    + `${N} nodes, ${edges.count} edges, ${inPattern.length} in the pattern`,
)
console.log(
  `backend: ${
    isNativeAvailable() && options.native
      ? `native, ${nativeDeviceMode()} device `
        + `(accelerator available: ${nativeDevice()})`
      : "interpreter"
  }`,
)

const single = graphTensors(edges, N)
const batched = graphTensors(batchEdges(edges, B, N), B * N)

const targetTiled = tensorFrom(tile(targetData, B), [
  B * N,
  4,
])

function tensorFrom(
  data: Float32Array,
  shape: number[],
): AnyTensor {
  return fromFlat(data, shape)
}

function tile(
  data: Float32Array,
  times: number,
): Float32Array {
  const out = new Float32Array(data.length * times)
  for (let i = 0; i < times; i++) {
    out.set(data, i * data.length)
  }
  return out
}

const model = new GraphNCA(C, options.hidden)
const params = model.parameters()
const optimizer = new Adam(params, { lr: options.lr })

function rollout(
  x0: AnyTensor,
  steps: number,
  // Dims come from command-line flags, so they are `number` — a wildcard
  // the shape algebra accepts anywhere. A model written with literal
  // channel counts gets them checked; a CLI cannot.
  graph: GraphTensors<number>,
): AnyTensor {
  let x = x0
  for (let i = 0; i < steps; i++) {
    const alive = aliveMask(x, graph)
    x = model
      .forward(x, graph, options.updateRate)
      .mul(alive)
  }
  return x
}

function visibleLoss(
  x: AnyTensor,
  against: AnyTensor,
): AnyTensor {
  return x.narrow(1, 0, 4).sub(against).pow(2).mean()
}

type StepResult = [AnyTensor, AnyTensor, AnyTensor]

const rollouts = new Map<
  number,
  (x0: AnyTensor) => StepResult
>()

/**
 * Compiled per rollout length. A graph traced once has a fixed depth, so
 * a variable-length rollout means one compiled graph per length; the
 * reference samples any length in [48, 80], and this draws from
 * `buckets` evenly spaced lengths in the same range instead.
 */
function trainStep(
  x0: Float32Array,
  steps: number,
): { loss: number; mse: number; state: Float32Array } {
  let step = rollouts.get(steps)
  if (!step) {
    step = compile((input: AnyTensor): StepResult => {
      const noised = options.noise > 0
        ? input.add(
          (
            normal([B * N, C] as [
              number,
              number,
            ]) as AnyTensor
          ).mul(options.noise),
        )
        : input
      const x = rollout(noised, steps, batched)
      const mse = visibleLoss(x, targetTiled)
      const loss = options.overflow > 0
        ? mse.add(
          x
            .sub(x.clamp(-1, 1))
            .abs()
            .mean()
            .mul(options.overflow),
        )
        : mse
      optimizer.zeroGrad()
      loss.backward()
      clipGradNorm(params, options.clip)
      optimizer.step()
      return [loss, mse, x]
    })
    rollouts.set(steps, step)
  }
  const [loss, mse, state] = step(
    tensorFrom(x0, [B * N, C]),
  )
  return {
    loss: loss.item(),
    mse: mse.item(),
    state: state.data as Float32Array,
  }
}

const horizons = Array.from(
  { length: options.buckets },
  (_, i) =>
    Math.round(
      options.horizon[0]!
        + ((options.horizon[1]! - options.horizon[0]!) * i)
          / Math.max(options.buckets - 1, 1),
    ),
)

const pool = new Float32Array(options.pool * N * C)
for (let i = 0; i < options.pool; i++) {
  pool.set(seedState(1, N, C, center), i * N * C)
}

const random = rng(options.seed + 1)

function sampleError(
  state: Float32Array,
  offset: number,
): number {
  let total = 0
  for (let node = 0; node < N; node++) {
    for (let c = 0; c < 4; c++) {
      const delta = state[offset + node * C + c]!
        - targetData[node * 4 + c]!
      total += delta * delta
    }
  }
  return total / (N * 4)
}

function wipe(
  batchState: Float32Array,
  b: number,
  nodes: Int32Array,
): void {
  for (const node of nodes) {
    batchState.fill(
      0,
      (b * N + node) * C,
      (b * N + node) * C + C,
    )
  }
}

const probeHole = damage.ball(pos, inPattern, {
  frac: 0.25,
  center: inPattern[Math.floor(inPattern.length / 2)]!,
})

function healProbe(
  grow = 80,
  heal = 160,
): {
  grown: number
  healed: number
  grownState: Float32Array
  healedState: Float32Array
} {
  const state = seedState(1, N, C, center)
  const run = (from: Float32Array, steps: number) => {
    let x = tensorFrom(from, [N, C]) as AnyTensor
    for (let i = 0; i < steps; i++) {
      const alive = aliveMask(x, single)
      x = model
        .forward(x, single, options.updateRate)
        .mul(alive)
        .detach()
    }
    return x.data as Float32Array
  }
  configure({ lazy: true })
  const grownState = run(state, grow)
  const grownError = sampleError(grownState, 0)
  const wounded = grownState.slice()
  for (const node of probeHole) {
    wounded.fill(0, node * C, node * C + C)
  }
  const healedState = run(wounded, heal)
  configure({ lazy: false })
  return {
    grown: grownError,
    healed: sampleError(healedState, 0),
    grownState,
    healedState,
  }
}

const checkpointPath = text("save", "")
  || `runs/gnca-${options.target}${options.tag ? `-${options.tag}` : ""}.json`
const svgPath = text("svg", "")
  || checkpointPath.replace(/\.json$/, ".svg")

function save(step: number): void {
  saveCheckpoint(
    checkpointPath,
    {
      step,
      channels: C,
      hidden: options.hidden,
      target: options.target,
      center,
      pos: Array.from(pos.data),
      dim: pos.dim,
      edges: {
        src: Array.from(edges.src),
        dst: Array.from(edges.dst),
      },
      targetRgba: Array.from(targetData),
    },
    params,
  )
}

function drawProbe(probe: {
  grownState: Float32Array
  healedState: Float32Array
}): void {
  writeFileSync(
    svgPath,
    renderSvg(
      pos.data,
      pos.dim,
      [
        visible(targetData, N, 4),
        visible(probe.grownState, N, C),
        visible(probe.healedState, N, C),
      ],
      { labels: ["target", "grown", "healed"], nodes: N },
    ),
  )
}

if (options.initFrom) {
  const padded = loadRule(
    params,
    readCheckpoint(options.initFrom),
  )
  console.log(
    `warm-started from ${options.initFrom}`
      + (padded > 0 ? ` (zero-padded ${padded} weights)` : ""),
  )
}

console.log(
  `training ${options.steps} steps, batch ${B}, `
    + `rollout lengths ${horizons.join("/")}, `
    + `${params.length} parameter tensors, `
    + `${params.reduce((n, p) => n + p.numel, 0)} weights`,
)

const batchState = new Float32Array(B * N * C)
const chosen = new Int32Array(B)
let lastReport = Date.now()

for (let step = 1; step <= options.steps; step++) {
  for (let b = 0; b < B; b++) {
    chosen[b] = random.int(options.pool)
  }
  for (let b = 0; b < B; b++) {
    batchState.set(
      pool.subarray(
        chosen[b]! * N * C,
        (chosen[b]! + 1) * N * C,
      ),
      b * N * C,
    )
  }

  const order = Array.from({ length: B }, (_, b) => b).sort(
    (a, b) =>
      sampleError(batchState, b * N * C)
      - sampleError(batchState, a * N * C),
  )
  const ranked = new Float32Array(B * N * C)
  const rankedChoice = new Int32Array(B)
  order.forEach((b, i) => {
    ranked.set(
      batchState.subarray(b * N * C, (b + 1) * N * C),
      i * N * C,
    )
    rankedChoice[i] = chosen[b]!
  })
  batchState.set(ranked)
  chosen.set(rankedChoice)

  if (step % 8 === 0) {
    batchState.set(seedState(1, N, C, center), 0)
  }

  for (let i = 0; i < Math.min(options.damage, B); i++) {
    const b = B - 1 - i
    wipe(
      batchState,
      b,
      i === 0
        ? damage.scatter(inPattern, { random })
        : damage.ball(pos, inPattern, {
          frac: random.range(0.2, 0.5),
          random,
        }),
    )
  }

  const horizon = horizons[random.int(horizons.length)]!
  const { loss, mse, state } = trainStep(
    batchState,
    horizon,
  )

  let worst = 0
  let worstError = -Infinity
  for (let b = 0; b < B; b++) {
    const error = sampleError(state, b * N * C)
    if (error > worstError) {
      worstError = error
      worst = b
    }
  }
  for (let b = 0; b < B; b++) {
    pool.set(
      b === worst
        ? seedState(1, N, C, center)
        : state.subarray(b * N * C, (b + 1) * N * C),
      chosen[b]! * N * C,
    )
  }

  if (step % options.report === 0 || step === 1) {
    const now = Date.now()
    const rate = step === 1 ? 0 : (
      (options.report * 1000)
      / Math.max(now - lastReport, 1)
    )
    lastReport = now
    console.log(
      `step ${String(step).padStart(6)}  loss ${loss.toFixed(6)}  `
        + `mse ${mse.toFixed(6)}  ${rate.toFixed(2)} steps/s`,
    )
  }
  if (step % options.probe === 0) {
    const probe = healProbe()
    console.log(
      `    probe: grown ${probe.grown.toFixed(4)}  `
        + `healed ${probe.healed.toFixed(4)}`,
    )
    save(step)
    drawProbe(probe)
  }
}

const probe = healProbe()
save(options.steps)
drawProbe(probe)
console.log(
  `done. probe: grown ${probe.grown.toFixed(4)}  `
    + `healed ${probe.healed.toFixed(4)}`,
)
console.log(`saved ${checkpointPath} and ${svgPath}`)
