"use tsover"

// Train a graph cellular automaton to grow a pattern on a random graph
// from a single seed node, and to heal after damage.
//
//     pnpm gnca                      # train with the defaults
//     pnpm gnca --steps 50           # a quick smoke run
//     pnpm gnca --target star --nodes 512
//
// Ported from ~/code/graph-cellular-automata/scripts/train.py. The
// recipe is the same one: a sample pool so the rule sees partly grown
// states, damage on the best-formed samples so healing is what it
// learns, and a heal probe as the number the whole thing chases —
// training loss can fall while regeneration stays broken.

import {
  Adam,
  Tensor,
  clipGradNorm,
  compile,
  configure,
  isNativeAvailable,
  nativeDevice,
  nativeDeviceMode,
  normal,
  useNative
} from "../../index.ts"
import * as damage from "./damage.ts"
import {
  type Edges,
  batchEdges,
  nearestNode,
  randomGeometricGraph,
  wattsStrogatzGraph
} from "./graphs.ts"
import {
  GraphNCA,
  type GraphTensors,
  aliveMask,
  graphTensors,
  seedState
} from "./model.ts"
import { rng } from "./rng.ts"
import { TARGETS } from "./targets.ts"

type AnyTensor = Tensor<any, any>

// ------------------------------------------------------------- options ---

const flags = parseFlags(process.argv.slice(2))
const options = {
  nodes: number("nodes", 1024),
  k: number("k", 8),
  channels: number("channels", 16),
  hidden: number("hidden", 128),
  steps: number("steps", 8000),
  pool: number("pool", 512),
  batch: number("batch", 8),
  /** Samples per batch to damage; the best-scoring ones are hit. */
  damage: number("damage", 3),
  /** Rollout length range. Healing needs longer than growing. */
  horizon: [number("horizon-min", 48), number("horizon-max", 80)],
  /**
   * How many distinct rollout lengths to draw from. Each gets its own
   * compiled graph, since a graph traced once has a fixed depth — see
   * the note on `rollouts` below.
   */
  buckets: number("buckets", 5),
  noise: number("noise", 0.02),
  /** Penalty on state magnitude outside [-1, 1]. */
  overflow: number("overflow", 1),
  lr: number("lr", 5e-4),
  updateRate: number("update-rate", 0.5),
  clip: number("clip", 1),
  target: text("target", "heart"),
  graph: text("graph", "auto"),
  beta: number("beta", 0.05),
  seed: number("seed", 0),
  report: number("report", 20),
  probe: number("probe", 500),
  native: !flags.has("no-native")
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
  if (!Number.isFinite(value))
    throw new Error(`--${name} expects a number, got ${raw}`)
  return value
}

function text(name: string, fallback: string): string {
  return flags.get(name) || fallback
}

// --------------------------------------------------- graph and target ---

configure({ seed: options.seed })
if (options.native && isNativeAvailable()) useNative()

const spec = TARGETS[options.target]
if (!spec)
  throw new Error(
    `unknown target ${options.target}; try one of ${Object.keys(TARGETS).join(", ")}`
  )

const dim = spec.seedAt.length
const useRing =
  options.graph === "ws" ||
  (options.graph === "auto" && options.target === "ring")
const built =
  useRing ?
    wattsStrogatzGraph({
      nodes: options.nodes,
      k: options.k,
      beta: options.beta,
      seed: options.seed
    })
  : randomGeometricGraph({
      nodes: options.nodes,
      k: options.k,
      seed: options.seed,
      dim
    })
const pos = built.pos
const edges: Edges = built.edges
const N = pos.n
const C = options.channels
const B = options.batch
const targetData = spec.build(pos)
const center = nearestNode(pos, spec.seedAt)
const inPattern = damage.patternNodes(targetData)

console.log(
  `${options.target}: ${dim}-d ${useRing ? "ring" : "geometric"} graph, ` +
    `${N} nodes, ${edges.count} edges, ${inPattern.length} in the pattern`
)
console.log(
  `backend: ${
    isNativeAvailable() && options.native ?
      `native, ${nativeDeviceMode()} device ` +
      `(accelerator available: ${nativeDevice()})`
    : "interpreter"
  }`
)

// The rule sees a batch as one big graph: B copies side by side in the
// node dimension, with the edge list replicated at matching offsets.
const single = graphTensors(edges, N)
const batched = graphTensors(batchEdges(edges, B, N), B * N)

// Target as a tensor, once for a single copy and once tiled over the
// batch, so the loss is a plain subtraction.
const target = tensorFrom(targetData, [N, 4])
const targetTiled = tensorFrom(tile(targetData, B), [B * N, 4])

function tensorFrom(
  data: Float32Array,
  shape: number[]
): AnyTensor {
  const t = Tensor.zeros(shape) as AnyTensor
  ;(t.data as Float32Array).set(data)
  return t
}

function tile(data: Float32Array, times: number): Float32Array {
  const out = new Float32Array(data.length * times)
  for (let i = 0; i < times; i++) out.set(data, i * data.length)
  return out
}

// ------------------------------------------------------ model and loss ---

const model = new GraphNCA(C, options.hidden)
const params = model.parameters()
const optimizer = new Adam(params, { lr: options.lr })

/**
 * Roll the automaton forward `steps` times, masking dead nodes before
 * each step exactly as the growing NCA does.
 */
function rollout(
  x0: AnyTensor,
  steps: number,
  graph: GraphTensors
): AnyTensor {
  let x = x0
  for (let i = 0; i < steps; i++) {
    const alive = aliveMask(x, graph)
    x = model.forward(x, graph, options.updateRate).mul(alive)
  }
  return x
}

/** Squared error against the target on the four visible channels. */
function visibleLoss(
  x: AnyTensor,
  against: AnyTensor
): AnyTensor {
  return x.narrow(1, 0, 4).sub(against).pow(2).mean()
}

/**
 * One training step as a single graph: noise the inputs, roll out,
 * score, differentiate, clip, and apply the Adam update — all of it
 * evaluated in one pass over the graph.
 *
 * Compiled per rollout length. A graph traced once has a fixed depth, so
 * a variable-length rollout means one compiled graph per length; the
 * reference samples any length in [48, 80], and this draws from
 * `buckets` evenly spaced lengths in the same range instead. Variable
 * enough to keep the rule from locking onto one horizon, few enough
 * that the compiled graphs are worth their memory.
 */
const rollouts = new Map<
  number,
  (x0: AnyTensor) => [AnyTensor, AnyTensor]
>()

function trainStep(
  x0: Float32Array,
  steps: number
): { loss: number; mse: number; state: Float32Array } {
  let step = rollouts.get(steps)
  if (!step) {
    step = compile((input: AnyTensor) => {
      const noised =
        options.noise > 0 ?
          input.add(
            (normal([B * N, C] as [number, number]) as AnyTensor).mul(
              options.noise
            )
          )
        : input
      const x = rollout(noised, steps, batched)
      const mse = visibleLoss(x, targetTiled)
      // States that run away are the usual failure after damage, so
      // penalise magnitude outside [-1, 1] directly.
      const loss =
        options.overflow > 0 ?
          mse.add(
            x.sub(x.clamp(-1, 1)).abs().mean().mul(options.overflow)
          )
        : mse
      optimizer.zeroGrad()
      loss.backward()
      clipGradNorm(params, options.clip)
      optimizer.step()
      return [loss, mse, x] as any
    }) as any
    rollouts.set(steps, step!)
  }
  const inputs = tensorFrom(x0, [B * N, C])
  const [loss, mse, state] = (step as any)(inputs) as [
    AnyTensor,
    AnyTensor,
    AnyTensor
  ]
  return {
    loss: loss.item(),
    mse: mse.item(),
    state: state.data as Float32Array
  }
}

const horizons = Array.from({ length: options.buckets }, (_, i) =>
  Math.round(
    options.horizon[0]! +
      ((options.horizon[1]! - options.horizon[0]!) * i) /
        Math.max(options.buckets - 1, 1)
  )
)

// ------------------------------------------------------------- the pool ---
// The sample pool is the reason the rule learns to keep a pattern rather
// than to draw one: most steps start from a state some earlier rollout
// left behind, not from the bare seed.

const pool = new Float32Array(options.pool * N * C)
for (let i = 0; i < options.pool; i++)
  pool.set(seedState(1, N, C, center), i * N * C)

const random = rng(options.seed + 1)

/** Squared error of one flat sample against the target, visible channels. */
function sampleError(
  state: Float32Array,
  offset: number
): number {
  let total = 0
  for (let node = 0; node < N; node++)
    for (let c = 0; c < 4; c++) {
      const delta =
        state[offset + node * C + c]! - targetData[node * 4 + c]!
      total += delta * delta
    }
  return total / (N * 4)
}

/** Zero every channel of the given nodes in sample `b` of the batch. */
function wipe(
  batchState: Float32Array,
  b: number,
  nodes: Int32Array
): void {
  for (const node of nodes)
    batchState.fill(
      0,
      (b * N + node) * C,
      (b * N + node) * C + C
    )
}

// ------------------------------------------------------------- the probe ---
// Grow, punch a hole, heal. Training loss can fall while regeneration
// stays broken, so this is the number to watch. The hole is a fixed
// blob, so the probe is comparable across steps and across runs.

const probeHole = damage.ball(pos, inPattern, {
  frac: 0.25,
  center: inPattern[Math.floor(inPattern.length / 2)]!
})

function healProbe(
  grow = 80,
  heal = 160
): { grown: number; healed: number } {
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
  for (const node of probeHole)
    wounded.fill(0, node * C, node * C + C)
  const healedState = run(wounded, heal)
  configure({ lazy: false })
  return {
    grown: grownError,
    healed: sampleError(healedState, 0)
  }
}

// -------------------------------------------------------------- training ---

console.log(
  `training ${options.steps} steps, batch ${B}, ` +
    `rollout lengths ${horizons.join("/")}, ` +
    `${params.length} parameter tensors, ` +
    `${params.reduce((n, p) => n + p.numel, 0)} weights`
)

const batchState = new Float32Array(B * N * C)
const chosen = new Int32Array(B)
let lastReport = Date.now()

for (let step = 1; step <= options.steps; step++) {
  // Draw a batch from the pool.
  for (let b = 0; b < B; b++) chosen[b] = random.int(options.pool)
  for (let b = 0; b < B; b++)
    batchState.set(
      pool.subarray(chosen[b]! * N * C, (chosen[b]! + 1) * N * C),
      b * N * C
    )

  // Rank worst-first, so batchState[0] is the least-formed pattern and
  // the last entries the most. Damage lands on the best ones: healing a
  // grown pattern is the behaviour we want, and gradient spent on an
  // already-broken state is spent twice over.
  const order = Array.from({ length: B }, (_, b) => b).sort(
    (a, b) =>
      sampleError(batchState, b * N * C) -
      sampleError(batchState, a * N * C)
  )
  const ranked = new Float32Array(B * N * C)
  const rankedChoice = new Int32Array(B)
  order.forEach((b, i) => {
    ranked.set(
      batchState.subarray(b * N * C, (b + 1) * N * C),
      i * N * C
    )
    rankedChoice[i] = chosen[b]!
  })
  batchState.set(ranked)
  chosen.set(rankedChoice)

  // Every eighth step, train the worst sample from the bare seed, so the
  // rule keeps practising the long horizon from nothing.
  if (step % 8 === 0)
    batchState.set(seedState(1, N, C, center), 0)

  // Damage the best-formed samples: one gets scattered noise, the rest
  // get balls covering 20-50% of the pattern.
  for (let i = 0; i < Math.min(options.damage, B); i++) {
    const b = B - 1 - i
    wipe(
      batchState,
      b,
      i === 0 ?
        damage.scatter(inPattern, { random })
      : damage.ball(pos, inPattern, {
          frac: random.range(0.2, 0.5),
          random
        })
    )
  }

  const horizon = horizons[random.int(horizons.length)]!
  const { loss, mse, state } = trainStep(batchState, horizon)

  // Write the rollout back into the pool, and reset the worst sample to
  // a fresh seed so the pool never fills with dead ends.
  let worst = 0
  let worstError = -Infinity
  for (let b = 0; b < B; b++) {
    const error = sampleError(state, b * N * C)
    if (error > worstError) {
      worstError = error
      worst = b
    }
  }
  for (let b = 0; b < B; b++)
    pool.set(
      b === worst ?
        seedState(1, N, C, center)
      : state.subarray(b * N * C, (b + 1) * N * C),
      chosen[b]! * N * C
    )

  if (step % options.report === 0 || step === 1) {
    const now = Date.now()
    const rate =
      step === 1 ?
        0
      : (options.report * 1000) / Math.max(now - lastReport, 1)
    lastReport = now
    console.log(
      `step ${String(step).padStart(6)}  loss ${loss.toFixed(6)}  ` +
        `mse ${mse.toFixed(6)}  ${rate.toFixed(2)} steps/s`
    )
  }
  if (step % options.probe === 0) {
    const { grown, healed } = healProbe()
    console.log(
      `    probe: grown ${grown.toFixed(4)}  healed ${healed.toFixed(4)}`
    )
  }
}

const { grown, healed } = healProbe()
console.log(
  `done. probe: grown ${grown.toFixed(4)}  healed ${healed.toFixed(4)}`
)
