"use tsover"

/**
 * An MLP classifier — §3.6 of the plan, on the pieces that exist today.
 *
 * The point of the example is the shape story, not the dataset: one
 * `sequential` whose widths are checked at construction, a `forward` that
 * infers `[64, 784] -> [64, 10]`, an `AdamW` driven by a warmup-cosine
 * schedule, and gradient clipping between `backward()` and `step()`.
 *
 * The data is synthetic and generated here (ten class prototypes plus
 * Gaussian noise) so that the example runs offline and reproducibly: a
 * fixed `configure({ seed })` makes the prototypes, the noise, the
 * parameter init and therefore the whole loss curve replay exactly.
 *
 * The loop is the plain `zeroGrad` / `backward` / `step` triple rather
 * than a compiled step: `compile()` bakes the optimizer's `lr` into the
 * traced graph as a constant, so a schedule that actually moves needs the
 * eager loop today. Everything else here is unchanged by that choice.
 */

import { AdamW, clipGradNorm, configure, crossEntropy, Linear, randn, ReLU, sequential, type Tensor, tensor, warmupCosine } from "../index.ts"
import { accuracy } from "./util.ts"

const FEATURES = 784
const CLASSES = 10
const BATCH = 64
const TRAIN = 2048
const TEST = 512
// 400 steps is the run the README quotes. `test/examples.test.ts` overrides
// it to a handful, so the smoke test exercises this file rather than a copy
// of it.
const STEPS = Number(process.env["TYPENET_EXAMPLE_STEPS"] ?? 400)

configure({ seed: 7 })

// --- the data ------------------------------------------------------------
// One prototype vector per class; a sample is its class prototype plus
// noise. Built with the library's own ops, so the shapes are checked the
// same way the model's are: [N, 10] one-hot @ [10, 784] -> [N, 784].

const prototypes = randn([CLASSES, FEATURES]) * 0.12

// The split size is a runtime quantity, so its dim stays the wildcard
// `number` — and `narrow` below pins the batch dim to the literal 64 that
// the model's shapes are actually checked against.
function makeSplit(n: number): {
  x: Tensor<[number, typeof FEATURES]>
  labels: number[]
} {
  const labels = Array.from({ length: n }, (_, i) => i % CLASSES)
  // Deterministic interleaving by construction, then one shuffle, so that
  // contiguous batches below are class-balanced without an index tensor.
  for (let i = labels.length - 1; i > 0; i--) {
    const j = Math.floor(((i * 1103515245 + 12345) % 2147483648) / 2147483648 * (i + 1))
    ;[labels[i], labels[j]] = [labels[j]!, labels[i]!]
  }
  const onehot = tensor(labels).oneHot(CLASSES) // Tensor<[number, 10]>
  const x = onehot.matmul(prototypes) + randn([n, FEATURES]) * 1.0
  return { x, labels }
}

const train = makeSplit(TRAIN)
const test = makeSplit(TEST)

// --- the model -----------------------------------------------------------
// The widths are checked where they are written: swapping the 256 in the
// second Linear for anything else is a compile error, not a runtime one.

const model = sequential(
  new Linear(FEATURES, 256),
  new ReLU(),
  new Linear(256, CLASSES),
)

const opt = new AdamW(model.parameters(), { lr: 3e-4, weightDecay: 0.01 })
const schedule = warmupCosine({ base: 3e-4, warmupSteps: 40, totalSteps: STEPS })

// --- the loop ------------------------------------------------------------

const started = performance.now()
let last = 0

for (let step = 0; step < STEPS; step++) {
  const start = (step * BATCH) % (TRAIN - BATCH)
  // `narrow` carries the literal length: a [64, 784] window of the
  // [2048, 784] training set, typed as such.
  const x = train.x.narrow(0, start, BATCH)
  const y = train.labels.slice(start, start + BATCH)

  const logits = model.forward(x) // Tensor<[64, 10]>
  const loss = crossEntropy(logits, y)

  opt.lr = schedule(step)
  opt.zeroGrad()
  loss.backward()
  clipGradNorm(model.parameters(), 1)
  opt.step()

  last = loss.item()
  if (step % 50 === 0 || step === STEPS - 1) {
    console.log(
      `step ${String(step).padStart(3)}  lr ${opt.lr.toExponential(2)}  loss ${last.toFixed(4)}`,
    )
  }
}

const elapsed = (performance.now() - started) / 1000

// --- what it learned -----------------------------------------------------

const testLogits = model.forward(test.x)
console.log(
  `\nfinal train loss ${last.toFixed(4)}  test accuracy ${accuracy(testLogits, test.labels)}%  (${STEPS} steps in ${elapsed.toFixed(1)}s)`,
)
