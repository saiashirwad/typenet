"use tsover"

// An MLP classifier. `sequential` checks the widths at construction; the data is
// generated here so the example runs offline and replays from a fixed seed.

import { AdamW, clipGradNorm, configure, crossEntropy, Linear, randn, ReLU, sequential, Tensor, tensor, warmupCosine } from "../index.ts"
import { accuracy } from "./util.ts"

const FEATURES = 784
const CLASSES = 10
const BATCH = 64
const TRAIN = 2048
const TEST = 512
// 400 steps is the run the README quotes; the tests override it.
const STEPS = Number(process.env["TYPENET_EXAMPLE_STEPS"] ?? 400)

configure({ seed: 7 })

const prototypes = randn([CLASSES, FEATURES]) * 0.12

function makeSplit(n: number): {
  x: Tensor<[number, typeof FEATURES]>
  labels: number[]
} {
  const labels = Array.from({ length: n }, (_, i) => i % CLASSES)
  // A seeded shuffle, since Math.random would not replay from the configured seed.
  for (let i = labels.length - 1; i > 0; i--) {
    const j = Math.floor(((i * 1103515245 + 12345) % 2147483648) / 2147483648 * (i + 1))
    ;[labels[i], labels[j]] = [labels[j]!, labels[i]!]
  }
  const onehot = tensor(labels).oneHot(CLASSES)
  const x = onehot.matmul(prototypes) + randn([n, FEATURES]) * 1.0
  return { x, labels }
}

const train = makeSplit(TRAIN)
const test = makeSplit(TEST)

const model = sequential(
  new Linear(FEATURES, 256),
  new ReLU(),
  new Linear(256, CLASSES),
)

const opt = new AdamW(model.parameters(), { lr: 3e-4, weightDecay: 0.01 })
const schedule = warmupCosine({ base: 3e-4, warmupSteps: 40, totalSteps: STEPS })

const started = performance.now()
let last = 0

for (let step = 0; step < STEPS; step++) {
  const start = (step * BATCH) % (TRAIN - BATCH)
  const x = train.x.narrow(0, start, BATCH)
  const y = Tensor.indices(train.labels.slice(start, start + BATCH), [BATCH])

  const logits = model.forward(x)
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

const testLogits = model.forward(test.x)
console.log(
  `\nfinal train loss ${last.toFixed(4)}  test accuracy ${accuracy(testLogits, test.labels)}%  (${STEPS} steps in ${elapsed.toFixed(1)}s)`,
)
