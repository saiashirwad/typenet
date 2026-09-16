"use tsover"

import { AdamW, clipGradNorm, configure, crossEntropy, Tensor, warmupCosine } from "../index.ts"
import { makeData, mlpModel } from "./mlp-net.ts"
import { accuracy } from "./util.ts"

const BATCH = 64
const TRAIN = 2048
const STEPS = 400

configure({ seed: 7 })

const { train, test } = makeData(TRAIN, 512)
const model = mlpModel()

const opt = new AdamW(model.parameters(), { lr: 3e-4, weightDecay: 0.01 })
const schedule = warmupCosine({ base: 3e-4, warmupSteps: 40, totalSteps: STEPS })

const started = performance.now()
let last = 0

for (let step = 0; step < STEPS; step++) {
  const start = (step * BATCH) % (TRAIN - BATCH)
  const x = train.x.narrow(0, start, BATCH)
  const y = Tensor.indices(train.labels.slice(start, start + BATCH), [BATCH])

  const loss = crossEntropy(model.forward(x), y)

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

console.log(
  `\nfinal train loss ${last.toFixed(4)}  test accuracy ${accuracy(model.forward(test.x), test.labels)}%  (${STEPS} steps in ${elapsed.toFixed(1)}s)`,
)
