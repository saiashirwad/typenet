"use tsover"

// Trains the character RNN and writes the checkpoint generate.ts samples from. For text that reads
// like language: `pnpm fetch:corpus`, point TEXT at it, HIDDEN 256, UNROLL 64, LEARNING_RATE 2e-3, STEPS 3000.

import { readFileSync } from "node:fs"
import { basename } from "node:path"
import { fileURLToPath } from "node:url"
import { AdamW, clipGradNorm, configure, type IndexTensor, Tensor } from "../../index.ts"
import { alphabetOf, saveCheckpoint, seeded } from "./checkpoint.ts"
import { CharacterRNN } from "./net.ts"

const EMBED = 32
const BATCH = 32
const LEARNING_RATE = 5e-3
const STEPS = 2000
const HIDDEN = 128
const UNROLL = 32
const SEED = 1234
const TEXT = fileURLToPath(new URL("data/essay.txt", import.meta.url))
const CHECKPOINT = fileURLToPath(new URL("char-rnn.json", import.meta.url))

const text = readFileSync(TEXT, "utf8")
const alphabet = alphabetOf(text)

configure({ seed: SEED })

const model = new CharacterRNN({ vocab: alphabet.length, embed: EMBED, hidden: HIDDEN, unroll: UNROLL }, alphabet)
const codes = Array.from(model.encode(text).data, Number)
const parameters = model.parameters()
const optimizer = new AdamW(parameters, { lr: LEARNING_RATE, weightDecay: 0 })
// `encode` drops a character the alphabet does not hold, and the alphabet is this text's.
if (codes.length !== text.length) {
  throw new Error(`encode kept ${codes.length} of ${text.length} characters, so the alphabet is not the text's`)
}

/** `BATCH` random windows of `UNROLL` characters, and the characters that follow them. */
function windows(): {
  x: IndexTensor<[typeof BATCH, number]>
  y: IndexTensor<[typeof BATCH, number]>
} {
  const xs: number[] = []
  const ys: number[] = []
  for (let b = 0; b < BATCH; b++) {
    const start = Math.floor(Math.random() * (codes.length - UNROLL - 1))
    for (let t = 0; t < UNROLL; t++) {
      xs.push(codes[start + t]!)
      ys.push(codes[start + t + 1]!)
    }
  }
  return { x: Tensor.indices(xs, [BATCH, UNROLL]), y: Tensor.indices(ys, [BATCH, UNROLL]) }
}

console.log(
  `character rnn: ${basename(TEXT)} (${text.length.toLocaleString("en-US")} characters, `
    + `${alphabet.length} symbols), embed ${EMBED}, hidden ${HIDDEN}, batch ${BATCH}, `
    + `unroll ${UNROLL}, ${parameters.reduce((n, p) => n + p.numel, 0).toLocaleString("en-US")} parameters`,
)
console.log(`an untrained model costs ln(${alphabet.length}) = ${Math.log(alphabet.length).toFixed(4)}`)

const started = performance.now()
let first = Number.NaN
let last = Number.NaN

for (let step = 1; step <= STEPS; step++) {
  const { x, y } = windows()
  const loss = model.lossOn(x, y)

  // Decay to zero by the last step: a rate that is right at the start keeps the loss bouncing in
  // a band around the minimum at the end.
  optimizer.lr = LEARNING_RATE * (1 - (step - 1) / STEPS)
  optimizer.zeroGrad()
  loss.backward()
  clipGradNorm(parameters, 5)
  optimizer.step()

  last = loss.item()
  if (step === 1) first = last
  if (step === 1 || step % 100 === 0 || step === STEPS) {
    console.log(`step ${String(step).padStart(4)}  loss ${last.toFixed(4)}`)
  }
}

const elapsed = (performance.now() - started) / 1000

model.eval()
const sample = model.decode(
  model.generate(model.encode("\n"), { length: 400, temperature: 0.6, rng: seeded(SEED) }).data,
)
console.log(`\nloss ${first.toFixed(4)} -> ${last.toFixed(4)} in ${STEPS} steps (${elapsed.toFixed(1)}s)`)
console.log(`\nsample at temperature 0.6:\n  ${sample.replaceAll("\n", "\n  ")}`)

const entries = saveCheckpoint(CHECKPOINT, model, SEED)
console.log(`\nwrote ${basename(CHECKPOINT)} (${entries} tensors); sample it with pnpm vite-node examples/char-rnn/generate.ts`)
