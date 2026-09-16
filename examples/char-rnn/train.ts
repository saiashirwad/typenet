"use tsover"

// Trains the character RNN in net.ts, prints the loss curve, and writes a checkpoint that
// generate.ts samples from.
//
//   pnpm example:char-rnn
//
// There is no built-in text: TYPENET_RNN_TEXT names the corpus and defaults to the small
// committed one, `data/essay.txt`. That file is 1.8 kB on purpose, small enough to watch the
// model leave uniform noise within a minute.
//
// For a sample that reads like language, train on Tiny Shakespeare, the corpus the original
// char-rnn post uses. This configuration reached a loss of 1.50 in 3,000 steps (about half an
// hour) and samples words, character names and blank-verse line breaks:
//
//   pnpm fetch:corpus
//   TYPENET_RNN_TEXT=examples/char-rnn/data/tiny-shakespeare.txt \
//   TYPENET_RNN_HIDDEN=256 TYPENET_RNN_UNROLL=64 TYPENET_RNN_LR=2e-3 \
//   TYPENET_RNN_STEPS=3000 pnpm example:char-rnn
//
// Every shape below is inferred: `windows()` builds `[BATCH, UNROLL]` index tensors and `lossOn`
// consumes them. The widths are numbers rather than literals, so the model typechecks for any
// configuration rather than only the one spelled out in the constants.

import { AdamW, clipGradNorm, configure, type IndexTensor, Tensor } from "../../index.ts"
import { alphabetOf, readCorpus, saveCheckpoint, seeded } from "./checkpoint.ts"
import { CharacterRNN } from "./net.ts"

const EMBED = 32
const BATCH = 32
const LEARNING_RATE = Number(process.env["TYPENET_RNN_LR"] ?? 5e-3)
const STEPS = Number(process.env["TYPENET_RNN_STEPS"] ?? 2000)
const HIDDEN = Number(process.env["TYPENET_RNN_HIDDEN"] ?? 128)
const UNROLL = Number(process.env["TYPENET_RNN_UNROLL"] ?? 32)
const SEED = Number(process.env["TYPENET_RNN_SEED"] ?? 1234)
const CHECKPOINT = process.env["TYPENET_RNN_CHECKPOINT"] ?? "char-rnn.json"
const TEXT_PATH = process.env["TYPENET_RNN_TEXT"] ?? "examples/char-rnn/data/essay.txt"

const text = readCorpus(TEXT_PATH)

configure({ seed: SEED })

const model = new CharacterRNN({ vocab: alphabetOf(text).length, embed: EMBED, hidden: HIDDEN, unroll: UNROLL }, alphabetOf(text))
const codes = Array.from(model.encode(text).data, Number)
const parameters = model.parameters()
const optimizer = new AdamW(parameters, { lr: LEARNING_RATE, weightDecay: 0 })
// `encode` drops a character the alphabet does not hold; the alphabet is built from this same
// text, so anything short of all of it means the two disagree, not that the text had a stray.
if (codes.length !== text.length) {
  throw new Error(`encode kept ${codes.length} of ${text.length} characters, so the alphabet is not the text's`)
}

/** A random window of `UNROLL` characters and the characters that follow it, as `[BATCH, UNROLL]`. */
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
  `character rnn: ${TEXT_PATH} (${text.length.toLocaleString("en-US")} characters, `
    + `${model.alphabet.length} symbols), embed ${EMBED}, hidden ${HIDDEN}, batch ${BATCH}, `
    + `unroll ${UNROLL}, ${parameters.reduce((n, p) => n + p.numel, 0).toLocaleString("en-US")} parameters`,
)
console.log(`an untrained model costs ln(${model.alphabet.length}) = ${Math.log(model.alphabet.length).toFixed(4)}`)

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
  const every = STEPS <= 50 ? 10 : STEPS <= 500 ? 25 : 100
  if (step === 1 || step % every === 0 || step === STEPS) {
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
console.log(`\nwrote ${CHECKPOINT} (${entries} tensors); sample it with pnpm example:char-rnn:generate`)
