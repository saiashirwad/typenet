"use tsover"

// Samples from the checkpoint train.ts writes. The prompt warms the state and is not printed.

import { fileURLToPath } from "node:url"
import { loadCheckpoint, seeded } from "./checkpoint.ts"

const CHECKPOINT = fileURLToPath(new URL("char-rnn.json", import.meta.url))
const PROMPT = "\n"
const LENGTH = 500
const TEMPERATURE = 0.6
const SEED = 1234

const { model, signature } = loadCheckpoint(CHECKPOINT)

console.log(
  `loaded ${signature.alphabet.length} symbols, embed ${signature.embed}, `
    + `hidden ${signature.hidden}, unroll ${signature.unroll}, trained at seed ${signature.seed}`,
)

const prompt = model.encode(PROMPT)
if (prompt.shape[0] !== PROMPT.length) {
  throw new Error(`the prompt holds a character outside the model's alphabet: ${JSON.stringify(PROMPT)}`)
}
const continuation = model.decode(
  model
    .generate(prompt, { length: LENGTH, temperature: TEMPERATURE, rng: seeded(SEED) })
    .narrow(0, prompt.shape[0]!, LENGTH)
    .data,
)

console.log(`\ntemperature ${TEMPERATURE}, after ${JSON.stringify(PROMPT)}:\n`)
console.log(continuation.replaceAll("\n", "\n  "))
