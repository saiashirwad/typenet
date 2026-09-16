"use tsover"

// Generates text from a trained character RNN: the file the training run writes.
//
//   pnpm example:char-rnn                    # trains, then writes char-rnn.json
//   pnpm example:char-rnn:generate           # samples from it
//   TYPENET_RNN_PROMPT="ROMEO: " TYPENET_RNN_LENGTH=600 pnpm example:char-rnn:generate
//
// A prompt warms the hidden state one character at a time, and the printed text is the
// continuation after it: the prompt is context, not output. The default prompt is a single
// newline, which is enough to start the recurrence and gives the model nothing to copy.

import { loadCheckpoint, seeded } from "./checkpoint.ts"

const PATH = process.env["TYPENET_RNN_CHECKPOINT"] ?? "char-rnn.json"
const PROMPT = process.env["TYPENET_RNN_PROMPT"] ?? "\n"
const LENGTH = Number(process.env["TYPENET_RNN_LENGTH"] ?? 500)
const TEMPERATURE = Number(process.env["TYPENET_RNN_TEMPERATURE"] ?? 0.6)
const SEED = Number(process.env["TYPENET_RNN_SEED"] ?? 1234)

const { model, signature } = loadCheckpoint(PATH)

console.log(
  `loaded ${PATH}: ${signature.alphabet.length} symbols, embed ${signature.embed}, `
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

console.log(
  `\ntemperature ${TEMPERATURE}${PROMPT === "" ? "" : `, after ${JSON.stringify(PROMPT)}`}:\n`,
)
console.log(continuation.replaceAll("\n", "\n  "))
