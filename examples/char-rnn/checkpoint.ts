"use tsover"

// Checkpoints and text loading for the character RNN examples.
//
// Saving goes through `Module.stateDict()`, so it writes exactly what the module reports: the
// parameters, by dotted name, at their own dtype and shape. The alphabet travels with them,
// because a model is only meaningful next to the symbols it was trained on.

import { readFileSync, writeFileSync } from "node:fs"
import { type DType, type StateDict, type StateEntry } from "../../index.ts"
import { CharacterRNN, type CharacterRNNConfig } from "./net.ts"

const FORMAT = "typenet.char-rnn/1"

/** Every distinct character of `text`, in code-point order: the vocabulary, read off the data. */
export function alphabetOf(text: string): string {
  return [...new Set(text)].sort().join("")
}

/**
 * Reads the corpus at `path`. There is no built-in text, so the path is required: a caller that
 * forgets to name one gets a `readFileSync` error rather than a silent fallback to some other
 * text than the one it meant.
 */
export function readCorpus(path: string): string {
  const text = readFileSync(path, "utf8")
  if (text.length === 0) throw new Error(`${path} is empty`)
  return text
}

/** The seed a sample replays from, and the reason `Math.random` is not used for one. */
export function seeded(seed: number): () => number {
  let state = seed >>> 0
  return () => {
    // mulberry32.
    state = (state + 0x6d2b79f5) >>> 0
    let t = state
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

/** The architecture a checkpoint was trained at, kept next to the weights so it can be rebuilt. */
export interface CheckpointSignature {
  readonly vocab: number
  readonly embed: number
  readonly hidden: number
  readonly unroll: number
  readonly alphabet: string
  readonly seed: number
}

interface CheckpointFile {
  readonly format: string
  readonly signature: CheckpointSignature
  readonly state: Record<string, { shape: number[]; dtype: string; data: number[] }>
}

/**
 * What saving needs: the alphabet to record, the widths to record, and the state to write. A
 * narrow structural type rather than the model class itself, because `CharacterRNN`'s widths are
 * invariant and a model built from inferred literals (`CharacterRNN<65, 32, 128>`) is not
 * assignable to one built from numbers.
 */
export interface CheckpointableModel {
  readonly config: CharacterRNNConfig<number, number, number>
  readonly alphabet: readonly string[]
  stateDict(): StateDict
}

export function saveCheckpoint(path: string, model: CheckpointableModel, seed: number): number {
  const { vocab, embed, hidden, unroll } = model.config
  const state: CheckpointFile["state"] = {}
  for (const [name, entry] of Object.entries(model.stateDict())) {
    state[name] = {
      shape: [...entry.shape],
      dtype: entry.dtype,
      data: Array.from(entry.data, Number),
    }
  }
  const file: CheckpointFile = {
    format: FORMAT,
    signature: { vocab, embed, hidden, unroll, alphabet: model.alphabet.join(""), seed },
    state,
  }
  writeFileSync(path, `${JSON.stringify(file, null, 2)}\n`)
  return Object.keys(state).length
}

/** Rebuilds the model a checkpoint was written from, weights and alphabet included. */
export function loadCheckpoint(
  path: string,
): { model: CharacterRNN<number, number, number>; signature: CheckpointSignature } {
  const file = JSON.parse(readFileSync(path, "utf8")) as CheckpointFile
  if (file.format !== FORMAT) {
    throw new Error(`${path}: expected format ${FORMAT}, got ${JSON.stringify(file.format)}`)
  }
  const { vocab, embed, hidden, unroll, alphabet } = file.signature
  const model = new CharacterRNN({ vocab, embed, hidden, unroll }, alphabet)
  // Strict: a checkpoint whose weights disagree with the signature above throws here, rather
  // than sampling from a model that is half the one the file was written from.
  const state: Record<string, StateEntry> = {}
  for (const [name, entry] of Object.entries(file.state)) {
    state[name] = {
      shape: entry.shape,
      dtype: entry.dtype as DType,
      data: Float32Array.from(entry.data),
    }
  }
  model.loadStateDict(state)
  model.eval()
  return { model, signature: file.signature }
}
