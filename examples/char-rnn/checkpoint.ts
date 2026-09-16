"use tsover"

// Checkpoints for the character RNN: `stateDict()` plus the alphabet it was trained on.

import { readFileSync, writeFileSync } from "node:fs"
import { type DType, type StateDict, type StateEntry } from "../../index.ts"
import { CharacterRNN, type CharacterRNNConfig } from "./net.ts"

const FORMAT = "typenet.char-rnn/1"

/** Every distinct character of `text`, in code-point order: the vocabulary, read off the data. */
export function alphabetOf(text: string): string {
  return [...new Set(text)].sort().join("")
}

/** mulberry32: a sample replays from its seed, which `Math.random` would not give. */
export function seeded(seed: number): () => number {
  let state = seed >>> 0
  return () => {
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
 * A structural type rather than `CharacterRNN` itself, because its widths are invariant and a
 * model built from inferred literals is not assignable to one built from numbers.
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
  const state: Record<string, StateEntry> = {}
  for (const [name, entry] of Object.entries(file.state)) {
    state[name] = {
      shape: entry.shape,
      dtype: entry.dtype as DType,
      data: Float32Array.from(entry.data),
    }
  }
  // Strict: weights that disagree with the signature throw rather than sampling half a model.
  model.loadStateDict(state)
  model.eval()
  return { model, signature: file.signature }
}
