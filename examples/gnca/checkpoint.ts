// Saving and resuming a run. A checkpoint is self-contained — the rule's
// weights *and* the graph it was trained on — because the graph is what
// gives the weights meaning: the same rule on a different graph is a
// different model. JSON, because the whole thing is well under a megabyte
// and being able to read it is worth more than the bytes.
//
// Mirrors load_checkpoint / load_rule in
// ~/code/graph-cellular-automata/src/gnca/inference.py.

import {
  readFileSync,
  writeFileSync,
  mkdirSync
} from "node:fs"
import { dirname } from "node:path"
import type { Tensor } from "../../index.ts"
import type { Edges, Points } from "./graphs.ts"

type AnyTensor = Tensor<any, any>

export interface Checkpoint {
  /** Step the checkpoint was written at. */
  step: number
  channels: number
  hidden: number
  /** Which pattern, and where its seed node went. */
  target: string
  center: number
  /** Node positions, flat row-major. */
  pos: number[]
  dim: number
  edges: { src: number[]; dst: number[] }
  /** The target RGBA, so a checkpoint can be scored without rebuilding it. */
  targetRgba: number[]
  /** One entry per parameter tensor, in `parameters()` order. */
  weights: { shape: number[]; values: number[] }[]
}

export function saveCheckpoint(
  path: string,
  checkpoint: Omit<Checkpoint, "weights">,
  params: AnyTensor[]
): void {
  mkdirSync(dirname(path), { recursive: true })
  const full: Checkpoint = {
    ...checkpoint,
    weights: params.map(p => ({
      shape: [...p.shape],
      // Round-trip through JSON exactly: 9 significant digits is more
      // than a float32 needs.
      values: Array.from(p.data, v =>
        Number(v.toPrecision(9))
      )
    }))
  }
  writeFileSync(path, JSON.stringify(full))
}

export function readCheckpoint(path: string): Checkpoint {
  return JSON.parse(
    readFileSync(path, "utf8")
  ) as Checkpoint
}

/** The graph a checkpoint was trained on. */
export function checkpointGraph(checkpoint: Checkpoint): {
  pos: Points
  edges: Edges
} {
  return {
    pos: {
      data: Float32Array.from(checkpoint.pos),
      n: checkpoint.pos.length / checkpoint.dim,
      dim: checkpoint.dim
    },
    edges: {
      src: Float32Array.from(checkpoint.edges.src),
      dst: Float32Array.from(checkpoint.edges.dst),
      count: checkpoint.edges.src.length
    }
  }
}

/**
 * Load weights into a model's parameters, zero-padding the first layer's
 * input columns when the checkpoint's perception is narrower than the
 * model's.
 *
 * That padding is what makes the reference's ablation ladder work: add a
 * feature to the percept, warm-start from the run before it, and the
 * loaded rule is *functionally identical* to begin with — the new columns
 * contribute nothing until training moves them — so the comparison
 * measures the feature rather than a fresh initialisation.
 *
 * Returns how many columns were padded.
 */
export function loadRule(
  params: AnyTensor[],
  checkpoint: Checkpoint
): number {
  if (checkpoint.weights.length !== params.length)
    throw new Error(
      `checkpoint has ${checkpoint.weights.length} parameter tensors, model has ${params.length}`
    )
  let padded = 0
  params.forEach((p, i) => {
    const saved = checkpoint.weights[i]!
    const target = p.data as Float32Array
    if (
      saved.shape.length === 2 &&
      p.shape.length === 2 &&
      saved.shape[1] === p.shape[1] &&
      saved.shape[0]! < p.shape[0]!
    ) {
      // A narrower input side: copy row by row and leave the rest zero.
      // typenet weights are [in, out], so the pad is trailing rows.
      const cols = p.shape[1]!
      target.fill(0)
      target.set(saved.values, 0)
      padded += (p.shape[0]! - saved.shape[0]!) * cols
      return
    }
    if (saved.values.length !== target.length)
      throw new Error(
        `checkpoint parameter ${i} has shape [${saved.shape}], model wants [${p.shape}]`
      )
    target.set(saved.values)
  })
  return padded
}
