import { mkdirSync, readFileSync, writeFileSync } from "node:fs"
import { dirname } from "node:path"
import type { AnyTensor } from "../../src/tensor.ts"
import type { Edges, Points } from "./graphs.ts"

export interface Checkpoint {
  step: number
  channels: number
  hidden: number
  target: string
  center: number
  pos: number[]
  dim: number
  edges: { src: number[]; dst: number[] }
  targetRgba: number[]
  weights: { shape: number[]; values: number[] }[]
}

export function saveCheckpoint(
  path: string,
  checkpoint: Omit<Checkpoint, "weights">,
  params: AnyTensor[],
): void {
  mkdirSync(dirname(path), { recursive: true })
  const full: Checkpoint = {
    ...checkpoint,
    weights: params.map(p => ({
      shape: [...p.shape],
      // Round-trip through JSON exactly: 9 significant digits is more
      // than a float32 needs.
      values: Array.from(p.data, v => Number(v.toPrecision(9))),
    })),
  }
  writeFileSync(path, JSON.stringify(full))
}

export function readCheckpoint(path: string): Checkpoint {
  return JSON.parse(
    readFileSync(path, "utf8"),
  ) as Checkpoint
}

export function checkpointGraph(checkpoint: Checkpoint): {
  pos: Points
  edges: Edges
} {
  return {
    pos: {
      data: Float32Array.from(checkpoint.pos),
      n: checkpoint.pos.length / checkpoint.dim,
      dim: checkpoint.dim,
    },
    edges: {
      src: Int32Array.from(checkpoint.edges.src),
      dst: Int32Array.from(checkpoint.edges.dst),
      count: checkpoint.edges.src.length,
    },
  }
}

/**
 * Load weights into a model's parameters, zero-padding the first layer's
 * input columns when the checkpoint's perception is narrower than the
 * model's.
 *
 * Returns how many columns were padded.
 */
export function loadRule(
  params: AnyTensor[],
  checkpoint: Checkpoint,
): number {
  if (checkpoint.weights.length !== params.length) {
    throw new Error(
      `checkpoint has ${checkpoint.weights.length} parameter tensors, model has ${params.length}`,
    )
  }
  let padded = 0
  params.forEach((p, i) => {
    const saved = checkpoint.weights[i]!
    const target = p.data as Float32Array
    if (
      saved.shape.length === 2
      && p.shape.length === 2
      && saved.shape[1] === p.shape[1]
      && saved.shape[0]! < p.shape[0]!
    ) {
      // typenet weights are [in, out], so the pad is trailing rows.
      const cols = p.shape[1]!
      target.fill(0)
      target.set(saved.values, 0)
      padded += (p.shape[0]! - saved.shape[0]!) * cols
      return
    }
    if (saved.values.length !== target.length) {
      throw new Error(
        `checkpoint parameter ${i} has shape [${saved.shape}], model wants [${p.shape}]`,
      )
    }
    target.set(saved.values)
  })
  return padded
}
