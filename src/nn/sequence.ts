"use tsover"

import type { Drop, Shape } from "../shape.ts"
import { type AnyTensor, Tensor } from "../tensor.ts"

/** One step's result: the output collected for this position, and the state to carry on with. */
export interface ScanStep<Out extends Shape, St extends Shape> {
  readonly output: Tensor<Out>
  readonly state: Tensor<St>
}

export interface ScanResult<B extends number, Out extends Shape, St extends Shape> {
  /** Every step's output along a new time axis. `Out` is one step's whole shape, batch included, so the tail is `Drop<Out, 1>`. */
  readonly outputs: Tensor<[B, number, ...Drop<Out, 1>]>
  /** The state after the last step. */
  readonly state: Tensor<St>
}

/** Runs `steps` iterations of `next` from `initial`, concatenating the outputs along a new time axis. */
export function scan<B extends number, St extends Shape, Out extends Shape>(
  initial: Tensor<St>,
  steps: number,
  next: (state: Tensor<St>, index: number) => ScanStep<Out, St>,
): ScanResult<B, Out, St> {
  if (!Number.isInteger(steps) || steps < 0) {
    throw new Error(`scan: steps must be a non-negative integer, got ${steps}`)
  }
  const batch = initial.shape[0]!
  let state = initial
  const pieces: AnyTensor[] = []
  for (let i = 0; i < steps; i++) {
    const step = next(state, i)
    if (step.output.shape[0] !== batch || step.state.shape[0] !== batch) {
      throw new Error(
        `scan: step ${i} returned a batch of ${step.output.shape[0]}/${step.state.shape[0]}, `
          + `but the initial state is batched ${batch}`,
      )
    }
    state = step.state
    pieces.push(step.output as AnyTensor)
  }
  if (pieces.length === 0) {
    throw new Error("scan: steps was 0, so there is no output to return")
  }
  // One-node stack: catting the outputs one at a time would copy the whole run on every step.
  const outputs = Tensor.stackList(pieces, 1) as unknown as Tensor<[B, number, ...Drop<Out, 1>]>
  return { outputs, state }
}

/** A stepped recurrence for loops whose length is not known up front: carries a `[B, ...St]` state, and `outputs()` concatenates everything recorded so far. */
export class Sequence<B extends number, St extends Shape = Shape> {
  private readonly history: Tensor<Shape>[] = []
  private current: Tensor<St>
  private readonly batch: number

  private constructor(initial: Tensor<St>) {
    this.current = initial
    this.batch = initial.shape[0]!
  }

  /** Starts from `initial`, whose first axis is the batch every later step has to agree with. */
  static of<B extends number, St extends Shape>(initial: Tensor<St>): Sequence<B, St> {
    return new Sequence<B, St>(initial)
  }

  /** The state after the steps taken so far. */
  get state(): Tensor<St> {
    return this.current
  }

  /** How many outputs have been recorded. */
  get length(): number {
    return this.history.length
  }

  /** Steps once, recording `output` and carrying on with `state`, which need not share its shape. */
  step<Out extends Shape>(output: Tensor<Out>, state: Tensor<St>): Sequence<B, St> {
    if (output.shape[0] !== this.batch || state.shape[0] !== this.batch) {
      throw new Error(
        `Sequence.step: batch ${output.shape[0]}/${state.shape[0]} does not match the state's ${this.batch}`,
      )
    }
    this.current = state
    this.history.push(output as unknown as Tensor<Shape>)
    return this
  }

  /** Advances the state without recording an output, which is how a prompt is consumed. */
  advance(state: Tensor<St>): Sequence<B, St> {
    if (state.shape[0] !== this.batch) {
      throw new Error(
        `Sequence.advance: batch ${state.shape[0]} does not match the state's ${this.batch}`,
      )
    }
    this.current = state
    return this
  }

  /** Everything recorded, as `[B, T, ...]` over the outputs' own shape. */
  outputs<Out extends Shape = Shape>(): Tensor<[B, number, ...Drop<Out, 1>]> {
    if (this.history.length === 0) {
      throw new Error("Sequence.outputs: nothing has been stepped yet")
    }
    return Tensor.stackList(this.history, 1) as unknown as Tensor<[B, number, ...Drop<Out, 1>]>
  }
}
