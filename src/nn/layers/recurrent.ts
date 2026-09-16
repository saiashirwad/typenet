"use tsover"

// A vanilla (Elman) recurrent layer. The recurrence is a method rather than a loop inside
// `forward`, because there is no sequence axis to fold: `forward` sees one timestep, and the
// caller owns the loop. That is also what keeps every shape literal, since a state is `[B, H]`
// whatever the caller's sequence length happens to be.
//
//   h_t = tanh(x_t @ weightIH + b_ih + h_{t-1} @ weightHH + b_hh)
//
// Both projections stay separate so each can be inspected or initialised on its own. The weights
// follow `nn.RNN`'s convention, `U(-1/√H, 1/√H)` on the weights and the biases.

import { Tensor } from "../../tensor.ts"
import * as init from "../init.ts"
import { Module } from "../module.ts"
import { type Parameter, parameter } from "../parameter.ts"
import { SHAPE_EFFECT } from "../sequential.ts"

/** `forward` can read the batch from a state or from a count, and `B` is only inferable from one of them. */
export type StateOrBatch<B extends number, H extends number> =
  | { readonly state: Tensor<[B, H]> }
  | { readonly batch: B }

export class Rnn<In extends number, H extends number> extends Module {
  // A step maps `[B, In]` to `[B, H]`, so it neither appends an axis nor merges one.
  declare readonly [SHAPE_EFFECT]: "identity"

  readonly inputSize: In
  readonly hiddenSize: H
  /** `[In, H]`, this library's matmul order: the transpose of PyTorch's `weight_ih`. */
  readonly weightIH: Parameter<[In, H]>
  /** `[H, H]`, the transpose of PyTorch's `weight_hh`. */
  readonly weightHH: Parameter<[H, H]>
  readonly biasIH: Parameter<[H]> | null
  readonly biasHH: Parameter<[H]> | null

  constructor(inputSize: In, hiddenSize: H, options: { bias?: boolean } = {}) {
    super()
    this.inputSize = inputSize
    this.hiddenSize = hiddenSize
    const k = 1 / Math.sqrt(hiddenSize)
    // Fan-based initialisers are wrong here: a recurrent weight multiplies the state at every
    // step of the unroll, so its bound has to shrink with `hiddenSize` whatever `inputSize` is,
    // which is what `nn.RNN`'s flat `1/√H` does.
    this.weightIH = parameter(init.uniform<[In, H]>([inputSize, hiddenSize], { low: -k, high: k }))
    this.weightHH = parameter(init.uniform<[H, H]>([hiddenSize, hiddenSize], { low: -k, high: k }))
    this.biasIH = options.bias === false ? null : parameter(init.uniform<[H]>([hiddenSize], { low: -k, high: k }))
    this.biasHH = options.bias === false ? null : parameter(init.uniform<[H]>([hiddenSize], { low: -k, high: k }))
  }

  /** The zero state a sequence starts from, shaped `[B, H]`. */
  zeroState<B extends number>(batch: B): Tensor<[B, H]> {
    return Tensor.zeros([batch, this.hiddenSize])
  }

  /**
   * One timestep. `state` is the previous `[B, H]`; pass `{ batch }` instead to start from zero,
   * which is the only way `B` can be inferred when there is no state yet.
   */
  forward<B extends number, O extends StateOrBatch<B, H>>(
    x: Tensor<[B, In]>,
    options: O & (O extends { state: Tensor<[B, H]> } ? unknown : { batch: B }),
  ): Tensor<[B, H]> {
    const state = "state" in options ? options.state : this.zeroState(options.batch)
    const pre = x.matmul(this.weightIH).add(state.matmul(this.weightHH))
    const biased = this.biasIH === null || this.biasHH === null
      ? pre
      : pre.add(this.biasIH).add(this.biasHH)
    return biased.tanh()
  }
}

/** The same step without a batch axis, `[In] -> [H]`, for sampling one character at a time. */
export function stepUnbatched<In extends number, H extends number>(
  layer: Rnn<In, H>,
  x: Tensor<[In]>,
  state: Tensor<[H]>,
): Tensor<[H]> {
  return layer.forward(x.unsqueeze(0), { state: state.unsqueeze(0) }).squeeze(0)
}
