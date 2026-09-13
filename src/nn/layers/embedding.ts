import type { IndexTensor, Shape } from "../../shape.ts"
import { gatherRows, Tensor } from "../../tensor.ts"
import * as init from "../init.ts"
import { Module } from "../module.ts"
import { type Parameter, parameter } from "../parameter.ts"
import { SHAPE_EFFECT } from "../sequential.ts"

/**
 * A lookup table of `V` rows of width `D`, addressed by an index tensor of
 * any rank (W5.1-A): `forward` emits the fused `gatherRows` node (W4.1),
 * so a `[B, T]` batch of token ids never round-trips through a flatten.
 *
 * Declares `["appendDim", D]` (W4.9's protocol) because the structural
 * probe cannot read this layer at all: its `forward` takes an
 * {@link IndexTensor}, and a plain `Tensor<S>` is not assignable to
 * `IndexTensor<S>` — without the declaration `sequential(...)` would treat
 * `Embedding` as the identity, which is exactly the silent no-op W4.9
 * exists to close.
 */
export class Embedding<V extends number, D extends number> extends Module {
  declare readonly [SHAPE_EFFECT]: [effect: "appendDim", D: D]

  readonly weight: Parameter<[V, D]>
  readonly numEmbeddings: V
  readonly embeddingDim: D

  constructor(numEmbeddings: V, embeddingDim: D) {
    super()
    this.numEmbeddings = numEmbeddings
    this.embeddingDim = embeddingDim
    // PyTorch's default: N(0, 1) per row. Explicit type argument for the
    // same nested-generic-call reason `Linear`'s constructor documents.
    this.weight = parameter(
      init.normal<[V, D]>([numEmbeddings, embeddingDim]),
    )
  }

  /**
   * A float tensor is a compile error here (`Tensor<S>` is not
   * `IndexTensor<S>`) — `Tensor.indices`/`toIndex()` (tensor.ts) are the
   * two ways to mint one, and `nn.functional.arangeIndex` a third for the
   * common `0..N-1` case. Weight tying: pass `this.weight` itself to
   * `TiedLinear.of` (W5.10-A) rather than constructing a second Embedding.
   */
  forward<S extends Shape>(
    ids: IndexTensor<S>,
  ): Tensor<[...S, D]> {
    return gatherRows(this.weight, ids)
  }
}
