import type { IndexTensor, Shape } from "../../shape.ts"
import { gatherRows, Tensor } from "../../tensor.ts"
import * as init from "../init.ts"
import { Module } from "../module.ts"
import { type Parameter, parameter } from "../parameter.ts"
import { SHAPE_EFFECT } from "../sequential.ts"

/** A lookup table of `V` rows of width `D`, addressed by an index tensor of any rank via the fused `gatherRows` node. */
export class Embedding<V extends number, D extends number> extends Module {
  declare readonly [SHAPE_EFFECT]: [effect: "appendDim", D: D]

  readonly weight: Parameter<[V, D]>
  readonly numEmbeddings: V
  readonly embeddingDim: D

  constructor(numEmbeddings: V, embeddingDim: D) {
    super()
    this.numEmbeddings = numEmbeddings
    this.embeddingDim = embeddingDim
    // PyTorch's default init: N(0, 1) per row.
    this.weight = parameter(
      init.normal<[V, D]>([numEmbeddings, embeddingDim]),
    )
  }

  /** `ids` must be an {@link IndexTensor}; for weight tying, pass `this.weight` itself to `TiedLinear.of`. */
  forward<S extends Shape>(
    ids: IndexTensor<S>,
  ): Tensor<[...S, D]> {
    return gatherRows(this.weight, ids)
  }
}
