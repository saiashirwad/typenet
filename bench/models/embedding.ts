// Bench-only embedding stand-in (no Embedding module in src/nn yet), read
// by bench/macro-embedding.ts.

import { Module, randn } from "../../index.ts"
import type { IndexTensor } from "../../src/shape.ts"
import { type AnyTensor, fromFlat } from "../../src/tensor.ts"

export interface EmbeddingConfig {
  vocabSize: number
  embedDim: number
}

/** `weight.indexSelect(ids, 0)` lookup; backward is `scatterAdd` via `.backward()`. */
export class Embedding extends Module {
  readonly weight: AnyTensor

  constructor(readonly config: EmbeddingConfig) {
    super()
    this.weight = (randn([config.vocabSize, config.embedDim]) as AnyTensor)
      .mul(0.02)
      .detach()
      .requiresGrad()
  }

  /** `ids`: rank-1 index tensor of `count` indices -> `[count, embedDim]`. */
  forward(ids: IndexTensor<[number]>): AnyTensor {
    return this.weight.indexSelect(ids, 0)
  }
}

/** `count` random integer ids in `[0, vocabSize)` as a float32 index tensor
 * (indexSelect/scatterAdd accept float32 indices), branded with `.toIndex()`. */
export function randomIds(vocabSize: number, count: number): IndexTensor<[number]> {
  const data = new Float32Array(count)
  for (let i = 0; i < count; i++) {
    data[i] = Math.floor(Math.random() * vocabSize)
  }
  return fromFlat(data, [count]).toIndex()
}
