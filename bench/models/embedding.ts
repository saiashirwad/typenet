// Embedding forward + backward, hand-composed from `indexSelect` /
// `scatterAdd` (PLAN-V2 §4.2, W0.3) — there is no `Embedding` module in
// `src/nn.ts` yet, so this is the bench-only stand-in, read by
// `bench/macro-embedding.ts`.

import { Module, randn } from "../../index.ts"
import type { IndexTensor } from "../../src/shape.ts"
import { type AnyTensor, fromFlat } from "../../src/tensor.ts"

export interface EmbeddingConfig {
  vocabSize: number
  embedDim: number
}

/** `weight.indexSelect(ids, 0)` — an `F.embedding` lookup with no fused
 * kernel; the backward is `scatterAdd`, exercised via `.backward()`. */
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

/** `count` random integer ids in `[0, vocabSize)`, as a rank-1 float32
 * index tensor (`indexSelect`/`scatterAdd` accept a float32 index for
 * compatibility with plain JS number arrays), branded with `.toIndex()`
 * because those methods take an `IndexTensor` (OWNER-5, D25). */
export function randomIds(vocabSize: number, count: number): IndexTensor<[number]> {
  const data = new Float32Array(count)
  for (let i = 0; i < count; i++) {
    data[i] = Math.floor(Math.random() * vocabSize)
  }
  return fromFlat(data, [count]).toIndex()
}
