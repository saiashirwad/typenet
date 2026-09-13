// Embedding forward + backward (`indexSelect` + `scatterAdd`) at
// V ∈ {65, 4096, 50257}, batch×block = 16 384 ids (PLAN-V2 §4.2, W0.3).
// Read by later items via the `emb-65` / `emb-4096` / `emb-50257` case
// ids — do not rename them.

import { disableNative, useNative } from "../index.ts"
import { configure } from "../src/lazy.ts"
import type { AnyTensor } from "../src/tensor.ts"
import { bench, type BenchCaseSpec, isSmokeRun, type Mode } from "./lib/harness.ts"
import { EMBEDDING_FULL, EMBEDDING_SMOKE } from "./lib/sizes.ts"
import { Embedding, randomIds } from "./models/embedding.ts"

const SIZE_CONFIG = isSmokeRun() ? EMBEDDING_SMOKE : EMBEDDING_FULL
const IDS_PER_STEP = SIZE_CONFIG.idsPerStep // batch × block, per §4.2 (full only)
const EMBED_DIM = SIZE_CONFIG.embedDim

interface EmbCase extends BenchCaseSpec {
  vocabSize: number
  backward: boolean
}

const CASES: readonly EmbCase[] = SIZE_CONFIG.vocabs.flatMap(vocabSize => [
  // Primary, load-bearing id: forward+backward.
  { id: `emb-${vocabSize}`, vocabSize, backward: true },
  // Forward-only companion.
  { id: `emb-${vocabSize}-fwd`, vocabSize, backward: false },
])

function setMode(mode: Mode): void {
  if (mode === "native") {
    configure({ lazy: true })
    useNative()
  } else if (mode === "interp") {
    disableNative()
    configure({ lazy: true })
  } else {
    disableNative()
    configure({ lazy: false })
  }
}

async function main(): Promise<void> {
  const models = new Map<number, Embedding>()
  const modelFor = (vocabSize: number): Embedding => {
    let m = models.get(vocabSize)
    if (!m) {
      m = new Embedding({ vocabSize, embedDim: EMBED_DIM })
      models.set(vocabSize, m)
    }
    return m
  }

  await bench("macro-embedding", CASES, (kase, mode) => {
    setMode(mode)
    const model = modelFor(kase.vocabSize)
    model.zeroGrad()
    const ids = randomIds(kase.vocabSize, IDS_PER_STEP) as AnyTensor
    const out = model.forward(ids)
    if (kase.backward) {
      out.sum().backward()
    } else {
      out.data // force materialization
    }
  })

  configure({ lazy: false })
  disableNative()
}

await main()
