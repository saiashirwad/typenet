// indexSelect / scatterAdd run directly, with no autograd graph, at
// embedding scale.

import { disableNative, useNative } from "../index.ts"
import { rand } from "../src/factories.ts"
import { configure } from "../src/lazy.ts"
import type { IndexTensor } from "../src/shape.ts"
import { type AnyTensor, fromFlat } from "../src/tensor.ts"
import { bench, type BenchCaseSpec, isSmokeRun, type Mode } from "./lib/harness.ts"
import { EMBEDDING_FULL, EMBEDDING_SMOKE } from "./lib/sizes.ts"

const SIZE_CONFIG = isSmokeRun() ? EMBEDDING_SMOKE : EMBEDDING_FULL
const IDS = SIZE_CONFIG.idsPerStep
const EMBED_DIM = SIZE_CONFIG.embedDim

type Kind = "indexSelect" | "scatterAdd"

interface GatherScatterCase extends BenchCaseSpec {
  kind: Kind
  vocabSize: number
}

const CASES: readonly GatherScatterCase[] = SIZE_CONFIG.vocabs.flatMap(vocabSize => [
  { id: `indexSelect-v${vocabSize}`, kind: "indexSelect" as const, vocabSize },
  { id: `scatterAdd-v${vocabSize}`, kind: "scatterAdd" as const, vocabSize },
])

/** Branded via `.toIndex()`: `indexSelect`/`scatterAdd` take an
 * `IndexTensor`, never a bare tensor. The brand check is a one-time
 * integrality scan here, outside every timed region. */
function randomIds(vocabSize: number, count: number): IndexTensor<[number]> {
  const data = new Float32Array(count)
  for (let i = 0; i < count; i++) data[i] = Math.floor(Math.random() * vocabSize)
  return fromFlat(data, [count]).toIndex()
}

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
  const weights = new Map<number, AnyTensor>()
  const ids = new Map<number, IndexTensor<[number]>>()
  const grads = new Map<number, AnyTensor>()

  await bench("micro-gather-scatter", CASES, (kase, mode) => {
    setMode(mode)

    let w = weights.get(kase.vocabSize)
    if (!w) {
      w = rand([kase.vocabSize, EMBED_DIM]) as AnyTensor
      weights.set(kase.vocabSize, w)
    }
    let idx = ids.get(kase.vocabSize)
    if (!idx) {
      idx = randomIds(kase.vocabSize, IDS)
      ids.set(kase.vocabSize, idx)
    }

    if (kase.kind === "indexSelect") {
      const out = w.indexSelect(idx, 0)
      out.data // force materialization
      return
    }

    let g = grads.get(kase.vocabSize)
    if (!g) {
      g = rand([IDS, EMBED_DIM]) as AnyTensor
      grads.set(kase.vocabSize, g)
    }
    const out = g.scatterAdd(idx, kase.vocabSize, 0)
    out.data // force materialization
  })

  configure({ lazy: false })
  disableNative()
}

await main()
