// `indexSelect` / `scatterAdd` at embedding scale (PLAN-V2 §4.2, W0.3) —
// the raw ops macro-embedding.ts composes into a model; here they run
// directly, with no autograd graph, at V ∈ {65, 4096, 50257} and
// batch×block = 16 384 ids.

import { disableNative, useNative } from "../index.ts"
import { rand } from "../src/factories.ts"
import { configure } from "../src/lazy.ts"
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

function randomIds(vocabSize: number, count: number): AnyTensor {
  const data = new Float32Array(count)
  for (let i = 0; i < count; i++) data[i] = Math.floor(Math.random() * vocabSize)
  return fromFlat(data, [count])
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
  const ids = new Map<number, AnyTensor>()
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
