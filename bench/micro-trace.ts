// JS graph-build time only: trace to a lazy graph, no evaluation, for the
// MLP and nanoGPT S/M/L. The `trace-mlp` / `trace-gpt-{s,m,l}` case ids
// are load-bearing; do not rename them.
//
// No TransformerBlock module exists yet, so the GPT graph is hand-composed
// from CausalSelfAttention plus a GELU MLP; forward-pass correctness is
// not what this measures.

import { Linear, Module } from "../index.ts"
import { rand, randn } from "../src/factories.ts"
import { configure } from "../src/lazy.ts"
import type { IndexTensor } from "../src/shape.ts"
import { type AnyTensor, fromFlat } from "../src/tensor.ts"
import { bench, type BenchCaseSpec, isSmokeRun } from "./lib/harness.ts"
import { MLP_LEGACY, MLP_SMOKE, NANOGPT_SIZES, NANOGPT_SMOKE } from "./lib/sizes.ts"
import { CausalSelfAttention } from "./models/attention.ts"

function layerNorm(x: AnyTensor, gamma: AnyTensor, beta: AnyTensor, eps = 1e-5): AnyTensor {
  const mean = x.mean(2, true)
  const centered = x.sub(mean)
  const variance = centered.pow(2).mean(2, true)
  const normed = centered.div(variance.add(eps).sqrt())
  return normed.mul(gamma).add(beta)
}

function gelu(x: AnyTensor): AnyTensor {
  const c = Math.sqrt(2 / Math.PI)
  const inner = x.add(x.pow(3).mul(0.044715)).mul(c)
  return x.mul(0.5).mul(inner.tanh().add(1))
}

class GptBlock extends Module {
  readonly attn: CausalSelfAttention
  readonly mlpUp: Linear<number, number>
  readonly mlpDown: Linear<number, number>
  readonly ln1Gamma: AnyTensor
  readonly ln1Beta: AnyTensor
  readonly ln2Gamma: AnyTensor
  readonly ln2Beta: AnyTensor

  constructor(nEmbd: number, nHead: number, batch: number, seqLen: number) {
    super()
    this.attn = new CausalSelfAttention({ batch, seqLen, nEmbd, nHead })
    this.mlpUp = new Linear(nEmbd, 4 * nEmbd)
    this.mlpDown = new Linear(4 * nEmbd, nEmbd)
    this.ln1Gamma = rand([nEmbd]) as AnyTensor
    this.ln1Beta = rand([nEmbd]) as AnyTensor
    this.ln2Gamma = rand([nEmbd]) as AnyTensor
    this.ln2Beta = rand([nEmbd]) as AnyTensor
  }

  forward(x: AnyTensor): AnyTensor {
    const attnOut = this.attn.forward(layerNorm(x, this.ln1Gamma, this.ln1Beta))
    const h = x.add(attnOut)
    const mlpOut = this.mlpDown.forward(
      gelu(this.mlpUp.forward(layerNorm(h, this.ln2Gamma, this.ln2Beta) as never) as AnyTensor) as never,
    ) as AnyTensor
    return h.add(mlpOut)
  }
}

class TraceGpt extends Module {
  readonly tokEmb: AnyTensor
  readonly blocks: GptBlock[]
  readonly head: Linear<number, number>

  constructor(readonly batch: number, readonly seqLen: number, nEmbd: number, nHead: number, nLayer: number, vocabSize: number) {
    super()
    this.tokEmb = (randn([vocabSize, nEmbd]) as AnyTensor).mul(0.02)
    this.blocks = Array.from({ length: nLayer }, () => new GptBlock(nEmbd, nHead, batch, seqLen))
    this.head = new Linear(nEmbd, vocabSize)
  }

  /** Builds the forward graph only; the caller must never force the result. */
  forward(ids: IndexTensor<[number]>): AnyTensor {
    let x = this.tokEmb.indexSelect(ids, 0).reshape([this.batch, this.seqLen, this.tokEmb.shape[1]!])
    for (const block of this.blocks) x = block.forward(x)
    return this.head.forward(x as never) as AnyTensor
  }
}

/** Branded eagerly via `.toIndex()` so nothing inside the traced forward
 * ever has to read `.data`. */
function randomIds(count: number, vocabSize: number): IndexTensor<[number]> {
  const data = new Float32Array(count)
  for (let i = 0; i < count; i++) data[i] = Math.floor(Math.random() * vocabSize)
  return fromFlat(data, [count]).toIndex()
}

interface TraceCase extends BenchCaseSpec {
  build: () => AnyTensor
}

async function main(): Promise<void> {
  // All setup happens once, eagerly, outside the timed region; only the
  // forward trace is timed, under interp (lazy, no native).
  configure({ lazy: false })

  const smoke = isSmokeRun()
  const mlpCfg = smoke ? MLP_SMOKE[0]! : MLP_LEGACY.find(c => c.id === "mlp-legacy-b64")!
  const gptSizes = smoke ? [NANOGPT_SMOKE] : NANOGPT_SIZES
  const mlpLayer1 = new Linear(mlpCfg.inputDim, mlpCfg.hiddenDim)
  const mlpLayer2 = new Linear(mlpCfg.hiddenDim, mlpCfg.outputDim)
  const mlpInput = randn([mlpCfg.batch, mlpCfg.inputDim]) as AnyTensor

  const gpts = new Map<string, { model: TraceGpt; ids: IndexTensor<[number]> }>()
  for (const size of gptSizes) {
    const model = new TraceGpt(size.batch, size.blockSize, size.nEmbd, size.nHead, size.nLayer, size.vocabSize)
    const ids = randomIds(size.batch * size.blockSize, size.vocabSize)
    gpts.set(size.id, { model, ids })
  }

  const CASES: readonly TraceCase[] = [
    {
      id: "trace-mlp",
      modes: ["interp"],
      build: () => (mlpLayer2.forward(mlpLayer1.forward(mlpInput as never).relu() as never) as AnyTensor),
    },
    ...gptSizes.map(size => {
      const letter = size.id.split("-")[1]!
      const { model, ids } = gpts.get(size.id)!
      return {
        id: `trace-gpt-${letter}`,
        modes: ["interp"] as const,
        build: () => model.forward(ids),
      }
    }),
  ]

  await bench("micro-trace", CASES, (kase, mode) => {
    configure({ lazy: mode !== "eager" })
    // Never force evaluation: it would mix eval time into what is
    // supposed to be pure graph construction.
    kase.build()
  })

  configure({ lazy: false })
}

await main()
