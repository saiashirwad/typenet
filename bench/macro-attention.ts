// One causal MHA block, forward and forward+backward, at the nanoGPT
// S/M/L configs, via bench/models/attention.ts. The `attn-<letter>` /
// `attn-<letter>-fwd` case ids are load-bearing; do not rename them.

import { disableNative, useNative } from "../index.ts"
import { configure } from "../src/lazy.ts"
import type { AnyTensor } from "../src/tensor.ts"
import { bench, type BenchCaseSpec, isSmokeRun, type Mode } from "./lib/harness.ts"
import { NANOGPT_SIZES, NANOGPT_SMOKE } from "./lib/sizes.ts"
import { CausalSelfAttention, randomAttentionInput } from "./models/attention.ts"

interface AttnCase extends BenchCaseSpec {
  batch: number
  seqLen: number
  nEmbd: number
  nHead: number
  backward: boolean
}

// Batched (rank-4) matmul never takes the Accelerate fast path in eager
// mode (the shortcut only fires for batchCount === 1), so attn-l's
// backward at eager speed is a multi-second naive triple-loop GEMM per
// call (~15 s fwd / ~42 s fwd+bwd measured) -- minutes per mode at the
// harness's sample floor. Interp pays the identical per-op cost, so the
// backward case runs interp-only while attn-s / attn-m keep the full
// eager+interp comparison.
const HEAVY_BACKWARD_MODES: readonly Mode[] = ["interp"]

const SIZES = isSmokeRun() ? [NANOGPT_SMOKE] : NANOGPT_SIZES

const CASES: readonly AttnCase[] = SIZES.flatMap(size => {
  const letter = size.id.split("-")[1]! // "nanogpt-s" -> "s"
  const shared = { batch: size.batch, seqLen: size.blockSize, nEmbd: size.nEmbd, nHead: size.nHead }
  return [
    // Forward+backward. `native` is excluded: backward chains a `permute`
    // onto a view-derived tensor and the native GEMM rejects the resulting
    // stride pattern (MatMulUnexpectedStriding). Forward alone runs clean
    // on native (the `-fwd` case); re-enable native here once strided GEMM
    // lands.
    {
      id: `attn-${letter}`,
      ...shared,
      backward: true,
      modes: letter === "l" ? HEAVY_BACKWARD_MODES : ["eager", "interp"],
    },
    { id: `attn-${letter}-fwd`, ...shared, backward: false },
  ]
})

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
  const models = new Map<string, CausalSelfAttention>()
  const modelFor = (kase: AttnCase): CausalSelfAttention => {
    let m = models.get(kase.id)
    if (!m) {
      m = new CausalSelfAttention(kase)
      models.set(kase.id, m)
    }
    return m
  }

  await bench("macro-attention", CASES, (kase, mode) => {
    setMode(mode)
    const model = modelFor(kase)
    model.zeroGrad()
    const x = randomAttentionInput(kase) as AnyTensor
    const out = model.forward(x)
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
