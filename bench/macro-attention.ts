// One causal MHA block, forward and forward+backward, at the nanoGPT
// S/M/L head configs of PLAN-V2 §0 (PLAN-V2 §4.2, W0.3). Hand-composed
// from today's ops via `bench/models/attention.ts` — the same shape
// `examples/gat.ts` writes. Read by W4.6, W5.2 and gate G4.5, none of
// which create it: the `attn-s` / `attn-m` / `attn-l` case ids are load-
// bearing and must not be renamed.

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
// mode (`eager.ts`'s shortcut only fires for `batchCount === 1`), so a
// causal-attention forward+backward at the L config (batch·head = 384,
// T = 256) is a genuinely multi-second naive triple-loop GEMM — measured
// at ~15 s fwd / ~42 s fwd+bwd per call on this machine. At the harness's
// 13-sample floor that is minutes per mode; running it in *two* slow
// modes (eager and interp, which share the same naive kernel) needs
// ~18 minutes for this one case alone. `attn-l`'s backward case is
// therefore interp-only — eager would only reconfirm the identical
// per-op cost interp already pays, at double the wall-clock — while
// `attn-s` / `attn-m` (both far smaller) keep the full eager+interp
// comparison.
const HEAVY_BACKWARD_MODES: readonly Mode[] = ["interp"]

// Smoke keeps only the single owner-specified tiny config — the smallest
// possible "1-2 sizes" reading of PLAN-V2's S/M/L sweep.
const SIZES = isSmokeRun() ? [NANOGPT_SMOKE] : NANOGPT_SIZES

const CASES: readonly AttnCase[] = SIZES.flatMap(size => {
  const letter = size.id.split("-")[1]! // "nanogpt-s" -> "s"
  const shared = { batch: size.batch, seqLen: size.blockSize, nEmbd: size.nEmbd, nHead: size.nHead }
  return [
    // The primary, load-bearing id: forward+backward, one full training-
    // shaped pass — this is the number later items read. `native` is
    // excluded here: the backward of a batched (rank-4) matmul chains a
    // `permute` onto an already-view-derived tensor, and today's native
    // GEMM rejects the resulting stride pattern
    // (`MatMulUnexpectedStriding … non-contiguous lhs`) — a pre-existing
    // native-backend gap (no strided-GEMM support yet, §2.9's
    // `_NO_STRIDED_GEMM`), not something this bench script can paper
    // over without touching native/ (out of scope for W0.3). Forward
    // alone runs clean on native (see `-fwd` below); re-enable native
    // here once strided GEMM lands.
    {
      id: `attn-${letter}`,
      ...shared,
      backward: true,
      modes: letter === "l" ? HEAVY_BACKWARD_MODES : ["eager", "interp"],
    },
    // Forward-only companion, per §4.2's "forward and forward+backward" —
    // runs in every mode, native included.
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
