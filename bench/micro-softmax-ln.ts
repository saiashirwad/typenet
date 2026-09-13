// softmax (causal and not) and layer norm at [N,C], C ∈ {64,128,384,1024},
// plus cross-entropy fwd+bwd at [16384,65] (PLAN-V2 §4.2, W0.3). Neither
// softmax nor layer norm has a fused row-wise kernel yet (D8 is a later
// wave) — softmax uses the builtin `.softmax()` (itself composed from
// max/sub/exp/div, see `tensor.ts`'s `softmaxShift`), and layer norm is
// hand-composed here from `mean`/`sub`/`pow`/`sqrt`/`div`.
//
// `ce-16384x65` is load-bearing: G4.3 reads it.

import { crossEntropy, disableNative, useNative } from "../index.ts"
import { rand, randn } from "../src/factories.ts"
import { configure } from "../src/lazy.ts"
import { type AnyTensor, fromFlat } from "../src/tensor.ts"
import { bench, type BenchCaseSpec, isSmokeRun, type Mode } from "./lib/harness.ts"
import { SOFTMAX_LN_FULL, SOFTMAX_LN_SMOKE } from "./lib/sizes.ts"

const SMOKE = isSmokeRun()
const SIZE_CONFIG = SMOKE ? SOFTMAX_LN_SMOKE : SOFTMAX_LN_FULL
const { rows: ROWS, widths: WIDTHS, ceRows: CE_ROWS, ceCols: CE_COLS } = SIZE_CONFIG

type Kind = "softmax" | "softmax-causal" | "layernorm" | "cross-entropy"

interface SoftmaxLnCase extends BenchCaseSpec {
  kind: Kind
  rows: number
  cols: number
}

const CASES: readonly SoftmaxLnCase[] = [
  ...WIDTHS.map(c => ({ id: `softmax-${c}`, kind: "softmax" as const, rows: ROWS, cols: c })),
  // Causal softmax only makes sense on a square score-like matrix.
  ...WIDTHS.map(c => ({ id: `softmax-causal-${c}`, kind: "softmax-causal" as const, rows: c, cols: c })),
  ...WIDTHS.map(c => ({ id: `layernorm-${c}`, kind: "layernorm" as const, rows: ROWS, cols: c })),
  // Load-bearing (full only): G4.3 reads this exact id at 16384x65 —
  // `ce-smoke` is a smoke-only stand-in, never read by that gate.
  {
    id: SMOKE ? "ce-smoke" : `ce-${CE_ROWS}x${CE_COLS}`,
    kind: "cross-entropy" as const,
    rows: CE_ROWS,
    cols: CE_COLS,
  },
]

function causalMask(n: number): AnyTensor {
  const data = new Float32Array(n * n)
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) data[i * n + j] = j > i ? -1e9 : 0
  }
  return fromFlat(data, [n, n])
}

/** Hand-composed layer norm over the last axis, with affine parameters —
 * no `LayerNorm` module exists in `src/nn.ts` yet. */
function layerNorm(x: AnyTensor, gamma: AnyTensor, beta: AnyTensor, eps = 1e-5): AnyTensor {
  const mean = x.mean(1, true)
  const centered = x.sub(mean)
  const variance = centered.pow(2).mean(1, true)
  const normed = centered.div(variance.add(eps).sqrt())
  return normed.mul(gamma).add(beta)
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
  const inputs = new Map<string, AnyTensor>()
  const masks = new Map<number, AnyTensor>()
  const affine = new Map<number, { gamma: AnyTensor; beta: AnyTensor }>()
  const targets = new Map<number, number[]>()

  await bench("micro-softmax-ln", CASES, (kase, mode) => {
    setMode(mode)
    const key = `${kase.rows}x${kase.cols}`
    let x = inputs.get(key)
    if (!x) {
      x = randn([kase.rows, kase.cols]) as AnyTensor
      if (kase.kind === "cross-entropy") x = x.requiresGrad()
      inputs.set(key, x)
    }

    switch (kase.kind) {
      case "softmax": {
        const out = x.softmax(1)
        out.data
        return
      }
      case "softmax-causal": {
        let mask = masks.get(kase.cols)
        if (!mask) {
          mask = causalMask(kase.cols)
          masks.set(kase.cols, mask)
        }
        const out = x.add(mask).softmax(1)
        out.data
        return
      }
      case "layernorm": {
        let ab = affine.get(kase.cols)
        if (!ab) {
          ab = { gamma: rand([kase.cols]) as AnyTensor, beta: rand([kase.cols]) as AnyTensor }
          affine.set(kase.cols, ab)
        }
        const out = layerNorm(x, ab.gamma, ab.beta)
        out.data
        return
      }
      case "cross-entropy": {
        let t = targets.get(kase.rows)
        if (!t) {
          t = Array.from({ length: kase.rows }, () => Math.floor(Math.random() * kase.cols))
          targets.set(kase.rows, t)
        }
        x.zeroGrad()
        const loss = crossEntropy(x as never, t)
        ;(loss as unknown as AnyTensor).backward()
        return
      }
    }
  })

  configure({ lazy: false })
  disableNative()
}

await main()
