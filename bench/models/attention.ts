// Bench-only causal multi-head self-attention built from primitives
// (matmul, permute, view, softmax); no MultiHeadAttention module exists yet.
// Read by bench/macro-attention.ts and bench/micro-trace.ts.

import { Linear, Module, randn } from "../../index.ts"
import { type AnyTensor, fromFlat } from "../../src/tensor.ts"

export interface AttentionConfig {
  batch: number
  seqLen: number
  nEmbd: number
  nHead: number
}

/** Lower-triangular-visible additive mask, `[1, 1, seqLen, seqLen]`:
 * `0` where a position may attend, `-1e9` where it may not (causal). */
function causalMask(seqLen: number): AnyTensor {
  const data = new Float32Array(seqLen * seqLen)
  for (let i = 0; i < seqLen; i++) {
    for (let j = 0; j < seqLen; j++) {
      data[i * seqLen + j] = j > i ? -1e9 : 0
    }
  }
  return fromFlat(data, [1, 1, seqLen, seqLen])
}

/**
 * One causal self-attention block: separate Q/K/V/output projections,
 * scaled dot-product attention with a causal mask, output projection.
 * Forward only.
 */
export class CausalSelfAttention extends Module {
  readonly q: Linear<number, number>
  readonly k: Linear<number, number>
  readonly v: Linear<number, number>
  readonly proj: Linear<number, number>
  readonly nHead: number
  readonly headDim: number
  private readonly mask: AnyTensor

  constructor(readonly config: AttentionConfig) {
    super()
    const { nEmbd, nHead, seqLen } = config
    if (nEmbd % nHead !== 0) {
      throw new Error(
        `CausalSelfAttention: nEmbd (${nEmbd}) must be divisible by nHead (${nHead})`,
      )
    }
    this.nHead = nHead
    this.headDim = nEmbd / nHead
    this.q = new Linear(nEmbd, nEmbd)
    this.k = new Linear(nEmbd, nEmbd)
    this.v = new Linear(nEmbd, nEmbd)
    this.proj = new Linear(nEmbd, nEmbd)
    this.mask = causalMask(seqLen)
  }

  /** `x`: `[batch, seqLen, nEmbd]` -> `[batch, seqLen, nEmbd]`. */
  /** `x`: `[batch, seqLen, nEmbd]` -> `[batch, seqLen, nEmbd]`. */
  forward(x: AnyTensor): AnyTensor {
    const { batch, seqLen, nEmbd } = this.config
    const { nHead, headDim } = this

    // Native GEMM requires contiguous operands and `permute` is a
    // metadata-only strided view, so this identity reshape forces the
    // materializing copy `matmul` needs (dropping it reproduces
    // MatMulUnexpectedStriding on the native path).
    const contiguous = (t: AnyTensor): AnyTensor => t.reshape([...t.shape])

    const toHeads = (t: AnyTensor): AnyTensor =>
      contiguous(
        t
          .view([batch, seqLen, nHead, headDim])
          .permute(0, 2, 1, 3), // [B, H, T, Dh]
      )

    const q = toHeads(this.q.forward(x as never) as AnyTensor)
    const k = toHeads(this.k.forward(x as never) as AnyTensor)
    const v = toHeads(this.v.forward(x as never) as AnyTensor)

    const kt = contiguous(k.permute(0, 1, 3, 2)) // [B, H, Dh, T]
    const scores = (q.matmul(kt) as AnyTensor).mul(1 / Math.sqrt(headDim)) // [B, H, T, T]
    const masked = scores.add(this.mask)
    const alpha = masked.softmax(3)
    const out = alpha.matmul(v) as AnyTensor // [B, H, T, Dh]

    const merged = contiguous(out.permute(0, 2, 1, 3)).reshape([batch, seqLen, nEmbd])
    return this.proj.forward(merged as never) as AnyTensor
  }
}

/** Fresh random `[batch, seqLen, nEmbd]` input for `config`. */
export function randomAttentionInput(config: AttentionConfig): AnyTensor {
  return randn([config.batch, config.seqLen, config.nEmbd]) as AnyTensor
}
