"use tsover"

// nanoGPT written against **today's API only** (PLAN-V2 §4.2, W0.12): hand-
// rolled causal attention in the style of `examples/gat.ts`, softmax and
// layernorm composed from primitive ops (there is no `LayerNorm` in `nn.ts`
// yet), `indexSelect`-based token/position embedding, and
// `crossEntropy(logits, number[])` for the loss. This is a *baseline* — the
// "naive composition on today's runtime" row of §6.2 — not a preview of the
// layer catalog Wave 5 adds.
//
// W5.7 rewrites this model against the new API and must reproduce
// `bench/macro-nanogpt.ts`'s 50-step loss curve to 1e-6 (that file documents
// the exact seed / data / hyperparameters the reproduction depends on).

import { crossEntropy, fromFlat, Linear, Module, ones, rand, Tensor, zeros } from "../../index.ts"
import type { IndexTensor } from "../../src/shape.ts"

type AnyTensor = Tensor<any>

export interface NanoGptConfig {
  readonly batch: number
  readonly blockSize: number
  readonly nEmbd: number
  readonly nHead: number
  readonly nLayer: number
  readonly vocabSize: number
}

/**
 * GELU (tanh approximation, Hendrycks & Gimpel) composed from today's API —
 * there is no built-in `gelu`.
 */
function gelu(x: AnyTensor): AnyTensor {
  const c = Math.sqrt(2 / Math.PI)
  const inner = x.add(x.pow(3).mul(0.044715)).mul(c)
  return x.mul(0.5).mul(inner.tanh().add(1))
}

/** Composed from `mean`/`sub`/`sqrt` — there is no built-in `LayerNorm`. */
class LayerNorm extends Module {
  readonly gamma: AnyTensor
  readonly beta: AnyTensor
  private readonly eps: number

  constructor(dim: number, eps = 1e-5) {
    super()
    this.gamma = ones([dim]).requiresGrad()
    this.beta = zeros([dim]).requiresGrad()
    this.eps = eps
  }

  forward(x: AnyTensor): AnyTensor {
    const mu = x.mean(-1, true)
    const centered = x.sub(mu)
    const variance = centered.mul(centered).mean(-1, true)
    const xhat = centered.div(variance.add(this.eps).sqrt())
    return xhat.mul(this.gamma).add(this.beta)
  }
}

/**
 * Hand-rolled causal multi-head self-attention, in the style of
 * `examples/gat.ts`'s hand-rolled attention head: one fused QKV projection,
 * split into heads via `view` + `permute`, a precomputed additive causal
 * mask (`-1e9` above the diagonal, the same convention `gat.ts` uses for
 * "no edge"), and `softmax` on the last axis.
 */
class CausalSelfAttention extends Module {
  readonly qkv: Linear<number, number>
  readonly proj: Linear<number, number>
  private readonly nHead: number
  private readonly headDim: number
  private readonly mask: AnyTensor

  constructor(nEmbd: number, nHead: number, blockSize: number) {
    super()
    if (nEmbd % nHead !== 0) {
      throw new Error(
        `CausalSelfAttention: nEmbd ${nEmbd} is not divisible by nHead ${nHead}`,
      )
    }
    this.nHead = nHead
    this.headDim = nEmbd / nHead
    this.qkv = new Linear(nEmbd, 3 * nEmbd)
    this.proj = new Linear(nEmbd, nEmbd)

    const maskData = new Float32Array(blockSize * blockSize)
    for (let i = 0; i < blockSize; i++) {
      for (let j = 0; j < blockSize; j++) {
        maskData[i * blockSize + j] = j <= i ? 0 : -1e9
      }
    }
    this.mask = fromFlat(maskData, [1, 1, blockSize, blockSize])
  }

  forward(x: AnyTensor): AnyTensor {
    const [B, T, C] = x.shape as number[]
    const qkv = this.qkv.forward(x)
    const q = qkv.narrow(-1, 0, C!)
    const k = qkv.narrow(-1, C!, C!)
    const v = qkv.narrow(-1, 2 * C!, C!)

    // The extra `.view()` after `.permute()` is a same-shape reshape, a
    // no-op in eager JS — but a `View` node's *native* evaluator always
    // materializes a contiguous copy first (`get(input)?.contiguous()?`,
    // `native/src/lib.rs`), which this needs before the result becomes a
    // matmul operand: candle's native matmul supports a plain last-two-dims
    // transpose directly but rejects this permute's "swap the middle two
    // axes" stride pattern as an operand (`MatMulUnexpectedStriding`,
    // `"non-contiguous lhs"`) — a pre-existing native-backend gap (no
    // general strided-GEMM support yet, PLAN-V2 §2.9's `_NO_STRIDED_GEMM`),
    // not something fixable from here.
    const splitHeads = (t: AnyTensor): AnyTensor =>
      t.view([B!, T!, this.nHead, this.headDim])
        .permute(0, 2, 1, 3)
        .view([B!, this.nHead, T!, this.headDim])
    const qh = splitHeads(q)
    const kh = splitHeads(k)
    const vh = splitHeads(v)

    const scale = 1 / Math.sqrt(this.headDim)
    const scores = qh.matmul(kh.transpose(-2, -1)).mul(scale)
    const attn = scores.add(this.mask).softmax(-1)
    const merged = attn.matmul(vh).permute(0, 2, 1, 3).view([B!, T!, C!])
    return this.proj.forward(merged)
  }
}

class MLP extends Module {
  readonly fc: Linear<number, number>
  readonly proj: Linear<number, number>

  constructor(nEmbd: number) {
    super()
    this.fc = new Linear(nEmbd, 4 * nEmbd)
    this.proj = new Linear(4 * nEmbd, nEmbd)
  }

  forward(x: AnyTensor): AnyTensor {
    return this.proj.forward(gelu(this.fc.forward(x)))
  }
}

class Block extends Module {
  readonly ln1: LayerNorm
  readonly attn: CausalSelfAttention
  readonly ln2: LayerNorm
  readonly mlp: MLP

  constructor(cfg: NanoGptConfig) {
    super()
    this.ln1 = new LayerNorm(cfg.nEmbd)
    this.attn = new CausalSelfAttention(cfg.nEmbd, cfg.nHead, cfg.blockSize)
    this.ln2 = new LayerNorm(cfg.nEmbd)
    this.mlp = new MLP(cfg.nEmbd)
  }

  forward(x: AnyTensor): AnyTensor {
    const a = x.add(this.attn.forward(this.ln1.forward(x)))
    return a.add(this.mlp.forward(this.ln2.forward(a)))
  }
}

/**
 * nanoGPT (Karpathy's `model.py` shape) built entirely from today's public
 * API: `indexSelect`-based token + learned position embedding, `nLayer`
 * pre-norm transformer blocks, a final `LayerNorm`, and an untied `Linear`
 * output head (weight tying is a Wave 5 concern, not a baseline one).
 */
export class NanoGptLegacy extends Module {
  readonly wte: AnyTensor
  readonly wpe: AnyTensor
  readonly blocks: Block[]
  readonly lnF: LayerNorm
  readonly head: Linear<number, number>
  private readonly cfg: NanoGptConfig
  private readonly posIndex: IndexTensor<[number]>

  constructor(cfg: NanoGptConfig) {
    super()
    this.cfg = cfg
    // Same fan-in-scaled uniform init `Linear`'s own constructor uses.
    const k = 1 / Math.sqrt(cfg.nEmbd)
    this.wte = rand([cfg.vocabSize, cfg.nEmbd]).mul(2 * k).sub(k).detach().requiresGrad()
    this.wpe = rand([cfg.blockSize, cfg.nEmbd]).mul(2 * k).sub(k).detach().requiresGrad()
    this.blocks = Array.from({ length: cfg.nLayer }, () => new Block(cfg))
    this.lnF = new LayerNorm(cfg.nEmbd)
    this.head = new Linear(cfg.nEmbd, cfg.vocabSize)
    // `.toIndex()` brands it for `indexSelect` (OWNER-5, D25). Done in
    // the constructor, eagerly: the brand check reads `.data`, which
    // would force the graph if it happened inside `forward` under
    // `compile()`.
    this.posIndex = fromFlat(
      Float32Array.from({ length: cfg.blockSize }, (_, i) => i),
      [cfg.blockSize],
    ).toIndex()
  }

  /** `idx`: `[B, T]` float32 token ids (T must equal `cfg.blockSize`). Returns `[B, T, vocabSize]` logits. */
  forward(idx: AnyTensor): AnyTensor {
    const [B, T] = idx.shape as number[]
    // `view` does not carry the index brand forward, and re-branding with
    // `.toIndex()` would read `.data` — fatal here, because `forward` runs
    // inside `compile()`'s trace where `idx` is a placeholder. The ids
    // `generateBatch` produced are integral by construction, so the brand
    // is asserted rather than re-checked.
    const flatIdx = idx.view([B! * T!]) as unknown as IndexTensor<[number]>
    const tok = this.wte.indexSelect(flatIdx).view([B!, T!, this.cfg.nEmbd])
    const pos = this.wpe.indexSelect(this.posIndex) // [T, C], broadcasts over batch
    let x: AnyTensor = tok.add(pos)
    for (const block of this.blocks) x = block.forward(x)
    x = this.lnF.forward(x)
    return this.head.forward(x)
  }
}

/** `logits`: `[B, T, V]`; `targets`: `B * T` next-token ids, row-major.
 *
 * `crossEntropy` takes a branded `IndexTensor` (OWNER-5, D26), so the
 * plain array is wrapped here. It is deliberately a **float32** leaf
 * (`fromFlat` + `.toIndex()`, not `Tensor.indices`, which builds int32):
 * the loss reaches the ids through `oneHot`, and `serializeLazyGraph`
 * only lets an integer leaf feed a gather/scatter index — an int32 leaf
 * into `oneHot` is rejected when this model is traced by `compile()`.
 * float32 is also the dtype the pre-D26 array path used, so the loss
 * curve this baseline pins is bit-for-bit what it was. */
export function nanoGptCrossEntropy(logits: AnyTensor, targets: readonly number[]): AnyTensor {
  const [B, T, V] = logits.shape as number[]
  // The brand is asserted, not checked: `.toIndex()` reads `.data`, and
  // this runs inside `compile()`'s trace, where reading a tensor's values
  // is an error. `generateBatch` floors every target into `[0, vocabSize)`,
  // so integrality holds by construction.
  const ids = fromFlat(Float32Array.from(targets), [targets.length]) as unknown as IndexTensor<
    [number]
  >
  return crossEntropy(logits.view([B! * T!, V!]), ids as never)
}

/**
 * A deterministic synthetic next-token batch — there is no dataset here,
 * only what pins the loss curve numerically. Draws `batch * (blockSize +
 * 1)` uniform values through the shared seeded generator (`configure({
 * seed })` in `src/lazy.ts` — the same stream `Tensor.rand`/`Tensor.randn`
 * draw from) and floors them into ids in `[0, vocabSize)`. Row `b`'s first
 * `blockSize` ids are the input; the same row shifted one position over is
 * the next-token target.
 */
export function generateBatch(cfg: NanoGptConfig): { idx: AnyTensor; targets: number[] } {
  const { batch, blockSize, vocabSize } = cfg
  const u = rand([batch, blockSize + 1]).data as Float32Array
  const idxData = new Float32Array(batch * blockSize)
  const targets: number[] = new Array(batch * blockSize)
  const toId = (u01: number): number => Math.min(vocabSize - 1, Math.floor(u01 * vocabSize))
  for (let b = 0; b < batch; b++) {
    const row = b * (blockSize + 1)
    for (let t = 0; t < blockSize; t++) {
      idxData[b * blockSize + t] = toId(u[row + t]!)
      targets[b * blockSize + t] = toId(u[row + t + 1]!)
    }
  }
  return { idx: fromFlat(idxData, [batch, blockSize]), targets }
}
