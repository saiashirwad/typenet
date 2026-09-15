"use tsover"

// nanoGPT written against today's API only: hand-rolled causal attention,
// softmax and layernorm composed from primitives, indexSelect-based token
// and position embeddings, crossEntropy(logits, number[]) for the loss.
// This is the naive-composition baseline, not a preview of the layer
// catalog.

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

/** GELU (tanh approximation) composed from primitives; no built-in `gelu`. */
function gelu(x: AnyTensor): AnyTensor {
  const c = Math.sqrt(2 / Math.PI)
  const inner = x.add(x.pow(3).mul(0.044715)).mul(c)
  return x.mul(0.5).mul(inner.tanh().add(1))
}

/** No built-in LayerNorm, so composed from primitives. */
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
 * Hand-rolled causal multi-head self-attention: one fused QKV projection,
 * heads split via `view` + `permute`, a precomputed additive causal mask,
 * softmax on the last axis.
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

    // The second `.view()` is a same-shape no-op in eager JS, but the
    // native evaluator materializes a contiguous copy there, which this
    // needs before the result becomes a matmul operand: candle's matmul
    // rejects this permute's swap-the-middle-two-axes stride pattern
    // (MatMulUnexpectedStriding, "non-contiguous lhs").
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
 * nanoGPT (Karpathy's `model.py` shape) built entirely from the public API:
 * `indexSelect`-based token + learned position embedding, `nLayer` pre-norm
 * transformer blocks, a final `LayerNorm`, and an untied `Linear` output
 * head.
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
    // Branded eagerly in the constructor: `.toIndex()` reads `.data`, which
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
    // `view` does not carry the index brand, and re-branding with
    // `.toIndex()` would read `.data` (an error inside `compile()`'s
    // trace, where `idx` is a placeholder). The ids are integral by
    // construction, so the brand is asserted rather than re-checked.
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
 * The ids are wrapped as a **float32** leaf (`fromFlat` + `.toIndex()`, not
 * `Tensor.indices`, which builds int32): the loss reaches them through
 * `oneHot`, and the tracer only lets an integer leaf feed a gather/scatter
 * index, so an int32 leaf into `oneHot` is rejected under `compile()`. */
export function nanoGptCrossEntropy(logits: AnyTensor, targets: readonly number[]): AnyTensor {
  const [B, T, V] = logits.shape as number[]
  // Asserted, not checked: `.toIndex()` reads `.data`, which is an error
  // inside `compile()`'s trace. `generateBatch` floors every target into
  // `[0, vocabSize)`, so integrality holds by construction.
  const ids = fromFlat(Float32Array.from(targets), [targets.length]) as unknown as IndexTensor<
    [number]
  >
  return crossEntropy(logits.view([B! * T!, V!]), ids as never)
}

/**
 * Deterministic synthetic next-token batch: draws `batch * (blockSize + 1)`
 * uniforms from the shared seeded generator and floors them into ids in
 * `[0, vocabSize)`. Row `b`'s first `blockSize` ids are the input; the same
 * row shifted one position over is the next-token target.
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
