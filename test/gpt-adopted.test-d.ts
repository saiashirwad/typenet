/**
 * The §3.7 GPT sample, compiled against the signatures PLAN-V2 adopts
 * (D19 / D20 / D22 / D23 / D25 / D26, `LastDimCheck` form A) and the real
 * `src/shape.ts`. Checked-in version of `scratchpad/review/gpt-adopted.ts`,
 * with the `Embedding` / `crossEntropy` cases from
 * `scratchpad/final-verify/api-forms.ts` ported onto the adopted forms —
 * that probe used the *rejected* variadic `LastDimCheck` spelling and hid
 * the consequence behind an `as never`, so it is not copied as written.
 *
 * The layers that do not exist yet (LayerNorm, Embedding, Dropout, GELU,
 * sdpa, the index-typed crossEntropy) are `declare`d here with exactly
 * the signatures their items will implement; `Linear` and `Module` are
 * the real ones. When those land, this file becomes a straight import —
 * until then it is the proof that the shape algebra in W0.13 is enough to
 * type an attention block end to end.
 *
 * `flatten` / `unflatten` are free-function stand-ins for the `Tensor`
 * methods W1.8 adds, for the same reason (see flatten.test-d.ts).
 */
import { assertChecked } from "../src/cast.ts"
import { Linear, Module } from "../src/nn.ts"
import { DimDiv, DimMul } from "../src/shape.ts"
import type { DimDivCheck, FlattenCheck, FlattenShape, IndexTensor, Init, LastDimCheck, Shape, UnflattenCheck, UnflattenShape } from "../src/shape.ts"
import type { Tensor } from "../src/tensor.ts"

type Equal<A, B> = (<T>() => T extends A ? 1 : 2) extends (<T>() => T extends B ? 1 : 2) ? true : false
type Expect<T extends true> = T

declare function flatten<S extends Shape, const F extends number, const T extends number>(
  t: Tensor<S>,
  from: F & FlattenCheck<S, F, T>,
  to: T,
): Tensor<FlattenShape<S, F, T>>

declare function unflatten<S extends Shape, const D extends number, const Sizes extends number[]>(
  t: Tensor<S>,
  dim: D & UnflattenCheck<S, D, Sizes>,
  sizes: Sizes,
): Tensor<UnflattenShape<S, D, Sizes>>

declare class LayerNorm<D extends number> {
  constructor(d: D)
  forward<S extends Shape>(x: Tensor<S> & LastDimCheck<S, D>): Tensor<S>
}

declare class Embedding<V extends number, D extends number> {
  readonly weight: Tensor<[V, D]>
  constructor(v: V, d: D)
  forward<S extends Shape>(ids: IndexTensor<S>): Tensor<[...S, D]>
}

declare class Dropout {
  constructor(p?: number)
  forward<S extends Shape>(x: Tensor<S>): Tensor<S>
}

declare class GELU {
  forward<S extends Shape>(x: Tensor<S>): Tensor<S>
}

declare function sdpa<B extends number, H extends number, T extends number, K extends number>(
  q: Tensor<[B, H, T, K]>,
  k: Tensor<[B, H, K, T]>,
  v: Tensor<[B, H, T, K]>,
  o?: { causal?: boolean; dropout?: number },
): Tensor<[B, H, T, K]>

declare function crossEntropy<S extends Shape>(logits: Tensor<S>, targets: IndexTensor<Init<S>>): Tensor<[]>

// ---- MHA, exactly as §3.7 prints it ----------------------------------------
class MultiHeadAttention<D extends number, H extends number> extends Module {
  readonly qkv: Linear<D, DimMul<3, D>>
  readonly proj: Linear<D, D>
  readonly d: D
  readonly h: H
  readonly causal: boolean
  readonly p: number

  constructor(d: D, h: H & DimDivCheck<D, H>, o: { causal?: boolean; dropout?: number } = {}) {
    super()
    this.d = d
    this.h = h as H
    this.causal = o.causal ?? false
    this.p = o.dropout ?? 0
    this.qkv = new Linear(d, DimMul(3, d))
    this.proj = new Linear(d, d)
  }

  forward<B extends number, T extends number>(x: Tensor<[B, T, D]>): Tensor<[B, T, D]> {
    const dh = DimDiv(this.d, this.h)
    const qkv = this.qkv.forward(x)
    const q = assertChecked<[B, T, D]>(qkv.narrow(2, 0, this.d))
    const k = assertChecked<[B, T, D]>(qkv.narrow(2, this.d, this.d))
    const v = assertChecked<[B, T, D]>(qkv.narrow(2, DimMul(2, this.d), this.d))
    const q4 = unflatten(q, 2, [this.h, dh]).permute(0, 2, 1, 3)
    const k4 = unflatten(k, 2, [this.h, dh]).permute(0, 2, 3, 1)
    const v4 = unflatten(v, 2, [this.h, dh]).permute(0, 2, 1, 3)
    const ctx = sdpa(q4, k4, v4, { causal: this.causal, dropout: this.p })
    const merged = flatten(ctx.permute(0, 2, 1, 3), 2, 3)
    return this.proj.forward(assertChecked<[B, T, D]>(merged))
  }
}

// the head dim is derived, and the divisibility precondition is enforced
const _mha = new MultiHeadAttention(384, 6)
// @ts-expect-error 384 heads do not divide into 5
const _mhaBad = new MultiHeadAttention(384, 5)

// ---- a block, and the Embedding / crossEntropy path ------------------------
class Block<D extends number, H extends number> extends Module {
  readonly ln1: LayerNorm<D>
  readonly attn: MultiHeadAttention<D, H>
  readonly ln2: LayerNorm<D>
  readonly fc: Linear<D, DimMul<4, D>>
  readonly act: GELU
  readonly proj: Linear<DimMul<4, D>, D>
  readonly drop: Dropout

  constructor(d: D, h: H & DimDivCheck<D, H>) {
    super()
    this.ln1 = new LayerNorm(d)
    this.attn = new MultiHeadAttention<D, H>(d, h, { causal: true })
    this.ln2 = new LayerNorm(d)
    this.fc = new Linear(d, DimMul(4, d))
    this.act = new GELU()
    this.proj = new Linear(DimMul(4, d), d)
    this.drop = new Dropout(0.1)
  }

  forward<B extends number, T extends number>(x: Tensor<[B, T, D]>): Tensor<[B, T, D]> {
    const h = x.add(this.attn.forward(this.ln1.forward(x)))
    const mlp = this.drop.forward(this.proj.forward(this.act.forward(this.fc.forward(this.ln2.forward(h)))))
    return h.add(mlp)
  }
}

function _gpt<B extends number, T extends number, V extends number, D extends number>(
  tok: IndexTensor<[B, T]>,
  emb: Embedding<V, D>,
  block: Block<D, 6>,
  ln: LayerNorm<D>,
  head: Linear<D, V>,
  targets: IndexTensor<[B, T]>,
) {
  const x = emb.forward(tok)
  type _1 = Expect<Equal<typeof x.shape, [B, T, D]>>
  const h = block.forward(x)
  type _2 = Expect<Equal<typeof h.shape, [B, T, D]>>
  const logits = head.forward(ln.forward(h))
  const loss = crossEntropy(logits, targets)
  type _3 = Expect<Equal<typeof loss.shape, []>>
  return loss
}

// the concrete end of the same path: `Init<S>` picks the target shape off
// the logits with no manual reshape, at rank 3 and at rank 2
declare const ids: IndexTensor<[8, 256]>
declare const labels: IndexTensor<[8]>
declare const logits3: Tensor<[8, 256, 65]>
declare const logits2: Tensor<[8, 10]>
const _loss3 = crossEntropy(logits3, ids)
const _loss2 = crossEntropy(logits2, labels)
// @ts-expect-error targets must be [8, 256], not [8]
const _lossBad = crossEntropy(logits3, labels)

declare const floats: Tensor<[8, 256]>
// @ts-expect-error an embedding takes index tensors, not floats
const _embBad = new Embedding(65, 384).forward(floats)

export { _embBad, _gpt, _loss2, _loss3, _lossBad, _mha, _mhaBad, Block, MultiHeadAttention }
