import { assertChecked } from "../../cast.ts"
import { type DimDiv, DimDiv as dimDiv, type DimDivCheck, type DimMul, DimMul as dimMul } from "../../shape.ts"
import type { Tensor } from "../../tensor.ts"
import { sdpa } from "../functional.ts"
import { Module } from "../module.ts"
import { SHAPE_EFFECT } from "../sequential.ts"
import { Linear } from "./linear.ts"

/** Multi-head self-attention over `[B, T, D]`; `D` must be divisible by `H` (checked at the constructor in both type and value). */
export class MultiHeadAttention<D extends number, H extends number> extends Module {
  declare readonly [SHAPE_EFFECT]: [effect: "mapLast", In: D, Out: D]

  /** The fused `[D, 3D]` projection, or `null` under `qkvFused: false`. */
  readonly qkv: Linear<D, DimMul<3, D>> | null
  /** The three separate `[D, D]` projections, or `null` when fused. */
  readonly wq: Linear<D, D> | null
  readonly wk: Linear<D, D> | null
  readonly wv: Linear<D, D> | null
  readonly proj: Linear<D, D>

  readonly d: D
  readonly h: H
  /** `D / H`, carried as a literal by the `DimDiv` type/value twin. */
  readonly headDim: DimDiv<D, H>
  readonly causal: boolean
  readonly p: number

  constructor(
    d: D,
    h: H & DimDivCheck<D, H>,
    options: {
      causal?: boolean
      dropout?: number
      bias?: boolean
      qkvFused?: boolean
    } = {},
  ) {
    super()
    const heads = h as H
    // Runtime twin of DimDivCheck: the type only proves divisibility for
    // literals, so the value side repeats it.
    if (!Number.isInteger(d) || !Number.isInteger(heads) || heads <= 0 || d <= 0) {
      throw new Error(
        `MultiHeadAttention: d and h must be positive integers, got d=${d}, h=${heads}`,
      )
    }
    if (d % heads !== 0) {
      throw new Error(
        `MultiHeadAttention: ${heads} heads do not divide a model width of ${d} `
          + `(head width would be ${d / heads})`,
      )
    }
    this.d = d
    this.h = heads
    this.headDim = dimDiv(d, heads)
    this.causal = options.causal ?? false
    this.p = options.dropout ?? 0
    const bias = options.bias !== false
    const fused = options.qkvFused !== false
    this.qkv = fused ? new Linear(d, dimMul(3, d), { bias }) : null
    this.wq = fused ? null : new Linear(d, d, { bias })
    this.wk = fused ? null : new Linear(d, d, { bias })
    this.wv = fused ? null : new Linear(d, d, { bias })
    this.proj = new Linear(d, d, { bias })
  }

  /** `[B, T, D] -> [B, T, D]`; head split/merge go through `unflatten`/`permute`/`flatten` because `view` cannot reduce with generic `B`/`T`. */
  forward<B extends number, T extends number>(
    x: Tensor<[B, T, D]>,
  ): Tensor<[B, T, D]> {
    if (x.shape.length !== 3) {
      throw new Error(
        `MultiHeadAttention.forward: expects [B, T, ${this.d}], got a rank-${x.shape.length} tensor`,
      )
    }
    if (x.shape[2] !== this.d) {
      throw new Error(
        `MultiHeadAttention.forward: expects a model width of ${this.d}, got [${x.shape.join(", ")}]`,
      )
    }
    const { h, headDim } = this
    let q: Tensor<[B, T, D]>
    let k: Tensor<[B, T, D]>
    let v: Tensor<[B, T, D]>
    if (this.qkv) {
      const fused = this.qkv.forward(x)
      // `narrow` only knows the cut length as "some number", so assert
      // the result really is [B, T, D]; one helper serves all three cuts.
      const cut = (offset: number): Tensor<[B, T, D]> => assertChecked<[B, T, D]>(fused.narrow(2, offset, this.d))
      q = cut(0)
      k = cut(this.d)
      v = cut(2 * this.d)
    } else {
      q = this.wq!.forward(x)
      k = this.wk!.forward(x)
      v = this.wv!.forward(x)
    }
    // `k` goes straight to `[B, H, Dh, T]`, the transposed form `sdpa` takes.
    const q4 = q.unflatten(2, [h, headDim]).permute(0, 2, 1, 3)
    const k4 = k.unflatten(2, [h, headDim]).permute(0, 2, 3, 1)
    const v4 = v.unflatten(2, [h, headDim]).permute(0, 2, 1, 3)
    const ctx = sdpa(q4, k4, v4, { causal: this.causal, dropout: this.training ? this.p : 0 })
    // `DimMul<H, DimDiv<D, H>>` does not reduce to `D` for generic `D`/`H`,
    // so assert the merged shape.
    const merged = assertChecked<[B, T, D]>(ctx.permute(0, 2, 1, 3).flatten(2, 3))
    return this.proj.forward(merged)
  }
}
