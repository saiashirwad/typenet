"use tsover"

import type { DimDivCheck, DimMul } from "../../shape.ts"
import { DimMul as dimMul } from "../../shape.ts"
import type { Tensor } from "../../tensor.ts"
import { Module } from "../module.ts"
import { SHAPE_EFFECT } from "../sequential.ts"
import { GELU } from "./activation.ts"
import { MultiHeadAttention } from "./attention.ts"
import { Dropout } from "./dropout.ts"
import { Linear } from "./linear.ts"
import { LayerNorm } from "./norm.ts"

/**
 * A pre-norm transformer block (W5.2-A step 4, §3.7):
 *
 * ```
 * h = x + attn(ln1(x))
 * y = h + drop(proj(gelu(fc(ln2(h)))))
 * ```
 *
 * Pre-norm (`ln` INSIDE the residual branch, not after the sum) because
 * that is what makes a deep stack trainable without a warmup schedule —
 * the residual stream stays an identity path from embedding to head.
 *
 * The MLP is the standard `4·D` expansion, carried as `DimMul<4, D>` so
 * the two `Linear`s' widths are derived rather than restated: a `D` of 384
 * gives a `Linear<384, 1536>` and a `Linear<1536, 384>` with no literal
 * `1536` written anywhere.
 *
 * The two residual adds are the tsover `+` operator (hence the file's
 * `"use tsover"` directive) rather than `.add(...)`: `x + y` goes through
 * the same `Broadcast`-checked overload `.add` does, so nothing is weaker
 * — it just reads like the equation it is.
 */
export class TransformerBlock<D extends number, H extends number> extends Module {
  /** Same reasoning as {@link MultiHeadAttention}'s: the block owns `D`. */
  declare readonly [SHAPE_EFFECT]: [effect: "mapLast", In: D, Out: D]

  readonly ln1: LayerNorm<D>
  readonly attn: MultiHeadAttention<D, H>
  readonly ln2: LayerNorm<D>
  readonly fc: Linear<D, DimMul<4, D>>
  readonly act: GELU
  readonly proj: Linear<DimMul<4, D>, D>
  readonly drop: Dropout

  constructor(
    d: D,
    h: H & DimDivCheck<D, H>,
    options: {
      causal?: boolean
      dropout?: number
      bias?: boolean
      qkvFused?: boolean
      eps?: number
    } = {},
  ) {
    super()
    const p = options.dropout ?? 0
    this.ln1 = new LayerNorm(d, { eps: options.eps })
    // D20's forwarding discipline, and the reason this call carries
    // EXPLICIT type arguments: `h`'s declared type here is already
    // `H & DimDivCheck<D, H>`, so letting inference re-derive `H` from it
    // would make the callee's own `DimDivCheck<D, H>` a check of the
    // intersection against itself — trivially satisfied, and the
    // divisibility precondition would be silently discharged one level up
    // from where the widths are actually known.
    this.attn = new MultiHeadAttention<D, H>(d, h, {
      causal: options.causal ?? true,
      dropout: p,
      bias: options.bias,
      qkvFused: options.qkvFused,
    })
    this.ln2 = new LayerNorm(d, { eps: options.eps })
    this.fc = new Linear(d, dimMul(4, d), { bias: options.bias !== false })
    this.act = new GELU()
    this.proj = new Linear(dimMul(4, d), d, { bias: options.bias !== false })
    this.drop = new Dropout(p)
  }

  forward<B extends number, T extends number>(
    x: Tensor<[B, T, D]>,
  ): Tensor<[B, T, D]> {
    const h = x + this.attn.forward(this.ln1.forward(x))
    const mlp = this.drop.forward(
      this.proj.forward(this.act.forward(this.fc.forward(this.ln2.forward(h)))),
    )
    return h + mlp
  }
}
