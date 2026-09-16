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

// `"use tsover"` lets the residual adds be written `x + y` (the same
// Broadcast-checked overload `.add` uses).

// Pre-norm block: h = x + attn(ln1(x)); y = h + drop(proj(gelu(fc(ln2(h))))).
export class TransformerBlock<D extends number, H extends number> extends Module {
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
    // Explicit type arguments: inference from the `h` intersection would
    // re-derive `H` and discharge the divisibility check against itself.
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
