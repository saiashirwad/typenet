import { assertChecked } from "../../cast.ts"
import { type DimDiv, DimDiv as dimDiv, type DimDivCheck, type DimMul, DimMul as dimMul } from "../../shape.ts"
import type { Tensor } from "../../tensor.ts"
import { sdpa } from "../functional.ts"
import { Module } from "../module.ts"
import { SHAPE_EFFECT } from "../sequential.ts"
import { Linear } from "./linear.ts"

/**
 * Multi-head self-attention over `[B, T, D]` (W5.2-A, §3.7).
 *
 * The head width is DERIVED — `DimDiv<D, H>` — and the divisibility
 * precondition rides on the constructor's own parameter as
 * `h: H & DimDivCheck<D, H>`, so `new MultiHeadAttention(384, 5)` is a
 * compile error at the construction site rather than a runtime throw from
 * inside `unflatten`.
 *
 * **The `h` field is the bare `H`, never `H & DimDivCheck<D, H>`.** A
 * check intersection is fine as a *parameter* type — it is evaluated once,
 * at the call — but as a *property* type it is re-evaluated at every read,
 * and every downstream use of `this.h` (the `unflatten` sizes tuple, the
 * `DimDiv` value twin, a `Block<D, H>` forwarding its own `h`) inherits an
 * `ErrorMessage` branch it cannot discharge. That is the failure PLAN-V2's
 * Notes call "poisons every downstream use", and it is why the constructor
 * assigns through a widening cast rather than letting the parameter type
 * flow into the field.
 *
 * D20's forwarding discipline applies at every construction site: a
 * wrapper that takes `h: H & DimDivCheck<D, H>` must pass it on with
 * EXPLICIT type arguments (`new MultiHeadAttention<D, H>(d, h)`), or
 * inference re-derives `H` from the intersection and the check is silently
 * discharged against itself. {@link TransformerBlock} is written that way.
 *
 * Causal masking is a property of the `softmax{causal}` node (step 6): no
 * `[T, T]` buffer is ever allocated, on any path, which is what lets one
 * traced program serve every `T` the planner sees rather than one mask
 * constant per sequence length.
 */
export class MultiHeadAttention<D extends number, H extends number> extends Module {
  /**
   * `["mapLast", D, D]`, not `"identity"` (W4.9): attention OWNS the model
   * width, so a `MultiHeadAttention(128, 4)` fed from a `Linear(64, 64)`
   * inside `sequential(...)` must stay a width mismatch. The declaration is
   * also load-bearing rather than belt-and-braces here — `forward`'s
   * parameter is `Tensor<[B, T, D]>`, a rank-3 tuple the structural probe
   * only matches when the chain's running shape happens to be rank 3 with
   * the right width, and falls through to `S` (a silent identity) when it
   * does not.
   *
   * The one thing the protocol's three arms cannot say is "rank 3 only":
   * `mapLast` promises nothing about the prefix. The runtime twin is the
   * rank check in {@link forward}, which names the shape it got.
   */
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
    // The runtime twin of `DimDivCheck` (the shape-type laws' dual
    // type/value rule): the type says "H divides D" only when both are
    // literals, and a `MultiHeadAttention<number, number>` built from two
    // runtime widths is deliberately let through by law 1 — so the value
    // side has to say it too, naming both widths.
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
    // One `[D, 3D]` GEMM by default: three `[D, D]` matmuls over the same
    // activation are three kernel launches and three passes over `x` for
    // arithmetic one GEMM already does.
    this.qkv = fused ? new Linear(d, dimMul(3, d), { bias }) : null
    this.wq = fused ? null : new Linear(d, d, { bias })
    this.wk = fused ? null : new Linear(d, d, { bias })
    this.wv = fused ? null : new Linear(d, d, { bias })
    this.proj = new Linear(d, d, { bias })
  }

  /**
   * `[B, T, D] -> [B, T, D]`, generic in the batch and sequence axes: one
   * traced program serves every `(B, T)` the caller has.
   *
   * The head split and merge go through `unflatten`/`permute`/`flatten`
   * and never `view` (D21): `view` needs the element count to reduce to a
   * literal, and `B`/`T` are generics here, so `[B, T, D] -> [B*T, D]` is
   * simply unavailable to it. `unflatten`/`flatten` are pure tuple
   * surgery and reduce either way.
   */
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
      // ASSERT 1 of 2. `narrow(2, offset, this.d)` cuts a `[B, T, 3D]`
      // down to a runtime width that happens to be `D`; the algebra only
      // knows the cut length as "some number", because `ResizeDim` is
      // told the length and not that the length IS `D`. The three cuts
      // share one assertion site rather than growing three: the runtime
      // twin is `narrow`'s own bounds check plus the `x.shape[2]` check
      // above, which together prove `3D` really was three `D`s.
      const cut = (offset: number): Tensor<[B, T, D]> => assertChecked<[B, T, D]>(fused.narrow(2, offset, this.d))
      q = cut(0)
      k = cut(this.d)
      v = cut(2 * this.d)
    } else {
      q = this.wq!.forward(x)
      k = this.wk!.forward(x)
      v = this.wv!.forward(x)
    }
    // `k` is permuted straight to `[B, H, Dh, T]` — the transposed form
    // `sdpa` takes — so the score matmul costs no extra permute node.
    const q4 = q.unflatten(2, [h, headDim]).permute(0, 2, 1, 3)
    const k4 = k.unflatten(2, [h, headDim]).permute(0, 2, 3, 1)
    const v4 = v.unflatten(2, [h, headDim]).permute(0, 2, 1, 3)
    const ctx = sdpa(q4, k4, v4, { causal: this.causal, dropout: this.training ? this.p : 0 })
    // ASSERT 2 of 2. Merging the heads back gives `DimMul<H, DimDiv<D, H>>`,
    // which is `D` for every literal pair but does NOT reduce to `D` while
    // `D` and `H` are generics — hotscript cannot cancel a multiplication
    // against a division it never performed. The runtime twin is
    // `flatten`'s own product check on the two axes it folds.
    const merged = assertChecked<[B, T, D]>(ctx.permute(0, 2, 1, 3).flatten(2, 3))
    return this.proj.forward(merged)
  }
}
