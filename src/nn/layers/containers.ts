"use tsover"

import type { DimEq, ErrorMessage, LastDimCheck, Shape } from "../../shape.ts"
import type { Tensor } from "../../tensor.ts"
import { Module } from "../module.ts"
import { SHAPE_EFFECT, type ShapeEffect } from "../sequential.ts"

/**
 * An ordered, homogeneous list of submodules (W5.2-A step 5) — the stack
 * of `TransformerBlock`s a GPT is mostly made of.
 *
 * It deliberately has NO `forward`. `Sequential` composes layers of
 * *different* types and derives the chain's shape from their tuple;
 * `ModuleList` holds `n` copies of ONE type and leaves the loop to the
 * caller, which is what a block stack with a KV cache, an early exit, or
 * an activation checkpoint boundary actually needs. Because it has no
 * `forward`, `sequential(...)` rejects it by name — `ApplyLayer`'s last
 * arm — rather than accepting it and silently doing nothing.
 *
 * Its only real job is discovery: `Module`'s reflection already recurses
 * through arrays, so `namedParameters()` reports `blocks.items.3.attn.proj.weight`
 * and `stateDict()` round-trips a whole stack with no per-block wiring.
 *
 * Typed as a single `M` rather than a heterogeneous tuple on purpose: a
 * stack of blocks that all map `[B, T, D] -> [B, T, D]` composes in a
 * plain `for` loop with the running tensor keeping its type, which is the
 * property the caller wants and a tuple would take away.
 */
export class ModuleList<M extends Module> extends Module {
  readonly items: readonly M[]

  constructor(items: Iterable<M>) {
    super()
    this.items = [...items]
  }

  /**
   * `n` modules built by `make(i)` — the shape a block stack is actually
   * written in (`ModuleList.of(6, () => new TransformerBlock(384, 6))`),
   * and the one place where "six blocks" does not mean "one block used
   * six times": each call makes its own parameters.
   */
  static of<M extends Module>(n: number, make: (i: number) => M): ModuleList<M> {
    if (!Number.isInteger(n) || n < 0) {
      throw new Error(`ModuleList.of: n must be a non-negative integer, got ${n}`)
    }
    return new ModuleList(Array.from({ length: n }, (_, i) => make(i)))
  }

  get length(): number {
    return this.items.length
  }

  /** Positive or negative index, bounds-checked — never `undefined`. */
  at(i: number): M {
    const idx = i < 0 ? this.items.length + i : i
    const m = this.items[idx]
    if (m === undefined) {
      throw new Error(
        `ModuleList: index ${i} is out of range for a list of ${this.items.length}`,
      )
    }
    return m
  }

  [Symbol.iterator](): IterableIterator<M> {
    return this.items[Symbol.iterator]()
  }
}

/**
 * The declared effect {@link Residual} inherits from its inner module.
 *
 * An undeclared inner module reports `"identity"`, which is the honest
 * answer for a wrapper whose whole contract is "shape in, same shape out":
 * the `+` below could not have typechecked otherwise.
 */
type ResidualEffect<M> = M extends { readonly [SHAPE_EFFECT]: infer E extends ShapeEffect } ? E : "identity"

/**
 * Can `Residual` wrap this module — i.e. is its shape effect an
 * endomorphism on `S`?
 *
 * Fail-open (law 1) in three directions: a fully generic `S` decides
 * nothing, an undeclared inner module decides nothing (the probe that
 * would read it lives in `sequential.ts` and is not exported — and a
 * third-party layer must keep composing exactly as it does today), and a
 * `mapLast` whose widths are generics decides nothing because `DimEq`
 * already treats a naked `number` as equal to everything.
 *
 * It errors on exactly two things it can prove: a `mapLast` that changes
 * the width (`Linear<4, 8>` — there is nothing to add `x` to), and an
 * `appendDim` (`Embedding` — the rank grows, so the residual stream and
 * the branch are not the same tensor at all).
 */
type ResidualCheck<M, S extends Shape> =
    number[] extends S ? unknown
  : M extends { readonly [SHAPE_EFFECT]: [effect: "mapLast", In: infer In extends number, Out: infer Out extends number] } ?
      DimEq<In, Out> extends false ? ErrorMessage<
        `Residual: the wrapped layer maps ${In} features to ${Out}, so its output cannot be added to its input`
      >
    : LastDimCheck<S, In>
  : M extends { readonly [SHAPE_EFFECT]: [effect: "appendDim", D: number] } ? ErrorMessage<
    `Residual: the wrapped layer adds an axis, so its output cannot be added to its input`
  >
  : unknown

/**
 * `x -> x + inner(x)` (W5.2-A step 5): the residual connection as a layer,
 * so a `sequential(...)` chain can carry one without the caller writing a
 * `forward` by hand.
 *
 * `Residual` reports the inner module's own {@link SHAPE_EFFECT}, not a
 * flat `"identity"` — so `sequential(new Linear(4, 8), new Residual(new LayerNorm(8)))`
 * still width-checks against the `8`, and swapping a bare `LayerNorm(8)`
 * for a `Residual(new LayerNorm(8))` changes nothing about how the chain
 * around it types. A wrapped layer that is NOT shape-preserving is caught
 * by {@link ResidualCheck} at the `forward` call.
 *
 * Note what this does not do: a pre-norm transformer block is
 * `x + attn(ln(x))`, i.e. the norm is INSIDE the branch, so it is
 * `new Residual(sequential(new LayerNorm(d), attn))` and not a `Residual`
 * around the attention alone. {@link TransformerBlock} writes that out
 * directly; this container is for the hand-rolled case.
 */
export class Residual<M extends Module> extends Module {
  declare readonly [SHAPE_EFFECT]: ResidualEffect<M>

  constructor(readonly inner: M) {
    super()
  }

  forward<S extends Shape>(
    x: Tensor<S> & ResidualCheck<M, S>,
  ): Tensor<S> {
    // `M extends Module`, and `Module` declares no `forward` — it cannot,
    // since every layer's is a different signature. So the call needs one
    // bridge, and this is it: a cast to the EXACT contract
    // {@link ResidualCheck} has just verified (`Tensor<S> -> Tensor<S>`),
    // not an `as any` that would erase `S` and let any shape through. The
    // runtime check below is its twin, for the fail-open cases the type
    // side deliberately lets past.
    const branch = (this.inner as unknown as {
      forward(t: Tensor<S>): Tensor<S>
    }).forward(x)
    // The runtime twin of `ResidualCheck`: the type side is fail-open for
    // an undeclared inner module, so the value side has to name the
    // mismatch when one shows up anyway. `+`'s own `Broadcast` would
    // otherwise quietly broadcast a `[B, 1]` branch over a `[B, D]` input.
    if (branch.shape.length !== x.shape.length || branch.shape.some((s, i) => s !== x.shape[i])) {
      throw new Error(
        `Residual: ${this.inner.constructor.name} maps [${x.shape.join(", ")}] to `
          + `[${branch.shape.join(", ")}], so its output cannot be added to its input`,
      )
    }
    return x + branch
  }
}
