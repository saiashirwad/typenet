"use tsover"

import type { DimEq, ErrorMessage, LastDimCheck, Shape } from "../../shape.ts"
import type { Tensor } from "../../tensor.ts"
import { Module } from "../module.ts"
import { SHAPE_EFFECT, type ShapeEffect } from "../sequential.ts"

/** No `forward`, so `sequential` rejects it and the caller drives the loop. */
export class ModuleList<M extends Module> extends Module {
  readonly items: readonly M[]

  constructor(items: Iterable<M>) {
    super()
    this.items = [...items]
  }

  static of<M extends Module>(n: number, make: (i: number) => M): ModuleList<M> {
    if (!Number.isInteger(n) || n < 0) {
      throw new Error(`ModuleList.of: n must be a non-negative integer, got ${n}`)
    }
    return new ModuleList(Array.from({ length: n }, (_, i) => make(i)))
  }

  get length(): number {
    return this.items.length
  }

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

/** The declared effect {@link Residual} inherits from its inner module; an undeclared one reports `"identity"`. */
type ResidualEffect<M> = M extends { readonly [SHAPE_EFFECT]: infer E extends ShapeEffect } ? E : "identity"

/** Errors when the wrapped module provably changes width or rank, and fails open otherwise. */
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

/** `x -> x + inner(x)`. Reports the inner module's {@link SHAPE_EFFECT}, so the surrounding chain still width-checks through it. */
export class Residual<M extends Module> extends Module {
  declare readonly [SHAPE_EFFECT]: ResidualEffect<M>

  constructor(readonly inner: M) {
    super()
  }

  forward<S extends Shape>(
    x: Tensor<S> & ResidualCheck<M, S>,
  ): Tensor<S> {
    // `Module` declares no `forward`, so cast to the contract ResidualCheck just verified.
    const branch = (this.inner as unknown as { forward(t: Tensor<S>): Tensor<S> }).forward(x)
    // Runtime twin of ResidualCheck: `+` would otherwise quietly broadcast a mismatched branch.
    if (branch.shape.length !== x.shape.length || branch.shape.some((s, i) => s !== x.shape[i])) {
      throw new Error(
        `Residual: ${this.inner.constructor.name} maps [${x.shape.join(", ")}] to `
          + `[${branch.shape.join(", ")}], so its output cannot be added to its input`,
      )
    }
    return x + branch
  }
}
