"use tsover"

/**
 * The type showcase: everything in this file is checked by `tsc`, and the
 * eight negative cases below are checked *twice*: once by
 * `@ts-expect-error` (which fails the build if the error stops firing) and
 * once by `test/examples.test.ts`, which strips the directives, re-runs
 * `tsc`, and compares the message against the `// tsc(NNNN):` comment
 * quoted above each case.
 *
 * Running it (`pnpm example:shapes`) prints the runtime shape beside each
 * inferred type: the value twins (`DimAdd`, `DimMul`) compute at run time
 * exactly what their type twins compute at compile time.
 */

import { cat, DimAdd, DimMul, Linear, Module, randn, ReLU, sequential, Tensor } from "../index.ts"

// 1. Shape inference through `sequential`
// `sequential` is typed as the *tuple* of its layers, so `forward` composes
// their shape effects instead of collapsing to a bare `Tensor<number[]>`.
// The width chain (784 -> 256 -> 256 -> 10) is checked at construction; the
// batch dimension rides along untouched.

const mlp = sequential(
  new Linear(784, 256),
  new ReLU(),
  new Linear(256, 10),
)

// The annotation is the assertion: `Tensor` is invariant in its shape, so
// this only compiles if `forward` really inferred `[64, 10]`.
const logits: Tensor<[64, 10]> = mlp.forward(randn([64, 784]))

// The same chain at a batch size nobody has picked yet. `B` stays generic
// all the way through, which is what makes one model serve every batch.
function classify<B extends number>(x: Tensor<[B, 784]>): Tensor<[B, 10]> {
  return mlp.forward(x)
}

const batch32: Tensor<[32, 10]> = classify(randn([32, 784]))

// 2. A generic `forward<B, T>`, with `DimMul` carrying the widths
// `DimMul` is a type *and* a value: the type multiplies the literal dims,
// the function multiplies the numbers, and they are the same symbol, so
// `new Linear(d, DimMul(4, d))` is a `Linear<D, DimMul<4, D>>` with
// nothing written down twice and nothing forced.

class FeedForward<D extends number> extends Module {
  readonly up: Linear<D, DimMul<4, D>>
  readonly act = new ReLU()
  readonly down: Linear<DimMul<4, D>, D>

  constructor(d: D) {
    super()
    this.up = new Linear(d, DimMul(4, d))
    this.down = new Linear(DimMul(4, d), d)
  }

  // Rank 3 in, rank 3 out, for every batch `B` and every sequence length
  // `T`: `Linear` owns the last axis and lets any prefix ride along.
  forward<B extends number, T extends number>(x: Tensor<[B, T, D]>): Tensor<[B, T, D]> {
    return x + this.down.forward(this.act.forward(this.up.forward(x)))
  }
}

const ff = new FeedForward(16)
const mixed: Tensor<[2, 5, 16]> = ff.forward(randn([2, 5, 16]))
// The hidden width is `DimMul<4, 16>` = 64, computed by the type, not
// declared by the programmer.
const hidden: Tensor<[16, 64]> = ff.up.weight

// 3. `DimAdd` in a concatenation
// Concatenation adds the widths. The signature says so, so the caller's
// downstream `Linear` can be sized off the sum with no arithmetic of its own.

function concatFeatures<B extends number, L extends number, R extends number>(
  left: Tensor<[B, L]>,
  right: Tensor<[B, R]>,
): Tensor<[B, DimAdd<L, R>]> {
  return cat(left, right, 1)
}

const joined: Tensor<[8, 20]> = concatFeatures(randn([8, 12]), randn([8, 8]))

// 4. `DimMul` in a head split, and the `flatten`/`unflatten` round trip
// Splitting `[B, T, H*Dh]` into `[B, T, H, Dh]` is the move every attention
// implementation makes. Here the input width is *stated* as `DimMul<H, Dh>`,
// so the split is checked rather than trusted, and `flatten` puts it back.

function splitHeads<
  B extends number,
  T extends number,
  H extends number,
  Dh extends number,
>(
  x: Tensor<[B, T, DimMul<H, Dh>]>,
  heads: H,
  headDim: Dh,
): Tensor<[B, T, H, Dh]> {
  return x.unflatten(2, [heads, headDim])
}

const heads: Tensor<[2, 5, 4, 8]> = splitHeads(randn([2, 5, 32]), 4, 8)
const merged: Tensor<[2, 5, 32]> = heads.flatten(2, 3)

// The eight compile-time errors
// Each case quotes the message `tsc` prints for it. `test/examples.test.ts`
// re-derives every one of them from the compiler and fails if a single
// character has drifted.
//
// A note on the `​` you cannot see at the end of the quoted shape errors:
// every message type ends in a zero-width space (`ErrorMessage` in
// `src/shape.ts`), which is what makes a shape complaint render as a plain
// sentence instead of a wall of type machinery. The test strips it before
// comparing.
//
// Seven of them live inside a function that is never called, and the eighth
// in a `forward` that is never called: each is *also* a runtime error,
// raised with the same sentence by the same shape algebra, as the last
// lines of this file demonstrate.

function theEightErrors(): void {
  // (1) matmul checks the inner dimensions, and names both operands.
  //
  // tsc(2345): Argument of type 'Tensor<[2, 3]>' is not assignable to parameter of type
  // tsc(2345): 'Tensor<[2, 3]> & "matmul: inner dimensions do not match ([2, 3] @ [2, 3])"'.
  // @ts-expect-error
  randn([2, 3]).matmul(randn([2, 3]))

  // (2) The tsover operators are shape-checked too: `[2, 3]` and `[4]` have
  //     no common broadcast, so `+` does not apply.
  //
  // tsc(2365): Operator '+' cannot be applied to types 'Tensor<[2, 3]>' and 'Tensor<[4]>'.
  // @ts-expect-error
  randn([2, 3]) + randn([4])

  // (3) A reshape that does not preserve the element count, with the count
  //     computed for you.
  //
  // tsc(2345): Argument of type '[7, 2]' is not assignable to parameter of type
  // tsc(2345): '[7, 2] & "Cannot view tensor of shape [2, 3] as [7, 2] (6 vs 14 elements)"'.
  // @ts-expect-error
  randn([2, 3]).view([7, 2])

  // (4) A permutation that is not a permutation.
  //
  // tsc(2345): Argument of type '[0, 0, 1]' is not assignable to parameter of type
  // tsc(2345): '[0, 0, 1] & "permute([0, 0, 1]) repeats a dimension"'.
  // @ts-expect-error
  randn([2, 3, 4]).permute(0, 0, 1)

  // (5) Concatenation may differ on the concatenated axis and nowhere else.
  //
  // tsc(2345): Argument of type 'Tensor<[2, 4]>' is not assignable to parameter of type
  // tsc(2345): 'Tensor<[2, 4]> & "cat: shapes [2, 3] and [2, 4] differ outside dim 0"'.
  // @ts-expect-error
  cat(randn([2, 3]), randn([2, 4]), 0)

  // (6) A width mismatch between two layers of a chain, caught at
  //     construction, before any tensor exists, and naming both widths.
  //
  // tsc(2345): Argument of type '[Linear<2, 8>, ReLU, Linear<16, 3>]' is not assignable to parameter of type
  // tsc(2345): 'readonly [Linear<2, 8>, ReLU, Linear<16, 3>] & "sequential: layer expects 16 input features but the previous layer outputs 8"'.
  // @ts-expect-error
  sequential(new Linear(2, 8), new ReLU(), new Linear(16, 3))

  // (7) A chain that is internally consistent but does not accept this input.
  //
  // tsc(2345): Argument of type 'Tensor<[4, 5]>' is not assignable to parameter of type
  // tsc(2345): 'Tensor<[4, 5]> & "sequential: input shape does not fit the layer chain"'.
  // @ts-expect-error
  sequential(new Linear(2, 8), new Linear(8, 3)).forward(randn([4, 5]))
}

// (8) The derived width in a *generic* body: a block that widens by 4x and
//     forgets to project back is rejected inside its own definition, with
//     no instantiation needed.
//
// tsc(2322): Type 'Tensor<[B, T, DimMul<4, D>]>' is not assignable to type 'Tensor<[B, T, D]>'.
class Broken<D extends number> extends Module {
  readonly up: Linear<D, DimMul<4, D>>

  constructor(d: D) {
    super()
    this.up = new Linear(d, DimMul(4, d))
  }

  forward<B extends number, T extends number>(x: Tensor<[B, T, D]>): Tensor<[B, T, D]> {
    // @ts-expect-error
    return this.up.forward(x)
  }
}

// What it looks like at run time

const shown: [string, readonly number[], string][] = [
  ["mlp.forward([64, 784])", logits.shape, "Tensor<[64, 10]>"],
  ["classify([32, 784])", batch32.shape, "Tensor<[B, 10]>, B = 32"],
  ["FeedForward(16).forward([2, 5, 16])", mixed.shape, "Tensor<[B, T, D]>"],
  ["FeedForward(16).up.weight", hidden.shape, "Tensor<[D, DimMul<4, D>]>"],
  ["cat([8, 12], [8, 8], 1)", joined.shape, "Tensor<[B, DimAdd<L, R>]>"],
  ["splitHeads([2, 5, 32], 4, 8)", heads.shape, "Tensor<[B, T, H, Dh]>"],
  ["heads.flatten(2, 3)", merged.shape, "Tensor<[B, T, DimMul<H, Dh>]>"],
]

console.log("shape gallery — inferred type vs the shape at run time\n")
for (const [expr, shape, inferred] of shown) {
  console.log(`  ${expr.padEnd(36)} ${`[${shape.join(", ")}]`.padEnd(16)} ${inferred}`)
}
console.log(
  `\n  ${new Broken(4).up.outFeatures} = DimMul(4, 4) at run time — the same arithmetic the type did`,
)

// The eight errors are never run (`theEightErrors` is not called). This is
// the first of them, executed on purpose, to show that the compile-time
// message and the runtime message are one sentence written once.
void theEightErrors
try {
  const two = randn([2, 3])
  // @ts-expect-error the same error case (1) is rejected for at compile time
  two.matmul(randn([2, 3]))
} catch (e) {
  console.log(`\n  the same check, at run time: ${(e as Error).message}`)
}

console.log("\n  8 compile-time errors above; `pnpm typecheck` is what runs them")
