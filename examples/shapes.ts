"use tsover"

import { cat, DimAdd, DimMul, Linear, Module, randn, ReLU, sequential, Tensor } from "../index.ts"

const mlp = sequential(
  new Linear(784, 256),
  new ReLU(),
  new Linear(256, 10),
)

const logits: Tensor<[64, 10]> = mlp.forward(randn([64, 784]))

function classify<B extends number>(x: Tensor<[B, 784]>): Tensor<[B, 10]> {
  return mlp.forward(x)
}

const batch32: Tensor<[32, 10]> = classify(randn([32, 784]))

class FeedForward<D extends number> extends Module {
  readonly up: Linear<D, DimMul<4, D>>
  readonly act = new ReLU()
  readonly down: Linear<DimMul<4, D>, D>

  constructor(d: D) {
    super()
    this.up = new Linear(d, DimMul(4, d))
    this.down = new Linear(DimMul(4, d), d)
  }

  forward<B extends number, T extends number>(x: Tensor<[B, T, D]>): Tensor<[B, T, D]> {
    return x + this.down.forward(this.act.forward(this.up.forward(x)))
  }
}

const ff = new FeedForward(16)
const mixed: Tensor<[2, 5, 16]> = ff.forward(randn([2, 5, 16]))
const hidden: Tensor<[16, 64]> = ff.up.weight

function concatFeatures<B extends number, L extends number, R extends number>(
  left: Tensor<[B, L]>,
  right: Tensor<[B, R]>,
): Tensor<[B, DimAdd<L, R>]> {
  return cat(left, right, 1)
}

const joined: Tensor<[8, 20]> = concatFeatures(randn([8, 12]), randn([8, 8]))

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

/** Seven rejections, never run: `pnpm typecheck` is what executes them. */
export function compileTimeErrors(): void {
  // matmul: inner dimensions do not match ([2, 3] @ [2, 3])
  // @ts-expect-error
  randn([2, 3]).matmul(randn([2, 3]))

  // '+' over [2, 3] and [4]
  // @ts-expect-error
  randn([2, 3]) + randn([4])

  // Cannot view tensor of shape [2, 3] as [7, 2] (6 vs 14 elements)
  // @ts-expect-error
  randn([2, 3]).view([7, 2])

  // permute([0, 0, 1]) repeats a dimension
  // @ts-expect-error
  randn([2, 3, 4]).permute(0, 0, 1)

  // cat: shapes [2, 3] and [2, 4] differ outside dim 0
  // @ts-expect-error
  cat(randn([2, 3]), randn([2, 4]), 0)

  // sequential: layer expects 16 input features but the previous layer outputs 8
  // @ts-expect-error
  sequential(new Linear(2, 8), new ReLU(), new Linear(16, 3))

  // sequential: input shape does not fit the layer chain
  // @ts-expect-error
  sequential(new Linear(2, 8), new Linear(8, 3)).forward(randn([4, 5]))
}

class Broken<D extends number> extends Module {
  readonly up: Linear<D, DimMul<4, D>>

  constructor(d: D) {
    super()
    this.up = new Linear(d, DimMul(4, d))
  }

  forward<B extends number, T extends number>(x: Tensor<[B, T, D]>): Tensor<[B, T, D]> {
    // [B, T, DimMul<4, D>] is not [B, T, D]
    // @ts-expect-error
    return this.up.forward(x)
  }
}

const shown: [string, readonly number[], string][] = [
  ["mlp.forward([64, 784])", logits.shape, "Tensor<[64, 10]>"],
  ["classify([32, 784])", batch32.shape, "Tensor<[B, 10]>, B = 32"],
  ["FeedForward(16).forward([2, 5, 16])", mixed.shape, "Tensor<[B, T, D]>"],
  ["FeedForward(16).up.weight", hidden.shape, "Tensor<[D, DimMul<4, D>]>"],
  ["cat([8, 12], [8, 8], 1)", joined.shape, "Tensor<[B, DimAdd<L, R>]>"],
  ["splitHeads([2, 5, 32], 4, 8)", heads.shape, "Tensor<[B, T, H, Dh]>"],
  ["heads.flatten(2, 3)", merged.shape, "Tensor<[B, T, DimMul<H, Dh>]>"],
]

console.log("shape gallery, inferred type vs the shape at run time\n")
for (const [expr, shape, inferred] of shown) {
  console.log(`  ${expr.padEnd(36)} ${`[${shape.join(", ")}]`.padEnd(16)} ${inferred}`)
}
console.log(
  `\n  ${new Broken(4).up.outFeatures} = DimMul(4, 4) at run time, the same arithmetic the type did`,
)

try {
  // the first case above, reaching the kernel that rejects it the same way
  // @ts-expect-error
  randn([2, 3]).matmul(randn([2, 3]))
} catch (e) {
  console.log(`\n  the same check, at run time: ${(e as Error).message}`)
}
