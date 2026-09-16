# typenet

A tensor library for TypeScript where a shape is part of a tensor's type. `Tensor<[2, 3]>` times `Tensor<[3, 4]>` gives `Tensor<[2, 4]>`; mismatched inner dimensions fail the build. Execution is eager or lazy, autograd is reverse-mode, and an optional Rust backend targets CPU or Metal. Operators are shape-checked through [tsover](https://tsover.swmansion.com).

Pre-alpha. There are no releases yet, and the API changes between commits without a deprecation window.

## Shapes are types

A tensor's type carries its shape as a tuple of literal numbers.

```ts
"use tsover"
import { randn, Tensor } from "typenet"

const a = randn([2, 3])
const w = randn([3, 4])

const h: Tensor<[2, 4]> = a.matmul(w)
const c: Tensor<[2, 3]> = randn([2, 1]) + randn([1, 3]) // broadcast
const loss: Tensor<[]> = ((h - 1) ** 2).mean()

// @ts-expect-error matmul: inner dimensions do not match ([2, 3] @ [2, 3])
a.matmul(randn([2, 3]))
```

A size that isn't fixed yet is `number`, and it stays a wildcard.

```ts
import { Linear, randn, Tensor } from "typenet"

const layer = new Linear(784, 128)
const batch: Tensor<[number, 784]> = randn([8, 784])
const out: Tensor<[number, 128]> = layer.forward(batch)
```

`DimAdd`, `DimMul`, `DimSub` and `DimDiv` are each a type and a function. The type computes on literal sizes, and the function computes on numbers.

```ts
"use tsover"
import { cat, DimMul } from "typenet"
import { Linear, Module, randn, type Tensor } from "typenet"

class FeedForward<D extends number> extends Module {
  readonly up: Linear<D, DimMul<4, D>>
  readonly down: Linear<DimMul<4, D>, D>

  constructor(d: D) {
    super()
    this.up = new Linear(d, DimMul(4, d))
    this.down = new Linear(DimMul(4, d), d)
  }

  forward<B extends number, T extends number>(
    x: Tensor<[B, T, D]>,
  ): Tensor<[B, T, D]> {
    return this.down.forward(this.up.forward(x).relu())
  }
}

const ff = new FeedForward(16)
const hidden: Tensor<[16, 64]> = ff.up.weight // DimMul<4, 16> = 64
const joined: Tensor<[8, 20]> = cat(randn([8, 12]), randn([8, 8]), 1)
```

## API

```ts
"use tsover"
import { arange, eye, ones, randn, tensor, zeros } from "typenet"
import { AdamW, clipGradNorm, crossEntropy, mseLoss } from "typenet"
import { Linear, ReLU, sequential, Tensor } from "typenet"

// the argument is the shape, and the literal comes back as a type
tensor([[1, 2], [3, 4]]) // Tensor<[2, 2]>
zeros([2, 3]) // Tensor<[2, 3]>
ones([4]) // Tensor<[4]>
arange(10) // Tensor<[10]>
eye(3) // Tensor<[3, 3]>

// math is differentiable and shape-checked
const a = randn([2, 3])
const w = randn([3, 4])
a.matmul(w) // Tensor<[2, 4]>
a.add(randn([3])) // broadcast -> Tensor<[2, 3]>
a.relu().softmax(1) // Tensor<[2, 3]>
a.pow(2).mean() // Tensor<[]>

// reductions and views carry their shape through
a.sum() // Tensor<[]>
a.sum(1) // Tensor<[2]>
a.sum(-1, true) // Tensor<[2, 1]>
randn([2, 3, 4]).permute(2, 0, 1) // Tensor<[4, 2, 3]>
a.view([3, 2]).T // Tensor<[2, 3]>

// widths are checked where they are written
const net = sequential(new Linear(4, 8), new ReLU(), new Linear(8, 10))
net.forward(randn([16, 4])) // Tensor<[16, 10]>

// a loss reads its target shape off the logits
const opt = new AdamW(net.parameters(), { lr: 3e-4, weightDecay: 0.01 })
const loss = mseLoss(net.forward(randn([16, 4])), randn([16, 10]))
opt.zeroGrad()
loss.backward()
clipGradNorm(net.parameters(), 1)
opt.step()

// crossEntropy takes class ids one rank below the logits
crossEntropy(
  net.forward(randn([16, 4])),
  Tensor.indices(Array(16).fill(0), [16]),
)
```

## A character-level GPT

`examples/gpt.ts` trains a small GPT on a passage encoded in the file. The output head is tied. `TiedLinear.of(this.wte)` reuses the `[V, D]` token table instead of allocating a second matrix, so `parameters()` reports it once.

```ts
"use tsover"
import { crossEntropy, type DimDivCheck, Embedding } from "typenet"
import { functional, type IndexTensor, LayerNorm, Module } from "typenet"
import { ModuleList, Tensor, TiedLinear, TransformerBlock } from "typenet"

/** V vocab, T context, D width, H heads. */
class GPT<
  V extends number,
  T extends number,
  D extends number,
  H extends number,
> extends Module {
  readonly wte: Embedding<V, D>
  readonly wpe: Embedding<T, D>
  readonly blocks: ModuleList<TransformerBlock<D, H>>
  readonly lnf: LayerNorm<D>
  readonly head: TiedLinear<D, V>
  readonly pos: IndexTensor<[T]>

  constructor(cfg: {
    vocab: V
    block: T
    dModel: D
    heads: H & DimDivCheck<D, H>
    layers: number
  }) {
    super()
    const { vocab, block, dModel: d, heads: h, layers } = cfg
    this.wte = new Embedding(vocab, d)
    this.wpe = new Embedding(block, d)
    this.blocks = ModuleList.of(
      layers,
      () => new TransformerBlock<D, H>(d, h, { causal: true }),
    )
    this.lnf = new LayerNorm(d)
    this.head = TiedLinear.of(this.wte)
    this.pos = this.registerBuffer("pos", functional.arangeIndex(block))
  }

  forward<B extends number>(idx: IndexTensor<[B, T]>): Tensor<[B, T, V]> {
    // the position table broadcasts over the batch, and each block returns [B, T, D]
    const { wte, wpe, pos } = this
    let h: Tensor<[B, T, D]> = wte.forward(idx) + wpe.forward(pos)
    for (const block of this.blocks) h = block.forward(h)
    return this.head.forward(this.lnf.forward(h))
  }
}

const model = new GPT({ vocab: 32, block: 32, dModel: 64, heads: 4, layers: 2 })
const ids = Tensor.indices(Array(4 * 32).fill(0), [4, 32])
const labels = Tensor.indices(Array(4 * 32).fill(1), [4, 32])
const loss = crossEntropy(model.forward(ids), labels) // Tensor<[]>
loss.backward()
```

## Development

```sh
pnpm install
pnpm test              # vitest
pnpm typecheck         # tsc over the project and the test files
pnpm typecheck:budget  # type-level cost against the checked-in baseline
pnpm check:readme      # compiles every ts block in this file
pnpm format            # dprint
pnpm build:native      # the Rust addon
```

`examples/` holds five runnable models: `pnpm example:shapes`, `example:mlp`, `example:gpt`, `example:xor`, `example:spiral`.
