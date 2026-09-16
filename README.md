# typenet

A tensor library for TypeScript with compile-time shape checking. `Tensor<[2, 3]>` times `Tensor<[3, 4]>` is `Tensor<[2, 4]>`, and mismatched inner dimensions fail the build. It runs eagerly or lazily, differentiates in reverse mode, and has an optional Rust backend for CPU or Metal. Arithmetic operators are checked the same way, with [tsover](https://tsover.swmansion.com).

Pre-alpha. No releases yet, and the API changes between commits.

```ts
"use tsover"
import { randn } from "typenet"

const a = randn([2, 3]) // Tensor<[2, 3]>
const w = randn([3, 4]) // Tensor<[3, 4]>

const h = a.matmul(w) // Tensor<[2, 4]>
const c = randn([2, 1]) + randn([1, 3]) // broadcast -> Tensor<[2, 3]>
const loss = ((h - 1) ** 2).mean() // Tensor<[]>

// @ts-expect-error matmul: inner dimensions do not match ([2, 3] @ [2, 3])
a.matmul(randn([2, 3]))
```

```ts
import { Linear, randn, Tensor } from "typenet"

const layer = new Linear(784, 128)

// the batch stays generic through the layer, so this is written once
function forward<B extends number>(x: Tensor<[B, 784]>): Tensor<[B, 128]> {
  return layer.forward(x)
}

forward(randn([8, 784])) // Tensor<[8, 128]>
forward(randn([32, 784])) // Tensor<[32, 128]>
```

```ts
"use tsover"
import { cat, DimMul } from "typenet"
import { Linear, Module, randn, type Tensor } from "typenet"

class FeedForward<D extends number> extends Module {
  readonly up: Linear<D, DimMul<4, D>>
  readonly down: Linear<DimMul<4, D>, D>

  constructor(d: D) {
    super()
    this.up = new Linear(d, DimMul(4, d)) // the type and the value do the same arithmetic
    this.down = new Linear(DimMul(4, d), d)
  }

  forward<B extends number, T extends number>(
    x: Tensor<[B, T, D]>,
  ): Tensor<[B, T, D]> {
    return this.down.forward(this.up.forward(x).relu())
  }
}

const ff = new FeedForward(16)
const hidden = ff.up.weight // Tensor<[16, 64]>
const joined = cat(randn([8, 12]), randn([8, 8]), 1) // Tensor<[8, 20]>
```

## API

```ts
"use tsover"
import { arange, eye, ones, randn, tensor, zeros } from "typenet"
import { AdamW, clipGradNorm, crossEntropy, mseLoss } from "typenet"
import { Linear, ReLU, sequential, Tensor } from "typenet"

// creation
tensor([[1, 2], [3, 4]]) // Tensor<[2, 2]>
zeros([2, 3]) // Tensor<[2, 3]>
ones([4]) // Tensor<[4]>
arange(10) // Tensor<[10]>
eye(3) // Tensor<[3, 3]>

// math
const a = randn([2, 3])
const w = randn([3, 4])
a.matmul(w) // Tensor<[2, 4]>
a.add(randn([3])) // broadcast -> Tensor<[2, 3]>
a.relu().softmax(1) // Tensor<[2, 3]>
a.pow(2).mean() // Tensor<[]>

// reductions and views
a.sum() // Tensor<[]>
a.sum(1) // Tensor<[2]>
a.sum(-1, true) // Tensor<[2, 1]>
randn([2, 3, 4]).permute(2, 0, 1) // Tensor<[4, 2, 3]>
a.view([3, 2]).T // Tensor<[2, 3]>

// layers
const net = sequential(new Linear(4, 8), new ReLU(), new Linear(8, 10))
net.forward(randn([16, 4])) // Tensor<[16, 10]>

// training
const opt = new AdamW(net.parameters(), { lr: 3e-4, weightDecay: 0.01 })
const loss = mseLoss(net.forward(randn([16, 4])), randn([16, 10]))
opt.zeroGrad()
loss.backward()
clipGradNorm(net.parameters(), 1)
opt.step()

// class ids are one rank below the logits
crossEntropy(
  net.forward(randn([16, 4])),
  Tensor.indices(Array(16).fill(0), [16]),
)
```

## A character-level GPT

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
    this.head = TiedLinear.of(this.wte) // shares the [V, D] token table
    this.pos = this.registerBuffer("pos", functional.arangeIndex(block))
  }

  forward<B extends number>(idx: IndexTensor<[B, T]>): Tensor<[B, T, V]> {
    const { wte, wpe, pos } = this
    let h = wte.forward(idx) + wpe.forward(pos) // Tensor<[B, T, D]>
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
