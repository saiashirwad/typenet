# typenet

A tensor library for TypeScript where a shape is part of a tensor's type. `Tensor<[2, 3]>` and `Tensor<[3, 4]>` multiply to `Tensor<[2, 4]>`; a matmul whose inner dims disagree is a compile error rather than a runtime one. It has eager and lazy evaluation, reverse-mode autograd, a typed `nn` module system, and an optional Rust/candle backend for CPU or Metal. Operators (`+ - * / **`) are shape-checked too, through [tsover](https://tsover.swmansion.com).

## The type system

A shape is a tuple of literal numbers in the tensor's type. Below, `+` broadcasts, `**` and `mean` reduce, and the last line is a compile error because the inner dimensions do not agree:

```ts
"use tsover"
import { randn, Tensor } from "typenet"

const a = randn([2, 3]) // Tensor<[2, 3]>
const w = randn([3, 4]) // Tensor<[3, 4]>

const h: Tensor<[2, 4]> = a.matmul(w) // inner 3s agree
const c: Tensor<[2, 3]> = randn([2, 1]) + randn([1, 3]) // broadcast
const loss: Tensor<[]> = ((h - 1) ** 2).mean()

// @ts-expect-error matmul: inner dimensions do not match ([2, 3] @ [2, 3])
a.matmul(randn([2, 3]))
```

A dimension nobody has picked yet is `number`, and it stays a wildcard through every layer:

```ts
import { Linear, randn, Tensor } from "typenet"

const layer = new Linear(784, 128)
const n: number = 8 // the batch, whatever it turns out to be
const batch: Tensor<[number, 784]> = randn([n, 784])
const out: Tensor<[number, 128]> = layer.forward(batch)
```

`DimAdd`, `DimMul`, `DimSub` and `DimDiv` are each a type and a value with the same name. The type computes on literal dims, the value on numbers. A derived width is written once and carries its own type; `cat` adds the width it concatenates:

```ts
"use tsover"
import { cat, DimMul } from "typenet"
import { Linear, Module, randn, type Tensor } from "typenet"

class FeedForward<D extends number> extends Module {
  readonly up: Linear<D, DimMul<4, D>>
  readonly down: Linear<DimMul<4, D>, D>

  constructor(d: D) {
    super()
    this.up = new Linear(d, DimMul(4, d)) // 4x, in type and value
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

Creation, math, reductions, views, layers, optimizers and a loss, with the shape of each result written beside it:

```ts
"use tsover"
import { arange, eye, ones, randn, tensor, zeros } from "typenet"
import { AdamW, clipGradNorm, crossEntropy, mseLoss } from "typenet"
import { Linear, ReLU, sequential, Tensor } from "typenet"

// creation: the argument is the shape, and the literal comes back as a type
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

// layers compose their shape effects, widths checked where written
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

## A GPT that typechecks

An MLP only chains two numbers. A transformer uses more of the type system: the head split, the position-embedding broadcast (`[B, T, D] + [T, D]`), the weight tie between a `[V, D]` token table and a `D -> V` output head, and a loss that reads its target shape off the logits. `examples/gpt.ts` is that model, with no cast anywhere in it:

```ts
"use tsover"
import { crossEntropy, type DimDivCheck, Embedding } from "typenet"
import { functional, type IndexTensor, LayerNorm, Module } from "typenet"
import { ModuleList, Tensor, TiedLinear, TransformerBlock } from "typenet"

/** V vocab, T context, D width, H heads. All literals, so forward is checked. */
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
    // The head SHARES the token table: one [V, D] parameter, updated once.
    this.head = TiedLinear.of(this.wte)
    this.pos = this.registerBuffer("pos", functional.arangeIndex(block))
  }

  forward<B extends number>(idx: IndexTensor<[B, T]>): Tensor<[B, T, V]> {
    // [B, T, D] + [T, D] broadcasts; each block is an endomorphism on [B, T, D].
    const { wte, wpe, pos } = this
    let h: Tensor<[B, T, D]> = wte.forward(idx) + wpe.forward(pos)
    for (const block of this.blocks) h = block.forward(h)
    return this.head.forward(this.lnf.forward(h))
  }
}

const model = new GPT({ vocab: 32, block: 32, dModel: 64, heads: 4, layers: 2 })
// token ids are an IndexTensor, one rank below the logits
const ids = Tensor.indices(Array(4 * 32).fill(0), [4, 32])
const labels = Tensor.indices(Array(4 * 32).fill(1), [4, 32])
const loss = crossEntropy(model.forward(ids), labels) // Tensor<[]>
loss.backward()
```

## Development

```sh
pnpm install
pnpm test              # vitest: runtime, operators, grad checks
pnpm typecheck         # tsover's tsc, examples and type tests included
pnpm typecheck:budget  # type-level cost against the checked-in baseline
pnpm check:readme      # compiles every ts block in this file
pnpm format            # dprint
pnpm build:native      # the Rust addon (needs a Rust toolchain)
```

`examples/` holds the runnable models: `pnpm example:shapes`, `example:mlp`, `example:gpt`, `example:xor`, `example:spiral`.
