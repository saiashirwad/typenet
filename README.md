# typenet

Type-safe tensor arithmetic for TypeScript. Shapes are tracked in the type system — broadcasting, matmul, reshapes and reductions are all checked at compile time.

Includes autograd, layers, and optimizers.

Operators (`+ - * / **`) work on tensors via [tsover](https://tsover.swmansion.com), a TypeScript fork with operator overloading.

## The type system

A tensor's shape is a tuple of literal numbers in its type, so every operation on it is an operation on that tuple. Nothing is inferred at run time that was not already decided at compile time:

```ts
"use tsover"
import { randn } from "typenet"

const a = randn([2, 3]) // Tensor<[2, 3]>
const w = randn([3, 4]) // Tensor<[3, 4]>

const h = a.matmul(w) // Tensor<[2, 4]>
const s = h + randn([4]) // Tensor<[2, 4]>, broadcast
const l = ((s - 1) ** 2).mean() // Tensor<[]>

const m = randn([2, 1]) + randn([1, 3]) // Tensor<[2, 3]>
```

`examples/shapes.ts` is a gallery of this — run it with `pnpm example:shapes`, or read it, which is the same thing since it is checked by `tsc` either way.

### Shapes compose through layers

`sequential` is typed as the _tuple_ of its layers, so `forward` composes their shape effects instead of collapsing to an untyped tensor. The width chain is checked at construction; the batch dimension rides along untouched:

```ts
"use tsover"
import { Linear, randn, ReLU, sequential, type Tensor } from "typenet"

const mlp = sequential(
  new Linear(784, 256),
  new ReLU(),
  new Linear(256, 10),
)

const logits: Tensor<[64, 10]> = mlp.forward(randn([64, 784]))

// The same chain at a batch size nobody has picked yet.
function classify<B extends number>(x: Tensor<[B, 784]>): Tensor<[B, 10]> {
  return mlp.forward(x)
}
```

### Dimension arithmetic is a type and a value

`DimAdd`, `DimMul`, `DimSub` and `DimDiv` are each a type _and_ a function of the same name. The type does the arithmetic on the literal dims, the function does it on the numbers, and they are the same symbol — so a derived width is written once and carries its own type:

```ts
"use tsover"
import { DimMul, Linear, Module, randn, ReLU, type Tensor } from "typenet"

class FeedForward<D extends number> extends Module {
  readonly up: Linear<D, DimMul<4, D>> // 4x the model width
  readonly act = new ReLU()
  readonly down: Linear<DimMul<4, D>, D>

  constructor(d: D) {
    super()
    this.up = new Linear(d, DimMul(4, d))
    this.down = new Linear(DimMul(4, d), d)
  }

  // Rank 3 in, rank 3 out, for every batch B and sequence length T.
  forward<B extends number, T extends number>(
    x: Tensor<[B, T, D]>,
  ): Tensor<[B, T, D]> {
    return x + this.down.forward(this.act.forward(this.up.forward(x)))
  }
}

const ff = new FeedForward(16)
const hidden: Tensor<[16, 64]> = ff.up.weight // DimMul<4, 16> = 64
const mixed: Tensor<[2, 5, 16]> = ff.forward(randn([2, 5, 16]))
```

The same pair drives concatenation (`DimAdd`) and the head split every attention implementation does (`DimMul`), both of them checked rather than trusted:

```ts
"use tsover"
import { cat, DimAdd, DimMul, randn, type Tensor } from "typenet"

function concatFeatures<B extends number, L extends number, R extends number>(
  left: Tensor<[B, L]>,
  right: Tensor<[B, R]>,
): Tensor<[B, DimAdd<L, R>]> {
  return cat(left, right, 1)
}

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

const joined: Tensor<[8, 20]> = concatFeatures(randn([8, 12]), randn([8, 8]))
const heads: Tensor<[2, 5, 4, 8]> = splitHeads(randn([2, 5, 32]), 4, 8)
const merged: Tensor<[2, 5, 32]> = heads.flatten(2, 3) // and back again
```

### The errors are sentences

A shape error is rendered as a sentence naming the shapes involved, not as a wall of type machinery. This is what `tsc` prints for `randn([2, 3]).matmul(randn([2, 3]))`:

```
error TS2345: Argument of type 'Tensor<[2, 3]>' is not assignable to parameter of type
'Tensor<[2, 3]> & "matmul: inner dimensions do not match ([2, 3] @ [2, 3])"'.
```

The eight cases in `examples/shapes.ts` are checked twice — by `@ts-expect-error`, which fails the build if one stops firing, and by `test/examples.test.ts`, which strips the directives, re-runs `tsc`, and compares what comes back with the message quoted above each case. So these are the messages, not a description of them:

| what you wrote                                                          | what `tsc` says                                                                      |
| ----------------------------------------------------------------------- | ------------------------------------------------------------------------------------ |
| `randn([2, 3]).matmul(randn([2, 3]))`                                   | `matmul: inner dimensions do not match ([2, 3] @ [2, 3])`                            |
| `randn([2, 3]) + randn([4])`                                            | `Operator '+' cannot be applied to types 'Tensor<[2, 3]>' and 'Tensor<[4]>'.`        |
| `randn([2, 3]).view([7, 2])`                                            | `Cannot view tensor of shape [2, 3] as [7, 2] (6 vs 14 elements)`                    |
| `randn([2, 3, 4]).permute(0, 0, 1)`                                     | `permute([0, 0, 1]) repeats a dimension`                                             |
| `cat(randn([2, 3]), randn([2, 4]), 0)`                                  | `cat: shapes [2, 3] and [2, 4] differ outside dim 0`                                 |
| `sequential(new Linear(2, 8), new ReLU(), new Linear(16, 3))`           | `sequential: layer expects 16 input features but the previous layer outputs 8`       |
| `sequential(new Linear(2, 8), new Linear(8, 3)).forward(randn([4, 5]))` | `sequential: input shape does not fit the layer chain`                               |
| a `forward<B, T>` that widens by 4x and forgets to project back         | `Type 'Tensor<[B, T, DimMul<4, D>]>' is not assignable to type 'Tensor<[B, T, D]>'.` |

The last one is the one that matters most: it is caught _inside_ a generic body, with no instantiation, so a layer is wrong where it is written rather than where it is used.

### What the type system tracks

| Property     | Mechanism                                                                   |
| ------------ | --------------------------------------------------------------------------- |
| shape        | tuple of literals: `Tensor<[32, 784]>`                                      |
| dynamic dims | `number` is a wildcard: `Tensor<[number, 784]>` takes any batch             |
| dtype        | `"float32"` (default), `"float64"`, `"int32"`, or `"int64"`, via `.to(...)` |
| gradients    | `.requiresGrad()` returns a new taped leaf over the same storage            |

The shape algebra lives in `src/shape.ts` (types only): `Broadcast`, `MatMul` (dot, mat-vec, vec-mat, batched), `ResolveView` (reshape with `-1`), `Transpose`/`Permute`/`Squeeze`/`Unsqueeze`, `ReduceDim`, `Stack`, `Cat`, and the `Dim*` arithmetic above.

Every check is **fail-open**: a shape that is still generic decides nothing and is allowed through, so a generic layer body compiles once rather than once per instantiation. A check fires only when the mismatch is provable.

## A complete network

Feature dimensions are literal, the batch dimension stays generic:

```ts
"use tsover"
import { Linear, Module, SGD, type Tensor, tensor } from "typenet"

class XorNet extends Module {
  hidden = new Linear(2, 8)
  out = new Linear(8, 1)

  forward<B extends number>(
    x: Tensor<[B, 2]>,
  ): Tensor<[B, 1]> {
    const h = this.hidden.forward(x).tanh() // Tensor<[B, 8]>
    return this.out.forward(h).sigmoid() // Tensor<[B, 1]>
  }
}

const X = tensor([
  [0, 0],
  [0, 1],
  [1, 0],
  [1, 1],
]) // Tensor<[4, 2]>
const Y = tensor([[0], [1], [1], [0]]) // Tensor<[4, 1]>

const net = new XorNet()
const optim = new SGD(net.parameters(), {
  lr: 0.5,
  momentum: 0.9,
})

for (let epoch = 0; epoch < 1500; epoch++) {
  const loss = ((net.forward(X) - Y) ** 2).mean()
  optim.zeroGrad()
  loss.backward()
  optim.step()
}
```

## Examples

```sh
pnpm example:shapes   # the type gallery above, plus eight compile-time errors
pnpm example:mlp      # 784 -> 256 -> 10 classifier, AdamW + warmup-cosine
pnpm example:xor      # MLP learns XOR, MSE + SGD
pnpm example:spiral   # 3-class spiral, crossEntropy + Adam
pnpm example:gat      # graph attention network
```

`pnpm example:mlp` trains a `sequential(Linear(784, 256), ReLU, Linear(256, 10))` on a synthetic 10-class dataset generated in the example itself (so it runs offline, and a fixed seed replays it exactly): 400 steps of batch 64, `AdamW` at a warmup-cosine learning rate with gradient clipping. It reaches a **training loss of 0.1735 and 87.9% test accuracy in 13-14 s** on an Apple M5, in eager mode. The loss and the accuracy are the same on every run — the seed fixes the data, the init and the shuffle; only the wall time moves.

It trains with the plain `zeroGrad` / `backward` / `step` loop rather than `compile()`, because a compiled step bakes the optimizer's `lr` into the traced graph as a constant, and the schedule has to move.

## Operator overloading

[tsover](https://tsover.swmansion.com) is a TypeScript fork with operator overloading. It's installed here as the `typescript` package and applied via the vite plugin, covering `vitest` and `vite-node`. Opt in with a `"use tsover"` directive; inside that scope `+ - * / **` work on tensors with full shape inference, including cross-broadcasts like `[2, 1] + [1, 3] -> [2, 3]`.

For editor support, point your editor at the workspace TypeScript — in VS Code:

```json
{ "typescript.tsdk": "node_modules/typescript/lib" }
```

## Autograd

Reverse-mode, tape-based:

```ts
"use tsover"
import { tensor } from "typenet"

const x = tensor([1, 2]).requiresGrad()
const y = tensor([3, 4]).requiresGrad()
x.mul(y).add(x).pow(2).sum().backward()
x.grad // Tensor<[2]>
```

Gradients flow through arithmetic, `pow`/`exp`/`log`/`sqrt`/`abs`, activations, `matmul`, reductions, shape ops, and gather/scatter; broadcasts are reduced correctly. `noGrad(fn)` disables taping, `.detach()` cuts the graph. Every backward rule is checked against central finite differences in `test/gradcheck.test.ts`, in both eager and lazy modes.

## Lazy mode

Eager mode, the default, runs every operation immediately. `lazy(fn)` runs `fn` with graph building turned on instead — operations return unevaluated nodes, forced only by `.data`, `.item()`, `.toArray()`, or `compile()`'s serializer — and puts the previous mode back when `fn` returns, even if it throws:

```ts
"use tsover"
import { lazy, tensor } from "typenet"

const out = lazy(() => tensor([1, 2, 3]).add(tensor([10, 20, 30])))
out.toArray() // [11, 22, 33] — forces the graph
```

`configure({ lazy: true })` sets the same flag globally, with no scope of its own — the right tool for a REPL, where there is no enclosing function to scope it to, and the wrong one anywhere else. A script that flips the flag, calls something twice, and flips it back has no `try`/`finally`:

```ts
"use tsover"
import { configure } from "typenet"

declare function run(): void

configure({ lazy: true })
run()
run() // if either call throws, lazy mode never gets turned back off
configure({ lazy: false })
```

`lazy(fn)` and its `eager(fn)` counterpart (for forcing eager mode inside an outer lazy scope) are `withContext({ lazy: true }, fn)` / `withContext({ lazy: false }, fn)` under the hood — reach for `configure` only at a REPL prompt.

## Compiled training steps

`compile(fn, exampleInputs)` traces `fn` against the examples up front and replays the graph on every call (omitting the examples still traces on the first call, deprecated). Reading a tensor's values inside `fn` (`.data`, `.item()`, ...) throws — the graph is recorded, not run. A whole training step fits inside one — forward, backward, gradient clipping and the optimizer update all evaluated in a single pass, with nothing read back to JavaScript in between:

```ts
"use tsover"
import { clipGradNorm, compile, Linear, randn, SGD, type Tensor } from "typenet"

const net = new Linear(2, 1)
const optim = new SGD(net.parameters(), { lr: 0.1 })
const X = randn([32, 2])
const Y = randn([32, 1])

const step = compile(
  (x: Tensor<[32, 2]>, y: Tensor<[32, 1]>) => {
    const loss = ((net.forward(x) - y) ** 2).mean()
    optim.zeroGrad()
    loss.backward()
    clipGradNorm(net.parameters(), 1)
    optim.step()
    return loss
  },
  [X, Y],
)
for (let i = 0; i < 1000; i++) step(X, Y)
```

The graph can be deep: a cellular automaton rolled out over dozens of time steps and differentiated end to end is tens of thousands of nodes, which is fine. Two limits follow from tracing once: JavaScript control flow that depends on tensor _values_ cannot be captured (shape-dependent control flow is fine, shapes are known at trace time), and the graph has a fixed depth, so a variable-length loop needs one compiled graph per length. A scalar read from JavaScript at trace time — an optimizer's `lr`, say — is a constant in the traced graph, so a learning-rate schedule belongs in an eager loop for now.

## API sketch

Creation:

```ts
"use tsover"
import {
  arange,
  eye,
  full,
  ones,
  rand,
  randn,
  scalar,
  tensor,
  zeros,
} from "typenet"

tensor([[1, 2], [3, 4]]) // shape inferred: Tensor<[2, 2]>
zeros([2, 3])
ones([4])
full([2], 7)
rand([3])
randn([3])
eye(3)
arange(10)
scalar(42)
```

Math — differentiable and shape-checked:

```ts
"use tsover"
import { randn } from "typenet"

const a = randn([2, 3])
const b = randn([2, 3])
const w = randn([3, 4])

a.add(b)
a.sub(b)
a.mul(b)
a.div(b)
a.pow(2)
a.neg()
a.exp()
a.log()
a.sqrt()
a.abs()
a.relu()
a.sigmoid()
a.tanh()
a.softmax(1)
a.logSoftmax(1)
a.matmul(w) // [2, 3] @ [3, 4] -> [2, 4]
randn([3]).dot(randn([3])) // Tensor<[]>
a.maximum(b)
a.minimum(b)
a.clamp(-1, 1)
a.gt(0)
a.ge(0)
a.lt(0)
a.le(0)
a.eq(0) // 1/0 masks, no gradient
```

Reductions and shape:

```ts
"use tsover"
import { randn, Tensor } from "typenet"

const a = randn([2, 3])
const t = randn([2, 3, 4])

a.sum()
a.sum(1)
a.sum(-1, true)
a.mean()
a.max()
a.argmax(1)

a.view([3, -1])
a.reshape([2, 3])
t.squeeze()
a.unsqueeze(-1)
t.transpose(0, 2)
t.permute(2, 0, 1)
a.T
t.narrow(1, 0, 2)
t.slice([2, [1, 3], null]) // number = end, [start, end], null = keep
a.broadcastTo([8, 2, 3]) // expand-only
Tensor.stack([a, a], 0)
Tensor.cat(a, a, 1)
```

Randomness, layers and optimizers:

```ts
"use tsover"
import {
  Adam,
  clipGradNorm,
  configure,
  crossEntropy,
  Linear,
  mseLoss,
  rand,
  randn,
  SGD,
  Tensor,
} from "typenet"

// rand/randn: { resample: "once" } (the default) fills a plain leaf
// immediately, fixed for the tensor's life; { resample: "perCall" } is
// a graph node redrawn on every evaluation. Both draw from the seeded
// generator: configure({ seed }) makes a run — including Linear's
// init — reproducible.
configure({ seed: 0 })
rand([8, 1])
randn([8, 3], { resample: "perCall" })

const net = new Linear(784, 128) // weights Tensor<[784, 128]>
const params = net.parameters()

const pred = net.forward(randn([16, 784]))
mseLoss(pred, randn([16, 128]))
// crossEntropy takes class ids as an IndexTensor, one rank below the
// logits: [16] ids against [16, 128] logits.
crossEntropy(
  net.forward(randn([16, 784])),
  Tensor.indices(Array(16).fill(0), [16]),
)

new SGD(params, { lr: 0.1, momentum: 0.9, weightDecay: 0 })
new Adam(params, { lr: 3e-4, betas: [0.9, 0.999], eps: 1e-8, weightDecay: 0 })
clipGradNorm(params, 1) // between backward() and step()

// data out
pred.item()
pred.get(1, 2)
pred.toArray() // NestedArray<S>, typed nesting depth
```

## Graphs and message passing

`indexSelect` and `scatterAdd` are each other's gradient, and between
them they express message passing. With an edge list as the index,
gathering is "read each edge's source node" and scattering is "sum each
node's incoming messages":

```ts
"use tsover"
import { fromFlat, ones, randn } from "typenet"

const nodes = 5
const x = randn([nodes, 3])
const src = fromFlat(new Int32Array([0, 1, 2, 3]), [4], "int32").toIndex()
const dst = fromFlat(new Int32Array([1, 2, 3, 4]), [4], "int32").toIndex()
const invDegree = ones([nodes, 1])

const messages = x
  .indexSelect(src)
  .sub(x.indexSelect(dst))
  .tanh()
const aggregated = messages
  .scatterAdd(dst, nodes)
  .mul(invDegree)
```

Index tensors hold integral values, and `indexSelect` / `scatterAdd`
demand that in the _type_: their index parameter is an `IndexTensor`, not
a plain `Tensor<[n]>`, so an ordinary tensor in that position is a
compile error rather than a runtime surprise. There are two ways to make
one — `Tensor.indices(data, shape)` builds an int32 leaf directly, and
`.toIndex()` brands an existing tensor; both check integrality once, at
the call. Prefer `int32` / `int64` storage, which is exact across the
full integer range; `float32` indices remain legal for compatibility, and
an f32 mantissa addresses 16.7M rows exactly.

## Native backend

Eager mode runs typed-array kernels in JavaScript. Lazy and compiled
graphs can instead go to a Rust addon built on [candle](https://github.com/huggingface/candle):

```sh
pnpm build:native            # needs a Rust toolchain
```

```ts
"use tsover"
import { useNative } from "typenet"

useNative() // candle on the CPU device
useNative({ device: "gpu" }) // the best accelerator available
```

CPU is the default, which is not the obvious choice. Measured on an Apple
M5, candle's CPU device (Accelerate for matmul) matches Metal on chained
large matmuls, loses to it by ~1.5x on purely elementwise graphs, and
beats it by ~7x on the gather/scatter graphs message passing produces —
Metal's `index_select`/`index_add` kernels are slow and such graphs are
made of many small dispatches. Reach for `"gpu"` when a workload is
dominated by large elementwise tensors.

The native path handles float32 compute with CPU-resident leaves.
`int32` / `int64` leaves are allowed as gather/scatter indices (read as
their native width, so they have no f32 mantissa limit). With native
enabled, a graph with a float64 leaf — or an integer leaf used as a
compute operand — throws instead of silently falling back to the JS
interpreter: keep the graph in float32 or call `disableNative()`.
Set `TYPENET_CHECK_SHAPES=1` to make the Rust side recompute every node
shape and assert it matches what JS serialized.

A graph that contains an op the addon has no kernel for runs on the JS
interpreter instead, and says so — once per op, with a line naming it.
`jsCounters().nativeFallbacks` counts them and `TYPENET_STRICT_NATIVE=1`
turns the notice into a throw, which is how a benchmark asserts it is
measuring the fast path rather than assuming it.

Graphs small enough that a kernel launch would cost more than the
arithmetic (≤ 65536 elements) skip candle altogether and run on a fused
loop evaluator: chains of elementwise ops collapse into single passes, so
their intermediate values never reach memory.

With `useNative()` active, eager mode is no longer "ignore native": a
large packed float32 matmul (> 65536 multiply-accumulates, unbatched)
goes to Accelerate's `sgemm` instead of the JS triple loop. BLAS
reassociates the accumulation, so results match JS to f32 rounding
rather than bit-for-bit; call `disableNative()` for exact replay.
Elementwise ops stay in JS and stay bit-identical.

`TYPENET_EVALUATOR=loops|cpu|gpu` overrides the choice and
`TYPENET_PROFILE=1` reports wall time and throughput per op kind, for
measuring one against another.

## Development

```sh
pnpm install
pnpm test          # vitest: runtime, operators, numerical grad checks
pnpm typecheck     # tsover's tsc, includes test/types.test-d.ts and examples/
pnpm check:readme  # every ts block in this file is compiled
```

Every TypeScript block in this README is extracted and compiled by
`scripts/check-readme.mjs`, with `typenet` resolved to this checkout — so
a block that has gone stale fails a script rather than a reader.

## Status

Work in progress — the API and type system are still evolving.
