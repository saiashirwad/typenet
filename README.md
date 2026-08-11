# typenet

Type-safe tensor arithmetic for TypeScript. Shapes are tracked in the type system — broadcasting, matmul, reshapes and reductions are all checked at compile time.

Includes autograd, layers, and optimizers.

Operators (`+ - * / **`) work on tensors via [tsover](https://tsover.swmansion.com), a TypeScript fork with operator overloading.

```ts
"use tsover"
import { randn } from "typenet"

const a = randn([2, 3]) // Tensor<[2, 3]>
const w = randn([3, 4]) // Tensor<[3, 4]>

const h = a.matmul(w) // Tensor<[2, 4]>
const s = h + randn([4]) // Tensor<[2, 4]>, broadcast
const l = ((s - 1) ** 2).mean() // Tensor<[]>

const m = randn([2, 1]) + randn([1, 3]) // Tensor<[2, 3]>

a.matmul(randn([5, 4]))
// compile error: matmul: inner dimensions do not match
```

## A complete network

Feature dimensions are literal, the batch dimension stays generic:

```ts
"use tsover"
import {
  tensor,
  Tensor,
  Linear,
  Module,
  SGD
} from "typenet"
import type { TensorParams } from "typenet"

class XorNet extends Module {
  hidden = new Linear(2, 8)
  out = new Linear(8, 1)

  forward<B extends number, P extends TensorParams>(
    x: Tensor<[B, 2], P>
  ): Tensor<[B, 1], P> {
    const h = this.hidden.forward(x).tanh() // Tensor<[B, 8]>
    return this.out.forward(h).sigmoid() // Tensor<[B, 1]>
  }
}

const X = tensor([
  [0, 0],
  [0, 1],
  [1, 0],
  [1, 1]
]) // Tensor<[4, 2]>
const Y = tensor([[0], [1], [1], [0]]) // Tensor<[4, 1]>

const net = new XorNet()
const optim = new SGD(net.parameters(), {
  lr: 0.5,
  momentum: 0.9
})

for (let epoch = 0; epoch < 1500; epoch++) {
  const loss = ((net.forward(X) - Y) ** 2).mean()
  optim.zeroGrad()
  loss.backward()
  optim.step()
}
```

More in `examples/`:

```sh
pnpm example:xor      # MLP learns XOR, MSE + SGD
pnpm example:spiral   # 3-class spiral, crossEntropy + Adam
pnpm example:gat      # graph attention network
```

## What the type system tracks

| Property        | Mechanism                                                         |
| --------------- | ----------------------------------------------------------------- |
| shape           | tuple of literals: `Tensor<[32, 784]>`                            |
| dynamic dims    | `number` is a wildcard: `Tensor<[number, 784]>` takes any batch   |
| dtype           | `"float32"` (default) or `"float64"`, via `.to("float64")`        |
| `requires_grad` | `.requires_grad()` flips the type-level flag and enables autograd |

The shape algebra lives in `src/shape.ts` (types only): `Broadcast`, `MatMul` (dot, mat-vec, vec-mat, batched), `ResolveView` (reshape with `-1`), `Transpose`/`Permute`/`Squeeze`/`Unsqueeze`, `ReduceDim`, `Stack`, `Cat`. Errors say what went wrong: `Cannot view tensor of shape [2, 3] as [7, 2] (6 vs 14 elements)`.

## Operator overloading

[tsover](https://tsover.swmansion.com) is a TypeScript fork with operator overloading. It's installed here as the `typescript` package and applied via the vite plugin, covering `vitest` and `vite-node`. Opt in with a `"use tsover"` directive; inside that scope `+ - * / **` work on tensors with full shape inference, including cross-broadcasts like `[2, 1] + [1, 3] -> [2, 3]`.

For editor support, point your editor at the workspace TypeScript — in VS Code:

```json
{ "typescript.tsdk": "node_modules/typescript/lib" }
```

## Autograd

Reverse-mode, tape-based:

```ts
const x = tensor([1, 2]).requires_grad()
const y = tensor([3, 4]).requires_grad()
x.mul(y).add(x).pow(2).sum().backward()
x.grad // Tensor<[2]>
```

Gradients flow through arithmetic, `pow`/`exp`/`log`/`sqrt`/`abs`, activations, `matmul`, reductions, shape ops, and gather/scatter; broadcasts are reduced correctly. `noGrad(fn)` disables taping, `.detach()` cuts the graph. Every backward rule is checked against central finite differences in `test/gradcheck.test.ts`, in both eager and lazy modes.

## Compiled training steps

`compile(fn)` traces a function once and replays the graph on every call. A whole training step fits inside one — forward, backward, gradient clipping and the optimizer update all evaluated in a single pass, with nothing read back to JavaScript in between:

```ts
const step = compile(
  (x: Tensor<[B, 2]>, y: Tensor<[B, 1]>) => {
    const loss = ((net.forward(x) - y) ** 2).mean()
    optim.zeroGrad()
    loss.backward()
    clipGradNorm(net.parameters(), 1)
    optim.step()
    return loss
  }
)
for (let i = 0; i < 1000; i++) step(X, Y)
```

The graph can be deep: a cellular automaton rolled out over dozens of time steps and differentiated end to end is tens of thousands of nodes, which is fine. Two limits follow from tracing once: JavaScript control flow that depends on tensor _values_ cannot be captured (shape-dependent control flow is fine, shapes are known at trace time), and the graph has a fixed depth, so a variable-length loop needs one compiled graph per length.

## API sketch

```ts
// creation
tensor([[1, 2], [3, 4]]) // shape inferred: Tensor<[2, 2]>
zeros([2, 3]); ones([4]); full([2], 7); rand([3]); randn([3])
eye(3); arange(10); scalar(42)

// math — differentiable and shape-checked
a.add(b); a.sub(b); a.mul(b); a.div(b); a.pow(2)
a.neg(); a.exp(); a.log(); a.sqrt(); a.abs()
a.relu(); a.sigmoid(); a.tanh(); a.softmax(1); a.logSoftmax(1)
a.matmul(b); a.dot(b)
a.maximum(b); a.minimum(b); a.clamp(-1, 1)
a.gt(0); a.ge(0); a.lt(0); a.le(0); a.eq(0) // 1/0 masks, no gradient

// reductions
a.sum(); a.sum(1); a.sum(-1, true); a.mean(); a.max(); a.argmax(1)

// shape
a.view([3, -1]); a.reshape([2, 3]); a.squeeze(); a.unsqueeze(-1)
a.transpose(0, 2); a.permute(2, 0, 1); a.T; a.narrow(1, 0, 4)
Tensor.stack([a, b], 0); Tensor.cat(a, b, 1)

// gather and scatter, for message passing on a graph
x.indexSelect(src)              // each edge's source state
messages.scatterAdd(dst, nodes) // each node's incoming messages

// random values redrawn on every evaluation, unlike rand/randn
uniform([n, 1]); normal([n, c]); configure({ seed: 0 })

// nn / optim
new Linear(784, 128) // weights Tensor<[784, 128]>
net.parameters(); mseLoss(pred, target); crossEntropy(logits, targets)
new SGD(params, { lr, momentum?, weightDecay? })
new Adam(params, { lr?, betas?, eps?, weightDecay? })
clipGradNorm(params, 1) // between backward() and step()

// data out
a.item(); a.get(1, 2); a.toArray() // NestedArray<S>, typed nesting depth
```

## Graphs and message passing

`indexSelect` and `scatterAdd` are each other's gradient, and between
them they express message passing. With an edge list as the index,
gathering is "read each edge's source node" and scattering is "sum each
node's incoming messages":

```ts
const messages = x
  .indexSelect(src)
  .sub(x.indexSelect(dst))
  .tanh()
const aggregated = messages
  .scatterAdd(dst, nodes)
  .mul(invDegree)
```

Index tensors hold integral values in `float32` — there is no integer
dtype, and an f32 mantissa addresses 16.7M rows exactly.

`examples/gnca` is a full application of this: a graph cellular automaton
that grows a pattern from one seed node and heals after damage, ported
from [graph-cellular-automata](https://github.com/saiashirwad/graph-cellular-automata)
and checked against the PyTorch original in `test/gnca.test.ts` — same
graph, same rollout, same gradients to 3e-5.

```sh
pnpm gnca --steps 200          # train
pnpm vite-node examples/gnca/bench.ts
```

## Native backend

Eager mode runs typed-array kernels in JavaScript. Lazy and compiled
graphs can instead go to a Rust addon built on [candle](https://github.com/huggingface/candle):

```sh
pnpm build:native            # needs a Rust toolchain
```

```ts
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

Graphs small enough that a kernel launch would cost more than the
arithmetic (≤ 65536 elements) skip candle altogether and run on a fused
loop evaluator: chains of elementwise ops collapse into single passes, so
their intermediate values never reach memory.

`TYPENET_EVALUATOR=loops|cpu|gpu` overrides the choice and
`TYPENET_PROFILE=1` reports wall time and throughput per op kind, for
measuring one against another.

## Development

```sh
pnpm install
pnpm test        # vitest: runtime, operators, numerical grad checks
pnpm typecheck   # tsover's tsc, includes test/types.test-d.ts
```

## Status

Work in progress — the API and type system are still evolving.
