# typenet

A type-safe tensor and neural-network library for TypeScript. Tensor **shapes live in the type system**: broadcasting, matmul, reshapes and reductions are all computed at the type level, so shape bugs are compile errors instead of runtime crashes. On top of that sits a real runtime — typed-array storage, NumPy-style broadcasting, reverse-mode autograd, layers and optimizers — plus **operator overloading** via [tsover](https://tsover.swmansion.com), so `((pred - target) ** 2).mean()` is real code.

```ts
"use tsover";
import { tensor, randn } from "typenet";

const a = randn([2, 3]); // Tensor<[2, 3]>
const w = randn([3, 4]); // Tensor<[3, 4]>

const h = a.matmul(w); // Tensor<[2, 4]>
const s = h + randn([4]); // Tensor<[2, 4]>  (broadcast)
const l = ((s - 1) ** 2).mean(); // Tensor<[]>      (scalar)

a.matmul(randn([5, 4]));
// ^ compile error: matmul: inner dimensions do not match
s + randn([3]);
// ^ compile error: Operator '+' cannot be applied
```

## A complete neural network

Shapes are checked end to end — the batch dim is generic, the feature dims are literal:

```ts
"use tsover";
import { tensor, Tensor, Linear, Module, SGD } from "typenet";
import type { TensorParams } from "typenet";

class XorNet extends Module {
  hidden = new Linear(2, 8);
  out = new Linear(8, 1);

  forward<B extends number, P extends TensorParams>(x: Tensor<[B, 2], P>): Tensor<[B, 1], P> {
    const h = this.hidden.forward(x).tanh(); // Tensor<[B, 8]>
    return this.out.forward(h).sigmoid(); // Tensor<[B, 1]>
  }
}

const X = tensor([
  [0, 0],
  [0, 1],
  [1, 0],
  [1, 1],
]); // Tensor<[4, 2]>
const Y = tensor([[0], [1], [1], [0]]); // Tensor<[4, 1]>

const net = new XorNet();
const optim = new SGD(net.parameters(), { lr: 0.5, momentum: 0.9 });

for (let epoch = 0; epoch < 1500; epoch++) {
  const loss = ((net.forward(X) - Y) ** 2).mean();
  optim.zeroGrad();
  loss.backward();
  optim.step();
}
```

Run the full examples:

```sh
pnpm example:xor      # MLP learns XOR with MSE + SGD
pnpm example:spiral   # 3-class spiral, crossEntropy + Adam
```

## What the type system tracks

| Property      | Mechanism                                                                                                    |
| ------------- | ------------------------------------------------------------------------------------------------------------ |
| shape         | tuple of number literals, e.g. `Tensor<[32, 784]>`                                                           |
| dynamic dims  | `number` acts as a wildcard: `Tensor<[number, 784]>` type-checks against any batch size, enforced at runtime |
| dtype         | `"float32"` (default) / `"float64"`, switch with `.to("float64")`                                            |
| device        | `"cpu"` / `"gpu"` tag (V1 executes on CPU either way)                                                        |
| requires_grad | `.requires_grad()` flips the type-level flag and enables autograd                                            |

Shape algebra implemented in `src/shape.ts` (types only, zero runtime cost):

- `Broadcast<A, B>` — NumPy broadcasting rules
- `MatMul<A, B>` — full PyTorch matmul semantics (dot, mat-vec, vec-mat, batched with broadcast batch dims)
- `ResolveView<S, V>` — reshape with `-1` inference
- `Transpose` / `Permute` / `Squeeze` / `Unsqueeze` — with negative-index support
- `ReduceDim<S, D, KeepDim>` — `sum` / `mean` / `max` / `argmax`
- `Stack` / `Cat`

Errors are readable: `a.view([7, 2])` on 6 elements fails with
`Cannot view tensor of shape [2, 3] as [7, 2] (6 vs 14 elements)`.

## Operator overloading (tsover)

[tsover](https://tsover.swmansion.com) is a TypeScript fork by Software Mansion that adds operator overloading. This repo installs it as the `typescript` package (see `package.json`) and runs its code transform through the vite plugin (`vite.config.ts`), which powers both `vitest` and `vite-node`.

Files (or single functions) opt in with a directive:

```ts
"use tsover";
```

Inside such a scope, `+ - * / **` (and their `+=` forms) work on tensors with full shape checking. `**` requires a scalar exponent.

One subtlety worth knowing: tsover resolves operator overloads **without generic inference** — it just checks assignability against each declared signature. `Tensor` therefore enumerates the legal right-hand shapes for its resolved shape `S` as a union (`BroadcastsInto<S>`: every suffix of S with any subset of dims set to 1) and uses a `deferOperation` catch-all so `smaller + bigger` resolves through the bigger operand's overloads. Consequence: `a + b` supports operands where one shape broadcasts _into_ the other (covers scalars, rows, columns, biases, …). For a true cross-broadcast like `[2,1] + [1,3]`, use the method form `a.add(b)`, which is fully generic.

Editor setup: point your editor at the workspace TypeScript. In VS Code:

```json
{ "typescript.tsdk": "node_modules/typescript/lib" }
```

## Autograd

Reverse-mode, tape-based, PyTorch-shaped:

```ts
const x = tensor([1, 2]).requires_grad();
const y = tensor([3, 4]).requires_grad();
x.mul(y).add(x).pow(2).sum().backward();
x.grad; // Tensor<[2]> — populated
```

- `backward()` on a scalar (or pass an explicit gradient), grads accumulate on leaves until `zeroGrad()`
- gradients through broadcasting are reduced correctly (`sumTo`)
- differentiable: arithmetic, `pow`, `exp/log/sqrt/abs`, `relu/sigmoid/tanh/softmax/logSoftmax`, `matmul` (all variants), `sum/mean` (per-dim too), `view/squeeze/unsqueeze/transpose/permute`, `cat/stack`
- `noGrad(fn)` disables taping, `.detach()` cuts the graph
- every gradient is verified against central finite differences in `test/autograd.test.ts`

## API sketch

```ts
// creation
tensor([[1, 2], [3, 4]])       // shape inferred: Tensor<[2, 2]>
zeros([2, 3]) ones([4]) full([2], 7) rand([3]) randn([3])
eye(3) arange(10) scalar(42)

// math (all differentiable, all shape-checked)
a.add(b) a.sub(b) a.mul(b) a.div(b) a.pow(2)
a.neg() a.exp() a.log() a.sqrt() a.abs()
a.relu() a.sigmoid() a.tanh() a.softmax(1) a.logSoftmax(1)
a.matmul(b) a.dot(b)

// reductions
a.sum() a.sum(1) a.sum(-1, true) a.mean() a.max() a.argmax(1)

// shape
a.view([3, -1]) a.reshape([...]) a.squeeze() a.squeezeDim(0)
a.unsqueeze(-1) a.transpose(0, 2) a.permute(2, 0, 1) a.T
Tensor.stack([a, b], 0) Tensor.cat(a, b, 1)

// nn / optim
new Linear(784, 128) — weights Tensor<[784, 128]>, Kaiming-ish init
Module.parameters() mseLoss(pred, target) crossEntropy(logits, targets)
new SGD(params, { lr, momentum?, weightDecay? })
new Adam(params, { lr?, betas?, eps?, weightDecay? })

// data out
a.item() a.get(1, 2) a.toArray()  // NestedArray<S> — typed nesting depth!
```

## Development

```sh
pnpm install
pnpm test        # vitest (runtime + operator tests, numerical grad checks)
pnpm typecheck   # tsover's tsc; also validates test/types.test-d.ts
```

## V1 limitations

- CPU only; `.gpu()` is a type-level tag.
- `max()` is not differentiable (exists for stable softmax + metrics).
- Operators need the shapes above; cross-broadcasts use `.add()` et al.
- No slicing/fancy indexing yet — `get`, `narrow`-via-`cat`/`stack` patterns only.
