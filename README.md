# typenet

Type-safe tensor arithmetic for TypeScript. Shapes are tracked in the type system — broadcasting, matmul, reshapes and reductions are all checked at compile time.

Includes autograd, layers, optimizers, and an optional WebGPU backend.

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
import { tensor, Tensor, Linear, Module, SGD } from "typenet"
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

const X = tensor([[0, 0], [0, 1], [1, 0], [1, 1]]) // Tensor<[4, 2]>
const Y = tensor([[0], [1], [1], [0]]) // Tensor<[4, 1]>

const net = new XorNet()
const optim = new SGD(net.parameters(), { lr: 0.5, momentum: 0.9 })

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
pnpm example:dawn     # WebGPU compute in Node through Dawn
pnpm example:xor:dawn # XOR trained on GPU tensors in Node
```

## What the type system tracks

| Property        | Mechanism                                                          |
| --------------- | ------------------------------------------------------------------ |
| shape           | tuple of literals: `Tensor<[32, 784]>`                             |
| dynamic dims    | `number` is a wildcard: `Tensor<[number, 784]>` takes any batch    |
| dtype           | `"float32"` (default) or `"float64"`, via `.to("float64")`         |
| device          | `"cpu"` typed arrays or `"gpu"` TypeGPU storage                    |
| `requires_grad` | `.requires_grad()` flips the type-level flag and enables autograd  |

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

Gradients flow through arithmetic, `pow`/`exp`/`log`/`sqrt`/`abs`, activations, `matmul`, reductions, and shape ops; broadcasts are reduced correctly. `noGrad(fn)` disables taping, `.detach()` cuts the graph. Every gradient is checked against finite differences in `test/autograd.test.ts`.

## WebGPU

Initialize once, move tensors and modules over synchronously, await only when data comes back:

```ts
import { initTypeGPU, tensor, Linear, SGD } from "typenet"

const root = await initTypeGPU()
const x = tensor([[1, 2], [3, 4]]).gpu()
const net = new Linear(2, 1).gpu()
const optim = new SGD(net.parameters(), { lr: 0.01 })

const loss = net.forward(x).pow(2).mean() // queues GPU work
loss.backward()
optim.step()

console.log(await loss.read()) // Float32Array

optim.dispose()
root.destroy()
```

If you already own a TypeGPU root, pass it to `configureTypeGPU(root)`. GPU tensors expose `read()`, `toCPU()`, `write()`, `dispose()`; synchronous host access throws, since WebGPU readback is async. `float64` is rejected on GPU (WebGPU has no f64) rather than silently downcast.

The Dawn examples run the same backend headless in Node via the [`webgpu`](https://www.npmjs.com/package/webgpu) package. Dawn is a dev-only dependency of the examples.

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

// reductions
a.sum(); a.sum(1); a.sum(-1, true); a.mean(); a.max(); a.argmax(1)

// shape
a.view([3, -1]); a.reshape([2, 3]); a.squeeze(); a.unsqueeze(-1)
a.transpose(0, 2); a.permute(2, 0, 1); a.T
Tensor.stack([a, b], 0); Tensor.cat(a, b, 1)

// nn / optim
new Linear(784, 128) // weights Tensor<[784, 128]>
net.parameters(); mseLoss(pred, target); crossEntropy(logits, targets)
new SGD(params, { lr, momentum?, weightDecay? })
new Adam(params, { lr?, betas?, eps?, weightDecay? })

// data out
a.item(); a.get(1, 2); a.toArray() // NestedArray<S>, typed nesting depth
await gpuTensor.read(); await gpuTensor.toCPU()
```

## Development

```sh
pnpm install
pnpm test        # vitest: runtime, operators, numerical grad checks
pnpm typecheck   # tsover's tsc, includes test/types.test-d.ts
```

Browser GPU parity harness: `pnpm exec vite`, open `/test/gpu.html` in a WebGPU-capable browser, expect `PASS`.

## Status

Work in progress — the API and type system are still evolving.
