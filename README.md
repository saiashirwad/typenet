# typenet

A tensor and neural-network library for TypeScript where shapes live in the type system. Broadcasting, matmul, reshapes and reductions are computed at the type level, so shape mismatches are compile errors. Underneath is a PyTorch-shaped runtime: typed-array storage, NumPy broadcasting, reverse-mode autograd, layers, optimizers, and an optional WebGPU backend. Operators work on tensors via [tsover](https://tsover.swmansion.com), a TypeScript fork with operator overloading.

```ts
"use tsover"
import { randn } from "typenet"

const a = randn([2, 3]) // Tensor<[2, 3]>
const w = randn([3, 4]) // Tensor<[3, 4]>

const h = a.matmul(w) // Tensor<[2, 4]>
const s = h + randn([4]) // Tensor<[2, 4]>, broadcast
const l = ((s - 1) ** 2).mean() // Tensor<[]>

a.matmul(randn([5, 4]))
// compile error: matmul: inner dimensions do not match
s + randn([3])
// compile error: Operator '+' cannot be applied
```

## A complete network

Shapes are checked end to end. The batch dimension is generic, the feature dimensions are literal:

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

Runnable versions live in `examples/`:

```sh
pnpm example:xor      # MLP learns XOR, MSE + SGD
pnpm example:spiral   # 3-class spiral, crossEntropy + Adam
pnpm example:gat      # graph attention network on a synthetic graph
pnpm example:dawn     # WebGPU compute in Node through Dawn
pnpm example:xor:dawn # XOR trained on GPU tensors in Node
```

## What the type system tracks

| Property        | Mechanism                                                              |
| --------------- | ---------------------------------------------------------------------- |
| shape           | tuple of number literals, e.g. `Tensor<[32, 784]>`                     |
| dynamic dims    | `number` is a wildcard: `Tensor<[number, 784]>` accepts any batch size |
| dtype           | `"float32"` (default) or `"float64"`, switch with `.to("float64")`     |
| device          | `"cpu"` typed arrays or `"gpu"` TypeGPU storage                        |
| `requires_grad` | `.requires_grad()` flips the type-level flag and enables autograd      |

The shape algebra is in `src/shape.ts` (types only, no runtime cost): `Broadcast` (NumPy rules), `MatMul` (full PyTorch semantics — dot, mat-vec, vec-mat, batched), `ResolveView` (reshape with `-1` inference), `Transpose` / `Permute` / `Squeeze` / `Unsqueeze` (with negative indices), `ReduceDim`, `Stack`, `Cat`. Errors state the problem: `a.view([7, 2])` on 6 elements fails with `Cannot view tensor of shape [2, 3] as [7, 2] (6 vs 14 elements)`.

## Operator overloading

[tsover](https://tsover.swmansion.com) is a TypeScript fork by Software Mansion that adds operator overloading. This repo installs it as the `typescript` package and applies its transform through the vite plugin (see `vite.config.ts`), which covers both `vitest` and `vite-node`. Files or single functions opt in with a `"use tsover"` directive; inside such a scope `+ - * / **` (and their `+=` forms) work on tensors with full shape checking.

One caveat: tsover resolves operator overloads without generic inference — it only checks assignability against declared signatures. `Tensor` therefore enumerates the legal right-hand shapes for its resolved shape as a union (every suffix with any subset of dims set to 1) and defers to the bigger operand's overloads. In practice `a + b` works whenever one shape broadcasts _into_ the other (scalars, rows, columns, biases). For a true cross-broadcast like `[2, 1] + [1, 3]`, use the method form `a.add(b)`, which is fully generic.

For editor support, point your editor at the workspace TypeScript. In VS Code:

```json
{ "typescript.tsdk": "node_modules/typescript/lib" }
```

## Autograd

Reverse-mode and tape-based:

```ts
const x = tensor([1, 2]).requires_grad()
const y = tensor([3, 4]).requires_grad()
x.mul(y).add(x).pow(2).sum().backward()
x.grad // Tensor<[2]>
```

- `backward()` on a scalar, or pass an explicit gradient; grads accumulate on leaves until `zeroGrad()`
- gradients through broadcasting are reduced correctly
- differentiable: arithmetic, `pow`, `exp`/`log`/`sqrt`/`abs`, `relu`/`sigmoid`/`tanh`/`softmax`/`logSoftmax`, `matmul` (all variants), `sum`/`mean` (per-dim too), `view`/`squeeze`/`unsqueeze`/`transpose`/`permute`, `cat`/`stack`
- `noGrad(fn)` disables taping, `.detach()` cuts the graph
- every gradient is checked against central finite differences in `test/autograd.test.ts`

## WebGPU

The TypeGPU backend is explicit: initialize once, move tensors and modules over synchronously, await only when data returns to JavaScript.

```ts
import { initTypeGPU, tensor, Linear, SGD } from "typenet"

const root = await initTypeGPU()
const x = tensor([
  [1, 2],
  [3, 4]
]).gpu()
const net = new Linear(2, 1).gpu()
const optim = new SGD(net.parameters(), { lr: 0.01 })

const loss = net.forward(x).pow(2).mean() // queues GPU work
loss.backward()
optim.step()

console.log(await loss.read()) // Float32Array
console.log((await loss.toCPU()).item()) // CPU tensor

optim.dispose()
root.destroy()
```

If your application already owns a TypeGPU root, pass it to `configureTypeGPU(root)` instead. GPU tensors expose `read()`, `toCPU()`, `write()` and `dispose()`; synchronous host access (`data`, `item()`, `get()`, `toArray()`, `cpu()`) throws, since WebGPU readback is asynchronous. Views share their GPU allocation, so disposing any alias invalidates all of them.

The backend covers broadcasting, unary math, reductions, matmul, permutation, cat/stack, autograd, `Linear`, cross-entropy, SGD and Adam. WebGPU has no float64 shader type, so `.to("float64").gpu()` is rejected rather than silently downcast.

The Dawn examples (`example:dawn`, `example:xor:dawn`) run the same backend headless in Node: the [`webgpu`](https://www.npmjs.com/package/webgpu) package provides a native Dawn device, which is handed to `tgpu.initFromDevice({ device })` and `configureTypeGPU(root)`. Dawn is a dev-only dependency of the examples; typenet itself never imports it.

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

The browser GPU parity harness is `test/gpu.html`: run `pnpm exec vite`, open `/test/gpu.html` in a WebGPU-capable browser, expect `PASS`. It covers kernels, autograd, modules, losses, optimizers and backend error paths.

## Limitations

- GPU kernels are correctness-first (one thread per output); tiled matmul, hierarchical reductions and pipeline caching are future work.
- GPU tensors are float32-only, and moving an already-recorded CPU autograd graph to the GPU creates a new GPU leaf.
- `max()` is not differentiable (it exists for stable softmax and metrics).
- Operators cover shapes that broadcast into one another; cross-broadcasts use the method forms.
- No slicing or fancy indexing yet.
