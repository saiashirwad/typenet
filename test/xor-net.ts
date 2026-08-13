import { tensor } from "../src/factories.ts"
import { Tensor } from "../src/tensor.ts"

type AnyTensor = Tensor<any>

/**
 * A shared 2→4→1 MLP (tanh hidden, sigmoid output) training fixture for
 * the XOR truth table. `params` is `AnyTensor[]` (already requiresGrad'd)
 * so optimizers and `backward()` work against it directly.
 */
export function makeXorNet() {
  const x = tensor([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
  ])
  const y = tensor([[0], [1], [1], [0]])
  const w1 = tensor([
    [0.5, -0.5, 0.25, -0.25],
    [0.1, 0.2, -0.3, 0.4],
  ]).requiresGrad()
  const b1 = tensor([
    0.1,
    -0.1,
    0.05,
    -0.05,
  ]).requiresGrad()
  const w2 = tensor([
    [0.6],
    [-0.6],
    [0.3],
    [-0.3],
  ]).requiresGrad()
  const b2 = tensor([0.2]).requiresGrad()
  const params = [w1, b1, w2, b2] as AnyTensor[]
  const forward = () => {
    const h = x.matmul(w1).add(b1).tanh()
    return h.matmul(w2).add(b2).sigmoid()
  }
  const loss = () => forward().sub(y).pow(2).mean()
  return { x, y, params, forward, loss }
}
