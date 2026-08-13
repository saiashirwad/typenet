import { Linear, Module, type Tensor, tensor } from "../index.ts"

/**
 * The shared 2 -> 8 -> 1 XOR MLP used by the examples (tanh hidden,
 * sigmoid output).
 */
export class XorNet extends Module {
  hidden = new Linear(2, 8)
  out = new Linear(8, 1)

  forward<B extends number>(
    x: Tensor<[B, 2]>,
  ): Tensor<[B, 1]> {
    const h = this.hidden.forward(x).tanh()
    return this.out.forward(h).sigmoid()
  }
}

export const XOR_X = tensor([
  [0, 0],
  [0, 1],
  [1, 0],
  [1, 1],
])

export const XOR_Y = tensor([[0], [1], [1], [0]])
