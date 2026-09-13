import type { DimCheck, Shape } from "../../shape.ts"
import { type AnyTensor, Tensor } from "../../tensor.ts"
import { Module } from "../module.ts"

class Activation extends Module {
  // Named `fn`, not `apply` — `Module.apply()` (W1.7's Module v2) claims
  // that name, and a private field of the same name as an inherited
  // public method is a TS2415 error.
  constructor(
    private readonly fn: (x: AnyTensor) => AnyTensor,
  ) {
    super()
  }

  forward<S extends Shape>(
    x: Tensor<S>,
  ): Tensor<S> {
    return this.fn(x) as Tensor<S>
  }
}

export class ReLU extends Activation {
  constructor() {
    super(x => x.relu())
  }
}

export class LeakyReLU extends Activation {
  constructor(negativeSlope = 0.01) {
    super(x => x.leakyRelu(negativeSlope))
  }
}

export class Tanh extends Activation {
  constructor() {
    super(x => x.tanh())
  }
}

export class Sigmoid extends Activation {
  constructor() {
    super(x => x.sigmoid())
  }
}

/**
 * Generic in the dim so `DimCheck` sees a literal at the call, not the
 * wide `number` (`IsValidDim<S, number>` is always true).
 */
export class Softmax<const D extends number = -1> extends Module {
  constructor(readonly dim: D = -1 as D) {
    super()
  }

  forward<S extends Shape>(
    x: Tensor<S> & DimCheck<S, D>,
  ): Tensor<S> {
    return (x as AnyTensor).softmax(this.dim) as Tensor<S>
  }
}
