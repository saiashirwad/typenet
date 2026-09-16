import type { DimCheck, Shape } from "../../shape.ts"
import { type AnyTensor, gelu, silu, type Tensor } from "../../tensor.ts"
import { Module } from "../module.ts"
import { SHAPE_EFFECT } from "../sequential.ts"

class Activation extends Module {
  declare readonly [SHAPE_EFFECT]: "identity"

  // Named `fn`, not `apply`: a field clashing with the inherited Module.apply() is a TS2415 error.
  constructor(private readonly fn: (x: AnyTensor) => AnyTensor) {
    super()
  }

  forward<S extends Shape>(x: Tensor<S>): Tensor<S> {
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

/** GELU, tanh approximation, over the fused `gelu` node. */
export class GELU extends Activation {
  constructor() {
    super(x => gelu(x))
  }
}

export class SiLU extends Activation {
  constructor() {
    super(x => silu(x))
  }
}

/** Generic in `D` so `DimCheck` sees a literal; declares `"identity"` because the probe cannot read its `DimCheck` intersection parameter. */
export class Softmax<const D extends number = -1> extends Module {
  declare readonly [SHAPE_EFFECT]: "identity"

  constructor(readonly dim: D = -1 as D) {
    super()
  }

  forward<S extends Shape>(
    x: Tensor<S> & DimCheck<S, D>,
  ): Tensor<S> {
    return (x as AnyTensor).softmax(this.dim) as Tensor<S>
  }
}
