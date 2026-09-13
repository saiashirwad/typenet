import type { DimCheck, Shape } from "../../shape.ts"
import { type AnyTensor, gelu, silu, Tensor } from "../../tensor.ts"
import { Module } from "../module.ts"
import { SHAPE_EFFECT } from "../sequential.ts"

class Activation extends Module {
  // Every activation here is shape-preserving and width-agnostic — the
  // canonical "identity" effect (W4.9) — declared once on the shared base
  // rather than repeated on each subclass. Their plain `Tensor<S> ->
  // Tensor<S>` `forward` already probes to the same answer structurally,
  // so this is belt-and-braces, not load-bearing: it survives a future
  // `forward` signature (an intersection type like `DimCheck`, say) that
  // would stop probing, the exact trap `Softmax` below is in today.
  declare readonly [SHAPE_EFFECT]: "identity"

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
 * GELU (tanh approximation), over the fused `gelu` node (W4.1) rather than
 * the six-node `0.5x(1+tanh(...))` composed spelling.
 */
export class GELU extends Activation {
  constructor() {
    super(x => gelu(x))
  }
}

/** SiLU / swish: `x * sigmoid(x)`, over the fused `silu` node (W4.1). */
export class SiLU extends Activation {
  constructor() {
    super(x => silu(x))
  }
}

/**
 * Generic in the dim so `DimCheck` sees a literal at the call, not the
 * wide `number` (`IsValidDim<S, number>` is always true).
 *
 * Declares `"identity"` (W4.9): `forward`'s parameter type is
 * `Tensor<S> & DimCheck<S, D>`, an intersection the structural probe
 * cannot read (it only matches a bare `Tensor<S>` parameter) — without the
 * declaration, `sequential(new Linear(4,4), new Softmax(-1), ...)` would
 * fall through to `ApplyLayer`'s `S` fallback by accident rather than by
 * the fallback being correct, and a future change to the probe could stop
 * agreeing with it silently.
 */
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
