import type { Shape } from "../shape.ts"
import type { TensorStorage } from "../storage.ts"
import { _internal, type AnyTensor, type Tensor } from "../tensor.ts"

declare const PARAM: unique symbol
export type Parameter<S extends Shape = Shape> = Tensor<S> & { readonly [PARAM]: true }

const paramBrand = new WeakSet<AnyTensor>()

export function isParameter(t: AnyTensor): t is Parameter {
  return paramBrand.has(t)
}

/** Enables gradients and brands the tensor so `Module.namedParameters()` collects it. */
export function parameter<S extends Shape>(t: Tensor<S>): Parameter<S> {
  const p = t.requiresGrad() as Parameter<S>
  paramBrand.add(p as AnyTensor)
  return p
}

/** Records that `b`'s storage stands for `a`'s, so `Module` sees one parameter and one `stateDict` entry. Sharing one `.grad` needs the same `Tensor` object at both sites (see `TiedLinear`). */
const storageAlias = new WeakMap<TensorStorage, TensorStorage>()

export function tie<S extends Shape>(a: Parameter<S>, b: Parameter<NoInfer<S>>): void {
  const target = canonicalStorage(_internal.sourceOf(a as AnyTensor))
  storageAlias.set(_internal.sourceOf(b as AnyTensor), target)
}

function canonicalStorage(s: TensorStorage): TensorStorage {
  let cur = s
  const seen = new Set<TensorStorage>()
  while (storageAlias.has(cur) && !seen.has(cur)) {
    seen.add(cur)
    cur = storageAlias.get(cur)!
  }
  return cur
}

/** The identity `Module` dedups tensors by: storage resolved through any `tie()` alias, not object identity. */
export function storageKey(t: AnyTensor): TensorStorage {
  return canonicalStorage(_internal.sourceOf(t))
}
