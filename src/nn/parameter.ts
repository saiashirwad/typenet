import type { Shape } from "../shape.ts"
import type { TensorStorage } from "../storage.ts"
import { _internal, type AnyTensor, type Tensor } from "../tensor.ts"

/**
 * The `PARAM` phantom brand (§3.3 of PLAN-V2): a unique symbol property
 * that never exists at runtime, only in the type. `parameter()` is the
 * one place that mints it, backed by the `paramBrand` `WeakSet` below —
 * the "dual type/value twin" the shape-type laws require (UNDERSTAND.md
 * §2.1): the type says "this is a trainable weight", the `WeakSet` is
 * what `Module` actually trusts at runtime. A bare `Tensor<S>` that has
 * not been through `parameter()` is simply not assignable to
 * `Parameter<S>` — fail-open by construction, never a runtime surprise.
 */
declare const PARAM: unique symbol
export type Parameter<S extends Shape = Shape> = Tensor<S> & { readonly [PARAM]: true }

const paramBrand = new WeakSet<AnyTensor>()

/** The runtime twin of the `PARAM` brand — what `Module` actually checks. */
export function isParameter(t: AnyTensor): t is Parameter {
  return paramBrand.has(t)
}

/**
 * Turns a tensor into a leaf parameter: enables gradients (the tape
 * starts here, same as {@link Tensor#requiresGrad}) and brands the
 * result so `Module.namedParameters()` recognises it. Layers that
 * predate this brand (built with a bare `.requiresGrad()`) are still
 * collected — `Module` falls back to `needsGrad` — but everything wired
 * through `parameter()` also carries the nominal `Parameter<S>` type.
 */
export function parameter<S extends Shape>(t: Tensor<S>): Parameter<S> {
  const p = t.requiresGrad() as Parameter<S>
  paramBrand.add(p as AnyTensor)
  return p
}

/**
 * Weight tying (§3.3). `b`'s own storage cannot be rewritten in place —
 * `Tensor`'s `#source` field is immutable once constructed (tensor.ts)
 * — so `tie` cannot literally hand `b` `a`'s buffer from outside that
 * module. Instead it records, in a `WeakMap` keyed by storage object
 * identity, that `b`'s storage stands for `a`'s everywhere `Module`
 * cares about identity: `namedParameters`/`parameters` dedup, and the
 * single `stateDict` entry a tied pair must produce. Resolve an
 * arbitrary tensor's canonical storage with {@link storageKey}.
 *
 * This is the identity half of tying. Sharing the *gradient* — so both
 * uses accumulate into one `.grad` — needs the literal same `Tensor`
 * object at both use sites (a per-instance field, not a storage-level
 * one); that is `TiedLinear`'s job (W5.10), which stores the embedding's
 * own `Parameter` object rather than calling `tie` on two independently
 * constructed ones.
 */
const storageAlias = new WeakMap<TensorStorage, TensorStorage>()

export function tie<S extends Shape>(a: Parameter<S>, b: Parameter<NoInfer<S>>): void {
  const target = canonicalStorage(_internal.sourceOf(a as AnyTensor))
  storageAlias.set(_internal.sourceOf(b as AnyTensor), target)
}

/** Follows `tie`'s alias chain to the representative storage object. */
export function canonicalStorage(s: TensorStorage): TensorStorage {
  let cur = s
  const seen = new Set<TensorStorage>()
  while (storageAlias.has(cur) && !seen.has(cur)) {
    seen.add(cur)
    cur = storageAlias.get(cur)!
  }
  return cur
}

/**
 * The identity `Module` dedups tensors by: storage, resolved through
 * any `tie()` alias — never object identity (a `detach()`/`requiresGrad()`
 * view of one buffer must dedup with its source too, and those views
 * already share the same underlying storage object — see `makeView` in
 * tensor.ts — so this needs no special case for them).
 */
export function storageKey(t: AnyTensor): TensorStorage {
  return canonicalStorage(_internal.sourceOf(t))
}
