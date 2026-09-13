import { type DType, shapesEqual, type TensorStorage, type TypedArray } from "../storage.ts"
import { type AnyTensor, fromFlat, Tensor } from "../tensor.ts"
import { isParameter, type Parameter, storageKey } from "./parameter.ts"

/** One entry of a `stateDict()` — enough to reconstruct the tensor exactly. */
export interface StateEntry {
  readonly shape: readonly number[]
  readonly dtype: DType
  readonly data: TypedArray
}

/** Name-keyed snapshot of a `Module`'s parameters and buffers (§3.3). */
export type StateDict = Record<string, StateEntry>

/**
 * What `loadStateDict` found. A renamed field, a resized layer, or a
 * checkpoint from a different architecture is *reported*, never
 * silently mis-loaded or silently dropped (§3.3's "name-keyed state
 * dicts" fix).
 */
export interface LoadReport {
  /** Names this module expects that the state dict does not have. */
  readonly missing: readonly string[]
  /** Names in the state dict that this module does not have. */
  readonly unexpected: readonly string[]
  /** Names present on both sides whose shape or dtype disagree. */
  readonly mismatched: readonly string[]
}

function snapshotEntry(t: AnyTensor): StateEntry {
  return { shape: [...t.shape], dtype: t.dtype, data: t.snapshot() as unknown as TypedArray }
}

type WalkEntry = {
  readonly path: string
  readonly kind: "param" | "buffer" | "module"
  readonly value: AnyTensor | Module
}

/** One-time-per-field warning tracking for {@link Module.#warnUnreachable}. */
const warnedFields = new WeakMap<Module, Set<string>>()

/**
 * `Module` v2 (§3.3, PLAN-V2). A member is discovered two ways, merged:
 *
 * 1. **Automatically**, by reflecting over the instance's own enumerable
 *    properties (`Object.entries(this)`), recursing through nested
 *    `Module`s and arrays — this is today's behaviour, unchanged, so a
 *    plain `this.layer = new Linear(...)` keeps working with no ceremony.
 * 2. **Explicitly**, via {@link register}/{@link registerBuffer} — the
 *    escape for `#private` fields and getters, which are invisible to
 *    (1): a `#private` field is never an enumerable own property, and a
 *    class getter lives on the prototype, not the instance.
 *
 * Every parameter is deduped by **storage identity** (through `tie`'s
 * alias table — see parameter.ts), not object identity: a tied weight,
 * or a `detach()`/`requiresGrad()` view of one buffer, is counted once.
 */
export abstract class Module {
  #trainingMode = true
  #registered = new Map<string, Module | AnyTensor>()
  #bufferNames = new Set<string>()
  #paramEpoch = 0
  #lastParamKeys: string | null = null

  /**
   * Registers `v` under `name` so it is found even when it is not an
   * enumerable own property of `this` — a `#private` field or a value
   * only exposed through a getter. Returns `v` so the common form is
   * `this.#w = this.register("w", parameter(...))`.
   */
  register<T extends Module | Parameter | AnyTensor>(name: string, v: T): T {
    this.#registered.set(name, v as Module | AnyTensor)
    return v
  }

  /**
   * Like {@link register}, but marks `v` as a buffer: state that rides
   * along in `stateDict()`/`loadStateDict()` and `namedBuffers()` but is
   * never collected by `namedParameters()`/`parameters()` and never
   * receives a gradient. Unlike parameters, buffers are **never**
   * auto-discovered — an arbitrary tensor field might be scratch state,
   * not part of the module's persistent state — so this is the only way
   * a tensor becomes a buffer.
   */
  registerBuffer<T extends AnyTensor>(name: string, t: T): T {
    this.#registered.set(name, t)
    this.#bufferNames.add(name)
    return t
  }

  #ownEntries(): [string, unknown][] {
    const seen = new Set<string>()
    const out: [string, unknown][] = []
    for (const [k, v] of Object.entries(this)) {
      seen.add(k)
      out.push([k, v])
    }
    for (const [k, v] of this.#registered) {
      if (!seen.has(k)) out.push([k, v])
    }
    return out
  }

  /**
   * Depth-first walk of this module's whole subtree, yielding every
   * parameter, buffer and nested module with a dotted path relative to
   * `this`. `visited` guards against a cyclic module graph (a submodule
   * that references an ancestor) — the shared submodule still appears at
   * every path that reaches it, but its own children are only expanded
   * once, from the first path.
   */
  *#walk(prefix: string, visited: Set<Module>): Generator<WalkEntry> {
    if (visited.has(this)) return
    visited.add(this)
    for (const [key, raw] of this.#ownEntries()) {
      yield* this.#emit(prefix ? `${prefix}.${key}` : key, raw, this.#bufferNames.has(key), visited)
    }
  }

  *#emit(path: string, value: unknown, asBuffer: boolean, visited: Set<Module>): Generator<WalkEntry> {
    if (value instanceof Module) {
      yield { path, kind: "module", value }
      yield* value.#walk(path, visited)
      return
    }
    if (Array.isArray(value)) {
      for (let i = 0; i < value.length; i++) {
        yield* this.#emit(`${path}.${i}`, value[i], asBuffer, visited)
      }
      return
    }
    if (value instanceof Tensor) {
      if (asBuffer) {
        yield { path, kind: "buffer", value }
      } else if (isParameter(value) || value.needsGrad) {
        yield { path, kind: "param", value }
      }
    }
  }

  /**
   * dotted paths, declaration order, deduped by storage identity — the
   * fix for the double-push/double-update bug a tied weight hits today
   * (§3.3).
   */
  namedParameters(): ReadonlyMap<string, Parameter> {
    const out = new Map<string, Parameter>()
    const seen = new Set<TensorStorage>()
    for (const e of this.#walk("", new Set())) {
      if (e.kind !== "param") continue
      const key = storageKey(e.value as AnyTensor)
      if (seen.has(key)) continue
      seen.add(key)
      // The runtime WeakSet check (isParameter) or the needsGrad
      // fallback is the actual source of truth here; the cast bridges
      // the reflection boundary, where the static type has already been
      // erased to `unknown` by Object.entries. See parameter.ts.
      out.set(e.path, e.value as Parameter)
    }
    this.#warnUnreachable(seen)
    return out
  }

  parameters(): Parameter[] {
    return [...this.namedParameters().values()]
  }

  namedBuffers(): ReadonlyMap<string, AnyTensor> {
    const out = new Map<string, AnyTensor>()
    for (const e of this.#walk("", new Set())) {
      if (e.kind === "buffer") out.set(e.path, e.value as AnyTensor)
    }
    return out
  }

  /** Every *nested* module (not `this`), by dotted path. */
  namedModules(): ReadonlyMap<string, Module> {
    const out = new Map<string, Module>()
    for (const e of this.#walk("", new Set())) {
      if (e.kind === "module") out.set(e.path, e.value as Module)
    }
    return out
  }

  /**
   * Warns once per field, naming it, when a `Parameter` is reachable
   * from a public getter on this instance but was never collected —
   * the "getters are invisible to reflection" trap `register()` exists
   * to close. Skipped once `NODE_ENV === "production"`: it is a
   * development aid, not a runtime contract.
   */
  #warnUnreachable(collected: ReadonlySet<TensorStorage>): void {
    if (typeof process !== "undefined" && process.env.NODE_ENV === "production") return
    let warned = warnedFields.get(this)
    let proto: unknown = Object.getPrototypeOf(this)
    while (proto && proto !== Module.prototype) {
      for (const [name, desc] of Object.entries(Object.getOwnPropertyDescriptors(proto))) {
        if (typeof desc.get !== "function") continue
        if (warned?.has(name)) continue
        let value: unknown
        try {
          value = desc.get.call(this)
        } catch {
          continue
        }
        if (!(value instanceof Tensor)) continue
        if (!isParameter(value) && !value.needsGrad) continue
        if (collected.has(storageKey(value))) continue
        if (!warned) {
          warned = new Set()
          warnedFields.set(this, warned)
        }
        warned.add(name)
        console.warn(
          `typenet: Parameter "${name}" on ${this.constructor.name} is reachable but not registered — `
            + `wrap its assignment in register("${name}", ...) so it is collected by `
            + `namedParameters()/stateDict().`,
        )
      }
      proto = Object.getPrototypeOf(proto)
    }
  }

  /** `true` between `train()` and the next `eval()`/`train(false)`. */
  get training(): boolean {
    return this.#trainingMode
  }

  train(mode = true): this {
    this.#trainingMode = mode
    for (const m of this.namedModules().values()) {
      // Module's own `#trainingMode` is accessible on any Module
      // instance from inside Module's own methods — that is how JS
      // private class fields work, regardless of which subclass `m` is.
      m.#trainingMode = mode
    }
    return this
  }

  eval(): this {
    return this.train(false)
  }

  zeroGrad(): void {
    for (const p of this.parameters()) p.zeroGrad()
  }

  /** Calls `fn` on `this` and then on every nested module, each with its dotted path (`""` for `this`). */
  apply(fn: (m: Module, path: string) => void): this {
    fn(this, "")
    for (const [path, m] of this.namedModules()) fn(m, path)
    return this
  }

  /**
   * Bumps whenever the *set* of parameter names changes shape (a
   * parameter added or removed since the last read) — computed lazily
   * by diffing against the previous read rather than by intercepting
   * every field assignment, which would need a `Proxy` around every
   * `Module` instance for no benefit today. An `Optimizer` built against
   * a stale parameter set is a silent-corruption bug (§3.3); it snapshots
   * this at construction and can throw if a later read disagrees.
   */
  get parameterEpoch(): number {
    const keys = [...this.namedParameters().keys()].join(" ")
    if (keys !== this.#lastParamKeys) {
      this.#lastParamKeys = keys
      this.#paramEpoch++
    }
    return this.#paramEpoch
  }

  /** By name, so reordering two field declarations cannot permute a checkpoint (§3.3). */
  stateDict(): StateDict {
    const out: Record<string, StateEntry> = {}
    for (const [name, p] of this.namedParameters()) out[name] = snapshotEntry(p as AnyTensor)
    for (const [name, b] of this.namedBuffers()) out[name] = snapshotEntry(b)
    return out
  }

  /**
   * Restores by name. `strict` (default `true`) throws when anything is
   * missing, unexpected or mismatched; pass `{ strict: false }` to get
   * the {@link LoadReport} back instead and apply whatever did match.
   */
  loadStateDict(d: StateDict, o?: { strict?: boolean }): LoadReport {
    const strict = o?.strict ?? true
    const missing: string[] = []
    const unexpected: string[] = []
    const mismatched: string[] = []

    const current = new Map<string, AnyTensor>()
    for (const [name, p] of this.namedParameters()) current.set(name, p as AnyTensor)
    for (const [name, b] of this.namedBuffers()) current.set(name, b)

    for (const [name, target] of current) {
      const entry = d[name]
      if (!entry) {
        missing.push(name)
        continue
      }
      if (!shapesEqual(entry.shape, target.shape) || entry.dtype !== target.dtype) {
        mismatched.push(name)
        continue
      }
      const source = fromFlat(entry.data, [...entry.shape], entry.dtype) as AnyTensor
      target.copy_(source)
    }
    for (const name of Object.keys(d)) {
      if (!current.has(name)) unexpected.push(name)
    }

    const report: LoadReport = { missing, unexpected, mismatched }
    if (strict && (missing.length > 0 || unexpected.length > 0 || mismatched.length > 0)) {
      throw new Error(
        `loadStateDict: strict mismatch — missing: [${missing.join(", ")}], `
          + `unexpected: [${unexpected.join(", ")}], mismatched: [${mismatched.join(", ")}]`,
      )
    }
    return report
  }
}
