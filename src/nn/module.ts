import { type DType, shapesEqual, type TensorStorage, type TypedArray } from "../storage.ts"
import { type AnyTensor, fromFlat, Tensor } from "../tensor.ts"
import { isParameter, type Parameter, storageKey } from "./parameter.ts"

export interface StateEntry {
  readonly shape: readonly number[]
  readonly dtype: DType
  readonly data: TypedArray
}

export type StateDict = Record<string, StateEntry>

export interface LoadReport {
  readonly missing: readonly string[]
  readonly unexpected: readonly string[]
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

const warnedFields = new WeakMap<Module, Set<string>>()

/** Members are discovered by reflection over own properties plus register()/registerBuffer(); parameters are deduped by storage identity, so tied weights count once. */
export abstract class Module {
  #trainingMode = true
  #registered = new Map<string, Module | AnyTensor>()
  #bufferNames = new Set<string>()
  #paramEpoch = 0
  #lastParamKeys: string | null = null

  /** Makes `v` discoverable when reflection cannot see it, such as a `#private` field or a getter-only value. Returns `v`. */
  register<T extends Module | Parameter | AnyTensor>(name: string, v: T): T {
    this.#registered.set(name, v as Module | AnyTensor)
    return v
  }

  /** Like {@link register}, but the tensor is saved in `stateDict()` and never collected as a parameter. */
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

  namedParameters(): ReadonlyMap<string, Parameter> {
    const out = new Map<string, Parameter>()
    const seen = new Set<TensorStorage>()
    for (const e of this.#walk("", new Set())) {
      if (e.kind !== "param") continue
      const key = storageKey(e.value as AnyTensor)
      if (seen.has(key)) continue
      seen.add(key)
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

  /** Every nested module, excluding `this`, by dotted path. */
  namedModules(): ReadonlyMap<string, Module> {
    const out = new Map<string, Module>()
    for (const e of this.#walk("", new Set())) {
      if (e.kind === "module") out.set(e.path, e.value as Module)
    }
    return out
  }

  /** Warns about `Parameter`s reachable only through a getter, which reflection never collects. */
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
          `typenet: Parameter "${name}" on ${this.constructor.name} is reachable but not registered \u2014 `
            + `wrap its assignment in register("${name}", ...) so it is collected by `
            + `namedParameters()/stateDict().`,
        )
      }
      proto = Object.getPrototypeOf(proto)
    }
  }

  get training(): boolean {
    return this.#trainingMode
  }

  train(mode = true): this {
    this.#trainingMode = mode
    // A private field is reachable from the class body on any instance, whatever `m`'s subclass.
    for (const m of this.namedModules().values()) m.#trainingMode = mode
    return this
  }

  eval(): this {
    return this.train(false)
  }

  zeroGrad(): void {
    for (const p of this.parameters()) p.zeroGrad()
  }

  apply(fn: (m: Module, path: string) => void): this {
    fn(this, "")
    for (const [path, m] of this.namedModules()) fn(m, path)
    return this
  }

  /** Bumps when the set of parameter names changes, which is how an `Optimizer` detects a stale parameter set. */
  get parameterEpoch(): number {
    const keys = [...this.namedParameters().keys()].join("\x00")
    if (keys !== this.#lastParamKeys) {
      this.#lastParamKeys = keys
      this.#paramEpoch++
    }
    return this.#paramEpoch
  }

  /** Keyed by name, so reordering field declarations cannot permute a checkpoint. */
  stateDict(): StateDict {
    const out: Record<string, StateEntry> = {}
    for (const [name, p] of this.namedParameters()) out[name] = snapshotEntry(p as AnyTensor)
    for (const [name, b] of this.namedBuffers()) out[name] = snapshotEntry(b)
    return out
  }

  /** Strict by default, throwing on any mismatch; `{ strict: false }` applies what matched and returns the report. */
  loadStateDict(d: StateDict, options?: { strict?: boolean }): LoadReport {
    const strict = options?.strict ?? true
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
        `loadStateDict: strict mismatch \u2014 missing: [${missing.join(", ")}], `
          + `unexpected: [${unexpected.join(", ")}], mismatched: [${mismatched.join(", ")}]`,
      )
    }
    return report
  }
}
