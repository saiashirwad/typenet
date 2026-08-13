import * as nativeBackend from "./backends/native.ts"
import { nextSeed } from "./kernels.ts"
import { force, forceMany, formatLazyOp, lazily, serializeLazyGraph, topoOrder } from "./lazy.ts"
import { type CpuStorage, type LazyStorage, prod, shapesEqual, showShape, type TensorStorage } from "./storage.ts"
import { type AnyTensor, makeRaw, Tensor } from "./tensor.ts"

type GraphUpdate = {
  target: AnyTensor
  expr: AnyTensor
}

type UpdateTrace = {
  updates: GraphUpdate[]
  materialize: AnyTensor[]
}

let updateTrace: UpdateTrace | null = null

export function _activeUpdateTrace(): UpdateTrace | null {
  return updateTrace
}

type CompiledInput<T extends AnyTensor> = T extends Tensor<infer S, any> ? Tensor<S, any> | ArrayLike<number> : never

// Forcing keeps a tensor's identity (its storage is swapped in
// place), so a name survives materialization; names do NOT cross
// detach()/clone()/compile() placeholders, which create fresh tensor
// objects.

const tensorNames = new WeakMap<AnyTensor, string>()

export function printGraph(
  roots: AnyTensor | AnyTensor[],
): string {
  const rootList = Array.isArray(roots) ? roots : [roots]
  const ids = new Map<AnyTensor, number>()
  const entries = topoOrder(rootList).map(t => ({
    t,
    node: t._storage.kind === "lazy" ? t._storage.node : null,
  }))
  entries.forEach(({ t }, i) => ids.set(t, i))
  const label = (t: AnyTensor): string => tensorNames.get(t) ?? `%${ids.get(t)!}`
  const width = Math.max(
    1,
    ...entries.map(({ t }) => label(t).length),
  )
  const rootSet = new Set(rootList)
  return entries
    .map(({ t, node }) => {
      const lhs = label(t).padEnd(width)
      const shape = showShape(t.shape)
      const tail = `${shape} ${t.dtype}${rootSet.has(t) ? " ; root" : ""}`
      if (!node) return `${lhs} = leaf ${tail}`
      return `${lhs} = ${formatLazyOp(node, label)} ${tail}`
    })
    .join("\n")
}

export { tensorNames }

/**
 * The first call must pass CPU float32 tensors (their shapes/dtype
 * pin the traced graph); later calls may pass either tensors of the
 * same shape or flat `ArrayLike<number>` buffers of matching length.
 * Calling with a different argument count, shape, or dtype throws —
 * compiled graphs are shape-stable, recompile for a new shape.
 *
 * Tracing always happens under lazy semantics regardless of the
 * global `configure({ lazy })` flag, and the flag is restored
 * afterwards. `fn` must be a pure dataflow function of its inputs
 * with one exception: a full training step — `loss.backward()` plus
 * an optimizer `step()` — is traced into the graph, so a compiled
 * step evaluates forward, backward, and the parameter update in one
 * pass and writes the updated values back into the parameter (and
 * optimizer state) buffers on every call. Other forcing points
 * (`.data`, `.item()`, ...) inside `fn` remain unsupported.
 *
 * Call `.dispose()` when done to free the native prepared-graph handle
 * (no-op on the interpreter path, and safe to call more than once).
 */
export type CompiledFn<
  Args extends AnyTensor[],
  R extends AnyTensor | AnyTensor[],
> =
  & ((
    ...inputs: { [K in keyof Args]: CompiledInput<Args[K]> }
  ) => R)
  & { dispose(): void }

export function compile<
  Args extends AnyTensor[],
  R extends AnyTensor | AnyTensor[],
>(fn: (...args: Args) => R): CompiledFn<Args, R> {
  type State = {
    placeholders: AnyTensor[]
    outputs: AnyTensor[]
    tuple: boolean
    shapes: number[][]
    updates: GraphUpdate[]
    materialize: { t: AnyTensor; storage: LazyStorage }[]
    native: {
      json: string
      handle: number | null
      leafTensors: AnyTensor[]
      leafOffsets: number[]
      leafBytes: number
      rootShapes: number[][]
    } | null
    lazy: { t: AnyTensor; storage: LazyStorage }[]
  }
  let state: State | null = null

  const trace = (inputs: readonly unknown[]): State => {
    const placeholders = inputs.map((input, i) => {
      if (!(input instanceof Tensor)) {
        throw new Error(
          `compile() traces on the first call, so argument ${i} must be a Tensor (later calls may pass flat buffers)`,
        )
      }
      const t = force(input as AnyTensor)
      if (
        t._storage.kind !== "cpu"
        || t.dtype !== "float32"
      ) {
        throw new Error(
          `compile() only supports CPU float32 inputs, argument ${i} is ${t.dtype}`,
        )
      }
      return makeRaw(
        (t._storage.data as Float32Array).slice(),
        t.shape,
        "float32",
      )
    })
    const prevTrace = updateTrace
    const traced: UpdateTrace = {
      updates: [],
      materialize: [],
    }
    updateTrace = traced
    let result: unknown
    try {
      result = lazily(() => fn(...(placeholders as Args)))
    } finally {
      updateTrace = prevTrace
    }
    const tuple = Array.isArray(result)
    const outputs = (
      tuple ? result : [result]
    ) as AnyTensor[]
    outputs.forEach((out, i) => {
      if (!(out instanceof Tensor)) {
        throw new Error(
          `compile() expected fn to return a Tensor or Tensor[], got ${typeof out} at output ${i}`,
        )
      }
    })
    const updates = traced.updates
    const materialize = traced.materialize.map(t => {
      if (t._storage.kind !== "lazy") {
        throw new Error(
          "compile(): an optimizer step produced a non-lazy gradient — compiled training steps need lazy gradients",
        )
      }
      return { t, storage: t._storage }
    })
    const roots = [
      ...outputs,
      ...updates.map(u => u.expr),
      ...materialize.map(m => m.t),
    ]
    const lazy: State["lazy"] = topoOrder(roots)
      .filter(t => t._storage.kind === "lazy")
      .map(t => ({ t, storage: t._storage as LazyStorage }))
    const serialized = serializeLazyGraph(roots)
    return {
      placeholders,
      outputs,
      tuple,
      shapes: placeholders.map(p => [...p.shape]),
      updates,
      materialize,
      native: serialized
        ? {
          json: serialized.json,
          handle: null,
          leafTensors: serialized.leafTensors,
          leafOffsets: serialized.leafOffsets,
          leafBytes: serialized.leafBytes,
          rootShapes: serialized.rootShapes,
        }
        : null,
      lazy,
    }
  }

  const swapInputs = (
    state: State,
    inputs: readonly unknown[],
  ): void => {
    if (inputs.length !== state.placeholders.length) {
      throw new Error(
        `compiled function expected ${state.placeholders.length} arguments, got ${inputs.length}`,
      )
    }
    inputs.forEach((input, i) => {
      const storage = state.placeholders[i]!._storage
      const buffer = (storage as CpuStorage)
        .data as Float32Array
      if (input instanceof Tensor) {
        const t = force(input as AnyTensor)
        if (t._storage.kind !== "cpu") {
          throw new Error(
            `compiled function argument ${i}: expected a CPU tensor`,
          )
        }
        if (t.dtype !== "float32") {
          throw new Error(
            `compiled function argument ${i}: expected float32, got ${t.dtype}`,
          )
        }
        if (!shapesEqual(t.shape, state.shapes[i]!)) {
          throw new Error(
            `compiled function argument ${i}: expected shape ${showShape(state.shapes[i]!)}, got ${
              showShape(t.shape)
            } — compiled graphs are shape-stable, recompile for a new shape`,
          )
        }
        buffer.set(t._storage.data as Float32Array)
      } else if (
        input != null
        && typeof (input as ArrayLike<number>).length
          === "number"
      ) {
        if (
          (input as ArrayLike<number>).length
            !== buffer.length
        ) {
          throw new Error(
            `compiled function argument ${i}: expected ${buffer.length} values for shape ${showShape(state.shapes[i]!)}, got ${
              (input as ArrayLike<number>).length
            }`,
          )
        }
        buffer.set(
          Array.from(input as ArrayLike<number>, Number),
        )
      } else {
        throw new Error(
          `compiled function argument ${i}: expected a Tensor or flat ArrayLike<number>`,
        )
      }
    })
  }

  const applyUpdate = (
    u: GraphUpdate,
    values: Float32Array,
  ): void => {
    const storage = u.target._storage
    if (storage.kind !== "cpu") {
      throw new Error(
        "compiled function: an optimizer update target is not CPU storage — compiled graphs require parameters and optimizer state to stay put",
      )
    }
    if (storage.data.length !== values.length) {
      throw new Error(
        "compiled function: an optimizer update target changed size",
      )
    }
    ;(storage.data as Float32Array).set(values)
  }

  const runNative = (state: State): AnyTensor[] => {
    const native = state.native!
    const leaves = new Float32Array(native.leafBytes)
    native.leafTensors.forEach((leaf, i) => {
      const storage = leaf._storage
      if (storage.kind !== "cpu") {
        throw new Error(
          "compiled function: a captured tensor is not CPU storage — compiled graphs require captured leaves (e.g. parameters) to stay put",
        )
      }
      leaves.set(
        storage.data as Float32Array,
        native.leafOffsets[i]!,
      )
    })
    if (native.handle === null) {
      native.handle = nativeBackend.prepareGraphNative(
        native.json,
      )
    }
    const data = nativeBackend.evalPreparedNative(
      native.handle,
      leaves,
      nextSeed(),
    )
    let offset = 0
    const take = (shape: number[]): Float32Array => {
      const n = prod(shape)
      const view = data.subarray(offset, offset + n)
      offset += n
      return view
    }
    const outputs = native.rootShapes
      .slice(0, state.outputs.length)
      .map(shape => makeRaw(take(shape), shape, "float32"))
    for (const u of state.updates) {
      applyUpdate(u, take([...u.expr.shape]))
    }
    for (const m of state.materialize) {
      const values = take([...m.t.shape])
      m.storage.cache = makeRaw(
        values,
        [...m.t.shape],
        "float32",
      )
      ;(m.t as { _storage: TensorStorage })._storage = m.storage.cache._storage
    }
    return outputs
  }

  const runInterpreter = (state: State): AnyTensor[] => {
    for (const { t, storage } of state.lazy) {
      storage.cache = null
      ;(t as { _storage: TensorStorage })._storage = storage
    }
    forceMany([
      ...state.outputs,
      ...state.updates.map(u => u.expr),
      ...state.materialize.map(m => m.t),
    ])
    for (const u of state.updates) {
      applyUpdate(u, u.expr.data as Float32Array)
    }
    return state.outputs.map(out => makeRaw(out.data, out.shape, out.dtype))
  }

  const compiled = ((...inputs: readonly unknown[]) => {
    if (!state) state = trace(inputs)
    swapInputs(state, inputs)
    const outputs = state.native && nativeBackend.isNativeEnabled()
      ? runNative(state)
      : runInterpreter(state)
    return (state.tuple ? outputs : outputs[0]) as R
  }) as CompiledFn<Args, R>

  compiled.dispose = () => {
    const handle = state?.native?.handle
    if (handle == null) return
    nativeBackend.releaseGraphNative(handle)
    state!.native!.handle = null
  }

  return compiled
}
