import * as nativeBackend from "./backends/native.ts"
import { withContext } from "./context.ts"
import { formatLazyOp, topoOrder } from "./ir.ts"
import { nextSeed } from "./kernels.ts"
import { force, forceMany, serializeLazyGraph } from "./lazy.ts"
import { prod, shapesEqual, showShape } from "./storage.ts"
import { _internal, type AnyTensor, makeRaw, Tensor } from "./tensor.ts"

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

type CompiledInput<T extends AnyTensor> = T extends Tensor<infer S> ? Tensor<S> | ArrayLike<number> : never

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
  const entries = topoOrder(rootList).map(t => {
    const source = _internal.sourceOf(t)
    return {
      t,
      node: source.kind === "lazy" && !_internal.hasValue(t)
        ? source.node
        : null,
    }
  })
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
>(
  fn: (...args: Args) => R,
  /**
   * Example inputs to trace against up front. Omitting them falls back
   * to tracing on the first call.
   * @deprecated the no-examples overload traces on first call; pass the
   * tensors you used to pass on the first call instead.
   */
  exampleInputs?: [...Args],
): CompiledFn<Args, R> {
  type State = {
    placeholders: AnyTensor[]
    outputs: AnyTensor[]
    tuple: boolean
    shapes: number[][]
    updates: GraphUpdate[]
    materialize: AnyTensor[]
    native: {
      json: string
      handle: number | null
      leafTensors: AnyTensor[]
      leafOffsets: number[]
      leafBytes: number
      rootShapes: number[][]
      /** Leaf indices re-sent every eval: placeholders + update targets. */
      dirty: number[] | null
    } | null
    lazy: AnyTensor[]
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
        _internal.cpuOf(t) === null
        || t.dtype !== "float32"
      ) {
        throw new Error(
          `compile() only supports CPU float32 inputs, argument ${i} is ${t.dtype}`,
        )
      }
      return makeRaw(
        (_internal.cpuOf(t) as Float32Array).slice(),
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
      // Tracing forbids every value read: a `.data` / `.item()` inside
      // fn would force mid-trace and bake a constant into the graph.
      result = withContext(
        { lazy: true, tracing: true },
        () => fn(...(placeholders as Args)),
      )
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
      if (_internal.sourceOf(t).kind !== "lazy") {
        throw new Error(
          "compile(): an optimizer step produced a non-lazy gradient — compiled training steps need lazy gradients",
        )
      }
      return t
    })
    const roots = [
      ...outputs,
      ...updates.map(u => u.expr),
      ...materialize,
    ]
    const lazy: State["lazy"] = topoOrder(roots).filter(
      t => _internal.sourceOf(t).kind === "lazy",
    )
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
          dirty: null,
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
      const buffer = _internal.cpuOf(
        state.placeholders[i]!,
      ) as Float32Array
      if (input instanceof Tensor) {
        const t = force(input as AnyTensor)
        if (_internal.cpuOf(t) === null) {
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
        buffer.set(_internal.cpuOf(t) as Float32Array)
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
    if (_internal.sourceOf(u.target).kind !== "cpu") {
      throw new Error(
        "compiled function: an optimizer update target is not CPU storage — compiled graphs require parameters and optimizer state to stay put",
      )
    }
    const buffer = _internal.cpuOf(u.target)!
    if (buffer.length !== values.length) {
      throw new Error(
        "compiled function: an optimizer update target changed size",
      )
    }
    ;(buffer as Float32Array).set(values)
  }

  const pinnedBuffer = (leaf: AnyTensor): Float32Array => {
    const buffer = _internal.cpuOf(leaf)
    if (buffer === null) {
      throw new Error(
        "compiled function: a captured tensor is not CPU storage — compiled graphs require captured leaves (e.g. parameters) to stay put",
      )
    }
    return buffer as Float32Array
  }

  const runNative = (state: State): AnyTensor[] => {
    const native = state.native!
    if (native.handle === null) {
      // Prepare once and pin every leaf. Static captures (edge lists,
      // targets, degree tables) cross the FFI exactly once, here; only
      // the dirty set below is re-sent per eval. Mutating a captured
      // non-parameter leaf after compile() is therefore not seen — the
      // same "captured leaves stay put" contract as the error above.
      native.handle = nativeBackend.prepareGraphNative(
        native.json,
      )
      native.leafTensors.forEach((leaf, i) =>
        nativeBackend.pinLeafNative(
          native.handle!,
          i,
          pinnedBuffer(leaf),
        )
      )
      const resent = new Set<AnyTensor>([
        ...state.placeholders,
        // Update targets are JS-authoritative (applyUpdate writes them,
        // tests and checkpoints mutate them), so their current values
        // are re-sent every eval.
        ...state.updates.map(u => u.target),
      ])
      native.dirty = native.leafTensors.flatMap((t, i) => resent.has(t) ? [i] : [])
    }
    const dirtyIndex = Uint32Array.from(native.dirty!)
    let dirtyLength = 0
    for (const i of native.dirty!) {
      dirtyLength += pinnedBuffer(native.leafTensors[i]!).length
    }
    const dirty = new Float32Array(dirtyLength)
    let cursor = 0
    for (const i of native.dirty!) {
      const buffer = pinnedBuffer(native.leafTensors[i]!)
      dirty.set(buffer, cursor)
      cursor += buffer.length
    }
    const data = nativeBackend.evalPreparedNative(
      native.handle,
      dirty,
      dirtyIndex,
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
      _internal.setCpu(m, take([...m.shape]))
    }
    return outputs
  }

  const runInterpreter = (state: State): AnyTensor[] => {
    // Rewind: drop every materialized value from the previous replay so
    // the whole graph recomputes against the freshly swapped inputs.
    for (const t of state.lazy) {
      _internal.resetCpu(t)
    }
    forceMany([
      ...state.outputs,
      ...state.updates.map(u => u.expr),
      ...state.materialize,
    ])
    for (const u of state.updates) {
      applyUpdate(u, u.expr.data as Float32Array)
    }
    return state.outputs.map(out => makeRaw(out.data, out.shape, out.dtype))
  }

  if (exampleInputs) state = trace(exampleInputs)

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
