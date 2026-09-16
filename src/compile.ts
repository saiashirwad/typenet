import * as nativeBackend from "./backends/native.ts"
import { withContext } from "./context.ts"
import { formatLazyOp, topoOrder } from "./ir.ts"
import { nextSeed } from "./kernels.ts"
import { force, forceMany, serializeLazyGraph } from "./lazy.ts"
import { prod, shapesEqual, showShape } from "./storage.ts"
import { _internal, type AnyTensor, makeRaw, Tensor } from "./tensor.ts"

export type GraphUpdate = {
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

// Forcing swaps storage in place, so a tensor keeps its name; detach()/clone() and
// compile() placeholders make fresh tensors, so names do not cross them.
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

/** Shape-stable: later calls must match the traced shapes. Tracing covers a full training step. */
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
  /** @deprecated pass these on the first call instead. */
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
      // A .data/.item() read inside fn would force mid-trace and bake a constant into the graph.
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
          "compile(): an optimizer step produced a non-lazy gradient, compiled training steps need lazy gradients",
        )
      }
      return t
    })
    const roots = [
      ...outputs,
      ...updates.map(u => u.expr),
      ...materialize,
    ]
    const lazy = topoOrder(roots).filter(
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
            }, compiled graphs are shape-stable, recompile for a new shape`,
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
        "compiled function: an optimizer update target is not CPU storage, compiled graphs require parameters and optimizer state to stay put",
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

  const pinnedBytes = (leaf: AnyTensor): Uint8Array => {
    const buffer = _internal.cpuOf(leaf)
    if (buffer === null) {
      throw new Error(
        "compiled function: a captured tensor is not CPU storage, compiled graphs require captured leaves (e.g. parameters) to stay put",
      )
    }
    return new Uint8Array(
      buffer.buffer,
      buffer.byteOffset,
      buffer.byteLength,
    )
  }

  const runNative = (state: State): AnyTensor[] => {
    const native = state.native!
    if (native.handle === null) {
      // Pin every leaf once; only the dirty set is re-sent, so mutating a captured
      // non-parameter leaf after compile() is not seen.
      native.handle = nativeBackend.prepareGraphNative(
        native.json,
      )
      native.leafTensors.forEach((leaf, i) =>
        nativeBackend.pinLeafNative(
          native.handle!,
          i,
          pinnedBytes(leaf),
        )
      )
      const resent = new Set<AnyTensor>([
        ...state.placeholders,
        // Update targets are JS-authoritative, so re-send them every eval.
        ...state.updates.map(u => u.target),
      ])
      native.dirty = native.leafTensors.flatMap((t, i) => resent.has(t) ? [i] : [])
    }
    const dirtyIndex = Uint32Array.from(native.dirty!)
    let dirtyLength = 0
    for (const i of native.dirty!) {
      dirtyLength += pinnedBytes(native.leafTensors[i]!).length
    }
    const dirty = new Uint8Array(dirtyLength)
    let cursor = 0
    for (const i of native.dirty!) {
      const bytes = pinnedBytes(native.leafTensors[i]!)
      dirty.set(bytes, cursor)
      cursor += bytes.length
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
    // Drop the previous replay's materialized values so the graph recomputes against fresh inputs.
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
