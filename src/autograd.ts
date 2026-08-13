import { rawBinary } from "./eager.ts"
import { eagerly, force, forceMany, lazyMode } from "./lazy.ts"
import { arrayCtor, shapesEqual, showShape } from "./storage.ts"
import { _internal, type AnyTensor, makeRaw } from "./tensor.ts"

interface GradNode {
  name: string
  inputs: AnyTensor[]

  backward: (grad: AnyTensor) => (AnyTensor | null)[]
}

let gradEnabled = true

export function noGrad<T>(fn: () => T): T {
  const prev = gradEnabled
  gradEnabled = false
  try {
    return fn()
  } finally {
    gradEnabled = prev
  }
}

/**
 * Iterative for the same reason as `topoOrder` — the tape
 * behind a long rollout is thousands of nodes deep.
 */
function tapeOrder(root: AnyTensor): AnyTensor[] {
  const topo: AnyTensor[] = []
  const seen = new Set<AnyTensor>()
  const stack: {
    t: AnyTensor
    inputs: AnyTensor[]
    i: number
  }[] = []

  const push = (t: AnyTensor): void => {
    if (seen.has(t)) return
    seen.add(t)
    stack.push({
      t,
      inputs: _internal.gradNodeOf(t)?.inputs ?? [],
      i: 0,
    })
  }
  push(root)
  while (stack.length > 0) {
    const frame = stack[stack.length - 1]!
    if (frame.i < frame.inputs.length) {
      push(frame.inputs[frame.i++]!)
      continue
    }
    stack.pop()
    topo.push(frame.t)
  }
  return topo
}

function withGrad(
  result: AnyTensor,
  name: string,
  inputs: AnyTensor[],
  backward: (grad: AnyTensor) => (AnyTensor | null)[],
): AnyTensor {
  if (gradEnabled && inputs.some(t => t.needsGrad)) {
    _internal.setGradNode(result, { name, inputs, backward })
  }
  return result
}

/**
 * Reverse-mode sweep from `root`. The implementation lives here rather
 * than on the class body; `Tensor.backward` delegates.
 */
function runBackward(
  root: AnyTensor,
  gradient: AnyTensor | undefined,
  activeTrace: () => unknown,
): void {
  if (!root.needsGrad) {
    throw new Error(
      "backward() on a tensor that does not require grad",
    )
  }
  let seed: AnyTensor
  const lazyPath = lazyMode
    && _internal.sourceOf(root).kind === "lazy"
  if (gradient) {
    if (!shapesEqual(gradient.shape, root.shape)) {
      throw new Error(
        `backward() gradient shape ${showShape(gradient.shape)} does not match ${showShape(root.shape)}`,
      )
    }
    seed = lazyPath ? gradient : force(gradient)
  } else {
    if (root.numel !== 1) {
      throw new Error(
        "backward() without a gradient requires a scalar output",
      )
    }
    seed = makeRaw(
      new (arrayCtor(root.dtype))(root.numel).fill(1),
      root.shape,
      root.dtype,
    )
  }

  const topo = tapeOrder(root)

  const grads = new Map<AnyTensor, AnyTensor>()
  grads.set(root, seed)

  const walk = () =>
    noGrad(() => {
      for (let i = topo.length - 1; i >= 0; i--) {
        const t = topo[i]!
        const g = grads.get(t)
        if (!g) continue
        const node = _internal.gradNodeOf(t)
        if (node) {
          const inputGrads = node.backward(g)
          node.inputs.forEach((input, j) => {
            const ig = inputGrads[j]
            if (!ig || !input.needsGrad) return
            const existing = grads.get(input)
            grads.set(
              input,
              existing
                ? rawBinary(existing, ig, "add")
                : ig,
            )
          })
        } else if (t._requiresGrad) {
          t.grad = t.grad
            ? rawBinary(t.grad, g, "add")
            : g
        }
      }
    })

  if (lazyPath) {
    walk()
    // Materialize the whole forward topo plus every parameter grad
    // in a single multi-root forcing point (one native FFI hop when
    // native is enabled). The forward tensors are forced too so that
    // values read after an in-place optimizer step (which mutates
    // the leaf parameter data) still see pre-step values, matching
    // eager and phase-1 lazy semantics. During a compile() trace the
    // forcing is deferred: the lazy grads stay graph expressions and
    // the traced optimizer updates (plus the grads themselves) become
    // extra roots of the compiled graph, materialized on replay.
    if (!activeTrace()) {
      forceMany([
        ...topo,
        ...topo
          .filter(t => t._requiresGrad && t.grad)
          .map(t => t.grad as AnyTensor),
      ])
    }
    return
  }

  for (const t of topo) force(t)
  eagerly(walk)
}

export type { GradNode }
export { runBackward, tapeOrder, withGrad }
