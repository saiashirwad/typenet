// The autograd engine: a tape of GradNodes recorded on each op and
// walked in reverse by Tensor.backward(). withGrad attaches a node only
// when grad is enabled and some input needs grad. Depends on the
// Tensor class only as a type (AnyTensor), so this layer has no runtime
// edge upward.

import type { AnyTensor } from "./tensor.ts"

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
 * Post-order traversal of the autograd tape behind `root`: every
 * tensor appears after the inputs it was computed from, so walking the
 * result in reverse visits each node only once its own gradient is
 * complete. Iterative for the same reason as `topoOrder` — the tape
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
      inputs: t.gradNode ? t.gradNode.inputs : [],
      i: 0
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
  backward: (grad: AnyTensor) => (AnyTensor | null)[]
): AnyTensor {
  if (gradEnabled && inputs.some(t => t.needsGrad)) {
    result.gradNode = { name, inputs, backward }
  }
  return result
}

export type { GradNode }
export { tapeOrder, withGrad }
