// sum/mean/max/argmax by axis and size, including the > 4096-row GEMV-sum
// route.

import { disableNative, useNative } from "../index.ts"
import { rand } from "../src/factories.ts"
import { configure } from "../src/lazy.ts"
import type { AnyTensor } from "../src/tensor.ts"
import { bench, type BenchCaseSpec, isSmokeRun, type Mode } from "./lib/harness.ts"
import { REDUCE_SHAPES_FULL, REDUCE_SHAPES_SMOKE } from "./lib/sizes.ts"

type ReduceOp = "sum" | "mean" | "max" | "argmax"
const OPS: readonly ReduceOp[] = ["sum", "mean", "max", "argmax"]

// `gemv-route` (full only) sits at >= 4096 rows (GEMV_SUM_MIN_ROWS) so
// dim-0 sum/mean exercise the GEMV-sum rewrite; `below-route` is the same
// shape family just under the threshold, as a contrast.
const SHAPES = isSmokeRun() ? REDUCE_SHAPES_SMOKE : REDUCE_SHAPES_FULL

interface ReduceCase extends BenchCaseSpec {
  op: ReduceOp
  shape: readonly number[]
  axis: 0 | 1
}

const CASES: readonly ReduceCase[] = SHAPES.flatMap(({ label, shape }) =>
  OPS.flatMap(op =>
    ([0, 1] as const).map(axis => ({
      id: `reduce-${op}-${label}-axis${axis}`,
      op,
      shape,
      axis,
    }))
  )
)

const inputs = new Map<string, AnyTensor>()
function inputFor(shape: readonly number[]): AnyTensor {
  const key = shape.join("x")
  let t = inputs.get(key)
  if (!t) {
    t = rand([...shape]) as AnyTensor
    inputs.set(key, t)
  }
  return t
}

function applyOp(x: AnyTensor, op: ReduceOp, axis: 0 | 1): AnyTensor {
  switch (op) {
    case "sum":
      return x.sum(axis)
    case "mean":
      return x.mean(axis)
    case "max":
      return x.max(axis)
    case "argmax":
      return x.argmax(axis)
  }
}

function setMode(mode: Mode): void {
  if (mode === "native") {
    configure({ lazy: true })
    useNative()
  } else if (mode === "interp") {
    disableNative()
    configure({ lazy: true })
  } else {
    disableNative()
    configure({ lazy: false })
  }
}

async function main(): Promise<void> {
  await bench("micro-reduce", CASES, (kase, mode) => {
    setMode(mode)
    const x = inputFor(kase.shape)
    const out = applyOp(x, kase.op, kase.axis)
    out.data // force materialization
  })

  configure({ lazy: false })
  disableNative()
}

await main()
