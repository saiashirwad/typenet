// Run under `runOnSmallStack("deep-chain-20000")`: a standalone re-run of
// the vitest depth-20000 chain from test/deep.test.ts, on a deliberately
// small (256 KB) V8 stack. It has no framework to report through, so it
// asserts directly and lets a thrown AssertionError (or a native stack
// overflow) fail the process with a non-zero exit code.
//
// This deliberately does NOT re-run the `printGraph` assertion from
// test/deep.test.ts: `printGraph` (src/compile.ts) computes its column
// width via `Math.max(1, ...entries.map(...))`, and spreading tens of
// thousands of arguments into one call blows this stack on its own,
// independent of and at a shallower depth than anything under test here
// (measured threshold on this stack size: fine at ~10k graph nodes, blown by
// ~16k, well under this file's 40k-node graph). That is a pre-existing
// stack-safety bug in a file outside this work item's scope (not
// test/deep.test.ts, test/small-stack.ts or test/scenarios/*.ts) — flagged
// separately rather than routed around here.
import assert from "node:assert/strict"
import { compile } from "../../src/compile.ts"
import { tensor } from "../../src/factories.ts"
import { configure } from "../../src/lazy.ts"
import { Tensor } from "../../src/tensor.ts"

type AnyTensor = Tensor<any>

const DEPTH = 20000

function chain(x: AnyTensor, depth: number): AnyTensor {
  let h = x
  for (let i = 0; i < depth; i++) h = h.mul(1.0001).add(0)
  return h
}

const expected = 1.0001 ** DEPTH

// forces a chain far deeper than the JS stack
{
  configure({ lazy: true })
  const out = chain(tensor([1, 2]), DEPTH)
  assert.ok(Math.abs(out.get(0) - expected) < 1e-2, `get(0): ${out.get(0)} vs ${expected}`)
  assert.ok(Math.abs(out.get(1) - 2 * expected) < 1e-2, `get(1): ${out.get(1)} vs ${2 * expected}`)
}

// differentiates a deep chain, lazily
{
  configure({ lazy: true })
  const x = tensor([1, 2]).requiresGrad()
  chain(x as AnyTensor, DEPTH).sum().backward()
  assert.ok(Math.abs(x.grad!.get(0) - expected) < 1e-2, `grad(0): ${x.grad!.get(0)} vs ${expected}`)
}

// differentiates a deep chain, eagerly
{
  configure({ lazy: false })
  const x = tensor([1, 2]).requiresGrad()
  chain(x as AnyTensor, DEPTH).sum().backward()
  assert.ok(Math.abs(x.grad!.get(0) - expected) < 1e-2, `grad(0): ${x.grad!.get(0)} vs ${expected}`)
}

// compiles and replays a (shallower) deep chain
{
  configure({ lazy: false })
  const step = compile((x: Tensor<[2]>) => chain(x as AnyTensor, 2000).sum())
  const first = step(tensor([1, 2])).item()
  const second = step(tensor([1, 2])).item()
  assert.ok(Math.abs(second - first) < 1e-6, `${second} vs ${first}`)
  assert.ok(Math.abs(first - 3 * 1.0001 ** 2000) < 1e-2, `${first}`)
}

configure({ lazy: false })
console.log("ok")
