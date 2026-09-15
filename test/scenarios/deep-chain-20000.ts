// Run under `runOnSmallStack("deep-chain-20000")`: a standalone re-run of
// the depth-20000 chain from test/deep.test.ts on a 256 KB V8 stack. There
// is no framework to report through, so it asserts directly and lets a
// thrown AssertionError (or a native stack overflow) fail the process with
// a non-zero exit code.
//
// No `printGraph` assertion: its column-width computation spreads the whole
// graph into one `Math.max(1, ...entries.map(...))` call, which alone blows
// this stack by ~16k nodes, well under this file's 40k-node graph (a
// stack-safety bug in src/compile.ts, not what this scenario tests).
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

{
  configure({ lazy: true })
  const out = chain(tensor([1, 2]), DEPTH)
  assert.ok(Math.abs(out.get(0) - expected) < 1e-2, `get(0): ${out.get(0)} vs ${expected}`)
  assert.ok(Math.abs(out.get(1) - 2 * expected) < 1e-2, `get(1): ${out.get(1)} vs ${2 * expected}`)
}

{
  configure({ lazy: true })
  const x = tensor([1, 2]).requiresGrad()
  chain(x as AnyTensor, DEPTH).sum().backward()
  assert.ok(Math.abs(x.grad!.get(0) - expected) < 1e-2, `grad(0): ${x.grad!.get(0)} vs ${expected}`)
}

{
  configure({ lazy: false })
  const x = tensor([1, 2]).requiresGrad()
  chain(x as AnyTensor, DEPTH).sum().backward()
  assert.ok(Math.abs(x.grad!.get(0) - expected) < 1e-2, `grad(0): ${x.grad!.get(0)} vs ${expected}`)
}

// shallower than the blocks above: compile() materializes the graph eagerly
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
