// Run under runOnSmallStack("deep-chain-4000-mixed"): a depth-4000 chain mixing elementwise ops with
// views and a reduction on a 256 KB stack. Each non-elementwise op is value-preserving, so the closed form holds.
import assert from "node:assert/strict"
import { tensor } from "../../src/factories.ts"
import { configure } from "../../src/lazy.ts"
import { Tensor } from "../../src/tensor.ts"

type AnyTensor = Tensor<any>

const DEPTH = 4000
const N = 4

function mixedChain(x: AnyTensor, depth: number): AnyTensor {
  let h = x
  for (let i = 0; i < depth; i++) {
    h = h.mul(1.0001).add(0)
    switch (i % 4) {
      case 1:
        h = h.view([1, N])
        break
      case 2:
        h = h.transpose(0, 1).transpose(0, 1)
        break
      case 3:
        h = h.sum(0, true)
        break
        // case 0: plain elementwise, no extra op
    }
  }
  return h
}

const start = [1, 2, 3, 4]
const expected = 1.0001 ** DEPTH

function checkShapeAndValues(out: AnyTensor) {
  assert.deepEqual(out.shape, [1, N])
  for (let j = 0; j < N; j++) {
    const want = start[j]! * expected
    const got = out.get(0, j)
    assert.ok(Math.abs(got - want) < 1e-1, `get(0,${j}): ${got} vs ${want}`)
  }
}

{
  configure({ lazy: true })
  const out = mixedChain(tensor([start]), DEPTH)
  checkShapeAndValues(out)
}

{
  configure({ lazy: true })
  const x = tensor([start]).requiresGrad()
  mixedChain(x as AnyTensor, DEPTH).sum().backward()
  for (let j = 0; j < N; j++) {
    const got = x.grad!.get(0, j)
    assert.ok(Math.abs(got - expected) < 1e-1, `grad(0,${j}): ${got} vs ${expected}`)
  }
}

{
  configure({ lazy: false })
  const x = tensor([start]).requiresGrad()
  mixedChain(x as AnyTensor, DEPTH).sum().backward()
  for (let j = 0; j < N; j++) {
    const got = x.grad!.get(0, j)
    assert.ok(Math.abs(got - expected) < 1e-1, `grad(0,${j}): ${got} vs ${expected}`)
  }
}

configure({ lazy: false })
console.log("ok")
