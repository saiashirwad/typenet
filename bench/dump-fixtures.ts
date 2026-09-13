// Serialises six representative compiled-graph JSONs into
// `test/fixtures/graphs/*.json` (PLAN-V2 §4.2, W0.2): the round-trip
// corpus W1.1 reads and the seed corpus W2.5's fuzzer starts from. Every
// fixture is built with ordinary tensor ops under `configure({ lazy:
// true })` and dumped through `serializeLazyGraph` -- nothing here
// hand-writes JSON, per the item's own instruction.
//
// "MLP fwd+bwd+Adam" is the one fixture that can't just call `.backward()`
// and `Adam.step()`: outside a `compile()` trace, both force-evaluate
// immediately (the same "eager under the hood" behavior every other bench
// script in this file relies on), which would collapse the graph to
// leaves before it could be serialized. So that fixture writes out the
// same math by hand -- an analytic backward pass for this exact 2-layer
// net, then one Adam update per parameter -- entirely in ordinary lazy
// tensor ops. It is not wired to any real training loop; it exists to
// give the wire format a realistic mix of op kinds (matmul, relu, gt,
// pow, sum, sqrt, div, ...) over multiple roots.

import { mkdirSync, writeFileSync } from "node:fs"
import { join } from "node:path"
import { configure, rand, Tensor, tensor, zeros } from "../index.ts"
import { serializeLazyGraph } from "../src/lazy.ts"

type AnyTensor = Tensor<any>

const OUT_DIR = join(process.cwd(), "test", "fixtures", "graphs")

function dump(name: string, roots: AnyTensor[]): void {
  const serialized = serializeLazyGraph(roots)
  if (!serialized) {
    throw new Error(
      `dump-fixtures: "${name}" produced no lazy graph -- every root was already a concrete leaf`,
    )
  }
  const parsed = JSON.parse(serialized.json) as { nodes: unknown[]; roots: unknown[] }
  if (!Array.isArray(parsed.nodes) || parsed.nodes.length === 0) {
    throw new Error(`dump-fixtures: "${name}" serialized with an empty node list`)
  }
  if (!Array.isArray(parsed.roots) || parsed.roots.length === 0) {
    throw new Error(`dump-fixtures: "${name}" serialized with an empty root list`)
  }
  mkdirSync(OUT_DIR, { recursive: true })
  writeFileSync(join(OUT_DIR, `${name}.json`), serialized.json)
  console.log(
    `wrote test/fixtures/graphs/${name}.json (${parsed.nodes.length} nodes, ${parsed.roots.length} roots)`,
  )
}

// 1. MLP forward: Linear(784,256) -> relu -> Linear(256,10), one root.
function buildMlpForward(): AnyTensor[] {
  const x = rand([64, 784]) as AnyTensor
  const w1 = rand([784, 256]) as AnyTensor
  const b1 = zeros([256]) as AnyTensor
  const w2 = rand([256, 10]) as AnyTensor
  const b2 = zeros([10]) as AnyTensor
  const pred = x.matmul(w1).add(b1).relu().matmul(w2).add(b2)
  return [pred]
}

// 2. MLP forward + backward + Adam, by hand (see the file banner for why).
function buildMlpForwardBackwardAdam(): AnyTensor[] {
  const batch = 64
  const x = rand([batch, 784]) as AnyTensor
  const y = rand([batch, 10]) as AnyTensor
  const w1 = rand([784, 256]) as AnyTensor
  const b1 = zeros([256]) as AnyTensor
  const w2 = rand([256, 10]) as AnyTensor
  const b2 = zeros([10]) as AnyTensor

  const z1 = x.matmul(w1).add(b1)
  const h = z1.relu()
  const pred = h.matmul(w2).add(b2)
  const diff = pred.sub(y)
  const loss = diff.pow(2).mean()

  // Analytic backward for this exact graph: mseLoss = mean((pred-y)^2).
  const dpred = diff.mul(2 / pred.numel)
  const dW2 = h.transpose(0, 1).matmul(dpred)
  const db2 = dpred.sum(0)
  const dh = dpred.matmul(w2.transpose(0, 1))
  const dz1 = dh.mul(z1.gt(0)) // relu backward mask
  const dW1 = x.transpose(0, 1).matmul(dz1)
  const db1 = dz1.sum(0)

  const beta1 = 0.9
  const beta2 = 0.999
  const lr = 1e-3
  const eps = 1e-8
  // One Adam step from zeroed (m, v) state, i.e. t = 1.
  const adamStep = (p: AnyTensor, g: AnyTensor): AnyTensor => {
    const m = zeros(p.shape as number[]).mul(beta1).add(g.mul(1 - beta1))
    const v = zeros(p.shape as number[]).mul(beta2).add(g.mul(g).mul(1 - beta2))
    const mHat = m.div(1 - beta1)
    const vHat = v.div(1 - beta2)
    return p.sub(mHat.div(vHat.sqrt().add(eps)).mul(lr))
  }

  return [loss, adamStep(w1, dW1), adamStep(b1, db1), adamStep(w2, dW2), adamStep(b2, db2)]
}

// 3. An elementwise chain, mixing arithmetic with a transcendental.
function buildElementwiseChain(): AnyTensor[] {
  const a = rand([4096]) as AnyTensor
  const b = rand([4096]) as AnyTensor
  const c = rand([4096]) as AnyTensor
  let t = a.mul(b).add(c).tanh()
  t = t.relu().sub(a).mul(2).sigmoid()
  t = t.add(b).mul(c).pow(2).sqrt()
  return [t]
}

// 4. A reduce chain: sum / mean / max over different axes, recombined.
function buildReduceChain(): AnyTensor[] {
  const a = rand([32, 128]) as AnyTensor
  const sum0 = a.sum(0)
  const mean1 = a.mean(1)
  const maxAll = a.max()
  const combined = sum0.sum().add(mean1.mean()).add(maxAll)
  return [combined, sum0, mean1]
}

// 5. A gather/scatter pair at embedding scale: indexSelect then the
// exact-reverse scatterAdd.
function buildGatherScatterPair(): AnyTensor[] {
  const table = rand([65, 384]) as AnyTensor
  const ids = tensor([1, 4, 4, 7, 12, 30, 30, 63]).to("int32") as AnyTensor
  const gathered = table.indexSelect(ids)
  const scattered = gathered.scatterAdd(ids, 65)
  return [gathered, scattered]
}

// 6. A 4-D batched matmul: [B,H,T,K] @ [B,H,K,K2], as attention projects.
function buildBatchedMatmul(): AnyTensor[] {
  const a = rand([2, 3, 4, 8]) as AnyTensor
  const b = rand([2, 3, 8, 5]) as AnyTensor
  return [a.matmul(b)]
}

function main(): void {
  configure({ lazy: true })
  dump("mlp-forward", buildMlpForward())
  dump("mlp-forward-backward-adam", buildMlpForwardBackwardAdam())
  dump("elementwise-chain", buildElementwiseChain())
  dump("reduce-chain", buildReduceChain())
  dump("gather-scatter-pair", buildGatherScatterPair())
  dump("batched-matmul-4d", buildBatchedMatmul())
  configure({ lazy: false })
}

main()
