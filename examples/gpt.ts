"use tsover"

import {
  AdamW,
  clipGradNorm,
  configure,
  crossEntropy,
  type DimDivCheck,
  Embedding,
  functional,
  type IndexTensor,
  init,
  jsCounters,
  LayerNorm,
  Module,
  ModuleList,
  Tensor,
  TiedLinear,
  TransformerBlock,
  warmupCosine,
} from "../index.ts"

// H carries DimDivCheck<D, H>, and the explicit type arguments on TransformerBlock below keep
// inference from re-deriving H and discharging that check against itself.
class GPT<
  V extends number,
  T extends number,
  D extends number,
  H extends number,
> extends Module {
  readonly wte: Embedding<V, D>
  readonly wpe: Embedding<T, D>
  readonly blocks: ModuleList<TransformerBlock<D, H>>
  readonly lnf: LayerNorm<D>
  readonly head: TiedLinear<D, V>
  // A buffer, so it rides along in stateDict() and is never collected as a parameter.
  readonly pos: IndexTensor<[T]>

  constructor(cfg: {
    vocab: V
    block: T
    dModel: D
    heads: H & DimDivCheck<D, H>
    layers: number
    dropout?: number
  }) {
    super()
    const { vocab, block, dModel: d, heads: h, layers } = cfg
    const p = cfg.dropout ?? 0
    this.wte = new Embedding(vocab, d)
    this.wpe = new Embedding(block, d)
    this.blocks = ModuleList.of(
      layers,
      () => new TransformerBlock<D, H>(d, h, { causal: true, dropout: p }),
    )
    this.lnf = new LayerNorm(d)
    // The head shares wte's [V, D] table, which `parameters()` reports once, and transposes it
    // in the GEMM. `new Linear(d, vocab)` would need a [D, V] weight, so `TiedLinear.of` exists.
    this.head = TiedLinear.of(this.wte)
    this.pos = this.registerBuffer("pos", functional.arangeIndex(block))
    // GPT-2's embedding std: with a tied head, the default N(0, 1) would start the loss well above ln(V).
    init.normal_(this.wte.weight, { std: 0.02 })
    init.normal_(this.wpe.weight, { std: 0.02 })
  }

  forward<B extends number>(idx: IndexTensor<[B, T]>): Tensor<[B, T, V]> {
    let h: Tensor<[B, T, D]> = this.wte.forward(idx) + this.wpe.forward(this.pos)
    for (const block of this.blocks) h = block.forward(h)
    return this.head.forward(this.lnf.forward(h))
  }
}

const ALPHABET = "abcdefghijklmnopqrstuvwxyz ,.;'\n"
const VOCAB = 32
if (ALPHABET.length !== VOCAB) {
  throw new Error(`the alphabet holds ${ALPHABET.length} characters, but VOCAB says ${VOCAB}`)
}

const CORPUS = `the shape is the type and the type is the shape.
a tensor knows how wide it is before it knows what it holds;
the compiler counts the axes, the kernel only moves the bytes.
what cannot be written down cannot be run, and what can be run
was written down first, in a tuple of literal numbers.
`.repeat(24)

const codeOf = new Map([...ALPHABET].map((c, i) => [c, i]))
const encoded = [...CORPUS.toLowerCase()].flatMap(c => {
  const code = codeOf.get(c)
  return code === undefined ? [] : [code]
})

const BLOCK = 32
const D_MODEL = 64
const HEADS = 4
const LAYERS = 2
const BATCH = 16
const STEPS = 20

configure({ seed: 1234 })

const model = new GPT({
  vocab: VOCAB,
  block: BLOCK,
  dModel: D_MODEL,
  heads: HEADS,
  layers: LAYERS,
  dropout: 0.1,
})

const opt = new AdamW(model.parameters(), { lr: 3e-3, weightDecay: 0.1, betas: [0.9, 0.95] })
const schedule = warmupCosine({ base: 3e-3, warmupSteps: 5, totalSteps: STEPS })

function batchAt(step: number): {
  x: IndexTensor<[typeof BATCH, typeof BLOCK]>
  y: IndexTensor<[typeof BATCH, typeof BLOCK]>
} {
  const xs: number[] = []
  const ys: number[] = []
  for (let b = 0; b < BATCH; b++) {
    const start = ((step * BATCH + b) * 37) % (encoded.length - BLOCK - 1)
    for (let t = 0; t < BLOCK; t++) {
      xs.push(encoded[start + t]!)
      ys.push(encoded[start + t + 1]!)
    }
  }
  return {
    x: Tensor.indices(xs, [BATCH, BLOCK]),
    y: Tensor.indices(ys, [BATCH, BLOCK]),
  }
}

const params = model.parameters()
console.log(
  `GPT: vocab ${VOCAB}, context ${BLOCK}, width ${D_MODEL}, `
    + `${HEADS} heads, ${LAYERS} layers, ${params.length} tensors, `
    + `${params.reduce((n, p) => n + p.numel, 0).toLocaleString("en-US")} parameters `
    + `(the ${VOCAB}x${D_MODEL} token table is counted once: the LM head shares it)`,
)

model.train()
const started = performance.now()
let first = 0
let last = 0

for (let step = 0; step < STEPS; step++) {
  const { x, y } = batchAt(step)

  const loss = crossEntropy(model.forward(x), y)

  opt.lr = schedule(step)
  opt.zeroGrad()
  loss.backward()
  clipGradNorm(params, 1)
  opt.step()

  last = loss.item()
  if (step === 0) first = last
  console.log(
    `step ${String(step).padStart(3)}  lr ${opt.lr.toExponential(2)}  loss ${last.toFixed(4)}`,
  )
}

const elapsed = (performance.now() - started) / 1000

model.eval()
const held = batchAt(STEPS + 1)
const heldLoss = crossEntropy(model.forward(held.x), held.y).item()

console.log(
  `\nloss ${first.toFixed(4)} -> ${last.toFixed(4)} in ${STEPS} steps `
    + `(${elapsed.toFixed(1)}s, eager); held-out ${heldLoss.toFixed(4)}`,
)
console.log(`native fallbacks: ${jsCounters().nativeFallbacks}`)

/** Four rejections, never run: `pnpm typecheck` is what executes them. */
export function compileTimeErrors(): void {
  // 5 does not divide dModel 64
  // @ts-expect-error
  new GPT({ vocab: VOCAB, block: BLOCK, dModel: 64, heads: 5, layers: 2 })

  // floats where the embedding wants indices
  // @ts-expect-error
  model.forward(Tensor.zeros([BATCH, BLOCK]))

  // a 64-long context into a model built for 32
  // @ts-expect-error
  model.forward(Tensor.indices(new Array(BATCH * 64).fill(0), [BATCH, 64]))

  // [B] of targets against [B, T, V] of logits
  // @ts-expect-error
  crossEntropy(model.forward(batchAt(0).x), Tensor.indices(new Array(BATCH).fill(0), [BATCH]))
}
