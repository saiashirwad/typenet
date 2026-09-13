"use tsover"

/**
 * A typed GPT — §3.7 of the plan, on the pieces that exist today (W5.7b).
 *
 * WHY this example exists: an MLP's shapes are a chain of two numbers, and
 * a chain of two numbers proves very little. A transformer is where a
 * shape-typed library either earns its keep or does not — the head split
 * (`D -> H * (D/H)`), the position embedding broadcast (`[B,T,D] + [T,D]`),
 * the weight tie between a `[V,D]` table and a `D -> V` head, and a loss
 * that reads its target shape off the logits (`[B,T,V]` logits want
 * `[B,T]` targets, never `[B,T,V]` and never `[B*T]`). Every one of those
 * is a place where an untyped library lets a wrong tensor through and
 * reports it, if at all, as a wrong number some steps later.
 *
 * Here every one of them is a compile error at the line that causes it.
 * `forward` is generic in the batch size, so `GPT` is written once and
 * typechecks for every `B`; the only literals in the file are the ones the
 * config actually names. `_compileTimeErrors()` at the bottom is never
 * called — it exists so that four of those mistakes are written down and
 * *proved* to be rejected, since `@ts-expect-error` fails the build when
 * the error stops firing.
 *
 * The data is a short passage generated and encoded here, so the example
 * runs offline and a fixed `configure({ seed })` replays the whole loss
 * curve exactly.
 *
 * The loop is the plain `zeroGrad` / `backward` / `step` triple rather
 * than a compiled step, for `examples/mlp.ts`'s reason: `compile()` bakes
 * the optimizer's `lr` into the traced graph as a constant, and the
 * warmup-cosine schedule below has to move. Nothing else about the model
 * changes with that choice — `compileStep` lands on exactly this `forward`.
 */

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

// ---------------------------------------------------------------------------
// The model
// ---------------------------------------------------------------------------

/**
 * `GPT<V, T, D, H>`: vocabulary, context length, model width, heads.
 *
 * All four are type parameters rather than fields-of-type-`number`, so the
 * shapes inside `forward` are the shapes the constructor was given. `H`
 * carries `DimDivCheck<D, H>` on the constructor's own parameter (D20):
 * a head count that does not divide the model width is rejected here, at
 * the construction site, rather than inside `unflatten` at run time — and
 * the check is forwarded to `TransformerBlock` with EXPLICIT type
 * arguments, or inference would re-derive `H` from the intersection and
 * discharge the check against itself.
 */
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
  /** `0..T-1`, a buffer rather than a free variable: it rides along in
   * `stateDict()` and is never collected as a parameter. */
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
    // Weight tying: the head SHARES `wte`'s `[V, D]` buffer and transposes
    // it in the GEMM. `new Linear(d, vocab)` + `tie(head.weight,
    // wte.weight)` does NOT typecheck — `[D, V]` is not `[V, D]` — which
    // is exactly why `TiedLinear.of` exists. `parameters()` reports the
    // shared table once, so it is updated once per `step()`.
    this.head = TiedLinear.of(this.wte)
    this.pos = this.registerBuffer("pos", functional.arangeIndex(block))
    // GPT-2's own embedding init. `Embedding`'s default is PyTorch's
    // N(0, 1) per row, which through a TIED head makes the first logits
    // ~sqrt(D) wide and the first loss an order of magnitude above
    // ln(V) — the tie is exactly what couples the table's scale to the
    // output scale, so it is the model, not the layer, that has to say so.
    init.normal_(this.wte.weight, { std: 0.02 })
    init.normal_(this.wpe.weight, { std: 0.02 })
  }

  /**
   * `[B, T] token ids -> [B, T, V] logits`, generic in the batch size.
   *
   * `idx` is an {@link IndexTensor}, not a float tensor: `Embedding` takes
   * the branded index type, so passing activations where token ids belong
   * is a compile error rather than a lookup on `Math.round`ed floats.
   */
  forward<B extends number>(idx: IndexTensor<[B, T]>): Tensor<[B, T, V]> {
    // `[B, T, D] + [T, D]` — the position embedding broadcasts over the
    // batch, and the shape algebra says so: no `unsqueeze(0)`, no `expand`,
    // and a `wpe` built at the wrong width would not broadcast at all.
    let h: Tensor<[B, T, D]> = this.wte.forward(idx) + this.wpe.forward(this.pos)
    // The stack is a plain loop over a `ModuleList`, and the running
    // tensor keeps its type through every iteration because each block is
    // an endomorphism on `[B, T, D]`.
    for (const block of this.blocks) h = block.forward(h)
    return this.head.forward(this.lnf.forward(h))
  }
}

// ---------------------------------------------------------------------------
// The data
// ---------------------------------------------------------------------------
// A char-level corpus, written here so the example runs offline. The
// alphabet is a fixed literal set rather than "whatever characters the
// text happens to contain", because `V` has to be a literal type for
// `Embedding<V, D>` to carry it — and a vocabulary that silently changes
// size with the corpus is how a checkpoint stops loading.

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

// ---------------------------------------------------------------------------
// The configuration
// ---------------------------------------------------------------------------

const BLOCK = 32 // context length T
const D_MODEL = 64
const HEADS = 4 // 64 / 4 = 16 per head — derived, never written down
const LAYERS = 2
const BATCH = 16
// 20 steps is the run the README quotes; `test/examples-gpt.test.ts`
// overrides it so the smoke test exercises this file rather than a copy.
const STEPS = Number(process.env["TYPENET_EXAMPLE_STEPS"] ?? 20)

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

/**
 * One `[BATCH, BLOCK]` window of ids and the same window shifted by one —
 * the next-token objective, as two index tensors. Deterministic in `step`
 * so the curve replays.
 */
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

// ---------------------------------------------------------------------------
// The loop
// ---------------------------------------------------------------------------

const params = model.parameters()
console.log(
  `GPT: vocab ${VOCAB}, context ${BLOCK}, width ${D_MODEL}, `
    + `${HEADS} heads, ${LAYERS} layers — ${params.length} tensors, `
    + `${params.reduce((n, p) => n + p.numel, 0).toLocaleString("en-US")} parameters `
    + `(the ${VOCAB}x${D_MODEL} token table is counted once: the LM head shares it)`,
)

model.train()
const started = performance.now()
let first = 0
let last = 0

for (let step = 0; step < STEPS; step++) {
  const { x, y } = batchAt(step)

  const logits = model.forward(x) // Tensor<[16, 32, 32]>
  // `crossEntropy` reads the target shape off the logits: `[B, T, V]`
  // logits want `[B, T]` ids. Flattening either side by hand is not just
  // unnecessary, it is a compile error.
  const loss = crossEntropy(logits, y)

  opt.lr = schedule(step)
  opt.zeroGrad()
  loss.backward()
  clipGradNorm(params, 1)
  opt.step()

  last = loss.item()
  if (step === 0) first = last
  // Every step on the 20-step default (the whole curve is the output);
  // thinned out on a longer run so the log stays readable.
  const every = STEPS <= 25 ? 1 : 10
  if (step % every === 0 || step === STEPS - 1) {
    console.log(
      `step ${String(step).padStart(3)}  lr ${opt.lr.toExponential(2)}  loss ${last.toFixed(4)}`,
    )
  }
}

const elapsed = (performance.now() - started) / 1000

// `eval()` turns dropout off everywhere in the tree, so the reported
// number is the model, not a sample of it.
model.eval()
const held = batchAt(STEPS + 1)
const heldLoss = crossEntropy(model.forward(held.x), held.y).item()

console.log(
  `\nloss ${first.toFixed(4)} -> ${last.toFixed(4)} in ${STEPS} steps `
    + `(${elapsed.toFixed(1)}s, eager); held-out ${heldLoss.toFixed(4)}`,
)
// Eager mode never serialises a graph, so nothing can have fallen off the
// native path — printed rather than assumed, because "it is still native"
// is exactly the claim a silent fallback would make a lie of.
console.log(`native fallbacks: ${jsCounters().nativeFallbacks}`)

// ---------------------------------------------------------------------------
// The mistakes, written down
// ---------------------------------------------------------------------------
// Never called. Each line below is here to be REJECTED: `@ts-expect-error`
// fails the build if the error ever stops firing, so these four are a test
// of the type system that lives in the example it documents. The
// `// tsc(NNNN):` comment above each one quotes what the compiler actually
// prints — `test/examples-gpt.test.ts` strips the directives, recompiles
// this file, and checks the quotes against the real diagnostics, so the
// messages in the README are the messages a user sees.

function _compileTimeErrors(): void {
  // (1) 4 heads divide a width of 64; 5 do not. Caught at the construction
  //     site rather than inside the head split at run time.
  //
  //     The message is the one thing the shape algebra cannot carry here:
  //     `DimDivCheck<64, 5>` IS the sentence "attention: 64 is not
  //     divisible by 5", but a parameter typed `H & DimDivCheck<D, H>`
  //     intersects a numeric literal with a string one, and that is
  //     `never`. The rejection lands on the right line; the sentence shows
  //     on hover over `DimDivCheck`, not in the diagnostic.
  //
  // tsc(2322): Type 'number' is not assignable to type 'never'.
  // @ts-expect-error
  new GPT({ vocab: VOCAB, block: BLOCK, dModel: 64, heads: 5, layers: 2 })

  // (2) Token ids are branded. A float tensor of exactly the right shape is
  //     still not something an embedding table can be indexed by.
  //
  // tsc(2345): Argument of type 'Tensor<[16, 32]>' is not assignable to parameter of type 'IndexTensor<[16, 32]>'.
  // tsc(2345): Property '[INDEX]' is missing in type 'Tensor<[16, 32]>' but required in type '{ readonly [INDEX]: true; }'.
  // @ts-expect-error
  model.forward(Tensor.zeros([BATCH, BLOCK]))

  // (3) The context length is part of the model's type, so a window longer
  //     than the position embedding is a compile error and not a gather off
  //     the end of a table.
  //
  // tsc(2345): Argument of type 'IndexTensor<[16, 64]>' is not assignable to parameter of type 'IndexTensor<[16, 32]>'.
  // tsc(2345): Type '64' is not assignable to type '32'.
  // @ts-expect-error
  model.forward(Tensor.indices(new Array(BATCH * 64).fill(0), [BATCH, 64]))

  // (4) `[16, 32, 32]` logits want `[16, 32]` targets: the class axis is the
  //     one `crossEntropy` reduces, never one the caller supplies, and the
  //     target shape is read off the logits rather than restated.
  //
  // tsc(2345): Argument of type 'IndexTensor<[16]>' is not assignable to parameter of type 'IndexTensor<[16, 32]>'.
  // tsc(2345): Type '[16]' is not assignable to type '[16, 32]'.
  // @ts-expect-error
  crossEntropy(model.forward(batchAt(0).x), Tensor.indices(new Array(BATCH).fill(0), [BATCH]))
}

void _compileTimeErrors
