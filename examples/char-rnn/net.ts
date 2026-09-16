"use tsover"

// A one-layer character RNN (Karpathy's min-char-rnn). `scan` and `Sequence` own the time loop.

import { categorical, crossEntropy, Embedding, type IndexTensor, Linear, Module, Rnn, scan, Sequence, Tensor } from "../../index.ts"

export interface Timestep<B extends number, E extends number, H extends number> {
  readonly x: Tensor<[B, E]>
  readonly hidden: Tensor<[B, H]>
}

export interface CharacterRNNConfig<V extends number, E extends number, H extends number> {
  readonly vocab: V
  readonly embed: E
  readonly hidden: H
  /** Timesteps per training batch. Sampling runs past it freely; only a batch is cut to it. */
  readonly unroll: number
}

/** The model plus the tokenizer it was built against, since the two must agree on the alphabet. */
export class CharacterRNN<
  V extends number,
  E extends number,
  H extends number,
> extends Module {
  readonly config: CharacterRNNConfig<V, E, H>
  readonly embed: Embedding<V, E>
  readonly rnn: Rnn<E, H>
  readonly head: Linear<H, V>
  readonly alphabet: readonly string[]
  private readonly codes = new Map<string, number>()

  constructor(config: CharacterRNNConfig<V, E, H>, alphabet: string) {
    super()
    this.config = config
    this.alphabet = [...alphabet]
    if (this.alphabet.length !== config.vocab) {
      throw new Error(
        `the alphabet holds ${this.alphabet.length} characters, but vocab says ${config.vocab}`,
      )
    }
    this.alphabet.forEach((c, i) => this.codes.set(c, i))
    this.embed = new Embedding(config.vocab, config.embed)
    this.rnn = new Rnn(config.embed, config.hidden)
    this.head = new Linear(config.hidden, config.vocab)
  }

  encode(text: string): IndexTensor<[number]> {
    const codes: number[] = []
    for (const c of text) {
      const code = this.codes.get(c)
      if (code !== undefined) codes.push(code)
    }
    return Tensor.indices(codes, [codes.length])
  }

  decode(codes: ArrayLike<number>): string {
    let text = ""
    for (let i = 0; i < codes.length; i++) text += this.alphabet[codes[i]!] ?? ""
    return text
  }

  zeroHidden<B extends number>(batch: B): Tensor<[B, H]> {
    return Tensor.zeros([batch, this.config.hidden])
  }

  /** One timestep: the previous state and this step's characters in, next-character logits out. */
  stepOne<B extends number>(
    step: Timestep<B, E, H>,
  ): { logits: Tensor<[B, V]>; hidden: Tensor<[B, H]> } {
    const hidden = this.rnn.forward(step.x, { state: step.hidden })
    return { logits: this.head.forward(hidden), hidden }
  }

  /** The next-character logits over `idx`, and the state left after the last step. */
  forward<B extends number, T extends number>(
    idx: IndexTensor<[B, T]>,
  ): { logits: Tensor<[B, T, V]>; hidden: Tensor<[B, H]> } {
    const embedded = this.embed.forward(idx) // [B, T, E]
    const run = scan<B, [B, H], [B, V]>(this.zeroHidden(idx.shape[0]!), idx.shape[1]!, (hidden, t) => {
      const step = this.stepOne({ x: embedded.select(1, t), hidden })
      return { output: step.logits, state: step.hidden }
    })
    // `T` is the caller's and the loop bound is the same number; the loop itself can only report a count.
    return { logits: run.outputs as Tensor<[B, T, V]>, hidden: run.state }
  }

  /** Mean cross-entropy over a window: step `t` of `idx` is trained against `next[:, t]`. */
  lossOn<B extends number>(
    idx: IndexTensor<[B, number]>,
    next: IndexTensor<[B, number]>,
  ): Tensor<[]> {
    return crossEntropy(this.forward(idx).logits, next)
  }

  /**
   * Reads `prompt`, then samples `length` more characters. The prompt is context, not output: the
   * result is `prompt` followed by the continuation, and the caller trims it.
   */
  generate(
    prompt: IndexTensor<[number]>,
    options: { length: number; temperature: number; rng: () => number },
  ): IndexTensor<[number]> {
    const { length: promptLength } = prompt.shape as [number]
    if (promptLength < 1) {
      throw new Error("generate() needs a prompt of at least one character to start the recurrence")
    }
    let seq = prompt
    // One sample is one sequence, so the batch is 1. The prompt only warms the state.
    let run = Sequence.of<1, [1, H]>(this.zeroHidden(1))
    for (let i = 0; i < promptLength - 1; i++) {
      run = run.advance(this.stepOne({ x: this.embedOne(seq.get(i)), hidden: run.state }).hidden)
    }
    for (let i = 0; i < options.length; i++) {
      const step = this.stepOne({ x: this.embedOne(seq.get(-1)), hidden: run.state })
      run = run.step(step.logits, step.hidden)
      seq = Tensor.cat(
        seq,
        categorical(step.logits, { temperature: options.temperature, rng: options.rng }),
      ).toIndex()
    }
    return seq
  }

  private embedOne(code: number): Tensor<[1, E]> {
    return this.embed.forward(Tensor.indices([code], [1]))
  }
}
