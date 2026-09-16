// Conv and pool spatial algebra against the real `src/shape.ts` exports. `ConvCheck`
// tests the span `H + 2P - K`, because hotscript's `Numbers.Div` truncates toward zero.
import { Linear } from "../src/nn/index.ts"
import type { ConvCheck, ConvOut, ErrorMessage, FlattenFrom, PoolOut, Shape } from "../src/shape.ts"
import type { Tensor } from "../src/tensor.ts"
import { CONV_FIT_FAIL_CASES } from "./shape-cases.ts"

type Equal<A, B> = (<T>() => T extends A ? 1 : 2) extends <T>() => T extends B ? 1 : 2 ? true : false
type Expect<T extends true> = T

// Every non-fitting row errors, including the rows whose truncated quotient looks like a legal 1-wide output.

type IsErrorMessage<T> = T extends ErrorMessage ? true : false
type XCase = typeof CONV_FIT_FAIL_CASES
type _tx0 = Expect<IsErrorMessage<ConvCheck<XCase[0]["h"], XCase[0]["k"], XCase[0]["s"], XCase[0]["p"]>>>
type _tx1 = Expect<IsErrorMessage<ConvCheck<XCase[1]["h"], XCase[1]["k"], XCase[1]["s"], XCase[1]["p"]>>>
type _tx2 = Expect<IsErrorMessage<ConvCheck<XCase[2]["h"], XCase[2]["k"], XCase[2]["s"], XCase[2]["p"]>>>
type _tx3 = Expect<IsErrorMessage<ConvCheck<XCase[3]["h"], XCase[3]["k"], XCase[3]["s"], XCase[3]["p"]>>>

// The trap spelled out: `ConvOut` reports 1 here (trunc(-1/2) + 1) where the true floor
// answer is 0, and `ConvCheck` errors anyway because it never looks at that number.
type _trapOutLooksLegal = Expect<Equal<ConvOut<4, 5, 2, 0>, 1>>
type _trapIsAnError = Expect<
  Equal<ConvCheck<4, 5, 2, 0>, ErrorMessage<"conv: kernel 5 with padding 0 does not fit a spatial extent of 4">>
>
type _trapIsNotOpen = Expect<Equal<Equal<ConvCheck<4, 5, 2, 0>, unknown>, false>>
type _poolTrapOutLooksLegal = Expect<Equal<PoolOut<2, 4, 4>, 1>>
type _poolTrapIsAnError = Expect<IsErrorMessage<ConvCheck<2, 4, 4, 0>>>

// A wide `number` is a wildcard, so the check answers `unknown` rather than failing.
type _wildOut = Expect<Equal<ConvOut<number, 3, 1, 0>, number>>
type _wildCheck = Expect<Equal<ConvCheck<number, 3, 1, 0>, unknown>>
type _wildKernel = Expect<Equal<ConvCheck<28, number, 1, 0>, unknown>>
type _wildStride = Expect<Equal<ConvCheck<28, 3, number, 0>, unknown>>
type _wildPad = Expect<Equal<ConvCheck<28, 3, 1, number>, unknown>>

declare class Conv2d<
  CIn extends number,
  COut extends number,
  K extends number,
  S extends number = 1,
  P extends number = 0,
> {
  constructor(cIn: CIn, cOut: COut, k: K, o?: { stride?: S; padding?: P; bias?: boolean })
  readonly weight: Tensor<[COut, CIn, K, K]>
  forward<B extends number, H extends number, W extends number>(
    x: Tensor<[B, CIn, H, W]> & ConvCheck<H, K, S, P, "height"> & ConvCheck<W, K, S, P, "width">,
  ): Tensor<[B, COut, ConvOut<H, K, S, P>, ConvOut<W, K, S, P>]>
}

declare class MaxPool2d<K extends number, S extends number = K> {
  constructor(k: K, o?: { stride?: S })
  forward<B extends number, C extends number, H extends number, W extends number>(
    x: Tensor<[B, C, H, W]> & ConvCheck<H, K, S, 0, "height"> & ConvCheck<W, K, S, 0, "width">,
  ): Tensor<[B, C, PoolOut<H, K, S>, PoolOut<W, K, S>]>
}

declare class ReLUish {
  forward<S extends Shape>(x: Tensor<S>): Tensor<S>
}

declare class Flatten {
  forward<S extends Shape>(x: Tensor<S>): Tensor<FlattenFrom<S>>
}

// A six-layer CNN chain, literal end to end, through the real `Linear`.
declare const mnist: Tensor<[64, 1, 28, 28]>

function _cnn() {
  const h1 = new ReLUish().forward(new Conv2d(1, 8, 3).forward(mnist))
  const p1 = new MaxPool2d(2).forward(h1)
  const h2 = new ReLUish().forward(new Conv2d(8, 16, 3).forward(p1))
  const p2 = new MaxPool2d(2).forward(h2)
  const flat = new Flatten().forward(p2)
  const out = new Linear(400, 10).forward(flat)
  type _1 = Expect<Equal<typeof h1.shape, [64, 8, 26, 26]>>
  type _2 = Expect<Equal<typeof p1.shape, [64, 8, 13, 13]>>
  type _3 = Expect<Equal<typeof h2.shape, [64, 16, 11, 11]>>
  type _4 = Expect<Equal<typeof p2.shape, [64, 16, 5, 5]>>
  type _5 = Expect<Equal<typeof flat.shape, [64, 400]>>
  type _6 = Expect<Equal<typeof out.shape, [64, 10]>>
  return { flat, h1, h2, out, p1, p2 }
}

// The other two shapes real models use: "same" padding and a strided block.
function _samePadding(x: Tensor<[8, 3, 32, 32]>) {
  const same = new Conv2d(3, 16, 3, { padding: 1 }).forward(x)
  type _1 = Expect<Equal<typeof same.shape, [8, 16, 32, 32]>>
  const strided = new Conv2d(16, 32, 3, { padding: 1, stride: 2 }).forward(same)
  type _2 = Expect<Equal<typeof strided.shape, [8, 32, 16, 16]>>
  return { same, strided }
}

// Trap (b): a conv inside a generic body over naked generic spatial dims. It must compile
// and still resolve to literals at instantiation, which needs the `IsExact` guards.
function _genericSpatialDims<B extends number, H extends number, W extends number>(t: Tensor<[B, 1, H, W]>) {
  return new Conv2d(1, 8, 3).forward(t)
}
type _g1 = Expect<Equal<ReturnType<typeof _genericSpatialDims<4, 28, 28>>, Tensor<[4, 8, 26, 26]>>>

// The same for pooling and for a whole generic block: the checks stay open down the chain.
function _genericBlock<B extends number, H extends number, W extends number>(t: Tensor<[B, 1, H, W]>) {
  const c = new Conv2d(1, 8, 3).forward(t)
  const p = new MaxPool2d(2).forward(c)
  return new Flatten().forward(p)
}
type _g2 = Expect<Equal<ReturnType<typeof _genericBlock<4, 28, 28>>, Tensor<[4, 1352]>>>

declare const tiny: Tensor<[64, 1, 2, 2]>
declare const four: Tensor<[64, 1, 4, 4]>

function _negatives() {
  // @ts-expect-error a 3-wide kernel does not fit a 2-wide input
  new Conv2d(1, 8, 3).forward(tiny)

  // @ts-expect-error a 5-wide kernel at stride 2 does not fit a 4-wide
  // input: the truncation trap, which a quotient-based check would let
  // through
  new Conv2d(1, 8, 5, { stride: 2 }).forward(four)

  const flat = new Flatten().forward(new MaxPool2d(2).forward(four))
  // @ts-expect-error [64, 4] does not matmul with [256, 10]
  new Linear(256, 10).forward(flat)
}

export { _cnn, _genericBlock, _genericSpatialDims, _negatives, _samePadding }
