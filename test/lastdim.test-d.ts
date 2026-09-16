// The `naked*` and `generic*` functions below are the regression trap: the variadic-infer
// spelling `S extends [...number[], infer L]` defers on a naked `S` and rejects every generic caller.
import type { LastDimCheck, Shape } from "../src/shape.ts"
import type { Tensor } from "../src/tensor.ts"

declare class LayerNormLike<D extends number> {
  forward<S extends Shape>(x: Tensor<S> & LastDimCheck<S, D>): Tensor<S>
}

// Law 1: a naked generic shape is accepted.
function _naked<S extends Shape>(x: Tensor<S>, n: LayerNormLike<16>) {
  return n.forward(x)
}

function _genericDims<B extends number, T extends number>(x: Tensor<[B, T, 16]>, n: LayerNormLike<16>) {
  return n.forward(x)
}

function _genericWidth<B extends number, T extends number, D extends number>(
  x: Tensor<[B, T, D]>,
  n: LayerNormLike<D>,
) {
  return n.forward(x)
}

// The dynamic shape is a wildcard, not an error.
function _dynamic(x: Tensor<number[]>, n: LayerNormLike<16>) {
  return n.forward(x)
}

declare const lit: Tensor<[4, 8, 384]>
const _ok = new LayerNormLike<384>().forward(lit)
// @ts-expect-error the last axis is 384, not 128
const _bad = new LayerNormLike<128>().forward(lit)

declare const vec: Tensor<[384]>
const _okRank1 = new LayerNormLike<384>().forward(vec)
// @ts-expect-error rank-1 input, still the wrong width
const _badRank1 = new LayerNormLike<128>().forward(vec)

export { _bad, _badRank1, _dynamic, _genericDims, _genericWidth, _naked, _ok, _okRank1 }
