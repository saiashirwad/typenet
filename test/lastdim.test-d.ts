/**
 * `LastDimCheck`, against the real export.
 *
 * Checked-in version of `scratchpad/final-verify/lastdim.ts`, reduced to
 * the adopted spelling. The probe compared it against a two-sided tuple
 * escape on `Last<S>`; both pass every case below, and `DimEq` was kept
 * because it is the comparison the rest of `shape.ts` already uses.
 *
 * What must NOT be used is the variadic-infer spelling
 * `S extends [...number[], infer L] ? … : ErrorMessage<…>`
 * (`scratchpad/final-verify/failopen.ts`): it defers on a naked generic
 * `S` with an `ErrorMessage` in reach and rejects every generic caller.
 * The three `naked*` / `generic*` functions below are that regression —
 * they are not decoration, they fail to compile the moment the check is
 * rewritten that way.
 */
import type { LastDimCheck, Shape } from "../src/shape.ts"
import type { Tensor } from "../src/tensor.ts"

declare class LayerNormLike<D extends number> {
  forward<S extends Shape>(x: Tensor<S> & LastDimCheck<S, D>): Tensor<S>
}

// law 1: a naked generic shape is accepted
function _naked<S extends Shape>(x: Tensor<S>, n: LayerNormLike<16>) {
  return n.forward(x)
}

// generic dims with a known arity are accepted
function _genericDims<B extends number, T extends number>(x: Tensor<[B, T, 16]>, n: LayerNormLike<16>) {
  return n.forward(x)
}

// ...including when the feature width itself is the generic
function _genericWidth<B extends number, T extends number, D extends number>(
  x: Tensor<[B, T, D]>,
  n: LayerNormLike<D>,
) {
  return n.forward(x)
}

// the dynamic shape is a wildcard, not an error
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
