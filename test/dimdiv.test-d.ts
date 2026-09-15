/**
 * `DimDiv` / `DimDivCheck`, against the real exports. Nothing here runs:
 * a `.test-d.ts` is a pure typecheck fixture and the `@ts-expect-error`s
 * are the assertions.
 *
 * Two ways to silently lose the check, both of which compile:
 *
 *   1. an intermediate constructor taking a plain `h: H` instead of
 *      `h: H & DimDivCheck<D, H>`;
 *   2. a forwarding site written `new MHA(d, h)` without explicit type
 *      arguments, letting inference re-widen `h`.
 */
import { DimDiv, DimMul } from "../src/shape.ts"
import type { DimDivCheck } from "../src/shape.ts"

type Equal<A, B> = (<T>() => T extends A ? 1 : 2) extends (<T>() => T extends B ? 1 : 2) ? true : false
type Expect<T extends true> = T

type _d1 = Expect<Equal<DimDiv<384, 6>, 64>>
type _d2 = Expect<Equal<DimDiv<12, 4>, 3>>
// a wide dim stays a wildcard rather than becoming an error
type _d3 = Expect<Equal<DimDiv<number, 6>, number>>
type _d4 = Expect<Equal<DimDiv<number, number>, number>>
// `/ 1` is the identity, and it reduces while the numerator is generic
type _d5 = Expect<Equal<DimDiv<384, 1>, 384>>

// The quotient truncates (hotscript's `Numbers.Div`, mirrored by
// `Math.trunc` in the value twin). Divisibility is `DimDivCheck`'s job,
// never the quotient's.
type _d6 = Expect<Equal<DimDiv<7, 2>, 3>>

function _identityReducesUnderAGeneric<D extends number>(d: D) {
  const same: D = DimDiv(d, 1)
  // the value twin carries the type: 384 / 6 is the literal 64
  const heads = DimDiv(384, 6)
  type _1 = Expect<Equal<typeof heads, 64>>
  return [same, heads] as const
}

declare class MHA<D extends number, H extends number> {
  constructor(d: D, h: H & DimDivCheck<D, H>)
  readonly headDim: DimDiv<D, H>
}

// The check is carried on the intermediate constructor's own parameter,
// and the forwarding site names its type arguments (see the header).
class Block<D extends number, H extends number> {
  readonly attn: MHA<D, H>
  constructor(d: D, h: H & DimDivCheck<D, H>) {
    this.attn = new MHA<D, H>(d, h)
  }
}

class Gpt<D extends number, H extends number> {
  readonly blocks: Block<D, H>[]
  constructor(d: D, h: H & DimDivCheck<D, H>, n: number) {
    this.blocks = Array.from({ length: n }, () => new Block<D, H>(d, h))
  }
}

const _ok = new Gpt(384, 6, 6)
// @ts-expect-error 384 is not divisible by 5
const _bad = new Gpt(384, 5, 6)

// A generic model width never trips the check: the remainder is a
// residual, not a nonzero literal (law 1, fail open).
function _genericWidth<D extends number, H extends number>(d: D, h: H & DimDivCheck<D, H>) {
  return new Gpt<D, H>(d, h, 4)
}

// ...and the wide `number` is a wildcard, as everywhere else.
function _wideWidth(d: number, h: number) {
  return new Gpt(d, h, 4)
}

const _headDim = new MHA(384, 6).headDim
type _h1 = Expect<Equal<typeof _headDim, 64>>

// The width a caller writes by hand and the one the algebra derives are
// the same type, in both directions.
function _roundTrip<D extends number, H extends number>() {
  const a: DimMul<H, DimDiv<D, H>> = null as any as DimMul<H, DimDiv<D, H>>
  return a
}

export { _bad, _genericWidth, _headDim, _identityReducesUnderAGeneric, _ok, _roundTrip, _wideWidth, Block, Gpt }
