"use tsover"

import { describe, expect, it } from "vitest"
import { tensor, Tensor } from "../src/tensor.ts"
import type { TensorParams } from "../src/tensor.ts"
import { mseLoss } from "../src/nn.ts"

type Equal<A, B> =
  (<T>() => T extends A ? 1 : 2) extends (
    <T>() => T extends B ? 1 : 2
  ) ?
    true
  : false
type Expect<T extends true> = T

// operator-form cross-broadcasts resolve with precise result shapes
const _col = tensor([[1], [2]]) // Tensor<[2, 1]>
const _rowVec = tensor([[10, 20, 30]]) // Tensor<[1, 3]>
const _outer = _col + _rowVec // Tensor<[2, 3]>
type _op1 = Expect<Equal<typeof _outer.shape, [2, 3]>>

const _a = tensor([
  [1, 2, 3],
  [4, 5, 6]
]) // Tensor<[2, 3]>
const _mixedRank = _a + tensor([1, 2, 3]) // [2,3] + [3] -> [2,3]
type _op2 = Expect<Equal<typeof _mixedRank.shape, [2, 3]>>

const _scalarRhs = _a * 2 // Tensor<[2, 3]>
type _op3 = Expect<Equal<typeof _scalarRhs.shape, [2, 3]>>
const _scalarLhs = 1 - _a // Tensor<[2, 3]>
type _op4 = Expect<Equal<typeof _scalarLhs.shape, [2, 3]>>

// type-only: incompatible shapes must not resolve. Wrapped in an uncalled
// function so the @ts-expect-error is checked but the expression never runs.
function _incompatibleShapes() {
  // @ts-expect-error [2,1] and [3,2] do not broadcast
  _col +
    tensor([
      [1, 2],
      [3, 4],
      [5, 6]
    ])
}

describe("tsover operators", () => {
  const a = tensor([
    [1, 2, 3],
    [4, 5, 6]
  ])
  const row = tensor([10, 20, 30])

  it("+ - * / on same shapes", () => {
    expect((a + a).toArray()).toEqual([
      [2, 4, 6],
      [8, 10, 12]
    ])
    expect((a - a).sum().item()).toBe(0)
    expect((a * a).get(1, 2)).toBe(36)
    expect((a / a).get(0, 0)).toBe(1)
  })

  it("broadcasts rhs into lhs", () => {
    expect((a + row).toArray()).toEqual([
      [11, 22, 33],
      [14, 25, 36]
    ])
  })

  it("resolves on the left operand when lhs is smaller", () => {
    expect((row + a).toArray()).toEqual([
      [11, 22, 33],
      [14, 25, 36]
    ])
  })

  it("cross-broadcasts: [2,1] + [1,3] -> [2,3]", () => {
    const col = tensor([[1], [2]])
    const rowVec = tensor([[10, 20, 30]])
    const out = col + rowVec
    expect(out.shape).toEqual([2, 3])
    expect(out.toArray()).toEqual([
      [11, 21, 31],
      [12, 22, 32]
    ])
  })

  it("scalar operands on either side", () => {
    expect((a * 2).get(0, 1)).toBe(4)
    expect((1 - a).get(0, 0)).toBe(0)
    const recip = (12 / row).toArray()
    expect(recip[0]).toBeCloseTo(1.2)
    expect(recip[1]).toBeCloseTo(0.6)
    expect(recip[2]).toBeCloseTo(0.4)
  })

  it("** with scalar exponent", () => {
    expect((a ** 2).toArray()).toEqual([
      [1, 4, 9],
      [16, 25, 36]
    ])
  })

  it("compound expressions build autograd graphs", () => {
    const x = tensor([1, 2, 3]).requires_grad()
    const target = tensor([2, 2, 2])
    const loss = ((x - target) ** 2).mean()
    expect(loss.item()).toBeCloseTo(2 / 3)
    loss.backward()

    const g = x.grad!.toArray()
    expect(g[0]! * 3).toBeCloseTo(-2)
    expect(g[1]! * 3).toBeCloseTo(0)
    expect(g[2]! * 3).toBeCloseTo(2)
    expect(loss.item()).toBeCloseTo(
      mseLoss(x as any, target as any).item()
    )
  })

  it("outer combine: [A,1] + [1,A] -> [A,A]", () => {
    const col = tensor([[1], [2], [3]])
    const outer = col + col.T
    expect(outer.shape).toEqual([3, 3])
    expect(outer.toArray()).toEqual([
      [2, 3, 4],
      [3, 4, 5],
      [4, 5, 6]
    ])
  })

  it("operators resolve inside generic functions", () => {
    function maskedScores<
      N extends number,
      P extends TensorParams
    >(
      src: Tensor<[N, 1], P>,
      dst: Tensor<[N, 1], P>,
      adj: Tensor<[N, N], any>
    ): Tensor<[N, N], P> {
      const scores = src + dst.T
      return scores * adj + (1 - adj) * -1e9
    }
    const src = tensor([[1], [2]])
    const adj = tensor([
      [1, 1],
      [0, 1]
    ])
    const out = maskedScores(src, src, adj)
    expect(out.get(0, 0)).toBe(2)
    expect(out.get(0, 1)).toBe(3)
    expect(out.get(1, 0)).toBe(-1e9)
    expect(out.get(1, 1)).toBe(4)
  })

  it("plain number arithmetic is untouched", () => {
    expect(1 + 2).toBe(3)
    expect("a" + "b").toBe("ab")
  })
})
