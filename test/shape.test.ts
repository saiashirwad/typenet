import { describe, expect, it } from "vitest"
import {
  broadcastShapes,
  broadcastToShape,
  catShape,
  ConvOut,
  DimDiv,
  flattenFrom,
  flattenShape,
  matmulShape,
  permuteShape,
  PoolOut,
  reduceShape,
  resizeDim,
  resolveView,
  sliceShape,
  unflattenShape,
} from "../src/shape.ts"
import {
  BROADCAST_CASES,
  BROADCAST_FAIL_CASES,
  BROADCAST_TO_CASES,
  BROADCAST_TO_FAIL_CASES,
  CAT_CASES,
  CAT_FAIL_CASES,
  CONV_CASES,
  CONV_FIT_FAIL_CASES,
  DIM_DIV_CASES,
  FLATTEN_CASES,
  FLATTEN_FAIL_CASES,
  FLATTEN_FROM_CASES,
  MATMUL_CASES,
  MATMUL_FAIL_CASES,
  PERMUTE_CASES,
  POOL_CASES,
  REDUCE_CASES,
  RESIZE_CASES,
  SLICE_CASES,
  SLICE_FAIL_CASES,
  UNFLATTEN_CASES,
  UNFLATTEN_FAIL_CASES,
  VIEW_CASES,
  VIEW_FAIL_CASES,
} from "./shape-cases.ts"

describe("runtime shape functions agree with the case table", () => {
  it.each(BROADCAST_CASES)("broadcast %j", c => {
    expect(broadcastShapes(c.a, c.b)).toEqual(c.out)
    expect(broadcastShapes(c.b, c.a)).toEqual(c.out)
  })

  it.each(BROADCAST_FAIL_CASES)("broadcast fail %j", c => {
    expect(() => broadcastShapes(c.a, c.b)).toThrow(
      /Cannot broadcast/,
    )
  })

  it.each(MATMUL_CASES)("matmul %j", c => {
    expect(matmulShape(c.a, c.b)).toEqual(c.out)
  })

  it.each(MATMUL_FAIL_CASES)("matmul fail %j", c => {
    expect(() => matmulShape(c.a, c.b)).toThrow(
      /inner dimensions do not match/,
    )
  })

  it.each(VIEW_CASES)("view %j", c => {
    expect(resolveView(c.s, c.v)).toEqual(c.out)
  })

  it.each(VIEW_FAIL_CASES)("view fail %j", c => {
    expect(() => resolveView(c.s, c.v)).toThrow(/Cannot view/)
  })

  it.each(CAT_CASES)("cat %j", c => {
    expect(catShape(c.a, c.b, c.dim)).toEqual(c.out)
  })

  it.each(CAT_FAIL_CASES)("cat fail %j", c => {
    expect(() => catShape(c.a, c.b, c.dim)).toThrow(/cat: /)
  })

  it.each(RESIZE_CASES)("resize %j", c => {
    expect(resizeDim(c.s, c.dim, c.length)).toEqual(c.out)
  })

  it.each(SLICE_CASES)("slice %j", c => {
    expect(sliceShape(c.s, c.spec)).toEqual(c.out)
  })

  it.each(PERMUTE_CASES)("permute %j", c => {
    expect(permuteShape(c.s, c.order)).toEqual(c.out)
  })

  it.each(REDUCE_CASES)("reduce %j", c => {
    expect(reduceShape(c.s, c.dim, c.keepdim)).toEqual(c.out)
  })

  it.each(BROADCAST_TO_CASES)("broadcastTo %j", c => {
    expect(broadcastToShape(c.from, c.to)).toEqual(c.to)
  })

  it.each(BROADCAST_TO_FAIL_CASES)(
    "broadcastTo fail %j",
    c => {
      expect(() => broadcastToShape(c.from, c.to)).toThrow(
        /is not a broadcast of/,
      )
    },
  )
})

describe("runtime shape functions: flatten, unflatten, slice ranges, DimDiv", () => {
  it.each(SLICE_FAIL_CASES)("slice fail %j", c => {
    expect(() => sliceShape(c.s, c.spec)).toThrow(/slice: /)
  })

  it.each(FLATTEN_CASES)("flatten %j", c => {
    expect(flattenShape(c.s, c.from, c.to)).toEqual(c.out)
  })

  it.each(FLATTEN_FAIL_CASES)("flatten fail %j", c => {
    expect(() => flattenShape(c.s, c.from, c.to)).toThrow(
      /flatten\(\)|out of range|start dim is after/,
    )
  })

  it.each(UNFLATTEN_CASES)("unflatten %j", c => {
    expect(unflattenShape(c.s, c.dim, c.sizes)).toEqual(c.out)
  })

  it.each(UNFLATTEN_FAIL_CASES)("unflatten fail %j", c => {
    expect(() => unflattenShape(c.s, c.dim, c.sizes)).toThrow(
      /unflatten\(|out of range/,
    )
  })

  // Explicit `<number, number>`: `c.a` and `c.b` are unions of every
  // literal in the table, and letting hotscript's `Div` distribute over
  // that product is a TS2589. The literal arithmetic is asserted row by
  // row in types.test-d.ts; this loop checks the runtime twin.
  it("DimDiv agrees with the table at runtime", () => {
    for (const c of DIM_DIV_CASES) {
      expect(DimDiv<number, number>(c.a, c.b)).toBe(c.out)
    }
  })
})

describe("conv shapes: the spatial ladder, the wildcards and the truncation trap", () => {
  // Explicit `<number, number, number, number>` throughout, for the same
  // reason as the DimDiv loop above: under `it.each` the case fields are
  // unions of every literal in the table, and letting hotscript's
  // arithmetic distribute over that cross product costs a quarter of a
  // million instantiations.
  it("ConvOut agrees with the table at runtime", () => {
    for (const c of CONV_CASES) {
      expect(ConvOut<number, number, number, number>(c.h, c.k, c.s, c.p)).toBe(c.out)
    }
  })

  it("PoolOut agrees with the table at runtime, and is ConvOut at padding 0", () => {
    for (const c of POOL_CASES) {
      expect(PoolOut<number, number, number>(c.h, c.k, c.s)).toBe(c.out)
      expect(PoolOut<number, number, number>(c.h, c.k, c.s)).toBe(
        ConvOut<number, number, number, number>(c.h, c.k, c.s, 0),
      )
    }
  })

  it("the two-block MNIST ladder reduces 28 -> 26 -> 13 -> 11 -> 5", () => {
    const a = ConvOut(28, 3, 1, 0)
    const b = PoolOut(a, 2, 2)
    const c = ConvOut(b, 3, 1, 0)
    const d = PoolOut(c, 2, 2)
    expect([a, b, c, d]).toEqual([26, 13, 11, 5])
    // the head width the Linear after Flatten has to be built with;
    // `flattenFrom<number[]>` keeps the runtime assertion from
    // re-deriving the literal type.
    expect(flattenFrom<number[]>([64, 16, d, d])).toEqual([64, 400])
  })

  it("flattenFrom agrees with the table at runtime", () => {
    for (const c of FLATTEN_FROM_CASES) {
      expect(flattenFrom<number[]>([...c.s])).toEqual(c.out)
    }
  })

  // Why ConvCheck tests the SPAN and not the quotient: Math.trunc (and
  // hotscript's Numbers.Div, which the type twin uses) truncate toward
  // zero, so a kernel that does not fit still reports a plausible output.
  it("a kernel that does not fit still produces a number, which is why ConvCheck tests the span", () => {
    for (const c of CONV_FIT_FAIL_CASES) {
      expect(c.h + 2 * c.p - c.k).toBe(c.span)
      expect(c.span).toBeLessThan(0)
      expect(ConvOut<number, number, number, number>(c.h, c.k, c.s, c.p)).toBe(c.out)
    }
  })

  it("the truncation trap, in the value world", () => {
    // a 5-wide kernel on a 4-wide input at stride 2 reads as a legal 1-wide
    // output, because trunc(-0.5) is 0 where floor(-0.5) is -1
    expect(ConvOut(4, 5, 2, 0)).toBe(1)
    expect(Math.floor((4 - 5) / 2) + 1).toBe(0)
    // the span, which is what ConvCheck looks at, has no such hole
    expect(4 + 2 * 0 - 5).toBeLessThan(0)
  })
})
