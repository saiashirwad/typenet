import { describe, expect, it } from "vitest"
import { broadcastShapes, broadcastToShape, catShape, matmulShape, permuteShape, reduceShape, resizeDim, resolveView } from "../src/shape.ts"
import {
  BROADCAST_CASES,
  BROADCAST_FAIL_CASES,
  BROADCAST_TO_CASES,
  BROADCAST_TO_FAIL_CASES,
  CAT_CASES,
  CAT_FAIL_CASES,
  MATMUL_CASES,
  MATMUL_FAIL_CASES,
  PERMUTE_CASES,
  REDUCE_CASES,
  RESIZE_CASES,
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
