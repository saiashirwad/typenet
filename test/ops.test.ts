import { readFileSync } from "node:fs"
import { join } from "node:path"
import { describe, expect, it } from "vitest"
import { OP_DESC } from "../src/ir.ts"
import { BINARY_OPS, NODE_OPS, UNARY_OPS } from "../src/ops.ts"

/**
 * The TS op tables and the Rust addon must list the same kinds. The
 * Rust side is handwritten (its lowering maps onto candle APIs, not a
 * scalar apply), so this test reads native/src/lib.rs and compares:
 * a kind added on one side fails here until the other side moves.
 */

const librs = readFileSync(
  join(__dirname, "..", "native", "src", "lib.rs"),
  "utf8",
)

function block(after: string): string {
  const start = librs.indexOf(after)
  expect(start, `found ${JSON.stringify(after)}`)
    .toBeGreaterThan(-1)
  const open = librs.indexOf("{", start)
  let depth = 0
  for (let i = open; i < librs.length; i++) {
    if (librs[i] === "{") depth++
    else if (librs[i] === "}") {
      depth--
      if (depth === 0) return librs.slice(open + 1, i)
    }
  }
  throw new Error("unbalanced braces")
}

const lowerFirst = (s: string) => s[0]!.toLowerCase() + s.slice(1)

describe("op kind lists match the Rust addon", () => {
  it("Node enum tags == OP_DESC keys + leaf", () => {
    const body = block("enum Node")
    const variants = [
      ...body.matchAll(/^\s{4}(\w+) \{/gm),
    ].map(m => lowerFirst(m[1]!))
    expect(variants.sort()).toEqual([...NODE_OPS].sort())
    expect(
      ["leaf", ...Object.keys(OP_DESC)].sort(),
    ).toEqual([...NODE_OPS].sort())
  })

  it("Bin::parse arms == BINARY_OPS", () => {
    const body = block("impl Bin")
    const arms = [
      ...body.matchAll(/"(\w+)" => Bin::/g),
    ].map(m => m[1]!)
    expect(arms.sort()).toEqual([...BINARY_OPS].sort())
  })

  it("Un::parse arms == UNARY_OPS", () => {
    const body = block("impl Un")
    const arms = [...body.matchAll(/"(\w+)" => Un::/g)].map(
      m => m[1]!,
    )
    expect(arms.sort()).toEqual([...UNARY_OPS].sort())
  })
})
