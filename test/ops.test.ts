import { readFileSync } from "node:fs"
import { join } from "node:path"
import { describe, expect, it } from "vitest"
import { OP_DESC } from "../src/ir.ts"
import { NON_WIRE_OPS, SUPPORT, WIRE_OPS } from "../src/lower-native.ts"
import { BINARY_OPS, NODE_OPS, RANDOM_KINDS, REDUCE_OPS, UNARY_OPS } from "../src/ops.ts"

/**
 * The TS op tables and the Rust addon must list the same kinds. The Rust side is handwritten,
 * so this reads native/src/lib.rs and compares each table against its match arms, both ways.
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
  it("Node enum tags == WIRE_OPS", () => {
    const body = block("enum Node")
    const variants = [
      ...body.matchAll(/^\s{4}(\w+) \{/gm),
    ].map(m => lowerFirst(m[1]!))
    expect(variants.sort()).toEqual([...WIRE_OPS].sort())
  })

  it("NODE_OPS == leaf + OP_DESC keys", () => {
    expect(
      ["leaf", ...Object.keys(OP_DESC)].sort(),
    ).toEqual([...NODE_OPS].sort())
  })

  it("every non-wire kind is classified in SUPPORT", () => {
    expect(NON_WIRE_OPS.length).toBeGreaterThan(0)
    for (const op of NON_WIRE_OPS) {
      expect(SUPPORT[op], `SUPPORT entry for ${op}`)
        .toBeDefined()
      // Every non-wire kind must carry a reason a user can read.
      expect(SUPPORT[op].kind).toBe("unsupported")
      if (SUPPORT[op].kind === "unsupported") {
        expect(SUPPORT[op].reason).toContain(op)
      }
    }
  })

  it("WIRE_OPS is a subset of NODE_OPS", () => {
    for (const op of WIRE_OPS) {
      expect(NODE_OPS as readonly string[]).toContain(op)
      expect(SUPPORT[op].kind).toBe("native")
    }
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

  it("Reduce::parse arms == REDUCE_OPS", () => {
    const body = block("impl Reduce")
    const arms = [...body.matchAll(/"(\w+)" => Reduce::/g)].map(
      m => m[1]!,
    )
    expect(arms.sort()).toEqual([...REDUCE_OPS].sort())
  })

  it("random_data kinds == RANDOM_KINDS", () => {
    const body = block("fn random_data")
    const arms = [...body.matchAll(/"(\w+)" =>/g)].map(
      m => m[1]!,
    )
    expect(arms.sort()).toEqual([...RANDOM_KINDS].sort())
  })
})
