// PLAN-V2 §4.1/§4.2, W0.7 — gate C3. See `test/loss-curve.ts` for the
// harness itself; this file only exercises its contract.
import { describe, expect, it } from "vitest"
import { isNativeAvailable } from "../src/backends/native.ts"
import { assertKnownEnvSwitches, expectIdenticalCurves, KNOWN_ENV_SWITCHES, lossCurve } from "./loss-curve.ts"

const available = isNativeAvailable()

// Each `lossCurve()` call spawns a fresh `vite-node` process (required so a
// `TYPENET_*` switch, read once at addon init, actually takes effect) —
// module load alone costs several seconds, well past vitest's default 5s
// test timeout, hence the generous timeouts below.
const SPAWN_TIMEOUT_MS = 60_000

describe.skipIf(!available)("loss curve", () => {
  it(
    "two runs of the same configuration are bit-identical",
    () => {
      const a = lossCurve({ steps: 5 })
      const b = lossCurve({ steps: 5 })
      expectIdenticalCurves(a, b, "same-config runs")
    },
    SPAWN_TIMEOUT_MS * 2,
  )

  it(
    "a different seed produces a different curve (the harness isn't trivially constant)",
    () => {
      const a = lossCurve({ steps: 5, seed: 1234 })
      const b = lossCurve({ steps: 5, seed: 5678 })
      expect(a).not.toEqual(b)
    },
    SPAWN_TIMEOUT_MS * 2,
  )

  it(
    "runs end to end under a real §2.9 kill switch without throwing",
    () => {
      const curve = lossCurve({ steps: 5, env: { TYPENET_NO_FUSION: "1" } })
      expect(curve.length).toBe(5)
      for (const bits of curve) {
        const value = new Float32Array(new Uint32Array([bits]).buffer)[0]!
        expect(Number.isFinite(value), `loss bit pattern ${bits} decodes to a non-finite value`).toBe(true)
      }
    },
    SPAWN_TIMEOUT_MS,
  )
})

describe("expectIdenticalCurves", () => {
  it("passes for two equal curves", () => {
    const a = Uint32Array.from({ length: 200 }, (_, i) => i)
    const b = Uint32Array.from({ length: 200 }, (_, i) => i)
    expect(() => expectIdenticalCurves(a, b, "equal")).not.toThrow()
  })

  it("fails loudly on a one-ulp difference at step 137", () => {
    const f32ToBits = (v: number) => new Uint32Array(new Float32Array([v]).buffer)[0]!
    const base = Uint32Array.from({ length: 200 }, (_, i) => f32ToBits(Math.sin(i)))
    const perturbed = Uint32Array.from(base)
    // Flip the least-significant bit of the f32 mantissa at step 137: the
    // smallest possible representable difference (one ulp).
    perturbed[137] = perturbed[137]! ^ 1
    expect(() => expectIdenticalCurves(base, perturbed, "one-ulp-at-137")).toThrow(/step 137/)
  })

  it("fails loudly on curves of different lengths", () => {
    const a = Uint32Array.from([1, 2, 3])
    const b = Uint32Array.from([1, 2])
    expect(() => expectIdenticalCurves(a, b, "length-mismatch")).toThrow(/length/)
  })
})

describe("lossCurve env validation", () => {
  it("throws on an unknown env switch name instead of running the default configuration", () => {
    expect(() => lossCurve({ steps: 1, env: { TYPENET_NO_ARENAA: "1" } })).toThrow(/unknown env switch/)
    expect(() => lossCurve({ steps: 1, env: { NOT_EVEN_PREFIXED: "1" } })).toThrow(/unknown env switch/)
  })

  it("accepts every switch KNOWN_ENV_SWITCHES declares (pure validation, no process spawned)", () => {
    for (const key of KNOWN_ENV_SWITCHES) {
      expect(() => assertKnownEnvSwitches({ [key]: "1" })).not.toThrow()
    }
  })
})
