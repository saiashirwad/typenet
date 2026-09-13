/**
 * JS-side structural counters (PLAN-V2 §5A.2a).
 *
 * The Rust `counters()` can only see what reaches the addon. These see what
 * *doesn't*: every graph that fell off the native path, and why. That makes
 * "this model still runs native" a testable claim rather than an assumption,
 * which is the whole reason the fallback is loud — a silent interpreter
 * fallback on a transformer is a 30x slowdown that no gate would catch.
 *
 * `jsCounters()` is deliberately not re-exported from `index.ts` yet: in the
 * S1 sub-wave `index.ts` has a single writer (W1.10-A), so tests and benches
 * import this module directly. A-L1 adds the barrel export along with
 * `loweredNodes`/`wireNodes` when it lands the lowering half.
 */

export interface JsCounters {
  /** Graphs handed to the JS interpreter because the addon cannot run them. */
  nativeFallbacks: number
  /** The same count, split by the op that forced it. */
  fallbacksByOp: Record<string, number>
}

let nativeFallbacks = 0
let fallbacksByOp: Record<string, number> = {}
const notified = new Set<string>()

export function jsCounters(): JsCounters {
  return {
    nativeFallbacks,
    fallbacksByOp: { ...fallbacksByOp },
  }
}

export function resetJsCounters(): void {
  nativeFallbacks = 0
  fallbacksByOp = {}
  notified.clear()
}

/** Whether `TYPENET_STRICT_NATIVE=1` turns the notice below into a throw. */
function strictNative(): boolean {
  return (
    typeof process !== "undefined"
    && process.env?.TYPENET_STRICT_NATIVE === "1"
  )
}

/**
 * Record — and, once per distinct reason per process, print — a whole-graph
 * fallback to the JS interpreter.
 *
 * Once per reason, not once per call: a training loop would otherwise print
 * the same line ten thousand times, and a notice nobody reads is the same as
 * no notice. The counter still moves on every call, which is what tests
 * assert on.
 */
export function noteFallback(op: string, reason: string): void {
  nativeFallbacks++
  fallbacksByOp[op] = (fallbacksByOp[op] ?? 0) + 1
  if (strictNative()) {
    throw new Error(
      `typenet: TYPENET_STRICT_NATIVE=1 and this graph cannot run natively — ${reason}`,
    )
  }
  if (notified.has(reason)) return
  notified.add(reason)
  console.warn(
    `typenet: running this graph on the JS interpreter — ${reason}. `
      + "Set TYPENET_STRICT_NATIVE=1 to make this throw.",
  )
}
