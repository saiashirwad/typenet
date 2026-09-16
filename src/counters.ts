/** JS-side counters for interpreter fallbacks; the Rust counters() cannot see them. */

export interface JsCounters {
  nativeFallbacks: number
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

function strictNative(): boolean {
  return (
    typeof process !== "undefined"
    && process.env?.TYPENET_STRICT_NATIVE === "1"
  )
}

/** Warns once per distinct reason. */
export function noteFallback(op: string, reason: string): void {
  nativeFallbacks++
  fallbacksByOp[op] = (fallbacksByOp[op] ?? 0) + 1
  if (strictNative()) {
    throw new Error(
      `typenet: TYPENET_STRICT_NATIVE=1 and this graph cannot run natively, ${reason}`,
    )
  }
  if (notified.has(reason)) return
  notified.add(reason)
  console.warn(
    `typenet: running this graph on the JS interpreter, ${reason}. `
      + "Set TYPENET_STRICT_NATIVE=1 to make this throw.",
  )
}
