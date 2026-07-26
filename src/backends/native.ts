import { createRequire } from "node:module"

/**
 * Optional Rust/candle backend (phase 2). The native addon is loaded
 * lazily via require() so the library keeps working when the .node
 * binary has not been built — a clear error is thrown only when
 * useNative() is called explicitly or a lazy graph is forced while
 * native mode is enabled.
 */

export type NativeModule = {
  evalGraph(graphJson: string, leaves: Float32Array): ArrayBuffer
  deviceName(): string
}

let moduleCache: NativeModule | null | undefined
let nativeEnabled = false

function loadNative(): NativeModule | null {
  if (moduleCache !== undefined) return moduleCache
  try {
    const require = createRequire(import.meta.url)
    moduleCache = require("@typenet/native") as NativeModule
  } catch {
    moduleCache = null
  }
  return moduleCache
}

/** True when the @typenet/native addon is built and loadable. */
export function isNativeAvailable(): boolean {
  return loadNative() !== null
}

/** Device the native backend evaluates on ("metal" or "cpu"). */
export function nativeDevice(): string | null {
  return loadNative()?.deviceName() ?? null
}

/**
 * Enable the native backend for lazy-graph evaluation. Throws when
 * the addon is not built; run `pnpm build:native` first. Only affects
 * lazy mode — eager execution is unchanged.
 */
export function useNative(): void {
  if (!loadNative())
    throw new Error(
      "@typenet/native is not built. Run `pnpm build:native` " +
        "(requires a Rust toolchain) before calling useNative()."
    )
  nativeEnabled = true
}

/** Disable the native backend (lazy graphs use the interpreter). */
export function disableNative(): void {
  nativeEnabled = false
}

export function isNativeEnabled(): boolean {
  return nativeEnabled && loadNative() !== null
}

/**
 * Evaluate a serialized lazy graph in one FFI hop. Internal — called
 * from force() in src/tensor.ts.
 */
export function evalGraphNative(
  graphJson: string,
  leaves: Float32Array
): Float32Array {
  const mod = loadNative()
  if (!mod)
    throw new Error(
      "@typenet/native is not built. Run `pnpm build:native`."
    )
  try {
    return new Float32Array(mod.evalGraph(graphJson, leaves))
  } catch (error) {
    throw new Error(
      `native backend: ${error instanceof Error ? error.message : String(error)}`
    )
  }
}
