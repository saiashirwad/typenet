import { createRequire } from "node:module"

/**
 * Optional Rust/candle backend (phase 2). The native addon is loaded
 * lazily via require() so the library keeps working when the .node
 * binary has not been built — a clear error is thrown only when
 * useNative() is called explicitly or a lazy graph is forced while
 * native mode is enabled.
 */

export type NativeModule = {
  evalGraph(
    graphJson: string,
    leaves: Float32Array,
    seed: number
  ): ArrayBuffer
  prepareGraph(graphJson: string): number
  evalPrepared(
    handle: number,
    leaves: Float32Array,
    seed: number
  ): ArrayBuffer
  releaseGraph(handle: number): void
  deviceName(): string
}

let moduleCache: NativeModule | null | undefined
let nativeEnabled = false
let deviceMode: "cpu" | "gpu" = "cpu"

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

/** Best accelerator the addon found ("metal" or "cpu"), whether used or not. */
export function nativeDevice(): string | null {
  return loadNative()?.deviceName() ?? null
}

/** Which device non-tiny graphs currently run on. */
export function nativeDeviceMode(): "cpu" | "gpu" {
  return deviceMode
}

/**
 * Enable the native backend for lazy-graph evaluation. Throws when the
 * addon is not built; run `pnpm build:native` first. Only affects lazy
 * mode — eager execution is unchanged.
 *
 * `device` picks what non-tiny graphs run on: `"cpu"` (the default) is
 * candle's CPU device, which on macOS uses Accelerate for matmul;
 * `"gpu"` is the best accelerator available. CPU is the default because
 * it wins on the graph shapes typenet produces — see `pickTarget` in
 * src/tensor.ts for the measurements. Reach for `"gpu"` when a workload
 * is dominated by large elementwise tensors.
 */
export function useNative(
  options: { device?: "cpu" | "gpu" } = {}
): void {
  if (!loadNative())
    throw new Error(
      "@typenet/native is not built. Run `pnpm build:native` " +
        "(requires a Rust toolchain) before calling useNative()."
    )
  // Each call fully specifies the configuration, so a plain useNative()
  // always means the default device and no earlier choice lingers.
  deviceMode = options.device ?? "cpu"
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
 * from force() in src/tensor.ts. `seed` drives any random nodes in the
 * graph; it is an argument rather than part of the JSON so that
 * replaying a graph keeps hitting the same prepared plan.
 */
export function evalGraphNative(
  graphJson: string,
  leaves: Float32Array,
  seed: number
): Float32Array {
  const mod = loadNative()
  if (!mod)
    throw new Error(
      "@typenet/native is not built. Run `pnpm build:native`."
    )
  try {
    return new Float32Array(
      mod.evalGraph(graphJson, leaves, seed >>> 0)
    )
  } catch (error) {
    throw new Error(
      `native backend: ${error instanceof Error ? error.message : String(error)}`
    )
  }
}

/**
 * Parse and plan a graph once, returning a handle to evaluate it by.
 *
 * `compile()` replays one graph thousands of times; going through
 * `evalGraphNative` every call would ship the whole JSON across the FFI
 * boundary and hash it just to find the plan again, and for a rolled-out
 * automaton that string is hundreds of kilobytes.
 */
export function prepareGraphNative(
  graphJson: string
): number {
  const mod = loadNative()
  if (!mod)
    throw new Error(
      "@typenet/native is not built. Run `pnpm build:native`."
    )
  try {
    return mod.prepareGraph(graphJson)
  } catch (error) {
    throw new Error(
      `native backend: ${error instanceof Error ? error.message : String(error)}`
    )
  }
}

/** Evaluate a graph prepared by {@link prepareGraphNative}. */
export function evalPreparedNative(
  handle: number,
  leaves: Float32Array,
  seed: number
): Float32Array {
  const mod = loadNative()
  if (!mod)
    throw new Error(
      "@typenet/native is not built. Run `pnpm build:native`."
    )
  try {
    return new Float32Array(
      mod.evalPrepared(handle, leaves, seed >>> 0)
    )
  } catch (error) {
    throw new Error(
      `native backend: ${error instanceof Error ? error.message : String(error)}`
    )
  }
}

/** Release a prepared graph. */
export function releaseGraphNative(handle: number): void {
  loadNative()?.releaseGraph(handle)
}
