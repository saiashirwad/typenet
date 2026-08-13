import { createRequire } from "node:module"

export type NativeModule = {
  evalGraph(
    graphJson: string,
    leaves: Uint8Array,
    seed: number,
  ): ArrayBuffer
  prepareGraph(graphJson: string): number
  pinLeaf(
    handle: number,
    leaf: number,
    data: Uint8Array,
  ): void
  evalPrepared(
    handle: number,
    dirty: Uint8Array,
    dirtyIndex: Uint32Array,
    seed: number,
  ): ArrayBuffer
  sgemm(
    a: Float32Array,
    b: Float32Array,
    m: number,
    k: number,
    n: number,
  ): ArrayBuffer
  releaseGraph(handle: number): void
  preparedGraphCount(): number
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
 * Throws when the addon is not built; run `pnpm build:native` first.
 * Only affects lazy mode — eager execution is unchanged.
 */
export function useNative(
  options: { device?: "cpu" | "gpu" } = {},
): void {
  if (!loadNative()) {
    throw new Error(
      "@typenet/native is not built. Run `pnpm build:native` "
        + "(requires a Rust toolchain) before calling useNative().",
    )
  }
  // Each call fully specifies the configuration, so a plain useNative()
  // always means the default device and no earlier choice lingers.
  deviceMode = options.device ?? "cpu"
  nativeEnabled = true
}

export function disableNative(): void {
  nativeEnabled = false
}

export function isNativeEnabled(): boolean {
  return nativeEnabled && loadNative() !== null
}

/** Internal save/restore hooks for the context stack. */
export function _nativeState(): {
  enabled: boolean
  device: "cpu" | "gpu"
} {
  return { enabled: nativeEnabled, device: deviceMode }
}

export function _setNativeState(state: {
  enabled: boolean
  device: "cpu" | "gpu"
}): void {
  nativeEnabled = state.enabled
  deviceMode = state.device
}

const MISSING_ADDON = "@typenet/native is not built. Run `pnpm build:native`."

function withNative<T>(fn: (mod: NativeModule) => T): T
function withNative<T>(
  fn: (mod: NativeModule) => T,
  fallback: () => T,
): T
function withNative<T>(
  fn: (mod: NativeModule) => T,
  fallback?: () => T,
): T {
  const mod = loadNative()
  if (!mod) {
    if (fallback) return fallback()
    throw new Error(MISSING_ADDON)
  }
  try {
    return fn(mod)
  } catch (error) {
    throw new Error(
      `native backend: ${error instanceof Error ? error.message : String(error)}`,
    )
  }
}

/**
 * `seed` drives any random nodes in the graph; it is an argument rather
 * than part of the JSON so that replaying a graph keeps hitting the
 * same prepared plan.
 */
export function evalGraphNative(
  graphJson: string,
  leaves: Uint8Array,
  seed: number,
): Float32Array {
  return withNative(mod => new Float32Array(mod.evalGraph(graphJson, leaves, seed >>> 0)))
}

export function prepareGraphNative(
  graphJson: string,
): number {
  return withNative(mod => mod.prepareGraph(graphJson))
}

export function pinLeafNative(
  handle: number,
  leaf: number,
  data: Uint8Array,
): void {
  withNative(mod => mod.pinLeaf(handle, leaf, data))
}

export function evalPreparedNative(
  handle: number,
  dirty: Uint8Array,
  dirtyIndex: Uint32Array,
  seed: number,
): Float32Array {
  return withNative(mod => new Float32Array(mod.evalPrepared(handle, dirty, dirtyIndex, seed >>> 0)))
}

/** Null when the addon is not loaded (caller falls back to JS). */
export function sgemmNative(
  a: Float32Array,
  b: Float32Array,
  m: number,
  k: number,
  n: number,
): Float32Array | null {
  return withNative(
    mod => new Float32Array(mod.sgemm(a, b, m, k, n)),
    () => null,
  )
}

/** No-op when the addon is not loaded. */
export function releaseGraphNative(handle: number): void {
  withNative(mod => mod.releaseGraph(handle), () => undefined)
}

export function preparedGraphCountNative(): number {
  return withNative(mod => mod.preparedGraphCount(), () => 0)
}
