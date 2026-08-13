import { _nativeState, _setNativeState } from "./backends/native.ts"
import { isLazyMode, setLazyMode } from "./ir.ts"
import { rngState, setRngState } from "./kernels.ts"

/**
 * The runtime knobs as one structured view, with save/patch/restore
 * scoping — the same stack pattern as `noGrad` / `eagerly`, not
 * AsyncLocalStorage (the library is synchronous). `configure()` mutates
 * the same underlying state as a script-level default; `withContext`
 * scopes a change to a callback and puts everything back.
 */
export interface RuntimeContext {
  lazy: boolean
  native: boolean
  device: "cpu" | "gpu"
  seed: number
  tracing: boolean
}

let tracing = false

/** True while compile() replays fn to record the graph. */
export function isTracing(): boolean {
  return tracing
}

export function context(): Readonly<RuntimeContext> {
  const native = _nativeState()
  return {
    lazy: isLazyMode(),
    native: native.enabled,
    device: native.device,
    seed: rngState().seed,
    tracing,
  }
}

export function withContext<T>(
  patch: Partial<RuntimeContext>,
  fn: () => T,
): T {
  const prevLazy = isLazyMode()
  const prevNative = _nativeState()
  const prevRng = rngState()
  const prevTracing = tracing
  if (patch.lazy !== undefined) setLazyMode(patch.lazy)
  if (
    patch.native !== undefined
    || patch.device !== undefined
  ) {
    _setNativeState({
      enabled: patch.native ?? prevNative.enabled,
      device: patch.device ?? prevNative.device,
    })
  }
  if (patch.seed !== undefined) {
    setRngState({
      seed: patch.seed >>> 0,
      stream: 0,
      active: prevRng.active,
    })
  }
  if (patch.tracing !== undefined) tracing = patch.tracing
  try {
    return fn()
  } finally {
    setLazyMode(prevLazy)
    _setNativeState(prevNative)
    setRngState(prevRng)
    tracing = prevTracing
  }
}
