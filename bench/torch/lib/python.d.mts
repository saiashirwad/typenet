// Type declarations for python.mjs, consumed by bench/torch/run.ts (a .ts
// file importing a plain .mjs script — see python.mjs for the
// implementation, which is the single source of truth).

export type TorchProbe =
  | { ok: true; version: string; mps: boolean }
  | { ok: false; reason: string }

export interface RunTorchResult {
  status: number | null
  stdout: string
  stderr: string
  error: Error | undefined
}

/** The interpreter path this process will run every torch script under: `$TORCH_PYTHON`, or the default venv on this machine. */
export function torchPython(): string

/**
 * Spawns `torchPython() -c "import torch; ..."` once per process and caches
 * the answer. Never throws.
 */
export function hasTorch(): TorchProbe

/** Runs a torch bench script under `torchPython()`. */
export function runTorch(scriptPath: string, args?: readonly string[]): RunTorchResult

/** Test-only: drop the cached `hasTorch()` answer so a test can force a re-probe. */
export function _resetHasTorchCacheForTests(): void
