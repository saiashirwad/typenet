// The single place $TORCH_PYTHON is resolved for the whole repo. The system
// interpreter on PATH has no torch installed; the default below points at
// a venv with torch and a working MPS backend.

import { spawnSync } from "node:child_process"

const DEFAULT_TORCH_PYTHON = "/Users/texoport/code/graph-cellular-automata/.venv/bin/python"

/** The interpreter path this process will run every torch script under: `$TORCH_PYTHON`, or the default venv on this machine. */
export function torchPython() {
  return process.env.TORCH_PYTHON ?? DEFAULT_TORCH_PYTHON
}

/** @type {{ ok: true, version: string, mps: boolean } | { ok: false, reason: string } | undefined} */
let cachedAnswer

/** Probes once per process and caches the answer. Never throws: every
 * failure mode comes back as `{ ok: false, reason }` instead. */
export function hasTorch() {
  if (cachedAnswer !== undefined) return cachedAnswer
  const python = torchPython()
  const probe = "import torch; print(torch.__version__, torch.backends.mps.is_available())"
  const result = spawnSync(python, ["-c", probe], { encoding: "utf8" })

  if (result.error) {
    cachedAnswer = {
      ok: false,
      reason: `interpreter "${python}" could not be run (${result.error.message})`,
    }
    return cachedAnswer
  }
  if (result.status !== 0) {
    const lastStderrLine = (result.stderr ?? "").trim().split("\n").filter(Boolean).at(-1) ?? "no output"
    cachedAnswer = {
      ok: false,
      reason: `interpreter "${python}" has no working torch (exit ${result.status}: ${lastStderrLine})`,
    }
    return cachedAnswer
  }

  const parts = result.stdout.trim().split(/\s+/)
  const version = parts[0] ?? "unknown"
  const mps = parts[1] === "True"
  cachedAnswer = { ok: true, version, mps }
  return cachedAnswer
}

export function runTorch(scriptPath, args = []) {
  const python = torchPython()
  const result = spawnSync(python, [scriptPath, ...args], {
    encoding: "utf8",
    maxBuffer: 64 * 1024 * 1024,
  })
  return {
    status: result.status,
    stdout: result.stdout ?? "",
    stderr: result.stderr ?? "",
    error: result.error,
  }
}

/** Test-only: drop the cached `hasTorch()` answer so a test can force a re-probe. */
export function _resetHasTorchCacheForTests() {
  cachedAnswer = undefined
}
