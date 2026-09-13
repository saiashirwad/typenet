// bench/torch/lib/python.mjs — the single place `$TORCH_PYTHON` is resolved
// for the whole repo (PLAN-V2 §4.2/§5A.4, W0.4).
//
// Nothing else in the repo may name a python interpreter — not the system
// interpreter on `PATH`, not a second default. The system interpreter on
// this machine has no torch installed, so falling back to it would produce
// a skip that *looks* like "no torch anywhere" and is not: torch 2.13.0
// with a working MPS backend is one env var away, at the path defaulted
// below. `bench/torch/run.ts` is the only importer today; W5.8's fixture
// generator (`scripts/regen-reference.mjs`) imports this module too, once
// it exists.

import { spawnSync } from "node:child_process"

const DEFAULT_TORCH_PYTHON = "/Users/texoport/code/graph-cellular-automata/.venv/bin/python"

/** The interpreter path this process will run every torch script under: `$TORCH_PYTHON`, or the default venv on this machine. */
export function torchPython() {
  return process.env.TORCH_PYTHON ?? DEFAULT_TORCH_PYTHON
}

/** @type {{ ok: true, version: string, mps: boolean } | { ok: false, reason: string } | undefined} */
let cachedAnswer

/**
 * Spawns `torchPython() -c "import torch; ..."` once per process and caches
 * the answer (repeated calls, even across many bench scripts in one
 * `run.ts` invocation, never re-spawn). Never throws: a missing
 * interpreter, a `TORCH_PYTHON` pointed at a file that isn't python, and a
 * python with no torch installed all come back as `{ ok: false, reason }`
 * rather than an exception, so a caller can print a loud one-line skip
 * naming both the variable and the path it tried instead of crashing.
 */
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

/**
 * Runs a torch bench script under `torchPython()` and returns its result —
 * never resolves an interpreter of its own beyond `torchPython()`.
 * @param {string} scriptPath
 * @param {readonly string[]} args
 */
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
