import { spawn } from "node:child_process"
import path from "node:path"
import { fileURLToPath } from "node:url"

const here = path.dirname(fileURLToPath(import.meta.url))

/** vite-node's CLI entry: scenarios are TypeScript and this project has no
 * build step, so a plain `node <file>.ts` can't run them. Resolved once,
 * relative to this file, so it doesn't depend on the caller's cwd. */
const viteNodeBin = path.resolve(here, "../node_modules/vite-node/dist/cli.mjs")

export interface SmallStackResult {
  /** Process exit code, or `null` if it was killed by a signal. */
  code: number | null
  stdout: string
  stderr: string
}

/** Runs `test/scenarios/<scenario>.ts` to completion in a fresh Node process
 * with a deliberately small V8 stack (`--stack-size=256`), and reports its
 * exit code. This is the *one* place the small-stack command is spelled out
 * — callers pass just the scenario's basename.
 *
 * The point is to catch any algorithm on a deep graph that recurses per
 * node instead of working iteratively: a 256 KB stack blows on a few
 * thousand JS frames, well short of the default (much larger) stack every
 * other test in this file runs under, so a regression to recursion shows up
 * here even when it wouldn't under vitest's own stack. */
export function runOnSmallStack(scenario: string): Promise<SmallStackResult> {
  const scenarioPath = path.resolve(here, "scenarios", `${scenario}.ts`)
  return new Promise((resolve, reject) => {
    const child = spawn(
      process.execPath,
      ["--stack-size=256", viteNodeBin, scenarioPath],
      { stdio: ["ignore", "pipe", "pipe"] },
    )
    let stdout = ""
    let stderr = ""
    child.stdout.on("data", chunk => stdout += chunk)
    child.stderr.on("data", chunk => stderr += chunk)
    child.on("error", reject)
    child.on("close", code => resolve({ code, stdout, stderr }))
  })
}
