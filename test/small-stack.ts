import { spawn } from "node:child_process"
import path from "node:path"
import { fileURLToPath } from "node:url"

const here = path.dirname(fileURLToPath(import.meta.url))

/** vite-node's CLI entry: scenarios are TypeScript and this project has no
 * build step, so a plain `node <file>.ts` can't run them. */
const viteNodeBin = path.resolve(here, "../node_modules/vite-node/dist/cli.mjs")

export interface SmallStackResult {
  /** Process exit code, or `null` if it was killed by a signal. */
  code: number | null
  stdout: string
  stderr: string
}

/** Runs `test/scenarios/<scenario>.ts` in a fresh Node process with a
 * small V8 stack (`--stack-size=256`). A 256 KB stack blows
 * on a few thousand JS frames, so an algorithm that recurses per node on
 * a deep graph fails here even though it passes under vitest's own stack. */
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
