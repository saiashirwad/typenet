import { execFileSync } from "node:child_process"
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs"
import { tmpdir } from "node:os"
import { basename, dirname, resolve } from "node:path"
import { fileURLToPath } from "node:url"
import { describe, expect, it } from "vitest"

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..")

function bin(name: string): string {
  return resolve(root, "node_modules", ".bin", process.platform === "win32" ? `${name}.cmd` : name)
}

/** The example's own 20-step run, shared: it is a ~15 s child process, and lazy so no timeout is dodged. */
let cached: { out: string; losses: number[] } | null = null

function trainingRun(): { out: string; losses: number[] } {
  if (!cached) {
    let out = ""
    try {
      out = execFileSync(bin("vite-node"), ["examples/gpt.ts"], {
        cwd: root,
        encoding: "utf8",
        stdio: ["ignore", "pipe", "pipe"],
      })
    } catch (err) {
      const e = err as { stdout?: string; stderr?: string }
      throw new Error(`examples/gpt.ts failed:\n${e.stderr ?? ""}\n${e.stdout ?? ""}`)
    }
    cached = {
      out,
      losses: [...out.matchAll(/^step\s+\d+\s+lr \S+\s+loss (\d+\.\d+)$/gm)].map(m => Number(m[1])),
    }
  }
  return cached
}

describe("examples/gpt.ts", () => {
  it("trains for 20 steps, starting at ln(V)", () => {
    const { out, losses } = trainingRun()
    expect(losses).toHaveLength(20)
    // An untrained model over a 32-symbol alphabet costs ln(32) = 3.4657 per token, so
    // starting anywhere else means the init or the tie is wrong.
    expect(losses[0]!).toBeGreaterThan(3.3)
    expect(losses[0]!).toBeLessThan(3.6)
    // Not just the endpoints: a curve that dips once and then diverges does not pass.
    const mean = (xs: number[]) => xs.reduce((a, b) => a + b, 0) / xs.length
    expect(mean(losses.slice(10))).toBeLessThan(mean(losses.slice(0, 10)))
    // 28 tensors, not 29: the [32, 64] table is the LM head's weight too, and parameters()
    // dedups it by storage identity.
    expect(out).toContain("28 tensors")
    expect(out).toContain("104,192 parameters")
    // Eager mode serialises no graph, so this is trivially true until the example compile()s.
    expect(out).toContain("native fallbacks: 0")
  }, 300_000)
})

/**
 * `pnpm typecheck` already proves every @ts-expect-error fires, since tsc reports an unused one.
 * This proves the converse: with the directives gone, each of those lines is still an error, so
 * none of them is load-bearing for some unrelated reason.
 */
function errorLinesWithoutDirectives(example: string): { directives: number[]; errors: number[] } {
  const source = readFileSync(resolve(root, example), "utf8")
  const lines = source.split("\n")
  const directives = lines.flatMap((line, i) => /^\s*\/\/ @ts-expect-error/.test(line) ? [i + 1] : [])

  const dir = mkdtempSync(resolve(tmpdir(), "typenet-example-"))
  try {
    const file = resolve(dir, basename(example))
    // Commented out rather than deleted, so the line numbers still line up.
    writeFileSync(
      file,
      source
        .replaceAll(/^(\s*)\/\/ @ts-expect-error.*$/gm, "$1// (stripped)")
        .replaceAll(/from "(\.\.\/)+index\.ts"/g, `from ${JSON.stringify(resolve(root, "index.ts"))}`),
    )
    let output = ""
    try {
      execFileSync(bin("tsc"), [
        "--noEmit",
        "--strict",
        "--target",
        "esnext",
        "--module",
        "esnext",
        "--moduleResolution",
        "bundler",
        "--allowImportingTsExtensions",
        "--moduleDetection",
        "force",
        "--skipLibCheck",
        "--lib",
        "esnext,dom",
        file,
      ], { cwd: root, encoding: "utf8", stdio: ["ignore", "pipe", "pipe"] })
    } catch (e) {
      const err = e as { stdout?: string; stderr?: string }
      output = `${err.stdout ?? ""}${err.stderr ?? ""}`
    }
    const errors = [...output.matchAll(/\.ts\((\d+),\d+\): error TS\d+:/g)].map(m => Number(m[1]))
    return { directives, errors }
  } finally {
    rmSync(dir, { recursive: true, force: true })
  }
}

describe.each(["examples/shapes.ts", "examples/gpt.ts"])("%s's compile-time errors", example => {
  it("every stripped @ts-expect-error leaves a real diagnostic behind", () => {
    const { directives, errors } = errorLinesWithoutDirectives(example)
    expect(directives.length).toBeGreaterThan(0)
    for (const line of directives) {
      // The rejected expression starts on the next line and may wrap onto the one after it.
      expect(errors.some(e => e > line && e <= line + 2), `${example}:${line}: nothing errors under it`).toBe(true)
    }
  }, 300_000)
})

describe("the README's GPT walkthrough", () => {
  it("is compiled by check-readme (no skip marker)", () => {
    const readme = readFileSync(resolve(root, "README.md"), "utf8")
    expect(readme).not.toMatch(/check-readme: skip, this block is examples\/gpt\.ts/)
  })
})
