import { execFileSync } from "node:child_process"
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs"
import { tmpdir } from "node:os"
import { dirname, resolve } from "node:path"
import { fileURLToPath } from "node:url"
import { describe, expect, it } from "vitest"

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..")
const examplePath = resolve(root, "examples/gpt.ts")
const exampleSource = readFileSync(examplePath, "utf8")

function bin(name: string): string {
  return resolve(root, "node_modules", ".bin", process.platform === "win32" ? `${name}.cmd` : name)
}

function run(command: string, args: string[], env: Record<string, string> = {}): string {
  try {
    return execFileSync(command, args, {
      cwd: root,
      env: { ...process.env, ...env },
      encoding: "utf8",
      stdio: ["ignore", "pipe", "pipe"],
    })
  } catch (err) {
    const e = err as { stdout?: string; stderr?: string; message?: string }
    throw new Error(
      `${command} ${args.join(" ")} failed:\n${e.stderr ?? e.message ?? String(err)}\n${e.stdout ?? ""}`,
    )
  }
}

/**
 * Shared example output, run lazily from inside a test: it is a ~15 s child process, and a
 * collection-time cost is one no test timeout covers.
 */
let cachedRun: { out: string; losses: number[] } | null = null

function trainingRun(): { out: string; losses: number[] } {
  if (cachedRun) return cachedRun
  // Steps passed explicitly so this measures 20 steps, not whatever the default happens to be.
  const out = run(bin("vite-node"), ["examples/gpt.ts"], { TYPENET_EXAMPLE_STEPS: "20" })
  const losses = [...out.matchAll(/^step\s+\d+\s+lr \S+\s+loss (\d+\.\d+)$/gm)]
    .map(m => Number(m[1]))
  cachedRun = { out, losses }
  return cachedRun
}

describe("examples/gpt.ts", () => {
  it("trains for 20 steps", () => {
    expect(trainingRun().losses).toHaveLength(20)
  }, 300_000)

  it("starts at ln(V) and the loss falls", () => {
    const { losses } = trainingRun()
    // An untrained model over a 32-symbol alphabet costs ln(32) = 3.4657 per token, so
    // starting anywhere else means the init or the tie is wrong.
    expect(losses[0]!).toBeGreaterThan(3.3)
    expect(losses[0]!).toBeLessThan(3.6)
    expect(losses.at(-1)!).toBeLessThan(losses[0]!)
    // Not just the endpoints: the second half must be below the first, so a curve that
    // dips once and then diverges does not pass.
    const mean = (xs: number[]) => xs.reduce((a, b) => a + b, 0) / xs.length
    expect(mean(losses.slice(10))).toBeLessThan(mean(losses.slice(0, 10)))
  }, 300_000)

  it("reports the tied token table once", () => {
    // 28 tensors, not 29: the [32, 64] table is the LM head's weight too, and
    // parameters() dedups it by storage identity.
    expect(trainingRun().out).toContain("28 tensors")
    expect(trainingRun().out).toContain("104,192 parameters")
  }, 300_000)

  it("never falls off the native path", () => {
    // Eager mode serialises no graph, so this is trivially true today, but it fails
    // loudly if the example moves to compile().
    expect(trainingRun().out).toContain("native fallbacks: 0")
  }, 300_000)
})

describe("examples/gpt.ts is cast-free", () => {
  it("no escape hatch appears in the example", () => {
    const banned = /\bas (any|unknown|never)\b|assertChecked|AnyTensor/
    exampleSource.split("\n").forEach((line, i) => {
      expect(banned.test(line), `examples/gpt.ts:${i + 1}: ${line.trim()}`).toBe(false)
    })
  })
})

type ErrorCase = {
  code: number
  quoted: string
  directiveLine: number
  /** 1-based line where the next case's quote begins, or the end of file. */
  endLine: number
}

/** Every ErrorMessage ends with U+200B, which is invisible in a comment and must not be part of the comparison. */
function normalize(text: string): string {
  return text.replaceAll("​", "").replaceAll(/\s+/g, " ").trim()
}

/** A quoted diagnostic split into its first line and each elaboration under it, because tsc prints them separately. */
function fragmentsOf(quoted: string): string[] {
  return quoted.split(/(?<=\.)\s+(?=[A-Z])/).filter(f => f.trim() !== "")
}

function parseCases(source: string): ErrorCase[] {
  const lines = source.split("\n")
  const cases: ErrorCase[] = []
  for (let i = 0; i < lines.length; i++) {
    const first = /^\s*\/\/ tsc\((\d+)\):\s?(.*)$/.exec(lines[i]!)
    if (!first) continue
    const code = Number(first[1])
    const parts = [first[2]!]
    let j = i + 1
    for (; j < lines.length; j++) {
      const more = /^\s*\/\/ tsc\(\d+\):\s?(.*)$/.exec(lines[j]!)
      if (!more) break
      parts.push(more[1]!)
    }
    expect(
      /^\s*\/\/ @ts-expect-error\s*$/.test(lines[j] ?? ""),
      `examples/gpt.ts:${j + 1}: a // tsc(...) quote with no @ts-expect-error under it`,
    ).toBe(true)
    cases.push({ code, quoted: parts.join(" "), directiveLine: j + 1, endLine: lines.length })
    i = j
  }
  for (let k = 0; k < cases.length - 1; k++) cases[k]!.endLine = cases[k + 1]!.directiveLine - 1
  return cases
}

type Diagnostic = { line: number; code: number; message: string }

/**
 * Compiles the example with every @ts-expect-error removed, in a temp dir. Directives become a
 * comment rather than being deleted so line numbers still line up; `raw` is the whole compiler output.
 */
function compileWithoutDirectives(): { diagnostics: Diagnostic[]; raw: string } {
  const dir = mkdtempSync(resolve(tmpdir(), "typenet-gpt-"))
  try {
    const file = resolve(dir, "gpt.ts")
    const stripped = exampleSource
      .replaceAll(/^(\s*)\/\/ @ts-expect-error.*$/gm, "$1// (directive stripped)")
      .replaceAll(/from "\.\.\//g, () => `from "${root}/`)
    writeFileSync(file, stripped)
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
    const diagnostics = output
      .split("\n")
      .flatMap(line => {
        const m = /^.*gpt\.ts\((\d+),\d+\): error TS(\d+): (.*)$/.exec(line)
        return m ? [{ line: Number(m[1]), code: Number(m[2]), message: m[3]! }] : []
      })
    return { diagnostics, raw: output }
  } finally {
    rmSync(dir, { recursive: true, force: true })
  }
}

describe("examples/gpt.ts's compile errors", () => {
  const cases = parseCases(exampleSource)
  const { diagnostics, raw } = compileWithoutDirectives()
  const compilerSaid = normalize(raw)

  it("is a gallery of exactly four compile-time errors", () => {
    expect(cases).toHaveLength(4)
  })

  it.each(cases.map((c, i) => [i + 1, c] as const))(
    "case %i fires with the code it claims, on its own line",
    (_index, testCase) => {
      const found = diagnostics.find(d =>
        d.code === testCase.code
        && d.line > testCase.directiveLine
        && d.line <= testCase.endLine
      )
      expect(
        found,
        `no TS${testCase.code} between examples/gpt.ts:${testCase.directiveLine} and :${testCase.endLine}`
          + `, got ${JSON.stringify(diagnostics)}`,
      ).toBeDefined()
    },
  )

  it.each(cases.map((c, i) => [i + 1, c] as const))(
    "case %i's quoted message is what tsc prints",
    (_index, testCase) => {
      for (const fragment of fragmentsOf(testCase.quoted)) {
        expect(compilerSaid, `tsc never printed: ${fragment}`).toContain(normalize(fragment))
      }
    },
  )
})

const readme = readFileSync(resolve(root, "README.md"), "utf8")

describe("the README's GPT walkthrough", () => {
  it("is compiled by check-readme (no skip marker)", () => {
    // The block imports from "typenet", which check-readme rewrites to index.ts, so the package
    // root must export every layer the block names.
    expect(readme).not.toMatch(/check-readme: skip, this block is examples\/gpt\.ts/)
  })
})
