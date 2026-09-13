// W5.7b — `examples/gpt.ts`, the typed GPT.
//
// Three claims are made about that file, and each one is only worth as
// much as a test of it:
//
//   1. it RUNS — 20 steps of a real training loop, with a loss that
//      actually falls and `jsCounters().nativeFallbacks === 0`;
//   2. it is CAST-FREE — no `as any`/`as unknown`/`as never`, no
//      `assertChecked`, no `AnyTensor`. A showcase whose shapes are
//      asserted rather than inferred demonstrates nothing;
//   3. its four `@ts-expect-error` cases fail for the REASON quoted above
//      them. `pnpm typecheck` proves each directive fires; only
//      recompiling the file without them proves *which* error fired, and
//      that is the half the README quotes.
//
// Plus the README's own GPT block, checked against the file line by line —
// see the `check-readme: skip` on it. That block cannot be compiled by
// `scripts/check-readme.mjs` (the layers it names are not re-exported from
// the package root yet, so the `"typenet"` spelling a reader would copy
// does not resolve to them), so the drift check has to come from here.

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

// ---------------------------------------------------------------------------
// 1. It runs
// ---------------------------------------------------------------------------

/**
 * The example's output, run once and shared by the cases below.
 *
 * Run lazily from inside a test rather than at collection time: it is a
 * ~15 s child process, and a collection-time cost is one no test timeout
 * covers.
 */
let cachedRun: { out: string; losses: number[] } | null = null

function trainingRun(): { out: string; losses: number[] } {
  if (cachedRun) return cachedRun
  // The example's own default is 20 steps; passing it explicitly makes the
  // acceptance criterion ("20 steps with a decreasing loss") the thing this
  // test measures rather than whatever the default happens to be.
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
    // An untrained model over a 32-symbol alphabet costs ln(32) = 3.4657
    // per token. Starting anywhere else means the init or the tie is
    // wrong, which is a bug a merely-decreasing curve would hide.
    expect(losses[0]!).toBeGreaterThan(3.3)
    expect(losses[0]!).toBeLessThan(3.6)
    expect(losses.at(-1)!).toBeLessThan(losses[0]!)
    // Not just the endpoints: the second half is below the first, so a
    // curve that dips once and then diverges does not pass.
    const mean = (xs: number[]) => xs.reduce((a, b) => a + b, 0) / xs.length
    expect(mean(losses.slice(10))).toBeLessThan(mean(losses.slice(0, 10)))
  }, 300_000)

  it("reports the tied token table once", () => {
    // 28 tensors, not 29: the [32, 64] table is the LM head's weight too,
    // and `parameters()` dedups it by storage identity.
    expect(trainingRun().out).toContain("28 tensors")
    expect(trainingRun().out).toContain("104,192 parameters")
  }, 300_000)

  it("never falls off the native path", () => {
    // Eager mode serialises no graph at all, so this is the trivially-true
    // case — which is exactly why it is asserted rather than assumed: the
    // day the example moves to `compile()`, this line stops being trivial
    // and starts being the gate.
    expect(trainingRun().out).toContain("native fallbacks: 0")
  }, 300_000)
})

// ---------------------------------------------------------------------------
// 2. It is cast-free
// ---------------------------------------------------------------------------

describe("examples/gpt.ts is cast-free", () => {
  it("no escape hatch appears in the example", () => {
    // W5.7b's acceptance grep, as a test.
    const banned = /\bas (any|unknown|never)\b|assertChecked|AnyTensor/
    exampleSource.split("\n").forEach((line, i) => {
      expect(banned.test(line), `examples/gpt.ts:${i + 1}: ${line.trim()}`).toBe(false)
    })
  })
})

// ---------------------------------------------------------------------------
// 3. The four compile errors, checked against the compiler
// ---------------------------------------------------------------------------

type ErrorCase = {
  /** The TS error code the case claims, e.g. 2345. */
  code: number
  /** The `// tsc(NNNN):` lines above the directive, joined into one line. */
  quoted: string
  /** 1-based line of the `@ts-expect-error` directive. */
  directiveLine: number
  /** 1-based line where the next case's quote begins, or the end of file. */
  endLine: number
}

/** `​` (U+200B) terminates every `ErrorMessage`; it is invisible in a
 * comment and must not be part of the comparison. */
function normalize(text: string): string {
  return text.replaceAll("​", "").replaceAll(/\s+/g, " ").trim()
}

/** A quoted diagnostic split into its first line and each indented
 * elaboration under it — `tsc` prints them as separate lines, so they are
 * compared as separate fragments. */
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
 * Compiles the example with every `@ts-expect-error` removed, in a temp
 * directory so the project's own typecheck is untouched.
 *
 * Every `from "../…"` is re-pointed at this checkout, since the copy no
 * longer sits next to `examples/`. The directives are replaced by a
 * comment rather than deleted so the reported line numbers still line up
 * with the real file.
 *
 * `raw` is the whole compiler output: `tsc` prints a diagnostic's
 * elaboration ("Type '64' is not assignable to type '32'.") on its own
 * indented line with no file anchor, so the anchored `diagnostics` list
 * below cannot be what the quoted messages are matched against.
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
          + ` — got ${JSON.stringify(diagnostics)}`,
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

// ---------------------------------------------------------------------------
// 4. The README's GPT block has not drifted from the file
// ---------------------------------------------------------------------------

const readme = readFileSync(resolve(root, "README.md"), "utf8")

/** The one ```ts block under the `## A GPT that typechecks` heading. */
function readmeGptBlock(): string[] {
  const lines = readme.split("\n")
  const start = lines.findIndex(l => l.trim() === "## A GPT that typechecks")
  expect(start, "README.md has no `## A GPT that typechecks` section").toBeGreaterThan(-1)
  const open = lines.findIndex((l, i) => i > start && l.trim() === "```ts")
  expect(open, "the GPT section has no ts block").toBeGreaterThan(start)
  const close = lines.findIndex((l, i) => i > open && l.trim() === "```")
  return lines.slice(open + 1, close)
}

describe("the README's GPT walkthrough", () => {
  it("quotes the model out of examples/gpt.ts, line for line", () => {
    // Matched against the file with its whitespace collapsed, not line by
    // line: `dprint` formats TypeScript inside a markdown fence at its own
    // line width, so a long expression can be wrapped in the README and
    // not in the file. Collapsing whitespace makes the check about the
    // code and not about where the formatter chose to break it.
    const flat = normalize(exampleSource)
    const block = readmeGptBlock()
    expect(block.length).toBeGreaterThan(20)
    for (const line of block) {
      const trimmed = normalize(line)
      if (trimmed === "") continue
      // The README imports from "typenet"; the example from "../index.ts".
      if (trimmed.startsWith("import ") || trimmed.startsWith("} from ")) continue
      expect(
        flat.includes(trimmed),
        `README.md quotes a line examples/gpt.ts does not have: ${trimmed}`,
      ).toBe(true)
    }
  })

  it("is compiled by check-readme (no skip marker)", () => {
    // The block imports from "typenet", which scripts/check-readme.mjs
    // rewrites to index.ts, so the package root must export every layer
    // the block names — that is the whole point of checking it.
    expect(readme).not.toMatch(/check-readme: skip — this block is examples\/gpt\.ts/)
  })

  it("quotes every compiler message the example claims", () => {
    const normalized = normalize(readme)
    for (const testCase of parseCases(exampleSource)) {
      // The table quotes the sentence the type error turns on — the last
      // fragment of the diagnostic, which is the one that names the two
      // shapes (or the missing brand) rather than restating the call.
      const sentence = normalize(fragmentsOf(testCase.quoted).at(-1)!)
      expect(normalized, `README.md does not quote: ${sentence}`).toContain(sentence)
    }
  })
})
