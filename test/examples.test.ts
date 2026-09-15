import { execFileSync } from "node:child_process"
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs"
import { tmpdir } from "node:os"
import { dirname, resolve } from "node:path"
import { fileURLToPath } from "node:url"
import { describe, expect, it } from "vitest"

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..")

function bin(name: string): string {
  return resolve(root, "node_modules", ".bin", process.platform === "win32" ? `${name}.cmd` : name)
}

function run(
  command: string,
  args: string[],
  env: Record<string, string> = {},
): string {
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

describe("examples", () => {
  it("example:shapes runs and prints the gallery", () => {
    const out = run(bin("vite-node"), ["examples/shapes.ts"])
    expect(out).toContain("shape gallery")
    // The value twins agree with the types they carry.
    expect(out).toContain("[2, 5, 4, 8]")
    expect(out).toContain("16 = DimMul(4, 4)")
    // The same shape algebra raises the same sentence at run time.
    expect(out).toContain("matmul: inner dimensions do not match ([2, 3] @ [2, 3])")
  }, 120_000)

  it("example:mlp trains (a handful of steps)", () => {
    // The example's own STEPS, overridden so the smoke test exercises this
    // file rather than a copy of it. The README's numbers are the 400-step
    // default, which is too slow for the suite.
    const out = run(bin("vite-node"), ["examples/mlp.ts"], {
      TYPENET_EXAMPLE_STEPS: "12",
    })
    const losses = [...out.matchAll(/loss (\d+\.\d+)/g)].map(m => Number(m[1]))
    expect(losses.length).toBeGreaterThan(1)
    // 12 steps of warmup is not convergence, but a step that is wired up
    // wrong does not move the loss at all.
    expect(losses.at(-1)!).toBeLessThan(losses[0]!)
    expect(out).toContain("test accuracy")
  }, 120_000)
})

// `pnpm typecheck` already proves every `@ts-expect-error` in the gallery
// fires (an unused directive is itself an error). What it cannot prove is
// that the quoted message is the message the compiler actually produces.
// So: strip the directives, compile the file again, and read the
// diagnostics back.

type GalleryCase = {
  /** The TS error code the case claims, e.g. 2345. */
  code: number
  /** The message quoted in the `// tsc(NNNN):` comment, joined into one line. */
  quoted: string
  /** 1-based line of the `@ts-expect-error` directive. */
  directiveLine: number
  /** 1-based line where the next case's quote begins, or the end of file. */
  endLine: number
}

const galleryPath = resolve(root, "examples/shapes.ts")
const gallerySource = readFileSync(galleryPath, "utf8")

/** `​` (U+200B) terminates every `ErrorMessage`; it is invisible in a
 * comment and must not be part of the comparison. */
function normalize(text: string): string {
  return text.replaceAll("​", "").replaceAll(/\s+/g, " ").trim()
}

function parseGallery(source: string): GalleryCase[] {
  const lines = source.split("\n")
  const cases: GalleryCase[] = []
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
    // The directive may be several lines below the quote (case 8 declares a
    // class first), so scan forward for it.
    let directive = j
    while (directive < lines.length && !/^\s*\/\/ @ts-expect-error\s*$/.test(lines[directive]!)) {
      directive++
    }
    expect(
      directive,
      `examples/shapes.ts:${i + 1}: a // tsc(...) quote with no @ts-expect-error under it`,
    ).toBeLessThan(lines.length)
    cases.push({
      code,
      quoted: parts.join(" "),
      directiveLine: directive + 1,
      endLine: lines.length,
    })
    i = directive
  }
  for (let k = 0; k < cases.length - 1; k++) cases[k]!.endLine = cases[k + 1]!.directiveLine - 1
  return cases
}

type Diagnostic = { line: number; code: number; message: string }

/** Compiles the gallery with every `@ts-expect-error` removed, in a temp
 * directory so the project's own typecheck is untouched. */
function compileWithoutDirectives(): Diagnostic[] {
  const dir = mkdtempSync(resolve(tmpdir(), "typenet-gallery-"))
  try {
    const file = resolve(dir, "shapes.ts")
    const stripped = gallerySource
      .replaceAll(/^(\s*)\/\/ @ts-expect-error.*$/gm, "$1// (directive stripped)")
      .replaceAll(`"../index.ts"`, JSON.stringify(resolve(root, "index.ts")))
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
    return output
      .split("\n")
      .flatMap(line => {
        const m = /^.*shapes\.ts\((\d+),\d+\): error TS(\d+): (.*)$/.exec(line)
        return m ? [{ line: Number(m[1]), code: Number(m[2]), message: m[3]! }] : []
      })
  } finally {
    rmSync(dir, { recursive: true, force: true })
  }
}

describe("examples/shapes.ts", () => {
  const cases = parseGallery(gallerySource)
  const diagnostics = compileWithoutDirectives()

  it("is a gallery of exactly eight compile-time errors", () => {
    expect(cases).toHaveLength(8)
  })

  it.each(cases.map((c, i) => [i + 1, c] as const))(
    "case %i's quoted message is what tsc prints",
    (_index, testCase) => {
      const found = diagnostics.find(d =>
        d.code === testCase.code
        && d.line > testCase.directiveLine
        && d.line <= testCase.endLine
      )
      expect(
        found,
        `no TS${testCase.code} between examples/shapes.ts:${testCase.directiveLine} and :${testCase.endLine}`
          + ` — got ${JSON.stringify(diagnostics)}`,
      ).toBeDefined()
      expect(normalize(found!.message)).toContain(normalize(testCase.quoted))
    },
  )

  it("every quoted message appears in the README", () => {
    const readme = readFileSync(resolve(root, "README.md"), "utf8")
    for (const testCase of cases) {
      // The README quotes the sentence the shape algebra produced (the
      // part inside the `"..."` of an assignability error), or, for a
      // diagnostic with no such sentence, the whole first line.
      const inner = /"([^"]+)"/.exec(testCase.quoted)
      const sentence = normalize(inner ? inner[1]! : testCase.quoted)
      expect(normalize(readme), `README.md does not quote: ${sentence}`).toContain(sentence)
    }
  })
})

describe("examples are cast-free", () => {
  it("no example reaches for an escape hatch", () => {
    // The showcase is worthless if shapes are asserted rather than inferred.
    const banned = /\bas (any|unknown|never)\b|assertChecked|AnyTensor/
    const files = ["examples/shapes.ts", "examples/mlp.ts"]
    for (const file of files) {
      const source = readFileSync(resolve(root, file), "utf8")
      source.split("\n").forEach((line, i) => {
        expect(banned.test(line), `${file}:${i + 1}: ${line.trim()}`).toBe(false)
      })
    }
  })
})

describe("check:readme", () => {
  it("every ts block in README.md compiles", () => {
    const out = run(process.execPath, [resolve(root, "scripts/check-readme.mjs")])
    expect(out).toMatch(/check-readme: \d+ code block\(s\) typecheck/)
  }, 120_000)
})
