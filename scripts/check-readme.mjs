#!/usr/bin/env node
// W5.7a — the README typechecker.
//
// Extracts every ```ts fenced block from README.md, writes each one to a
// temp directory as a standalone module, and compiles the lot with the
// workspace tsc (tsover). A block that does not compile fails the script,
// with the diagnostic re-anchored to its line in README.md rather than to
// the temp file nobody wrote.
//
// WHY every block is standalone rather than sharing a hidden prelude: a
// README block that only compiles against setup the reader cannot see is
// exactly the block that goes stale. Each one carries its own imports, and
// `import { … } from "typenet"` is rewritten to the repo's index.ts here
// so the published spelling is what the reader copies.
//
// A block that genuinely must not be compiled opts out with an HTML
// comment on the line above its fence:
//
//     <!-- check-readme: skip — why this one is not a program -->
//
// The reason is mandatory, and every skip is printed in the summary, so
// opting out is a visible choice and not a silent one.

import { execFileSync } from "node:child_process"
import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs"
import { tmpdir } from "node:os"
import { dirname, resolve } from "node:path"
import { fileURLToPath } from "node:url"

const __dirname = dirname(fileURLToPath(import.meta.url))
const root = resolve(__dirname, "..")
const readmePath = resolve(root, "README.md")

/** Languages whose blocks are compiled. `sh`, `json`, … are prose. */
const CHECKED = new Set(["ts", "tsx", "typescript"])

/**
 * Fenced blocks, with the 1-based README line of the first line of code and
 * the `check-readme:` directive that precedes the fence, if any.
 */
function extractBlocks(markdown) {
  const lines = markdown.split("\n")
  const blocks = []
  let directive
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i]
    const comment = /^<!--\s*check-readme:\s*(.*?)\s*-->\s*$/.exec(line)
    if (comment) {
      directive = comment[1]
      continue
    }
    const fence = /^```([A-Za-z0-9_+-]*)\s*$/.exec(line)
    if (!fence) {
      if (line.trim() !== "") directive = undefined
      continue
    }
    const lang = fence[1]
    const start = i + 1
    let end = start
    while (end < lines.length && !/^```\s*$/.test(lines[end])) end++
    if (CHECKED.has(lang)) {
      blocks.push({
        lang,
        line: start + 1, // 1-based line of the first code line
        code: lines.slice(start, end).join("\n"),
        directive,
      })
    }
    directive = undefined
    i = end
  }
  return blocks
}

function tscBinary() {
  const local = resolve(root, "node_modules", ".bin", process.platform === "win32" ? "tsc.cmd" : "tsc")
  return existsSync(local) ? local : "tsc"
}

/** The import spelling a reader would use, pointed at this checkout. */
function rewriteImports(code) {
  const indexPath = resolve(root, "index.ts")
  return code.replaceAll(/(["'])typenet\1/g, JSON.stringify(indexPath))
}

const markdown = readFileSync(readmePath, "utf8")
const blocks = extractBlocks(markdown)

const skipped = []
const compiled = []
for (const block of blocks) {
  if (block.directive?.startsWith("skip")) {
    const reason = block.directive.replace(/^skip\s*[—:-]?\s*/, "").trim()
    if (reason === "") {
      console.error(
        `README.md:${block.line}: a check-readme skip needs a reason — write`
          + ` "<!-- check-readme: skip — why -->"`,
      )
      process.exit(1)
    }
    skipped.push({ ...block, reason })
    continue
  }
  compiled.push(block)
}

const dir = mkdtempSync(resolve(tmpdir(), "typenet-readme-"))
try {
  const files = compiled.map((block, i) => {
    const name = `block-${String(i + 1).padStart(2, "0")}.ts`
    const path = resolve(dir, name)
    mkdirSync(dirname(path), { recursive: true })
    writeFileSync(path, rewriteImports(block.code) + "\n")
    return { ...block, name, path }
  })

  let output = ""
  try {
    execFileSync(tscBinary(), [
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
      ...files.map(f => f.path),
    ], { cwd: root, encoding: "utf8", stdio: ["ignore", "pipe", "pipe"] })
  } catch (e) {
    output = `${e.stdout ?? ""}${e.stderr ?? ""}`
  }

  const diagnostics = output
    .split("\n")
    .filter(l => /error TS\d+/.test(l))

  if (diagnostics.length > 0) {
    for (const line of diagnostics) {
      // "<path>(row,col): error TSxxxx: …" -> "README.md:<readme line>: …"
      const m = /^(.*?)\((\d+),(\d+)\): (.*)$/.exec(line)
      if (!m) {
        console.error(line)
        continue
      }
      const file = files.find(f => m[1].endsWith(f.name))
      if (!file) {
        console.error(line)
        continue
      }
      console.error(`README.md:${file.line + Number(m[2]) - 1}:${m[3]}: ${m[4]}`)
    }
    console.error(
      `\ncheck-readme: ${diagnostics.length} error(s) in ${compiled.length} code block(s)`,
    )
    process.exit(1)
  }

  for (const block of skipped) {
    console.log(`  skipped README.md:${block.line} — ${block.reason}`)
  }
  console.log(
    `check-readme: ${compiled.length} code block(s) typecheck`
      + (skipped.length > 0 ? `, ${skipped.length} skipped` : ""),
  )
} finally {
  rmSync(dir, { recursive: true, force: true })
}
