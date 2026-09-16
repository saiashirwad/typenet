#!/usr/bin/env node
// Compiles every ts block in README.md with "typenet" pointed at index.ts, and reports
// diagnostics on their README line. A block opts out with a `check-readme: skip, why` comment.

import { execFileSync } from "node:child_process"
import { existsSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs"
import { tmpdir } from "node:os"
import { dirname, resolve } from "node:path"
import { fileURLToPath } from "node:url"

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..")
const indexPath = resolve(root, "index.ts")

const CHECKED = new Set(["ts", "tsx", "typescript"])

function extractBlocks(markdown) {
  const lines = markdown.split("\n")
  const blocks = []
  let directive
  for (let i = 0; i < lines.length; i++) {
    const comment = /^<!--\s*check-readme:\s*(.*?)\s*-->\s*$/.exec(lines[i])
    if (comment) {
      directive = comment[1]
      continue
    }
    const fence = /^```([A-Za-z0-9_+-]*)\s*$/.exec(lines[i])
    if (!fence) {
      if (lines[i].trim() !== "") directive = undefined
      continue
    }
    const start = i + 1
    let end = start
    while (end < lines.length && !/^```\s*$/.test(lines[end])) end++
    if (CHECKED.has(fence[1])) {
      blocks.push({ line: start + 1, code: lines.slice(start, end).join("\n"), directive })
    }
    directive = undefined
    i = end
  }
  return blocks
}

const skipped = []
const compiled = []
for (const block of extractBlocks(readFileSync(resolve(root, "README.md"), "utf8"))) {
  if (!block.directive?.startsWith("skip")) {
    compiled.push(block)
    continue
  }
  const reason = block.directive.replace(/^skip\s*[,:-]?\s*/, "").trim()
  if (reason === "") {
    console.error(`README.md:${block.line}: a check-readme skip needs a reason, write "<!-- check-readme: skip, why -->"`)
    process.exit(1)
  }
  skipped.push({ ...block, reason })
}

const dir = mkdtempSync(resolve(tmpdir(), "typenet-readme-"))
try {
  const files = compiled.map((block, i) => {
    const name = `block-${String(i + 1).padStart(2, "0")}.ts`
    writeFileSync(resolve(dir, name), block.code.replaceAll(/(["'])typenet\1/g, JSON.stringify(indexPath)) + "\n")
    return { ...block, name, path: resolve(dir, name) }
  })

  const localTsc = resolve(root, "node_modules", ".bin", process.platform === "win32" ? "tsc.cmd" : "tsc")
  let output = ""
  try {
    execFileSync(existsSync(localTsc) ? localTsc : "tsc", [
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

  const diagnostics = output.split("\n").filter(l => /error TS\d+/.test(l))
  if (diagnostics.length > 0) {
    for (const line of diagnostics) {
      const m = /^(.*?)\((\d+),(\d+)\): (.*)$/.exec(line)
      const file = m && files.find(f => m[1].endsWith(f.name))
      // Map the temp file's line back onto the README, or pass the diagnostic through unchanged.
      console.error(file ? `README.md:${file.line + Number(m[2]) - 1}:${m[3]}: ${m[4]}` : line)
    }
    console.error(`\ncheck-readme: ${diagnostics.length} error(s) in ${compiled.length} code block(s)`)
    process.exit(1)
  }

  for (const block of skipped) console.log(`  skipped README.md:${block.line}, ${block.reason}`)
  console.log(`check-readme: ${compiled.length} code block(s) typecheck` + (skipped.length > 0 ? `, ${skipped.length} skipped` : ""))
} finally {
  rmSync(dir, { recursive: true, force: true })
}
