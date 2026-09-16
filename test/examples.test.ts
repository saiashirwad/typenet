import { execFileSync } from "node:child_process"
import { mkdtempSync, readFileSync, rmSync } from "node:fs"
import { tmpdir } from "node:os"
import { dirname, resolve } from "node:path"
import { fileURLToPath } from "node:url"
import { describe, expect, it } from "vitest"
import { alphabetOf, loadCheckpoint, saveCheckpoint, seeded } from "../examples/char-rnn/checkpoint.ts"
import { CharacterRNN } from "../examples/char-rnn/net.ts"
import { makeData, mlpModel } from "../examples/mlp-net.ts"
import { AdamW, configure, crossEntropy, Tensor } from "../index.ts"

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..")

function run(command: string, args: string[]): string {
  try {
    return execFileSync(command, args, {
      cwd: root,
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
  it("shapes.ts runs and prints the gallery", () => {
    const viteNode = resolve(root, "node_modules", ".bin", process.platform === "win32" ? "vite-node.cmd" : "vite-node")
    const out = run(viteNode, ["examples/shapes.ts"])
    expect(out).toContain("shape gallery")
    expect(out).toContain("[2, 5, 4, 8]")
    expect(out).toContain("16 = DimMul(4, 4)")
    // The runtime message matches the type-level one.
    expect(out).toContain("matmul: inner dimensions do not match ([2, 3] @ [2, 3])")
  }, 120_000)

  // The runnable examples train for hundreds of steps, which no suite should pay for; these
  // exercise the same pieces those entry points import, for long enough to see the loss move.
  it("the mlp model and its data learn", () => {
    configure({ seed: 7 })
    const { train } = makeData(256, 16)
    const model = mlpModel()
    const opt = new AdamW(model.parameters(), { lr: 3e-3, weightDecay: 0.01 })

    const losses: number[] = []
    for (let step = 0; step < 12; step++) {
      const at = step * 16
      const loss = crossEntropy(
        model.forward(train.x.narrow(0, at, 16)),
        Tensor.indices(train.labels.slice(at, at + 16), [16]),
      )
      opt.zeroGrad()
      loss.backward()
      opt.step()
      losses.push(loss.item())
    }
    expect(losses.at(-1)!).toBeLessThan(losses[0]!)
  }, 120_000)

  it("the character rnn learns and round-trips through a checkpoint", () => {
    const text = readFileSync(resolve(root, "examples/char-rnn/data/essay.txt"), "utf8")
    const alphabet = alphabetOf(text)
    expect(alphabet).toHaveLength(30)

    configure({ seed: 1234 })
    const model = new CharacterRNN({ vocab: alphabet.length, embed: 16, hidden: 32, unroll: 16 }, alphabet)
    const codes = Array.from(model.encode(text).data, Number)
    expect(codes).toHaveLength(text.length)

    const opt = new AdamW(model.parameters(), { lr: 5e-3, weightDecay: 0 })
    const losses: number[] = []
    for (let step = 0; step < 30; step++) {
      const xs: number[] = []
      const ys: number[] = []
      for (let b = 0; b < 8; b++) {
        const start = ((step * 8 + b) * 17) % (codes.length - 17)
        for (let t = 0; t < 16; t++) {
          xs.push(codes[start + t]!)
          ys.push(codes[start + t + 1]!)
        }
      }
      const loss = model.lossOn(Tensor.indices(xs, [8, 16]), Tensor.indices(ys, [8, 16]))
      opt.zeroGrad()
      loss.backward()
      opt.step()
      losses.push(loss.item())
    }
    // An untrained model over 30 symbols costs ln(30) = 3.4012 and nothing else; starting
    // anywhere else means the loss, the target or the alphabet is wrong.
    expect(losses[0]!).toBeGreaterThan(3.3)
    expect(losses[0]!).toBeLessThan(3.6)
    expect(losses.at(-1)!).toBeLessThan(losses[0]!)

    const dir = mkdtempSync(resolve(tmpdir(), "typenet-rnn-"))
    try {
      const path = resolve(dir, "char-rnn.json")
      expect(saveCheckpoint(path, model, 1234)).toBe(Object.keys(model.stateDict()).length)
      const loaded = loadCheckpoint(path)
      expect(loaded.signature).toMatchObject({ vocab: 30, embed: 16, hidden: 32, unroll: 16, seed: 1234 })

      // Same weights, same seed, so the reloaded model must sample the same characters.
      model.eval()
      const sample = (m: CharacterRNN<number, number, number>) =>
        m.decode(m.generate(m.encode("\n"), { length: 40, temperature: 0.6, rng: seeded(1234) }).data)
      expect(sample(loaded.model)).toBe(sample(model))
      expect(sample(model)).toHaveLength(41)
    } finally {
      rmSync(dir, { recursive: true, force: true })
    }
  }, 120_000)
})

describe("examples are cast-free", () => {
  it("no example reaches for an escape hatch", () => {
    // Shapes must be inferred rather than asserted, or the example proves nothing.
    const banned = /\bas (any|unknown|never)\b|assertChecked|AnyTensor/
    const files = [
      "examples/shapes.ts",
      "examples/mlp.ts",
      "examples/mlp-net.ts",
      "examples/gpt.ts",
      "examples/char-rnn/train.ts",
      "examples/char-rnn/generate.ts",
    ]
    for (const file of files) {
      readFileSync(resolve(root, file), "utf8").split("\n").forEach((line, i) => {
        expect(banned.test(line.replace(/\/\/.*$/, "")), `${file}:${i + 1}: ${line.trim()}`).toBe(false)
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
