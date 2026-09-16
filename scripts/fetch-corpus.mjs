// Downloads Tiny Shakespeare (1.1 MB, public domain) to the path TYPENET_RNN_TEXT points at in
// examples/char-rnn.

import { mkdirSync, writeFileSync } from "node:fs"
import { dirname, resolve } from "node:path"
import { fileURLToPath } from "node:url"

const URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
const target = resolve(dirname(fileURLToPath(import.meta.url)), "../examples/char-rnn/data/tiny-shakespeare.txt")

const response = await fetch(URL)
if (!response.ok) throw new Error(`GET ${URL} -> ${response.status} ${response.statusText}`)
const text = await response.text()
mkdirSync(dirname(target), { recursive: true })
writeFileSync(target, text)
console.log(`fetch-corpus: wrote ${text.length.toLocaleString("en-US")} characters to ${target}`)
