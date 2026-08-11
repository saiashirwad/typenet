// Surface point clouds as 3-d growth targets. The procedural sphere,
// torus and jack paint a thin shell inside a random cube of nodes, so only
// a fifth of the graph is the pattern and a demo looks like dust. Here
// every node sits on a real mesh surface: the graph *is* the shape, alpha
// is 1 everywhere, and colour comes from surface normals, so a regrown ear
// is a regrown colour you can check.
//
// Ported from ~/code/graph-cellular-automata/src/gnca/pointclouds.py.
// Clouds live as .npz under that repo's data/pointclouds/, produced by its
// scripts/fetch_pointclouds.py.

import { readFileSync } from "node:fs"
import { inflateRawSync } from "node:zlib"
import { hueRgb } from "./targets.ts"
import type { Points } from "./graphs.ts"
import type { Target } from "./targets.ts"

/**
 * Cloud name to a seed hint in [0,1]^3 — the node nearest that point
 * becomes the seed. The hints aim at a distinctive tip, so growth has a
 * clear place to start from.
 */
export const POINTCLOUDS: Record<string, number[]> = {
  bunny: [0.5, 0.85, 0.55], // near the ears
  spot: [0.5, 0.75, 0.7], // head and horns
  armadillo: [0.5, 0.8, 0.55], // head
  teapot: [0.75, 0.55, 0.55] // spout tip
}

// --- just enough of the npz format ---------------------------------
// An .npz is a zip of .npy members. Both formats are simple enough to
// read directly, which beats taking a dependency to load four files.

/**
 * Members of a zip, by name, decompressed. Stored and deflated only.
 *
 * Read through the central directory rather than by walking local headers
 * from the front: numpy writes ZIP64, which puts 0xFFFFFFFF in the local
 * header's size fields and the real sizes in an extra field, so a walk
 * that trusted those sizes would step into the middle of the file.
 */
function unzip(bytes: Buffer): Map<string, Buffer> {
  const out = new Map<string, Buffer>()
  let at = centralDirectoryStart(bytes)
  while (
    at + 46 <= bytes.length &&
    bytes.readUInt32LE(at) === 0x02014b50
  ) {
    const method = bytes.readUInt16LE(at + 10)
    const nameLength = bytes.readUInt16LE(at + 28)
    const extraLength = bytes.readUInt16LE(at + 30)
    const commentLength = bytes.readUInt16LE(at + 32)
    const name = bytes
      .subarray(at + 46, at + 46 + nameLength)
      .toString("latin1")
    let compressed = bytes.readUInt32LE(at + 20)
    let offset = bytes.readUInt32LE(at + 42)
    // ZIP64 extra field (header id 1): 8-byte uncompressed size, then
    // compressed size, then the local header offset — each present only
    // if the 32-bit field it replaces was maxed out.
    if (
      compressed === 0xffffffff ||
      offset === 0xffffffff
    ) {
      const extraAt = at + 46 + nameLength
      let field = extraAt
      while (field + 4 <= extraAt + extraLength) {
        const id = bytes.readUInt16LE(field)
        const size = bytes.readUInt16LE(field + 2)
        if (id === 0x0001) {
          let read = field + 4
          const uncompressedMaxed =
            bytes.readUInt32LE(at + 24) === 0xffffffff
          if (uncompressedMaxed) read += 8
          if (compressed === 0xffffffff) {
            compressed = Number(bytes.readBigUInt64LE(read))
            read += 8
          }
          if (offset === 0xffffffff)
            offset = Number(bytes.readBigUInt64LE(read))
          break
        }
        field += 4 + size
      }
    }
    // The local header repeats the name and may carry a different extra
    // field, so the body's start has to come from it, not from here.
    const localName = bytes.readUInt16LE(offset + 26)
    const localExtra = bytes.readUInt16LE(offset + 28)
    const start = offset + 30 + localName + localExtra
    const body = bytes.subarray(start, start + compressed)
    if (method === 0) out.set(name, body)
    else if (method === 8)
      out.set(name, inflateRawSync(body))
    else
      throw new Error(
        `unzip: ${name} uses compression method ${method}, expected stored or deflate`
      )
    at += 46 + nameLength + extraLength + commentLength
  }
  if (out.size === 0)
    throw new Error("unzip: no zip members found")
  return out
}

/** Offset of the first central directory entry. */
function centralDirectoryStart(bytes: Buffer): number {
  // The end-of-central-directory record is last, but a trailing comment
  // may follow it, so scan back for its signature.
  for (let at = bytes.length - 22; at >= 0; at--) {
    if (bytes.readUInt32LE(at) !== 0x06054b50) continue
    const start = bytes.readUInt32LE(at + 16)
    if (start !== 0xffffffff) return start
    // ZIP64: a locator sits just before, pointing at the ZIP64 record,
    // which holds the real offset.
    const locator = at - 20
    if (
      locator < 0 ||
      bytes.readUInt32LE(locator) !== 0x07064b58
    )
      throw new Error(
        "unzip: ZIP64 end record missing its locator"
      )
    const record = Number(
      bytes.readBigUInt64LE(locator + 8)
    )
    if (bytes.readUInt32LE(record) !== 0x06064b50)
      throw new Error("unzip: bad ZIP64 end record")
    return Number(bytes.readBigUInt64LE(record + 48))
  }
  throw new Error(
    "unzip: no end-of-central-directory record"
  )
}

/** A little-endian float32 .npy array, as its values and shape. */
function readNpy(bytes: Buffer): {
  values: Float32Array
  shape: number[]
} {
  if (bytes.subarray(1, 6).toString("latin1") !== "NUMPY")
    throw new Error("readNpy: not a .npy file")
  const major = bytes[6]!
  // v1 stores the header length in 2 bytes, v2 and up in 4
  const headerLength =
    major === 1 ?
      bytes.readUInt16LE(8)
    : bytes.readUInt32LE(8)
  const headerAt = major === 1 ? 10 : 12
  const header = bytes
    .subarray(headerAt, headerAt + headerLength)
    .toString("latin1")
  const dtype = /'descr':\s*'([^']+)'/.exec(header)?.[1]
  if (dtype !== "<f4")
    throw new Error(
      `readNpy: expected little-endian float32, got ${dtype}`
    )
  if (/'fortran_order':\s*True/.test(header))
    throw new Error(
      "readNpy: column-major arrays are not supported"
    )
  const shape = (
    /'shape':\s*\(([^)]*)\)/.exec(header)?.[1] ?? ""
  )
    .split(",")
    .map(s => s.trim())
    .filter(s => s.length > 0)
    .map(Number)
  const start = headerAt + headerLength
  const count = shape.reduce((a, b) => a * b, 1)
  // Copy rather than view: the offset is not guaranteed to be aligned to 4.
  const values = new Float32Array(count)
  for (let i = 0; i < count; i++)
    values[i] = bytes.readFloatLE(start + i * 4)
  return { values, shape }
}

// --- clouds ---------------------------------------------------------

/** Centre and scale so the cloud fills [margin, 1 - margin]^dim. */
function toUnitCube(
  pos: Float32Array,
  dim: number,
  margin = 0.08
): Float32Array {
  const n = pos.length / dim
  const lo = new Float32Array(dim).fill(Infinity)
  const hi = new Float32Array(dim).fill(-Infinity)
  for (let i = 0; i < n; i++)
    for (let d = 0; d < dim; d++) {
      const v = pos[i * dim + d]!
      if (v < lo[d]!) lo[d] = v
      if (v > hi[d]!) hi[d] = v
    }
  let span = 0
  for (let d = 0; d < dim; d++)
    span = Math.max(span, hi[d]! - lo[d]!)
  // Longest axis maps to [margin, 1 - margin]; the others stay in
  // proportion, so the shape is not stretched.
  const scale = (1 - 2 * margin) / (span || 1)
  const out = new Float32Array(pos.length)
  for (let i = 0; i < n; i++)
    for (let d = 0; d < dim; d++) {
      const mid = (lo[d]! + hi[d]!) / 2
      out[i * dim + d] =
        (pos[i * dim + d]! - mid) * scale + 0.5
    }
  return out
}

/** Rainbow by azimuth, dimmed by elevation: a readable regeneration cue. */
function colourByNormals(
  normals: Float32Array,
  dim: number
): Target {
  const n = normals.length / dim
  const out = new Float32Array(n * 4)
  for (let i = 0; i < n; i++) {
    const x = normals[i * dim]!
    const y = normals[i * dim + 1]!
    const z = normals[i * dim + 2]!
    const length = Math.max(Math.hypot(x, y, z), 1e-8)
    const h =
      (Math.atan2(y / length, x / length) + Math.PI) /
      (2 * Math.PI)
    const shade = 0.45 + 0.55 * ((z / length + 1) * 0.5)
    const rgb = hueRgb(h)
    out[i * 4] = rgb[0] * shade
    out[i * 4 + 1] = rgb[1] * shade
    out[i * 4 + 2] = rgb[2] * shade
    out[i * 4 + 3] = 1
  }
  return out
}

/**
 * Load a prepared cloud: its positions in the unit cube, an RGBA target
 * coloured by surface normal, and where to put the seed.
 *
 * `nodes` subsamples evenly when it is smaller than the stored cloud —
 * evenly rather than randomly, because reproducing numpy's PRNG is not
 * worth it and a stride keeps the surface evenly covered. A cloud cannot
 * be *grown*: there are no surface points to invent.
 */
export function loadCloud(
  path: string,
  options: { nodes?: number; seedAt?: number[] } = {}
): { pos: Points; target: Target; seedAt: number[] } {
  const members = unzip(readFileSync(path))
  const posMember = members.get("pos.npy")
  const normalMember = members.get("normals.npy")
  if (!posMember) throw new Error(`${path} has no pos.npy`)
  if (!normalMember)
    throw new Error(
      `${path} has no normals.npy — this reader does not estimate ` +
        `normals; regenerate the cloud with the reference's ` +
        `scripts/fetch_pointclouds.py`
    )
  const loaded = readNpy(posMember)
  const dim = loaded.shape[1] ?? 3
  let pos = loaded.values
  let normals = readNpy(normalMember).values
  let n = loaded.shape[0] ?? pos.length / dim

  const wanted = options.nodes
  if (wanted !== undefined && wanted < n) {
    const keptPos = new Float32Array(wanted * dim)
    const keptNormals = new Float32Array(wanted * dim)
    for (let i = 0; i < wanted; i++) {
      const from = Math.floor((i * n) / wanted)
      for (let d = 0; d < dim; d++) {
        keptPos[i * dim + d] = pos[from * dim + d]!
        keptNormals[i * dim + d] = normals[from * dim + d]!
      }
    }
    pos = keptPos
    normals = keptNormals
    n = wanted
  }

  const name = path
    .replace(/^.*\//, "")
    .replace(/\.npz$/, "")
  return {
    pos: { data: toUnitCube(pos, dim), n, dim },
    target: colourByNormals(normals, dim),
    seedAt: options.seedAt ??
      POINTCLOUDS[name] ?? [0.5, 0.5, 0.5]
  }
}
