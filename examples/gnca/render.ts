/** One frame: RGBA rows, flat, four floats per node. */
export type Frame = Float32Array

const clamp01 = (v: number) =>
  v < 0
    ? 0
    : v > 1
    ? 1
    : v

export function renderSvg(
  positions: Float32Array,
  dim: number,
  frames: Frame[],
  options: {
    size?: number
    labels?: string[]
    nodes?: number
  } = {},
): string {
  const size = options.size ?? 320
  const labels = options.labels ?? []
  const n = options.nodes ?? Math.floor(positions.length / dim)
  const radius = Math.max(1.2, (size / Math.sqrt(n)) * 0.55)
  const gap = 12
  const labelRoom = labels.length > 0 ? 20 : 0

  const order = Array.from({ length: n }, (_, i) => i)
  if (dim === 3) {
    order.sort(
      (a, b) => positions[a * dim + 2]! - positions[b * dim + 2]!,
    )
  }

  const panels = frames.map((frame, panel) => {
    const x0 = panel * (size + gap)
    const circles = order
      .map(i => {
        const alpha = clamp01(frame[i * 4 + 3]!)
        if (alpha < 0.02) return ""
        const cx = x0 + clamp01(positions[i * dim]!) * size
        // SVG y grows downward; flip so the picture matches the maths
        const cy = (1 - clamp01(positions[i * dim + 1]!)) * size
        const rgb = [0, 1, 2]
          .map(c => Math.round(clamp01(frame[i * 4 + c]!) * 255))
          .join(",")
        return (
          `<circle cx="${cx.toFixed(1)}" cy="${cy.toFixed(1)}" `
          + `r="${radius.toFixed(1)}" fill="rgb(${rgb})" `
          + `fill-opacity="${alpha.toFixed(3)}"/>`
        )
      })
      .join("")
    const label = labels[panel] === undefined
      ? ""
      : `<text x="${x0 + size / 2}" y="${size + 14}" fill="#888" `
        + `font-family="ui-monospace, monospace" font-size="11" `
        + `text-anchor="middle">${labels[panel]}</text>`
    return circles + label
  })

  const width = frames.length * size + (frames.length - 1) * gap
  return (
    `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" `
    + `height="${size + labelRoom}" viewBox="0 0 ${width} ${size + labelRoom}">`
    + `<rect width="100%" height="100%" fill="#111"/>`
    + panels.join("")
    + `</svg>\n`
  )
}

export function visible(
  state: Float32Array,
  nodes: number,
  channels: number,
): Frame {
  const out = new Float32Array(nodes * 4)
  for (let i = 0; i < nodes; i++) {
    for (let c = 0; c < 4; c++) {
      out[i * 4 + c] = state[i * channels + c]!
    }
  }
  return out
}
