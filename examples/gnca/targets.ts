// Ported from ~/code/graph-cellular-automata/src/gnca/targets.py.

import { point, type Points } from "./graphs.ts"

/** RGBA rows, flat and row-major: 4 floats per node. */
export type Target = Float32Array

type Colour = readonly [number, number, number]

/** HSV to RGB at full saturation and value, hue in [0, 1]. */
export function hueRgb(h: number): Colour {
  const sector = Math.floor(h * 6) % 6
  const f = h * 6 - Math.floor(h * 6)
  const table: Colour[] = [
    [1, f, 0],
    [1 - f, 1, 0],
    [0, 1, f],
    [0, 1 - f, 1],
    [f, 0, 1],
    [1, 0, 1 - f],
  ]
  return table[sector]!
}

function polar(
  pos: Points,
  i: number,
  cx = 0.5,
  cy = 0.5,
): { r: number; h: number } {
  const dx = point(pos, i, 0) - cx
  const dy = point(pos, i, 1) - cy
  return {
    r: Math.hypot(dx, dy),
    h: (Math.atan2(dy, dx) + Math.PI) / (2 * Math.PI),
  }
}

function paint(
  pos: Points,
  rule: (i: number) => { inside: boolean; colour: Colour },
): Target {
  const out = new Float32Array(pos.n * 4)
  for (let i = 0; i < pos.n; i++) {
    const { inside, colour } = rule(i)
    if (!inside) continue
    out[i * 4] = colour[0]
    out[i * 4 + 1] = colour[1]
    out[i * 4 + 2] = colour[2]
    out[i * 4 + 3] = 1
  }
  return out
}

export function heart(pos: Points): Target {
  return paint(pos, i => {
    const x = (point(pos, i, 0) - 0.5) * 2.6
    const y = (point(pos, i, 1) - 0.45) * 2.6
    const a = x * x + y * y - 1
    const r = Math.hypot(x, y)
    return {
      inside: a * a * a - x * x * y * y * y < 0,
      colour: [
        0.5 + 0.5 * Math.sin(8 * r),
        0.5 + 0.5 * Math.sin(8 * r + 2.1),
        0.5 + 0.5 * Math.sin(8 * r + 4.2),
      ],
    }
  })
}

export function star(pos: Points, arms = 5): Target {
  return paint(pos, i => {
    const { r, h } = polar(pos, i)
    return {
      inside: r < 0.2 + 0.13 * Math.cos(arms * h * 2 * Math.PI),
      colour: hueRgb(h),
    }
  })
}

export function annulus(pos: Points): Target {
  return paint(pos, i => {
    const { r, h } = polar(pos, i)
    return {
      inside: r > 0.17 && r < 0.33,
      colour: hueRgb(h),
    }
  })
}

export function lobes(pos: Points): Target {
  return paint(pos, i => {
    const x = point(pos, i, 0)
    const y = point(pos, i, 1)
    const left = (x - 0.31) ** 2 + (y - 0.5) ** 2 < 0.031
    const right = (x - 0.69) ** 2 + (y - 0.5) ** 2 < 0.031
    const bridge = Math.abs(y - 0.5) < 0.035 && Math.abs(x - 0.5) < 0.21
    const colour: Colour = right
      ? [1, 0.42, 0.6]
      : bridge && !left
      ? [0.95, 0.85, 0.35]
      : [0.3, 0.78, 0.95]
    return { inside: left || right || bridge, colour }
  })
}

export function ring(pos: Points): Target {
  return paint(pos, i => ({
    inside: true,
    colour: hueRgb(polar(pos, i).h),
  }))
}

export function sphere(pos: Points): Target {
  return paint(pos, i => {
    const d = [0, 1, 2].map(k => point(pos, i, k) - 0.5)
    const r = Math.hypot(d[0]!, d[1]!, d[2]!)
    const h = (Math.atan2(d[1]!, d[0]!) + Math.PI) / (2 * Math.PI)
    const shade = 0.45 + 0.55 * ((d[2]! / 0.39 + 1) / 2)
    const rgb = hueRgb(h)
    return {
      inside: r > 0.19 && r < 0.39,
      colour: [
        rgb[0] * shade,
        rgb[1] * shade,
        rgb[2] * shade,
      ],
    }
  })
}

export function torus(
  pos: Points,
  R = 0.27,
  tube = 0.19,
): Target {
  return paint(pos, i => {
    const d = [0, 1, 2].map(k => point(pos, i, k) - 0.5)
    const q = Math.hypot(d[0]!, d[1]!) - R
    const h = (Math.atan2(d[1]!, d[0]!) + Math.PI) / (2 * Math.PI)
    return {
      inside: q * q + d[2]! * d[2]! < tube * tube,
      colour: hueRgb(h),
    }
  })
}

export function jack(
  pos: Points,
  half = 0.42,
  thick = 0.17,
): Target {
  const arms: Colour[] = [
    [1, 0.42, 0.6],
    [0.55, 0.92, 0.45],
    [0.35, 0.7, 1],
  ]
  return paint(pos, i => {
    const d = [0, 1, 2].map(k => point(pos, i, k) - 0.5)
    for (let axis = 0; axis < 3; axis++) {
      const others = [0, 1, 2].filter(k => k !== axis)
      const radial = Math.hypot(
        d[others[0]!]!,
        d[others[1]!]!,
      )
      if (Math.abs(d[axis]!) < half && radial < thick) {
        return { inside: true, colour: arms[axis]! }
      }
    }
    return { inside: false, colour: [0, 0, 0] }
  })
}

/**
 * Pattern name to its builder and where to put the seed node. The seed
 * is not always the centre — the annulus has nothing there. The length
 * of the seed point is what tells the rest of the code whether a target
 * is 2-d or 3-d.
 */
export const TARGETS: Record<
  string,
  { build: (pos: Points) => Target; seedAt: number[] }
> = {
  heart: { build: heart, seedAt: [0.5, 0.45] },
  star: { build: star, seedAt: [0.5, 0.5] },
  // on the ring, not in the hole
  annulus: { build: annulus, seedAt: [0.75, 0.5] },
  lobes: { build: lobes, seedAt: [0.31, 0.5] },
  // node 0 on the Watts-Strogatz ring
  ring: { build: ring, seedAt: [1, 0.5] },
  sphere: { build: sphere, seedAt: [0.8, 0.5, 0.5] },
  torus: { build: torus, seedAt: [0.77, 0.5, 0.5] },
  jack: { build: jack, seedAt: [0.5, 0.5, 0.5] },
}
