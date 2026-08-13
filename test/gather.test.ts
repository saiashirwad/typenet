import { afterEach, describe, expect, it } from "vitest"
import { disableNative, isNativeAvailable, useNative } from "../src/backends/native.ts"
import { configure, Tensor, tensor } from "../src/tensor.ts"

type AnyTensor = Tensor<any>

afterEach(() => {
  configure({ lazy: false })
  disableNative()
})

function allPaths(fn: () => AnyTensor): {
  eager: AnyTensor
  lazy: AnyTensor
  native: AnyTensor | null
} {
  configure({ lazy: false })
  const eager = fn()
  configure({ lazy: true })
  const lazy = fn()
  let native: AnyTensor | null = null
  if (isNativeAvailable()) {
    useNative()
    native = fn()
    native.data // force before disabling
    disableNative()
  }
  configure({ lazy: false })
  return { eager, lazy, native }
}

function expectAgree(
  fn: () => AnyTensor,
  tolerance = 1e-5,
): void {
  const { eager, lazy, native } = allPaths(fn)
  for (
    const [label, other] of [
      ["lazy", lazy],
      ["native", native],
    ] as const
  ) {
    if (!other) continue
    expect(other.shape, `${label} shape`).toEqual(
      eager.shape,
    )
    const a = eager.data
    const b = other.data
    for (let i = 0; i < a.length; i++) {
      expect(
        Math.abs(a[i]! - b[i]!),
        `${label} element ${i}: ${b[i]} vs eager ${a[i]}`,
      ).toBeLessThan(tolerance)
    }
  }
}

const rows = () =>
  tensor([
    [1, 2],
    [3, 4],
    [5, 6],
    [7, 8],
  ])

describe("indexSelect", () => {
  it("gathers rows, repeats included", () => {
    expect(
      rows()
        .indexSelect(tensor([2, 0, 2]))
        .toArray(),
    ).toEqual([
      [5, 6],
      [1, 2],
      [5, 6],
    ])
  })

  it("gathers along an inner dim", () => {
    expect(
      rows()
        .indexSelect(tensor([1, 0]), 1)
        .toArray(),
    ).toEqual([
      [2, 1],
      [4, 3],
      [6, 5],
      [8, 7],
    ])
  })

  it("accepts an empty index", () => {
    const out = rows().indexSelect(tensor([]) as any)
    expect(out.shape).toEqual([0, 2])
  })

  it("agrees across eager, lazy and native", () => {
    expectAgree(() =>
      rows()
        .indexSelect(tensor([3, 1, 1, 0]))
        .mul(2)
        .indexSelect(tensor([0, 2]))
    )
  })

  it("rejects an out-of-range index", () => {
    expect(() =>
      rows()
        .indexSelect(tensor([0, 9]))
        .toArray()
    ).toThrow(/index 9 out of range for 4 rows/)
  })

  it("rejects a fractional index", () => {
    expect(() =>
      rows()
        .indexSelect(tensor([0.5]))
        .toArray()
    ).toThrow(/out of range/)
  })

  it("rejects a non-rank-1 index", () => {
    expect(() => rows().indexSelect(tensor([[0], [1]]) as any)).toThrow(/requires a rank-1 index/)
  })
})

describe("scatterAdd", () => {
  it("sums colliding rows and zero-fills the rest", () => {
    expect(
      rows()
        .scatterAdd(tensor([1, 1, 0, 3]), 4)
        .toArray(),
    ).toEqual([
      [5, 6],
      [4, 6],
      [0, 0],
      [7, 8],
    ])
  })

  it("scatters along an inner dim", () => {
    expect(
      tensor([[1, 2, 3]])
        .scatterAdd(tensor([0, 0, 1]), 2, 1)
        .toArray(),
    ).toEqual([[3, 3]])
  })

  it("agrees across eager, lazy and native", () => {
    expectAgree(() =>
      rows()
        .scatterAdd(tensor([2, 0, 2, 1]), 3)
        .add(1)
        .scatterAdd(tensor([0, 0, 0]), 1)
    )
  })

  it("is the exact reverse of indexSelect", () => {
    const index = tensor([0, 2, 2, 3])
    const gathered = rows().indexSelect(index)
    const back = gathered.scatterAdd(index, 4)
    expect(back.toArray()).toEqual([
      [1, 2],
      [0, 0],
      [10, 12],
      [7, 8],
    ])
  })

  it("rejects an index length that does not match the source", () => {
    expect(() =>
      // @ts-expect-error index must have one entry per source row
      rows().scatterAdd(tensor([0, 1]), 4)
    ).toThrow(/2 indices for 4 rows along dim 0/)
  })

  it("rejects an out-of-range index", () => {
    expect(() =>
      rows()
        .scatterAdd(tensor([0, 1, 2, 9]), 4)
        .toArray()
    ).toThrow(/index 9 out of range for 4 rows/)
  })

  it("rejects a negative length", () => {
    expect(() => rows().scatterAdd(tensor([0, 1, 2, 3]), -1)).toThrow(/non-negative integer length/)
  })
})

describe("narrow", () => {
  it("slices a contiguous window", () => {
    expect(rows().narrow(0, 1, 2).toArray()).toEqual([
      [3, 4],
      [5, 6],
    ])
    expect(rows().narrow(1, 1, 1).toArray()).toEqual([
      [2],
      [4],
      [6],
      [8],
    ])
  })

  it("agrees across eager, lazy and native", () => {
    expectAgree(() => rows().narrow(0, 1, 3).narrow(1, 0, 1).mul(3))
  })

  it("rejects a window past the end", () => {
    expect(() => rows().narrow(0, 3, 2)).toThrow(
      /narrow\(0, 3, 2\) is out of range for \[4, 2\]/,
    )
    expect(() => rows().narrow(0, -1, 2)).toThrow(
      /out of range/,
    )
  })
})

describe("comparisons", () => {
  const a = () => tensor([-1, 0, 1, 2])

  it("produce 1/0 masks", () => {
    expect(a().gt(0).toArray()).toEqual([0, 0, 1, 1])
    expect(a().ge(0).toArray()).toEqual([0, 1, 1, 1])
    expect(a().lt(0).toArray()).toEqual([1, 0, 0, 0])
    expect(a().le(0).toArray()).toEqual([1, 1, 0, 0])
    expect(a().eq(1).toArray()).toEqual([0, 0, 1, 0])
  })

  it("broadcast", () => {
    expect(
      tensor([[1], [3]])
        .gt(tensor([0, 2, 4]))
        .toArray(),
    ).toEqual([
      [1, 0, 0],
      [1, 1, 0],
    ])
  })

  it("stop gradients", () => {
    const x = tensor([-1, 1]).requires_grad()
    const mask = (x as AnyTensor).gt(0)
    expect(mask.gradNode).toBeNull()
    expect(mask.needsGrad).toBe(false)
  })

  it("agree across eager, lazy and native", () => {
    expectAgree(() => a().gt(0).add(a().le(1)).mul(a().eq(2).add(1)))
  })
})

describe("maximum, minimum and clamp", () => {
  it("take the elementwise extreme", () => {
    expect(
      tensor([1, 5])
        .maximum(tensor([4, 2]))
        .toArray(),
    ).toEqual([4, 5])
    expect(
      tensor([1, 5])
        .minimum(tensor([4, 2]))
        .toArray(),
    ).toEqual([1, 2])
    expect(tensor([1, 5]).maximum(3).toArray()).toEqual([
      3,
      5,
    ])
  })

  it("clamp into a range, either end open", () => {
    const x = () => tensor([-3, -1, 0, 1, 3])
    expect(x().clamp(-1, 1).toArray()).toEqual([
      -1,
      -1,
      0,
      1,
      1,
    ])
    expect(x().clamp(0).toArray()).toEqual([0, 0, 0, 1, 3])
    expect(x().clamp(null, 0).toArray()).toEqual([
      -3,
      -1,
      0,
      0,
      0,
    ])
  })

  it("agree across eager, lazy and native", () => {
    expectAgree(() => {
      const x = tensor([-3, -1, 0.5, 2, 4])
      return x.sub(x.clamp(-1, 1)).abs().add(x.maximum(0))
    })
  })
})

describe.skipIf(!isNativeAvailable())(
  "native gather/scatter at a size that uses candle",
  () => {
    it.each(["cpu", "gpu"] as const)(
      "matches eager on the %s device",
      device => {
        const n = 300
        const c = 40
        const e = 2048
        const x = Tensor.rand([n, c]) as AnyTensor
        const index = Tensor.zeros([e]) as AnyTensor
        for (let i = 0; i < e; i++) {
          ;(index.data as Float32Array)[i] = (i * 7) % n
        }
        const build = () =>
          x
            .indexSelect(index)
            .clamp(-0.5, 0.5)
            .scatterAdd(index, n)
            .gt(0.1)
        configure({ lazy: false })
        const eager = build()
        useNative({ device })
        configure({ lazy: true })
        const native = build()
        const a = eager.data
        const b = native.data
        expect(b.length).toBe(a.length)
        for (let i = 0; i < a.length; i++) {
          expect(
            Math.abs(a[i]! - b[i]!),
            `${device} element ${i}`,
          ).toBeLessThan(1e-4)
        }
      },
    )

    // Above LOOP_EVALUATOR_MAX_WORK (65536 elements) the graph runs
    // through candle rather than the tiny-graph loop evaluator, so this
    // covers the other native code path.
    it("matches eager for a large gather/scatter round trip", () => {
      const n = 512
      const c = 64
      const e = 4096
      const x = Tensor.rand([n, c]) as AnyTensor
      const src = Tensor.zeros([e]) as AnyTensor
      const dst = Tensor.zeros([e]) as AnyTensor
      for (let i = 0; i < e; i++) {
        ;(src.data as Float32Array)[i] = (i * 7) % n
        ;(dst.data as Float32Array)[i] = (i * 13) % n
      }
      const build = () =>
        x
          .indexSelect(src)
          .sub(x.indexSelect(dst))
          .tanh()
          .scatterAdd(dst, n)
      configure({ lazy: false })
      const eager = build()
      useNative()
      configure({ lazy: true })
      const native = build()
      expect(native.shape).toEqual([n, c])
      const a = eager.data
      const b = native.data
      let worst = 0
      for (let i = 0; i < a.length; i++) {
        worst = Math.max(worst, Math.abs(a[i]! - b[i]!))
      }
      expect(worst).toBeLessThan(1e-4)
    })
  },
)

// Degenerate shapes: a zero-length dim is legal and the parallel kernels
// divide by it, so a chunk size of zero would panic inside the addon
// rather than return an empty result.
describe("empty dimensions", () => {
  const each = (fn: () => AnyTensor) => {
    for (
      const native of isNativeAvailable()
        ? [false, true]
        : [false]
    ) {
      configure({ lazy: false })
      const eager = fn()
      if (native) useNative()
      configure({ lazy: true })
      const other = fn()
      expect(other.shape).toEqual(eager.shape)
      expect(Array.from(other.data)).toEqual(
        Array.from(eager.data),
      )
      disableNative()
      configure({ lazy: false })
    }
  }

  it("gathers nothing", () => {
    each(() => rows().indexSelect(Tensor.zeros([0]) as any))
  })

  it("scatters into no rows along an inner dim", () => {
    each(() =>
      (Tensor.zeros([3, 0]) as AnyTensor).scatterAdd(
        Tensor.zeros([0]) as any,
        0,
        1,
      )
    )
  })

  it("scatters an empty source into real rows", () => {
    each(() =>
      (Tensor.zeros([0, 2]) as AnyTensor)
        .scatterAdd(Tensor.zeros([0]) as any, 3)
        .add(1)
    )
  })

  it("multiplies matrices with a zero inner dim", () => {
    each(() =>
      (Tensor.zeros([4, 0]) as AnyTensor).matmul(
        Tensor.zeros([0, 3]) as any,
      )
    )
  })

  it("narrows and cats an empty window", () => {
    each(() => Tensor.cat(rows().narrow(1, 0, 0), rows(), 1))
  })
})

describe.skipIf(!isNativeAvailable())(
  "large reductions",
  () => {
    // Summing away the outer dim of a tall matrix takes a BLAS route above
    // 4096 rows, which reassociates the summation. Comparing it to the
    // sequential sum would be the wrong test: a sum of 20000 terms of
    // magnitude 0.5 that cancels down to ~3 has an absolute error set by
    // the *terms*, so the two orders legitimately differ by far more than
    // the result's last digits. What matters is that the new route is no
    // less accurate, so both are measured against an f64 reference.
    const build = (rows: number, cols: number) => {
      const x = Tensor.zeros([rows, cols]) as AnyTensor
      const data = x.data as Float32Array
      for (let i = 0; i < data.length; i++) {
        data[i] = Math.sin(i * 12.9898) * 0.5
      }
      const exact = new Float64Array(cols)
      for (let i = 0; i < rows; i++) {
        for (let c = 0; c < cols; c++) {
          exact[c]! += data[i * cols + c]!
        }
      }
      let terms = 0
      for (const v of data) terms += Math.abs(v)
      return { x, exact, terms: terms / cols }
    }

    it.each([4095, 4096, 20000])(
      "sums %i rows at least as accurately as eager",
      rows => {
        const cols = 12
        const { x, exact, terms } = build(rows, cols)
        configure({ lazy: false })
        const eager = x.sum(0)
        useNative()
        configure({ lazy: true })
        const native = x.sum(0)
        expect(native.shape).toEqual([cols])
        // The f32 error band for a sum of this many terms of this size.
        const band = terms * 1e-5
        for (let c = 0; c < cols; c++) {
          const eagerError = Math.abs(
            eager.data[c]! - exact[c]!,
          )
          const nativeError = Math.abs(
            native.data[c]! - exact[c]!,
          )
          expect(
            nativeError,
            `column ${c}: ${native.data[c]} vs exact ${exact[c]}`,
          ).toBeLessThan(band)
          expect(
            nativeError,
            `column ${c}: native error ${nativeError} vs eager ${eagerError}`,
          ).toBeLessThan(Math.max(eagerError * 8, band))
        }
      },
    )

    it("keeps the dim when asked", () => {
      const cols = 3
      const { x, exact, terms } = build(8192, cols)
      useNative()
      configure({ lazy: true })
      const kept = x.sum(0, true)
      expect(kept.shape).toEqual([1, cols])
      for (let c = 0; c < cols; c++) {
        expect(
          Math.abs(kept.get(0, c) - exact[c]!),
        ).toBeLessThan(terms * 1e-5)
      }
    })

    it("still reduces the inner dim correctly", () => {
      // Short sums, so this one can be exact to f32 rounding.
      const { x } = build(8192, 5)
      configure({ lazy: false })
      const eager = x.sum(1)
      useNative()
      configure({ lazy: true })
      const native = x.sum(1)
      expect(native.shape).toEqual([8192])
      let worst = 0
      for (let i = 0; i < 8192; i++) {
        worst = Math.max(
          worst,
          Math.abs(eager.data[i]! - native.data[i]!),
        )
      }
      expect(worst).toBeLessThan(1e-6)
    })
  },
)
