import { afterEach, describe, expect, it } from "vitest"
import { disableNative, isNativeAvailable, preparedGraphCountNative, useNative } from "../src/backends/native.ts"
import { Linear, mseLoss, sequential, Tanh } from "../src/nn.ts"
import { compile, configure, Tensor, tensor } from "../src/tensor.ts"

type AnyTensor = Tensor<any>

const available = isNativeAvailable()

function expectClose(a: AnyTensor, b: AnyTensor): void {
  expect(b.shape).toEqual(a.shape)
  const ad = a.data
  const bd = b.data
  expect(bd.length).toBe(ad.length)
  for (let i = 0; i < ad.length; i++) {
    expect(Math.abs(ad[i]! - bd[i]!)).toBeLessThan(1e-4)
  }
}

function forwardEager(
  x: AnyTensor,
  w: AnyTensor,
): AnyTensor {
  return x.matmul(w).tanh().pow(2).sum(1)
}

const xData = [
  [1, 2],
  [3, 4],
  [5, 6],
]
const wData = [
  [0.5, -1, 0.25],
  [1.5, 0.75, -0.5],
]

afterEach(() => {
  configure({ lazy: false })
  disableNative()
})

describe("compile (interpreter)", () => {
  it("matches eager numerically (matmul/unary/reduce)", () => {
    const compiled = compile((x: AnyTensor, w: AnyTensor) => forwardEager(x, w))
    const result = compiled(tensor(xData), tensor(wData))
    expectClose(
      forwardEager(tensor(xData), tensor(wData)),
      result,
    )
  })

  it("replays with swapped data and gives updated results", () => {
    const compiled = compile((x: AnyTensor, w: AnyTensor) => forwardEager(x, w))
    const first = compiled(tensor(xData), tensor(wData))
    expectClose(
      forwardEager(tensor(xData), tensor(wData)),
      first,
    )
    const x2 = tensor([
      [0, 1],
      [1, 0],
      [2, 2],
    ])
    const w2 = tensor(wData).mul(2)
    expectClose(forwardEager(x2, w2), compiled(x2, w2))
    const x3 = Float32Array.of(1, 1, 1, 1, 1, 1)
    const x3Tensor = Tensor.zeros([3, 2])
    x3Tensor.data.set(x3)
    expectClose(
      forwardEager(x3Tensor, tensor(wData)),
      compiled(x3, tensor(wData)),
    )
    expectClose(
      forwardEager(tensor(xData), tensor(wData)),
      first,
    )
  })

  it("supports multiple outputs", () => {
    const compiled = compile(
      (x: AnyTensor, w: AnyTensor) => [
        x.matmul(w),
        x.sum(1),
      ],
    )
    const [mm, rs] = compiled(tensor(xData), tensor(wData))
    expectClose(tensor(xData).matmul(tensor(wData)), mm)
    expectClose(tensor(xData).sum(1), rs)
  })

  it("sees in-place updates to captured parameters", () => {
    const w = tensor(wData).requires_grad()
    const compiled = compile((x: AnyTensor) => x.matmul(w).sum())
    const x = tensor(xData)
    const before = compiled(x).item()
    w.data.set(tensor(wData).mul(2).data)
    const after = compiled(x).item()
    expect(after).toBeCloseTo(2 * before, 4)
  })

  it("errors on shape mismatch between calls", () => {
    const compiled = compile((x: AnyTensor) => x.sum())
    compiled(tensor(xData))
    expect(() => compiled(tensor([1, 2, 3]))).toThrow(
      /expected shape \[3, 2\], got \[3\]/,
    )
  })

  it("errors on argument count mismatch", () => {
    const compiled = compile((x: AnyTensor, w: AnyTensor) => x.matmul(w))
    compiled(tensor(xData), tensor(wData))
    expect(() => (compiled as any)(tensor(xData))).toThrow(
      /expected 2 arguments, got 1/,
    )
  })

  it("errors when the first call passes a flat buffer", () => {
    const compiled = compile((x: AnyTensor) => x.sum())
    expect(() => compiled(Float32Array.of(1, 2, 3) as any)).toThrow(/must be a Tensor/)
  })

  it("traces under lazy semantics regardless of the global flag", () => {
    configure({ lazy: true })
    const compiled = compile((x: AnyTensor) => x.mul(2).add(1))
    configure({ lazy: false })
    const result = compiled(tensor([1, 2, 3]))
    expect(result.toArray()).toEqual([3, 5, 7])
    expect(tensor([1]).add(1).item()).toBe(2)
  })

  it("runs a compiled XOR-style forward step against eager", () => {
    const makeNet = () =>
      sequential(
        new Linear(2, 8),
        new Tanh(),
        new Linear(8, 1),
      )
    const eagerNet = makeNet()
    const compiledNet = makeNet()
    for (
      const [p, q] of eagerNet
        .parameters()
        .map(
          (p, i) => [p, compiledNet.parameters()[i]!] as const,
        )
    ) {
      q.data.set(p.data)
    }
    const x = tensor([
      [0, 0],
      [0, 1],
      [1, 0],
      [1, 1],
    ])
    const t = tensor([[0], [1], [1], [0]])
    const step = compile((xIn: AnyTensor, tIn: AnyTensor) => mseLoss(compiledNet.forward(xIn), tIn))
    const loss = step(x, t)
    expectClose(mseLoss(eagerNet.forward(x), t), loss)
    for (const p of compiledNet.parameters()) {
      p.data.set(p.data.map(v => v * 0.9))
    }
    for (const p of eagerNet.parameters()) {
      p.data.set(p.data.map(v => v * 0.9))
    }
    expectClose(mseLoss(eagerNet.forward(x), t), step(x, t))
  })
})

describe.skipIf(!available)("compile (native)", () => {
  it("matches eager numerically through the native path", () => {
    useNative()
    const compiled = compile((x: AnyTensor, w: AnyTensor) => forwardEager(x, w))
    expectClose(
      forwardEager(tensor(xData), tensor(wData)),
      compiled(tensor(xData), tensor(wData)),
    )
  })

  it("replays with swapped data through the native path", () => {
    useNative()
    const compiled = compile((x: AnyTensor, w: AnyTensor) => forwardEager(x, w))
    compiled(tensor(xData), tensor(wData))
    const x2 = tensor([
      [2, 0],
      [0, 2],
      [1, 1],
    ])
    const w2 = tensor(wData).mul(0.5)
    expectClose(forwardEager(x2, w2), compiled(x2, w2))
  })

  it("matches eager for a compiled XOR-style forward step", () => {
    useNative()
    const net = sequential(
      new Linear(2, 8),
      new Tanh(),
      new Linear(8, 1),
    )
    const x = tensor([
      [0, 0],
      [0, 1],
      [1, 0],
      [1, 1],
    ])
    const t = tensor([[0], [1], [1], [0]])
    const step = compile((xIn: AnyTensor, tIn: AnyTensor) => mseLoss(net.forward(xIn), tIn))
    const loss = step(x, t)
    configure({ lazy: false })
    const reference = mseLoss(net.forward(x), t)
    expectClose(reference, loss)
    step.dispose()
  })

  it("dispose releases the native prepared-graph handle", () => {
    useNative()
    const before = preparedGraphCountNative()
    const compiled = []
    for (let i = 0; i < 16; i++) {
      // Distinct graphs so each prepare allocates its own handle.
      const scale = i + 1
      const fn = compile((x: AnyTensor) => x.mul(scale).sum())
      fn(tensor([1, 2, 3, 4]))
      compiled.push(fn)
    }
    expect(preparedGraphCountNative()).toBe(before + 16)
    for (const fn of compiled) fn.dispose()
    expect(preparedGraphCountNative()).toBe(before)
    for (const fn of compiled) fn.dispose()
    expect(preparedGraphCountNative()).toBe(before)
  })
})
