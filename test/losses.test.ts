import { describe, expect, it } from "vitest"
import { tensor } from "../src/factories.ts"
import { accuracy, crossEntropy, mseLoss } from "../src/nn/index.ts"
import { Tensor } from "../src/tensor.ts"

describe("losses", () => {
  it("mseLoss averages squared error", () => {
    const loss = mseLoss(tensor([1, 2, 3]), tensor([2, 2, 2]))
    expect(loss.item()).toBeCloseTo(2 / 3)
  })

  it("crossEntropy over [B,C]/[B] matches a manual computation", () => {
    const logits = tensor([
      [Math.log(1), Math.log(3)],
      [Math.log(4), Math.log(4)],
    ])
    const loss = crossEntropy(logits, Tensor.indices([1, 0], [2]))
    const expected = -(Math.log(3 / 4) + Math.log(0.5)) / 2
    expect(loss.item()).toBeCloseTo(expected, 5)
  })

  it("validates the batch size", () => {
    expect(() =>
      crossEntropy(
        tensor([[1, 2], [3, 4]]),
        // @ts-expect-error one target per row: batch is 2, not 1. This is
        // also a compile error, and the runtime check is still exercised.
        Tensor.indices([0], [1]),
      )
    ).toThrow(/1 targets for batch of 2/)
  })

  it("validates target range", () => {
    expect(() => crossEntropy(tensor([[1, 2]]), Tensor.indices([5], [1]))).toThrow(
      /out of range/,
    )
  })

  it("crossEntropy over [B,T,V]/[B,T] needs no manual reshape", () => {
    const logitsBTV = tensor([
      [
        [Math.log(1), Math.log(3), Math.log(1)],
        [Math.log(2), Math.log(2), Math.log(2)],
      ],
      [
        [Math.log(1), Math.log(1), Math.log(4)],
        [Math.log(5), Math.log(1), Math.log(1)],
      ],
    ])
    const targets = Tensor.indices([1, 0, 2, 0], [2, 2])
    const loss = crossEntropy(logitsBTV, targets)

    // Same number via hand-flattening to [4,3]/[4].
    const flatLoss = crossEntropy(
      logitsBTV.flatten(0, 1),
      Tensor.indices([1, 0, 2, 0], [4]),
    )
    expect(loss.item()).toBeCloseTo(flatLoss.item(), 6)

    const p = (row: number[], target: number) => {
      const m = Math.max(...row)
      const exps = row.map(v => Math.exp(v - m))
      const sum = exps.reduce((a, b) => a + b, 0)
      return exps[target]! / sum
    }
    const rows = [
      [Math.log(1), Math.log(3), Math.log(1)],
      [Math.log(2), Math.log(2), Math.log(2)],
      [Math.log(1), Math.log(1), Math.log(4)],
      [Math.log(5), Math.log(1), Math.log(1)],
    ]
    const ids = [1, 0, 2, 0]
    const manual = -rows.reduce((acc, row, i) => acc + Math.log(p(row, ids[i]!)), 0) / rows.length
    expect(loss.item()).toBeCloseTo(manual, 5)
  })

  it("ignoreIndex excludes rows from the mean and its gradient", () => {
    const logits = tensor([
      [2, 1, 0.1],
      [0.5, 1.5, -1],
      [1, 1, 1],
    ]).requiresGrad()
    // Row 2's target is the ignoreIndex sentinel.
    const targets = Tensor.indices([0, 1, -100], [3])
    const loss = crossEntropy(logits, targets, { ignoreIndex: -100 })

    const logitsRef = tensor([
      [2, 1, 0.1],
      [0.5, 1.5, -1],
    ]).requiresGrad()
    const lossRef = crossEntropy(logitsRef, Tensor.indices([0, 1], [2]))
    expect(loss.item()).toBeCloseTo(lossRef.item(), 5)

    loss.backward()
    lossRef.backward()
    const grad = logits.grad!.toArray() as number[][]
    const gradRef = logitsRef.grad!.toArray() as number[][]
    for (let i = 0; i < 2; i++) {
      for (let j = 0; j < 3; j++) {
        expect(grad[i]![j]!).toBeCloseTo(gradRef[i]![j]!, 5)
      }
    }
    expect(grad[2]).toEqual([0, 0, 0])
  })

  it("labelSmoothing blends the one-hot target toward uniform", () => {
    const logits = tensor([[2, 1, 0.1]])
    const targets = Tensor.indices([0], [1])
    const sharp = crossEntropy(logits, targets)
    const smoothed = crossEntropy(logits, targets, { labelSmoothing: 0.1 })
    expect(smoothed.item()).not.toBeCloseTo(sharp.item(), 5)
  })

  it("accuracy matches an argmax-by-hand count", () => {
    const logits = tensor([
      [0.1, 0.9],
      [0.8, 0.2],
      [0.3, 0.7],
    ])
    const targets = Tensor.indices([1, 1, 1], [3])
    expect(accuracy(logits, targets)).toBeCloseTo(2 / 3, 6)
  })

  it("accuracy takes the same [B,T,V]/[B,T] target shape as crossEntropy", () => {
    const logits = tensor([
      [[0.1, 0.9], [0.8, 0.2]],
      [[0.3, 0.7], [0.6, 0.4]],
    ])
    const targets = Tensor.indices([1, 0, 1, 1], [2, 2])
    expect(accuracy(logits, targets)).toBeCloseTo(3 / 4, 6)
  })
})
