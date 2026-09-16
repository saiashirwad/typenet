"use tsover"

import { Linear, randn, ReLU, sequential, Tensor, tensor } from "../index.ts"

export const FEATURES = 784
export const CLASSES = 10

export interface Split {
  x: Tensor<[number, typeof FEATURES]>
  labels: number[]
}

export function mlpModel() {
  return sequential(
    new Linear(FEATURES, 256),
    new ReLU(),
    new Linear(256, CLASSES),
  )
}

/** Gaussian clusters around `CLASSES` random prototypes, so the example runs offline and replays from the configured seed. */
export function makeData(train: number, test: number): { train: Split; test: Split } {
  const prototypes = randn([CLASSES, FEATURES]) * 0.12

  const split = (n: number): Split => {
    const labels = Array.from({ length: n }, (_, i) => i % CLASSES)
    // A seeded shuffle, since Math.random would not replay from the configured seed.
    for (let i = labels.length - 1; i > 0; i--) {
      const j = Math.floor(((i * 1103515245 + 12345) % 2147483648) / 2147483648 * (i + 1))
      ;[labels[i], labels[j]] = [labels[j]!, labels[i]!]
    }
    return { x: tensor(labels).oneHot(CLASSES).matmul(prototypes) + randn([n, FEATURES]), labels }
  }

  return { train: split(train), test: split(test) }
}
