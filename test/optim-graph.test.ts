import { afterEach, describe, expect, it } from "vitest"
import { disableNative, isNativeAvailable, useNative } from "../src/backends/native.ts"
import { compile } from "../src/compile.ts"
import { tensor } from "../src/factories.ts"
import { configure } from "../src/lazy.ts"
import { Adam, clipGradNorm, SGD } from "../src/optim.ts"
import { Tensor } from "../src/tensor.ts"
import { expectClose } from "./helpers.ts"
import { makeXorNet } from "./xor-net.ts"

type AnyTensor = Tensor<any>

const available = isNativeAvailable()

afterEach(() => {
  configure({ lazy: false })
  disableNative()
})

describe("optimizer dtype validation", () => {
  it("rejects integer parameters", () => {
    const i32 = tensor([1, 2, 3]).to("int32").requiresGrad()
    const i64 = tensor([1, 2, 3]).to("int64").requiresGrad()
    expect(() => new SGD([i32], { lr: 0.1 })).toThrow(/float32 or float64/)
    expect(() => new Adam([i64], { lr: 0.1 })).toThrow(/float32 or float64/)
  })
})

describe("optimizer dispose and reuse", () => {
  it("SGD and Adam keep working after dispose()", () => {
    const sgdParam = tensor([1, 2, 3]).requiresGrad()
    const sgd = new SGD([sgdParam], { lr: 0.1, momentum: 0.9 })
    sgdParam.grad = tensor([0.5, -0.25, 0.1])
    sgd.step()
    sgd.dispose()
    sgdParam.grad = tensor([0.5, -0.25, 0.1])
    expect(() => sgd.step()).not.toThrow()

    const adamParam = tensor([1, 2, 3]).requiresGrad()
    const adam = new Adam([adamParam], { lr: 0.1 })
    adamParam.grad = tensor([0.5, -0.25, 0.1])
    adam.step()
    adam.dispose()
    adamParam.grad = tensor([0.5, -0.25, 0.1])
    expect(() => adam.step()).not.toThrow()
  })
})

describe("optimizer in the lazy graph", () => {
  it("lazy SGD (momentum + weight decay) tracks eager across steps", () => {
    const eager = makeXorNet()
    const lazy = makeXorNet()
    const eagerOpt = new SGD(eager.params, {
      lr: 0.5,
      momentum: 0.9,
      weightDecay: 0.01,
    })
    const lazyOpt = new SGD(lazy.params, {
      lr: 0.5,
      momentum: 0.9,
      weightDecay: 0.01,
    })
    for (let step = 0; step < 30; step++) {
      configure({ lazy: false })
      const eagerLoss = eager.loss()
      eagerOpt.zeroGrad()
      eagerLoss.backward()
      eagerOpt.step()
      configure({ lazy: true })
      const lazyLoss = lazy.loss()
      lazyOpt.zeroGrad()
      lazyLoss.backward()
      lazyOpt.step()
      configure({ lazy: false })
      expect(lazyLoss.item()).toBeCloseTo(
        eagerLoss.item(),
        4,
      )
      eager.params.forEach((p, i) => {
        expectClose(p, lazy.params[i]!)
        expectClose(p.grad!, lazy.params[i]!.grad!)
      })
    }
  })

  it("lazy Adam tracks eager across steps", () => {
    const eager = makeXorNet()
    const lazy = makeXorNet()
    const eagerOpt = new Adam(eager.params, { lr: 0.05 })
    const lazyOpt = new Adam(lazy.params, { lr: 0.05 })
    for (let step = 0; step < 30; step++) {
      configure({ lazy: false })
      const eagerLoss = eager.loss()
      eagerOpt.zeroGrad()
      eagerLoss.backward()
      eagerOpt.step()
      configure({ lazy: true })
      const lazyLoss = lazy.loss()
      lazyOpt.zeroGrad()
      lazyLoss.backward()
      lazyOpt.step()
      configure({ lazy: false })
      expect(lazyLoss.item()).toBeCloseTo(
        eagerLoss.item(),
        4,
      )
      eager.params.forEach((p, i) => expectClose(p, lazy.params[i]!))
    }
  })

  it("keeps eager-mode step numerics unchanged when lazy is off", () => {
    const net = makeXorNet()
    const opt = new SGD(net.params, {
      lr: 0.5,
      momentum: 0.9,
    })
    const loss = net.loss()
    opt.zeroGrad()
    loss.backward()
    opt.step()
    // One eager SGD step with momentum 0.9: v = g, p -= lr * v.
    const g = net.params[0]!.grad!.data
    expect(net.params[0]!.data[0]!).toBeCloseTo(
      0.5 - 0.5 * g[0]!,
      6,
    )
  })
})

describe("compiled training step (forward + backward + optimizer)", () => {
  const compiledRun = (steps: number) => {
    const reference = makeXorNet()
    const compiled = makeXorNet()
    const refOpt = new SGD(reference.params, {
      lr: 0.5,
      momentum: 0.9,
    })
    const opt = new SGD(compiled.params, {
      lr: 0.5,
      momentum: 0.9,
    })
    const step = compile((x: AnyTensor, y: AnyTensor) => {
      const hidden = x
        .matmul(compiled.params[0]!)
        .add(compiled.params[1]!)
        .tanh()
      const out = hidden
        .matmul(compiled.params[2]!)
        .add(compiled.params[3]!)
        .sigmoid()
      const loss = out.sub(y).pow(2).mean()
      opt.zeroGrad()
      loss.backward()
      opt.step()
      return loss
    })
    const refLosses: number[] = []
    const losses: number[] = []
    for (let i = 0; i < steps; i++) {
      configure({ lazy: false })
      const refLoss = reference.loss()
      refOpt.zeroGrad()
      refLoss.backward()
      refOpt.step()
      refLosses.push(refLoss.item())
      losses.push(step(compiled.x, compiled.y).item())
    }
    configure({ lazy: false })
    return { reference, compiled, refLosses, losses }
  }

  it("matches the eager loss trajectory and final params", () => {
    const { reference, compiled, refLosses, losses } = compiledRun(50)
    expect(losses[0]!).toBeCloseTo(refLosses[0]!, 5)
    for (let i = 0; i < losses.length; i++) {
      expect(losses[i]!).toBeCloseTo(refLosses[i]!, 3)
    }
    expect(losses[losses.length - 1]!).toBeLessThan(
      losses[0]!,
    )
    reference.params.forEach((p, i) => expectClose(p, compiled.params[i]!, 1e-3))
    reference.params.forEach((p, i) => expectClose(p.grad!, compiled.params[i]!.grad!, 1e-3))
  })

  it("learns XOR end to end as one compiled graph", () => {
    const { losses } = compiledRun(400)
    expect(losses[losses.length - 1]!).toBeLessThan(0.05)
  })

  // Adam's bias correction depends on the step count, which a graph
  // traced once cannot hold as a constant. It rides along as a leaf
  // instead, so a compiled Adam step has to advance in lockstep with an
  // eager one — including over the first few steps, where the
  // corrections are furthest from 1.
  it("tracks eager Adam step by step when compiled", () => {
    const reference = makeXorNet()
    const compiled = makeXorNet()
    const refOpt = new Adam(reference.params, { lr: 0.05 })
    const opt = new Adam(compiled.params, { lr: 0.05 })
    const step = compile((x: AnyTensor, y: AnyTensor) => {
      const h = x
        .matmul(compiled.params[0]!)
        .add(compiled.params[1]!)
        .tanh()
      const out = h
        .matmul(compiled.params[2]!)
        .add(compiled.params[3]!)
        .sigmoid()
      const loss = out.sub(y).pow(2).mean()
      opt.zeroGrad()
      loss.backward()
      opt.step()
      return loss
    })
    const losses: number[] = []
    for (let i = 0; i < 40; i++) {
      configure({ lazy: false })
      const refLoss = reference.loss()
      refOpt.zeroGrad()
      refLoss.backward()
      refOpt.step()
      const loss = step(compiled.x, compiled.y).item()
      expect(loss, `step ${i + 1}`).toBeCloseTo(
        refLoss.item(),
        4,
      )
      losses.push(loss)
    }
    configure({ lazy: false })
    reference.params.forEach((p, i) => expectClose(p, compiled.params[i]!, 1e-4))
    expect(losses[losses.length - 1]!).toBeLessThan(
      losses[0]!,
    )
  })

  it("clips gradients inside a compiled step", () => {
    // A loss scaled up hard produces gradients far above the clip, so
    // every step is clipped and the parameter moves by exactly
    // lr * maxNorm / ||g|| along the gradient — matching eager.
    const reference = makeXorNet()
    const compiled = makeXorNet()
    const refOpt = new SGD(reference.params, { lr: 0.1 })
    const opt = new SGD(compiled.params, { lr: 0.1 })
    const step = compile((x: AnyTensor, y: AnyTensor) => {
      const h = x
        .matmul(compiled.params[0]!)
        .add(compiled.params[1]!)
        .tanh()
      const out = h
        .matmul(compiled.params[2]!)
        .add(compiled.params[3]!)
        .sigmoid()
      const loss = out.sub(y).pow(2).mean().mul(1000)
      opt.zeroGrad()
      loss.backward()
      clipGradNorm(compiled.params, 1)
      opt.step()
      return loss
    })
    for (let i = 0; i < 10; i++) {
      configure({ lazy: false })
      const refLoss = reference.loss().mul(1000)
      refOpt.zeroGrad()
      refLoss.backward()
      const norm = clipGradNorm(reference.params, 1)
      expect(norm.item()).toBeGreaterThan(1)
      refOpt.step()
      expect(
        step(compiled.x, compiled.y).item(),
      ).toBeCloseTo(refLoss.item(), 2)
    }
    configure({ lazy: false })
    reference.params.forEach((p, i) => expectClose(p, compiled.params[i]!, 1e-3))
  })
})

describe("clipGradNorm", () => {
  it("scales to exactly maxNorm when over", () => {
    const a = Tensor.of([3, 4]).requiresGrad() as AnyTensor
    a.mul(1).sum().backward()
    ;(a.grad!.data as Float32Array).set([3, 4])
    const norm = clipGradNorm([a], 1)
    expect(norm.item()).toBeCloseTo(5, 5)
    expect(a.grad!.get(0)).toBeCloseTo(3 / 5, 5)
    expect(a.grad!.get(1)).toBeCloseTo(4 / 5, 5)
  })

  it("leaves gradients alone when under", () => {
    const a = Tensor.of([1, 1]).requiresGrad() as AnyTensor
    a.mul(1).sum().backward()
    ;(a.grad!.data as Float32Array).set([0.3, 0.4])
    clipGradNorm([a], 10)
    expect(a.grad!.get(0)).toBeCloseTo(0.3, 6)
    expect(a.grad!.get(1)).toBeCloseTo(0.4, 6)
  })

  it("takes the norm across all parameters jointly", () => {
    const a = Tensor.of([0]).requiresGrad() as AnyTensor
    const b = Tensor.of([0]).requiresGrad() as AnyTensor
    a.mul(1).sum().backward()
    b.mul(1).sum().backward()
    ;(a.grad!.data as Float32Array).set([3])
    ;(b.grad!.data as Float32Array).set([4])
    expect(clipGradNorm([a, b], 5).item()).toBeCloseTo(5, 5)
    // already at the limit, so unchanged bar the 1e-6 epsilon
    expect(a.grad!.get(0)).toBeCloseTo(3, 4)
    expect(b.grad!.get(0)).toBeCloseTo(4, 4)
  })

  it("matches eager in lazy mode", () => {
    const build = () => {
      const a = Tensor.of([
        1,
        2,
        3,
      ]).requiresGrad() as AnyTensor
      a.pow(3).sum().mul(10).backward()
      clipGradNorm([a], 2)
      return a
    }
    configure({ lazy: false })
    const eager = build()
    configure({ lazy: true })
    const lazy = build()
    configure({ lazy: false })
    expectClose(eager.grad!, lazy.grad!, 1e-5)
  })

  it("rejects a non-positive maxNorm", () => {
    const a = Tensor.of([1]).requiresGrad() as AnyTensor
    a.mul(1).sum().backward()
    expect(() => clipGradNorm([a], 0)).toThrow(
      /maxNorm must be positive/,
    )
  })
})

describe.skipIf(!available)(
  "compiled training step (native)",
  () => {
    it("matches eager through the native path", () => {
      useNative()
      const reference = makeXorNet()
      const compiled = makeXorNet()
      const refOpt = new SGD(reference.params, {
        lr: 0.5,
        momentum: 0.9,
      })
      const opt = new SGD(compiled.params, {
        lr: 0.5,
        momentum: 0.9,
      })
      const step = compile((x: AnyTensor, y: AnyTensor) => {
        const h = x
          .matmul(compiled.params[0]!)
          .add(compiled.params[1]!)
          .tanh()
        const out = h
          .matmul(compiled.params[2]!)
          .add(compiled.params[3]!)
          .sigmoid()
        const loss = out.sub(y).pow(2).mean()
        opt.zeroGrad()
        loss.backward()
        opt.step()
        return loss
      })
      for (let i = 0; i < 20; i++) {
        configure({ lazy: false })
        disableNative()
        const refLoss = reference.loss()
        refOpt.zeroGrad()
        refLoss.backward()
        refOpt.step()
        useNative()
        const loss = step(compiled.x, compiled.y)
        expect(loss.item()).toBeCloseTo(refLoss.item(), 3)
      }
      disableNative()
      configure({ lazy: false })
      reference.params.forEach((p, i) => expectClose(p, compiled.params[i]!, 1e-3))
      reference.params.forEach((p, i) =>
        expectClose(
          p.grad!,
          compiled.params[i]!.grad!,
          1e-3,
        )
      )
    })

    it("matches eager for compiled Adam with clipping", () => {
      const reference = makeXorNet()
      const compiled = makeXorNet()
      const refOpt = new Adam(reference.params, {
        lr: 0.05,
      })
      const opt = new Adam(compiled.params, { lr: 0.05 })
      const step = compile((x: AnyTensor, y: AnyTensor) => {
        const h = x
          .matmul(compiled.params[0]!)
          .add(compiled.params[1]!)
          .tanh()
        const out = h
          .matmul(compiled.params[2]!)
          .add(compiled.params[3]!)
          .sigmoid()
        const loss = out.sub(y).pow(2).mean()
        opt.zeroGrad()
        loss.backward()
        clipGradNorm(compiled.params, 0.5)
        opt.step()
        return loss
      })
      for (let i = 0; i < 20; i++) {
        configure({ lazy: false })
        disableNative()
        const refLoss = reference.loss()
        refOpt.zeroGrad()
        refLoss.backward()
        clipGradNorm(reference.params, 0.5)
        refOpt.step()
        useNative()
        expect(
          step(compiled.x, compiled.y).item(),
          `step ${i + 1}`,
        ).toBeCloseTo(refLoss.item(), 4)
      }
      disableNative()
      configure({ lazy: false })
      reference.params.forEach((p, i) => expectClose(p, compiled.params[i]!, 1e-4))
    })
  },
)
