import { describe, expect, it } from "vitest"
import { Linear, Module } from "../src/nn.ts"
import { Adam, AdamW, constant, cosine, linearDecay, oneCycle, Optimizer, SGD, stepDecay, warmup, warmupCosine } from "../src/optim.ts"
import { type AnyTensor, Tensor } from "../src/tensor.ts"

const closeArray = (actual: number[], expected: number[], digits = 6) => {
  expect(actual).toHaveLength(expected.length)
  actual.forEach((v, i) => expect(v, `index ${i}`).toBeCloseTo(expected[i]!, digits))
}

describe("optimizer lr is public and mutable", () => {
  it("SGD.lr and Adam.lr can be reassigned directly, no cast needed", () => {
    const sgd = new SGD([Tensor.of([1]).requiresGrad() as AnyTensor], { lr: 0.1 })
    sgd.lr = 0.2
    expect(sgd.lr).toBe(0.2)

    const adam = new Adam([Tensor.of([1]).requiresGrad() as AnyTensor], { lr: 0.1 })
    adam.lr = 0.05
    expect(adam.lr).toBe(0.05)
  })

  it("a reassigned lr takes effect on the very next eager step()", () => {
    const p = Tensor.of([1]).requiresGrad() as AnyTensor
    p.mul(1).sum().backward()
    const opt = new SGD([p], { lr: 1e-3 })
    ;(p.grad!.data as Float32Array).set([1])
    opt.step()
    const after1 = p.data[0]!
    opt.lr = 1e-1
    ;(p.grad!.data as Float32Array).set([1])
    opt.step()
    const after2 = p.data[0]!
    // step 1 moved by 1e-3, step 2 by 1e-1 — a 100x jump.
    expect(Math.abs(after1 - 1)).toBeCloseTo(1e-3, 6)
    expect(Math.abs(after2 - after1)).toBeCloseTo(1e-1, 6)
  })
})

describe("AdamW (decoupled weight decay)", () => {
  it("matches a hand-computed decoupled-decay reference over 5 steps at 1e-6", () => {
    const lr = 0.1
    const beta1 = 0.9
    const beta2 = 0.999
    const eps = 1e-8
    const wd = 0.1
    const g = 0.5

    // Independent reference: Loshchilov & Hutter 2019, eq. 12 —
    // theta_t = theta_{t-1} - lr*wd*theta_{t-1} - lr*mHat/(sqrt(vHat)+eps).
    // `Math.fround` after each step mirrors the real parameter's float32
    // storage (the moment estimates below stay double precision, exactly
    // like the library's own Float64Array state).
    let refP = Math.fround(1)
    let refM = 0
    let refV = 0
    const refHistory: number[] = []
    for (let t = 1; t <= 5; t++) {
      refM = beta1 * refM + (1 - beta1) * g
      refV = beta2 * refV + (1 - beta2) * g * g
      const mHat = refM / (1 - beta1 ** t)
      const vHat = refV / (1 - beta2 ** t)
      const delta = lr * mHat / (Math.sqrt(vHat) + eps) + lr * wd * refP
      refP = Math.fround(refP - delta)
      refHistory.push(refP)
    }

    const p = Tensor.of([1]).requiresGrad() as AnyTensor
    p.mul(1).sum().backward()
    const opt = new AdamW([p], { lr, betas: [beta1, beta2], eps, weightDecay: wd })
    for (let t = 0; t < 5; t++) {
      ;(p.grad!.data as Float32Array).set([g])
      opt.step()
      expect(p.data[0]!, `step ${t + 1}`).toBeCloseTo(refHistory[t]!, 6)
    }
  })

  it("defaults weightDecay to 0.01 (matching PyTorch)", () => {
    const p1 = Tensor.of([1]).requiresGrad() as AnyTensor
    p1.mul(1).sum().backward()
    const withDefault = new AdamW([p1], { lr: 0.1 })
    ;(p1.grad!.data as Float32Array).set([0])
    withDefault.step()
    // Zero gradient: only the decay term should move the parameter.
    expect(p1.data[0]!).toBeCloseTo(1 - 0.1 * 0.01 * 1, 6)
  })

  it("with weightDecay: 0, AdamW matches plain Adam exactly", () => {
    const pA = Tensor.of([1]).requiresGrad() as AnyTensor
    const pW = Tensor.of([1]).requiresGrad() as AnyTensor
    pA.mul(1).sum().backward()
    pW.mul(1).sum().backward()
    const adam = new Adam([pA], { lr: 0.05, weightDecay: 0 })
    const adamW = new AdamW([pW], { lr: 0.05, weightDecay: 0 })
    for (let t = 0; t < 5; t++) {
      ;(pA.grad!.data as Float32Array).set([0.3])
      ;(pW.grad!.data as Float32Array).set([0.3])
      adam.step()
      adamW.step()
      expect(pW.data[0]!).toBeCloseTo(pA.data[0]!, 7)
    }
  })
})

describe("schedules", () => {
  it("constant is flat regardless of step", () => {
    const s = constant(3)
    closeArray([0, 1, 100].map(s), [3, 3, 3])
  })

  it("cosine anneals base -> min and holds", () => {
    const s = cosine({ base: 10, steps: 4 })
    closeArray(
      [0, 1, 2, 3, 4, 5].map(s),
      [10, 8.53553391, 5, 1.46446609, 0, 0],
    )
  })

  it("linearDecay anneals base -> min and holds", () => {
    const s = linearDecay({ base: 10, steps: 4 })
    closeArray([0, 1, 2, 3, 4, 5].map(s), [10, 7.5, 5, 2.5, 0, 0])
  })

  it("stepDecay multiplies by gamma every `every` steps", () => {
    const s = stepDecay({ base: 1, every: 2, gamma: 0.5 })
    closeArray([0, 1, 2, 3, 4].map(s), [1, 1, 0.5, 0.5, 0.25])
  })

  it("warmup ramps linearly to inner(0) then hands off to inner", () => {
    const s = warmup(constant(10), 4)
    closeArray([0, 1, 2, 3, 4, 5].map(s), [2.5, 5, 7.5, 10, 10, 10])
  })

  it("warmupCosine composes warmup + cosine", () => {
    const s = warmupCosine({ base: 10, warmupSteps: 2, totalSteps: 6 })
    closeArray(
      [0, 1, 2, 3, 4, 5, 6].map(s),
      [5, 10, 10, 8.53553391, 5, 1.46446609, 0],
    )
  })

  it("oneCycle ramps up then anneals down", () => {
    const s = oneCycle({ base: 10, steps: 10 })
    closeArray(
      Array.from({ length: 10 }, (_, i) => i).map(s),
      [
        0.4,
        2.8,
        7.6,
        10,
        9.50484632,
        8.11745654,
        6.11262022,
        3.88741978,
        1.88258346,
        0.49519368,
      ],
    )
  })
})

describe("optimizer parameterEpoch guard", () => {
  it("adding a submodule after constructing the optimizer throws at the next step()", () => {
    class Net extends Module {
      a = new Linear(2, 2)
    }
    const net = new Net()
    const opt = new SGD(net, { lr: 0.1 }) // Structural mutation after construction: a new submodule appears.
    ;(net as unknown as { b: Linear<2, 2> }).b = new Linear(2, 2)
    expect(() => opt.step()).toThrow(/parameter set changed/)
  })

  it("does not throw when the module's parameter set is untouched", () => {
    class Net extends Module {
      a = new Linear(2, 2)
    }
    const net = new Net()
    const opt = new SGD(net, { lr: 0.1 })
    expect(() => opt.step()).not.toThrow()
  })

  it("a plain AnyTensor[] source has no module to drift, so it never throws", () => {
    const p = Tensor.of([1]).requiresGrad() as AnyTensor
    const opt = new SGD([p], { lr: 0.1 })
    expect(() => opt.step()).not.toThrow()
  })

  it("building from a Module collects the same parameters as .parameters()", () => {
    class Net extends Module {
      a = new Linear(2, 3)
    }
    const net = new Net()
    const opt = new SGD(net, { lr: 0.1 }) as unknown as { params: AnyTensor[] }
    expect(opt.params).toHaveLength(net.parameters().length)
  })
})

describe("maxGradNorm option", () => {
  it("clips inside step() automatically when set", () => {
    const p = Tensor.of([3, 4]).requiresGrad() as AnyTensor
    p.mul(1).sum().backward()
    ;(p.grad!.data as Float32Array).set([3, 4]) // norm 5
    const opt = new SGD([p], { lr: 0, maxGradNorm: 1 })
    opt.step()
    // lr:0 means the parameter itself never moves; check the grad was scaled.
    expect(p.grad!.get(0)).toBeCloseTo(3 / 5, 5)
    expect(p.grad!.get(1)).toBeCloseTo(4 / 5, 5)
  })

  it("leaves gradients alone when maxGradNorm is not set", () => {
    const p = Tensor.of([3, 4]).requiresGrad() as AnyTensor
    p.mul(1).sum().backward()
    ;(p.grad!.data as Float32Array).set([3, 4])
    const opt = new SGD([p], { lr: 0 })
    opt.step()
    expect(p.grad!.get(0)).toBeCloseTo(3, 5)
    expect(p.grad!.get(1)).toBeCloseTo(4, 5)
  })
})

describe("optimizer stateDict / loadStateDict", () => {
  it("SGD round-trips momentum state by parameter name", () => {
    class Net extends Module {
      a = new Linear(2, 2)
    }
    const src = new Net()
    const opt = new SGD(src, { lr: 0.1, momentum: 0.9 })
    src.a.weight.mul(1).sum().backward()
    ;(src.a.weight.grad!.data as Float32Array).fill(1)
    opt.step()
    const saved = opt.stateDict()
    expect(Object.keys(saved.state)).toContain("a.weight")

    const dst = new Net()
    const restored = new SGD(dst, { lr: 0.1, momentum: 0.9 })
    restored.loadStateDict(saved)
    expect(restored.stateDict().state["a.weight"]!.velocity).toEqual(
      saved.state["a.weight"]!.velocity,
    )
  })

  it("Adam round-trips m/v/step by parameter name", () => {
    class Net extends Module {
      a = new Linear(2, 2)
    }
    const src = new Net()
    const opt = new Adam(src, { lr: 0.1 })
    src.a.weight.mul(1).sum().backward()
    ;(src.a.weight.grad!.data as Float32Array).fill(1)
    opt.step()
    opt.step()
    const saved = opt.stateDict()
    expect(saved.step).toBe(2)

    const dst = new Net()
    const restored = new Adam(dst, { lr: 0.1 })
    restored.loadStateDict(saved)
    const restoredState = restored.stateDict()
    expect(restoredState.step).toBe(2)
    expect(restoredState.state["a.weight"]!.m).toEqual(saved.state["a.weight"]!.m)
    expect(restoredState.state["a.weight"]!.v).toEqual(saved.state["a.weight"]!.v)
  })
})

describe("Optimizer base", () => {
  it("still rejects integer-dtype parameters (unchanged behaviour)", () => {
    const i32 = Tensor.of([1, 2, 3]).to("int32").requiresGrad() as AnyTensor
    expect(() => new SGD([i32], { lr: 0.1 })).toThrow(/float32 or float64/)
  })
})
