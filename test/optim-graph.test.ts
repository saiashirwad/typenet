import { afterEach, describe, expect, it } from "vitest"
import {
  Tensor,
  compile,
  configure,
  tensor
} from "../src/tensor.ts"
import { Adam, SGD } from "../src/optim.ts"
import {
  disableNative,
  isNativeAvailable,
  useNative
} from "../src/backends/native.ts"

type AnyTensor = Tensor<any, any>

const available = isNativeAvailable()

afterEach(() => {
  configure({ lazy: false })
  disableNative()
})

function expectClose(
  a: AnyTensor,
  b: AnyTensor,
  tol = 1e-4
): void {
  expect(b.shape).toEqual(a.shape)
  const ad = a.data
  const bd = b.data
  expect(bd.length).toBe(ad.length)
  for (let i = 0; i < ad.length; i++)
    expect(Math.abs(ad[i]! - bd[i]!)).toBeLessThan(tol)
}

// A small fixed-init MLP step (matmul/bias/tanh/sigmoid + mse), run
// in either mode; returns the loss and the (mutated) parameters.
function makeNet() {
  const x = tensor([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
  ])
  const y = tensor([[0], [1], [1], [0]])
  const w1 = tensor([
    [0.5, -0.5, 0.25, -0.25],
    [0.1, 0.2, -0.3, 0.4]
  ]).requires_grad()
  const b1 = tensor([
    0.1, -0.1, 0.05, -0.05
  ]).requires_grad()
  const w2 = tensor([
    [0.6],
    [-0.6],
    [0.3],
    [-0.3]
  ]).requires_grad()
  const b2 = tensor([0.2]).requires_grad()
  const params = [w1, b1, w2, b2] as AnyTensor[]
  const forward = () => {
    const h = x.matmul(w1).add(b1).tanh()
    return h.matmul(w2).add(b2).sigmoid()
  }
  const loss = () => forward().sub(y).pow(2).mean()
  return { x, y, params, forward, loss }
}

function cloneParams(params: AnyTensor[]): AnyTensor[] {
  return params.map(p =>
    (Tensor.zeros(p.shape) as AnyTensor)
      .write(p.data)
      .requires_grad()
  )
}

describe("optimizer in the lazy graph", () => {
  it("lazy SGD (momentum + weight decay) tracks eager across steps", () => {
    const eager = makeNet()
    const lazy = makeNet()
    const eagerOpt = new SGD(eager.params, {
      lr: 0.5,
      momentum: 0.9,
      weightDecay: 0.01
    })
    const lazyOpt = new SGD(lazy.params, {
      lr: 0.5,
      momentum: 0.9,
      weightDecay: 0.01
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
        4
      )
      eager.params.forEach((p, i) => {
        expectClose(p, lazy.params[i]!)
        expectClose(p.grad!, lazy.params[i]!.grad!)
      })
    }
  })

  it("lazy Adam tracks eager across steps", () => {
    const eager = makeNet()
    const lazy = makeNet()
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
        4
      )
      eager.params.forEach((p, i) =>
        expectClose(p, lazy.params[i]!)
      )
    }
  })

  it("keeps eager-mode step numerics unchanged when lazy is off", () => {
    const net = makeNet()
    const opt = new SGD(net.params, {
      lr: 0.5,
      momentum: 0.9
    })
    const loss = net.loss()
    opt.zeroGrad()
    loss.backward()
    opt.step()
    // One eager SGD step with momentum 0.9: v = g, p -= lr * v.
    const g = net.params[0]!.grad!.data
    expect(net.params[0]!.data[0]!).toBeCloseTo(
      0.5 - 0.5 * g[0]!,
      6
    )
  })
})

describe("compiled training step (forward + backward + optimizer)", () => {
  const compiledRun = (steps: number) => {
    const reference = makeNet()
    const compiled = makeNet()
    const refOpt = new SGD(reference.params, {
      lr: 0.5,
      momentum: 0.9
    })
    const opt = new SGD(compiled.params, {
      lr: 0.5,
      momentum: 0.9
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
    const { reference, compiled, refLosses, losses } =
      compiledRun(50)
    // Loss decreases and tracks eager closely at every step.
    expect(losses[0]!).toBeCloseTo(refLosses[0]!, 5)
    for (let i = 0; i < losses.length; i++)
      expect(losses[i]!).toBeCloseTo(refLosses[i]!, 3)
    expect(losses[losses.length - 1]!).toBeLessThan(
      losses[0]!
    )
    reference.params.forEach((p, i) =>
      expectClose(p, compiled.params[i]!, 1e-3)
    )
    // Grads after a compiled step see this step's values.
    reference.params.forEach((p, i) =>
      expectClose(p.grad!, compiled.params[i]!.grad!, 1e-3)
    )
  })

  it("learns XOR end to end as one compiled graph", () => {
    const { losses } = compiledRun(400)
    expect(losses[losses.length - 1]!).toBeLessThan(0.05)
  })

  it("rejects Adam inside compile() with a clear error", () => {
    const net = makeNet()
    const opt = new Adam(net.params, {})
    const step = compile((x: AnyTensor, y: AnyTensor) => {
      const h = x
        .matmul(net.params[0]!)
        .add(net.params[1]!)
        .tanh()
      const out = h
        .matmul(net.params[2]!)
        .add(net.params[3]!)
        .sigmoid()
      const loss = out.sub(y).pow(2).mean()
      opt.zeroGrad()
      loss.backward()
      opt.step()
      return loss
    })
    expect(() => step(net.x, net.y)).toThrow(
      /Adam\.step\(\) inside compile\(\) is not supported/
    )
  })
})

describe.skipIf(!available)(
  "compiled training step (native)",
  () => {
    it("matches eager through the native path", () => {
      useNative()
      const reference = makeNet()
      const compiled = makeNet()
      const refOpt = new SGD(reference.params, {
        lr: 0.5,
        momentum: 0.9
      })
      const opt = new SGD(compiled.params, {
        lr: 0.5,
        momentum: 0.9
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
      reference.params.forEach((p, i) =>
        expectClose(p, compiled.params[i]!, 1e-3)
      )
      reference.params.forEach((p, i) =>
        expectClose(
          p.grad!,
          compiled.params[i]!.grad!,
          1e-3
        )
      )
    })
  }
)
