import { describe, expect, it, vi } from "vitest"
import { Linear } from "../src/nn/index.ts"
import { isParameter, type LoadReport, Module, type Parameter, parameter, tie } from "../src/nn/index.ts"
import { SGD } from "../src/optim/index.ts"
import { type AnyTensor, Tensor } from "../src/tensor.ts"

describe("Module.parameters dedup", () => {
  it("a literally shared weight is counted once and updated once per step (not double)", () => {
    class Net extends Module {
      a: Parameter<[1]>
      b: Parameter<[1]>
      constructor() {
        super()
        const w = parameter(Tensor.zeros([1]))
        w.fill_(3)
        this.a = w
        this.b = w
      }
    }
    const net = new Net()
    expect(net.parameters()).toHaveLength(1)
    expect([...net.namedParameters().keys()]).toEqual(["a"])

    const opt = new SGD(net.parameters(), { lr: 0.1 })
    // loss = a, so d(loss)/da = 1 and a correct step moves a by exactly -lr; double-counting
    // the shared object would move it by -2*lr.
    opt.zeroGrad()
    net.a.backward()
    opt.step()
    expect(net.a.item()).toBeCloseTo(3 - 0.1, 6)
  })

  it("tie() dedups two independently-created parameters by storage", () => {
    class Net extends Module {
      a: Parameter<[2]>
      b: Parameter<[2]>
      constructor() {
        super()
        this.a = parameter(Tensor.zeros([2]))
        this.b = parameter(Tensor.zeros([2]))
        tie(this.a, this.b)
      }
    }
    const net = new Net()
    expect(net.parameters()).toHaveLength(1)
    expect([...net.namedParameters().keys()]).toEqual(["a"])
  })
})

describe("Module.stateDict / loadStateDict", () => {
  it("is name-keyed: reordering field declarations does not permute a loaded checkpoint", () => {
    class NetA extends Module {
      x: Parameter<[2]>
      y: Parameter<[3]>
      constructor() {
        super()
        this.x = parameter(Tensor.zeros([2]))
        this.y = parameter(Tensor.zeros([3]))
        this.x.fill_(1)
        this.y.fill_(2)
      }
    }
    // Same fields, declared in the opposite order.
    class NetB extends Module {
      y: Parameter<[3]>
      x: Parameter<[2]>
      constructor() {
        super()
        this.y = parameter(Tensor.zeros([3]))
        this.x = parameter(Tensor.zeros([2]))
      }
    }
    const a = new NetA()
    const dict = a.stateDict()
    const b = new NetB()
    b.loadStateDict(dict)
    expect(b.x.toArray()).toEqual([1, 1])
    expect(b.y.toArray()).toEqual([2, 2, 2])
  })

  it("registerBuffer contributes to stateDict but never to parameters", () => {
    class Net extends Module {
      w = parameter(Tensor.zeros([2]))
      mask: AnyTensor
      constructor() {
        super()
        this.mask = this.registerBuffer("mask", Tensor.ones([2]))
      }
    }
    const net = new Net()
    expect(net.namedParameters().has("mask")).toBe(false)
    expect(net.namedBuffers().has("mask")).toBe(true)
    const dict = net.stateDict()
    expect(dict["mask"]?.data).toBeDefined()
    expect(dict["w"]?.data).toBeDefined()
  })

  it("reports missing/unexpected/mismatched and throws by default (strict)", () => {
    class Net extends Module {
      w = parameter(Tensor.zeros([2]))
    }
    const net = new Net()
    expect(() => net.loadStateDict({})).toThrow(/missing/)

    const report: LoadReport = net.loadStateDict({}, { strict: false })
    expect(report.missing).toEqual(["w"])
    expect(report.unexpected).toEqual([])
    expect(report.mismatched).toEqual([])

    const wrongShape = { w: { shape: [3], dtype: "float32" as const, data: new Float32Array(3) } }
    const report2 = net.loadStateDict(wrongShape, { strict: false })
    expect(report2.mismatched).toEqual(["w"])

    const extra = {
      w: { shape: [2], dtype: "float32" as const, data: new Float32Array(2) },
      ghost: { shape: [1], dtype: "float32" as const, data: new Float32Array(1) },
    }
    const report3 = net.loadStateDict(extra, { strict: false })
    expect(report3.unexpected).toEqual(["ghost"])
  })
})

describe("Module register() and the unreachable-parameter warning", () => {
  class Net extends Module {
    #registered: Parameter<[2]>
    #unregistered: Parameter<[2]>
    constructor() {
      super()
      this.#registered = this.register("registered", parameter(Tensor.zeros([2])))
      this.#unregistered = parameter(Tensor.zeros([2]))
    }
    get registered(): Parameter<[2]> {
      return this.#registered
    }
    get unregistered(): Parameter<[2]> {
      return this.#unregistered
    }
  }

  it("collects a #private field registered with register()", () => {
    const net = new Net()
    expect(net.namedParameters().has("registered")).toBe(true)
    expect(isParameter(net.registered)).toBe(true)
  })

  it("warns once, naming the field, for a #private field that was not registered", () => {
    const warnSpy = vi.spyOn(console, "warn").mockImplementation(() => {})
    try {
      const net = new Net()
      const params = net.namedParameters()
      expect(params.has("unregistered")).toBe(false)
      expect(warnSpy).toHaveBeenCalledTimes(1)
      expect(warnSpy.mock.calls[0]?.[0]).toContain("unregistered")
      // One-time: a second read does not warn again.
      net.namedParameters()
      expect(warnSpy).toHaveBeenCalledTimes(1)
    } finally {
      warnSpy.mockRestore()
    }
  })
})

describe("Module traversal", () => {
  it("namedParameters uses dotted paths, in declaration order, through nesting and arrays", () => {
    class Net extends Module {
      a = new Linear(4, 3)
      b = new Linear(3, 2)
      list = [new Linear(2, 2)]
    }
    const net = new Net()
    expect([...net.namedParameters().keys()]).toEqual([
      "a.weight",
      "a.bias",
      "b.weight",
      "b.bias",
      "list.0.weight",
      "list.0.bias",
    ])
  })

  it("namedModules lists every nested module by dotted path", () => {
    class Sub extends Module {}
    class Net extends Module {
      sub = new Sub()
      list = [new Sub()]
    }
    const net = new Net()
    expect([...net.namedModules().keys()].sort()).toEqual(["list.0", "sub"])
  })

  it("apply() visits this and every nested module", () => {
    class Sub extends Module {}
    class Net extends Module {
      sub = new Sub()
    }
    const net = new Net()
    const seen: string[] = []
    net.apply((_m, path) => seen.push(path))
    expect(seen).toEqual(["", "sub"])
  })

  it("train()/eval() propagate to nested modules", () => {
    class Sub extends Module {}
    class Net extends Module {
      sub = new Sub()
    }
    const net = new Net()
    expect(net.training).toBe(true)
    expect(net.sub.training).toBe(true)
    net.eval()
    expect(net.training).toBe(false)
    expect(net.sub.training).toBe(false)
    net.train()
    expect(net.training).toBe(true)
    expect(net.sub.training).toBe(true)
  })

  it("zeroGrad() clears every collected parameter's gradient", () => {
    class Net extends Module {
      w = parameter(Tensor.zeros([2]))
    }
    const net = new Net()
    net.w.grad = Tensor.ones([2])
    net.zeroGrad()
    expect(net.w.grad).toBeNull()
  })
})

describe("Module.parameterEpoch", () => {
  it("is stable across reads until the parameter set changes", () => {
    class Net extends Module {
      a = parameter(Tensor.zeros([1]))
      b?: Parameter<[1]>
    }
    const net = new Net()
    const e0 = net.parameterEpoch
    expect(net.parameterEpoch).toBe(e0)
    net.b = parameter(Tensor.zeros([1]))
    expect(net.parameterEpoch).not.toBe(e0)
  })
})
