import {
  compile,
  configure,
  disableNative,
  isNativeAvailable,
  Linear,
  Module,
  nativeDevice,
  randn,
  SGD,
  type Tensor,
  tensor,
  type TensorParams,
  useNative,
} from "../index.ts"

// Benchmark: eager CPU vs lazy interpreter vs lazy + native (candle).
// Each workload is rebuilt from fresh leaves every iteration so lazy
// graphs are re-serialized and re-evaluated — the timer covers graph
// build + forcing for lazy modes and the kernels themselves for eager.

function timeIter(
  fn: () => void,
  {
    warmup = 2,
    iters = 5,
  }: { warmup?: number; iters?: number } = {},
): number {
  for (let i = 0; i < warmup; i++) fn()
  const t0 = performance.now()
  for (let i = 0; i < iters; i++) fn()
  return (performance.now() - t0) / iters
}

// --- workload 1: matmul chain --------------------------------------------

function matmulChain(size: number, chain: number): void {
  let h = randn([size, size])
  const weights = Array.from({ length: chain }, () => randn([size, size]))
  for (const w of weights) h = h.matmul(w)
  void h.data // forcing point
}

// --- workload 2: elementwise chain ---------------------------------------

function elementwiseChain(size: number, ops: number): void {
  const other = randn([size, size])
  let t = randn([size, size])
  const unary = [
    () => t.tanh(),
    () => t.sigmoid(),
    () => t.exp(),
    () => t.sqrt().add(1),
    () => t.mul(0.5).sub(other.mul(0.01)),
  ]
  for (let i = 0; i < ops; i++) {
    t = unary[i % unary.length]()
  }
  void t.data // forcing point
}

// --- workload 3: XOR MLP training loop ------------------------------------

class XorNet extends Module {
  hidden = new Linear(2, 8)
  out = new Linear(8, 1)

  forward<B extends number, P extends TensorParams>(
    x: Tensor<[B, 2], P>,
  ): Tensor<[B, 1], P> {
    const h = this.hidden.forward(x).tanh()
    return this.out.forward(h).sigmoid()
  }
}

const XOR_X = tensor([
  [0, 0],
  [0, 1],
  [1, 0],
  [1, 1],
])
const XOR_Y = tensor([[0], [1], [1], [0]])

function xorTrain(steps: number): number {
  const net = new XorNet()
  const optim = new SGD(net.parameters(), {
    lr: 0.5,
    momentum: 0.9,
  })
  let loss = net.forward(XOR_X).sub(XOR_Y).pow(2).mean()
  for (let step = 0; step < steps; step++) {
    const pred = net.forward(XOR_X)
    loss = pred.sub(XOR_Y).pow(2).mean()
    optim.zeroGrad()
    loss.backward() // forcing point
    optim.step()
  }
  return loss.item()
}

// The whole training step (forward + backward + SGD update) traced
// once and replayed as one graph — the phase B task 4 path. Replay
// evaluates everything in a single native hop (or one interpreter
// pass) and writes updated params/velocities back into the leaves.
function xorTrainCompiled(steps: number): number {
  const net = new XorNet()
  const optim = new SGD(net.parameters(), {
    lr: 0.5,
    momentum: 0.9,
  })
  const step = compile(
    (x: Tensor<[4, 2]>, y: Tensor<[4, 1]>) => {
      const loss = net.forward(x).sub(y).pow(2).mean()
      optim.zeroGrad()
      loss.backward()
      optim.step()
      return loss
    },
  )
  let loss = step(XOR_X, XOR_Y)
  for (let i = 1; i < steps; i++) loss = step(XOR_X, XOR_Y)
  return loss.item()
}

// --- runner ----------------------------------------------------------------

type Mode = {
  name: string
  setup: () => void
}

const modes: Mode[] = [
  {
    name: "eager CPU",
    setup: () => {
      configure({ lazy: false })
      disableNative()
    },
  },
  {
    name: "lazy (interpreter)",
    setup: () => {
      configure({ lazy: true })
      disableNative()
    },
  },
]

if (isNativeAvailable()) {
  modes.push({
    name: "lazy + native",
    setup: () => {
      configure({ lazy: true })
      useNative()
    },
  })
}

const workloads = [
  {
    name: "matmul chain [256,256] x10",
    run: () => matmulChain(256, 10),
    iters: 10,
  },
  {
    name: "matmul chain [512,512] x10",
    run: () => matmulChain(512, 10),
    iters: 3,
  },
  {
    name: "elementwise chain [1024,1024] x20",
    run: () => elementwiseChain(1024, 20),
    iters: 5,
  },
  {
    name: "xor train 200 steps",
    run: () => {
      workloads[3].finalLoss = xorTrain(200)
    },
    iters: 3,
    finalLoss: undefined as number | undefined,
  },
]

console.log(`native available: ${isNativeAvailable()}`)
console.log(`native device:    ${nativeDevice()}\n`)

const header = ["workload", ...modes.map(m => m.name)]
console.log(header.map(c => c.padEnd(22)).join(""))
console.log("-".repeat(22 * header.length))

for (const w of workloads) {
  const cells = [w.name.padEnd(22)]
  for (const m of modes) {
    m.setup()
    const ms = timeIter(w.run, {
      warmup: 1,
      iters: w.iters,
    })
    cells.push(`${ms.toFixed(1)} ms/iter`.padEnd(22))
  }
  console.log(cells.join(""))
  if (w.finalLoss !== undefined) {
    console.log(
      `  xor final loss (last mode): ${w.finalLoss.toFixed(6)}`,
    )
  }
}

// Compiled training step (phase B task 4): the whole XOR step —
// forward, backward, and the SGD update — is one replayed graph.
for (
  const mode of [
    {
      name: "compiled (interpreter)",
      setup: () => {
        configure({ lazy: false })
        disableNative()
      },
    },
    ...(isNativeAvailable()
      ? [
        {
          name: "compiled + native",
          setup: () => {
            configure({ lazy: false })
            useNative()
          },
        },
      ]
      : []),
  ]
) {
  mode.setup()
  let finalLoss = 0
  const ms = timeIter(
    () => {
      finalLoss = xorTrainCompiled(200)
    },
    { warmup: 1, iters: 3 },
  )
  console.log(
    `${"xor train 200 steps".padEnd(22)}${`${ms.toFixed(1)} ms/iter`.padEnd(22)}  [${mode.name}]`,
  )
  console.log(`  xor final loss: ${finalLoss.toFixed(6)}`)
}

configure({ lazy: false })
disableNative()
