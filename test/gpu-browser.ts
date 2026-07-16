import { initTypeGPU } from "../src/typegpu.ts"
import { Tensor, tensor } from "../src/tensor.ts"
import { Linear, crossEntropy } from "../src/nn.ts"
import { Adam, SGD } from "../src/optim.ts"

const result = document.querySelector("#result")!

function close(
  actual: ArrayLike<number>,
  expected: readonly number[]
): void {
  if (actual.length !== expected.length)
    throw new Error(
      `length ${actual.length} !== ${expected.length}`
    )
  expected.forEach((value, i) => {
    if (Math.abs(actual[i]! - value) > 1e-4)
      throw new Error(
        `value ${actual[i]} !== ${value} at ${i}`
      )
  })
}

async function parity(
  cpu: Tensor<any, any>,
  gpu: Tensor<any, any>
): Promise<void> {
  close(await gpu.read(), Array.from(await cpu.read()))
}

async function main() {
  if (!navigator.gpu) {
    result.textContent = "SKIP: WebGPU unavailable"
    return
  }
  const root = await initTypeGPU()
  let stage = "upload"
  try {
    const a = tensor([
      [1, 2, 3],
      [4, 5, 6]
    ]).gpu()
    const b = tensor([10, 20, 30]).gpu()
    stage = "binary"
    close(await a.add(b).read(), [11, 22, 33, 14, 25, 36])
    stage = "unary"
    close(await a.relu().exp().read(), [
      Math.E,
      Math.exp(2),
      Math.exp(3),
      Math.exp(4),
      Math.exp(5),
      Math.exp(6)
    ])
    const unaryInput = tensor([-2, -0.5, 0.5, 2])
    await parity(unaryInput.neg(), unaryInput.gpu().neg())
    await parity(unaryInput.abs(), unaryInput.gpu().abs())
    await parity(unaryInput.relu(), unaryInput.gpu().relu())
    await parity(
      unaryInput.leakyRelu(0.2),
      unaryInput.gpu().leakyRelu(0.2)
    )
    await parity(
      unaryInput.sigmoid(),
      unaryInput.gpu().sigmoid()
    )
    await parity(unaryInput.tanh(), unaryInput.gpu().tanh())
    const positive = tensor([0.25, 1, 4])
    await parity(positive.pow(1.5), positive.gpu().pow(1.5))
    await parity(positive.log(), positive.gpu().log())
    await parity(positive.sqrt(), positive.gpu().sqrt())
    stage = "binary-broadcast"
    const column = tensor([[1], [2]])
    const row = tensor([2, 4, 8])
    await parity(
      column.add(row as any),
      column.gpu().add(row.gpu() as any)
    )
    await parity(
      column.sub(row as any),
      column.gpu().sub(row.gpu() as any)
    )
    await parity(
      column.mul(row as any),
      column.gpu().mul(row.gpu() as any)
    )
    await parity(
      column.div(row as any),
      column.gpu().div(row.gpu() as any)
    )
    stage = "reduce"
    close(await a.sum(1).read(), [6, 15])
    await parity(
      tensor([
        [1, 2, 3],
        [4, 5, 6]
      ]).sum(),
      a.sum()
    )
    await parity(
      tensor([
        [1, 2, 3],
        [4, 5, 6]
      ]).max(0),
      a.max(0)
    )
    stage = "permute"
    close(await a.T.read(), [1, 4, 2, 5, 3, 6])
    stage = "matmul"
    close(
      await a
        .matmul(
          tensor([
            [1, 2],
            [3, 4],
            [5, 6]
          ]).gpu()
        )
        .read(),
      [22, 28, 49, 64]
    )
    const vector = tensor([1, 2, 3])
    await parity(
      vector.dot(vector),
      vector.gpu().dot(vector.gpu())
    )
    const batched = Tensor.ones([2, 3, 4])
    const weight = Tensor.ones([4, 5])
    await parity(
      batched.matmul(weight),
      batched.gpu().matmul(weight.gpu())
    )
    stage = "autograd"
    const x = tensor([1, 2, 3]).gpu().requires_grad()
    x.pow(2).sum().backward()
    close(await x.grad!.read(), [2, 4, 6])
    stage = "cat-argmax"
    close(
      await Tensor.cat(
        tensor([[1], [2]]).gpu(),
        tensor([[3], [4]]).gpu(),
        1
      ).read(),
      [1, 3, 2, 4]
    )
    close(
      await tensor([
        [1, 9, 2],
        [8, 3, 4]
      ])
        .gpu()
        .argmax(1)
        .read(),
      [1, 0]
    )
    await parity(
      Tensor.stack([tensor([1, 2]), tensor([3, 4])], 1),
      Tensor.stack(
        [tensor([1, 2]).gpu(), tensor([3, 4]).gpu()],
        1
      )
    )
    await parity(
      tensor([
        [1, 2, 3],
        [4, 5, 6]
      ]).view([3, 2]),
      a.view([3, 2])
    )
    stage = "linear"
    const linear = new Linear(2, 1).gpu()
    linear.weight.write([1, 2])
    linear.bias!.write([0])
    close(
      await linear.forward(tensor([[3, 4]]).gpu()).read(),
      [11]
    )
    stage = "crossEntropy"
    const logits = tensor([
      [2, 1, 0],
      [0, 1, 2]
    ]).gpu()
    const gpuLoss = crossEntropy(
      logits,
      tensor([0, 2]).gpu()
    )
    const cpuLoss = crossEntropy(
      tensor([
        [2, 1, 0],
        [0, 1, 2]
      ]),
      [0, 2]
    ).item()
    close(await gpuLoss.read(), [cpuLoss])
    stage = "sgd"
    const sgdParam = tensor([1, 2]).gpu().requires_grad()
    sgdParam.pow(2).sum().backward()
    new SGD([sgdParam], { lr: 0.1, momentum: 0.9 }).step()
    close(await sgdParam.read(), [0.8, 1.6])
    stage = "adam"
    const adamParam = tensor([1, 2]).gpu().requires_grad()
    adamParam.pow(2).sum().backward()
    new Adam([adamParam], { lr: 0.1 }).step()
    close(await adamParam.read(), [0.9, 1.9])
    stage = "errors"
    let mixed = false
    try {
      tensor([1])
        .gpu()
        .add(tensor([1]) as any)
    } catch {
      mixed = true
    }
    if (!mixed)
      throw new Error(
        "mixed-device operation did not throw"
      )
    let float64 = false
    try {
      tensor([1]).to("float64").gpu()
    } catch {
      float64 = true
    }
    if (!float64)
      throw new Error("float64 upload did not throw")
    const disposed = tensor([1]).gpu()
    disposed.dispose()
    let useAfterDispose = false
    try {
      await disposed.read()
    } catch {
      useAfterDispose = true
    }
    if (!useAfterDispose)
      throw new Error("disposed tensor read did not throw")
    result.textContent = "PASS"
  } catch (error) {
    throw new Error(
      `${stage}: ${error instanceof Error ? error.stack : error}`
    )
  } finally {
    root.destroy()
  }
}

main().catch(error => {
  result.textContent = `FAIL: ${error instanceof Error ? error.stack : error}`
})
