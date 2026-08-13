import { Adam, clipGradNorm, compile } from "../../index.ts"
import type { AnyTensor } from "../../src/tensor.ts"
import { aliveMask, type GraphNCA, type GraphTensors } from "./model.ts"

/**
 * Roll a state forward `steps` times: one model update, then zero out
 * nodes the alive mask kills. Shared by the bench and train examples.
 */
export function rollout(
  model: GraphNCA<any, any>,
  x: AnyTensor,
  graph: GraphTensors<any>,
  steps: number,
  updateRate = 0.5,
): AnyTensor {
  for (let i = 0; i < steps; i++) {
    x = model.forward(x, graph, updateRate).mul(aliveMask(x, graph))
  }
  return x
}

/**
 * A whole training step traced once: rollout, visible-loss, backward,
 * grad clip, Adam. Shared by the two bench scripts.
 */
export function compiledTrainStep(
  model: GraphNCA<any, any>,
  optimizer: Adam,
  graph: GraphTensors<any>,
  target: AnyTensor,
  x0: AnyTensor,
  steps: number,
) {
  const params = model.parameters()
  return compile((input: AnyTensor) => {
    const loss = rollout(model, input, graph, steps)
      .narrow(1, 0, 4)
      .sub(target)
      .pow(2)
      .mean()
    optimizer.zeroGrad()
    loss.backward()
    clipGradNorm(params, 1)
    optimizer.step()
    return loss
  }, [x0])
}
