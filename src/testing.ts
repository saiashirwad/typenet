import type { GradNode } from "./autograd.ts"
import { _internal, type AnyTensor } from "./tensor.ts"

/** Test-only peepholes into Tensor's private fields. */
export const testing = {
  storageOf(
    t: AnyTensor,
  ): "cpu" | "lazy" | "materialized" {
    const source = _internal.sourceOf(t)
    if (source.kind === "cpu") return "cpu"
    return _internal.hasValue(t) ? "materialized" : "lazy"
  },
  gradNodeOf(t: AnyTensor): GradNode | null {
    return _internal.gradNodeOf(t)
  },
}
