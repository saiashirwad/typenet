import type { GradNode } from "./autograd.ts"
import { _internal, type AnyTensor } from "./tensor.ts"

/**
 * Test-only peepholes into Tensor's private fields. `storageOf` reports
 * where a value stands: `"cpu"` for a plain CPU leaf, `"lazy"` for an
 * unevaluated graph node, `"materialized"` for a lazy tensor whose
 * value has been forced.
 */
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
