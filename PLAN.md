# typenet roadmap — the plan to get everything good

Consolidated plan, 2026-07-27. Branch: `lazy-native`. History and phase 1/2
details live in `LAZY.md`; this file is the forward-looking plan. Update it as
phases land.

## Target state

typenet keeps its exact public API and type safety — eager by default, tsover
operator overloads, literal shape types (`Tensor<[B, 8]>`) everywhere.
Underneath, training loops run as **compiled graphs**: traced once, executed by
candle (Metal/CPU), one FFI hop per step, gradients verified by
finite-difference tests. typonet (`~/code/typonet`) retires into a sketchbook —
its good ideas graduate here.

## Design invariants (do not break)

1. **No public type signature changes.** All work is runtime plumbing below
   the type level. `test/types.test-d.ts` must keep passing untouched.
2. **Eager mode stays the default and byte-identical.** Lazy/compiled is
   opt-in (`configure({ lazy: true })`, `compile()`).
3. **Type/runtime shape sync.** Lazy nodes compute runtime shapes from the
   same pure shape math the types mirror — the two must never drift.
4. Every phase lands tested + benchmarked before the next starts. Update this
   file and LAZY.md per phase.

## Key decisions (settled, don't re-litigate)

- **No generator DSL.** Generators add nothing — not for laziness (chained
  method calls build the graph; typonet's `yield*` only attaches debug names)
  and not for types (shape types flow through ordinary method return types;
  `yield*` returns its input unchanged). Effect needs `yield*` to unwrap
  effect channels; tensor ops return plain typed values. Plain functions +
  tsover operators compose fine (tsover rewrites operators at compile time,
  orthogonal to runtime tracing).
- **Merge direction: typonet → typenet,** not the reverse. typonet has the
  better execution model (build-once/replay-many, symbolic autograd, optimizer
  in graph, gradcheck, hardened type algebra); typenet has the machinery
  (eager mode, real backends, tsover, dtypes, tests). Maintaining two Tensor
  APIs and two shape algebras is the losing option.
- **Known tracing limitation (accepted, JAX-style):** data-dependent JS
  control flow (`if` on tensor *values*) can't be captured in a compiled
  graph. Shape-dependent control flow is fine — shapes are known at trace time.

## Phase A — safety net

1. **Gradcheck.** Port typonet's finite-difference checker
   (`typonet/src/gradcheck.ts`): every backward rule in typenet verified
   against central differences, tolerance ~1e-4 rel for f32 (typonet used
   f64/1e-11; adapt). Run in eager AND lazy modes.
2. **Negative type tests.** Port typonet's `errors.test-d.ts` approach into
   `test/types.test-d.ts`: assert bad shapes fail via `@ts-expect-error`
   (broadcast mismatch, matmul inner-dim mismatch, bad permute, bad view,
   sequential chain mismatch). Proves checks fire, not just pass.

Gate: gradcheck green in both modes; negative tests green.

## Phase B — `compile()`: build once, replay many

3. **`compile(fn)` API.** Trace `fn` once in lazy mode with placeholder
   leaves; serialize once; cache the graph. Each call swaps leaf
   `Float32Array`s and does ONE multi-root native eval (interpreter fallback
   when native unavailable). No generators:

   ```ts
   const step = compile((x: Tensor<[4,2]>, t: Tensor<[4,1]>) =>
     ((net.forward(x) - t) ** 2).mean())
   step(inputData, targetData)  // reuse: swap buffers, one FFI hop
   ```

   API details to pin during impl: how params/grads are exposed (return
   grads? auto-step?), cache invalidation honesty, error on shape change
   between calls.
4. **Optimizer updates in the graph.** In lazy/compiled mode,
   `optimizer.step()` appends update nodes (typonet model: momentum/Adam
   state as graph inputs carried between steps) instead of CPU-side `.data`
   loops. Full training step = one graph; grads never read back to JS.
   Depends on gradcheck (Phase A) landing first.

Gate: compiled XOR ≤ eager CPU (~3.5ms/200 steps); large workloads keep
their 6–23x; grad parity tests still green.

## Phase C — ergonomics + type hardening

5. **`.named("h")` + `printGraph(t)`** — readable SSA-ish IR dumps with shapes
   (typonet's one genuinely better DX feature, minus `yield*`).
6. **Type-algebra hardening** ported from typonet's `shape.ts` + README
   pitfalls section: fail-open checks on unresolved generics (no
   false-positive `ErrorMessage` in generic builders), `NoInfer` on repeated
   inference sites, any other tricks that apply to typenet's checks.

## Phase D — speed on big graphs (CONDITIONAL, only if benchmarks demand)

7. Graph-level optimization in Rust now that graphs are stable: elementwise
   fusion, dead-code elimination, buffer pooling. This is where "actually
   fast" lives for real workloads. No speculative work — benchmark first.

## Suggested commit cadence

One commit per numbered task, messages in the repo's lowercase style. Your
scratch files (`examples/basic.ts`, `examples/_gatdsl.ts`,
`examples/_yieldshape.ts`) stay uncommitted, they're yours.

## Current status

- [x] Phase 1: lazy graph (commit 304fe6c)
- [x] Phase 2: candle native backend (bb1ffa5)
- [x] Symbolic backward + multi-root eval (0e603d9)
- [x] Phase A — gradcheck + negative type tests
- [x] Phase B task 3 — compile(): build-once replay-many graphs
- [x] Phase B task 4 — optimizer in graph
- [ ] Phase C — naming/printing + type hardening
- [ ] Phase D — conditional graph optimization
