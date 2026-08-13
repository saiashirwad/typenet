# typenet roadmap — the plan to get everything good

Consolidated plan, 2026-07-27. Branch: `lazy-native`. This file is the
forward-looking plan and also holds the measured history that other files
cite (see "History: measured numbers" at the bottom). Update it as phases
land.

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
   file per phase.

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
  control flow (`if` on tensor _values_) can't be captured in a compiled
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
   const step = compile(
     (x: Tensor<[4, 2]>, t: Tensor<[4, 1]>) =>
       ((net.forward(x) - t) ** 2).mean(),
   )
   step(inputData, targetData) // reuse: swap buffers, one FFI hop
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

## Phase D — speed on big graphs (landed for tiny graphs)

7. Graph-level optimization in Rust now that graphs are stable. Benchmarks
   demanded the tiny-graph side, not the big-graph side (Metal already
   wins 6–23x there): landed the tiny-graph plan/exec evaluator with
   elementwise fusion, DCE, and a prepared-plan cache — see "History:
   measured numbers" below. Buffer pooling deliberately skipped (fusion
   made it moot). Remaining conditional work, only if future benchmarks
   demand: Metal-side kernel fusion for large graphs.

Gate: compiled ≤ eager ~3.5ms/200 steps — passed with margin: compiled

- native XOR is 1.6ms, the fastest mode overall (eager 2.1ms).

## Suggested commit cadence

One commit per numbered task, messages in the repo's lowercase style. Your
scratch files (`examples/basic.ts`, `examples/_gatdsl.ts`,
`examples/_yieldshape.ts`) stay uncommitted, they're yours.

## Phase E — graph neural nets, and training one at speed

Target: describe and train the graph cellular automaton in
`~/code/graph-cellular-automata`. Landed — the ops added were
gather/scatter, narrow, comparisons, clamp, in-graph randomness,
compilable Adam, and gradient clipping, plus a stack-safety rewrite of
the graph walkers, the device-default finding, and an optimization pass
that took a rolled-out training step from 141 ms to ~16 ms per time step
(see "History: measured numbers" below).

`examples/gnca` is the port; `test/gnca.test.ts` checks it against the
PyTorch original rather than trusting it.

Gate: forward and every gradient within 3e-5 relative of PyTorch, in all
four execution modes — passed. It trains.

Still open, and both are real projects rather than tasks:

8. **Parallel elementwise kernels.** candle's CPU elementwise ops are
   single-threaded — one core busy out of ten. That is a real fact, and
   the leading suspect for the remaining gap against PyTorch on MPS,
   though the split has not been measured per-op. Either give the fused loop
   evaluator strided views so it can win outright (it already has fusion,
   rayon and BLAS, and trails candle only on `permute`/`narrow`, which are
   free views there), or parallelize candle's kernels upstream.
9. **Gradient checkpointing.** Peak memory is every activation the
   backward pass needs — the same as PyTorch, ~3.4 GB at the reference
   settings, and linear in rollout length. Recomputing instead of storing
   would trade that for compute.

## Current status

- [x] Phase 1: lazy graph (commit 304fe6c)
- [x] Phase 2: candle native backend (bb1ffa5)
- [x] Symbolic backward + multi-root eval (0e603d9)
- [x] Phase A — gradcheck + negative type tests
- [x] Phase B task 3 — compile(): build-once replay-many graphs
- [x] Phase B task 4 — optimizer in graph
- [x] Phase C task 5 — .named() + printGraph() graph dumps
- [x] Phase C task 6 — type-algebra hardening
- [x] Phase D — graph optimization: tiny-graph evaluator + fusion + plan cache
- [x] Phase E — gather/scatter + the rest of the graph-CA op set, stack
      safety, CPU device default, 9x on a rolled-out training step
- [ ] Parallel elementwise kernels (the remaining ~3x vs PyTorch MPS)
- [ ] Gradient checkpointing (peak memory is linear in rollout length)

## History: measured numbers

The sections other files cite. All measured on this machine (Apple M5,
macOS aarch64) unless noted.

### Device defaults (Phase E finding)

candle's CPU device (Accelerate for matmul) is the native default, which
is not the obvious choice. Measured: CPU matches Metal on chained large
matmuls, loses to it by ~1.5x on purely elementwise graphs, and beats it
by ~7x on the gather/scatter graphs message passing produces — candle's
Metal `index_select`/`index_add` kernels are slow and such graphs are
made of many small dispatches. `useNative({ device: "gpu" })` opts back
in for elementwise-dominated workloads.

### Phase D (tiny graphs)

The tiny-graph plan/exec loop evaluator (elementwise fusion, DCE,
prepared-plan cache) took compiled native XOR to 1.6 ms per 200 steps,
the fastest mode overall (eager CPU: 2.1 ms). Graphs whose total element
count is at most 65536 (`LOOP_EVALUATOR_MAX_WORK` in `src/lazy.ts`) are
sent there with `device: "loops"`; everything larger goes to candle.

### Phase E (GNCA at speed)

An optimization pass over the compiled rolled-out training step
(multi-root eval, in-graph Adam, buffer release during eval on the
candle path) took it from 141 ms to ~16 ms per rolled-out time step at
the reference settings (8192 nodes, 76592 edges). Published race as of
2026-08: 0.7 training steps/s (typenet, candle CPU) vs 2.7 (PyTorch
MPS) on the README GNCA training recipe. Nothing in-repo times
`torch.mps` yet; treat those numbers as current until a checked bench
lands.

### Native dtype/device support (current behavior)

Native is f32-on-CPU-leaves only, and with native enabled that is now an
error: `serializeLazyGraph` throws on a float64 or non-CPU leaf instead
of silently handing the graph to the JS interpreter. Keep the graph in
float32 or call `disableNative()`. `TYPENET_CHECK_SHAPES=1` (always on
in debug builds) makes the Rust side recompute every node shape and
compare it against the JS-sent one.
