# Lazy evaluation + native backend — design & roadmap

Working branch: `lazy-native`. Reference implementation cloned at
`/tmp/effect-torch-study` (github.com/mikearnaldi/effect-torch) — re-clone if gone.

## The idea in one paragraph

typenet's type-level shapes (`Tensor<[2, 3]>`, `Broadcast`, `MatMulCheck`, tsover
operator overloads) are erased at compile time — they cost nothing at runtime and
do not care how or when numbers are computed. So we split each op in two:

1. **Type level (unchanged):** methods in `src/tensor.ts` keep their exact
   signatures; output shapes are still computed by the compiler.
2. **Runtime (changed):** instead of computing immediately, ops append a node to
   a lazy graph. Each node records its output _shape_ (pure shape math, no data
   touched), so type-level shape and runtime shape stay in sync by construction.

Execution happens only at explicit **forcing points**: `.read()`, `.item()`,
`.toCPU()`, `.backward()`, `optimizer.step()`. This is the same design as
effect-torch (`packages/native/src/lib.rs` — `LazyNode` enum ~line 335,
`eval_lazy` ~line 1127).

Effect is **not** used. Errors are thrown `TensorError`s; no DI layers; device
selection stays the existing `configureTypeGPU`-style init.

## Vocabulary

- **Lazy graph / scheduler** = the "recipe" layer. Decides _what_ to compute and
  defers _when_. Not a backend.
- **Executor / backend** = the "stove" that crunches numbers: CPU typed-array
  kernels, TypeGPU/WebGPU kernels, or (phase 2) Rust/candle.
- The lazy graph is the **boundary** that makes backends swappable. Phase 1
  builds the boundary; phase 2 plugs candle into it.

## Phase 1 — lazy graph on existing backends (this branch)

Status: implemented (CPU + TypeGPU paths; tests green).

What landed:

- `configure({ lazy: true })` / `isLazy()` in `src/tensor.ts` (re-exported
  from `index.ts`). Eager remains the default; lazy is a global opt-in flag.
- `LazyNode` union + `LazyStorage` (`kind: "lazy"`) in `src/tensor.ts`.
  Variants: `binary`, `unary`, `matmul`, `reduce` (dim), `reduceAll`,
  `broadcastTo`, `permute`, `view` (reshape/squeeze/unsqueeze), `narrow`,
  `cat`, `oneHot`. There is no explicit `leaf` variant — eager tensors
  (incl. scalar coercions) serve directly as graph leaves. Each node
  carries `shape`, `dtype`, `device`, computed by the same pure shape
  helpers (`broadcastShapes`, stride math) as eager mode — no data is
  touched at build time.
- The `raw*` functions (`rawBinary`, `rawUnary`, `rawReduce`,
  `rawReduceAll`, `rawBroadcastTo`, `rawPermute`, `rawMatmul`, `rawNarrow`,
  `rawOneHot`, `rawCat`, `reshapeRaw`) check the flag first and emit a node
  instead of computing. Full-tensor `sum()`/`max()`, `oneHot()` and
  `Tensor.cat` were extracted into `rawReduceAll` / `rawOneHot` / `rawCat`
  so both modes share one code path.
- Interpreter: `evalNode` walks the graph recursively and calls the
  existing `raw*` kernels with the flag temporarily off (`eagerly()`).
  Memoization: each `LazyStorage` caches its evaluated tensor; `force(t)`
  swaps the tensor's storage in place, so shared subgraphs evaluate once
  and every alias of a tensor sees the materialized result.
- Forcing points: `data`, `read()`, `item()`/`get()`/`toArray()` (via
  `data`), `toCPU()`, `write()`, `clone()`, `backward()`.
  `backward()` on a lazy loss builds the backward pass as lazy
  expressions and forces the forward topo plus all parameter grads in
  one `forceMany` (see phase 2 — lazy symbolic backward); the eager
  path is unchanged.
- Disposal: lazy nodes hold no buffers (CPU or GPU) until evaluated; after
  materialization the usual `.dispose()` rules apply.
- Tests: `test/lazy.test.ts` compares lazy vs eager numerically for
  broadcast binary ops, matmul, reduces, a unary chain, view/permute, cat,
  oneHot, mixed lazy/eager operands, and a full XOR forward + backward +
  SGD step.

Known limitations:

- GPU lazy graphs are wired (nodes carry `device: "gpu"` and dispatch to
  the TypeGPU kernels) but only exercised on CPU in tests; browser/dawn
  parity not yet verified.
- Validation that needs data (e.g. `oneHot` target range on CPU) fires at
  forcing time, not at op call time, in lazy mode.
- `pnpm typecheck` currently fails only on `examples/basic.ts:22`, which
  contains a deliberate matmul shape error (pre-existing on HEAD, demo
  scratch file) — library and tests typecheck clean.

Nice-to-haves later (not phase 1): op fusion in the interpreter, buffer reuse,
batched WGSL dispatches.

## Phase 2 — Rust/candle native backend

Status: implemented (v2 — multi-root eval + lazy symbolic backward).
Lazy CPU graphs are evaluated by a napi-rs/candle addon in one FFI
hop per forcing point; `backward()` on a lazy loss builds the entire
backward pass as lazy expressions and forces the forward topo plus
all parameter grads in a single multi-root hop. Tests green on Metal.

What landed:

- `native/` package (`@typenet/native`, private workspace package):
  napi-rs crate with `candle-core 0.9` (`metal` + `accelerate` on
  macOS), modeled on `/tmp/effect-torch-study/packages/native/src/lib.rs`.
  Two exported functions: `evalGraph(graphJson, leaves)` and
  `deviceName()`. Zero-copy readback via
  `napi_create_external_arraybuffer` (the f32 result Vec is handed to
  JS as an external ArrayBuffer, freed by the finalizer).
  Release profile: `lto = true, codegen-units = 1`.
- Bridge shape: `serializeLazyGraph` in `src/tensor.ts` walks the
  `LazyNode` tree once (memoized on tensor identity, so shared
  subexpressions serialize as one node), emitting a topological JSON
  node list with leaves as `{ op: "leaf", leaf, offset, shape }`
  placeholders plus one concatenated `Float32Array`. Rust rebuilds it
  with candle ops on the best device (Metal if `Device::new_metal(0)`
  succeeds, else CPU) and returns f32 bytes. `force()` uses it when
  lazy mode is on, the graph device is `"cpu"`, and native is enabled
  — otherwise it falls back to the interpreter (GPU leaves and
  float64 graphs also fall back).
- Multi-root eval: the graph JSON carries a `roots` index list
  (default: last node, so single-root callers are unchanged). Rust
  evaluates the node list once — nodes shared across roots compute
  once — and returns all roots as one concatenated f32 buffer, which
  the TS side slices per root shape (zero-copy subarray views into
  the external ArrayBuffer). `forceMany(tensors)` in `src/tensor.ts`
  is the multi-root forcing point: it serializes every pending root
  into one graph and makes one FFI hop; without native it simply
  forces each tensor through the interpreter, whose per-`LazyStorage`
  cache still evaluates shared subgraphs once.
- Lazy symbolic backward: `backward()` on a lazy loss no longer
  materializes the forward and re-runs the GradNode engine eagerly.
  The seed stays a plain ones leaf and the existing reverse-topo
  GradNode walk runs with lazy mode still on, so the backward rules
  (which compose public ops — mul/matmul/sumTo/broadcastTo/etc.)
  build lazy grad expressions whose closures capture the lazy forward
  tensors. Forward values the grad rules need (e.g. the tanh output)
  are therefore shared subgraphs, not recomputed. Grad accumulation
  across steps becomes a lazy `add` node against the previous
  (materialized) `.grad` leaf; nothing is mutated per-force. Finally
  `forceMany` materializes the whole forward topo plus every
  parameter grad in one hop. The forward topo is forced too (not
  just the grads) so values read after an in-place optimizer step —
  which mutates leaf parameter data — still see pre-step values,
  matching eager and phase-1 semantics. Eager mode, and calling
  `.backward()` on an eager loss while lazy mode is on, take the
  original code path unchanged.
- Activation: `useNative()` / `disableNative()` / `isNativeAvailable()`
  / `nativeDevice()` in `src/backends/native.ts`, re-exported from
  `index.ts`. The addon is loaded lazily via `createRequire`, so the
  library works fine when the `.node` binary is missing; `useNative()`
  throws a clear "run pnpm build:native" error in that case. Eager mode
  is completely unaffected.
- Op coverage (f32 only): all `LazyNode` variants — binary (add, sub,
  mul, div with broadcasting, plus the composite grad ops negDiv,
  halfDiv, mulSign, reluGrad, leakyReluGrad, sigmoidGrad, tanhGrad),
  unary (pow, neg, exp, log, sqrt, abs, relu, leakyRelu, sigmoid, tanh,
  scalePowGrad), matmul (with batch-dim broadcasting, which candle does
  not do natively), reduce/reduceAll (sum, max, argmax — argmax returns
  the first index on ties, matching the CPU kernel), broadcastTo,
  permute, view, narrow, cat, oneHot. Errors come back as JS errors
  prefixed `native backend:`.
- Tests: `test/native.test.ts` — parity vs CPU eager within 1e-4 for
  the same op set as `test/lazy.test.ts` plus batched matmul, argmax,
  an error-path test, a full XOR forward (native) + backward +
  SGD step, and gradient parity: lazy-vs-eager and native-vs-eager
  parameter grads for a matmul/broadcast/reduce/unary chain, the
  crossEntropy path (tensor and array targets), an XOR training step,
  grad accumulation across repeated `backward()` calls, and a
  shared-subexpression multi-root case (`z = x*x` used by two
  parents). Skips gracefully via
  `describe.skipIf(!isNativeAvailable())` when the binary isn't built.

Build:

```sh
pnpm install        # links native/ into the workspace
pnpm build:native   # napi build --release → native/typenet-native.node
```

Prerequisites: a Rust toolchain (`rustup`, 1.75+). First candle build
takes ~1–2 min on an M-series Mac; incremental rebuilds ~15–20 s.
`native/target/` and `native/*.node` are gitignored.

Deviations from the original plan:

- **Synchronous FFI, not tokio async.** The forcing points `data`,
  `item()`, `backward()` are synchronous public methods whose
  signatures may not change, so `evalGraph` blocks (one hop) instead of
  returning a Promise. Graphs are test-scale; revisit async if large
  graphs make the blocking hop visible.
- Errors are plain `Error`s (there is no `TensorError` class in the
  codebase — the interpreter throws plain errors too).

Known limitations:

- f32 only; float64 and GPU-device lazy graphs silently fall back to
  the CPU/TypeGPU interpreter.
- The whole graph is re-serialized at every forcing point (no caching
  of the JSON or of candle tensors across forces). For training loops
  this means one serialization + FFI hop per step, ~40–50 µs for a
  tiny graph — fine, but still ~4x slower than pure eager CPU on
  graph sizes where the compute itself is microseconds (see the
  benchmark table). Closing that gap needs structural graph caching
  across steps or a binary wire format, not more tuning of this path.
- Small-graph device hint: graphs touching ≤ 65536 elements
  (`CPU_HINT_MAX_WORK` in `src/tensor.ts`) carry `device: "cpu"` in
  the graph JSON and evaluate on candle's CPU device even when Metal
  is available. Metal charges per-kernel dispatch plus a sync per
  readback (~3 ms for a 32-node training-step graph) while candle
  CPU evaluates the same graph in ~25 µs; Metal only pays off on
  large compute-dense graphs.
- No op fusion; composite grad ops are lowered to several candle
  elementwise kernels.
- Metal device selection is "first Metal device or CPU"; no way to
  force CPU short of the small-graph hint or not enabling native.

Data-dependent reads in backward: none. The backward rules only
compose shape-driven ops. `softmax`/`logSoftmax` detach the max (a
symbolic op, no data read); `crossEntropy` builds its one-hot mask
from a plain JS array or via the lazy `oneHot` node; `max`/`argmax`
do not propagate gradients at all (no GradNode), so no max-mask is
needed; optimizers read `.data` only after `backward()` has forced
the grads.

### Benchmarks

Run with `pnpm vite-node examples/bench.ts` (2026-07-27, Apple M5,
`nativeDevice()`: metal; tiny graphs pinned to candle CPU via the
device hint). Wall time per iteration, fresh leaf tensors each
iteration so lazy graphs are rebuilt and re-serialized; the timer
covers graph build + forcing for the lazy modes. Averages over 3–10
iterations after 1 warmup.

| Workload                                    | eager CPU | lazy (interpreter) | lazy + native |
| ------------------------------------------- | --------- | ------------------ | ------------- |
| matmul chain: 10 × ([256,256] @ [256,256])  | 145.4 ms  | 145.9 ms           | 13.3 ms       |
| matmul chain: 10 × ([512,512] @ [512,512])  | 1082.8 ms | 1193.9 ms          | 49.6 ms       |
| elementwise chain: 20 ops on [1024,1024]    | 294.7 ms  | 314.0 ms           | 42.0 ms       |
| XOR MLP train, 200 steps (fwd+bwd+SGD)      | 2.6 ms    | 4.3 ms             | 11.3 ms       |
| XOR MLP train, 200 steps, compiled (task 4) | —         | 2.1 ms             | 5.0 ms        |

XOR numbers updated 2026-07-27 (Apple M5) after Phase B task 4: the
lazy-mode `step()` is now itself a graph eval (one extra forcing point
per step in the un-compiled modes — hence the regression there), and
the compiled rows replay the whole training step as one graph. The
previous run (same day, before task 4): 153.1 / 1296.5 / 323.9 /
3.5 ms eager and 2.8 / 6.2 ms for the lazy XOR modes.

Previous run (2026-07-27, Apple M5, single-root eval + eager
backward): 147.5 / 1213.9 / 286.0 / 2.8 ms eager, and **740.5 ms**
for native XOR — the problem this phase fixed.

Takeaways:

- Native (Metal) wins big on large compute-dense graphs: ~11x on the
  256² matmul chain, ~23x on 512², ~7x on the elementwise chain.
  One serialization + FFI hop is easily amortized there.
- The lazy interpreter alone neither helps nor hurts — it ends up
  running the same CPU typed-array kernels.
- XOR training went from 740.5 ms to 6.2 ms per 200 steps (~120x):
  `backward()` is now one multi-root hop per step (measured: 1 FFI
  call per step, ~32 graph nodes) instead of one per topo tensor,
  and the tiny graph runs on candle CPU instead of paying Metal
  dispatch + readback overhead (~3 ms/hop).
- Native is still ~2x slower than eager CPU on the un-compiled XOR
  loop, and task 4 widened that gap (6.2 → 11.3 ms): the in-graph
  `step()` adds a second serialization + FFI hop per step. The answer
  for tiny training loops is `compile()` — one cached graph, no
  re-serialization (see the task 4 section). On micro-graphs eager
  wins because it has zero per-op overhead; native pays off for
  few-force, big-graph workloads.

## Phase B task 4 — optimizer updates in the graph

Status: implemented (tests green, interpreter + native paths).

What landed:

- In lazy mode, `SGD.step()` / `Adam.step()` build their parameter
  updates as lazy graph expressions (`src/optim.ts`) instead of
  looping over `.data` on CPU: weight decay folds into the grad,
  momentum velocities / Adam moments live in ordinary CPU leaf
  tensors owned by the optimizer (the typonet model — state as graph
  inputs carried between steps), and the new values
  (`p - lr·v`, `v'`, …) are forced in one multi-root hop and written
  back into the same leaf buffers in place. Public signatures are
  unchanged (`step()` stays sync, no arguments), and the eager and
  GPU paths are byte-identical — the graph path only triggers for
  CPU float32 params while lazy mode (or a compile trace) is active.
- `compile()` composes with this: a compiled function may contain
  the full training step — `opt.zeroGrad(); loss.backward();
opt.step(); return loss`. During tracing an update collector
  (`_activeUpdateTrace` in `src/tensor.ts`) is installed: backward's
  lazy forcing point is deferred (grads stay graph expressions) and
  `step()` registers `(target, expr)` update pairs instead of
  forcing. The update expressions become extra roots of the cached
  graph, and every replay — native or interpreter — evaluates
  forward + backward + updates in one pass, writes the update roots
  back into the parameter/state leaf buffers, and materializes the
  grad tensors so `.grad` reads after a compiled step see this
  step's values. Grads never touch a JS-side element loop; the
  returned loss is the pre-step value (single eval, updates are
  written back afterwards), matching eager.
- Post-step read semantics: the compiled step evaluates everything
  from pre-step leaf values and only then mutates the leaves, which
  is exactly the invariant the un-compiled lazy path enforces by
  forcing the forward topo in `backward()` — so post-step reads see
  pre-step values in both.
- Tests: `test/optim-graph.test.ts` — lazy SGD (momentum + weight
  decay) and lazy Adam tracking eager step-by-step over 30 steps
  (params, grads, and losses within f32 tolerance), an eager-mode
  step-numerics guard, a compiled 50-step training run matching the
  eager loss trajectory step-by-step plus final params and grads, a
  compiled 400-step XOR loop that learns (final loss < 0.05), a
  clear error for Adam inside `compile()`, and a native-path
  compiled-step parity test via the `describe.skipIf` pattern.
- `examples/bench.ts` gained a compiled-XOR section
  (`xorTrainCompiled` traces the whole step once, then replays).

Benchmark (XOR MLP, 200 steps, 2026-07-27, Apple M5 — see the phase 2
table): eager CPU 2.6 ms; **compiled (interpreter) 2.1 ms**;
compiled + native 5.0 ms. The gate (compiled ≤ eager ~3.5 ms) passes
on the interpreter path; the native path is over it but improved vs
its 6.2 ms pre-task-4 number. Why native is still slower on this
micro-graph: the replayed step is ~100 tiny ops and candle charges
per-op dispatch (~25–50 µs/step total) while eager JS just runs the
same arithmetic inline; the interpreter replay avoids both the FFI
hop and serialization, which is why it beats eager. Closing the
native gap needs graph-level fusion / fewer kernels (Phase D), not
more plumbing here.

Known limitations:

- Adam inside `compile()` throws: bias correction depends on the
  host-side step count, which a build-once graph cannot carry (the
  constants would freeze at trace-time `t`). Un-compiled lazy Adam
  is unaffected — its graph is rebuilt per step with fresh
  constants. SGD (incl. momentum, weight decay) works in compiled
  steps since all its scalars are constant.
- `zeroGrad()` inside a compiled function runs only at trace time;
  replay always produces fresh grads (no cross-call accumulation),
  matching the standard per-step `zeroGrad` idiom.
- The un-compiled lazy modes pay one extra forcing point per step
  for the update graph (XOR: interpreter 2.8 → 4.3 ms, native
  6.2 → 11.3 ms) — the compiled step is the fast path for tiny
  training loops.
- Graph-path optimizer state is f32 (eager keeps f64
  velocities/moments); covered by the step-by-step parity tests
  within f32 tolerance.

## Phase B task 3 — `compile()`: build once, replay many

Status: implemented (tests green, interpreter + native paths).

What landed:

- `compile(fn)` in `src/tensor.ts`, re-exported from `index.ts`. The
  first call traces `fn` once under lazy semantics (regardless of the
  global `configure({ lazy })` flag, which is restored afterwards)
  with placeholder leaf tensors, serializes the graph once via
  `serializeLazyGraph`, and caches it. Generic over `fn`'s parameter
  and return types, so input/output shapes are preserved at the type
  level; existing signatures untouched.
- Replay: each call copies the caller's input data into the
  placeholder leaf buffers and evaluates the whole graph once. With
  native enabled that's one multi-root FFI hop — the cached JSON is
  reused and only the concatenated leaf buffer is rebuilt per call
  from the live leaf tensors (`serializeLazyGraph` now also returns
  the leaf tensor list, offsets, and byte count). Without native, the
  traced lazy tensors' caches are reset and `forceMany` re-evaluates
  through the interpreter.
- Closure-captured tensors (module parameters) are ordinary graph
  leaves read live on every call, so in-place optimizer `step()`
  updates are visible to the compiled function. Grads are not
  exposed — backward/optimizer-in-graph is task 4.
- Inputs: the trace (first) call must pass CPU float32 tensors so
  shapes are pinned; later calls may pass same-shape tensors or flat
  `ArrayLike<number>` buffers. `fn` may return a single tensor or a
  tuple.
- Cache invalidation honesty: wrong argument count, shape, or dtype
  on any call throws a clear error ("compiled graphs are
  shape-stable, recompile for a new shape"). Results are fresh
  tensors per call — earlier results are never clobbered.
- Tests: `test/compile.test.ts` — eager parity for a
  matmul/unary/reduce graph, replay with swapped tensor and flat
  buffer data, multiple outputs, in-place parameter updates, the
  three error paths, tracing under a flipped global lazy flag, and a
  compiled XOR-style forward step (interpreter, plus native parity
  via the `describe.skipIf` pattern).

Known limitations (task 3, as landed — see task 4 for the update):

- Forward graphs only at this point: forcing points (`.data`,
  `.item()`, `.backward()`) inside `fn` are unsupported — they would
  bake in trace-time values. Task 4 (above) puts backward +
  optimizer in the graph.
- CPU float32 only (same coverage as the native bridge); f64/GPU
  graphs and GPU-captured leaves throw or fall back as before.
- Captured leaves must stay on the CPU tensors they were traced with
  — moving a parameter to GPU after compiling throws on the native
  path.
- The interpreter fallback still pays per-op dispatch (no structural
  win there); the native path amortizes to one serialization-free
  FFI hop per call.

## Phase C task 5 — `.named()` + `printGraph()`

Debug ergonomics ported from typonet (its `Graph.names` + `Graph.print()`),
minus the `yield*` naming mechanism — names attach via a plain method:

```ts
const h = x.matmul(w).named("h")
printGraph(loss) // or printGraph([a, b]) for several roots
```

Output is an SSA-ish listing in topological order, one line per node,
labels padded for alignment, roots marked:

```
x    = leaf [2, 3] float32
w    = leaf [3, 4] float32
h    = matmul(x, w) [2, 4] float32
%3   = relu(h) [2, 4] float32
loss = reduceAll.sum(%3) [] float32 ; root
```

Design decisions:

- **Names are metadata only.** They live in a module-level
  `WeakMap<Tensor, string>` in `src/tensor.ts`, never consulted by
  compute, autograd, or graph serialization. `named(name)` sets the
  label and returns `this`, so it chains and preserves the tensor's
  type parameters.
- **Name survival.** Forcing keeps a tensor's identity (its storage is
  swapped in place), so a name survives materialization. Names do NOT
  cross operations that create fresh tensor objects — `detach()`,
  `clone()`, and `compile()` placeholders start unnamed.
- **Auto labels.** Unnamed nodes print as `%0`, `%1`, ... in traversal
  order (post-order DFS from the roots), stable for a given graph.
- **Eager tensors** have no graph and print as a single `leaf` line —
  honest, no crash. Since lazy `backward()` forces grads in one
  multi-root hop, a `.grad` tensor also prints as a leaf; print the
  loss _before_ calling `backward()` to see the forward graph.
- **Shared subgraphs** print once; multi-root calls mark each root
  with `; root`.

## How to resume

1. `git checkout lazy-native`
2. Read this file.
3. Phase 1 state: check `git log` and which `raw*` call sites in
   `src/tensor.ts` emit lazy nodes vs still compute eagerly.
4. Verify: `pnpm test && pnpm typecheck`.
5. Reference: `/tmp/effect-torch-study` (re-clone if missing:
   `git clone --depth 1 https://github.com/mikearnaldi/effect-torch /tmp/effect-torch-study`).
