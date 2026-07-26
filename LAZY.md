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
   a lazy graph. Each node records its output *shape* (pure shape math, no data
   touched), so type-level shape and runtime shape stay in sync by construction.

Execution happens only at explicit **forcing points**: `.read()`, `.item()`,
`.toCPU()`, `.backward()`, `optimizer.step()`. This is the same design as
effect-torch (`packages/native/src/lib.rs` — `LazyNode` enum ~line 335,
`eval_lazy` ~line 1127).

Effect is **not** used. Errors are thrown `TensorError`s; no DI layers; device
selection stays the existing `configureTypeGPU`-style init.

## Vocabulary

- **Lazy graph / scheduler** = the "recipe" layer. Decides *what* to compute and
  defers *when*. Not a backend.
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
  `backward()` first forces every tensor in the GradNode topo order, then
  runs the existing autograd engine unchanged inside `eagerly()`.
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

Status: implemented (v1). Lazy CPU graphs are evaluated by a
napi-rs/candle addon in one FFI hop per forcing point; tests green on
Metal.

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
  `LazyNode` tree once, emitting a topological JSON node list with
  leaves as `{ op: "leaf", leaf, offset, shape }` placeholders plus one
  concatenated `Float32Array`. Rust rebuilds it with candle ops on the
  best device (Metal if `Device::new_metal(0)` succeeds, else CPU) and
  returns f32 bytes. `force()` uses it when lazy mode is on, the graph
  device is `"cpu"`, and native is enabled — otherwise it falls back to
  the interpreter (GPU leaves and float64 graphs also fall back).
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
  an error-path test, and a full XOR forward (native) + backward +
  SGD step. Skips gracefully via `describe.skipIf(!isNativeAvailable())`
  when the binary isn't built.

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
  of the JSON or of candle tensors across forces).
- No op fusion; composite grad ops are lowered to several candle
  elementwise kernels.
- Metal device selection is "first Metal device or CPU"; no way to
  force CPU short of not enabling native.

### Benchmarks

Run with `pnpm vite-node examples/bench.ts` (2026-07-27, Apple M5,
`nativeDevice()`: metal). Wall time per iteration, fresh leaf tensors
each iteration so lazy graphs are rebuilt and re-serialized; the timer
covers graph build + forcing for the lazy modes. Averages over 3–10
iterations after 1 warmup.

| Workload | eager CPU | lazy (interpreter) | lazy + native |
| --- | --- | --- | --- |
| matmul chain: 10 × ([256,256] @ [256,256]) | 147.5 ms | 146.8 ms | 13.2 ms |
| matmul chain: 10 × ([512,512] @ [512,512]) | 1213.9 ms | 1192.9 ms | 51.2 ms |
| elementwise chain: 20 ops on [1024,1024] | 286.0 ms | 291.5 ms | 43.8 ms |
| XOR MLP train, 200 steps (fwd+bwd+SGD) | 2.8 ms | 1.9 ms | 740.5 ms |

Takeaways:

- Native (Metal) wins big on large compute-dense graphs: ~11x on the
  256² matmul chain, ~23x on 512², ~6.5x on the elementwise chain.
  One serialization + FFI hop is easily amortized there.
- The lazy interpreter alone neither helps nor hurts — it ends up
  running the same CPU typed-array kernels.
- Native loses badly on the XOR training loop (~300x slower). The
  graphs are tiny (4×2 inputs), so compute is negligible and the cost
  is all overhead: `backward()` forces every tensor in the GradNode
  topo order, and each forcing point re-serializes and re-ships its
  subgraph across the FFI boundary. Hundreds of hops per step × 200
  steps dwarfs the ~2 ms of actual math. Expected — see "the whole
  graph is re-serialized at every forcing point" above. Native pays
  off for few-force, big-graph workloads, not chatty small-graph
  training loops.

## How to resume

1. `git checkout lazy-native`
2. Read this file.
3. Phase 1 state: check `git log` and which `raw*` call sites in
   `src/tensor.ts` emit lazy nodes vs still compute eagerly.
4. Verify: `pnpm test && pnpm typecheck`.
5. Reference: `/tmp/effect-torch-study` (re-clone if missing:
   `git clone --depth 1 https://github.com/mikearnaldi/effect-torch /tmp/effect-torch-study`).
