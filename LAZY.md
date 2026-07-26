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

## Phase 2 — Rust/candle native backend (not started)

Goal: a third executor behind the lazy-graph boundary, preferred in Node.
TypeGPU remains the browser executor; candle likely replaces it for Node.

Plan:

- New `native/` package: napi-rs crate, `candle-core` dep (metal + accelerate
  on macOS, cuda features later). Model on `/tmp/effect-torch-study/packages/native/src/lib.rs`:
  - tensors cross the bridge as opaque handles (Arc-backed candle tensors);
  - **zero-copy readback** via `napi_create_external_arraybuffer` (`lib.rs:36-95`);
  - eval on tokio blocking pool (`lib.rs:1127-1160`), release profile
    `lto = true, codegen-units = 1`.
- Bridge shape: TS serializes/walks the `LazyNode` tree once → Rust rebuilds it
  as candle ops → runs → returns buffer. One FFI hop per forcing point.
- No Effect: errors thrown as `TensorError`; skip cancellation initially
  (add `AbortSignal` later only if needed).
- Backend selection: `useNative()` init call alongside `configureTypeGPU`.
- Parity tests: same op-by-op comparison harness as `test/gpu-browser.ts`,
  candle vs CPU within 1e-4.

## How to resume

1. `git checkout lazy-native`
2. Read this file.
3. Phase 1 state: check `git log` and which `raw*` call sites in
   `src/tensor.ts` emit lazy nodes vs still compute eagerly.
4. Verify: `pnpm test && pnpm typecheck`.
5. Reference: `/tmp/effect-torch-study` (re-clone if missing:
   `git clone --depth 1 https://github.com/mikearnaldi/effect-torch /tmp/effect-torch-study`).
