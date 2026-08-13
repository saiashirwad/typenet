# typenet — cleanup issues

A running to-do list of slop and cleanup opportunities found in the 2026-08
sweep. Grouped by severity; each item is a concrete, file-annotated task.
Check the box when the fix lands (one commit per item, lowercase messages).

## Tier 1 — latent bugs / correctness risks

- [x] **Optimizers are dtype-blind.** `src/optim.ts` — `useGraphStep()` gates
  the graph path on `p.dtype === "float32"`, but the eager (`nums`) path had
  no dtype check: an `int64` param throws `TypeError` (bigint · number) in
  `step()`, an `int32` param silently truncates, and `finishGraphUpdates`
  papers over it with `(u.target.data as Float32Array)`. Fixed by rejecting
  `int32`/`int64` params in the `Optimizer` constructor with a clear error.
- [x] **`leaf_offsets` swallows unknown leaf dtype.** `native/src/lib.rs:1213-1217`
  — `LeafTy::parse(...).unwrap_or(LeafTy::F32)` masks the "unsupported leaf
  dtype" error every other call site propagates with `?`. Fix: `LeafTy::parse(...)?`.
- [x] **`Target::parse` silently falls back to CPU.** `native/src/lib.rs:188`
  — `_ => Target::Cpu` turns a typo like `TYPENET_EVALUATOR=metal`/`=cuda` into
  a silent CPU run. Fix: error on unknown non-empty hints.
- [ ] **`Adam.dispose()` leaves the optimizer unusable.** `src/optim.ts:432-438`
  empties `m`/`v` to `[]` and nulls graph state, so a later `step()` indexes
  `this.m[pi]!` → `undefined` and throws; `SGD.dispose()` (`293-295`) nulls only
  `velocities` and re-initializes gracefully. Fix: uniform null-and-re-init
  semantics on both.

## Tier 2 — dead code (safe deletions)

- [x] **Dead Rust structural ops.** `native/src/lib.rs:2173-2231` — `tiny_broadcast_to`,
  `tiny_permute`, `tiny_narrow` are never used (the loop evaluator does these as
  zero-copy `Buf` metadata rewrites). Delete them.
- [x] **Dead `lazily()`.** `src/lazy.ts:49` (exported at `:415`) is never called;
  only `eagerly` is used.
- [x] **Dead `evalGraphNative()`.** `src/backends/native.ts:135` plus the one-shot
  `evalGraph` NAPI binding are unused — the path uses `prepareGraphNative` +
  `evalPreparedNative`.
- [x] **Vestigial `Layer` interface.** `src/nn.ts:63-73` (re-exported at `index.ts:18`)
  — nothing implements it, and its rank-2-only `forward` contradicts `Linear`.
  Remove it or make `Linear`/`Activation` actually implement it.
- [x] **Dead test helper.** `test/optim-graph.test.ts:66-72` — `cloneParams` is
  never called.
- [x] **Dead import.** `test/random.test.ts:4` — `tensor` imported but unused.

## Tier 3 — stale comments / docs

- [ ] **"typenet has no integer dtype".** `src/tensor.ts:941-945` — now false
  (`int32`/`int64` exist). Reword to mirror `eager.ts:353-355`.
- [ ] **"use `Math.random`".** `src/lazy.ts:24-28` — `configure({ seed })` says it
  does not affect `Tensor.rand`/`randn`, "which use `Math.random`". Wrong: they
  draw from the seeded hash generator and `configure({seed})` reseeds them.
- [x] **Misattached doc comment.** `native/src/lib.rs:2282-2286` — a `///` block
  about "this stub remains…" now documents `reinsert_dim`; there is no stub.
- [ ] **Garbled `sent_shape` doc.** `native/src/lib.rs:349-351` — first line
  describes the previous function `node_shapes`.
- [ ] **Overstating header.** `test/shape-cases.ts:1-10` — claims every table runs
  in both `types.test-d.ts` and `shape.test.ts`, but the `BROADCAST_TO_*` tables
  are runtime-only.
- [ ] **Stale test name.** `test/native.test.ts:134` — title says "view / permute /
  narrow / broadcastTo" but the body tests only view/permute/transpose.
- [ ] **Phantom scratch files.** `PLAN.md:113-114` — `examples/_gatdsl.ts` and
  `examples/_yieldshape.ts` don't exist, and `examples/basic.ts` is tracked, not
  "uncommitted".

## Tier 4 — duplication (extract a shared helper)

- [ ] **dtype promotion ×6.** `src/ir.ts:304,402,437` + `src/eager.ts:47,196,309`
  — identical `a.dtype === "float64" || b.dtype === "float64" ? "float64" : "float32"`.
  Extract `promoteBinaryDtype()` into `storage.ts` (also gives one place to decide
  integer promotion).
- [ ] **`GraphUpdate` duplicated.** `src/optim.ts:10` vs `src/compile.ts:9-12`.
  Export from `compile.ts` and import in `optim.ts`.
- [ ] **`AnyTensor` alias ×6.** `src/optim.ts:8`, `src/nn.ts:4`, and ×4 in
  `examples/gnca/*` — import the exported `AnyTensor` from `tensor.ts:77` instead.
- [ ] **Redundant re-export + import.** `src/storage.ts:5-6` — the same four types
  are both `export type {...} from` and `import type {...} from` the same module.
- [ ] **XOR net fixtures ×5.** `test/lazy.test.ts:211-235,312-336`,
  `test/native.test.ts:176-200,277-301`, `test/optim-graph.test.ts:32-64` —
  identical `x/y/w1/b1/w2/b2` literals. Extract `test/xor-net.ts`.
- [ ] **`XorNet` + dataset ×3.** `examples/xor.ts`, `examples/bench.ts`, `README.md`.
  Extract a shared `examples/xor-net.ts`.
- [ ] **Test helpers.** `expectClose` ×4 (`compile/lazy/native/optim-graph`) and
  `bothWays` ×2 (`lazy/native`). Extract `test/helpers.ts` (fixes an arg-order
  drift risk).
- [ ] **Duplicate grad-check harness.** `test/autograd.test.ts:9-51` re-implements
  the strictly better `gradcheck.test.ts` harness; every numerical case overlaps.
  Keep only the analytic tests + `stack`/f64 variants.
- [ ] **GNCA example duplication.** rollout loop ×3, compiled training step ×2,
  two timing helpers, two accuracy loops across `examples/gnca/*` and `examples/`.
  Extract shared helpers.
- [ ] **Rust op tables ×3.** `native/src/lib.rs:537-592` (`eval_binary`/`eval_unary`),
  `640-691` (`Bin::parse`/`Un::parse`), `717-782` (`apply_bin`/`apply_un`) — a
  formula drift is undetectable. Single source of truth (macro/table) + a
  scalar-vs-tensor parity test.
- [ ] **Rust release/decode duplication.** buffer-release countdown ×2
  (`execute()` 1631-1639 vs `run_graph()` 2669-2675); LE-decode loops ×4
  (`read_leaf_*`). Extract shared helpers.

## Tier 5 — config / tooling

- [ ] **`prettier` is dead weight.** `package.json:26` (devDep) + `:32-40` (config)
  + `test/types.test-d.ts:301` (`// prettier-ignore`) — dprint is the only
  formatter and the configs conflict (`printWidth: 60` vs `lineWidth: 150`).
  Remove prettier.
- [ ] **Empty `paseo.json`.** A 3-byte `{}` placeholder with zero in-repo references.
  Populate or delete.
- [ ] **Hardcoded path.** `package.json:16` `gnca:bench` bakes
  `$HOME/code/graph-cellular-automata/.venv/bin/python`; the script already
  supports `GCA_ROOT`.
- [ ] **Dead dprint excludes.** `dprint.json:11-12` excludes `**/ideas/**` and
  `**/old/**` — neither directory exists.
- [ ] **Unreferenced examples.** `examples/bench.ts` (superseded by `gnca/bench*.ts`)
  and `examples/basic.ts` (scratch dup of the README intro). Delete or wire up a
  script.
- [ ] **Over-broad `as any` casts.** `src/nn.ts:59,85,125,173` (`forward` returns
  erase the shape contract) and `src/optim.ts:49,51,52` (no-op `as number` in the
  `nums` algebra). Narrow to the real return types.
