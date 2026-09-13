# CI

## What runs

`.github/workflows/ci.yml` defines two jobs:

- **`test`** — a `macos-14` (Apple Silicon) matrix with two legs, `simd: [default,
  no_simd]`. The `no_simd` leg sets `TYPENET_NO_SIMD=1`; it is a no-op until the
  SIMD kernel work lands, wired now so the leg exists before the code that
  needs it. Both legs run the same steps: `pnpm install --frozen-lockfile`,
  `pnpm build:native`, then `pnpm test:ci` (added by W0.1), which is
  `pnpm test && pnpm typecheck && pnpm typecheck:budget && cargo clippy
  --manifest-path native/Cargo.toml --all-targets -- -D warnings`. This is a
  gate: both legs must be green for the workflow to pass.
- **`bench`** — runs `pnpm bench:micro` and uploads `bench/results/*.jsonl` as
  a build artifact. `continue-on-error: true` and no other job depends on it.

**The bench job is not a gate.** GitHub-hosted `macos-14` runners are shared,
laptop-class, and noisy-neighbor VMs; they cannot hold a 3% no-regression
clause, and a red bench job here means nothing about a real regression. Its
only purpose is to leave a downloadable trail of `.jsonl` lines per commit.
Every number quoted in the plan's §6 gates and §0 facts is measured on a
dedicated, otherwise-idle machine — **Apple M5, 4 "Super" P-cores + 6
E-cores** (the same topology this repository's benches print via
`sysctl -n hw.perflevel0.physicalcpu hw.perflevel1.physicalcpu`) — never on a
CI runner. Treat `bench-results` CI artifacts as a smoke check that the
scripts still run end-to-end, not as a source of truth for any ratio or gate.

## Reproducing the test steps locally

```
pnpm test:ci
```

runs exactly the four gated steps the `test` job runs (after `pnpm install`
and `pnpm build:native`, which the workflow also runs first). Set
`TYPENET_NO_SIMD=1` in the environment first to reproduce the second matrix
leg:

```
TYPENET_NO_SIMD=1 pnpm test:ci
```

## Validating the workflow file

`act -n` (installed via `brew install act`) parses `ci.yml` and dry-runs the
job graph: it resolves the `simd` matrix into two legs (`test
(simd=default)-1`, `test (simd=no_simd)-2`) plus the `bench` job with no
expression or schema errors. It cannot execute past "Set up job" in this
sandbox because no Docker daemon is reachable here, and `act` has no
`macos-14` runner image to begin with (it only ever substitutes a Linux
container for any `runs-on` label) — so a full local `act` run would not
exercise anything macOS/Xcode/cargo-specific anyway. `actionlint` (installed
via `brew install actionlint`) reports zero problems against
`.github/workflows/ci.yml`. Both matrix legs were additionally verified green
by running their real commands directly on this machine (which is itself
`macos-14`-equivalent, Apple Silicon): `pnpm install --frozen-lockfile`,
`pnpm build:native`, `pnpm test:ci`, and again with `TYPENET_NO_SIMD=1` — see
the work item's acceptance log for the captured output. The authoritative
validation remains a pushed run on GitHub's actual `macos-14` runners.

## Suite wall clock

Baseline, current tree, `pnpm test` (`vitest run`), three consecutive runs:

| run | test files | tests                  | wall duration |
| --- | ---------- | ---------------------- | ------------- |
| 1   | 15         | 333 (+1 expected fail) | 20.07 s       |
| 2   | 15         | 333 (+1 expected fail) | 19.68 s       |
| 3   | 15         | 333 (+1 expected fail) | 18.87 s       |

(matches the plan's own re-taken measurement of "333 tests, 15 files, 18.6 s
wall" within run-to-run noise.)

Two `vite.config.ts` `test` options from UNDERSTAND §7.3 item 10 were tried and
reverted — the file ships unchanged:

- **`isolate: false` alone**: two runs at 20.08 s / 19.98 s — no improvement
  over baseline outside noise. `transform`/`import` cumulative time (the real
  cost, ~110-165 s of CPU time across workers) is already parallelized across
  this machine's 10 cores regardless of per-file isolation, so removing
  isolation buys nothing here.
- **`isolate: false` + `poolOptions.threads.singleThread: true`**: same ~20 s
  wall, but now **1 test file fails** (282/283 pass instead of 333/334) —
  running every test file in one shared thread/module registry surfaces
  cross-file state pollution. Rejected on correctness grounds alone, independent
  of speed.

Neither change is adopted; `pnpm test` still runs the vitest default
(per-file isolation, thread pool auto-sized to the machine).

**Pre-existing flake, unrelated to this item**: `pnpm vitest run
--sequence.shuffle` intermittently fails `test/random.test.ts > uniform >
fills the unit interval` (a statistical tolerance check, e.g. "expected
0.2836 to be close to 0.2887 ... difference is 0.0051, but expected 0.005") —
this reproduces on the _unmodified_ baseline config too, so it is a flaky
statistical assertion in that file, not a shuffle-induced ordering/isolation
bug and not something this item's `vite.config.ts`/CI change causes or fixes.
`test/random.test.ts` is outside this item's file ownership.
