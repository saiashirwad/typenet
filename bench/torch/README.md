# bench/torch/ — the PyTorch mirrors

Line-for-line PyTorch equivalents of the shapes/optimizers/losses in
`bench/models/**` and `bench/micro-*.ts`, so every typenet number in
`bench/results/*.jsonl` has a torch number next to it (PLAN-V2 §4.2, W0.4).

## The `TORCH_PYTHON` contract

**`bench/torch/lib/python.mjs` is the single place `$TORCH_PYTHON` is
resolved in this repo.** Nothing else may name a python interpreter — not
the one on `PATH`, not a second default — because the system interpreter on
this machine has no torch installed, and falling back to it would silently
produce a skip that _looks_ like "no torch anywhere" and is not.

```
TORCH_PYTHON ?? /Users/texoport/code/graph-cellular-automata/.venv/bin/python
```

**Measured on this machine**: `torch 2.13.0`, `torch.backends.mps.is_available() == True`
(python 3.12, at the default path above). Every torch-touching item in
PLAN-V2 (W0.4, W5.8, gate G5.5, D15's Metal trigger) is therefore
measurable today, not deferred.

Every script that needs torch — every `bench_*.py` here, and later W5.8's
`scripts/regen-reference.mjs` — imports `bench/torch/lib/python.mjs` and
calls `torchPython()` / `hasTorch()` / `runTorch()`. None of them resolves
an interpreter on their own. A missing `TORCH_PYTHON`, one pointed at a
file that isn't python, or one whose python has no torch installed are all
**loud, one-line skips naming the variable and the path that was tried** —
never a silent pass, and never a fallback to `PATH`.

A grep for a second hardcoded interpreter name across `bench/` and
`test/fixtures/` should find no interpreter resolution anywhere outside
this contract.

## Running it

```
pnpm vite-node bench/torch/run.ts               # discovers every bench_*.py, runs torch-cpu (+ torch-mps)
pnpm vite-node bench/torch/run.ts -- --only mlp # substring filter, same convention as bench/lib/cli.ts
pnpm vite-node bench/torch/run.ts -- --tag foo  # written into every JSONL line
```

Every script defaults to a **smoke run** — tiny sizes, 1 warm-up, 2 timed
samples — exactly like the rest of `bench/**` (`bench/README.md`,
PLAN-V2 §5A.0: Phase A never passes `--full`, and no agent may run one
that does). `--full` opts a script into the real `bench/lib/sizes.ts`-
equivalent shapes and a >= 10 sample floor. Smoke and full results never
share a file: smoke lines go to `bench/results/smoke/torch-<script>.jsonl`,
full lines to `bench/results/torch-<script>.jsonl`, and every line carries
`smoke: true | false`.

## The timing boundary

Identical on both sides of the language boundary, matching
`bench/lib/harness.ts` exactly:

- **Post-warmup, per-step wall time.** Warm-up iterations are run and
  discarded before any sample is timed.
- **`torch.mps.synchronize()` runs _inside_ the timed region** on the
  `torch-mps` mode (`bench/torch/lib/harness.py`'s `run_timed`) — a step's
  queued-but-not-yet-executed GPU work is never excluded from its own
  timing, the same way a compiled typenet step's readback is never
  excluded from its own timing.
- **Medians of >= 10** samples for a `--full` run (2 for a smoke run, which
  exists to prove the script still runs end to end, not to produce a
  trustworthy number).

**A ratio may never mix boundaries.** Comparing a `torch-cpu`/`torch-mps`
number against a typenet number is only valid when both were measured
under this same boundary — post-warmup wall time, GPU work synchronized
inside the timed region, medians of the same sample floor. Comparing a
smoke number (either side) against a full number, or a `torch-mps` number
against a typenet `native` number gathered under a different harness
version, is exactly the "mixing two different models" mistake PLAN-V2
§4.2 already forbids for `mlp-legacy` vs `mlp-modern` — the same rule
applies across languages, not just within one.

## Scripts

| script               | mirrors                                                     | cases                                                                                                                                 |
| -------------------- | ----------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| `bench_mlp.py`       | `bench/macro-mlp.ts` + `bench/models/mlp.ts`                | `mlp-legacy-*`: `Linear(784,256) -> ReLU -> Linear(256,10)`, `mseLoss`, `Adam(lr 1e-3)`                                               |
| `bench_attention.py` | `bench/macro-attention.ts` + `bench/models/attention.ts`    | `attn-{s,m,l,smoke}[-fwd]`: unfused Q/K/V/output causal self-attention at the nanoGPT head configs                                    |
| `bench_nanogpt.py`   | `bench/macro-nanogpt.ts` + `bench/models/nanogpt-legacy.ts` | `nanogpt-{s,m,l,smoke}-{fwd,bwd,step}`: karpathy-shaped nanoGPT, fused-QKV attention, GELU-tanh MLP, `cross_entropy`, `Adam(lr 3e-4)` |
| `bench_ops.py`       | `bench/micro-{matmul,elementwise,reduce,softmax-ln}.ts`     | `mm-square-*`, `ew-n*-chain*`, `reduce-*`, `softmax-w*`, `ce-logsumexp`                                                               |

Every JSONL line uses the same envelope as `bench/lib/report.ts`
(`ts`/`host`/`cores`/`git`/`dirty`/`mode`/`script`/`case`/`n`/`median_ms`/
`p10_ms`/`p90_ms`/`counters`/`smoke`/`tag?`), with `mode` one of
`"torch-cpu"` / `"torch-mps"` and every structural counter reported `-1` —
a torch run measures none of typenet's own structural counters (PLAN-V2
§2.9), the same convention `bench/typecheck.ts` uses for a `tsc` run.
