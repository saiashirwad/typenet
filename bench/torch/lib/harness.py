"""Shared timing utility for the torch bench mirrors (PLAN-V2 §4.2, W0.4).

Mirrors `bench/lib/harness.ts`'s timing shape closely enough that a line
from either side is directly comparable: post-warmup per-step wall time,
medians of a fixed sample count. `torch.mps.synchronize()` runs *inside*
the timed region on MPS (`run_timed`'s `device == "mps"` branch) — the
identical timing boundary rule `bench/torch/README.md` documents, so a
`torch-mps` number is never a queue-a-few-and-walk-away measurement.

Smoke (the default; see `bench/README.md` and PLAN-V2 §5A.0 — Phase A never
passes `--full`) floors to 1 warm-up / 2 timed samples, exactly like
`bench/lib/harness.ts`'s own smoke floor; `--full` opts into 3 warm-up / 10
timed samples, satisfying W0.4's "medians of >= 10" for a full run.
"""
import time


def percentile(sorted_values, p):
    if len(sorted_values) == 1:
        return sorted_values[0]
    idx = (len(sorted_values) - 1) * p
    lo = int(idx)
    hi = min(lo + 1, len(sorted_values) - 1)
    if lo == hi:
        return sorted_values[lo]
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (idx - lo)


def compute_stats(samples_ms):
    ordered = sorted(samples_ms)
    return {
        "n": len(ordered),
        "median_ms": percentile(ordered, 0.5),
        "p10_ms": percentile(ordered, 0.10),
        "p90_ms": percentile(ordered, 0.90),
    }


def run_timed(fn, device, full):
    """Runs `fn()` under a warm-up + timed-sample loop and returns
    `{n, median_ms, p10_ms, p90_ms}`. `device` is `"cpu"` or `"mps"` — on
    `"mps"` each timed call is followed by `torch.mps.synchronize()` inside
    the timed region, matching the identical boundary rule W0.4 states for
    the TS/torch comparison.
    """
    warmup = 3 if full else 1
    samples = 10 if full else 2

    for _ in range(warmup):
        fn()

    timings_ms = []
    for _ in range(samples):
        t0 = time.perf_counter()
        fn()
        if device == "mps":
            import torch

            torch.mps.synchronize()
        timings_ms.append((time.perf_counter() - t0) * 1000.0)

    return compute_stats(timings_ms)
