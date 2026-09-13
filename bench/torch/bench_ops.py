"""PyTorch mirror of typenet's micro-op benches — `bench/micro-matmul.ts`,
`bench/micro-elementwise.ts`, `bench/micro-reduce.ts` and
`bench/micro-softmax-ln.ts` (PLAN-V2 §4.2's `bench/micro-*.ts` table, W0.4).

One script rather than four (this item's Files line names a single
`bench_ops.py`): each of the four TS scripts gets its own case-id prefix
(`mm-`, `ew-`, `reduce-`, `softmax-`/`ce-`) so a line from any of them is
unambiguous in the shared `bench/results/torch-ops.jsonl` output.

Prints one JSON line per case to stdout: `{"case", "n", "median_ms",
"p10_ms", "p90_ms"}`.
"""
import json
import sys

import torch

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from lib.cli import parse_args
from lib.harness import run_timed
from sizes import ELEMENTWISE_FULL, ELEMENTWISE_SMOKE, MATMUL_FULL, MATMUL_SMOKE, REDUCE_SHAPES_FULL, \
    REDUCE_SHAPES_SMOKE, SOFTMAX_LN_FULL, SOFTMAX_LN_SMOKE


def maybe_run(only, case_id, fn, device, full, results):
    if only is not None and only not in case_id:
        return
    stats = run_timed(fn, device, full)
    results.append({"case": case_id, **stats})


def bench_matmul(device, full, only, results):
    """`torch.matmul` on square operands — `bench/micro-matmul.ts`'s
    `mm-square-*` cases."""
    cfg = MATMUL_FULL if full else MATMUL_SMOKE
    for n in cfg["square"]:
        a = torch.rand(n, n, device=device)
        b = torch.rand(n, n, device=device)
        maybe_run(only, f"mm-square-{n}", lambda a=a, b=b: torch.matmul(a, b), device, full, results)


def bench_elementwise(device, full, only, results):
    """A chain of `torch.tanh(a * b + c)` steps — `bench/micro-elementwise.ts`'s
    chain-of-ops cases."""
    cfg = ELEMENTWISE_FULL if full else ELEMENTWISE_SMOKE
    for n in cfg["sizes"]:
        a = torch.rand(n, device=device)
        b = torch.rand(n, device=device)
        c = torch.rand(n, device=device)
        for chain_len in cfg["chainLengths"]:

            def chain(a=a, b=b, c=c, chain_len=chain_len):
                x = a
                for _ in range(chain_len):
                    x = torch.tanh(x * b + c)
                return x

            maybe_run(only, f"ew-n{n}-chain{chain_len}", chain, device, full, results)


def bench_reduce(device, full, only, results):
    """`torch.sum(dim=-1)` — `bench/micro-reduce.ts`'s per-shape reduce cases."""
    shapes = REDUCE_SHAPES_FULL if full else REDUCE_SHAPES_SMOKE
    for shape_cfg in shapes:
        x = torch.rand(*shape_cfg["shape"], device=device)
        maybe_run(only, f"reduce-{shape_cfg['label']}", lambda x=x: torch.sum(x, dim=-1), device, full, results)


def bench_softmax_ln(device, full, only, results):
    """`torch.softmax` over a sweep of row widths, and `torch.cross_entropy`
    at the shared vocab shape — `bench/micro-softmax-ln.ts`'s cases."""
    cfg = SOFTMAX_LN_FULL if full else SOFTMAX_LN_SMOKE
    for width in cfg["widths"]:
        x = torch.rand(cfg["rows"], width, device=device)
        maybe_run(only, f"softmax-w{width}", lambda x=x: torch.softmax(x, dim=-1), device, full, results)

    logits = torch.rand(cfg["ceRows"], cfg["ceCols"], device=device, requires_grad=True)
    targets = torch.randint(0, cfg["ceCols"], (cfg["ceRows"],), device=device)
    maybe_run(
        only,
        "ce-logsumexp",
        lambda: torch.nn.functional.cross_entropy(logits, targets),
        device,
        full,
        results,
    )


def main():
    args = parse_args()
    device, full, only = args.device, args.full, args.only

    results = []
    bench_matmul(device, full, only, results)
    bench_elementwise(device, full, only, results)
    bench_reduce(device, full, only, results)
    bench_softmax_ln(device, full, only, results)

    for line in results:
        print(json.dumps(line))


if __name__ == "__main__":
    main()
