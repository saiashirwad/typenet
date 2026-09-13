"""PyTorch mirror of `bench/macro-mlp.ts` + `bench/models/mlp.ts`
(PLAN-V2 §4.2, W0.4).

Same frozen `mlp-legacy` shape as the TypeScript side: `Linear(784,256) ->
ReLU -> Linear(256,10)`, `mseLoss`, `Adam(lr 1e-3)`. One zero_grad + forward
+ loss + backward + step training step per timed sample, at the identical
timing boundary `bench/torch/README.md` documents.

Prints one JSON line per case to stdout: `{"case", "n", "median_ms",
"p10_ms", "p90_ms"}`. `bench/torch/run.ts` wraps each line with the
run-level envelope (ts/host/cores/git/mode/...) and appends it to
`bench/results/torch-mlp.jsonl` (or `bench/results/smoke/torch-mlp.jsonl`
for a smoke run, the default).
"""
import json
import sys

import torch

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from lib.cli import parse_args
from lib.harness import run_timed
from sizes import MLP_LEGACY, MLP_SMOKE


def build_case(case, device):
    net = torch.nn.Sequential(
        torch.nn.Linear(case["inputDim"], case["hiddenDim"]),
        torch.nn.ReLU(),
        torch.nn.Linear(case["hiddenDim"], case["outputDim"]),
    ).to(device)
    x = torch.rand(case["batch"], case["inputDim"], device=device)
    y = torch.rand(case["batch"], case["outputDim"], device=device)
    optim = torch.optim.Adam(net.parameters(), lr=1e-3)
    return net, x, y, optim


def train_step(net, x, y, optim):
    optim.zero_grad()
    loss = torch.nn.functional.mse_loss(net(x), y)
    loss.backward()
    optim.step()
    return loss


def main():
    args = parse_args()
    device = args.device
    cases = MLP_LEGACY if args.full else MLP_SMOKE

    for case in cases:
        if args.only is not None and args.only not in case["id"]:
            continue
        net, x, y, optim = build_case(case, device)
        stats = run_timed(lambda: train_step(net, x, y, optim), device, args.full)
        print(json.dumps({"case": case["id"], **stats}))


if __name__ == "__main__":
    main()
