"""PyTorch mirror of `bench/macro-attention.ts` + `bench/models/attention.ts`
(PLAN-V2 §4.2, W0.4).

One causal multi-head self-attention block — separate Q/K/V/output
projections (no fused QKV, matching `bench/models/attention.ts`), scaled
dot-product attention with an additive causal mask, output projection — run
forward-only and forward+backward, at the nanoGPT S/M/L head configs
(`nanogpt-s`/`-m`/`-l`, and the single `nanogpt-smoke` config by default;
Phase A never passes `--full`, PLAN-V2 §5A.0).

Case ids follow the TS side's convention exactly: `attn-<letter>` (forward
+backward) and `attn-<letter>-fwd` (forward only), where `<letter>` is the
size id's suffix (`s`/`m`/`l`, or `smoke`).

Prints one JSON line per case to stdout: `{"case", "n", "median_ms",
"p10_ms", "p90_ms"}`.
"""
import json
import math
import sys

import torch

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from lib.cli import parse_args
from lib.harness import run_timed
from sizes import NANOGPT_SIZES, NANOGPT_SMOKE


class CausalSelfAttention(torch.nn.Module):
    """Unfused Q/K/V/output projections, scaled dot-product attention with
    an additive causal mask (0 where a position may attend, -1e9 where it
    may not) — the same shape `bench/models/attention.ts` composes from
    typenet's primitives."""

    def __init__(self, n_embd, n_head, seq_len):
        super().__init__()
        if n_embd % n_head != 0:
            raise ValueError(f"CausalSelfAttention: n_embd ({n_embd}) must be divisible by n_head ({n_head})")
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.q = torch.nn.Linear(n_embd, n_embd)
        self.k = torch.nn.Linear(n_embd, n_embd)
        self.v = torch.nn.Linear(n_embd, n_embd)
        self.proj = torch.nn.Linear(n_embd, n_embd)
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        self.register_buffer("causal_mask", mask)

    def forward(self, x):
        batch, seq_len, n_embd = x.shape

        def to_heads(t):
            return t.view(batch, seq_len, self.n_head, self.head_dim).permute(0, 2, 1, 3)

        q, k, v = to_heads(self.q(x)), to_heads(self.k(x)), to_heads(self.v(x))
        scores = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        scores = scores.masked_fill(self.causal_mask[:seq_len, :seq_len], float("-1e9"))
        attn = torch.softmax(scores, dim=-1)
        merged = (attn @ v).permute(0, 2, 1, 3).reshape(batch, seq_len, n_embd)
        return self.proj(merged)


def random_input(batch, seq_len, n_embd, device):
    return torch.randn(batch, seq_len, n_embd, device=device)


def run_forward(model, x):
    return model(x)


def run_forward_backward(model, x):
    model.zero_grad(set_to_none=True)
    out = model(x)
    out.sum().backward()
    return out


def main():
    args = parse_args()
    device = args.device
    sizes = NANOGPT_SIZES if args.full else [NANOGPT_SMOKE]

    for size in sizes:
        letter = size["id"].split("-")[1]
        model = CausalSelfAttention(size["nEmbd"], size["nHead"], size["blockSize"]).to(device)
        x = random_input(size["batch"], size["blockSize"], size["nEmbd"], device)

        fwd_id = f"attn-{letter}-fwd"
        if args.only is None or args.only in fwd_id:
            stats = run_timed(lambda: run_forward(model, x), device, args.full)
            print(json.dumps({"case": fwd_id, **stats}))

        bwd_id = f"attn-{letter}"
        if args.only is None or args.only in bwd_id:
            stats = run_timed(lambda: run_forward_backward(model, x), device, args.full)
            print(json.dumps({"case": bwd_id, **stats}))


if __name__ == "__main__":
    main()
