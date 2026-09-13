"""PyTorch mirror of `bench/macro-nanogpt.ts` + `bench/models/nanogpt-legacy.ts`
(PLAN-V2 §4.2, W0.4) — karpathy/nanoGPT's `model.py` shape: token + learned
position embedding, `nLayer` pre-norm transformer blocks (fused-QKV causal
self-attention, GELU-tanh MLP), a final LayerNorm, an untied output head,
and `cross_entropy` on a synthetic next-token batch.

Three variants per size, matching the TS side's case-id convention exactly:
`nanogpt-<letter>-fwd` (forward only), `-bwd` (forward + backward, no
optimizer step), `-step` (forward + backward + `Adam.step()`).

Prints one JSON line per case to stdout: `{"case", "n", "median_ms",
"p10_ms", "p90_ms"}`.
"""
import json
import math
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from lib.cli import parse_args
from lib.harness import run_timed
from sizes import NANOGPT_SIZES, NANOGPT_SMOKE


def gelu_tanh(x):
    """The tanh approximation (Hendrycks & Gimpel), matching
    `bench/models/nanogpt-legacy.ts`'s hand-composed `gelu` exactly rather
    than `torch.nn.functional.gelu`'s exact-erf default."""
    c = math.sqrt(2.0 / math.pi)
    inner = c * (x + 0.044715 * x.pow(3))
    return 0.5 * x * (1.0 + torch.tanh(inner))


class CausalSelfAttention(nn.Module):
    """Fused QKV projection, split into heads, causal masked softmax
    attention — the same shape `bench/models/nanogpt-legacy.ts` composes."""

    def __init__(self, n_embd, n_head, block_size):
        super().__init__()
        if n_embd % n_head != 0:
            raise ValueError(f"CausalSelfAttention: n_embd {n_embd} is not divisible by n_head {n_head}")
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.qkv = nn.Linear(n_embd, 3 * n_embd)
        self.proj = nn.Linear(n_embd, n_embd)
        mask = torch.triu(torch.ones(block_size, block_size), diagonal=1).bool()
        self.register_buffer("causal_mask", mask)

    def forward(self, x):
        b, t, c = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.split(c, dim=-1)

        def split_heads(u):
            return u.view(b, t, self.n_head, self.head_dim).permute(0, 2, 1, 3)

        qh, kh, vh = split_heads(q), split_heads(k), split_heads(v)
        scale = 1.0 / math.sqrt(self.head_dim)
        scores = (qh @ kh.transpose(-2, -1)) * scale
        scores = scores.masked_fill(self.causal_mask[:t, :t], float("-1e9"))
        attn = torch.softmax(scores, dim=-1)
        merged = (attn @ vh).permute(0, 2, 1, 3).reshape(b, t, c)
        return self.proj(merged)


class MLP(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.fc = nn.Linear(n_embd, 4 * n_embd)
        self.proj = nn.Linear(4 * n_embd, n_embd)

    def forward(self, x):
        return self.proj(gelu_tanh(self.fc(x)))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg["nEmbd"])
        self.attn = CausalSelfAttention(cfg["nEmbd"], cfg["nHead"], cfg["blockSize"])
        self.ln2 = nn.LayerNorm(cfg["nEmbd"])
        self.mlp = MLP(cfg["nEmbd"])

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class NanoGPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.wte = nn.Embedding(cfg["vocabSize"], cfg["nEmbd"])
        self.wpe = nn.Embedding(cfg["blockSize"], cfg["nEmbd"])
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg["nLayer"])])
        self.ln_f = nn.LayerNorm(cfg["nEmbd"])
        self.head = nn.Linear(cfg["nEmbd"], cfg["vocabSize"])

    def forward(self, idx):
        b, t = idx.shape
        pos = torch.arange(t, device=idx.device)
        x = self.wte(idx) + self.wpe(pos)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.head(x)


def generate_batch(cfg, device):
    """A synthetic next-token batch, shaped exactly like
    `bench/models/nanogpt-legacy.ts`'s `generateBatch`: row `b`'s first
    `blockSize` ids are the input; the same row shifted one position over
    is the next-token target. Not bit-identical to the TS RNG stream (the
    two run on different RNGs entirely) — this bench compares timings and
    shapes, not numerics, across the language boundary."""
    batch, block_size, vocab = cfg["batch"], cfg["blockSize"], cfg["vocabSize"]
    u = torch.randint(0, vocab, (batch, block_size + 1), device=device)
    idx = u[:, :block_size]
    targets = u[:, 1 : block_size + 1]
    return idx, targets


def run_forward(model, idx):
    return model(idx)


def run_forward_backward(model, idx, targets):
    model.zero_grad(set_to_none=True)
    logits = model(idx)
    b, t, v = logits.shape
    loss = F.cross_entropy(logits.reshape(b * t, v), targets.reshape(b * t))
    loss.backward()
    return loss


def run_step(model, idx, targets, optim):
    loss = run_forward_backward(model, idx, targets)
    optim.step()
    return loss


def main():
    args = parse_args()
    device = args.device
    sizes = NANOGPT_SIZES if args.full else [NANOGPT_SMOKE]

    for size in sizes:
        letter = size["id"].split("-")[1]
        model = NanoGPT(size).to(device)
        idx, targets = generate_batch(size, device)
        optim = torch.optim.Adam(model.parameters(), lr=3e-4)

        fwd_id = f"nanogpt-{letter}-fwd"
        if args.only is None or args.only in fwd_id:
            stats = run_timed(lambda: run_forward(model, idx), device, args.full)
            print(json.dumps({"case": fwd_id, **stats}))

        bwd_id = f"nanogpt-{letter}-bwd"
        if args.only is None or args.only in bwd_id:
            stats = run_timed(lambda: run_forward_backward(model, idx, targets), device, args.full)
            print(json.dumps({"case": bwd_id, **stats}))

        step_id = f"nanogpt-{letter}-step"
        if args.only is None or args.only in step_id:
            stats = run_timed(lambda: run_step(model, idx, targets, optim), device, args.full)
            print(json.dumps({"case": step_id, **stats}))


if __name__ == "__main__":
    main()
