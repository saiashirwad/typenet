"""The other column of the published race: the same GNCA training step
on torch.mps.

Recipe (must match bench_gate.ts): 1024 nodes, k=8, batch 8, C=16,
H=128, one compiled bucket in [48, 80] (64 steps), forward + backward +
grad clip + Adam, three-run median.

The model comes from the reference repo, which owns the rule:

    ~/code/graph-cellular-automata/.venv/bin/python examples/gnca/bench_torch.py

Set GCA_ROOT to point elsewhere.
"""

import os
import sys
import time

import numpy as np
import torch

ROOT = os.environ.get(
    "GCA_ROOT",
    os.path.expanduser("~/code/graph-cellular-automata"),
)
sys.path.insert(0, os.path.join(ROOT, "src"))
from gnca import GraphNCA, alive_mask, knn_graph  # noqa: E402

NODES = 1024
K = 8
BATCH = 8
CHANNELS = 16
HIDDEN = 128
STEPS = 64

device = "mps" if torch.backends.mps.is_available() else "cpu"
torch.manual_seed(1)
rng = np.random.default_rng(0)

pos = rng.random((NODES, 2), dtype=np.float32)
edges = knn_graph(pos, k=K)

offsets = torch.arange(BATCH, dtype=torch.int64)[:, None] * NODES
bedges = (
    (edges[None] + offsets[..., None]).permute(1, 0, 2).reshape(2, -1)
)

deg = torch.zeros(BATCH * NODES, 1)
deg.index_add_(0, bedges[1], torch.ones(bedges.shape[1], 1))

model = GraphNCA(channels=CHANNELS, hidden=HIDDEN).to(device)
bedges = bedges.to(device)
deg = deg.to(device)

x0 = torch.zeros(BATCH * NODES, CHANNELS, device=device)
for b in range(BATCH):
    x0[b * NODES, 3:] = 1.0
target = torch.rand(BATCH * NODES, 4, device=device)

optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)


def step() -> float:
    x = x0.clone()
    for _ in range(STEPS):
        mask = alive_mask(x, bedges)
        x = model(x, bedges, update_rate=0.5, deg=deg) * mask
    loss = torch.mean((x[:, :4] - target) ** 2)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    if device == "mps":
        torch.mps.synchronize()
    return float(loss.detach().cpu())


print(
    f"backend: torch {torch.__version__}, device {device}\n"
    f"{NODES} nodes x {BATCH} = {BATCH * NODES}, "
    f"{bedges.shape[1]} edges, C={CHANNELS}, H={HIDDEN}, "
    f"{STEPS} rollout steps"
)

step()  # warmup

runs = []
for _ in range(3):
    start = time.perf_counter()
    step()
    runs.append((time.perf_counter() - start) * 1000)
runs.sort()
median = runs[1]
print(f"runs (ms): {', '.join(f'{r:.0f}' for r in runs)}")
print(
    f"median full step: {median:.1f} ms = "
    f"{1000 / median:.2f} training steps/s"
)
