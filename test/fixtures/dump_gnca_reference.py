"""Dump a PyTorch reference for the graph-CA parity test.

Run from the reference repo, which owns the model this is checked against:

    cd ~/code/graph-cellular-automata
    .venv/bin/python ~/code/typenet/test/fixtures/dump_gnca_reference.py \
        > ~/code/typenet/test/fixtures/gnca-reference.json

Weights come out already transposed into typenet's layout (x @ W, so
[in, out]) rather than PyTorch's [out, in]. update_rate is 1.0 so the
stochastic mask is all ones and the rollout is deterministic; everything
else about the rule is exercised.
"""
import json
import sys

import numpy as np
import torch

sys.path.insert(0, "src")
from gnca import GraphNCA, alive_mask, knn_graph  # noqa: E402

CHANNELS = 8
HIDDEN = 12
NODES = 40
K = 4
STEPS = 3
BATCH = 2

torch.manual_seed(0)
rng = np.random.default_rng(0)

pos = rng.random((NODES, 2), dtype=np.float32)
edges = knn_graph(pos, k=K)

# A batch is B copies of the graph side by side in the node dimension.
offsets = torch.arange(BATCH, dtype=torch.int64)[:, None] * NODES
bedges = (edges[None] + offsets[..., None]).permute(1, 0, 2).reshape(2, -1)

deg = torch.zeros(BATCH * NODES, 1)
deg.index_add_(0, bedges[1], torch.ones(bedges.shape[1], 1))

model = GraphNCA(channels=CHANNELS, hidden=HIDDEN)
# Non-trivial weights everywhere: the zero-init gate and last layer would
# hide any mistake in either.
with torch.no_grad():
    for p in model.parameters():
        p.copy_(torch.from_numpy(
            rng.standard_normal(tuple(p.shape)).astype(np.float32) * 0.3))

x0 = torch.from_numpy(
    rng.standard_normal((BATCH * NODES, CHANNELS)).astype(np.float32))
target = torch.from_numpy(
    rng.random((BATCH * NODES, 4), dtype=np.float32))

x = x0.clone().requires_grad_(True)
state = x
for _ in range(STEPS):
    mask = alive_mask(state, bedges)
    state = model(state, bedges, update_rate=1.0, deg=deg) * mask
mse = ((state[:, :4] - target) ** 2).mean()
loss = mse + (state - state.clamp(-1, 1)).abs().mean()
loss.backward()

# PyTorch keeps Linear weights as [out, in]; typenet as [in, out].
names = ["net.0.weight", "net.0.bias", "net.2.weight",
         "gate.weight", "gate.bias"]
params = dict(model.named_parameters())


def out(t):
    return [round(float(v), 7) for v in t.detach().flatten()]


json.dump({
    "channels": CHANNELS, "hidden": HIDDEN, "nodes": NODES,
    "k": K, "steps": STEPS, "batch": BATCH,
    "pos": out(torch.from_numpy(pos)),
    "edges": {"src": out(edges[0].float()), "dst": out(edges[1].float())},
    "weights": {
        name: out(params[name].T if params[name].dim() == 2 else params[name])
        for name in names
    },
    "gradients": {
        name: out(params[name].grad.T if params[name].dim() == 2
                  else params[name].grad)
        for name in names
    },
    "x0": out(x0),
    "target": out(target),
    "state": out(state),
    "mse": round(float(mse), 7),
    "loss": round(float(loss), 7),
    "xGrad": out(x.grad),
}, sys.stdout)
