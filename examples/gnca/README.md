# Graph cellular automata

A cellular automaton on a graph: one shared update rule, applied to every
node, that grows a pattern from a single seed and regrows it after damage.
Ported from [graph-cellular-automata](../../../graph-cellular-automata),
which is the PyTorch original and the source of truth for the recipe.

```sh
pnpm gnca                            # train with the reference defaults
pnpm gnca --steps 200                # a quick run
pnpm gnca --target star --nodes 512  # a different pattern, smaller graph
pnpm gnca --target bunny --nodes 1536      # a mesh surface, 3-d
pnpm gnca --init-from runs/gnca-heart.json   # warm start
pnpm vite-node examples/gnca/bench.ts
```

Every probe writes a checkpoint and a picture: `runs/gnca-<target>.json`
holds the weights _and_ the graph they were trained on (the same rule on a
different graph is a different model), and `runs/gnca-<target>.svg` shows
target, grown and healed side by side, which is the fastest way to tell
whether a run is working.

`--init-from` warm-starts from a checkpoint, zero-padding the first
layer's input columns if the saved percept was narrower. That padding is
what makes the reference's ablation ladder work: the loaded rule is
_functionally identical_ to begin with, since a zero column contributes
nothing, so adding a percept feature measures the feature rather than a
fresh initialisation.

## Targets

Eight procedural patterns (`heart`, `star`, `annulus`, `lobes`, `ring`
in 2-d; `sphere`, `torus`, `jack` in 3-d) and four mesh surface clouds
(`bunny`, `spot`, `armadillo`, `teapot`).

The clouds are the better 3-d targets and the ones the reference's own
experiments use: the procedural shells paint a thin surface inside a random
cube, so only a fifth of the graph is the pattern and a demo looks like
dust, whereas on a cloud every node sits on the mesh, alpha is 1
everywhere, and colour comes from the surface normal — so a regrown ear
comes back in a colour you can check.

They load from the reference repo's `.npz` files, which is why
`pointclouds.ts` contains a small zip and `.npy` reader; numpy writes
ZIP64, so the reader goes through the central directory rather than
trusting the local headers' sizes. Point `--clouds` elsewhere if your
copies live elsewhere. The loader is checked against Python's values for
the same file, to six decimals.

## The rule

Each node sees `[own state, mean of neighbour states, mean of gated
(neighbour − own) differences, log(1 + degree)]`, and a two-layer MLP
turns that into a residual increment applied to a random half of the nodes
each step.

Three details carry most of the weight, and all three are the reference's:

- The **degree scalar** restores the neighbour _count_ that mean
  aggregation erases.
- The **gate** is per edge and per channel (Perona-Malik style), so the
  rule can turn diffusion off across a pattern boundary instead of
  smearing it. It is zero-initialised, and `2·sigmoid(0) = 1` exactly, so
  at init the gate is plain identity diffusion.
- The **alive mask** kills nodes whose neighbourhood has no alpha, which
  is what keeps growth a front rather than a wash.

The gate is linear in `[x_src, x_dst, |x_src − x_dst|]`, so its two
endpoint terms are computed once over nodes and gathered per edge. Doing
it as one `(E, 3C)` matmul over edges instead costs several times as much:
there are many more edges than nodes.

## What the training loop does

A sample pool, so most steps start from a state some earlier rollout left
behind rather than from the bare seed — that is what teaches the rule to
_hold_ a pattern rather than draw one. Each batch is ranked worst-first
and damage lands on the best-formed samples, because healing a grown
pattern is the behaviour wanted and gradient spent on an already-broken
state is spent twice over. Every eighth step trains one sample from the
bare seed, to keep the long horizon in practice.

The number to watch is the **heal probe**, not the loss: grow for 80
steps, punch a fixed hole, heal for 160, and score. Training loss can fall
while regeneration stays broken.

## Files

| file             | contents                                                    |
| ---------------- | ----------------------------------------------------------- |
| `model.ts`       | the update rule, the alive mask, the seed state             |
| `graphs.ts`      | k-NN, random geometric and Watts-Strogatz graph builders    |
| `targets.ts`     | the eight patterns, 2-d and 3-d, sampled at node positions  |
| `pointclouds.ts` | mesh surface clouds as targets, with a small .npz reader    |
| `damage.ts`      | ways to break a grown pattern: balls, scatter, bands, cuts  |
| `train.ts`       | the training loop                                           |
| `bench.ts`       | where the time goes, per rolled-out step and per graph size |
| `checkpoint.ts`  | saving, resuming, and the zero-padded warm start            |
| `render.ts`      | node states as an SVG, so a run is inspectable              |
| `rng.ts`         | seeded host-side sampling                                   |

## Checked against the original

`test/gnca.test.ts` does not trust this port. A PyTorch script
(`test/fixtures/dump_gnca_reference.py`) dumps a graph, a set of weights,
a three-step rollout, its loss and every gradient; the test rebuilds the
graph from the same node positions and compares. The edge list matches
exactly, and the state, loss and all five parameter gradients agree to
3e-5 relative — f32 reassociation noise, not a difference in the maths. It
runs in eager mode, through the lazy interpreter, natively, and through a
compiled training step.

To regenerate the fixture after changing the reference:

```sh
cd ~/code/graph-cellular-automata
.venv/bin/python ~/code/typenet/test/fixtures/dump_gnca_reference.py \
    > ~/code/typenet/test/fixtures/gnca-reference.json
```

## Two deviations from the reference

**Rollout length.** The reference samples any length in [48, 80] per
step. A compiled graph is traced once and so has a fixed depth, so this
draws from five evenly spaced lengths in that range, one compiled graph
each (`--buckets` to change how many). Variable enough to keep the rule
off a single horizon, few enough that the graphs are worth their memory.

**Alive mask.** The reference takes a max of alpha over neighbours and
compares it to the threshold. This counts live neighbours instead, which
is the same predicate — "this node or any neighbour is above threshold" —
through a scatter-add rather than a scatter-max, and needs one op fewer.

## Does it actually learn?

Same settings, same recipe, side by side against the PyTorch original
(Apple M5; heart target, 1024 nodes, k=8, batch 8). Different random
streams, so the two are not expected to agree step for step — only to
descend the same way, which they do:

| step | PyTorch loss | typenet loss |
| ---- | ------------ | ------------ |
| 200  | 0.1496       | 0.1608       |
| 400  | 0.1458       | 0.1450       |
| 600  | 0.1326       | 0.1257       |
| 800  | 0.1202       | 0.1183       |
| 1000 | 0.1159       | 0.1120       |
| 1200 | 0.1037       | 0.0981       |
| 1400 | 0.0982       | 0.0850       |

And the heal probe, which is the number that matters, improves the way it
should — healed overtaking grown, meaning the rule is regenerating rather
than just drawing: PyTorch at step 1000 grown 0.139 / healed 0.132,
typenet at step 1200 grown 0.100 / healed 0.076. The reference's
from-scratch 8000-step run finishes at grown 0.046 / healed 0.018.

## Speed

At those settings (8192 nodes, 76592 edges, forward + backward + Adam):
about **16 ms per rolled-out time step**, or **0.7 training steps/s**
against PyTorch on MPS at **2.7**, so ~4x slower.

The leading suspect is that candle's CPU elementwise kernels are
single-threaded, so the process sits at 100% of one core out of ten.
`TYPENET_PROFILE=1` shows it. The "History: measured numbers" section of
`../../PLAN.md` has the numbers and what closing it would take.
