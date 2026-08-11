# Graph cellular automata

A cellular automaton on a graph: one shared update rule, applied to every
node, that grows a pattern from a single seed and regrows it after damage.
Ported from [graph-cellular-automata](../../../graph-cellular-automata),
which is the PyTorch original and the source of truth for the recipe.

```sh
pnpm gnca                            # train with the reference defaults
pnpm gnca --steps 200                # a quick run
pnpm gnca --target star --nodes 512  # a different pattern, smaller graph
pnpm vite-node examples/gnca/bench.ts
```

## The rule

Each node sees `[own state, mean of neighbour states, mean of gated
(neighbour − own) differences, log(1 + degree)]`, and a two-layer MLP
turns that into a residual increment applied to a random half of the nodes
each step.

Three details carry most of the weight, and all three are the reference's:

- The **degree scalar** restores the neighbour *count* that mean
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
*hold* a pattern rather than draw one. Each batch is ranked worst-first
and damage lands on the best-formed samples, because healing a grown
pattern is the behaviour wanted and gradient spent on an already-broken
state is spent twice over. Every eighth step trains one sample from the
bare seed, to keep the long horizon in practice.

The number to watch is the **heal probe**, not the loss: grow for 80
steps, punch a fixed hole, heal for 160, and score. Training loss can fall
while regeneration stays broken.

## Files

| file         | contents                                                    |
| ------------ | ----------------------------------------------------------- |
| `model.ts`   | the update rule, the alive mask, the seed state             |
| `graphs.ts`  | k-NN, random geometric and Watts-Strogatz graph builders     |
| `targets.ts` | the eight patterns, 2-d and 3-d, sampled at node positions  |
| `damage.ts`  | ways to break a grown pattern: balls, scatter, bands, cuts   |
| `train.ts`   | the training loop                                           |
| `bench.ts`   | where the time goes, per rolled-out step and per graph size  |
| `rng.ts`     | seeded host-side sampling                                   |

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

## Speed

On an Apple M5, at the reference settings (1024 nodes, batch 8, so 8192
nodes and 76592 edges, forward + backward + Adam): about **16 ms per
rolled-out time step**, or ~1 training step/s at the reference rollout
lengths. PyTorch on MPS does 2.9 steps/s on the same machine.

The gap is one thing: candle's CPU elementwise kernels are
single-threaded, so the process sits at 100% of one core out of ten.
`TYPENET_PROFILE=1` shows it. The Phase E section of `../../LAZY.md` has
the numbers and what closing it would take.
