import { ones, randn, tensor, zeros } from "../src/factories.ts"
import { Linear } from "../src/nn.ts"
import { DimAdd, DimMul } from "../src/shape.ts"
import type {
  Broadcast,
  CanBroadcast,
  Cat,
  ConvCheck,
  ConvOut,
  FlattenFrom,
  InferShape,
  MatMul,
  NormalizeDim,
  Permute,
  PoolOut,
  ReduceDim,
  ResizeDim,
  ResolveView,
  SliceShape,
  Squeeze,
  Stack,
  Transpose,
  Unsqueeze,
} from "../src/shape.ts"
import { fromFlat, Tensor } from "../src/tensor.ts"
import {
  BROADCAST_CASES,
  CAT_CASES,
  CONV_CASES,
  FLATTEN_FROM_CASES,
  MATMUL_CASES,
  PERMUTE_CASES,
  POOL_CASES,
  REDUCE_CASES,
  RESIZE_CASES,
  SLICE_CASES,
  VIEW_CASES,
} from "./shape-cases.ts"

type Equal<A, B> = (<T>() => T extends A ? 1 : 2) extends (
  <T>() => T extends B ? 1 : 2
) ? true
  : false
type Expect<T extends true> = T

type _b1 = Expect<Equal<Broadcast<[2, 3], [3]>, [2, 3]>>
type _b2 = Expect<
  Equal<Broadcast<[8, 1, 6, 1], [7, 1, 5]>, [8, 7, 6, 5]>
>
type _b3 = Expect<Equal<CanBroadcast<[2, 3], [4]>, false>>
type _b4 = Expect<
  Equal<Broadcast<[number, 3], [3]>, [number, 3]>
>

type _m1 = Expect<Equal<MatMul<[2, 3], [3, 4]>, [2, 4]>>
type _m2 = Expect<
  Equal<MatMul<[10, 2, 3], [3, 4]>, [10, 2, 4]>
>
type _m3 = Expect<Equal<MatMul<[3], [3]>, []>>
type _m4 = Expect<Equal<MatMul<[2, 3], [3]>, [2]>>
type _m5 = Expect<Equal<MatMul<[3], [3, 4]>, [4]>>
type _m6 = Expect<
  Equal<MatMul<[number, 784], [784, 128]>, [number, 128]>
>

type _v1 = Expect<
  Equal<ResolveView<[4, 6], [2, -1, 3]>, [2, 4, 3]>
>
type _t1 = Expect<
  Equal<Transpose<[2, 3, 4], 0, 2>, [4, 3, 2]>
>
type _t2 = Expect<
  Equal<Transpose<[2, 3, 4], -1, -2>, [2, 4, 3]>
>
type _p1 = Expect<
  Equal<Permute<[2, 3, 4], [2, 0, 1]>, [4, 2, 3]>
>
type _s1 = Expect<Equal<Squeeze<[1, 2, 1, 3]>, [2, 3]>>
type _u1 = Expect<Equal<Unsqueeze<[2, 3], 0>, [1, 2, 3]>>
type _u2 = Expect<Equal<Unsqueeze<[2, 3], -1>, [2, 3, 1]>>
type _r1 = Expect<Equal<ReduceDim<[2, 3, 4], 1>, [2, 4]>>
type _r2 = Expect<
  Equal<ReduceDim<[2, 3, 4], -1, true>, [2, 3, 1]>
>
type _k1 = Expect<Equal<Stack<[2, 3], 5, 0>, [5, 2, 3]>>
type _c1 = Expect<Equal<Cat<[2, 3], [4, 3], 0>, [6, 3]>>
type _n1 = Expect<Equal<NormalizeDim<[2, 3, 4], -1>, 2>>
type _i1 = Expect<
  Equal<InferShape<[[1, 2, 3], [4, 5, 6]]>, [2, 3]>
>

// The shared case table from shape-cases.ts, run through the type
// algebra; shape.test.ts runs the same rows through the value twins.
type BCase = typeof BROADCAST_CASES
type _tb0 = Expect<
  Equal<Broadcast<BCase[0]["a"], BCase[0]["b"]>, BCase[0]["out"]>
>
type _tb1 = Expect<
  Equal<Broadcast<BCase[1]["a"], BCase[1]["b"]>, BCase[1]["out"]>
>
type _tb2 = Expect<
  Equal<Broadcast<BCase[2]["a"], BCase[2]["b"]>, BCase[2]["out"]>
>
type _tb3 = Expect<
  Equal<Broadcast<BCase[3]["a"], BCase[3]["b"]>, BCase[3]["out"]>
>
type MCase = typeof MATMUL_CASES
type _tm0 = Expect<
  Equal<MatMul<MCase[0]["a"], MCase[0]["b"]>, MCase[0]["out"]>
>
type _tm1 = Expect<
  Equal<MatMul<MCase[1]["a"], MCase[1]["b"]>, MCase[1]["out"]>
>
type VCase = typeof VIEW_CASES
type _tv0 = Expect<
  Equal<ResolveView<VCase[0]["s"], VCase[0]["v"]>, VCase[0]["out"]>
>
type _tv1 = Expect<
  Equal<ResolveView<VCase[1]["s"], VCase[1]["v"]>, VCase[1]["out"]>
>
type CCase = typeof CAT_CASES
type _tc0 = Expect<
  Equal<Cat<CCase[0]["a"], CCase[0]["b"], CCase[0]["dim"]>, CCase[0]["out"]>
>
type _tc1 = Expect<
  Equal<Cat<CCase[1]["a"], CCase[1]["b"], CCase[1]["dim"]>, CCase[1]["out"]>
>
type RCase = typeof RESIZE_CASES
type _tr0 = Expect<
  Equal<
    ResizeDim<RCase[0]["s"], RCase[0]["dim"], RCase[0]["length"]>,
    RCase[0]["out"]
  >
>
type SCase = typeof SLICE_CASES
type _ts0 = Expect<
  Equal<SliceShape<SCase[0]["s"], SCase[0]["spec"]>, SCase[0]["out"]>
>
type _ts1 = Expect<
  Equal<SliceShape<SCase[1]["s"], SCase[1]["spec"]>, SCase[1]["out"]>
>
type _ts2 = Expect<
  Equal<SliceShape<SCase[2]["s"], SCase[2]["spec"]>, SCase[2]["out"]>
>
type PCase = typeof PERMUTE_CASES
type _tp0 = Expect<
  Equal<Permute<PCase[0]["s"], PCase[0]["order"]>, PCase[0]["out"]>
>
type DCase = typeof REDUCE_CASES
type _td0 = Expect<
  Equal<
    ReduceDim<DCase[0]["s"], DCase[0]["dim"], DCase[0]["keepdim"]>,
    DCase[0]["out"]
  >
>
type _td1 = Expect<
  Equal<
    ReduceDim<DCase[1]["s"], DCase[1]["dim"], DCase[1]["keepdim"]>,
    DCase[1]["out"]
  >
>

// Conv / pool spatial arithmetic (W0.16, D35). Same dual-table discipline:
// every row below is run through the value twins in `shape.test.ts`'s
// "conv shapes" block. The traps, the wildcards and the layer-shaped
// assertions live in `conv-shapes.test-d.ts`.
type CvCase = typeof CONV_CASES
type _tcv0 = Expect<
  Equal<ConvOut<CvCase[0]["h"], CvCase[0]["k"], CvCase[0]["s"], CvCase[0]["p"]>, CvCase[0]["out"]>
>
type _tcv1 = Expect<
  Equal<ConvOut<CvCase[1]["h"], CvCase[1]["k"], CvCase[1]["s"], CvCase[1]["p"]>, CvCase[1]["out"]>
>
type _tcv2 = Expect<
  Equal<ConvOut<CvCase[2]["h"], CvCase[2]["k"], CvCase[2]["s"], CvCase[2]["p"]>, CvCase[2]["out"]>
>
type _tcv3 = Expect<
  Equal<ConvOut<CvCase[3]["h"], CvCase[3]["k"], CvCase[3]["s"], CvCase[3]["p"]>, CvCase[3]["out"]>
>
type _tcv4 = Expect<
  Equal<ConvOut<CvCase[4]["h"], CvCase[4]["k"], CvCase[4]["s"], CvCase[4]["p"]>, CvCase[4]["out"]>
>
// every positive row is a kernel that fits, so every row's check is open —
// including row 4, where the kernel exactly fills the input and the output
// is 1 (an off-by-one in the span test would reject it)
type _tcc0 = Expect<
  Equal<ConvCheck<CvCase[0]["h"], CvCase[0]["k"], CvCase[0]["s"], CvCase[0]["p"]>, unknown>
>
type _tcc4 = Expect<
  Equal<ConvCheck<CvCase[4]["h"], CvCase[4]["k"], CvCase[4]["s"], CvCase[4]["p"]>, unknown>
>
type PoCase = typeof POOL_CASES
type _tpo0 = Expect<Equal<PoolOut<PoCase[0]["h"], PoCase[0]["k"], PoCase[0]["s"]>, PoCase[0]["out"]>>
type _tpo1 = Expect<Equal<PoolOut<PoCase[1]["h"], PoCase[1]["k"], PoCase[1]["s"]>, PoCase[1]["out"]>>
type _tpo2 = Expect<Equal<PoolOut<PoCase[2]["h"], PoCase[2]["k"], PoCase[2]["s"]>, PoCase[2]["out"]>>
type FfCase = typeof FLATTEN_FROM_CASES
type _tff0 = Expect<Equal<FlattenFrom<FfCase[0]["s"]>, FfCase[0]["out"]>>
type _tff1 = Expect<Equal<FlattenFrom<FfCase[1]["s"]>, FfCase[1]["out"]>>
type _tff2 = Expect<Equal<FlattenFrom<FfCase[2]["s"]>, FfCase[2]["out"]>>
type _tff3 = Expect<Equal<FlattenFrom<FfCase[3]["s"]>, FfCase[3]["out"]>>

// D22, and conv is its first dependent caller: `FlattenFrom` folds with the
// reseeded `Prod`, so inside a generic body it is the SAME type as the
// `DimMul` chain a caller writes by hand — mutually assignable, not merely
// equal-looking. Seed the fold with `1` instead and both assignments below
// stop compiling, which is the whole reason the seed is `S[0]`.
type _ByHand<B extends number, C extends number, H extends number, W extends number> = [B, DimMul<DimMul<C, H>, W>]
function _flattenFromIsTheHandWrittenProduct<
  B extends number,
  C extends number,
  H extends number,
  W extends number,
>(derived: FlattenFrom<[B, C, H, W]>, byHand: _ByHand<B, C, H, W>) {
  // Assignability in both directions is the claim, and it is the strongest
  // one available here: `Equal` is a conditional, and under generic dims
  // BOTH of these types are still residual `DimMul`s, so `Equal<...>` itself
  // defers to `boolean` and asserting on it would prove nothing. The two
  // annotations below are checked by the compiler now.
  const a: _ByHand<B, C, H, W> = derived
  const b: FlattenFrom<[B, C, H, W]> = byHand
  return { a, b }
}

function _tensors() {
  const a = tensor([
    [1, 2, 3],
    [4, 5, 6],
  ])
  type _1 = Expect<Equal<typeof a.shape, [2, 3]>>

  const mm = a.matmul(zeros([3, 7]))
  type _2 = Expect<Equal<typeof mm.shape, [2, 7]>>

  // @ts-expect-error inner dims do not match
  a.matmul(zeros([4, 7]))

  const v = a.view([3, 2])
  type _3 = Expect<Equal<typeof v.shape, [3, 2]>>

  const vi = a.view([-1, 2])
  type _4 = Expect<Equal<typeof vi.shape, [3, 2]>>

  // @ts-expect-error 4 does not divide 6 elements
  a.view([4, -1])

  const sq = ones([1, 2, 1, 5]).squeeze()
  type _5 = Expect<Equal<typeof sq.shape, [2, 5]>>

  const us = a.unsqueeze(1)
  type _6 = Expect<Equal<typeof us.shape, [2, 1, 3]>>

  const sum = a.sum(-1)
  type _7 = Expect<Equal<typeof sum.shape, [2]>>

  const sumK = a.sum(0, true)
  type _8 = Expect<Equal<typeof sumK.shape, [1, 3]>>

  const scalarLoss = a.mean()
  type _9 = Expect<Equal<typeof scalarLoss.shape, []>>

  // @ts-expect-error dim 5 out of range
  a.sum(5)

  const bcast = a.add(tensor([1, 2, 3]))
  type _10 = Expect<Equal<typeof bcast.shape, [2, 3]>>

  // @ts-expect-error [2,3] and [4] do not broadcast
  a.add(tensor([1, 2, 3, 4]))

  const tr = randn([5, 6, 7]).transpose(0, 2)
  type _11 = Expect<Equal<typeof tr.shape, [7, 6, 5]>>

  const pm = randn([5, 6, 7]).permute(1, 2, 0)
  type _12 = Expect<Equal<typeof pm.shape, [6, 7, 5]>>

  // @ts-expect-error not a permutation
  randn([5, 6, 7]).permute(0, 0, 1)

  const st = Tensor.stack([a, a], 0)
  type _13 = Expect<Equal<typeof st.shape, [2, 2, 3]>>

  const ct = Tensor.cat(a, a, 1)
  type _14 = Expect<Equal<typeof ct.shape, [2, 6]>>

  const mt = a.T
  type _15 = Expect<Equal<typeof mt.shape, [3, 2]>>

  const rg = a.requiresGrad()
  type _16 = Expect<
    Equal<
      typeof rg,
      Tensor<
        [2, 3]
      >
    >
  >

  const f64 = a.to("float64")
  type _17 = Expect<
    Equal<
      typeof f64,
      Tensor<
        [2, 3]
      >
    >
  >

  const sliced = randn([4, 5, 6]).slice([2, [1, 4], null])
  type _18 = Expect<Equal<typeof sliced.shape, [2, 3, 6]>>

  const bcastTo = tensor([1, 2, 3]).broadcastTo([2, 3])
  type _19 = Expect<Equal<typeof bcastTo.shape, [2, 3]>>

  const bcastRank = a.broadcastTo([2, 2, 3])
  type _20 = Expect<Equal<typeof bcastRank.shape, [2, 2, 3]>>

  // @ts-expect-error [2, 3] does not broadcast down to [3]
  a.broadcastTo([3])

  // @ts-expect-error [2, 3] and [4] cannot broadcast
  a.broadcastTo([4, 3])

  // slice needs one entry per axis
  // @ts-expect-error rank 2 needs two entries
  a.slice([2])
}

function _nn() {
  const layer = new Linear(784, 128)
  const batch = randn([32, 784])
  const out = layer.forward(batch)
  type _1 = Expect<Equal<typeof out.shape, [32, 128]>>

  const b: number = 32
  const dyn = randn([b, 784] as [number, 784])
  const out2 = layer.forward(dyn)
  type _2 = Expect<Equal<typeof out2.shape, [number, 128]>>

  // @ts-expect-error wrong input width
  layer.forward(randn([32, 100]))

  // Regression for issue #30: `Linear.forward` must not return `as any`.
  // The output shape must propagate as a concrete `MatMul<S, [In, Out]>`
  // literal so downstream code is type-checked against it. A bias-less
  // layer exercises the pure-matmul branch.
  const nobias = new Linear(4, 8, { bias: false })
  const y = nobias.forward(randn([3, 4]))
  type _3 = Expect<Equal<typeof y.shape, [3, 8]>>

  // ...and a biased layer keeps the same guarantee, including the bias-add
  // path — the branch that previously forced the `as any` erasure.
  const biased = new Linear(8, 16)
  const z = biased.forward(y)
  type _4 = Expect<Equal<typeof z.shape, [3, 16]>>
}

import { ReLU, Sequential, sequential, Softmax } from "../src/nn.ts"

function _sequential() {
  const net = sequential(
    new Linear(2, 16),
    new ReLU(),
    new Linear(16, 16),
    new ReLU(),
    new Linear(16, 3),
  )
  type _1 = Expect<
    Equal<
      typeof net,
      Sequential<
        readonly [
          Linear<2, 16>,
          ReLU,
          Linear<16, 16>,
          ReLU,
          Linear<16, 3>,
        ]
      >
    >
  >

  const out = net.forward(randn([32, 2]))
  type _2 = Expect<Equal<typeof out.shape, [32, 3]>>

  // rank-generic: a chain of Linears rewrites the last axis only
  const deep = net.forward(randn([4, 32, 2]))
  type _3 = Expect<Equal<typeof deep.shape, [4, 32, 3]>>

  // @ts-expect-error 16 -> 17 mismatch between layers
  sequential(new Linear(2, 16), new Linear(17, 3))

  // @ts-expect-error activation cannot bridge a 16 -> 17 mismatch
  sequential(new Linear(2, 16), new ReLU(), new Linear(17, 3))

  // @ts-expect-error wrong input width
  net.forward(randn([32, 5]))
}

function _catN(a: Tensor<[2, 3]>, b: Tensor<[2, 5]>) {
  const wide = Tensor.cat([a, b, a], 1)
  type _1 = Expect<Equal<typeof wide.shape, [2, 11]>>

  const pair = Tensor.cat(a, a)
  type _2 = Expect<Equal<typeof pair.shape, [4, 3]>>

  // @ts-expect-error shapes differ outside dim 0
  Tensor.cat([a, b], 0)

  const out = new Softmax(1).forward(a)
  type _3 = Expect<Equal<typeof out.shape, [2, 3]>>

  // @ts-expect-error dim 5 out of range for rank-2 input
  new Softmax(5).forward(a)

  return { wide, pair, out }
}

function _negative() {
  const a = tensor([
    [1, 2, 3],
    [4, 5, 6],
  ])
  const b = zeros([3, 4])

  // @ts-expect-error matmul result is [2,4], not [2,5]
  const badResult: Tensor<[2, 5]> = a.matmul(b)

  // @ts-expect-error .T of [2,3] is [3,2], not [2,3]
  const badT: Tensor<[2, 3]> = a.T

  const q = randn([2, 4, 16, 8])

  // @ts-expect-error batched matmul: inner dims 8 and 9 disagree
  q.matmul(randn([2, 4, 9, 16]))

  // @ts-expect-error batch dims [2,4] cannot broadcast with [3,5]
  q.matmul(randn([3, 5, 8, 16]))

  // @ts-expect-error vec@vec needs matching length
  randn([8]).matmul(randn([9]))

  // @ts-expect-error trailing dim 5 does not broadcast against 4
  randn([2, 3, 4]).add(randn([5]))

  return { badResult, badT }
}

function _genericDims<
  N extends number,
>(
  h: Tensor<[N, 24]>,
  adj: Tensor<[N, N]>,
): Tensor<[N, 16]> {
  const w = zeros([24, 8])
  const wh = h.matmul(w)
  type _1 = Expect<Equal<typeof wh.shape, [N, 8]>>

  const col = wh.matmul(zeros([8, 1]))
  type _2 = Expect<Equal<typeof col.shape, [N, 1]>>

  const row = col.T
  type _3 = Expect<Equal<typeof row.shape, [1, N]>>

  const masked = adj.mul(adj).add(adj)
  type _4 = Expect<Equal<typeof masked.shape, [N, N]>>

  const alpha = masked.softmax(1)
  const agg = alpha.matmul(wh)
  type _5 = Expect<Equal<typeof agg.shape, [N, 8]>>

  const two = Tensor.cat(agg, agg, 1)
  type _6 = Expect<Equal<typeof two.shape, [N, 16]>>
  return two
}

// The Broadcast suffix rule: broadcasting a same-generic-dim suffix
// (the bias of a generic layer) is *syntactically* the identity, so the
// result keeps unifying with generic parameters downstream with no
// re-anchoring annotation. This is what lets GraphNCA.forward infer its
// gate tensor.
function _genericBias<
  E extends number,
  C extends number,
>(edges: Tensor<[E, C]>, bias: Tensor<[C]>) {
  const out = edges.add(bias)
  type _1 = Expect<Equal<typeof out.shape, [E, C]>>
  const chained = out.sigmoid().mul(edges)
  type _2 = Expect<Equal<typeof chained.shape, [E, C]>>
  const scalar = edges.add(Tensor.scalar(1))
  type _3 = Expect<Equal<typeof scalar.shape, [E, C]>>
  return chained
}

// fromFlat reads tuple types off shape literals built from typed
// runtime values, and DimAdd/DimMul carry their arithmetic as types.
function _fromFlat<E extends number, N extends number>(count: E, nodes: N) {
  const edges = fromFlat(new Float32Array(count), [count])
  type _1 = Expect<Equal<typeof edges.shape, [E]>>
  const col = fromFlat(new Float32Array(nodes), [nodes, 1])
  type _2 = Expect<Equal<typeof col.shape, [N, 1]>>
  const lit = fromFlat([1, 2, 3, 4, 5, 6], [2, 3])
  type _3 = Expect<Equal<typeof lit.shape, [2, 3]>>
  const width = DimAdd(DimMul(3, 16), 1)
  type _4 = Expect<Equal<typeof width, 49>>
  return [edges, col, lit, width] as const
}

function _genericOuter<
  N extends number,
>(col: Tensor<[N, 1]>, row: Tensor<[1, N]>) {
  const sum = col.add(row)
  type _1 = Expect<Equal<typeof sum.shape, [N, N]>>
  const prod = col.mul(row)
  type _2 = Expect<Equal<typeof prod.shape, [N, N]>>
  const flipped = row.add(col)
  type _3 = Expect<Equal<typeof flipped.shape, [N, N]>>
  return sum
}

function _genericBatched<
  B extends number,
  N extends number,
>(q: Tensor<[B, N, 8]>, w: Tensor<[8, 16]>) {
  const out = q.matmul(w)
  type _1 = Expect<Equal<typeof out.shape, [B, N, 16]>>
  const tr = q.transpose(1, 2)
  type _2 = Expect<Equal<typeof tr.shape, [B, 8, N]>>
  const s = q.sum(0)
  type _3 = Expect<Equal<typeof s.shape, [N, 8]>>
  const c = Tensor.cat(q, q, 1)
  const cc = Tensor.cat(q, q, 2)
  type _4 = Expect<Equal<typeof cc.shape, [B, N, 16]>>
  return { out, c }
}

function _genericNegative<
  N extends number,
>(h: Tensor<[N, 24]>, q: Tensor<[2, N, 16, 8]>) {
  // @ts-expect-error inner dims 24 and 7 disagree
  h.matmul(zeros([7, 8]))

  // @ts-expect-error [N, 24] and [5] do not broadcast
  h.add(zeros([5]))

  // @ts-expect-error batch dims 2 and 3 do not broadcast
  q.matmul(randn([3, N, 8, 16] as [3, N, 8, 16]))

  // @ts-expect-error cat: [N, 24] and [N, 8] differ outside dim 0
  Tensor.cat(h, zeros([8, 8]), 0)

  return h
}

import { crossEntropy, mseLoss } from "../src/nn.ts"

// NoInfer pins each repeated inference site to its first occurrence, so a
// mismatched later argument is checked instead of re-inferring.

function _noInfer(
  pred: Tensor<[2, 3]>,
  logits: Tensor<[4, 3]>,
  badTargets: Tensor<[5]>,
) {
  const ok = mseLoss(
    pred,
    tensor([
      [1, 2, 3],
      [4, 5, 6],
    ]),
  )
  type _1 = Expect<Equal<typeof ok.shape, []>>

  // @ts-expect-error target shape must match prediction
  mseLoss(pred, zeros([2, 4]))

  // @ts-expect-error one target per row: batch is 4, not 5
  crossEntropy(logits, badTargets)

  return ok
}

function _gatherScatter<
  N extends number,
>(x: Tensor<[N, 16]>, nodes: Tensor<[1024, 16]>) {
  const src = zeros([4096])
  const gathered = nodes.indexSelect(src)
  type _1 = Expect<Equal<typeof gathered.shape, [4096, 16]>>

  const aggregated = gathered.scatterAdd(src, 1024)
  type _2 = Expect<
    Equal<typeof aggregated.shape, [1024, 16]>
  >

  const channels = nodes.indexSelect(zeros([3]), 1)
  type _3 = Expect<Equal<typeof channels.shape, [1024, 3]>>

  const generic = x.indexSelect(src)
  type _4 = Expect<Equal<typeof generic.shape, [4096, 16]>>

  // @ts-expect-error dim 2 does not exist on a rank-2 tensor
  nodes.indexSelect(src, 2)

  // @ts-expect-error one index per source row: 4096 rows, not 8
  gathered.scatterAdd(zeros([8]), 1024)

  // @ts-expect-error dim overload: same length rule, now on the dim form
  gathered.scatterAdd(zeros([8]), 1024, 0)

  const alongDim = channels.scatterAdd(zeros([3]), 7, 1)
  type _5 = Expect<Equal<typeof alongDim.shape, [1024, 7]>>

  return { gathered, aggregated }
}

function _getOneHot(labels: Tensor<[6]>, grid: Tensor<[2, 3]>) {
  const hot = labels.oneHot(10)
  type _1 = Expect<Equal<typeof hot.shape, [6, 10]>>

  const v: number = grid.get(0, 1)

  // @ts-expect-error get() takes one index per dim: rank 2 needs two
  grid.get(0)

  // @ts-expect-error oneHot() requires a rank-1 tensor
  grid.oneHot(10)

  return { hot, v }
}

function _compare(a: Tensor<[2, 3]>, b: Tensor<[3]>) {
  const mask = a.gt(b)
  type _1 = Expect<Equal<typeof mask.shape, [2, 3]>>

  const limited = a.clamp(-1, 1)
  type _2 = Expect<Equal<typeof limited.shape, [2, 3]>>

  const outer = zeros([2, 1]).maximum(zeros([1, 3]))
  type _3 = Expect<Equal<typeof outer.shape, [2, 3]>>

  // @ts-expect-error [2, 3] and [4] do not broadcast
  a.maximum(zeros([4]))

  return { mask, limited, outer }
}

type PlusOne<C extends number> = DimAdd<C, 1>
type Times3Plus1<C extends number> = DimAdd<DimMul<3, C>, 1>
type DimMsg<C extends number> = `dim is ${DimAdd<3, C>}`

type _smartConstructors = {
  addLiterals: Expect<Equal<DimAdd<3, 4>, 7>>
  mulLiterals: Expect<Equal<DimMul<6, 7>, 42>>
  addWildcard: Expect<Equal<DimAdd<number, 3>, number>>
  mulWildcard: Expect<Equal<DimMul<number, 3>, number>>
  mulByZero: Expect<Equal<DimMul<5, 0>, 0>>
  // a deferred constructor re-fires at instantiation
  plusOneInstantiates: Expect<Equal<PlusOne<3>, 4>>
  times3Plus1: Expect<Equal<Times3Plus1<32>, 97>>
  // deferred dims still interpolate into error messages
  dimMsg: Expect<Equal<DimMsg<5>, "dim is 8">>
}

// identity rules reduce EAGERLY, inside a generic body, no instantiation
function _scGeneric<N extends number, C extends number>() {
  const a: C = null as any as DimAdd<C, 0>
  const b: C = null as any as DimMul<C, 1>
  const c: 0 = null as any as DimMul<C, 0>
  const d: [N, C] = null as any as Broadcast<[N, C], [N, C]>
  const e: [C] = null as any as Broadcast<[C], [1]>
  return [a, b, c, d, e]
}

// ---------------------------------------------------------------------------
// W0.13 foundations: the tuple surgery, the division algebra, and the
// checks that ride on them. The probes these came from are checked in as
// test/{dimdiv,lastdim,flatten,gpt-adopted}.test-d.ts; what is here is the
// case-table half (run again through the value twins by shape.test.ts) and
// the negative cases.
// ---------------------------------------------------------------------------

import { DimDiv } from "../src/shape.ts"
import type {
  BatchPrefix,
  DimDivCheck,
  Drop,
  FlattenCheck,
  FlattenShape,
  IndexCheck,
  IndexTensor,
  Init,
  Last,
  LastDimCheck,
  Prod,
  ReduceDims,
  Shape,
  Slice,
  SliceCheck,
  Take,
  UnflattenCheck,
  UnflattenShape,
} from "../src/shape.ts"
import { DIM_DIV_CASES, FLATTEN_CASES, UNFLATTEN_CASES } from "./shape-cases.ts"

type _init1 = Expect<Equal<Init<[2, 3, 4]>, [2, 3]>>
type _init2 = Expect<Equal<BatchPrefix<[8, 256, 65]>, [8, 256]>>
type _last1 = Expect<Equal<Last<[2, 3, 4]>, 4>>
type _take1 = Expect<Equal<Take<[2, 3, 4], 2>, [2, 3]>>
type _take2 = Expect<Equal<Take<[2, 3, 4], 0>, []>>
type _drop1 = Expect<Equal<Drop<[2, 3, 4], 1>, [3, 4]>>
type _drop2 = Expect<Equal<Drop<[2, 3, 4], 3>, []>>

type _prod1 = Expect<Equal<Prod<[2, 3, 4]>, 24>>
type _prod2 = Expect<Equal<Prod<[]>, 1>>
type _prod3 = Expect<Equal<Prod<[7]>, 7>>
type _prod4 = Expect<Equal<Prod<number[]>, number>>

// D22: the fold is seeded with S[0], so the product of a generic shape is
// the same type the caller writes by hand — in both directions.
function _prodReseed<B extends number, T extends number>() {
  const fromAlgebra: DimMul<B, T> = null as any as Prod<[B, T]>
  const byHand: Prod<[B, T]> = null as any as DimMul<B, T>
  return [fromAlgebra, byHand]
}

type _dd1 = Expect<Equal<DimDiv<384, 6>, 64>>
type _dd2 = Expect<Equal<DimDiv<number, 6>, number>>
// the quotient truncates; divisibility is DimDivCheck's job
type _dd3 = Expect<Equal<DimDiv<7, 2>, 3>>

type DDCase = typeof DIM_DIV_CASES
type _tdd0 = Expect<Equal<DimDiv<DDCase[0]["a"], DDCase[0]["b"]>, DDCase[0]["out"]>>
type _tdd1 = Expect<Equal<DimDiv<DDCase[1]["a"], DDCase[1]["b"]>, DDCase[1]["out"]>>
type _tdd2 = Expect<Equal<DimDiv<DDCase[2]["a"], DDCase[2]["b"]>, DDCase[2]["out"]>>
type _tdd3 = Expect<Equal<DimDiv<DDCase[3]["a"], DDCase[3]["b"]>, DDCase[3]["out"]>>

type FCase = typeof FLATTEN_CASES
type _tfl0 = Expect<Equal<FlattenShape<FCase[0]["s"], FCase[0]["from"], FCase[0]["to"]>, FCase[0]["out"]>>
type _tfl1 = Expect<Equal<FlattenShape<FCase[1]["s"], FCase[1]["from"], FCase[1]["to"]>, FCase[1]["out"]>>
type _tfl2 = Expect<Equal<FlattenShape<FCase[2]["s"], FCase[2]["from"], FCase[2]["to"]>, FCase[2]["out"]>>
type _tfl3 = Expect<Equal<FlattenShape<FCase[3]["s"], FCase[3]["from"], FCase[3]["to"]>, FCase[3]["out"]>>

type UCase = typeof UNFLATTEN_CASES
type _tuf0 = Expect<Equal<UnflattenShape<UCase[0]["s"], UCase[0]["dim"], UCase[0]["sizes"]>, UCase[0]["out"]>>
type _tuf1 = Expect<Equal<UnflattenShape<UCase[1]["s"], UCase[1]["dim"], UCase[1]["sizes"]>, UCase[1]["out"]>>
type _tuf2 = Expect<Equal<UnflattenShape<UCase[2]["s"], UCase[2]["dim"], UCase[2]["sizes"]>, UCase[2]["out"]>>

type _rd1 = Expect<Equal<ReduceDims<[2, 3, 4], [0, 2]>, [3]>>
type _rd2 = Expect<Equal<ReduceDims<[2, 3, 4], [0, 2], true>, [1, 3, 1]>>
type _rd3 = Expect<Equal<ReduceDims<[2, 3, 4], [-1]>, [2, 3]>>
type _rd4 = Expect<Equal<ReduceDims<[2, 3, 4], []>, [2, 3, 4]>>
type _rd5 = Expect<Equal<ReduceDims<number[], [0]>, number[]>>

// The signature shapes these checks are meant to be worn in. `flatten`
// and `unflatten` become Tensor methods in W1.8; here they stand in as
// free functions so the checks are exercised against the real exports.
declare class LayerNormLike<D extends number> {
  forward<S extends Shape>(x: Tensor<S> & LastDimCheck<S, D>): Tensor<S>
}
declare class AttentionLike<D extends number, H extends number> {
  constructor(d: D, h: H & DimDivCheck<D, H>)
}
declare function flattenT<S extends Shape, const F extends number, const T extends number>(
  t: Tensor<S>,
  from: F & FlattenCheck<S, F, T>,
  to: T,
): Tensor<FlattenShape<S, F, T>>
declare function unflattenT<S extends Shape, const D extends number, const Sizes extends number[]>(
  t: Tensor<S>,
  dim: D & UnflattenCheck<S, D, Sizes>,
  sizes: Sizes,
): Tensor<UnflattenShape<S, D, Sizes>>
declare function sliceT<S extends Shape, const Spec extends readonly Slice[]>(
  t: Tensor<S>,
  spec: Spec & SliceCheck<S, Spec>,
): Tensor<SliceShape<S, Spec>>
declare function embedT<S extends Shape>(ids: IndexTensor<S>): Tensor<[...S, 8]>
declare function indexOnly<T>(t: T & IndexCheck<T>): void

function _w013Negative(
  x: Tensor<[2, 3, 4]>,
  grid: Tensor<[4, 5]>,
  feats: Tensor<[4, 8, 384]>,
  ids: IndexTensor<[4, 5]>,
) {
  // @ts-expect-error 384 is not divisible by 5 heads
  new AttentionLike(384, 5)

  // @ts-expect-error the last axis is 384, not 128
  new LayerNormLike<128>().forward(feats)

  // @ts-expect-error flatten range: dim 3 is out of range for a rank-3 shape
  flattenT(x, 1, 3)

  // @ts-expect-error unflatten dim 5 is out of range for a rank-3 shape
  unflattenT(x, 5, [2, 2])

  // @ts-expect-error slice window [1, 9] runs past an axis of extent 5
  sliceT(grid, [2, [1, 9]])

  // @ts-expect-error a float tensor is not an index tensor
  embedT(grid)

  // @ts-expect-error ...and IndexCheck says so on any tensor-shaped argument
  indexOnly(grid)

  // the same calls, made correctly
  const flat = flattenT(x, 1, 2)
  type _1 = Expect<Equal<typeof flat.shape, [2, 12]>>
  const split = unflattenT(x, 2, [2, 2])
  type _2 = Expect<Equal<typeof split.shape, [2, 3, 2, 2]>>
  const win = sliceT(grid, [2, [1, 3]])
  type _3 = Expect<Equal<typeof win.shape, [2, 2]>>
  const emb = embedT(ids)
  type _4 = Expect<Equal<typeof emb.shape, [4, 5, 8]>>
  indexOnly(ids)
  const normed = new LayerNormLike<384>().forward(feats)
  type _5 = Expect<Equal<typeof normed.shape, [4, 8, 384]>>
  return { emb, flat, normed, split, win }
}

// Law 1, once per new check: a naked generic shape decides nothing, so
// every one of them has to let the call through.
function _w013FailOpen<S extends Shape>(x: Tensor<S>, n: LayerNormLike<16>, len: number) {
  const a = n.forward(x)
  const b = flattenT(x, 0, 1)
  const c = unflattenT(x, 0, [1, len])
  const d = sliceT(x, [null, null])
  return { a, b, c, d }
}

// ...and generic dims with a known arity behave the same way.
function _w013GenericDims<B extends number, T extends number, H extends number, Dh extends number>(
  x: Tensor<[B, T, 16]>,
  ctx: Tensor<[B, T, H, Dh]>,
  n: LayerNormLike<16>,
  h: H,
  dh: Dh,
) {
  const a = n.forward(x)
  type _1 = Expect<Equal<typeof a.shape, [B, T, 16]>>
  const b = flattenT(ctx, 2, 3)
  type _2 = Expect<Equal<typeof b.shape, [B, T, DimMul<H, Dh>]>>
  const c = unflattenT(x, 2, [h, dh])
  type _3 = Expect<Equal<typeof c.shape, [B, T, H, Dh]>>
  const d = sliceT(x, [null, [0, 4], null])
  return { a, b, c, d }
}
