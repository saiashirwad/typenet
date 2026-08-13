import { Linear } from "../src/nn.ts"
import type {
  Broadcast,
  CanBroadcast,
  Cat,
  DimAdd,
  DimMul,
  InferShape,
  MatMul,
  NormalizeDim,
  Permute,
  ReduceDim,
  ResolveView,
  Squeeze,
  Stack,
  Transpose,
  Unsqueeze,
} from "../src/shape.ts"
import { ones, randn, Tensor, tensor, zeros } from "../src/tensor.ts"

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

  const rg = a.requires_grad()
  type _16 = Expect<
    Equal<
      typeof rg,
      Tensor<
        [2, 3],
        {
          requires_grad: true
          dtype: "float32"
        }
      >
    >
  >

  const f64 = a.to("float64")
  type _17 = Expect<
    Equal<
      typeof f64,
      Tensor<
        [2, 3],
        {
          requires_grad: false
          dtype: "float64"
        }
      >
    >
  >
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
}

import { ReLU, Sequential, sequential } from "../src/nn.ts"
import type { TensorParams } from "../src/tensor.ts"

function _sequential() {
  const net = sequential(
    new Linear(2, 16),
    new ReLU(),
    new Linear(16, 16),
    new ReLU(),
    new Linear(16, 3),
  )
  type _1 = Expect<Equal<typeof net, Sequential<2, 3>>>

  const out = net.forward(randn([32, 2]))
  type _2 = Expect<Equal<typeof out.shape, [32, 3]>>

  // @ts-expect-error 16 -> 17 mismatch between layers
  sequential(new Linear(2, 16), new Linear(17, 3))

  // prettier-ignore
  // @ts-expect-error activation cannot bridge a 16 -> 17 mismatch
  sequential(new Linear(2, 16), new ReLU(), new Linear(17, 3))

  // @ts-expect-error wrong input width
  net.forward(randn([32, 5]))
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
  P extends TensorParams,
>(
  h: Tensor<[N, 24], P>,
  adj: Tensor<[N, N], any>,
): Tensor<[N, 16], P> {
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

function _genericOuter<
  N extends number,
  P extends TensorParams,
>(col: Tensor<[N, 1], P>, row: Tensor<[1, N], any>) {
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
  P extends TensorParams,
>(q: Tensor<[B, N, 8], P>, w: Tensor<[8, 16], any>) {
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
  P extends TensorParams,
>(h: Tensor<[N, 24], P>, q: Tensor<[2, N, 16, 8], P>) {
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

function _noInfer<P extends TensorParams>(
  pred: Tensor<[2, 3], P>,
  logits: Tensor<[4, 3], P>,
  badTargets: Tensor<[5], any>,
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
  P extends TensorParams,
>(x: Tensor<[N, 16], P>, nodes: Tensor<[1024, 16], P>) {
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

  return { gathered, aggregated }
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

import { aliveMask, GraphNCA, type GraphTensors, type Percept } from "../examples/gnca/model.ts"

function _gnca(
  nodes: Tensor<[1024, 16]>,
  batched: Tensor<[8192, 16]>,
  graph: GraphTensors<1024, 9574>,
  batchedGraph: GraphTensors<8192, 76592>,
  wrongChannels: Tensor<[1024, 8]>,
  wrongNodes: Tensor<[512, 16]>,
) {
  const rule = new GraphNCA(16, 128)

  // The derived widths are literals, not `number`.
  type _1 = Expect<Equal<Percept<16>, 49>>
  type _2 = Expect<
    Equal<typeof rule.inner.weight.shape, [49, 128]>
  >
  type _3 = Expect<
    Equal<typeof rule.gate.weight.shape, [48, 16]>
  >
  type _4 = Expect<
    Equal<typeof rule.outer.weight.shape, [128, 16]>
  >

  const next = rule.forward(nodes, graph)
  type _5 = Expect<Equal<typeof next.shape, [1024, 16]>>
  const nextBatch = rule.forward(batched, batchedGraph)
  type _6 = Expect<
    Equal<typeof nextBatch.shape, [8192, 16]>
  >

  const alive = aliveMask(nodes, graph)
  type _7 = Expect<Equal<typeof alive.shape, [1024, 1]>>
  const masked = nodes.mul(alive)
  type _8 = Expect<Equal<typeof masked.shape, [1024, 16]>>

  // @ts-expect-error a rule built for 16 channels cannot step an 8-channel state
  rule.forward(wrongChannels, graph)

  // @ts-expect-error the state and the graph must agree on the node count
  rule.forward(wrongNodes, graph)

  // @ts-expect-error and a batched state needs the batched edge list
  rule.forward(batched, graph)

  return { next, nextBatch, alive }
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
