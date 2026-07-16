import type {
  Broadcast,
  MatMul,
  ResolveView,
  Transpose,
  Permute,
  Squeeze,
  Unsqueeze,
  ReduceDim,
  Stack,
  Cat,
  CanBroadcast,
  NormalizeDim,
  InferShape
} from "../src/shape.ts"
import {
  tensor,
  zeros,
  ones,
  randn,
  Tensor
} from "../src/tensor.ts"
import { Linear } from "../src/nn.ts"

type Equal<A, B> =
  (<T>() => T extends A ? 1 : 2) extends (
    <T>() => T extends B ? 1 : 2
  ) ?
    true
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
    [4, 5, 6]
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
          device: "cpu"
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
          device: "cpu"
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
    new Linear(16, 3)
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

function _genericDims<
  N extends number,
  P extends TensorParams
>(
  h: Tensor<[N, 24], P>,
  adj: Tensor<[N, N], any>
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
