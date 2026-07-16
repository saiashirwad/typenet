"use tsover";

import { randn } from "../src/tensor";

const a = randn([2, 3]);
const w = randn([3, 4]);

const h = a.matmul(w);
const s = h + randn([4]);
const l = ((s - 1) ** 2).mean();

const m = randn([2, 1]) + randn([1, 3]);

a.matmul(randn([5, 4]));
