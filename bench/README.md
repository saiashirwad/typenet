# bench/ — smoke vs full

Every script here defaults to a **smoke run**: tiny sizes, 1 warm-up, 2 timed
samples per case × mode — just enough to prove the script still runs end to
end, not to produce a number worth trusting. Run one directly with
`pnpm vite-node bench/<script>.ts`, or the whole suite with `pnpm bench`.

Pass `--full` to opt into real sizes and the statistical floors (≥ 3
warm-up, ≥ 10 samples): `pnpm vite-node bench/<script>.ts -- --full`.

Every JSONL line records `smoke: true | false`, and the two never share a
file: smoke rows go to `bench/results/smoke/<script>.jsonl`, full rows to
`bench/results/<script>.jsonl`. Never compare a smoke number against a full
one, or against another smoke number from a differently-sized case — smoke
sizes exist in `bench/lib/sizes.ts` (the `*_SMOKE` exports) purely to be
small and fast, not representative.

A script whose smoke config can't be made to fit well under 30s any other
way (`bench/macro-nanogpt.ts`'s fixed 50-step loss curve, for one) prints a
line saying so and skips that one piece rather than running it — `--full`
still runs it in full.
