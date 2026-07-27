# Handoff — resume typenet roadmap in a fresh session

Paste the prompt below into a new kimi session. Everything durable is already
in the repo: `PLAN.md` (roadmap, gates, settled decisions) and `LAZY.md`
(architecture of the lazy graph + candle backend, benchmarks, resume notes).

## Prompt to paste

```
Work on /Users/texoport/code/typenet, branch lazy-native. Read PLAN.md
(the roadmap — pick up at the first unchecked item) and LAZY.md
(architecture + resume instructions). Reference codebase for gradcheck,
negative type tests, and the type-algebra tricks: /Users/texoport/code/typonet
(especially src/gradcheck.ts, src/errors.test-d.ts, src/shape.ts, and the
pitfalls section of its README).

Do Phase A now: (1) finite-difference gradcheck for every backward rule,
run in both eager and lazy modes, f32-appropriate tolerances;
(2) negative type tests via @ts-expect-error in test/types.test-d.ts.
Follow PLAN.md's design invariants: no public type signature changes,
eager mode byte-identical. Verify with pnpm test and pnpm typecheck (the
pre-existing examples/basic.ts error is expected — it's a deliberate
scratch-file shape error, ignore it). Commit each task separately in the
repo's lowercase style, then check the boxes in PLAN.md.

Note: examples/basic.ts, examples/_gatdsl.ts, examples/_yieldshape.ts are
the owner's uncommitted scratch files — never commit or revert them.
```

## State snapshot (2026-07-27)

- Branch `lazy-native`, 4 commits ahead of main:
  - `304fe6c` lazy graph (phase 1)
  - `bb1ffa5` candle native backend (phase 2)
  - `0e603d9` symbolic backward + multi-root eval
  - `e4e295d` PLAN.md
- 92/92 tests pass; native backend runs on Metal; build via `pnpm build:native`.
- Benchmarks in LAZY.md: native wins 6–23x on large graphs; XOR at 6.2ms vs
  3.5ms eager (compile() in Phase B is the fix).
- Next: Phase A → B (compile + optimizer-in-graph) → C (naming, type
  hardening) → D (conditional Rust-side fusion).
