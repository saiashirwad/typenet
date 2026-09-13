// Re-export shim (W1.10, mirroring W1.7's `src/nn.ts`). The optimizer
// implementation moved into `src/optim/**` so `schedule.ts` and the
// growing optimizer set (`SGD`/`Adam`/`AdamW`) have room without one
// monolithic file; every existing import path (`../src/optim.ts`, from
// index.ts and every test) keeps working through this one line alone —
// new code should import from `./optim/index.ts` directly.
export * from "./optim/index.ts"
