// Re-export shim (W1.7). `src/nn.ts` (298 lines) moved verbatim into
// `src/nn/**` — see that tree for the actual implementation. Every
// existing import path (`../src/nn.ts`, from index.ts, test/nn.test.ts
// and examples/**) keeps working through this one line alone; new code
// should import from `./nn/index.ts` directly. Deliberately not covered
// by OWNER-5 (PLAN-V2 §4): this is a module *path*, not an API spelling.
// Deleted one release after the move.
export * from "./nn/index.ts"
