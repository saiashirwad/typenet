// Type declarations for typecheck-budget.mjs, consumed by bench/typecheck.ts
// (a .ts file importing a plain .mjs script — see scripts/typecheck-budget.mjs
// for the implementation, which is the single source of truth).

export interface ExtendedDiagnostics {
  instantiations?: number
  types?: number
  memoryUsedKB?: number
  checkTimeSeconds?: number
  totalTimeSeconds?: number
}

export function parseExtendedDiagnostics(output: string): ExtendedDiagnostics

/**
 * Runs tsc over the budget file set and returns the parsed counters. Throws
 * (with the raw tsc output attached as `.output`) on a compile error or on
 * unparseable output.
 */
export function measureBudget(): ExtendedDiagnostics
