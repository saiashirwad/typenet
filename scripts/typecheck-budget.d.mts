// Type declarations for typecheck-budget.mjs, consumed by bench/typecheck.ts.

export interface ExtendedDiagnostics {
  instantiations?: number
  types?: number
  memoryUsedKB?: number
  checkTimeSeconds?: number
  totalTimeSeconds?: number
}

export function parseExtendedDiagnostics(output: string): ExtendedDiagnostics

export function measureBudget(): ExtendedDiagnostics
