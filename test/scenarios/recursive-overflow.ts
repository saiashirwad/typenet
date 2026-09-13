// Run under `runOnSmallStack("recursive-overflow")`. This scenario is
// deliberately broken: it recurses one JS call per step instead of working
// iteratively, so it exists purely to prove the small-stack runner actually
// enforces the small stack rather than always reporting success — a
// negative control for test/deep.test.ts.
function recurse(n: number): number {
  if (n <= 0) return 0
  return 1 + recurse(n - 1)
}

console.log(recurse(1_000_000))
