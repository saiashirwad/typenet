// Run under runOnSmallStack("recursive-overflow"). Broken on purpose: it recurses one JS call per
// step, so it proves the runner enforces the stack rather than always reporting success.
function recurse(n: number): number {
  if (n <= 0) return 0
  return 1 + recurse(n - 1)
}

console.log(recurse(1_000_000))
