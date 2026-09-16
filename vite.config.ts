import { defineConfig } from "vitest/config"

import tsover from "typescript/plugin/vite"

// The tsover plugin re-checks the whole program per transformed file just to print warnings;
// `pnpm typecheck` is what gates type errors.
process.env.TSOVER_SKIP_DIAGNOSTICS ??= "1"

export default defineConfig({
  plugins: [tsover({ tsconfigPath: "tsconfig.vite.json" })],
  test: {
    include: ["test/**/*.test.ts"],
  },
})
