import { defineConfig } from "vitest/config"

import tsover from "typescript/plugin/vite"

export default defineConfig({
  plugins: [tsover({ tsconfigPath: "tsconfig.vite.json" })],
  test: {
    include: ["test/**/*.test.ts"],
  },
})
