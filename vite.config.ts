import { defineConfig } from "vitest/config"
import typegpu from "unplugin-typegpu/vite"

import tsover from "typescript/plugin/vite"

export default defineConfig({
  plugins: [typegpu(), tsover()],
  test: {
    include: ["test/**/*.test.ts"]
  }
})
