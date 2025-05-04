import { defineConfig } from "vitest/config";

import tsover from "typescript/plugin/vite";

export default defineConfig({
  plugins: [tsover()],
  test: {
    include: ["test/**/*.test.ts"],
  },
});
