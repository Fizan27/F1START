// vitest.config.mts
//
// What it does: settings for the website's tests (run with `npm test`). It
//   teaches the test runner the "@/" shortcut the code uses for the web folder.
// Which files use it: vitest reads it; the tests live in web/tests/.

import path from "node:path";
import { fileURLToPath } from "node:url";

import { defineConfig } from "vitest/config";

const here = path.dirname(fileURLToPath(import.meta.url));

export default defineConfig({
  resolve: { alias: { "@": here } },
  test: { include: ["tests/**/*.test.ts"] },
});
