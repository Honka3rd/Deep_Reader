import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";

const suites = ["identity", "pdf-modes", "auto-prepare", "readiness", "workflow"];
const files = [...suites.map(name => `test-library-${name}.mjs`), "test-reader-guards.mjs", "test-reader-recovery.mjs", "test-enhancement-confirmation.mjs"];
for (const file of files) {
  console.log(`\nRunning ${file}`);
  const result = spawnSync(process.execPath, [fileURLToPath(new URL(file, import.meta.url))], {
    stdio: "inherit", env: process.env, timeout: 180_000,
  });
  if (result.error || result.status !== 0) {
    console.error(result.error?.message || `${file} failed (${result.signal || result.status})`);
    process.exit(1);
  }
}
console.log(`\nPASS ${files.length} Library/reader browser suites (mocked APIs)`);
