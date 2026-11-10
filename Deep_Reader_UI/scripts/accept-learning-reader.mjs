// LLRN-21 reproducible UI gate. Playwright/Chrome paths may be supplied via env.
import { spawn } from "node:child_process";
import { createHash } from "node:crypto";
import { copyFile, mkdir, readFile, writeFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import path from "node:path";

const root = fileURLToPath(new URL("../", import.meta.url));
const output = path.resolve(process.env.LEARNING_ACCEPTANCE_DIR || "/tmp/llrn21");
await mkdir(output, { recursive: true });
const scripts = ["test-learning-controller", "test-reading-settings", "test-learning-reader",
  "test-learning-display", "test-learning-lifecycle", "test-learning-reader-api",
  "test-translation-inline", "test-translation-session-lifecycle", "test-reader-guards",
  "test-source-review-navigation"];
const report = { task: "LLRN-21", status: "incomplete", startedAt: new Date().toISOString(),
  viewports: [1280, 390], liveModel: false, suites: [], sourceHashes: {},
  limits: "Controlled HTTP browser tests plus isolated file-backed learning API/offline provider; no SQL browser or semantic-quality certification. Screenshots need visual review." };
const sha = data => createHash("sha256").update(data).digest("hex");
const sourceFiles = ["src/features/reading-settings/view.tsx", "src/features/reading-settings/features.tsx",
  "src/features/reading-settings/controller.ts", "src/features/learning-reader/controller.ts",
  "src/features/learning-reader/session.ts", "src/features/learning-reader/block-view.tsx",
  "scripts/learning-controller-api-fixture.py", "scripts/learning-controller-fixtures.mjs",
  ...scripts.map(name => `scripts/${name}.mjs`)];
for (const name of sourceFiles) report.sourceHashes[name] = sha(await readFile(path.join(root, name)));
const run = (command, args, name) => new Promise(resolve => {
  const start = Date.now(); let log = "";
  const child = spawn(command, args, { cwd: root, env: { ...process.env, LEARNING_ACCEPTANCE_DIR: output }, timeout: 240000,
    stdio: ["ignore", "pipe", "pipe"] });
  child.stdout.on("data", data => { log += data; }); child.stderr.on("data", data => { log += data; });
  child.on("error", error => { log += `${error.name}: ${error.message}\n`; });
  child.on("close", async (code, signal) => {
    await writeFile(path.join(output, `${name}.log`), log);
    const result = { name, exitCode: code, signal, durationMs: Date.now() - start, passed: code === 0 };
    report.suites.push(result); console.log(JSON.stringify(result)); resolve(result.passed);
  });
});
try {
  let passed = await run("npm", ["run", "check"], "build");
  for (const name of scripts) {
    if (!passed) break;
    passed = await run(process.execPath, [`scripts/${name}.mjs`], name);
  }
  for (const [name, hash] of Object.entries(report.sourceHashes)) {
    if (sha(await readFile(path.join(root, name))) !== hash) throw Error(`Source changed during acceptance: ${name}`);
  }
  // Preserve the existing display suites' screenshots alongside this gate's trace/request evidence.
  if (passed) for (const width of report.viewports) for (const prefix of ["llrn18-reader", "llrn19-display", "llrn19-display-block"]) {
    await copyFile(`/tmp/${prefix}-${width}.png`, path.join(output, `${prefix}-${width}.png`));
  }
  report.status = passed && report.suites.length === scripts.length + 1 ? "passed" : "failed";
} catch (error) { report.status = "failed"; report.error = error.message; }
finally {
  report.finishedAt = new Date().toISOString();
  await writeFile(path.join(output, "report.json"), JSON.stringify(report, null, 2) + "\n");
}
console.log(`LLRN-21 ${report.status}: ${output}/report.json`);
process.exitCode = report.status === "passed" ? 0 : 1;
