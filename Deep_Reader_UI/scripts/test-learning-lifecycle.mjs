// LLRN-21: actual StrictMode App, controlled HTTP; no user data/model writes.
import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { mkdir, writeFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import * as f from "./learning-controller-fixtures.mjs";

const backend = fileURLToPath(new URL("../../Deep_Reflective_Reader/", import.meta.url));
const options = JSON.parse(execFileSync(`${backend}.venv/bin/python`, ["-c",
  "from api_learning_projection import reading_options_response; print(reading_options_response().model_dump_json())"], { cwd: backend, encoding: "utf8" }));
const scheme = options.proficiency_schemes.find(s => s.language_code === "en");
const evidence = process.env.LEARNING_ACCEPTANCE_DIR || "/tmp/llrn21";
await mkdir(evidence, { recursive: true });
const pause = ms => new Promise(r => setTimeout(r, ms));
const until = async fn => { for (let i = 0; i < 250; i++) { if (await fn()) return; await pause(20); } throw Error("Lifecycle fixture timed out"); };
const { chromium } = await import(process.env.PLAYWRIGHT_MODULE || "playwright");
const browser = await chromium.launch({ headless: true, ...(process.env.CHROMIUM_PATH ? { executablePath: process.env.CHROMIUM_PATH } : {}) });
try { for (const width of [1280, 390]) {
  const context = await browser.newContext({ viewport: { width, height: 900 } });
  await context.tracing.start({ screenshots: true, snapshots: true, sources: true });
  const page = await context.newPage(), errors = [], unexpected = [], calls = [];
  page.setDefaultTimeout(10000);
  page.on("pageerror", e => errors.push(e.message));
  let mode = "normal", revision = 1, contentRevision = 0, selected = null;
  let rejectSave = false, holdSave = false, holdGenerate = false, release = null;
  let lastRefresh = null;
  const original = "Hello world. Bonjour le monde. مرحبا بالعالم. " + "LongUnbrokenWord".repeat(35);
  let block = f.block("u1:content:0", original);
  const operations = new Map();
  const third = { mode_id: "acceptance_preview", label: "Preview test mode", description: "Test-only multi-step panel.", settings_version: "1", eligible: true, reasons: [] };
  const project = () => ({ ...f.book("Book", mode), revision,
    modes: [...f.book().modes.map(m => ({ ...m, label: m.mode_id === "normal" ? "Normal reading" : "Language learning" })), third],
    saved_settings: selected ? [{ mode_id: "language_learning", version: "1", supported: true, selection: selected }] : [] });
  const state = () => ({ document_id: "Book", revision: contentRevision, source_token: `content-${contentRevision}`,
    disposition: "accepted", ocr_review_available: true, can_undo: contentRevision === 1, undo_action: contentRevision === 1 ? "edit" : null,
    max_chars: 10000, regions: [], cleanup_audit: { total: 0, offset: 0, limit: 10, items: [] },
    blocks: [{ block_id: block.content_block_id, content: block.sentences.map(s => s.text).join(""), deleted: false, chapter_id: "c", section_id: "s", task_unit_id: "u1" }] });
  const generations = () => calls.filter(c => c.path.endsWith("/learning/generate"));
  await page.route("**/api/**", async route => {
    const req = route.request(), url = new URL(req.url()), path = url.pathname, body = req.postData() ? req.postDataJSON() : null;
    calls.push({ path, method: req.method(), body });
    if (path === "/api/documents" || path === "/api/documents/management") return route.fulfill({ json: { total: 1, items: [{ doc_name: "Book", title: "Learning acceptance book", source: "structured", preparation_status: "ready" }] } });
    if (path.endsWith("/task-layout")) return route.fulfill({ json: { document_id: "Book", title: "Book", chapters: [{ chapter_id: "c", title: "Chapter", sections: [{ section_id: "s", title: "Section", task_units: [{ unit_id: "u1" }, { unit_id: "u2" }] }] }] } });
    if (path.endsWith("/task-units/content")) return route.fulfill({ json: { contents: body.task_unit_ids.map(id => ({ task_unit_id: id, document_id: "Book", chapter_id: "c", section_id: "s",
      content_blocks: [{ block_id: id === "u1" ? block.content_block_id : "u2:content:0", block_type: "content", content: id === "u1" ? state().blocks[0].content : "Unseen next page." }] })) } });
    if (path.endsWith("/reparse-structure/status")) return route.fulfill({ json: { status: "idle" } });
    if (path.endsWith("/reading-content")) {
      if (req.method() === "GET") return route.fulfill({ json: url.searchParams.has("refresh_mutation_id") ? lastRefresh : state() });
      assert.equal(body.expected_revision, contentRevision); assert.equal(body.source_token, state().source_token);
      assert(["edit", "undo"].includes(body.action)); contentRevision++;
      block = f.block(block.content_block_id, body.action === "undo" ? original : body.text);
      block.source_token = String(contentRevision + 1).repeat(64);
      const next = state();
      next.refresh = { document_id: "Book", structure_version: null, revision: contentRevision,
        source_token: next.source_token, mutation_id: body.mutation_id, mutation_revision: contentRevision,
        scope_current: true, generated_invalidated: true, body_changed: true, translation_block_ids: [block.content_block_id],
        blocks: [{ block_id: block.content_block_id, chapter_id: "c", section_id: "s", task_unit_id: "u1", block_order: 0,
          unit_source_hash: `hash-${contentRevision}`, deleted: false, manually_edited: contentRevision === 1,
          content_block: { block_id: block.content_block_id, block_type: "content", content: next.blocks[0].content } }], block_page: { total: 1, offset: 0, limit: 20 } };
      lastRefresh = next.refresh;
      return route.fulfill({ json: next });
    }
    if (path.endsWith("/reading-preferences/options")) return route.fulfill({ json: options });
    if (path.endsWith("/reading-preferences/workspace")) return route.fulfill({ json: { revision: 1, native_language: "zh" } });
    if (path.endsWith("/reading-preferences/read")) return route.fulfill({ json: project() });
    if (path.endsWith("/reading-preferences/apply")) {
      if (rejectSave) { rejectSave = false; return route.fulfill({ status: 422, json: { detail: { kind: "ineligible", code: "preference_settings_invalid", reasons: [{ code: "proficiency_invalid", field: "proficiency", message: "Test rejection; review your level." }] } } }); }
      if (holdSave) { holdSave = false; await new Promise(r => release = r); }
      assert.equal(body.expected_revision, revision); assert.equal(body.expected_workspace_revision, 1);
      mode = body.selection.mode_id; revision++;
      if (mode === "language_learning") selected = body.selection;
      return route.fulfill({ json: project() });
    }
    const endpoint = path.split("/learning/")[1];
    if (endpoint === "read") return route.fulfill({ json: { doc_name: "Book", settings: project(), blocks: [block] } });
    if (endpoint === "generate") {
      if (holdGenerate) { holdGenerate = false; await new Promise(r => release = r); }
      assert.equal(body.expected_source_token, block.source_token);
      const value = f.save(block, body.component, (block.components.find(c => c.component === body.component).saved?.result_revision || 0) + 1);
      if (body.component === "hints") value.saved.proficiency = selected;
      const op = f.operation(block, body.component, "succeeded", body.command_id); op.result_revision = value.saved.result_revision;
      operations.set(body.command_id, op); f.attach(block, op);
      return route.fulfill({ json: f.submission(body) });
    }
    if (["operations/read", "operations/wait"].includes(endpoint)) return route.fulfill({ json: operations.get(body.handle.operation_id) });
    unexpected.push(path); return route.fulfill({ status: 500, json: { detail: "Unexpected fixture request" } });
  });
  try {
    await page.goto(`${process.env.UI_URL || "http://127.0.0.1:5173"}/documents/Book`);
    // Register only in this browser realm. No test mode ships in the app or backend.
    await page.evaluate(async () => {
      const reactUrl = performance.getEntriesByType("resource").map(e => e.name)
        .find(url => new URL(url).pathname.endsWith("/react.js"));
      const { default: React } = await import(reactUrl || "/node_modules/.vite/deps/react.js");
      const featureUrl = performance.getEntriesByType("resource").map(e => e.name)
        .find(url => new URL(url).pathname === "/src/features/reading-settings/features.tsx");
      const { settingsFeatures } = await import(featureUrl || "/src/features/reading-settings/features.tsx");
      window.previewMounts = 0;
      function Panel() {
        const [step, setStep] = React.useState(1);
        React.useEffect(() => { window.previewMounts++; return () => { window.previewMounts--; }; }, []);
        return React.createElement("section", { "aria-label": "Test mode steps" },
          React.createElement("p", null, `Step ${step}`), React.createElement("button", { onClick: () => setStep(step + 1) }, "Next test step"));
      }
      settingsFeatures.acceptance_preview = { version: "1", Panel, problems: () => ["Preview-only test feature."], selection: () => { throw Error("Test mode must never be saved"); } };
    });
    await page.locator(".hierarchy-section-button").click();
    await page.locator(".reader-content-block-body").waitFor();
    const settings = page.locator(".reading-settings-dialog");
    const openSettings = async () => { await page.getByRole("button", { name: "Library", exact: true }).click(); await page.locator('.file-management-row[data-doc-name="Book"]').getByRole("button", { name: "Reading mode", exact: true }).click(); await settings.locator(".reading-mode-active").waitFor(); };
    const candidate = async id => { const option = settings.locator(`[data-mode-id="${id}"]`); await option.focus(); await page.keyboard.press("Enter"); };
    const dismissNotification = async () => { const close = settings.getByLabel("Close notification"); if (await close.count()) await close.click(); };
    const closeSettings = async () => { await dismissNotification(); await settings.getByRole("button", { name: "Back to Library", exact: true }).click(); await settings.waitFor({ state: "hidden" }); await page.getByRole("button", { name: "Close", exact: true }).click(); };
    const ready = () => until(async () => await page.locator(".learning-component-status").filter({ hasText: /Saved$/ }).count() === 2)
      .catch(async error => { console.error(await page.locator("body").innerText(), calls.slice(-8)); throw error; });
    await openSettings(); await candidate("acceptance_preview");
    await settings.getByRole("region", { name: "Test mode steps" }).waitFor().catch(async error => { console.error(errors, await page.locator("body").innerText()); throw error; });
    assert.match(await settings.locator(".reading-mode-active").innerText(), /Normal reading/);
    assert.equal(await settings.getByRole("button", { name: "Save and apply", exact: true }).isEnabled(), false);
    await settings.getByRole("button", { name: "Next test step", exact: true }).focus(); await page.keyboard.press("Enter");
    assert.match(await settings.innerText(), /Step 2/);
    assert.equal(await page.evaluate(() => window.previewMounts), 1);
    for (let i = 0; i < 14; i++) { await page.keyboard.press("Tab"); assert(await settings.evaluate(e => e.contains(document.activeElement)), "settings trap contains focus"); }
    await page.screenshot({ path: `${evidence}/mode-list-${width}.png` });
    if (width < 600) await settings.getByRole("button", { name: "Back to modes", exact: true }).click();
    else await candidate("language_learning");
    await settings.getByRole("region", { name: "Unsaved settings" }).waitFor();
    assert.equal(await page.evaluate(() => window.previewMounts), 1, "departure confirmation preserves active panel");
    assert.equal(await settings.getByRole("button", { name: "Next test step", exact: true }).count(), 0, "hidden panel is excluded from accessible controls");
    for (let i = 0; i < 6; i++) {
      await page.keyboard.press("Tab");
      assert(await settings.evaluate(e => e.contains(document.activeElement) && !document.activeElement.closest(".reading-settings-panel")), "confirmation cannot tab into hidden settings");
    }
    await settings.getByRole("button", { name: "Keep editing", exact: true }).click();
    assert.match(await settings.innerText(), /Step 2/);
    await page.keyboard.press("Escape");
    await settings.getByRole("button", { name: "Discard draft and continue", exact: true }).click();
    await settings.waitFor({ state: "hidden" });
    assert.equal(await page.evaluate(() => window.previewMounts), 0);
    await page.locator('.file-management-row[data-doc-name="Book"]').getByRole("button", { name: "Reading mode", exact: true }).click();
    await settings.locator(".reading-mode-active").waitFor(); await candidate("language_learning");
    await settings.getByLabel("Reading level", { exact: true }).click();
    await page.getByRole("option", { name: scheme.levels[0].label, exact: true }).click();
    const apply = settings.getByRole("button", { name: "Save and apply", exact: true });
    rejectSave = true; await apply.click(); await settings.getByText("Test rejection; review your level.", { exact: false }).waitFor();
    assert.equal(mode, "normal"); assert.match(await settings.getByLabel("Reading level", { exact: true }).innerText(), new RegExp(scheme.levels[0].label));
    await dismissNotification(); holdSave = true; await apply.evaluate(e => { e.click(); e.click(); });
    await until(() => release !== null); assert.equal(calls.filter(c => c.path.endsWith("/apply")).length, 2);
    assert.equal(generations().length, 0, "preview and save cannot start learning while modal is open");
    release(); release = null; await until(() => mode === "language_learning");
    await until(async () => (await settings.locator(".reading-mode-active").innerText()).includes("Language learning"));
    await closeSettings(); await ready(); assert.equal(generations().length, 2);
    assert.equal(await page.locator(".learning-vocabulary-empty").count(), 1);
    const action = page.getByRole("button", { name: "Refresh translation", exact: true });
    holdGenerate = true; await action.evaluate(e => { e.click(); e.click(); }); await until(() => release !== null);
    assert.equal(generations().length, 3, "duplicate click is deduplicated");
    assert.equal(await page.locator(".learning-translated-sentence").count(), 1);
    release(); release = null; await ready();
    // Real editor interactions with controlled source mutation responses.
    await page.locator(".reader-content-block-interaction-trigger").click(); await page.getByRole("menuitem", { name: "Edit", exact: true }).click();
    const editor = page.locator('[aria-labelledby="reading-content-dialog-title"]');
    await editor.getByLabel("Block text").fill("Edited source. 中文 quotation.");
    await editor.getByRole("button", { name: "Save", exact: true }).click();
    await page.getByRole("button", { name: "Keep and save", exact: true }).click();
    await editor.waitFor({ state: "hidden" }); await ready();
    assert.match(await page.locator(".learning-source-sentence").innerText(), /Edited source/);
    assert.equal(generations().length, 5);
    await page.getByRole("button", { name: "Undo last content change", exact: true }).click(); await ready();
    await until(async () => (await page.locator(".learning-source-sentence").innerText()) === original);
    await ready();
    assert.equal(generations().length, 7, "Undo has fresh source binding and cannot reuse the deleted result");
    assert.equal(new Set(generations().map(c => c.body.command_id)).size, 7);
    assert.equal(new Set(generations().map(c => c.body.expected_source_token)).size, 3);
    assert(generations().every(c => c.body.content_block_id === "u1:content:0"), "no unseen-page generation");
    assert(await page.locator(".learning-source-sentence").evaluate(e => e.scrollWidth <= e.clientWidth + 1), "long source wraps");
    await action.focus(); await page.screenshot({ path: `${evidence}/reader-${width}.png` });
    assert.equal(await action.evaluate(e => document.activeElement === e), true);
    await openSettings(); await candidate("normal");
    await settings.getByRole("button", { name: "Save and apply", exact: true }).click();
    await until(() => mode === "normal"); await closeSettings();
    await until(async () => await page.locator(".learning-sentence-pairs").count() === 0);
    assert.equal(await page.locator(".reader-content-block-body").innerText(), original);
    await page.locator(".reader-content-block-interaction-trigger").click();
    await page.getByRole("menuitem", { name: "View translations", exact: true }).waitFor(); await page.keyboard.press("Escape");
    assert.equal(generations().length, 7); assert.deepEqual(errors, []); assert.deepEqual(unexpected, []);
    const report = { width, pass: true, controlledHTTP: true, generationCommands: 7, sourceMutations: 2,
      thirdModePreviewOnly: true, requestCounts: Object.fromEntries([...new Set(calls.map(c => c.path))].map(path => [path, calls.filter(c => c.path === path).length])), calls };
    await writeFile(`${evidence}/requests-${width}.json`, JSON.stringify(report, null, 2));
    console.log(JSON.stringify({ width, pass: true, generationCommands: 7, thirdModePreviewOnly: true }));
  } finally {
    if (release) release();
    await page.unrouteAll({ behavior: "wait" });
    await context.tracing.stop({ path: `${evidence}/trace-${width}.zip` });
    await context.close();
  }
} } finally { await browser.close(); }
