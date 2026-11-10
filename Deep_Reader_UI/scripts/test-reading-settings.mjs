// Actual StrictMode App; controlled HTTP contracts. No library writes or model requests.
import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { fileURLToPath } from "node:url";
const { chromium } = await import(process.env.PLAYWRIGHT_MODULE || "playwright");
const backend = fileURLToPath(new URL("../../Deep_Reflective_Reader/", import.meta.url));
const options = JSON.parse(execFileSync(`${backend}.venv/bin/python`, ["-c", "from api_learning_projection import reading_options_response; print(reading_options_response().model_dump_json())"], { cwd: backend, encoding: "utf8" }));
const pause = ms => new Promise(resolve => setTimeout(resolve, ms));
const until = async fn => { for (let i = 0; i < 150; i++) { if (await fn()) return; await pause(20); } assert.fail("Fixture condition timed out"); };
const learning = language => { const s = options.proficiency_schemes.find(s => s.language_code === language); return { mode_id: "language_learning", version: "1", language_code: language, scheme_id: s.scheme_id, scheme_version: s.version, level_id: s.levels[0].level_id }; };
const issue = (code, field, message) => ({ code, field, message });
const browser = await chromium.launch({ headless: true, ...(process.env.CHROMIUM_PATH ? { executablePath: process.env.CHROMIUM_PATH } : {}) });
try {
for (const width of [1280, 390]) {
 const page = await browser.newPage({ viewport: { width, height: 900 } });
 const errors = [], calls = [], unexpected = [];
 let workspace = { revision: 0, native_language: null };
 const books = Object.fromEntries(["Book", "Other", "Same"].map(doc => [doc, { revision: 0, active_mode: "normal", override: null, learning: null }]));
 let fail = null, hold = null, release = null, hiddenUnknown = false, malformed = false, largeCatalog = false;
 function project(doc) {
  const b = books[doc] ?? { revision: 0, active_mode: "normal", override: null, learning: null }, source = b.override || (doc === "Same" ? "zh" : "en");
  const reasons = [];
  if (!workspace.native_language) reasons.push(issue("native_language_required", "native_language", "Set your native language."));
  if (source === workspace.native_language) reasons.push(issue("same_language", "source_language", "Original and native languages must differ."));
  if (!b.learning || b.learning.language_code !== source) reasons.push(issue("proficiency_required", "proficiency", "Choose a reading level."));
  const deletion = b.deleted ? [issue("book_deleted", "document_id", "These preferences belong to a deleted book identity.")] : [];
  reasons.unshift(...deletion);
  const activeReasons = b.active_mode === "normal" ? deletion : b.active_mode === "future_mode" ? [issue("mode_unknown", "active_mode", "This saved mode is not supported.")] : reasons;
  return { doc_name: doc, revision: b.revision, workspace: { ...workspace }, active_mode: b.active_mode,
   effective_mode: activeReasons.length ? null : b.active_mode, source_language_override: b.override, source_language: source,
   source_origin: b.override ? "override" : "profile", active_reasons: activeReasons,
   modes: [{ mode_id: "normal", label: "Normal reading", description: "Read with the usual tools.", settings_version: "1", eligible: !deletion.length, reasons: deletion },
    { mode_id: "language_learning", label: "Language learning", description: "Learn the original language.", settings_version: "1", eligible: !reasons.length, reasons },
    ...(hiddenUnknown ? [{ mode_id: "future_mode", label: "Future mode", description: "A future feature.", settings_version: "2", eligible: b.active_mode !== "future_mode", reasons: b.active_mode === "future_mode" ? activeReasons : [] }] : [])],
   saved_settings: b.learning ? [{ mode_id: "language_learning", version: "1", supported: true, selection: b.learning }] : [] };
 }
 page.on("pageerror", e => errors.push(e.message));
 await page.route("**/api/**", async route => {
  const req = route.request(), path = new URL(req.url()).pathname, method = req.method(), body = req.postData() ? req.postDataJSON() : null;
  if (path === "/api/documents" || path === "/api/documents/management") return route.fulfill({ json: { total: largeCatalog ? 200 : 3,
    items: [...Object.keys(books), ...(largeCatalog ? Array.from({ length: 197 }, (_, i) => `Unread ${i}`) : [])].map(doc_name => ({ doc_name, title: doc_name === "Book" ? "A long book title — ".repeat(12) : doc_name, source: "structured", preparation_status: "ready" })) } });
  if (!path.includes("reading-preferences")) { unexpected.push(path); return route.fulfill({ status: 500, json: { detail: "Unexpected" } }); }
  calls.push({ path, method, body });
  const heldResponse = body?.doc_name ? project(body.doc_name) : null;
  if (hold === path && await page.locator('.reading-settings-dialog').isVisible()) { hold = null; await new Promise(resolve => { release = resolve; }); }
  if (path.endsWith("/options")) return route.fulfill({ json: options });
  if (path.endsWith("/workspace") && method === "GET") return route.fulfill({ json: workspace });
  if (path.endsWith("/read")) return route.fulfill({ json: malformed ? { ...heldResponse, doc_name: "wrong-book" } : heldResponse });
  if (fail === "conflict") { fail = null; return route.fulfill({ status: 409, json: { detail: { kind: "conflict", code: "preference_revision_conflict" } } }); }
  if (fail === "rejected") { fail = null; return route.fulfill({ status: 422, json: { detail: { kind: "ineligible", code: "preference_settings_invalid", reasons: [issue("proficiency_invalid", "proficiency", "Choose a current level.")] } } }); }
  if (path.endsWith("/workspace")) {
   assert.equal(body.expected_revision, workspace.revision); workspace = { revision: workspace.revision + 1, native_language: body.native_language };
  } else if (path.endsWith("/apply")) {
   const b = books[body.doc_name];
   assert.equal(body.expected_revision, b.revision); assert.equal(body.expected_workspace_revision, workspace.revision);
   b.active_mode = body.selection.mode_id; b.override = body.source_language_override; b.revision++; b.deleted = false;
   if (body.selection.mode_id === "language_learning") b.learning = body.selection;
  }
  if (fail === "unknown") { fail = null; return route.fulfill({ status: 503, json: { detail: { kind: "unknown", code: "preference_commit_unknown", recovery: "read_preferences" } } }); }
  if (fail === "wrong-selection") { fail = null; const response = project(body.doc_name);
   response.saved_settings[0].selection = { ...response.saved_settings[0].selection, level_id: "unexpected-level" };
   return route.fulfill({ json: response });
  }
  return route.fulfill({ json: path.endsWith("/workspace") ? workspace : project(body.doc_name) });
 });
 page.setDefaultTimeout(10000);
 await page.goto(process.env.UI_URL || "http://127.0.0.1:5173");
 const pureChecks = await page.evaluate(async ({ options, book }) => {
  const model = await import("/src/features/reading-settings/model.ts");
  const dto = await import("/src/services/ReadingPreferencesService.ts");
  const failures = [];
  const check = (name, value) => { if (!value) failures.push(name); };
  check("valid book", dto.isBook(book)); check("valid options", dto.isOptions(options));
  check("missing revision", !dto.isBook({ ...book, revision: undefined }));
  check("contradictory mode", !dto.isBook({ ...book, effective_mode: null }));
  check("duplicate catalog", !dto.isBook({ ...book, modes: [...book.modes, book.modes[0]] }));
  check("duplicate language", !dto.isOptions({ ...options, language_codes: [...options.language_codes, options.language_codes[0]] }));
  check("invalid workspace revision", !dto.isWorkspace({ revision: -1, native_language: null }));
  const draft = model.initialDraft(book, "language_learning");
  check("candidate dirty", model.draftChanged(draft, book));
  check("missing native/level", model.learningProblems(draft, { book, options }).length === 2);
  const blocked = { ...book, modes: [{ ...book.modes[0], reasons: [{ code: "book_deleted", field: "document_id", message: "Deleted book" }] }] };
  check("explicit reapply can resolve tombstone", model.serverBlockers(blocked, "normal").length === 0);
  blocked.modes[0].reasons.push({ code: "future_policy", field: "document_id", message: "Unknown policy" });
  check("unknown reasons still block", model.serverBlockers(blocked, "normal")[0] === "Unknown policy");
  check("profile is not guessed", model.sourceLanguage({ ...draft, override: null }, { ...book, source_origin: "override" }) === null);
  return failures;
 }, { options, book: project("Book") });
 assert.deepEqual(pureChecks, []);
 await page.getByRole("button", { name: "Library", exact: true }).click();
 const surface = page.locator(".reading-settings-dialog");
 const row = doc => page.locator(`.file-management-row[data-doc-name="${doc}"]`);
 const open = async doc => { await row(doc).getByRole("button", { name: "Reading mode", exact: true }).click(); await surface.locator(".reading-mode-active").waitFor(); };
 const mode = async name => { await surface.getByRole("button", { name: new RegExp(`^${name}`) }).click(); };
 const select = async (label, text) => { await surface.getByLabel(label, { exact: true }).click(); await page.getByRole("option", { name: text, exact: true }).click(); };
 const dismiss = async () => { if (await surface.getByLabel("Close notification").count()) { await surface.getByLabel("Close notification").click(); await surface.locator(".app-notification").waitFor({ state: "hidden" }); } };
 const save = async () => { await surface.locator(".reading-settings-save").click(); await until(async () => !(await surface.locator(".reading-settings-save").innerText()).includes("wait")); };
 const back = async () => { await surface.locator(".reading-settings-cancel").click(); await surface.waitFor({ state: "hidden" }); };
 const confirmDiscard = async () => { await surface.getByRole("button", { name: "Discard draft and continue" }).click(); };
 assert(calls.every(c => c.path.endsWith("/read")), "Library labels only read book preferences");
 await page.getByRole("button", { name: "Workspace preferences", exact: true }).click();
 await surface.getByLabel("Native language", { exact: true }).waitFor();
 await surface.getByLabel("Native language", { exact: true }).click();
 assert.equal(await page.getByRole("option").count(), options.language_codes.length + 1);
 await page.getByRole("option", { name: "Chinese (zh)", exact: true }).click();
 await save(); assert.equal(workspace.native_language, "zh"); await dismiss(); await back();
 // R3: a canonical reimport is present, but preferences retain a deletion tombstone.
 for (const chosen of ["Normal reading", "Language learning"]) {
  Object.assign(books.Other,{deleted:true,active_mode:"normal",learning:null}); books.Other.revision++;
  const before=calls.filter(c=>c.path.endsWith("/apply")).length, revision=books.Other.revision;
  await open("Other"); await mode(chosen);
  if(chosen==="Language learning") await select("Reading level",options.proficiency_schemes.find(s=>s.language_code==="en").levels[0].label);
  assert.equal(calls.filter(c=>c.path.endsWith("/apply")).length,before,"reimport read must not restore settings");
  assert.equal(books.Other.deleted,true);
  assert.match(await surface.locator(".reading-mode-active-reason").innerText(), /re-imported.*Save and apply/);
  assert.equal(await surface.locator(".reading-settings-save").isEnabled(),true,"R3 explicit reapply must be available");
  if(chosen==="Normal reading") await page.screenshot({path:`/tmp/audit-e7073/reimport-settings-${width}.png`,fullPage:true,animations:"disabled"});
  await save(); await dismiss();
  assert.equal(books.Other.deleted,false); assert.equal(books.Other.revision,revision+1);
  assert.equal(calls.filter(c=>c.path.endsWith("/apply")).length,before+1);
  await back();
 }
 if(process.env.REIMPORT_ONLY){assert.deepEqual(errors,[]);assert.deepEqual(unexpected,[]);console.log(`PASS reimport ${width}: explicit normal/learning apply, read purity, CAS`);await page.close();continue;}
 Object.assign(books.Other,{active_mode:"normal",learning:null});
 await open("Book");
 await mode("Language learning");
 assert.equal(await surface.locator(".reading-settings-save").isEnabled(), false);
 assert.match(await surface.innerText(), /Preview · not applied/);
 const writesBefore = calls.filter(c => c.path.endsWith("/apply")).length;
 const enScheme = options.proficiency_schemes.find(s => s.language_code === "en");
 await select("Reading level", enScheme.levels[0].label);
 assert.equal(calls.filter(c => c.path.endsWith("/apply")).length, writesBefore);
 // Candidate switching and mobile back protect the current draft without writing.
 if (width < 600) await surface.getByRole("button", { name: "Back to modes" }).click();
 else await mode("Normal reading");
 await surface.getByRole("button", { name: "Keep editing" }).click();
 assert.match(await surface.getByLabel("Reading level", { exact: true }).innerText(), new RegExp(enScheme.levels[0].label));
 // Escape protects the draft and keeps focus inside the same dialog.
 await page.keyboard.press("Escape"); await surface.getByRole("region", { name: "Unsaved settings" }).waitFor();
 assert.equal(await surface.getByRole("button", { name: "Keep editing" }).evaluate(e => e === document.activeElement), true);
 await surface.getByRole("button", { name: "Keep editing" }).click();
 hold = "/api/documents/reading-preferences/apply";
 await surface.locator(".reading-settings-save").dblclick(); await until(() => !!release);
 assert.equal(calls.filter(c => c.path.endsWith("/apply")).length, writesBefore + 1);
 release(); release = null; await until(() => books.Book.active_mode === "language_learning");
 await until(async () => (await surface.locator(".reading-mode-active").innerText()).includes("Language learning"));
 await dismiss();
 await page.screenshot({ path: `/tmp/llrn17-settings-${width}.png`, fullPage: true });
 assert.equal(await surface.evaluate(e => e.scrollWidth <= e.clientWidth + 1), true);
 for (let i = 0; i < 14; i++) { await page.keyboard.press("Tab"); assert.equal(await surface.evaluate(e => e.contains(document.activeElement)), true); }
 await back();
 await until(async () => await row("Book").getByRole("button", { name: "Reading mode", exact: true }).evaluate(e => e === document.activeElement));
 assert.match(await row("Book").innerText(), /Language learning/);
 console.log(`checkpoint ${width}: saved and focus restored`);
 // Same-language learning is inspectable but cannot be applied.
 await open("Same"); await mode("Language learning");
 assert.match(await surface.innerText(), /matches your native language/);
 assert.equal(await surface.getByLabel("Reading level", { exact: true }).count(), 0);
 assert.equal(await surface.locator(".reading-settings-save").isEnabled(), false);
 await surface.locator(".reading-settings-cancel").click(); await confirmDiscard(); await surface.waitFor({ state: "hidden" });
 console.log(`checkpoint ${width}: same-language`);
 // Read failures/mismatched document never display or apply another book's state.
 malformed = true; await row("Other").getByRole("button", { name: "Reading mode", exact: true }).click();
 await surface.getByText("Reading settings could not be loaded.").waitFor();
 assert.equal(await surface.locator(".reading-settings-save:enabled").count(), 0);
 malformed = false; await surface.getByRole("button", { name: "Try again" }).click(); await surface.locator(".reading-mode-active").waitFor();
 await mode("Language learning"); await select("Reading level", enScheme.levels[1].label);
 fail = "rejected"; await save(); assert.match(await surface.innerText(), /Choose a current level/); assert.equal(books.Other.active_mode, "normal");
 assert.match(await surface.getByLabel("Reading level", { exact: true }).innerText(), new RegExp(enScheme.levels[1].label));
 fail = "conflict"; await dismiss(); await save();
 assert.equal(await surface.locator(".reading-settings-save").isEnabled(), false);
 await surface.getByRole("button", { name: "Read current settings" }).click(); await until(async () => await surface.locator(".reading-settings-save").isEnabled()); await dismiss();
 console.log(`checkpoint ${width}: rejected/conflict`);
 // Lost acknowledgment after commit: read first, never replay automatically.
 const beforeUnknown = calls.filter(c => c.path.endsWith("/apply")).length;
 fail = "unknown"; await save(); assert.equal(books.Other.active_mode, "language_learning");
 assert.equal(await surface.locator(".reading-settings-save").isEnabled(), false);
 await surface.getByRole("button", { name: "Read current settings" }).click();
 await until(async () => (await surface.locator(".reading-mode-active").innerText()).includes("Language learning"));
 assert.equal(calls.filter(c => c.path.endsWith("/apply")).length, beforeUnknown + 1);
 await dismiss(); await back();
 console.log(`checkpoint ${width}: unknown reconciled`);
 // A response with another level cannot be reported as this save's success.
 await open("Other"); if (width < 600) await mode("Language learning");
 await select("Reading level", enScheme.levels[2].label); fail = "wrong-selection";
 await save(); assert.equal(await surface.locator(".reading-settings-save").isEnabled(), false);
 assert.match(await surface.innerText(), /save outcome is unknown/);
 await surface.getByRole("button", { name: "Read current settings" }).click();
 await until(async () => await surface.locator(".reading-settings-save").isEnabled()); await dismiss(); await back();
 // Explicit source correction uses that language's registry scheme, not a shared hard-coded scale.
 await open("Other"); if (width < 600) await mode("Language learning");
 await select("Original language", "French (fr)");
 const fr = options.proficiency_schemes.find(s => s.language_code === "fr"); await select("Reading level", fr.levels[0].label);
 await save(); assert.equal(books.Other.learning.scheme_id, fr.scheme_id); assert.equal(books.Other.override, "fr"); await dismiss();
 if (width < 600) { await surface.getByRole("button", { name: "Back to modes" }).click(); }
 await mode("Normal reading"); await save(); assert.equal(books.Other.learning.language_code, "fr", "Switching normal retains saved learning settings"); await dismiss();
 await select("Original language", "Use saved profile"); await save(); assert.equal(books.Other.override, null); await dismiss(); await back();
 console.log(`checkpoint ${width}: source/normal retained`);
 // Unsupported active mode is never silently rewritten.
 books.Other.active_mode = "future_mode"; hiddenUnknown = true;
 await open("Other"); if (width < 600) await mode("Future mode");
 assert.match(await surface.innerText(), /not supported/); assert.equal(await surface.locator(".reading-settings-save").isEnabled(), false);
 if (width < 600) await surface.getByRole("button", { name: "Back to modes" }).click();
 await mode("Normal reading"); await save(); await dismiss(); await back(); hiddenUnknown = false;
 console.log(`checkpoint ${width}: unsupported`);
 // Late read after close/reopen cannot overwrite the new target.
 hold = "/api/documents/reading-preferences/read";
 await row("Book").getByRole("button", { name: "Reading mode", exact: true }).click(); await until(() => !!release);
 await surface.getByRole("button", { name: "Close reading settings" }).click(); await surface.waitFor({ state: "hidden" });
 await open("Same"); release(); release = null; await pause(120);
 assert.match(await surface.locator("#reading-settings-title").innerText(), /Same/);
 assert.match(await surface.locator(".reading-mode-active").innerText(), /Normal reading/); await back();
 // Closing an in-flight mutation does not mean rollback; late result belongs to its original book.
 await open("Book"); if (width < 600) await mode("Language learning");
 await select("Reading level", enScheme.levels[1].label);
 hold = "/api/documents/reading-preferences/apply";
 await surface.locator(".reading-settings-save").click(); await until(() => !!release);
 await surface.getByRole("button", { name: "Close reading settings" }).click();
 assert.match(await surface.innerText(), /may already have reached the server/);
 await confirmDiscard(); await surface.waitFor({ state: "hidden" }); await open("Same");
 release(); release = null; await until(() => books.Book.learning.level_id === enScheme.levels[1].level_id);
 await pause(100); assert.match(await surface.locator("#reading-settings-title").innerText(), /Same/);
 assert.equal(await surface.locator(".app-notification").count(), 0);
 await back();
 // Workspace change invalidates remembered book summaries; book settings reload on return.
 await open("Book"); if (width < 600) await mode("Language learning");
 await surface.getByRole("button", { name: "Workspace preferences", exact: true }).click();
 await surface.getByLabel("Native language", { exact: true }).waitFor(); await select("Native language", "English (en)");
 fail = "unknown"; await save(); assert.equal(await surface.locator(".reading-settings-save").isEnabled(), false);
 await surface.getByRole("button", { name: "Read current settings" }).click(); await until(async () => !(await surface.innerText()).includes("Read current settings to resolve")); await dismiss();
 await surface.getByRole("button", { name: "Back to book" }).click(); await surface.locator(".reading-mode-active").waitFor();
 assert.match(await surface.locator(".reading-mode-active").innerText(), /Unavailable/);
 if (width < 600) await mode("Language learning"); assert.equal(await surface.locator(".reading-settings-save").isEnabled(), false);
 await back();
 await row("Book").locator('.file-management-reading-mode-action').scrollIntoViewIfNeeded();
 await until(async () => (await row("Book").locator('.file-management-mode-label').innerText()) === "Language learning · Unavailable");
 // Long library is contained; settings reads remain on demand, even with 200 rows.
 const preferencesBeforeRefresh = calls.length; largeCatalog = true;
 const started = Date.now(); await page.getByRole("button", { name: "Refresh", exact: true }).click();
 await until(async () => await page.locator(".file-management-row").count() === 200);
 const listMs = Date.now() - started; assert.ok(listMs < 10000, `Library responsiveness: ${listMs}ms`);
 await page.waitForTimeout(300);
 assert(calls.length - preferencesBeforeRefresh < 20, "200-row Library must not read every book's preferences");
 await page.screenshot({ path: `/tmp/llrn17-library-${width}.png`, fullPage: true });
 assert.deepEqual(errors, []); assert.deepEqual(unexpected, []);
 assert.equal(calls.some(c => /generate|task-layout|prepare/.test(c.path)), false);
 assert.equal(await page.evaluate(() => Object.keys(localStorage).some(k => /preferences|reading.mode|native.language/i.test(k))), false);
 console.log(`PASS reading settings ${width}: registry, draft/keyboard/focus, same-language, CAS, unknown read recovery, source/level, unsupported mode, stale response, no generation (${calls.length} requests, 200-row refresh ${listMs}ms)`);
 await page.close();
}
} finally { await browser.close(); }
