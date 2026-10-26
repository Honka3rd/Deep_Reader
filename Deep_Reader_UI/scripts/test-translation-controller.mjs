// Mounted lifecycle, model and DTO regressions; controlled service, no production writes.
import assert from "node:assert/strict";
const { chromium } = await import(process.env.PLAYWRIGHT_MODULE || "playwright");
const browser = await chromium.launch({ headless: true, channel: process.env.PLAYWRIGHT_CHANNEL || "chrome" });
try {
  const page = await browser.newPage();
  const errors = [];
  page.on("pageerror", error => errors.push(error.message));
  await page.route("**/api/**", route => route.fulfill({ json: { documents: [] } }));
  await page.goto(process.env.UI_URL || "http://127.0.0.1:5173");
  await page.evaluate(async () => {
    const { default: React } = await import("/node_modules/.vite/deps/react.js");
    const { default: ReactDOMClient } = await import("/node_modules/.vite/deps/react-dom_client.js");
    const { default: ReactDOM } = await import("/node_modules/.vite/deps/react-dom.js");
    const { useBlockTranslation } = await import("/src/features/block-translation/controller.ts");
    const model = await import("/src/features/block-translation/model.ts");
    const dto = await import("/src/services/TranslationService.ts");
    const t = window.translationTest = { reads: [], writes: [], scope: "book", model, dto };
    t.blocks = ["a", "b"].map(id => ({ block_id: id, content: `Original ${id}`, ownerContext: { sourceHash: "source" } }));
    const service = {
      read: (target, signal) => new Promise((resolve, reject) => t.reads.push({ target, signal, resolve, reject })),
      generate: (command, signal) => new Promise((resolve, reject) => t.writes.push({ command, signal, resolve, reject })),
    };
    const container = document.createElement("div"); document.body.append(container);
    const root = ReactDOMClient.createRoot(container);
    function Probe() { t.c = useBlockTranslation({ docName: "book", blocks: t.blocks, scopeKey: t.scope, service }); return null; }
    t.render = () => ReactDOM.flushSync(() => root.render(React.createElement(React.StrictMode, null, React.createElement(Probe))));
    t.render();
    t.dispose = () => root.unmount();
    t.readResponse = target => ({ target: { ...target, document_id: "book", target_id: target.content_block_id,
      target_type: "content_block" }, language_codes: ["en", "fr", "de"], default_source_language: "en",
      effective_content_hash: "hash", reading_revision: 0, translations: [] });
    t.resolveRead = index => t.reads[index].resolve(t.readResponse(t.reads[index].target));
    t.result = (write, status = "completed") => {
      const data = t.readResponse(write.command), now = new Date().toISOString();
      const record = { source_language: "en", target_language: write.command.target_language, status,
        translated_text: "Bonjour", effective_content_hash: "hash", generated_at: now, updated_at: now,
        revision: 1, mutation_id: write.command.mutation_id };
      return { envelope: { target: data.target, status, reading_revision: 0, effective_content_hash: "hash", persisted: status !== "untranslatable" },
        payload: { source_language: "en", target_language: write.command.target_language, translated_text: status === "untranslatable" ? null : "Bonjour" },
        record: status === "untranslatable" ? null : record };
    };
  });
  const ready = async () => {
    await page.evaluate(() => { const t = window.translationTest; t.c.open(t.blocks[0], "translate"); });
    await page.evaluate(() => { const t = window.translationTest; t.resolveRead(t.reads.length - 1); });
    await page.waitForFunction(() => window.translationTest.c.status === "ready");
    await page.evaluate(() => window.translationTest.c.setTo("fr"));
    await page.waitForFunction(() => window.translationTest.c.canSubmit);
  };
  await page.evaluate(() => { const t = window.translationTest; t.c.open(t.blocks[0], "translate"); t.c.close(); t.resolveRead(0); });
  await page.waitForTimeout(50);
  assert.equal(await page.evaluate(() => window.translationTest.c.surface), null);
  assert.equal(await page.evaluate(() => window.translationTest.reads[0].signal.aborted), true);
  await page.evaluate(() => { const t = window.translationTest; t.c.open(t.blocks[0], "translate"); t.c.open(t.blocks[1], "translate"); t.resolveRead(1); t.resolveRead(2); });
  await page.waitForFunction(() => window.translationTest.c.status === "ready");
  assert.equal(await page.evaluate(() => window.translationTest.c.selection.content_block_id), "b");
  await page.evaluate(() => { const t = window.translationTest; t.c.open(t.blocks[0], "translate"); });
  await page.evaluate(() => {
    const t = window.translationTest, read = t.reads.at(-1);
    t.saved = t.result({ command: { ...read.target, target_language: "fr", mutation_id: "stored" } }).record;
    read.resolve({ ...t.readResponse(read.target), default_source_language: null, translations: [t.saved] });
  });
  await page.waitForFunction(() => window.translationTest.c.status === "ready");
  await page.evaluate(() => window.translationTest.c.setTo("fr"));
  await page.waitForFunction(() => window.translationTest.c.canSubmit);
  await page.evaluate(() => { void window.translationTest.c.submit(); });
  await page.waitForFunction(() => window.translationTest.c.surface === "saved");
  assert.equal(await page.evaluate(() => window.translationTest.writes.length), 0, "Saved target needs neither source default nor generation");
  assert.equal(await page.evaluate(() => window.translationTest.c.data.translations[0].mutation_id), "stored");
  await page.evaluate(() => { const t = window.translationTest; t.c.setFrom("de"); t.c.setTo("fr"); });
  await page.evaluate(() => { void window.translationTest.c.submit(); });
  assert.equal(await page.evaluate(() => window.translationTest.writes.length), 0, "Changing From does not regenerate a saved To");
  await page.evaluate(() => window.translationTest.c.setInlineLanguage("fr", true));
  await page.waitForFunction(() => window.translationTest.c.inlineLanguages.includes("fr"));
  await page.evaluate(() => window.translationTest.c.setInlineLanguage("fr", false));
  await page.waitForFunction(() => window.translationTest.c.inlineLanguages.length === 0);
  await page.evaluate(() => window.translationTest.c.setInlineLanguage("fr", true));
  await page.evaluate(() => window.translationTest.c.setFrom("fr"));
  await page.waitForFunction(() => window.translationTest.c.to === null);
  assert.equal(await page.evaluate(() => window.translationTest.c.canSubmit), false);
  await page.evaluate(() => { const t = window.translationTest; t.c.setTo("fr"); void t.c.submit(); });
  assert.equal(await page.evaluate(() => window.translationTest.writes.length), 0, "Saved target must not bypass equal-language guard");
  assert.deepEqual(await page.evaluate(() => window.translationTest.c.targetLanguageCodes), ["en", "de"]);
  await page.evaluate(() => window.translationTest.c.close());
  assert.equal(await page.evaluate(() => { const t = window.translationTest; return t.c.getInlineTranslations(t.blocks[0]).length; }), 1,
    "Closing the drawer retains inline selections");
  await page.evaluate(() => window.translationTest.c.invalidate({ translation_block_ids: ["a"] }));
  assert.equal(await page.evaluate(() => { const t = window.translationTest; return t.c.getInlineTranslations(t.blocks[0]).length; }), 0,
    "Server invalidation removes inline translations even when the drawer is closed");
  assert.deepEqual(await page.evaluate(() => window.translationTest.c.inlineLanguages), ["fr"], "Invalidation retains session preference");
  await page.evaluate(() => { const t = window.translationTest; t.c.open(t.blocks[0], "view"); });
  await page.evaluate(() => { const t = window.translationTest, read = t.reads.at(-1);
    read.resolve({ ...t.readResponse(read.target), translations: [t.saved] }); });
  await page.waitForFunction(() => window.translationTest.c.status === "ready");
  await page.evaluate(() => window.translationTest.c.setInlineLanguage("fr", true));
  await page.waitForFunction(() => window.translationTest.c.inlineLanguages.includes("fr"));
  await page.evaluate(() => { const t = window.translationTest; t.c.close(); t.originalBlocks = t.blocks;
    t.blocks = t.blocks.map(block => ({ ...block, content: "New source" })); t.render(); });
  assert.equal(await page.evaluate(() => { const t = window.translationTest; return t.c.getInlineTranslations(t.blocks[0]).length; }), 0,
    "Source edits hide obsolete inline translations immediately");
  await page.evaluate(() => { const t = window.translationTest; t.blocks = t.originalBlocks; t.render(); });
  assert.equal(await page.evaluate(() => { const t = window.translationTest; return t.c.getInlineTranslations(t.blocks[0]).length; }), 0,
    "Undo must not resurrect invalidated inline records");
  await ready();
  await page.evaluate(() => window.translationTest.c.setInlineLanguage("fr", false));
  await page.waitForFunction(() => window.translationTest.c.inlineLanguages.length === 0);
  await page.evaluate(() => { const t = window.translationTest; void t.c.submit(); void t.c.submit(); });
  await page.waitForFunction(() => window.translationTest.c.status === "generating");
  assert.equal(await page.evaluate(() => window.translationTest.c.surface), null);
  assert.equal(await page.evaluate(() => window.translationTest.writes.length), 1);
  await page.evaluate(() => { const t = window.translationTest; t.c.close(); t.writes[0].resolve(t.result(t.writes[0])); });
  await page.waitForTimeout(50);
  assert.equal(await page.evaluate(() => window.translationTest.c.surface), null);
  await ready();
  await page.evaluate(() => { const t = window.translationTest; void t.c.submit(); });
  await page.waitForFunction(() => window.translationTest.writes.length === 2);
  assert.equal(await page.evaluate(() => window.translationTest.writes[0].command.mutation_id),
    await page.evaluate(() => window.translationTest.writes[1].command.mutation_id), "Reconciled unknown retry reuses id");
  await page.evaluate(() => { const t = window.translationTest; t.writes[1].resolve(t.result(t.writes[1], "partial")); });
  await page.waitForFunction(() => window.translationTest.c.surface === "saved");
  assert.equal(await page.evaluate(() => window.translationTest.c.data.translations[0].status), "partial");
  await ready();
  await page.evaluate(() => { const t = window.translationTest; void t.c.submit(); });
  await page.waitForFunction(() => window.translationTest.writes.length === 3);
  await page.evaluate(() => { const t = window.translationTest; t.writes[2].resolve(t.result(t.writes[2], "untranslatable")); });
  await page.waitForFunction(() => window.translationTest.c.status === "failed");
  assert.equal(await page.evaluate(() => window.translationTest.c.surface), null);
  await ready();
  await page.evaluate(() => { const t = window.translationTest; void t.c.submit(); });
  await page.waitForFunction(() => window.translationTest.writes.length === 4);
  await page.evaluate(() => { const t = window.translationTest; t.writes[3].reject(new t.dto.TranslationRequestRejected("Rejected")); });
  await page.waitForFunction(() => window.translationTest.c.status === "failed");
  await ready();
  assert.equal(await page.evaluate(() => window.translationTest.c.canSubmit), true);
  await page.evaluate(() => { const t = window.translationTest; void t.c.submit(); });
  await page.waitForFunction(() => window.translationTest.writes.length === 5);
  await page.evaluate(() => { const t = window.translationTest; t.blocks = t.blocks.map(b => ({ ...b, content: "Changed" })); t.render(); t.writes[4].resolve(t.result(t.writes[4])); });
  await page.waitForTimeout(50);
  assert.equal(await page.evaluate(() => window.translationTest.c.surface), null);
  assert.equal(await page.evaluate(() => window.translationTest.writes[4].signal.aborted), true);
  await ready();
  await page.evaluate(() => { const t = window.translationTest; void t.c.submit(); });
  await page.waitForFunction(() => window.translationTest.writes.length === 6);
  await page.evaluate(() => { const t = window.translationTest; t.dispose(); t.writes[5].resolve(t.result(t.writes[5])); });
  assert.equal(await page.evaluate(() => window.translationTest.writes[5].signal.aborted), true);
  const checks = await page.evaluate(async () => {
    const t = window.translationTest, results = [], request = { doc_name: "book", content_block_id: "a" };
    const throws = async action => { try { await action(); return false; } catch { return true; } };
    let sent, value = t.readResponse(request);
    const service = new t.dto.TranslationService({ requestJsonWithStatus: async (path, options) => {
      sent = { path, options }; return { ok: true, status: 200, payload: value };
    } });
    const abort = new AbortController();
    await service.read({ ...request, text: "must not leak" }, abort.signal);
    results.push(sent.options.signal === abort.signal, JSON.stringify(JSON.parse(sent.options.body)) === JSON.stringify(request));
    results.push(t.model.profileFrom({ ...value, default_source_language: "unknown" }) === null);
    results.push(!t.model.validLanguages(["en"], "en", "freeform"));
    results.push(!t.model.validLanguages(["en"], "en", "en"));
    sent = null;
    results.push(await throws(() => service.generate({ ...request, source_language: "en", target_language: "en" })));
    results.push(sent === null);
    value = { ...value, target: { ...value.target, content_block_id: "wrong" } };
    results.push(await throws(() => service.read(request)));
    value = t.result({ command: { ...request, target_language: "fr", mutation_id: "id" } }, "untranslatable");
    results.push((await service.generate({ ...request, target_language: "fr" })).record === null);
    value = t.result({ command: { ...request, target_language: "fr", mutation_id: "id" } });
    results.push((await service.generate({ ...request, target_language: "fr", source_language: "de", mutation_id: "new" })).record.mutation_id === "id");
    value.record.translated_text = "Mismatch";
    results.push(await throws(() => service.generate({ ...request, target_language: "fr" })));
    for (const [status, payload, rejected] of [
      [502, { detail: "translation_provider_failed" }, true],
      [502, { detail: "translation_invalid_output" }, true],
      [503, { detail: "translation_capacity_unavailable" }, true],
      [503, { detail: "translation_profile_unavailable" }, true],
      [503, { detail: "translation_service_unavailable" }, false],
      [502, "Bad gateway", false], [504, "Gateway timeout", false],
      [408, {}, false], [422, { detail: "translation_request_invalid" }, true],
      [422, { detail: "translation_same_language" }, true],
    ]) {
      const failing = new t.dto.TranslationService({ requestJsonWithStatus: async () => ({ ok: false, status, payload }) });
      try { await failing.generate({ ...request, target_language: "fr" }); results.push(false); }
      catch (error) { results.push((error.name === "TranslationRequestRejected") === rejected); }
    }
    return results;
  });
  assert.ok(checks.every(Boolean));
  assert.deepEqual(errors, []);
  console.log("PASS: StrictMode lifecycle/cancel/out-of-order/duplicate/unknown replay/partial/untranslatable/failure/edit/unmount and DTO/model contracts");
} finally { await browser.close(); }
