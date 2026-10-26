// Held reads exercise bounded background loading and source/scope race protection.
import assert from "node:assert/strict";
const { chromium } = await import(process.env.PLAYWRIGHT_MODULE || "playwright");
const browser = await chromium.launch({ headless: true, channel: "chrome" });
try {
  const page = await browser.newPage();
  const errors = [];
  page.on("pageerror", error => errors.push(error.message));
  await page.route("**/api/**", route => route.fulfill({ json: { documents: [] } }));
  await page.goto(process.env.UI_URL || "http://127.0.0.1:5173");
  await page.evaluate(async () => {
    const { default: React } = await import("/node_modules/.vite/deps/react.js");
    const { default: Client } = await import("/node_modules/.vite/deps/react-dom_client.js");
    const { default: DOM } = await import("/node_modules/.vite/deps/react-dom.js");
    const { useBlockTranslation } = await import("/src/features/block-translation/controller.ts");
    const t = window.sessionTest = { doc: "Book", scope: "one", reads: [], writes: 0 };
    t.blocks = ["a", "b", "c", "d", "e"].map(block_id => ({ block_id, content: "Original", ownerContext: { taskUnitId: "unit", sourceHash: "source" } }));
    const service = { read: (target, signal) => new Promise((resolve, reject) => t.reads.push({ target, signal, resolve, reject })),
      generate: async () => { t.writes++; throw new Error("Unexpected generation"); } };
    const element = document.createElement("div"); document.body.append(element); const root = Client.createRoot(element);
    function Probe() { t.c = useBlockTranslation({ docName: t.doc, scopeKey: t.scope, blocks: t.blocks, service }); return null; }
    t.render = () => DOM.flushSync(() => root.render(React.createElement(React.StrictMode, null, React.createElement(Probe))));
    t.render(); t.dispose = () => root.unmount();
    t.finish = (index, text = "Saved", failed = false) => {
      const read = t.reads[index]; read.finished = true;
      if (failed) { read.reject(new Error("Unavailable")); return; }
      read.resolve({ target: { ...read.target, document_id: read.target.doc_name, target_type: "content_block", target_id: read.target.content_block_id },
        language_codes: ["en", "fr"], default_source_language: "en", effective_content_hash: "hash", reading_revision: 0,
        translations: [{ source_language: "en", target_language: "fr", translated_text: text, status: "completed", effective_content_hash: "hash",
          revision: 1, mutation_id: "id", generated_at: "2026-10-26T12:00:00Z", updated_at: "2026-10-26T12:00:00Z" }] });
    };
  });
  await page.evaluate(() => { const t = window.sessionTest; t.c.open(t.blocks[0], "view"); });
  await page.evaluate(() => window.sessionTest.finish(0));
  await page.waitForFunction(() => window.sessionTest.c.status === "ready");
  await page.evaluate(() => window.sessionTest.c.setInlineLanguage("fr", true));
  await page.evaluate(() => { const t = window.sessionTest; t.c.close(); t.c.onVisibleTranslationBlocksChange(t.blocks); });
  await page.waitForFunction(() => window.sessionTest.reads.length === 3);
  assert.equal(await page.evaluate(() => window.sessionTest.reads.filter(r => !r.finished && !r.signal.aborted).length), 2);
  await page.evaluate(() => window.sessionTest.finish(1));
  await page.waitForFunction(() => window.sessionTest.reads.length === 4);
  assert.equal(await page.evaluate(() => window.sessionTest.reads.filter(r => !r.finished && !r.signal.aborted).length), 2);
  // An explicit drawer result supersedes the older held background result for c.
  await page.evaluate(() => { const t = window.sessionTest; t.c.open(t.blocks[2], "view"); });
  await page.evaluate(() => { const t = window.sessionTest; t.finish(t.reads.length - 1, "New explicit result"); });
  await page.waitForFunction(() => window.sessionTest.c.status === "ready");
  assert.equal(await page.evaluate(() => window.sessionTest.reads[2].signal.aborted), true);
  await page.evaluate(() => window.sessionTest.finish(2, "STALE BACKGROUND"));
  assert.equal(await page.evaluate(() => { const t = window.sessionTest; return t.c.getInlineTranslations(t.blocks[2])[0].translated_text; }), "New explicit result");
  await page.evaluate(() => { const t = window.sessionTest; t.c.close(); t.oldReads = t.reads.length;
    t.doc = "Other"; t.scope = "two"; t.render(); });
  assert.deepEqual(await page.evaluate(() => window.sessionTest.c.inlineLanguages), ["fr"]);
  assert.equal(await page.evaluate(() => { const t = window.sessionTest; return t.c.getInlineTranslations(t.blocks[2]).length; }), 0);
  assert.equal(await page.evaluate(() => window.sessionTest.reads.slice(0, window.sessionTest.oldReads).filter(r => !r.finished && !r.signal.aborted).length), 0);
  await page.evaluate(() => { const t = window.sessionTest; t.c.onVisibleTranslationBlocksChange(t.blocks.slice(0, 1)); });
  await page.waitForFunction(() => window.sessionTest.reads.length > window.sessionTest.oldReads);
  await page.evaluate(() => { const t = window.sessionTest; t.finish(t.reads.length - 1, "", true); });
  await page.waitForFunction(() => { const t = window.sessionTest; return t.c.getInlineTranslationState(t.blocks[0]).status === "failed"; });
  const beforeRetry = await page.evaluate(() => window.sessionTest.reads.length);
  await page.evaluate(() => { const t = window.sessionTest; t.c.getInlineTranslationState(t.blocks[0]).onRetry(); });
  await page.waitForFunction(count => window.sessionTest.reads.length === count + 1, beforeRetry);
  await page.evaluate(() => { const t = window.sessionTest; t.finish(t.reads.length - 1, "Other book");
    for (let i = 0; i < t.oldReads; i++) if (!t.reads[i].finished) t.finish(i, "OLD BOOK"); });
  await page.waitForFunction(() => { const t = window.sessionTest; return t.c.getInlineTranslations(t.blocks[0])[0]?.translated_text === "Other book"; });
  await page.evaluate(() => { const t = window.sessionTest; t.c.onVisibleTranslationBlocksChange(t.blocks); });
  await page.waitForFunction(() => window.sessionTest.reads.filter(r => !r.finished && !r.signal.aborted).length === 2);
  await page.evaluate(() => { const t = window.sessionTest; t.c.invalidate({ translation_block_ids: ["a", "b"] }); });
  assert.deepEqual(await page.evaluate(() => window.sessionTest.c.inlineLanguages), ["fr"]);
  assert.equal(await page.evaluate(() => { const t = window.sessionTest; return t.c.getInlineTranslations(t.blocks[0]).length; }), 0);
  await page.evaluate(() => { const t = window.sessionTest;
    t.lateSourceReads = t.reads.map((read, index) => ({ read, index })).filter(({ read }) => !read.finished);
    t.blocks = t.blocks.map(block => ({ ...block, content: "Edited source" })); t.render(); });
  await page.evaluate(() => { const t = window.sessionTest;
    for (const { index } of t.lateSourceReads) t.finish(index, "OBSOLETE SOURCE"); });
  assert.equal(await page.evaluate(() => { const t = window.sessionTest; return t.blocks.some(block => t.c.getInlineTranslations(block).length); }), false,
    "Source changes reject late reads and do not start stale queued blocks");
  await page.evaluate(() => { const t = window.sessionTest; t.c.onVisibleTranslationBlocksChange([]); });
  await page.waitForFunction(() => window.sessionTest.reads.filter(r => !r.finished && !r.signal.aborted).length === 0);
  await page.evaluate(() => window.sessionTest.dispose());
  assert.equal(await page.evaluate(() => window.sessionTest.writes), 0); assert.deepEqual(errors, []);
  console.log("PASS: two-read concurrency, explicit result wins, cross-book cancellation/isolation, failure retry, preference retention, closed-drawer invalidation and page/unmount cleanup");
} finally { await browser.close(); }
