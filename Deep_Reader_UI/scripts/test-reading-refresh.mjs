// Mounted reader controller and pure merge contract; controlled HTTP, no production writes.
import assert from "node:assert/strict";
const { chromium } = await import(process.env.PLAYWRIGHT_MODULE || "playwright");
const browser = await chromium.launch({ headless: true,
  ...(process.env.CHROMIUM_PATH ? { executablePath: process.env.CHROMIUM_PATH } : {}) });
try {
  const page = await browser.newPage();
  const errors = [];
  page.on("pageerror", error => errors.push(error.message));
  const binding = { document_id: "Book", revision: 2, mutation_revision: 2, structure_version: "1", source_token: "source",
    mutation_id: "mutation", scope_current: true, generated_invalidated: true,
    block_page: { total: 25, offset: 0, limit: 20 }, blocks: [] };
  const rows = Array.from({ length: 25 }, (_, i) => ({ block_id: `b${i}`, task_unit_id: "u", chapter_id: "c", section_id: "s",
    block_order: i, insert_before_block_id: "tail", unit_source_hash: "fresh", deleted: false, manually_edited: false,
    content_block: { block_id: `b${i}`, content: `Restored ${i}`, block_type: "paragraph" } }));
  binding.blocks = rows.slice(0, 20);
  let held = null, hold = false, stale = false, fail = false;
  const requests = [];
  await page.route("**/api/**", route => {
    const url = new URL(route.request().url());
    if (url.searchParams.has("refresh_mutation_id")) {
      requests.push(url.searchParams.get("block_offset"));
      const offset = Number(url.searchParams.get("block_offset"));
      const json = { ...binding, block_page: { ...binding.block_page, offset }, blocks: rows.slice(offset, offset + 20) };
      if (stale && offset) json.revision++;
      if (fail && offset) return route.fulfill({ status: 503, json: { detail: "Read failed" } });
      if (hold && offset) { held = { route, json }; return; }
      return route.fulfill({ json });
    }
    return route.fulfill({ json: { documents: [] } });
  });
  await page.goto(process.env.UI_URL || "http://127.0.0.1:5173");
  await page.evaluate(async binding => {
    const { default: React } = await import("/node_modules/.vite/deps/react.js");
    const { default: ReactDOMClient } = await import("/node_modules/.vite/deps/react-dom_client.js");
    const { default: ReactDOM } = await import("/node_modules/.vite/deps/react-dom.js");
    const { useReaderContentController } = await import("/src/features/reader-content/controller.ts");
    const model = await import("/src/features/reader-content/readingRefreshModel.ts");
    const test = window.refreshTest = { binding, model, contents: 0, done: false, error: null, docName: "Book" };
    const section = { section_id: "s", task_units: [{ unit_id: "u" }] };
    const chapter = { chapter_id: "c", sections: [section] };
    const layout = test.layout = { document_id: "Book", chapters: [chapter] };
    const service = { fetchTaskUnitContents: async () => { test.contents++; return [{ document_id: "Book", task_unit_id: "u",
      chapter_id: "c", section_id: "s", content_blocks: [{ block_id: "tail", content: "Unchanged tail",
        metadata: { source_hash: "old" }, interaction_availability: { insight: { status: "completed", artifact_id: "stale" } } }] }]; } };
    const container = document.createElement("div"); document.body.append(container);
    const root = ReactDOMClient.createRoot(container);
    function Probe() { test.reader = useReaderContentController({ docName: test.docName, layout, service }); return null; }
    test.render = name => { test.docName = name; ReactDOM.flushSync(() => root.render(React.createElement(Probe))); };
    test.render("Book");
    test.select = () => test.reader.selectSection({ chapter, section, taskUnits: section.task_units });
    test.run = () => {
      test.done = false; test.error = null; test.abort = new AbortController();
      test.reader.refreshAffectedBlocks(binding, test.abort.signal)
        .catch(error => { test.error = error.message; }).finally(() => { test.done = true; });
    };
    test.dispose = () => root.unmount();
  }, binding);
  const select = async () => { await page.evaluate(() => window.refreshTest.select());
    await page.waitForFunction(() => window.refreshTest.reader.contentStatus === "success"); };
  const run = async () => { await page.evaluate(() => window.refreshTest.run()); await page.waitForFunction(() => window.refreshTest.done); };
  await select(); await run();
  assert.deepEqual(requests, ["0", "20"]);
  assert.deepEqual(await page.evaluate(() => ({ error: window.refreshTest.error, contents: window.refreshTest.contents,
    ids: window.refreshTest.reader.contentBlocks.map(b => b.block_id),
    hash: window.refreshTest.reader.contentBlocks.at(-1).ownerContext.sourceHash,
    availability: window.refreshTest.reader.contentBlocks.at(-1).interaction_availability ?? null })),
  { error: null, contents: 1, ids: [...rows.map(r => r.block_id), "tail"], hash: "fresh", availability: null });
  await run();
  assert.equal(await page.evaluate(() => window.refreshTest.reader.contentBlocks.length), 26, "Replay never duplicates blocks");
  for (const scenario of ["stale", "failure", "abort", "switch", "selection", "unmount"]) {
    await select(); stale = scenario === "stale"; fail = scenario === "failure";
    hold = ["abort", "switch", "selection", "unmount"].includes(scenario); held = null;
    await page.evaluate(() => window.refreshTest.run());
    if (hold) {
      for (let n = 0; n < 100 && !held; n++) await page.waitForTimeout(10);
      assert.ok(held);
      assert.equal(await page.evaluate(() => window.refreshTest.reader.contentBlocks.length), 1, "No partial page publication");
      if (scenario === "abort") await page.evaluate(() => window.refreshTest.abort.abort());
      if (scenario === "switch") await page.evaluate(() => window.refreshTest.render("Other"));
      if (scenario === "selection") await select();
      if (scenario === "unmount") await page.evaluate(() => window.refreshTest.dispose());
      await held.route.fulfill({ json: held.json });
    }
    await page.waitForFunction(() => window.refreshTest.done);
    assert.ok(await page.evaluate(() => window.refreshTest.error));
    assert.equal(await page.evaluate(() => window.refreshTest.reader.contentBlocks.length), 1);
    if (scenario === "switch") await page.evaluate(() => window.refreshTest.render("Book"));
    hold = stale = fail = false;
  }
  const checks = await page.evaluate(({ rows, binding }) => {
    const { mergeReadingRefresh: merge, validateRefreshPage: validate } = window.refreshTest.model;
    const layout = window.refreshTest.layout;
    const group = { taskUnitId: "u", chapterId: "c", sectionId: "s", blocks: [{ block_id: "tail", content: "Tail",
      interaction_availability: { insight: { status: "completed" } }, ownerContext: { taskUnitId: "u", sourceHash: "old" } }] };
    const one = { ...binding, generated_invalidated: false, block_page: { total: 1, offset: 0, limit: 20 }, blocks: [rows[0]] };
    const checks = [];
    const rejects = action => { try { action(); return false; } catch { return true; } };
    checks.push(rejects(() => merge([group], layout, binding, [...rows.slice(1), rows[1]])));
    checks.push(rejects(() => merge([group], layout, one, [{ ...rows[0], section_id: "wrong" }])));
    checks.push(rejects(() => merge([group], layout, one, [{ ...rows[0], insert_before_block_id: "missing" }])));
    checks.push(rejects(() => merge([group], layout, one, [{ ...rows[0], content_block: { block_id: "wrong", content: "X" } }])));
    checks.push(rejects(() => validate({ ...binding, scope_current: false }, binding, 0)));
    checks.push(rejects(() => validate({ ...binding, source_token: "new" }, binding, 0)));
    checks.push(rejects(() => validate({ ...binding, blocks: [] }, binding, 0)));
    const merged = merge([group], layout, one, one.blocks);
    checks.push(merged[0].blocks[1].interaction_availability.insight.status === "completed");
    checks.push(merged[0].blocks[0].metadata.manually_edited === false);
    const tombstone = { ...rows[0], content_block: null, deleted: true };
    checks.push(merge(merged, layout, one, [tombstone])[0].blocks.map(b => b.block_id).join() === "tail");
    checks.push(merge([group], layout, { ...binding, block_page: { total: 0, offset: 0, limit: 20 } }, [])[0].blocks[0].content === "Tail");
    const secondGroup = { taskUnitId: "v", chapterId: "c", sectionId: "s", blocks: [] };
    const crossLayout = { document_id: "Book", chapters: [{ chapter_id: "c", sections: [{ section_id: "s",
      task_units: [{ unit_id: "u" }, { unit_id: "v" }] }] }] };
    const second = { ...rows[1], task_unit_id: "v", block_order: 0, insert_before_block_id: null };
    const cross = { ...binding, block_page: { total: 2, offset: 0, limit: 20 } };
    const both = merge([group, secondGroup], crossLayout, cross, [rows[0], second]);
    checks.push(both[0].blocks[0].block_id === "b0" && both[1].blocks[0].block_id === "b1");
    checks.push(merge([group], crossLayout, one, [second])[0] === group); // Unloaded units are not fabricated.
    return checks;
  }, { rows, binding });
  assert.ok(checks.every(Boolean), JSON.stringify(checks)); assert.deepEqual(errors, []);
  const historyChecks = await page.evaluate(async binding => {
    const { mergeCleanupChange: merge } = await import("/src/features/reader-content/cleanupHistoryModel.ts");
    const id = "a".repeat(64), other = { record_id: "b".repeat(64), quote: "other", text: "other", page_indices: [], reason: "ocr_garbage" };
    const row = { ...other, record_id: id }, replacement = { ...row, text: "saved" };
    const batch = { revision: 1, deletions: [row, other], applied_count: 2, remaining_count: 2 };
    const untouched = { ...batch, revision: 0, deletions: [] };
    const history = { total: 2, offset: 0, limit: 10, items: [batch, untouched] };
    const refresh = { ...binding, cleanup_record_id: id, cleanup_change: { batch_revision: 1, record_id: id,
      deleted: false, record: replacement, applied_count: 2, remaining_count: 2 } };
    const result = merge(history, refresh, id, false), checks = [];
    checks.push(result.items[1] === untouched, result.items[0].deletions[1] === other,
      result.items[0].deletions[0] === replacement, batch.deletions[0] === row);
    const rejects = fn => { try { fn(); return false; } catch { return true; } };
    for (const patch of [{ scope_current: false }, { mutation_revision: 1 }, { cleanup_record_id: "wrong" },
      { cleanup_change: { ...refresh.cleanup_change, applied_count: 1 } },
      { cleanup_change: { ...refresh.cleanup_change, batch_revision: 99 } },
      { cleanup_change: { ...refresh.cleanup_change, deleted: true } }]) {
      checks.push(rejects(() => merge(history, { ...refresh, ...patch }, id, false)));
    }
    const deleted = merge(history, { ...refresh, cleanup_change: { ...refresh.cleanup_change,
      deleted: true, record: null, applied_count: 1, remaining_count: 1 } }, id, true);
    checks.push(deleted.items[0].deletions.length === 1, deleted.items[0].deletions[0] === other);
    return checks;
  }, binding);
  assert.ok(historyChecks.every(Boolean), JSON.stringify(historyChecks));
  console.log(`PASS paginated mounted refresh/replay/atomic publication, 6 stale/lifecycle cases and ${checks.length} merge/contract checks`);
  console.log(`PASS ${historyChecks.length} history delta identity, immutability and stale-response checks`);
} finally { await browser.close(); }
