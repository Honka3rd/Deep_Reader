// Actual App/routes and disposable PostgreSQL. Preview PNG is a UI-only fixture.
import assert from "node:assert/strict";
const api = process.env.READING_TEST_API;
assert.match(api || "", /^http:\/\/127\.0\.0\.1:\d+$/);
const { chromium } = await import(process.env.PLAYWRIGHT_MODULE || "playwright");
const browser = await chromium.launch({ headless: true,
  ...(process.env.CHROMIUM_PATH ? { executablePath: process.env.CHROMIUM_PATH } : {}) });
try {
  for (const width of [1280, 390]) {
    const page = await browser.newPage({ viewport: { width, height: 900 } });
    const screenshotPrefix = "/tmp/source-review-" + (process.env.READING_TEST_REAL_CASE || "synthetic");
    let capturedBody = false;
    const errors = [], requests = [];
    let failNext = false;
    page.on("pageerror", error => errors.push(error.message));
    await page.route("**/api/**", async route => {
      const url = new URL(route.request().url());
      const path = url.pathname.replace(/^\/api/, "");
      if (["/documents", "/documents/management"].includes(path)) return route.fulfill({ json: { items: [], total: 0 } });
      requests.push({ path, method: route.request().method() });
      assert.ok(!/prepare|reparse|smart-cleanup/.test(path), "No implicit preparation or model workflow");
      if (path.endsWith("/image")) return route.fulfill({ contentType: "image/png",
        body: Buffer.from("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jRZkAAAAASUVORK5CYII=", "base64") });
      if (failNext && route.request().method() === "POST" && path.endsWith("reading-content")) {
        failNext = false; return route.abort();
      }
      await route.fulfill({ response: await route.fetch({ url: api + path + url.search }) });
    });
    assert.equal((await page.request.post(api + "/fixture")).status(), 204);
    const stats = async () => (await page.request.get(api + "/stats")).json();
    const originalStats = await stats();
    const regionId = originalStats.region_id;
    assert.ok(regionId);
    const read = async () => (await page.request.get(api + "/documents/book/reading-content?region_id=" + encodeURIComponent(regionId))).json();
    const initial = await read();
    assert.ok(initial.regions.some(region => region.id === regionId && region.body_status === "linked"));
    let blockId = initial.blocks.find(block => !block.deleted).block_id;
    const originalText = initial.blocks.find(block => block.block_id === blockId).content;
    const corrected = suffix => originalStats.evidence.kind === "synthetic_source" ? suffix : originalText + "\n" + suffix;
    assert.ok(Array.from(corrected("Draft survives failed save")).length <= 10000);
    const readBlock = async () => (await page.request.get(api + "/documents/book/reading-content?block_id=" + encodeURIComponent(blockId))).json();
    const layoutReads = () => requests.filter(r => r.path === "/documents/task-layout").length;
    const writes = () => requests.filter(r => r.method === "POST" && r.path.endsWith("/reading-content")).length;
    await page.goto((process.env.UI_URL || "http://127.0.0.1:5173") + "/documents/book");
    const review = page.locator('div[aria-labelledby="reading-content-dialog-title"]');
    const region = () => review.locator('.source-review-region[data-region-id="' + regionId + '"]');
    const relatedReady = async () => {
      await review.getByLabel("Loading content changes").waitFor({ state: "hidden" });
      await region().locator('.source-review-blocks[aria-busy="false"]').waitFor();
    };
    const openReview = async () => {
      await page.getByRole("button", { name: "Review extracted text", exact: true }).click();
      await review.getByRole("tab", { name: "Flagged areas", exact: true }).click();
      const toggle = region().getByRole("button", { name: /^Related blocks/ });
      await toggle.waitFor();
      if (await toggle.getAttribute("aria-expanded") !== "true") await toggle.click();
      await relatedReady();
      assert.equal(await review.getByRole("button", { name: /Actions for block/ }).count(), 0, "Review never exposes editing");
      assert.equal(await review.getByLabel("Block text").count(), 0);
    };
    const selectBodyAction = async action => {
      await region().getByRole("button", { name: "Go to content", exact: true }).click();
      await review.waitFor({ state: "hidden" });
      // Use response identity, including punctuation in opaque IDs.
      const target = page.locator('.reader-content-block-list [data-block-id]').filter({ hasText: blockId });
      await target.waitFor();
      assert.equal(await target.getAttribute("data-block-id"), blockId);
      if (!capturedBody) {
        await page.screenshot({ path: screenshotPrefix + "-body-" + width + ".png", animations: "disabled" });
        capturedBody = true;
      }
      await target.getByRole("button", { name: "Content-block reading interactions for " + blockId, exact: true }).click();
      await page.getByRole("menuitem", { name: action, exact: true }).click();
    };
    const assertManual = async status => {
      const next = await read();
      const current = next.regions.find(r => r.id === regionId);
      assert.ok(current, "Body mutations retain the review region");
      assert.equal(current.manual_change.status, status);
      assert.ok(Number.isFinite(Date.parse(current.manual_change.at)));
      await region().getByText("Manually " + status, { exact: true }).waitFor();
      assert.equal(await region().locator("time").getAttribute("datetime"), current.manual_change.at);
      return next;
    };
    const edit = async (text, clear) => {
      await selectBodyAction("Edit");
      await page.getByLabel("Block text").fill(text);
      const before = layoutReads(), beforeWrites = writes();
      await page.getByRole("button", { name: "Save", exact: true }).click();
      const confirm = page.getByRole("dialog", { name: "Keep this block's interactions?" });
      assert.match(await confirm.innerText(), /Notes/);
      await confirm.getByRole("button", { name: clear ? "Delete interactions and save" : "Keep and save", exact: true })
        .evaluate(el => { el.click(); el.click(); });
      await review.waitFor({ state: "hidden" });
      assert.equal(layoutReads() - before, 1, "Exactly one layout refresh per body save");
      assert.equal(writes() - beforeWrites, 1, "Double click submits one mutation");
      await openReview();
      const next = await assertManual("modified");
      assert.equal(next.blocks.find(b => b.block_id === blockId).content, text);
      await region().getByText(text, { exact: true }).waitFor();
    };
    await openReview();
    await page.screenshot({ path: screenshotPrefix + "-blocks-" + width + ".png", animations: "disabled" });
    await edit(corrected("Reviewed OCR text"), false);
    assert.equal((await stats()).note_count, 1);
    await edit(corrected("Second correction"), true);
    assert.equal((await stats()).note_count, 0);
    await edit(corrected("Third correction"), false);
    await selectBodyAction("Edit");
    await page.getByLabel("Block text").fill(corrected("Draft survives failed save"));
    failNext = true;
    await page.getByRole("button", { name: "Save", exact: true }).click();
    await page.getByRole("button", { name: "Keep and save", exact: true }).click();
    await page.getByText("The change could not be confirmed.", { exact: false }).waitFor();
    assert.equal(await page.getByLabel("Block text").inputValue(), corrected("Draft survives failed save"));
    const retryBefore = layoutReads();
    await page.getByRole("button", { name: "Save", exact: true }).click();
    await page.getByRole("button", { name: "Keep and save", exact: true }).click();
    await review.waitFor({ state: "hidden" });
    assert.equal(layoutReads() - retryBefore, 1);
    await openReview();
    let lastDeletedText;
    // Delete each verified target through the body editor, including multi-block regions.
    for (let remaining = await read(); remaining.blocks.length; remaining = await read()) {
      const block = remaining.blocks.find(b => !b.deleted);
      if (!block) break;
      blockId = block.block_id; lastDeletedText = block.content;
      await selectBodyAction("Delete");
      await page.getByText("Delete this block and its interactions", { exact: false }).waitFor();
      const deleteBefore = layoutReads();
      await page.getByRole("button", { name: "Delete block", exact: true }).click();
      await review.waitFor({ state: "hidden" });
      assert.equal(layoutReads() - deleteBefore, 1);
      await openReview();
    }
    let state = await assertManual("deleted");
    assert.equal((await readBlock()).blocks.find(b => b.block_id === blockId).deleted, true);
    assert.ok(state.regions.find(r => r.id === regionId).related_deleted_count > 0);
    assert.equal(await region().getByRole("button", { name: "Go to content", exact: true }).isDisabled(), true);
    await page.screenshot({ path: screenshotPrefix + "-delete-" + width + ".png", animations: "disabled" });
    const undoBefore = layoutReads();
    await review.getByRole("button", { name: "Undo last content change" }).click();
    await relatedReady();
    state = await read();
    assert.equal(layoutReads() - undoBefore, 1);
    assert.equal(state.blocks.find(b => b.block_id === blockId).deleted, false);
    assert.equal(state.blocks.find(b => b.block_id === blockId).content, lastDeletedText);
    assert.equal((await stats()).note_count, 0, "Undo restores text, never cleared Note data");
    // A browser restart must read persisted manual history and current text.
    await page.reload(); await openReview();
    state = await read();
    const manualBeforeIgnore = state.regions.find(r => r.id === regionId).manual_change;
    assert.equal(await region().locator("time").getAttribute("datetime"), manualBeforeIgnore.at);
    const ignore = region().getByRole("button", { name: "Ignore region", exact: true });
    assert.equal(await ignore.isDisabled(), true, "Original must be loaded before Ignore");
    await region().getByRole("button", { name: "Original", exact: true }).click();
    await page.waitForFunction(() => Array.from(document.querySelectorAll("button")).some(b => b.textContent === "Ignore region" && !b.disabled));
    await ignore.click();
    const confirm = page.getByRole("dialog", { name: "Ignore this OCR region?" });
    await confirm.getByText("Existing body content and generation input stay unchanged.", { exact: false }).waitFor();
    await confirm.getByText("To delete content, use Delete in the body block menu.", { exact: false }).waitFor();
    await confirm.getByRole("button", { name: "Cancel", exact: true }).click();
    assert.equal((await read()).regions.length, initial.regions.length, "Cancel keeps the region");
    await ignore.click();
    const bodyBeforeIgnore = (await readBlock()).blocks;
    const ignoreBefore = layoutReads(), ignoreWrites = writes();
    await confirm.getByRole("button", { name: "Ignore region", exact: true }).evaluate(el => { el.click(); el.click(); });
    await region().waitFor({ state: "hidden" });
    await review.getByLabel("Loading content changes").waitFor({ state: "hidden" });
    assert.equal(layoutReads() - ignoreBefore, 1);
    assert.equal(writes() - ignoreWrites, 1);
    const ignored = (await (await page.request.get(api + "/documents/book/reading-content")).json()).regions;
    assert.equal(ignored.length, initial.regions.length - 1);
    assert.equal(ignored.some(r => r.id === regionId), false);
    assert.deepEqual((await readBlock()).blocks, bodyBeforeIgnore, "Ignore acknowledges review without changing retained body");
    const undoIgnoreBefore = layoutReads();
    await review.getByRole("button", { name: "Undo last content change" }).click();
    await relatedReady();
    assert.equal(layoutReads() - undoIgnoreBefore, 1);
    state = await read();
    assert.deepEqual(state.regions.find(r => r.id === regionId).manual_change, manualBeforeIgnore, "Ignore and its Undo do not fabricate manual timestamps");
    const finalStats = await stats();
    assert.equal(finalStats.note_count, 0);
    assert.equal(finalStats.model_calls, 0); assert.equal(finalStats.prepare_calls, 0);
    assert.equal(finalStats.source_sha256, originalStats.source_sha256, "Source remains immutable");
    assert.equal(finalStats.canonical_sha256, originalStats.canonical_sha256, "Canonical source remains immutable");
    assert.deepEqual(errors, []);
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth), false);
    await page.unrouteAll({ behavior: "wait" }); await page.close();
    console.log("PASS " + width + ": readonly review/body navigation, keep/clear Note, manual timestamps, retry, deletion retained, explicit Ignore, text-only Undo, one layout refresh per mutation");
  }
} finally { await browser.close(); }
