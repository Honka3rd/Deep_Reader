// Actual App and bounded task-unit pagination; fixture APIs only, no backend writes.
import assert from "node:assert/strict";
const { chromium } = await import(process.env.PLAYWRIGHT_MODULE || "playwright");
const browser = await chromium.launch({ headless: true,
  ...(process.env.CHROMIUM_PATH ? { executablePath: process.env.CHROMIUM_PATH } : {}) });
const sections = ["Many", "Single", "Empty", "Failed"].map(name => ({
  section_id: name, title: name,
  task_units: Array.from({ length: name === "Many" ? 20 : 1 }, (_, i) => ({ unit_id: `${name}-${i + 1}` })),
}));
try {
  for (const width of [1280, 390, 320]) {
    const page = await browser.newPage({ viewport: { width, height: 900 } });
    const errors = [], unexpected = [];
    let contentReads = 0, hold = false, held;
    page.on("pageerror", error => errors.push(error.message));
    await page.route("**/api/**", async route => {
      const path = new URL(route.request().url()).pathname;
      if (path === "/api/documents/reparse-structure/status") return route.fulfill({ json: { doc_name: "Book", status: "idle" } });
      if (path === "/api/documents/task-layout") return route.fulfill({ json: {
        document_id: "Book", title: "Book", chapters: [{ chapter_id: "chapter", title: "Chapter", sections }],
      } });
      if (path === "/api/documents/Book/reading-content") return route.fulfill({ json: {
        revision: 0, source_token: "fixture", disposition: "accepted", can_undo: false,
        max_chars: 10000, regions: [], blocks: [],
      } });
      if (path === "/api/documents/Book/task-units/content") {
        contentReads++;
        const ids = route.request().postDataJSON().task_unit_ids;
        if (hold) await new Promise(resolve => { held = resolve; });
        if (ids[0].startsWith("Failed")) return route.fulfill({ status: 503, json: { detail: "Unavailable fixture" } });
        return route.fulfill({ json: { contents: ids.map(id => ({
          document_id: "Book", task_unit_id: id, chapter_id: "chapter", section_id: id.split("-")[0],
          title: id, content_blocks: id.startsWith("Empty") ? [] : [{
            block_id: `${id}:content:0`, block_type: "paragraph",
            content: `${id} text. ` + "A paragraph of reading content with a stable identity. ".repeat(5),
          }],
        })) } });
      }
      unexpected.push(path);
      return route.fulfill({ status: 500, json: { detail: "Unexpected API" } });
    });
    await page.goto(`${process.env.UI_URL || "http://127.0.0.1:5173"}/documents/Book`);
    await page.locator('[data-item-id="chapter:chapter"] .hierarchy-tree-icon').click();
    const select = name => page.locator(".hierarchy-section-button").filter({ hasText: name }).click();
    const bar = page.locator(".reader-content-pagination-bar");
    const first = page.locator(".reader-content-page-first");
    const last = page.locator(".reader-content-page-last");
    const previous = page.locator(".reader-content-page-previous");
    const next = page.locator(".reader-content-page-next");
    const pageNumbers = async () => (await bar.innerText()).match(/Page (\d+) \/ (\d+)/).slice(1).map(Number);
    await select("Many");
    await page.waitForFunction(() => {
      const match = document.querySelector(".reader-content-pagination-bar")?.textContent.match(/Page (\d+) \/ (\d+)/);
      return match && Number(match[2]) > 2;
    });
    assert.deepEqual(await page.locator(".reader-content-page-controls").evaluate(el =>
      Array.from(el.children).map(child => Array.from(child.classList).find(name => name.startsWith("reader-content-page-")))),
    ["reader-content-page-first", "reader-content-page-previous", "reader-content-page-next", "reader-content-page-last"]);
    assert.equal(await first.isDisabled(), true);
    assert.deepEqual(await pageNumbers(), [1, 20]);
    assert.equal(await page.locator(".reader-content-task-unit-group").count(), 1);
    assert.equal(await page.locator(".reader-content-pagination-measure").count(), 0);
    assert.equal(await previous.isDisabled(), true);
    assert.equal(await last.getAttribute("aria-label"), "Last page");
    assert.equal(await first.getAttribute("title"), "First page");
    const readsBeforeNavigation = contentReads;
    await last.focus();
    await page.keyboard.press("Enter");
    await page.waitForFunction(() => document.querySelector(".reader-content-page-last")?.disabled);
    let [current, total] = await pageNumbers();
    assert.equal(current, total);
    assert.equal(await next.isDisabled(), true);
    assert.match(await page.locator(".reader-content-block-list").innerText(), /Many-20 text/);
    await previous.click();
    assert.equal((await pageNumbers())[0], total - 1);
    await first.focus();
    await page.keyboard.press("Space");
    await page.waitForFunction(() => document.querySelector(".reader-content-page-first")?.disabled);
    assert.equal((await pageNumbers())[0], 1);
    assert.match(await page.locator(".reader-content-block-list").innerText(), /Many-1 text/);
    await next.click();
    assert.equal((await pageNumbers())[0], 2);
    await last.click();
    await page.setViewportSize({ width, height: 760 });
    await page.waitForTimeout(250);
    await first.click();
    await last.click();
    [current, total] = await pageNumbers();
    assert.equal(current, total, "Last page retains the task-unit page count after resize");
    assert.equal(total, 20);
    assert.equal(await page.locator(".reader-content-task-unit-group").count(), 1);
    assert.equal(contentReads, readsBeforeNavigation, "Pagination must not fetch or prepare content");
    const geometry = await page.locator(".reader-content-page-controls").evaluate(el => {
      const parent = el.parentElement.getBoundingClientRect();
      return Array.from(el.children).map(child => {
        const r = child.getBoundingClientRect();
        return r.left >= parent.left - 1 && r.right <= parent.right + 1 && r.top >= parent.top - 1 && r.bottom <= parent.bottom + 1;
      });
    });
    assert.ok(geometry.every(Boolean), "All pagination buttons fit inside their bar");
    await bar.scrollIntoViewIfNeeded();
    await page.screenshot({ path: `/tmp/reader-pagination-${width}.png`, fullPage: true });
    await select("Single");
    await page.waitForFunction(() => document.querySelector(".reader-content-pagination-bar")?.textContent.includes("Page 1 / 1"));
    for (const button of [first, previous, next, last]) assert.equal(await button.isDisabled(), true);
    hold = true;
    await select("Many");
    await page.locator(".reader-content-blocks-loading-state").waitFor();
    assert.equal(await bar.count(), 0, "Loading state must not expose stale navigation");
    while (!held) await page.waitForTimeout(10);
    hold = false;
    held();
    await first.waitFor();
    assert.equal((await pageNumbers())[0], 1, "Changing section resets to the first page");
    await select("Empty");
    await page.locator(".reader-content-blocks-empty-state").waitFor();
    assert.equal(await bar.count(), 0);
    await select("Failed");
    await page.locator(".reader-content-blocks-error-state").waitFor();
    assert.equal(await bar.count(), 0);
    assert.deepEqual(errors, []);
    assert.deepEqual(unexpected, []);
    await page.close();
    console.log(`PASS ${width}px: direct-child order, first/last and adjacent navigation, keyboard, resize, bounds, reset, single/empty/loading/error, zero extra reads`);
  }
} finally { await browser.close(); }
