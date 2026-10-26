// Actual App with fixture reads; native fullscreen plus unavailable/rejected API paths.
import assert from "node:assert/strict";
const { chromium } = await import(process.env.PLAYWRIGHT_MODULE || "playwright");
const browser = await chromium.launch({ headless: true });
try {
  for (const width of [1280, 390]) for (const mode of ["native", "rejected", "unsupported", "deferred"]) {
    const page = await browser.newPage({ viewport: { width, height: 900 } });
    const errors = [], unexpected = [], requests = [];
    page.on("pageerror", error => errors.push(error.message));
    await page.addInitScript(mode => {
      window.storageWrites = [];
      const original = Storage.prototype.setItem;
      Storage.prototype.setItem = function(key, value) { window.storageWrites.push(key); return original.call(this, key, value); };
      if (mode === "unsupported") Object.defineProperty(document, "fullscreenEnabled", { value: false });
      if (mode === "rejected") Element.prototype.requestFullscreen = () => Promise.reject(new Error("Fullscreen denied"));
      if (mode === "deferred") Element.prototype.requestFullscreen = () => new Promise(resolve => { window.resolveFullscreen = resolve; });
    }, mode);
    await page.route("**/api/**", route => {
      const request = route.request(), url = new URL(request.url());
      const path = url.pathname; requests.push(path);
      if (path === "/api/documents/task-layout") return route.fulfill({ json: {
        document_id: request.postDataJSON().doc_name, title: "Book", chapters: [{ chapter_id: "c", title: "Chapter", sections: [{
          section_id: "s", title: "Body", task_units: ["u1", "u2", "u3"].map(unit_id => ({ unit_id })),
        }] }],
      } });
      if (path.endsWith("/reparse-structure/status")) return route.fulfill({ json: { status: "idle" } });
      if (path.endsWith("/reading-content")) return route.fulfill({ json: {
        revision: 0, source_token: "fixture", disposition: "accepted", can_undo: false, max_chars: 10000, regions: [], blocks: [],
        cleanup_audit: { total: 0, offset: 0, limit: 10, items: [] },
      } });
      if (path.endsWith("/task-units/content")) return route.fulfill({ json: {
        contents: request.postDataJSON().task_unit_ids.map(id => ({ document_id: "Book", task_unit_id: id,
          chapter_id: "c", section_id: "s", title: id,
          content_blocks: Array.from({ length: 35 }, (_, i) => ({ block_id: `${id}:content:${i}`, block_type: "paragraph",
            content: `Paragraph ${i}. ` + "Long reading content remains on the current page. ".repeat(8) })),
        })),
      } });
      unexpected.push(path); return route.fulfill({ status: 500, json: { detail: "Unexpected API" } });
    });
    await page.goto(`${process.env.UI_URL || "http://127.0.0.1:5173"}/documents/Book`);
    await page.locator(".hierarchy-section-button").click();
    await page.locator(".reader-content-page-next").click();
    const list = page.locator(".reader-content-block-list");
    await list.evaluate(el => { window.readerElement = el; el.scrollTop = 120; });
    const count = requests.length, storageCount = await page.evaluate(() => window.storageWrites.length);
    const toggle = page.locator(".reader-content-fullscreen-toggle");
    await toggle.click();
    await page.waitForFunction(() => document.documentElement.classList.contains("reader-fullscreen"));
    if (mode === "native") await page.waitForFunction(() => document.fullscreenElement === document.documentElement);
    const geometry = await page.locator(".reader-content-pane").evaluate(el => {
      const r = el.getBoundingClientRect();
      return { left: r.left, top: r.top, width: r.width, height: r.height, vw: innerWidth, vh: innerHeight };
    });
    assert.ok(Math.abs(geometry.left) < 1 && Math.abs(geometry.top) < 1);
    assert.ok(Math.abs(geometry.width - geometry.vw) < 1 && Math.abs(geometry.height - geometry.vh) < 1);
    assert.equal(await page.locator(".topbar").isVisible(), false);
    assert.equal(await page.locator(".navigation-pane").isVisible(), false);
    assert.equal(await page.locator(".reader-pane-resizer").isVisible(), false);
    assert.match(await page.locator(".reader-content-pagination-bar").innerText(), /Page 2 \/ 3/);
    assert.equal(await list.evaluate(el => window.readerElement === el && el.scrollTop === 120), true);
    assert.equal(requests.length, count, "Fullscreen must not reload layout, content or interaction data");
    if (mode === "deferred") {
      await toggle.click(); await page.evaluate(() => window.resolveFullscreen());
      await page.waitForTimeout(100);
      assert.equal(await page.locator("html.reader-fullscreen").count(), 0, "Late fullscreen outcome must not reopen reading mode");
    } else {
      await page.getByRole("button", { name: "Review source", exact: true }).click();
      const dialog = page.getByRole("dialog");
      await dialog.waitFor();
      assert.equal(await dialog.isVisible(), true, "Body-portalled source review remains visible");
      await page.keyboard.press("Escape"); await dialog.waitFor({ state: "hidden" });
      assert.equal(await toggle.getAttribute("aria-pressed"), "true", "Dialog Escape must not also close reading mode");
      await list.evaluate(el => { el.scrollTop = 0; });
      await page.locator(".reader-content-block-interaction-trigger").first().click();
      await page.getByRole("menuitem", { name: "Insights", exact: true }).waitFor();
      await page.keyboard.press("Escape");
      await page.locator(".reader-content-block-interaction-menu").waitFor({ state: "hidden" });
      assert.equal(await toggle.getAttribute("aria-pressed"), "true");
      if (mode === "native") await page.screenshot({ path: `/tmp/reader-fullscreen-${width}.png` });
      await list.evaluate(el => { el.scrollTop = 120; });
      if (mode === "native") await page.evaluate(() => document.exitFullscreen());
      else await page.keyboard.press("Escape");
      await page.waitForFunction(() => !document.documentElement.classList.contains("reader-fullscreen"));
      assert.equal(await list.evaluate(el => window.readerElement === el && el.scrollTop === 120), true);
      assert.equal(await toggle.evaluate(el => el === document.activeElement), true);
    }
    assert.match(await page.locator(".reader-content-pagination-bar").innerText(), /Page 2 \/ 3/);
    assert.equal(await page.locator(".topbar").isVisible(), true);
    assert.equal(await page.locator(".navigation-pane").isVisible(), true);
    assert.equal(await page.evaluate(() => window.storageWrites.length), storageCount);
    assert.deepEqual(errors, []); assert.deepEqual(unexpected, []);
    await page.close();
    console.log(`PASS ${width}px ${mode}: full viewport, same DOM/page/scroll, modal/menu, exit, focus, no reload/storage`);
  }
} finally { await browser.close(); }
