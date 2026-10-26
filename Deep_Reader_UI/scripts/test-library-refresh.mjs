// Real App/Fast Refresh lifecycle, intercepted catalog only; no source edits or book writes.
import assert from "node:assert/strict";
const { chromium } = await import(process.env.PLAYWRIGHT_MODULE || "playwright");
const browser = await chromium.launch({ headless: true });
try {
  for (const width of [1280, 390]) {
    const page = await browser.newPage({ viewport: { width, height: 900 } });
    page.setDefaultTimeout(10000);
    const errors = [], unexpected = [];
    let reads = 0, preparation = "building", hold = false, held;
    const catalog = status => ({ total: 1, items: [{ doc_name: "OCR Book", title: "OCR Book",
      source: "raw.pdf", source_type: "pdf", source_status: "available", preparation_status: status }] });
    page.on("pageerror", error => errors.push(error.message));
    await page.route("**/api/**", route => {
      const path = new URL(route.request().url()).pathname;
      if (path === "/api/documents") return route.fulfill({ json: catalog(preparation) });
      if (path === "/api/documents/management") {
        assert.equal(route.request().method(), "GET");
        reads++;
        if (hold) { held = route; return; }
        return route.fulfill({ json: catalog(preparation) });
      }
      unexpected.push(path);
      return route.fulfill({ status: 500, json: { detail: "Unexpected API" } });
    });
    await page.goto(process.env.UI_URL || "http://127.0.0.1:5173");
    await page.getByRole("button", { name: "Library", exact: true }).click();
    const library = page.locator(".file-management-dialog");
    const refresh = library.locator(".file-management-refresh-action");
    const status = library.locator(".file-management-structure-status");
    await status.getByText("Structure building", { exact: true }).waitFor();
    const node = await library.elementHandle();
    const replayEffects = async (expectOpen = true) => {
      await page.evaluate(async () => {
        const runtime = await import("/@react-refresh");
        const previous = window.libraryRefreshModule ?? await import("/src/App.tsx");
        const next = await import(`/src/App.tsx?t=${Date.now()}`);
        const error = runtime.validateRefreshBoundaryAndEnqueueUpdate("library-refresh-test", previous, next);
        if (error) throw new Error(error);
        window.libraryRefreshModule = next;
      });
      await page.waitForTimeout(150);
      if (expectOpen) assert.equal(await node.evaluate(el => el.isConnected), true, "Refresh preserves the open modal and React state");
    };
    const expectManualRefresh = async expected => {
      const before = reads;
      await refresh.click();
      await page.waitForTimeout(150);
      assert.equal(reads, before + 1, "Visible Refresh must issue one catalog request");
      await status.getByText(expected, { exact: true }).waitFor();
    };
    assert.equal(reads, 1);
    await replayEffects();
    assert.equal(reads, 1, "Effect replay does not poll or automatically reload the catalog");
    preparation = "ready";
    await expectManualRefresh("Structure prepared");
    preparation = "failed";
    await replayEffects();
    await expectManualRefresh("Structure failed");

    hold = true;
    await refresh.click();
    for (let i = 0; !held && i < 100; i++) await page.waitForTimeout(10);
    assert.ok(held);
    assert.equal(await refresh.isDisabled(), true);
    const beforeReplay = reads;
    await replayEffects();
    assert.equal(reads, beforeReplay, "Cancelled refresh is not retried automatically");
    assert.equal(await refresh.isDisabled(), false, "Effect replay must not strand the loading state");
    hold = false; preparation = "building";
    await expectManualRefresh("Structure building");
    await held.fulfill({ json: catalog("ready") }).catch(() => {});
    await page.waitForTimeout(150);
    assert.equal(await status.textContent(), "Structure building", "Cancelled response cannot overwrite a later refresh");
    await page.screenshot({ path: `/tmp/library-refresh-${width}.png`, fullPage: true });
    await library.getByRole("button", { name: "Close", exact: true }).click();
    const beforeCloseReplay = reads;
    await replayEffects(false);
    assert.equal(reads, beforeCloseReplay, "Closed Library stays idle on effect replay");
    await page.getByRole("button", { name: "Library", exact: true }).click();
    await status.getByText("Structure building", { exact: true }).waitFor();
    assert.equal(reads, beforeCloseReplay + 1);
    assert.deepEqual(errors, []); assert.deepEqual(unexpected, []);
    console.log(`PASS ${width}px: open-modal effect replay, manual refresh, aborted loading recovery, stale response, reopen`);
    await page.close();
  }
} finally { await browser.close(); }
