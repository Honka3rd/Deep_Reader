// Only intercepted API traffic: no real OCR, mutations or Library documents.
import assert from "node:assert/strict";
const { chromium } = await import(process.env.PLAYWRIGHT_MODULE || "playwright");
const browser = await chromium.launch({ headless: true,
  ...(process.env.CHROMIUM_PATH ? { executablePath: process.env.CHROMIUM_PATH } : {}) });
try {
  for (const width of [1280, 390]) {
    const page = await browser.newPage({ viewport: { width, height: 900 } });
    const errors = [], writes = [];
    let status = "ready", reads = 0, hold = false, deferred, fail = false;
    const payload = state => ({ items: state === "empty" ? [] : [{ doc_name: "OCR Book", title: "OCR Book",
      source: "raw.pdf", source_type: "pdf", source_status: "available", preparation_status: state }], total: 1 });
    page.on("pageerror", error => errors.push(error.message));
    await page.route("**/api/**", async route => {
      const path = new URL(route.request().url()).pathname;
      if (path === "/api/documents") return route.fulfill({ json: payload(status) });
      if (path === "/api/documents/management") {
        reads++;
        if (hold) { deferred = route; return; }
        if (fail) return route.fulfill({ status: 503, json: { detail: "Catalog unavailable" } });
        return route.fulfill({ json: payload(status) });
      }
      writes.push(path);
      return route.fulfill({ status: 500, json: { detail: "Unexpected API" } });
    });
    const row = page.locator(".file-management-row");
    const label = row.locator(".file-management-structure-status");
    const expectStatus = async text => {
      await page.waitForFunction(expected =>
        document.querySelector(".file-management-structure-status")?.textContent === expected, text,
      { timeout: 12000 });
    };
    const open = async () => {
      await page.getByRole("button", { name: "Library", exact: true }).click();
      await label.waitFor();
    };
    const close = () => page.locator(".file-management-dialog").getByRole("button", { name: "Close", exact: true }).click();
    const refresh = () => page.locator(".file-management-dialog").getByRole("button", { name: "Refresh", exact: true }).click();
    const expectIdle = async message => {
      const before = reads;
      await page.clock.fastForward(60_000);
      await page.waitForTimeout(100);
      assert.equal(reads, before, message);
    };
    await page.clock.install();
    await page.goto(process.env.UI_URL || "http://127.0.0.1:5173");
    await open();
    await expectStatus("Structure prepared");
    assert.equal(reads, 1, "Opening Library loads the catalog once");
    await expectIdle("Ready catalog must not poll");
    status = "building";
    await refresh();
    await expectStatus("Structure building");
    await expectIdle("Long-running OCR must not poll the management catalog");
    for (const selector of [".file-management-open-action", ".file-management-delete-action",
      ".file-management-rename-action", '[aria-label="Prepare PDF OCR Book"]']) {
      assert.equal(await row.locator(selector).isDisabled(), true);
    }
    await page.waitForTimeout(350);
    await page.screenshot({ path: `/tmp/library-building-${width}.png`, fullPage: true });
    await page.reload();
    await open();
    await expectStatus("Structure building");
    await expectIdle("Restored building status must not poll after page reload");
    await close();
    const closedReads = reads;
    await page.clock.fastForward(60_000);
    assert.equal(reads, closedReads, "Closed dialog must not poll");
    await open();
    status = "failed";
    await expectIdle("Remote completion must not trigger background catalog reads");
    await refresh();
    await expectStatus("Structure failed");
    assert.equal(await row.locator(".file-management-open-action").isDisabled(), true);
    assert.equal(await row.getByRole("button", { name: "Prepare PDF OCR Book" }).isDisabled(), false);
    await page.waitForTimeout(350);
    await page.screenshot({ path: `/tmp/library-failed-${width}.png`, fullPage: true });
    await expectIdle("Failed preparations must not poll");
    status = "ready";
    await refresh();
    await expectStatus("Structure prepared");
    assert.equal(await row.locator(".file-management-open-action").isDisabled(), false);
    await expectIdle("Manual refresh of a ready catalog must not start polling");
    status = "building";
    await refresh();
    await expectStatus("Structure building");
    await expectIdle("Building catalog remains idle across repeated timer intervals");
    hold = true;
    const before = reads;
    await refresh();
    for (let i = 0; !deferred && i < 50; i++) await page.waitForTimeout(100);
    assert.ok(deferred, "Expected the explicit refresh request");
    await page.clock.fastForward(60_000);
    assert.equal(reads, before + 1, "Waiting on explicit refresh must not start another request");
    await close();
    hold = false;
    status = "building";
    await open();
    await expectStatus("Structure building");
    await deferred.fulfill({ json: payload("ready") }).catch(() => {});
    await page.waitForTimeout(100);
    assert.equal(await label.textContent(), "Structure building", "Old response must not replace reopened state");
    status = "ready";
    await refresh();
    await expectStatus("Structure prepared");
    await expectIdle("Completion must stop polling");
    status = "empty";
    await refresh();
    await row.waitFor({ state: "detached" });
    await expectIdle("Empty catalog must not poll");
    status = "building";
    await refresh();
    await expectStatus("Structure building");
    fail = true;
    await refresh();
    await row.waitFor({ state: "detached" });
    await expectIdle("A failed catalog request must not retry automatically");
    fail = false;
    status = "ready";
    await refresh();
    await expectStatus("Structure prepared");
    assert.deepEqual(errors, []);
    assert.deepEqual(writes, [], "Reload/refresh/reopen must never start work");
    await page.close();
    console.log(`PASS ${width}px: no polling during OCR or after reload, explicit refresh/reopen, action gates, abort and stale response`);
  }
} finally {
  await browser.close();
}
