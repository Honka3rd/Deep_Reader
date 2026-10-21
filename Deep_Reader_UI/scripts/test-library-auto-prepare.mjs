// Intercept all API traffic: no real uploads, OCR or LLM work.
import assert from "node:assert/strict";
const { chromium } = await import(process.env.PLAYWRIGHT_MODULE || "playwright");
const browser = await chromium.launch({ headless: true,
  ...(process.env.CHROMIUM_PATH ? { executablePath: process.env.CHROMIUM_PATH } : {}) });
try {
  for (const width of [1280, 390]) {
    const page = await browser.newPage({ viewport: { width, height: 900 } });
    const writes = [], reads = [], errors = [], unexpected = [], aborted = [];
    let uploadCount = 0, ready = false, pendingRoute;
    page.on("pageerror", e => errors.push(e.message));
    page.on("requestfailed", request => {
      if (request.url().endsWith("/prepare-task-layout")) aborted.push(request.failure()?.errorText);
    });
    await page.route("**/api/**", async route => {
      const path = new URL(route.request().url()).pathname;
      if (["/api/documents", "/api/documents/management"].includes(path)) {
        return route.fulfill({ json: { items: [], total: 0 } });
      }
      if (path === "/api/documents/upload") {
        uploadCount++;
        return route.fulfill({ json: { doc_name: `Unique ${uploadCount}`, source_filename: "Book.pdf",
          source_type: "pdf", source_status: "uploaded", preparation_status: "not_prepared" } });
      }
      if (path === "/api/documents/prepare-task-layout") {
        writes.push(route.request().postDataJSON());
        pendingRoute = route;
        return;
      }
      if (path === "/api/documents/task-layout") {
        reads.push(route.request().postDataJSON());
        return route.fulfill(ready ? { json: { document_id: "opaque", chapters: [] } }
          : { status: 409, json: { detail: "Not ready" } });
      }
      unexpected.push(path);
      return route.fulfill({ status: 500 });
    });
    await page.goto(process.env.UI_URL || "http://127.0.0.1:5173");
    await page.getByRole("button", { name: "Library", exact: true }).click();
    await page.clock.install();
    await page.clock.pauseAt(new Date());
    const upload = async () => {
      await page.locator('input[type="file"]').setInputFiles({ name: "Book.pdf",
        mimeType: "application/pdf", buffer: Buffer.from("fixture") });
      await page.locator(".file-management-upload-action").click();
      await page.getByText("Preparation queued.", { exact: true }).waitFor();
    };
    const cancel = page.locator(".file-management-cancel-preparation-action");
    const prepare = page.locator(".file-management-prepare-uploaded-action");
    const open = page.locator(".file-management-open-uploaded-action");
    await upload();
    assert.equal(await open.isDisabled(), true);
    await cancel.click();
    await page.clock.runFor(10);
    assert.equal(writes.length, 0, "Queued cancellation must prevent dispatch");
    await page.getByText("Preparation cancelled before submission.", { exact: true }).waitFor();
    await prepare.click();
    await prepare.evaluate(button => button.click());
    await page.clock.runFor(10);
    await page.waitForFunction(() => document.querySelector(".file-management-post-upload-message")?.textContent === "Preparing document.");
    while (!pendingRoute) await new Promise(resolve => setTimeout(resolve, 10));
    assert.equal(writes.length, 1);
    assert.equal(writes[0].doc_name, "Unique 1");
    assert.equal(writes[0].pdf_extraction_mode, "fast");
    assert.equal(writes[0].force_rebuild, false);
    await page.clock.runFor(500);
    await page.screenshot({ path: `/tmp/library-auto-prepare-pending-${width}.png`, fullPage: true });
    await cancel.click();
    await page.getByText("Waiting cancelled. Server preparation may continue.", { exact: true }).waitFor();
    assert.equal(await open.isDisabled(), true);
    await prepare.click();
    await page.clock.runFor(10);
    await page.getByText("Reader is not ready. Server work may still be running; no new preparation was submitted.", { exact: true }).waitFor();
    assert.equal(writes.length, 1, "Cancelled work must reconcile, not blindly retry");
    assert.equal(reads.length, 1);
    const abandonedRoute = pendingRoute;
    pendingRoute = undefined;
    await page.locator(".file-management-upload-another-action").click();
    await upload();
    await page.clock.runFor(10);
    while (!pendingRoute) await new Promise(resolve => setTimeout(resolve, 10));
    assert.equal(writes.length, 2);
    await abandonedRoute.fulfill({ json: { document_id: "old", chapters: [], pdf_extraction: {
      requested_mode: "fast", effective_mode: "fast", provenance: "applied" } } }).catch(() => {});
    assert.equal(await open.isDisabled(), true, "Late cancelled response must not mark replacement ready");
    await page.locator(".file-management-dialog").getByRole("button", { name: "Close", exact: true }).click();
    await page.clock.runFor(500);
    await page.getByRole("button", { name: "Library", exact: true }).click();
    await page.clock.runFor(500);
    assert.equal(writes.length, 2, "Reopening must not resubmit");
    await cancel.click();
    ready = true;
    await prepare.click();
    await page.clock.runFor(10);
    await page.getByText("Document prepared. You can open it now.", { exact: true }).waitFor();
    assert.equal(await open.isEnabled(), true);
    assert.equal(writes.length, 2);
    assert.equal(reads.at(-1).doc_name, "Unique 2");
    assert.equal(new URL(page.url()).pathname, new URL(process.env.UI_URL || "http://127.0.0.1:5173").pathname,
      "Cancelled completion must not navigate to a document route");
    assert.ok(aborted.length >= 1, "Cancellation must abort the waiting fetch");
    await page.clock.runFor(500);
    await page.screenshot({ path: `/tmp/library-auto-prepare-${width}.png`, fullPage: true });
    assert.deepEqual(errors, []);
    assert.deepEqual(unexpected, []);
    await page.close();
    console.log(`PASS ${width}px: queued cancel, request abort, unique identity/mode, reconciliation, late response isolation, close/reopen`);
  }
} finally { await browser.close(); }
