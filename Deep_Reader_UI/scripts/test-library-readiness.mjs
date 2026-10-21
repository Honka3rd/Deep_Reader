// All API traffic is mocked; this suite must not invoke real preparation.
import assert from "node:assert/strict";
const { chromium } = await import(process.env.PLAYWRIGHT_MODULE || "playwright");
const browser = await chromium.launch({ headless: true,
  ...(process.env.CHROMIUM_PATH ? { executablePath: process.env.CHROMIUM_PATH } : {}) });
const base = process.env.UI_URL || "http://127.0.0.1:5173";
const layout = name => ({ document_id: name, title: name, chapters: [{ chapter_id: "c", title: name,
  sections: [{ section_id: "s", title: `${name} section`, task_units: [{ unit_id: "u" }] }] }] });
try {
  for (const width of [1280, 390]) {
    const page = await browser.newPage({ viewport: { width, height: 900 } });
    const errors = [], unexpected = [], writes = [], reads = [];
    let readyStatus = "ready", missing = true, deferred, hold = false, uploadRoute;
    page.on("pageerror", e => errors.push(e.message));
    await page.route("**/api/**", async route => {
      const path = new URL(route.request().url()).pathname;
      if (["/api/documents", "/api/documents/management"].includes(path)) {
        return route.fulfill({ json: { items: [
          ["Ready", readyStatus], ["Raw", "not_prepared"], ["Failed", "failed"],
          ["Unknown", "unknown"], ["Stale", "stale"], ["Missing", "ready"],
          ["Malformed", "ready"], ["Upload", "ready"],
        ].map(([name, status]) => ({ doc_name: name, title: name, source_type: "txt",
          source_status: "available", preparation_status: status })), total: 8 } });
      }
      if (path === "/api/documents/task-layout") {
        const body = route.request().postDataJSON();
        reads.push(body);
        assert.equal(body.refresh_task_units, false);
        assert.equal(body.include_anchor_page_evidence, false);
        if (hold) { deferred = route; return; }
        if (body.doc_name === "Missing" && missing) return route.fulfill({ status: 409, json: { detail: "Stale layout" } });
        if (body.doc_name === "Malformed") return route.fulfill({ json: { success: true } });
        return route.fulfill({ json: layout(body.doc_name) });
      }
      if (path === "/api/documents/prepare-task-layout") {
        const body = route.request().postDataJSON();
        writes.push(body);
        if (body.doc_name === "Upload") { uploadRoute = route; return; }
        assert.equal(body.doc_name, "Missing");
        assert.equal(body.pdf_extraction_mode, undefined);
        missing = false;
        return route.fulfill({ json: layout(body.doc_name) });
      }
      if (path === "/api/documents/upload") return route.fulfill({ json: {
        doc_name: "Upload", source_filename: "Upload.txt", source_type: "txt", source_status: "uploaded",
      } });
      unexpected.push(path);
      return route.fulfill({ status: 500 });
    });
    await page.goto(base);
    const library = page.locator(".file-management-dialog");
    const openRow = name => page.locator(".file-management-row").filter({ has: page.getByText(name, { exact: true }).first() })
      .locator(".file-management-open-action");
    await page.getByRole("button", { name: "Library", exact: true }).click();
    await openRow("Ready").waitFor();
    for (const name of ["Raw", "Failed", "Unknown", "Stale"]) assert.equal(await openRow(name).isDisabled(), true);
    assert.equal(reads.length, 0, "Catalog reads must not probe or generate layouts");
    await openRow("Ready").click();
    await page.locator(".hierarchy-section-button").filter({ hasText: "Ready section" }).waitFor();
    readyStatus = "failed";
    await page.getByRole("button", { name: "Library", exact: true }).click();
    await page.waitForFunction(() => document.querySelector(".file-management-row .file-management-open-action")?.disabled);
    assert.equal(await openRow("Ready").isDisabled(), true);
    assert.equal(await page.locator(".file-management-recent-action").isDisabled(), true);
    readyStatus = "ready";
    await page.locator(".file-management-refresh-action").click();
    await page.waitForFunction(() => !document.querySelector(".file-management-recent-action")?.disabled);
    hold = true;
    await page.locator(".file-management-recent-action").click();
    await page.getByRole("status").filter({ hasText: "Checking reader readiness." }).waitFor();
    while (!deferred) await new Promise(resolve => setTimeout(resolve, 10));
    await library.getByRole("button", { name: "Close", exact: true }).click();
    hold = false;
    await deferred.fulfill({ json: layout("Ready") }).catch(() => {});
    await page.getByRole("button", { name: "Library", exact: true }).click();
    await openRow("Missing").click();
    const dialog = page.getByRole("dialog", { name: "Prepare document", exact: true });
    await dialog.waitFor();
    assert.equal(writes.length, 0, "Failed Open must not prepare");
    assert.equal(await dialog.locator(".pdf-mode-control").count(), 0);
    await dialog.getByRole("button", { name: "Prepare", exact: true }).click();
    await dialog.getByText("Document prepared.", { exact: true }).waitFor();
    await dialog.getByRole("button", { name: "Open", exact: true }).click();
    await page.locator(".hierarchy-section-button").filter({ hasText: "Missing section" }).waitFor();
    assert.equal(writes.length, 1);
    await page.getByRole("button", { name: "Library", exact: true }).click();
    await openRow("Malformed").click();
    await dialog.waitFor();
    assert.match(page.url(), /documents\/Missing$/);
    await dialog.getByRole("button", { name: "Close", exact: true }).click();
    await page.locator('input[type="file"]').setInputFiles({ name: "Upload.txt", mimeType: "text/plain", buffer: Buffer.from("text") });
    await page.locator(".file-management-upload-action").click();
    await page.getByText("Preparing document.", { exact: true }).waitFor();
    assert.equal(await openRow("Upload").isDisabled(), true);
    assert.equal(await page.locator(".file-management-open-uploaded-action").isDisabled(), true);
    await page.locator(".file-management-cancel-preparation-action").click();
    assert.equal(await openRow("Upload").isDisabled(), true);
    await page.locator(".file-management-upload-another-action").click();
    assert.equal(await openRow("Upload").isDisabled(), true, "Discarding upload result must not clear the cancellation gate");
    assert.equal(writes.length, 2);
    await page.screenshot({ path: `/tmp/library-readiness-${width}.png`, fullPage: true });
    await uploadRoute?.fulfill({ json: layout("Upload") }).catch(() => {});
    assert.deepEqual(errors, []);
    assert.deepEqual(unexpected, []);
    await page.close();
    console.log(`PASS ${width}px: status gates, recent readiness, close cancellation, explicit TXT prepare, malformed layout, upload row bypass prevention`);
  }
} finally { await browser.close(); }
