// All API calls are intercepted; no production uploads, OCR or LLM calls.
import assert from "node:assert/strict";
const { chromium } = await import(process.env.PLAYWRIGHT_MODULE || "playwright");
const browser = await chromium.launch({ headless: true,
  ...(process.env.CHROMIUM_PATH ? { executablePath: process.env.CHROMIUM_PATH } : {}) });
const base = process.env.UI_URL || "http://127.0.0.1:5173";
const layout = { document_id: "Existing", title: "Existing", chapters: [] };
try {
  for (const width of [1280, 390]) {
    const page = await browser.newPage({ viewport: { width, height: 900 } });
    const requests = [], unexpected = [], errors = [];
    let uploads = 0, failFast = true, failExisting = true, release;
    page.on("pageerror", (e) => errors.push(e.message));
    await page.route("**/api/**", async (route) => {
      const path = new URL(route.request().url()).pathname;
      if (["/api/documents", "/api/documents/management"].includes(path)) {
        return route.fulfill({ json: { items: [
          { doc_name: "Existing", title: "Existing", source: "structured", source_type: "pdf", source_status: "available", preparation_status: "ready" },
          { doc_name: "Text", title: "Text", source: "raw", source_type: "txt", source_status: "available" },
        ], total: 2 } });
      }
      if (path === "/api/documents/upload") {
        uploads++;
        if (uploads === 1) return route.fulfill({ status: 409, json: {
          source_status: "failed", reason: "duplicate_conflict", source_filename: "Book.pdf",
        } });
        return route.fulfill({ json: { doc_name: `Unique ${uploads}`, source_filename: "Book.pdf",
          source_type: "pdf", source_status: "uploaded", preparation_status: "not_prepared" } });
      }
      if (path === "/api/documents/prepare-task-layout") {
        const body = route.request().postDataJSON();
        requests.push(body);
        if (body.doc_name === "Existing" && failExisting) {
          failExisting = false;
          return route.fulfill({ status: 502, json: { detail: "OCR failed." } });
        }
        if (body.pdf_extraction_mode === "fast" && failFast) {
          failFast = false;
          return route.fulfill({ status: 422, json: { reason: "no_text", detail: "No native text." } });
        }
        if (body.pdf_extraction_mode === "auto") await new Promise((resolve) => { release = resolve; });
        return route.fulfill({ json: { ...layout, pdf_extraction: {
          requested_mode: body.pdf_extraction_mode, effective_mode: body.pdf_extraction_mode, provenance: "applied",
        } } });
      }
      if (path === "/api/documents/task-layout") {
        assert.equal(route.request().postDataJSON().pdf_extraction_mode, undefined);
        return route.fulfill({ json: layout });
      }
      unexpected.push(path);
      return route.fulfill({ status: 500, json: { detail: "Unexpected request" } });
    });
    await page.goto(base);
    await page.getByRole("button", { name: "Library", exact: true }).click();
    const library = page.locator(".file-management-dialog");
    await page.locator('input[type="file"]').setInputFiles({ name: "Book.pdf", mimeType: "application/pdf", buffer: Buffer.from("fixture") });
    const modes = library.locator(".pdf-mode-control");
    await library.getByRole("combobox", { name: "Printed table of contents" }).click();
    await page.getByRole("option", { name: "Absent", exact: true }).click();
    assert.equal(requests.length, 0);
    assert.equal(await modes.getByRole("button", { name: "Fast", exact: true }).getAttribute("aria-pressed"), "true");
    await page.locator(".file-management-upload-action").click();
    await page.locator(".file-management-confirm-duplicate-upload-action").waitFor();
    assert.equal(await modes.getByRole("button", { name: "OCR", exact: true, includeHidden: true }).isDisabled(), true);
    await page.locator(".file-management-confirm-duplicate-upload-action").click();
    await page.getByText("no_text", { exact: true }).waitFor();
    assert.equal(requests.length, 1);
    assert.equal(requests[0].doc_name, "Unique 2");
    assert.equal(requests[0].pdf_extraction_mode, "fast");
    assert.equal(requests[0].toc_hint, "absent");
    await modes.getByRole("button", { name: "OCR", exact: true }).click();
    await page.locator(".file-management-prepare-uploaded-action").click();
    await page.getByText("Document prepared. You can open it now.", { exact: true }).waitFor();
    assert.equal(requests[1].pdf_extraction_mode, "ocr");
    assert.equal(requests[1].toc_hint, "absent");
    await page.locator(".file-management-upload-another-action").click();
    await page.locator('input[type="file"]').setInputFiles({ name: "Next.pdf", mimeType: "application/pdf", buffer: Buffer.from("fixture") });
    assert.equal(await modes.getByRole("button", { name: "OCR", exact: true }).getAttribute("aria-pressed"), "true");
    await modes.getByRole("button", { name: "Auto", exact: true }).click();
    await page.locator('input[type="file"]').setInputFiles({ name: "Replacement.pdf", mimeType: "application/pdf", buffer: Buffer.from("fixture") });
    assert.equal(await modes.getByRole("button", { name: "Auto", exact: true }).getAttribute("aria-pressed"), "true");
    await page.locator(".file-management-upload-action").click();
    await page.waitForFunction(() => document.querySelector(".file-management-prepare-uploaded-action")?.textContent === "Preparing");
    assert.equal(await modes.getByRole("button", { name: "Fast", exact: true }).isDisabled(), true);
    await library.getByRole("button", { name: "Close", exact: true }).click();
    await page.getByRole("button", { name: "Library", exact: true }).click();
    assert.equal(await modes.getByRole("button", { name: "Auto", exact: true }).getAttribute("aria-pressed"), "true");
    while (!release) await page.waitForTimeout(10);
    release();
    await page.getByText("Document prepared. You can open it now.", { exact: true }).waitFor();
    assert.equal(requests.length, 3);
    assert.equal(requests[2].doc_name, "Unique 3");
    assert.equal(requests[2].pdf_extraction_mode, "auto");
    await page.locator(".file-management-upload-another-action").click();
    await page.locator('input[type="file"]').setInputFiles({ name: "Text.txt", mimeType: "text/plain", buffer: Buffer.from("text") });
    assert.equal(await modes.count(), 0);
    await page.locator(".file-management-cancel-upload-action").click();
    assert.equal(await modes.count(), 0);
    await page.locator('input[type="file"]').setInputFiles({ name: "Scan.pdf", mimeType: "application/pdf", buffer: Buffer.from("fixture") });
    assert.equal(await modes.getByRole("button", { name: "Auto", exact: true }).getAttribute("aria-pressed"), "true");
    await modes.getByRole("button", { name: "OCR", exact: true }).click();
    await page.locator('input[type="file"]').setInputFiles({ name: "Scan-replacement.pdf", mimeType: "application/pdf", buffer: Buffer.from("fixture") });
    assert.equal(await modes.getByRole("button", { name: "OCR", exact: true }).getAttribute("aria-pressed"), "true");
    await page.locator(".file-management-upload-action").click();
    await page.getByText("Document prepared. You can open it now.", { exact: true }).waitFor();
    assert.equal(requests.length, 4);
    assert.equal(requests[3].pdf_extraction_mode, "ocr");
    await page.locator(".file-management-upload-another-action").click();
    assert.equal(await page.getByRole("button", { name: "Prepare PDF Text", exact: true }).count(), 0);
    await page.getByRole("button", { name: "Prepare PDF Existing", exact: true }).click();
    const dialog = page.getByRole("dialog", { name: "Prepare PDF", exact: true });
    await dialog.getByRole("combobox", { name: "Printed table of contents" }).click();
    await page.getByRole("option", { name: "Present", exact: true }).click();
    await dialog.getByRole("button", { name: "OCR", exact: true }).click();
    await dialog.getByRole("button", { name: "Prepare", exact: true }).click();
    await dialog.getByText("OCR failed. Existing layout is available.", { exact: true }).waitFor();
    assert.equal(requests.at(-1).toc_hint, "present");
    assert.equal(await dialog.getByRole("button", { name: "Open", exact: true }).isEnabled(), true);
    await page.screenshot({ path: `/tmp/library-pdf-modes-${width}.png`, fullPage: true });
    const bounds = await dialog.locator(".pdf-mode-control").boundingBox();
    assert.ok(bounds.x >= 0 && bounds.x + bounds.width <= width);
    await dialog.getByRole("button", { name: "Close", exact: true }).click();
    await page.evaluate(async () => {
      const { DocumentPreparationService } = await import("/src/services/DocumentPreparationService.ts");
      const { RestClient } = await import("/src/services/RestClient.ts");
      await new DocumentPreparationService(new RestClient()).prepareReader("Text");
    });
    assert.equal(requests.at(-1).pdf_extraction_mode, undefined);
    assert.deepEqual(unexpected, []);
    assert.deepEqual(errors, []);
    console.log(`PASS ${width}px: PDF modes, duplicate identity, fast retry, pending snapshot/reopen, TXT omission, failed reprepare/readback`);
    await page.close();
  }
} finally {
  await browser.close();
}
