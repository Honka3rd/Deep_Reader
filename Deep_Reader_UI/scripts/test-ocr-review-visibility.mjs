// Actual App and reader with controlled server capabilities; no live-document writes.
import assert from "node:assert/strict";
const { chromium } = await import(process.env.PLAYWRIGHT_MODULE || "playwright");
const browser = await chromium.launch({ headless: true,
  ...(process.env.CHROMIUM_PATH ? { executablePath: process.env.CHROMIUM_PATH } : {}) });
try {
  for (const width of [1280, 390]) {
    const page = await browser.newPage({ viewport: { width, height: 900 } });
    const errors = [], unexpected = [];
    const capabilities = { OcrPdf: true, AutoOcrPdf: true, NativePdf: false, TextFile: false, Unknown: undefined, SlowOcr: true };
    let held;
    page.on("pageerror", error => errors.push(error.message));
    await page.route("**/api/**", async route => {
      const url = new URL(route.request().url()), path = url.pathname;
      const name = decodeURIComponent(path.split("/")[3] || "");
      if (path === "/api/documents") return route.fulfill({ json: { total: 6,
        items: Object.keys(capabilities).map(doc_name => ({ doc_name, title: doc_name, source: "structured" })) } });
      if (path.endsWith("/reparse-structure/status")) return route.fulfill({ json: { status: "idle" } });
      if (path.endsWith("/task-layout")) return route.fulfill({ json: {
        document_id: route.request().postDataJSON().doc_name, chapters: [{ chapter_id: "c", title: "Chapter",
          sections: [{ section_id: "s", title: "Body", task_units: [{ unit_id: "u" }] }] }] } });
      if (path.endsWith("/reading-content") && route.request().method() === "GET") {
        const json = { document_id: name, revision: 1, source_token: name, disposition: "completed",
          ocr_review_available: capabilities[name], can_undo: true, undo_action: "edit", max_chars: 10000, regions: [],
          blocks: [{ block_id: "b", task_unit_id: "u", chapter_id: "c", section_id: "s", content: "Readable body text.", deleted: false }],
          cleanup_audit: { total: 0, offset: 0, limit: 10, items: [] } };
        if (name === "SlowOcr") { held = { route, json }; return; }
        return route.fulfill({ json });
      }
      if (path.endsWith("/task-units/content")) return route.fulfill({ json: { contents: [{
        document_id: name, chapter_id: "c", section_id: "s", task_unit_id: "u",
        content_blocks: [{ block_id: "b", block_type: "content", content: "Readable body text." }] }] } });
      unexpected.push(`${route.request().method()} ${path}`);
      return route.fulfill({ status: 500, json: { detail: "Unexpected request" } });
    });
    const button = page.getByRole("button", { name: "Review extracted text", exact: true });
    const toolbar = page.locator(".reader-content-actions");
    const dialog = page.locator('[aria-labelledby="reading-content-dialog-title"]');
    const select = async name => {
      await page.getByRole("combobox").fill(name);
      await page.getByRole("option", { name, exact: true }).click();
    };
    await page.goto(process.env.UI_URL || "http://127.0.0.1:5173");
    for (const name of ["OcrPdf", "NativePdf", "AutoOcrPdf", "TextFile", "Unknown"]) {
      const loaded = page.waitForResponse(response => response.url().includes(`/${name}/reading-content`));
      await select(name); await loaded;
      await toolbar.waitFor({ state: "attached" });
      if (capabilities[name] === true) {
        await button.waitFor();
        await button.click();
        await dialog.getByRole("tab", { name: "Flagged areas" }).click();
        await dialog.getByText("No flagged areas recorded. You can still review previous cleanup changes.").waitFor();
        await dialog.getByRole("button", { name: "Close content editor" }).click();
        await button.waitFor();
        await page.waitForFunction(() => document.activeElement?.getAttribute("aria-label") === "Review extracted text", null, { timeout: 3000 });
      } else {
        await page.waitForTimeout(100);
        assert.equal(await toolbar.getByRole("button").count(), 0, `${name}: no review/Undo toolbar buttons`);
      }
      if (name === "TextFile") {
        await page.getByRole("treeitem").first().click();
        await page.locator(".reader-content-block-interaction-trigger").first().click();
        await page.getByRole("menuitem", { name: "Edit", exact: true }).click();
        await dialog.getByRole("textbox", { name: "Block text" }).waitFor();
        assert.equal(await dialog.getByRole("textbox", { name: "Block text" }).inputValue(), "Readable body text.");
        await dialog.getByRole("button", { name: "Close content editor" }).click();
      }
      if (["OcrPdf", "TextFile"].includes(name)) await page.screenshot({ path: `/tmp/ocr-review-${name}-${width}.png`, animations: "disabled" });
    }
    await select("SlowOcr");
    while (!held) await page.waitForTimeout(10);
    assert.equal(await button.count(), 0, "Loading is not OCR proof");
    await select("TextFile");
    await held.route.fulfill({ json: held.json }).catch(() => {});
    await page.waitForTimeout(150);
    assert.equal(await toolbar.getByRole("button").count(), 0, "Late OCR response cannot enable another document");
    assert.deepEqual(errors, []); assert.deepEqual(unexpected, []);
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth), false);
    await page.close();
    console.log(`PASS ${width}: OCR-only toolbar, resolved review, unknown/loading, ordinary text editing and late response isolation`);
  }
} finally { await browser.close(); }
