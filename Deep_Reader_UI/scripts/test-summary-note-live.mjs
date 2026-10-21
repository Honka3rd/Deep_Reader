import assert from "node:assert/strict";
import { readFile, mkdir } from "node:fs/promises";
import path from "node:path";
const corpus = path.resolve("../Deep_Reflective_Reader/data/test-books/summary-note-20261021");
const cases = ["fa794acef055f235", "085747b65094d52f"];
const { chromium } = await import(process.env.PLAYWRIGHT_MODULE || "playwright");
const browser = await chromium.launch({ headless: true,
  ...(process.env.CHROMIUM_PATH ? { executablePath: process.env.CHROMIUM_PATH } : {}) });
await mkdir(path.join(corpus, "browser"), { recursive: true });
try {
  for (const width of [1280, 390]) for (const id of cases) {
    const targets = JSON.parse(await readFile(path.join(corpus, "targets", id + ".json"), "utf8"));
    const page = await browser.newPage({ viewport: { width, height: 900 } });
    const errors = [], writes = [];
    page.on("pageerror", error => errors.push(error.message));
    page.on("request", request => {
      if (/reading-interactions\/(summary\/(generate|refresh)|note\/(save|delete))/.test(request.url())) writes.push(request.url());
    });
    await page.goto(`http://127.0.0.1:5173/documents/${encodeURIComponent(targets[0].doc_name)}`);
    await page.getByLabel("Document reading interactions", { exact: true }).waitFor();
    await page.locator(".hierarchy-section-button").first().click();
    const unit = page.locator(".reader-content-task-unit-group").filter({ has: page.locator(".reader-content-task-unit-id", { hasText: targets[3].task_unit_id }) }).first();
    await unit.waitFor({ timeout: 120000 });
    const block = page.locator(".reader-content-block").filter({ has: page.locator(".reader-content-block-id", { hasText: targets[4].content_block_id }) }).first();
    const triggers = [page.getByLabel("Document reading interactions", { exact: true }),
      page.locator(".hierarchy-chapter-interaction-trigger").first(),
      page.locator(".hierarchy-section-interaction-trigger").first(),
      unit.locator(".reader-task-unit-interaction-trigger"), block.locator(".reader-content-block-interaction-trigger")];
    for (let index = 0; index < triggers.length; index++) {
      const open = async kind => {
        await triggers[index].click();
        for (const name of ["Insights", "Quiz", "Critical thinking", "Summary", "Note"]) {
          assert.equal(await page.getByRole("menuitem", { name: new RegExp(`^${name}(?:\\s|$)`) }).count(), 1, `${index}: ${name}`);
        }
        await page.getByRole("menuitem", { name: kind, exact: true }).click();
        const region = page.getByRole("region", { name: kind, exact: true });
        await region.waitFor();
        await region.getByLabel("Loading", { exact: true }).waitFor({ state: "hidden" });
        return region;
      };
      let before = writes.length;
      let drawer = await open("Summary");
      await drawer.getByRole("button", { name: "Refresh", exact: true }).waitFor();
      assert.equal(writes.length, before, "Opening must not generate");
      assert.ok((await drawer.locator("h3").textContent()).trim());
      const metrics = await drawer.evaluate(element => ({ width: element.clientWidth, scrollWidth: element.scrollWidth }));
      assert.ok(metrics.scrollWidth <= metrics.width + 1, "No horizontal overflow");
      if (index === 0) {
        const original = await drawer.locator("h3").textContent();
        await page.route("**/summary/refresh", route => route.fulfill({ status: 502, json: {
          envelope: { interaction_type: "summary", status: "generation_failed", reason: "provider_error" }, payload: null,
        } }), { times: 1 });
        await drawer.getByRole("button", { name: "Refresh", exact: true }).click();
        await drawer.getByRole("alert").waitFor();
        assert.equal(await drawer.locator("h3").textContent(), original);
        await page.screenshot({ path: path.join(corpus, "browser", `${id}-${width}-summary.png`) });
      }
      await drawer.getByRole("button", { name: "Close", exact: true }).click();
      before = writes.length;
      drawer = await open("Note");
      const editor = drawer.getByRole("textbox", { name: "Note", exact: true });
      const original = await editor.inputValue();
      assert.ok(original.length > 0);
      assert.equal(writes.length, before, "Note read must not save");
      await editor.fill(original + " Browser acceptance " + width);
      page.once("dialog", dialog => dialog.dismiss());
      await drawer.getByRole("button", { name: "Close", exact: true }).click();
      assert.ok(await drawer.isVisible(), "Dirty close is guarded");
      if (index === 0) {
        const pendingDraft = await editor.inputValue();
        await page.route("**/note/save", route => route.fulfill({ status: 503, json: { detail: "temporarily_unavailable" } }), { times: 1 });
        await drawer.getByRole("button", { name: "Save", exact: true }).click();
        await drawer.getByRole("alert").waitFor();
        assert.equal(await editor.inputValue(), pendingDraft, "Failed save retains draft");
      }
      await drawer.getByRole("button", { name: "Save", exact: true }).click();
      await drawer.getByLabel("Loading", { exact: true }).waitFor({ state: "hidden" });
      assert.equal(await drawer.getByRole("alert").count(), 0);
      assert.ok(await drawer.getByRole("button", { name: "Save", exact: true }).isDisabled());
      await editor.fill("\u{1F4DD}".repeat(10001));
      assert.ok(await drawer.getByRole("button", { name: "Save", exact: true }).isDisabled());
      await editor.fill(original);
      await drawer.getByRole("button", { name: "Save", exact: true }).click();
      await drawer.getByLabel("Loading", { exact: true }).waitFor({ state: "hidden" });
      if (index === 0) await page.screenshot({ path: path.join(corpus, "browser", `${id}-${width}-note.png`) });
      await drawer.getByRole("button", { name: "Close", exact: true }).click();
      drawer = await open("Note");
      assert.equal(await drawer.getByRole("textbox", { name: "Note", exact: true }).inputValue(), original);
      if (index === 0) {
        const countBeforeDelete = writes.length;
        page.once("dialog", dialog => dialog.dismiss());
        await drawer.getByRole("button", { name: "Delete note", exact: true }).click();
        assert.equal(writes.length, countBeforeDelete, "Cancelled delete makes no request");
        page.once("dialog", dialog => dialog.accept());
        await drawer.getByRole("button", { name: "Delete note", exact: true }).click();
        await drawer.getByLabel("Loading", { exact: true }).waitFor({ state: "hidden" });
        assert.equal(await drawer.getByRole("textbox", { name: "Note", exact: true }).inputValue(), "");
        await drawer.getByRole("textbox", { name: "Note", exact: true }).fill(original);
        await drawer.getByRole("button", { name: "Save", exact: true }).click();
        await drawer.getByLabel("Loading", { exact: true }).waitFor({ state: "hidden" });
      }
      await drawer.getByRole("button", { name: "Close", exact: true }).click();
    }
    assert.deepEqual(errors, []);
    console.log(JSON.stringify({ case: id, width, levels: 5, passed: true }));
    await page.close();
  }
} finally { await browser.close(); }
