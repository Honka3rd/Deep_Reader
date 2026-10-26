// Production App + routes + disposable SQL. No paid calls or production books.
import assert from "node:assert/strict";
const api = process.env.CLEANUP_TEST_API;
assert.match(api || "", /^http:\/\/127\.0\.0\.1:\d+$/);
const { chromium } = await import(process.env.PLAYWRIGHT_MODULE || "playwright");
const browser = await chromium.launch({ headless: true,
  ...(process.env.CHROMIUM_PATH ? { executablePath: process.env.CHROMIUM_PATH } : {}) });
try {
  for (const width of [1280, 390]) {
    const page = await browser.newPage({ viewport: { width, height: 900 } });
    const errors = [], mutations = [];
    let loseUndo = false, failContent = false;
    page.on("pageerror", error => errors.push(error.message));
    await page.route("**/api/**", async route => {
      const url = new URL(route.request().url());
      const path = url.pathname.replace(/^\/api/, "");
      if (["/documents", "/documents/management"].includes(path)) return route.fulfill({ json: { items: [], total: 0 } });
      if (path === "/documents/reading-interactions/note/read") return route.fulfill({ json: {
        target: route.request().postDataJSON().target, target_token: "note-fixture", max_chars: 10000, note: null,
      } });
      if (failContent && path.endsWith("/task-units/content")) {
        failContent = false;
        return route.fulfill({ status: 503, json: { detail: "Temporary read failure" } });
      }
      assert.notEqual(path, "/documents/prepare-task-layout");
      const response = await route.fetch({ url: api + path + url.search });
      if (path.endsWith("/reading-content") && route.request().method() === "POST") {
        mutations.push(route.request().postDataJSON());
        if (loseUndo) { loseUndo = false; return route.abort(); }
      }
      return route.fulfill({ response });
    });
    const configure = async scenario => assert.equal((await page.request.post(api + "/fixture", { data: { scenario } })).status(), 204);
    const state = async () => (await page.request.get(api + "/documents/book/reading-content?include_cleanup_audit=true&include_blocks=true")).json();
    const stats = async () => (await page.request.get(api + "/stats")).json();
    const panel = page.locator('div[aria-labelledby="reading-content-dialog-title"]');
    const openReview = async () => {
      await page.getByRole("button", { name: "Review source", exact: true }).click();
      await panel.getByLabel("Loading content changes").waitFor({ state: "hidden" });
    };
    const closeReview = async () => {
      await panel.getByRole("button", { name: "Close content editor" }).click();
      await panel.waitFor({ state: "hidden" });
    };
    const load = async scenario => {
      await configure(scenario);
      await page.goto(`${process.env.UI_URL || "http://127.0.0.1:5174"}/documents/book`);
      await page.locator(".hierarchy-section-button").first().click();
      await page.locator(".reader-content-block-body").first().waitFor();
    };
    const enhance = async () => {
      await page.locator("#repair-menu-button").click();
      await page.getByRole("menuitem", { name: "LLM enhance", exact: true }).click();
      const dialog = page.getByRole("dialog", { name: "Smart structuring", exact: true });
      await dialog.getByText("Ready for smart structuring.", { exact: true }).waitFor();
      await dialog.getByRole("button", { name: "Enhance", exact: true }).click();
      await dialog.getByText(/OCR cleanup completed/).waitFor();
      await dialog.getByRole("button", { name: "Close", exact: true }).click();
    };

    await load("completed");
    await openReview();
    await panel.getByText("No applied OCR cleanup recorded.").waitFor();
    assert.equal(mutations.length, 0);
    await closeReview();
    await enhance();
    await page.locator(".hierarchy-chapter-interaction-trigger").first().click();
    await page.getByRole("menuitem", { name: "Note", exact: true }).click();
    await page.getByRole("textbox", { name: "Note", exact: true }).fill("Retain this unsaved note after Undo.");
    // Concurrent view test: dispatch the review action while the draft stays mounted.
    await page.getByRole("button", { name: "Review source", exact: true, includeHidden: true }).evaluate(el => el.click());
    await panel.getByLabel("Loading content changes").waitFor({ state: "hidden" });
    await panel.getByText("No unresolved OCR regions recorded.").waitFor();
    assert.equal(await page.getByText("OCR needs review", { exact: true }).count(), 0);
    await panel.getByRole("button", { name: "Expand cleanup 1", exact: true }).click();
    await panel.locator("blockquote").filter({ hasText: "@@??" }).waitFor();
    await panel.getByText(/Page 1/).waitFor();
    await page.screenshot({ path: `/tmp/cleanup-review-${width}.png`, animations: "disabled" });
    await panel.getByRole("button", { name: "Collapse cleanup 1", exact: true }).click();
    await panel.locator("blockquote").waitFor({ state: "hidden" });
    const beforeUndo = await stats();
    loseUndo = true;
    await panel.getByRole("button", { name: "Undo OCR cleanup", exact: true }).evaluate(el => { el.click(); el.click(); });
    await panel.getByText(/change could not be confirmed/).waitFor();
    failContent = true;
    await panel.getByRole("button", { name: "Undo OCR cleanup", exact: true }).click();
    await panel.getByText("Undone", { exact: true }).waitFor();
    await panel.getByText("Changes saved. Reading refresh failed.", { exact: true }).waitFor();
    await panel.getByRole("button", { name: "Retry reading", exact: true }).click();
    await panel.getByText("Changes saved. Reading refresh failed.", { exact: true }).waitFor({ state: "hidden" });
    assert.equal(mutations.length, 2);
    assert.equal(mutations[0].mutation_id, mutations[1].mutation_id);
    assert.equal((await stats()).revision, beforeUndo.revision + 1);
    assert.equal((await stats()).calls, beforeUndo.calls);
    assert.equal((await state()).regions.length, 1);
    await closeReview();
    assert.equal(await page.getByRole("textbox", { name: "Note", exact: true }).inputValue(), "Retain this unsaved note after Undo.");
    page.once("dialog", prompt => prompt.dismiss());
    await page.getByRole("region", { name: "Note", exact: true }).getByRole("button", { name: "Close", exact: true }).click();
    assert.equal(await page.getByRole("textbox", { name: "Note", exact: true }).inputValue(), "Retain this unsaved note after Undo.");
    page.once("dialog", prompt => prompt.accept());
    await page.getByRole("region", { name: "Note", exact: true }).getByRole("button", { name: "Close", exact: true }).click();
    await page.locator(".reader-content-block-body").filter({ hasText: "@@??" }).first().waitFor();
    await page.getByText("OCR needs review", { exact: true }).waitFor();
    await page.reload();
    await openReview();
    await panel.getByText("Undone", { exact: true }).waitFor();
    assert.equal(await panel.getByRole("button", { name: "Undo last content change", exact: true }).isDisabled(), true);
    await closeReview();

    await load("remaining_regions");
    await enhance();
    await openReview();
    await panel.getByText("Source regions (2)", { exact: true }).waitFor();
    assert.equal(await panel.getByRole("button", { name: "Original", exact: true }).count(), 2);
    assert.equal((await state()).regions.some(r => r.id === "r1"), false);
    await page.screenshot({ path: `/tmp/cleanup-review-remaining-${width}.png`, animations: "disabled" });
    await configure("stale");
    const concurrent = await state();
    await panel.getByRole("button", { name: "Undo OCR cleanup", exact: true }).click();
    await panel.getByText(/Content changed elsewhere/).waitFor();
    assert.deepEqual(await state(), concurrent);
    await panel.getByRole("button", { name: "Reload", exact: true }).click();
    await panel.getByText("Later changes prevent batch undo").waitFor();
    assert.equal(await panel.getByRole("button", { name: "Undo OCR cleanup", exact: true }).count(), 0);
    await closeReview();

    await load("completed");
    await enhance();
    await configure("long_history");
    await openReview();
    await panel.getByText("1-10 of 23", { exact: true }).waitFor();
    await panel.getByRole("button", { name: "Next cleanup records", exact: true }).click();
    await panel.getByText("11-20 of 23", { exact: true }).waitFor();
    await panel.getByRole("button", { name: "Next cleanup records", exact: true }).click();
    await panel.getByText("21-23 of 23", { exact: true }).waitFor();
    assert.equal(await panel.getByRole("button", { name: "Next cleanup records", exact: true }).isDisabled(), true);
    await panel.getByRole("button", { name: "Previous cleanup records", exact: true }).click();
    await panel.getByText("11-20 of 23", { exact: true }).waitFor();
    assert.equal(await panel.evaluate(el => el.scrollWidth > el.clientWidth), false);
    await page.screenshot({ path: `/tmp/cleanup-review-history-${width}.png`, animations: "disabled" });
    assert.deepEqual(errors, []);
    console.log(`PASS ${width}px: actual audit/Undo API, original quote/page, collapsed/empty/paged history, resolved/remaining review, lost-ack idempotence, restored content, stale conflict, reload`);
    await page.close();
  }
} finally { await browser.close(); }
