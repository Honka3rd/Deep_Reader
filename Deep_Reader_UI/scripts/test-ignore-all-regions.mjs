// Actual App, multi-page review and controlled lost-response/replay transport.
import assert from "node:assert/strict";
const { chromium } = await import(process.env.PLAYWRIGHT_MODULE || "playwright");
const browser = await chromium.launch({ headless: true,
  ...(process.env.CHROMIUM_PATH ? { executablePath: process.env.CHROMIUM_PATH } : {}) });
try {
  for (const width of [1280, 390]) {
    const page = await browser.newPage({ viewport: { width, height: 900 } });
    page.setDefaultTimeout(10000);
    const errors = [], unexpected = [], commands = [];
    let revision = 0, layouts = 0, reads = 0, committed = 0, receipt, hold, late;
    let regions = Array.from({ length: 25 }, (_, i) => ({ id: `r${i}`, page_index: i,
      status: "unresolved", block_ids: [], mapping_status: "unavailable", body_status: "unknown" }));
    const state = () => ({ document_id: "Book", revision, source_token: "source", structure_version: "1",
      ocr_review_available: true, disposition: regions.length ? "needs_review" : "completed", can_undo: false,
      max_chars: 10000, regions, blocks: [], cleanup_audit: { total: 0, offset: 0, limit: 10, items: [] } });
    const result = command => ({ ...state(), refresh: { document_id: "Book", source_token: "source", structure_version: "1",
      revision, mutation_id: command.mutation_id, mutation_revision: revision, scope_current: true,
      generated_invalidated: false, body_changed: false, blocks: [], block_page: { total: 0, offset: 0, limit: 20 } } });
    page.on("pageerror", error => errors.push(error.message));
    await page.route("**/api/**", async route => {
      const url = new URL(route.request().url()), path = url.pathname;
      if (path === "/api/documents") return route.fulfill({ json: { items: [], total: 0 } });
      if (path.endsWith("/reparse-structure/status")) return route.fulfill({ json: { status: "idle" } });
      if (path.endsWith("/task-layout")) { layouts++; return route.fulfill({ json: { document_id: "Book", chapters: [] } }); }
      if (path.endsWith("/reading-content")) {
        if (route.request().method() === "GET") { reads++; return route.fulfill({ json: state() }); }
        const command = route.request().postDataJSON(); commands.push(command);
        assert.equal(command.action, "discard_all_regions");
        assert.equal(command.region_id, undefined, "Whole-document command has no selected-page IDs");
        if (hold) { late = { route, command }; return; }
        if (commands.length === 1) return route.fulfill({ status: 503, json: { detail: "Temporary failure" } });
        if (!receipt) { regions = []; revision++; committed++; receipt = command.mutation_id; return route.abort(); }
        assert.equal(command.mutation_id, receipt, "Lost response retry retains the receipt ID");
        return route.fulfill({ json: result(command) });
      }
      unexpected.push(`${route.request().method()} ${path}`);
      return route.fulfill({ status: 500, json: { detail: "Unexpected request" } });
    });
    await page.goto((process.env.UI_URL || "http://127.0.0.1:5173") + "/documents/Book");
    await page.getByRole("button", { name: "Review extracted text", exact: true }).click();
    const review = page.locator('[aria-labelledby="reading-content-dialog-title"]');
    await review.getByRole("tab", { name: "Flagged areas" }).click();
    await review.getByRole("button", { name: "Next flagged areas" }).click();
    assert.equal(await review.locator(".source-review-region").count(), 5);
    const ignore = review.getByRole("button", { name: "Ignore all", exact: true });
    const confirm = page.getByRole("dialog", { name: "Ignore all 25 flagged areas?", exact: true });
    await ignore.click();
    await confirm.getByText(/Carefully check the extracted text/).waitFor();
    assert.ok((await confirm.textContent()).includes("cannot be undone"));
    await page.waitForFunction(() => document.activeElement?.textContent === "Cancel").catch(async error => {
      console.error(await page.evaluate(() => document.activeElement?.outerHTML));
      await page.screenshot({ path: `/tmp/ignore-all-focus-${width}.png` }); throw error;
    });
    await page.screenshot({ path: `/tmp/ignore-all-confirm-${width}.png`, animations: "disabled" });
    await confirm.getByRole("button", { name: "Cancel", exact: true }).click();
    assert.equal(commands.length, 0, "Cancel sends no mutation");
    await ignore.click();
    const submit = confirm.getByRole("button", { name: "Ignore all permanently" });
    await submit.evaluate(button => { button.click(); button.click(); });
    await confirm.getByRole("alert").waitFor();
    assert.equal(commands.length, 1, "Duplicate clicks are ignored");
    await submit.click();
    await confirm.getByRole("alert").waitFor();
    const beforeReads = reads, beforeLayouts = layouts;
    await submit.click();
    await confirm.waitFor({ state: "hidden" });
    await review.getByRole("heading", { name: "Flagged areas (0)", exact: true }).waitFor();
    await page.waitForFunction(() => document.activeElement?.classList.contains("source-review-flagged-heading"));
    assert.equal(await ignore.isDisabled(), true);
    assert.equal(committed, 1);
    assert.equal(commands.length, 3);
    assert.equal(new Set(commands.map(command => command.mutation_id)).size, 1);
    assert.equal(reads, beforeReads, "Success merges response without refetching review");
    assert.equal(layouts, beforeLayouts, "Ignore does not reload task-layout or body");
    assert.equal(await review.getByRole("button", { name: "Undo last content change" }).isDisabled(), true);
    await page.screenshot({ path: `/tmp/ignore-all-complete-${width}.png`, animations: "disabled" });
    // A second fixture snapshot is held while navigating away: late success cannot reopen the dialog.
    regions = [{ id: "late", page_index: 1, status: "unresolved", block_ids: [] }];
    await review.getByRole("button", { name: "Refresh text review" }).click();
    await ignore.click(); hold = true;
    const pending = page.getByRole("dialog", { name: "Ignore all 1 flagged areas?", exact: true });
    await pending.getByRole("button", { name: "Ignore all permanently" }).click();
    while (!late) await page.waitForTimeout(10);
    assert.equal(await pending.getByRole("button", { name: "Cancel", exact: true }).isDisabled(), true);
    await page.keyboard.press("Escape"); assert.equal(await pending.isVisible(), true);
    await page.evaluate(() => { history.pushState(null, "", "/"); dispatchEvent(new PopStateEvent("popstate")); });
    await review.waitFor({ state: "hidden" });
    await late.route.fulfill({ json: result(late.command) }).catch(() => {});
    await page.waitForTimeout(100);
    assert.equal(await page.locator(".source-review-ignore-all-dialog").count(), 0);
    assert.deepEqual(errors, []); assert.deepEqual(unexpected, []);
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth), false);
    await page.close();
    console.log(`PASS ${width}: 25-region scope, confirmation/Cancel/focus, duplicate prevention, failed/lost-response retry, one commit, no reload/Undo, unmount/late result`);
  }
} finally { await browser.close(); }
