// Actual App/FastAPI/disposable SQL, deterministic provider and named transport faults.
import assert from "node:assert/strict";
const api = process.env.CLEANUP_TEST_API;
assert.match(api || "", /^http:\/\/127\.0\.0\.1:\d+$/);
const { chromium } = await import(process.env.PLAYWRIGHT_MODULE || "playwright");
const browser = await chromium.launch({ headless: true,
  ...(process.env.CHROMIUM_PATH ? { executablePath: process.env.CHROMIUM_PATH } : {}) });
try {
  for (const width of [1280, 390]) {
    const page = await browser.newPage({ viewport: { width, height: 900 } });
    const errors = [], requests = [], executions = [];
    let lost = false, unknown = false, failRead = false, failResume = false, holdLookup = false, holdContent = false, release;
    page.on("pageerror", error => errors.push(error.message));
    await page.route("**/api/**", async route => {
      const url = new URL(route.request().url()), path = url.pathname.replace(/^\/api/, "");
      requests.push(path);
      assert.notEqual(path, "/documents/prepare-task-layout");
      assert.notEqual(path, "/documents/prepare");
      if (["/documents", "/documents/management"].includes(path)) return route.fulfill({ json: { items: [], total: 0 } });
      if (path.endsWith("/note/read")) return route.fulfill({ json: {
        target: route.request().postDataJSON().target, target_token: "note-fixture", max_chars: 10000, note: null } });
      // Only the preview bitmap is synthetic; review/Ignore/Undo still use real APIs.
      if (path.endsWith("/image")) return route.fulfill({ contentType: "image/png", body: Buffer.from(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jD1sAAAAASUVORK5CYII=", "base64") });
      if (path.endsWith("/task-units/content") && failRead) {
        failRead = false; return route.fulfill({ status: 503, json: { detail: "Temporary read failure" } });
      }
      const lookup = /\/smart-cleanup\/[0-9a-f]{32}$/.test(path);
      if (lookup) {
        assert.equal(route.request().method(), "GET");
        assert.equal(route.request().postData(), null);
        if (unknown) return route.fulfill({ status: 503, json: {
          operation_id: path.split("/").at(-1), phase: "unknown", commit: "unknown", outcome: null,
          reason: "cleanup_reconciliation_unavailable", retry_action: "reconcile", reading_ready: null,
          pending_review_count: null, durability: "process_local" } });
      }
      if (path.endsWith("/smart-cleanup")) {
        executions.push(route.request().postDataJSON());
        if (failResume) { failResume = false; return route.fulfill({ status: 409, json: { detail: "cleanup_in_progress" } }); }
      }
      const response = await route.fetch({ url: api + path + url.search });
      if (path.endsWith("/task-units/content") && holdContent) await new Promise(resolve => { release = resolve; });
      if (lookup && holdLookup) await new Promise(resolve => { release = resolve; });
      if (path.endsWith("/smart-cleanup") && lost) { lost = false; return route.abort(); }
      await route.fulfill({ response });
    });
    const configure = async scenario => assert.equal((await page.request.post(api + "/fixture", { data: { scenario } })).status(), 204);
    const stats = async () => (await page.request.get(api + "/stats")).json();
    const reading = async () => (await page.request.get(api + "/documents/book/reading-content?include_blocks=true&include_cleanup_audit=true")).json();
    const dialog = page.getByRole("dialog", { name: "Smart cleanup", exact: true });
    const panel = page.locator('div[aria-labelledby="reading-content-dialog-title"]');
    const load = async (scenario = "completed", select = true) => {
      await configure(scenario);
      await page.goto(`${process.env.UI_URL || "http://127.0.0.1:5174"}/documents/book`);
      await page.evaluate(() => sessionStorage.clear());
      if (select) { await page.locator(".hierarchy-section-button").first().click(); await page.locator(".reader-content-block-body").first().waitFor(); }
    };
    const open = async () => {
      await page.locator("#repair-menu-button").click();
      await page.getByRole("menuitem", { name: "Smart cleanup", exact: true }).click(); await dialog.waitFor();
    };
    const close = async () => { await dialog.locator(".smart-cleanup-close").click(); await dialog.waitFor({ state: "hidden" }); };
    const submit = async () => {
      await dialog.getByText("Ready for smart cleanup.", { exact: true }).waitFor();
      await dialog.getByRole("button", { name: "Clean text", exact: true }).click();
    };
    const review = async () => {
      await page.getByRole("button", { name: "Review source", exact: true }).click();
      await panel.getByLabel("Loading content changes").waitFor({ state: "hidden" });
    };
    const closeReview = async () => { await panel.getByRole("button", { name: "Close content editor" }).click(); await panel.waitFor({ state: "hidden" }); };

    for (const scenario of ["failed", "partial"]) {
      await load(scenario);
      await page.locator("#repair-menu-button").click();
      await page.getByRole("menuitem", { name: "LLM enhance", exact: true }).click();
      const enhance = page.getByRole("dialog", { name: "Smart structuring", exact: true });
      await enhance.getByText("Ready for smart structuring.", { exact: true }).waitFor();
      await enhance.getByRole("button", { name: "Enhance", exact: true }).click();
      await enhance.getByRole("button", { name: "Smart cleanup", exact: true }).waitFor();
      const before = await stats();
      await configure("provider_ready");
      await enhance.getByRole("button", { name: "Smart cleanup", exact: true }).click();
      await dialog.getByText("Ready for smart cleanup.", { exact: true }).waitFor();
      assert.deepEqual(await stats(), before, "Opening retry only preflights");
      await submit();
      await dialog.getByText(scenario === "failed" ? /OCR cleanup completed/ : /Cleanup was skipped/).waitFor();
      const after = await stats();
      assert.equal(after.calls, before.calls + 1); assert.equal(after.builds, before.builds);
      await close();
    }

    await load("failed"); await open(); await submit();
    await dialog.getByText(/Cleanup could not finish/).waitFor();
    await configure("provider_ready");
    const first = executions.at(-1).operation_id;
    await dialog.getByRole("button", { name: "Retry cleanup" }).click();
    await dialog.getByText("Ready for smart cleanup.", { exact: true }).waitFor();
    assert.equal((await stats()).calls, 1);
    await submit(); await dialog.getByText(/OCR cleanup completed/).waitFor();
    assert.notEqual(executions.at(-1).operation_id, first); assert.equal((await stats()).calls, 2); await close();

    await load(); const ids = (await reading()).blocks.map(block => block.block_id);
    await open(); lost = true; await submit();
    await dialog.getByRole("button", { name: "Check stored state" }).waitFor();
    const saved = await stats(), posts = executions.length;
    unknown = true;
    await dialog.getByRole("button", { name: "Check stored state" }).click();
    await dialog.getByText(/result is not confirmed/).waitFor();
    assert.equal(await dialog.getByRole("button", { name: "Clean text" }).count(), 0);
    assert.equal(await dialog.getByRole("button", { name: "Retry cleanup" }).count(), 0);
    unknown = false; await configure("restart");
    await page.reload(); await open(); await dialog.getByText(/OCR cleanup completed/).waitFor();
    assert.deepEqual(await stats(), saved); assert.equal(executions.length, posts);
    assert.deepEqual((await reading()).blocks.map(block => block.block_id), ids);
    await close();

    await load(); await open(); failRead = true; await submit();
    await dialog.getByRole("button", { name: "Retry reading" }).waitFor();
    const refreshed = await stats();
    await dialog.getByRole("button", { name: "Retry reading" }).click();
    await dialog.getByText(/OCR cleanup completed/).waitFor();
    assert.deepEqual(await stats(), refreshed); await close();

    await load(); await open();
    await dialog.getByText("Ready for smart cleanup.", { exact: true }).waitFor();
    await page.clock.install(); holdContent = true; release = null;
    await submit();
    for (let n = 0; n < 100 && !release; n++) await page.waitForTimeout(50);
    assert.ok(release);
    await page.clock.fastForward(120001);
    assert.equal(await dialog.getByRole("button", { name: "Check stored state" }).count(), 0);
    await page.getByLabel("Cleanup in progress").waitFor();
    const waiting = await stats();
    holdContent = false; release(); await page.clock.resume();
    await dialog.getByText(/OCR cleanup completed/).waitFor();
    assert.deepEqual(await stats(), waiting); await close();

    await load("standalone_layout_failed", false); await open(); await submit();
    await dialog.getByRole("button", { name: "Resume reading" }).waitFor();
    const layoutBefore = await stats(), original = executions.at(-1);
    await configure("restart"); await page.reload(); await open();
    await dialog.getByRole("button", { name: "Resume reading" }).waitFor();
    failResume = true;
    await dialog.getByRole("button", { name: "Resume reading" }).click();
    await dialog.getByRole("button", { name: "Check stored state" }).click();
    await dialog.getByRole("button", { name: "Resume reading" }).waitFor();
    await configure("layout_ready");
    await dialog.getByRole("button", { name: "Resume reading" }).evaluate(el => { el.click(); el.click(); });
    await dialog.getByText(/OCR cleanup completed/).waitFor();
    assert.deepEqual(executions.at(-1), original); assert.deepEqual(await stats(), layoutBefore); await close();
    await page.locator(".hierarchy-section-button").first().click(); await page.locator(".reader-content-block-body").first().waitFor();

    await load(); await open(); await dialog.getByText("Ready for smart cleanup.", { exact: true }).waitFor();
    await configure("stale"); await submit();
    await dialog.getByText(/document or cleanup requirements changed/).waitFor();
    assert.equal((await stats()).calls, 0); await close();
    await page.locator(".reader-content-block-body").filter({ hasText: "Concurrent user edit." }).first().waitFor();

    await load(); await open(); lost = true; await submit();
    await dialog.getByRole("button", { name: "Check stored state" }).waitFor();
    await configure("undo"); const undone = await stats();
    await dialog.getByRole("button", { name: "Check stored state" }).click();
    await dialog.getByText(/earlier reading version/).waitFor();
    assert.deepEqual(await stats(), undone); await close();
    await page.locator(".reader-content-block-body").filter({ hasText: "@@??" }).first().waitFor();

    await load(); await open(); await submit(); await dialog.getByText(/OCR cleanup completed/).waitFor(); await close();
    await page.locator(".hierarchy-chapter-interaction-trigger").first().click();
    await page.getByRole("menuitem", { name: "Note", exact: true }).click();
    await page.getByRole("textbox", { name: "Note", exact: true }).fill("Keep my draft through cleanup Undo.");
    await page.getByRole("button", { name: "Review source", exact: true, includeHidden: true }).evaluate(el => el.click());
    await panel.getByText("No unresolved OCR regions recorded.").waitFor();
    await panel.getByRole("button", { name: "Expand cleanup 1", exact: true }).click();
    await panel.locator("blockquote").filter({ hasText: "@@??" }).waitFor();
    await page.screenshot({ path: `/tmp/smart-cleanup-recovery-audit-${width}.png`, animations: "disabled" });
    await panel.getByRole("button", { name: "Collapse cleanup 1", exact: true }).click();
    const beforeUndo = await stats();
    await panel.getByRole("button", { name: "Undo OCR cleanup", exact: true }).click();
    await panel.getByText("Undone", { exact: true }).waitFor();
    assert.equal((await stats()).calls, beforeUndo.calls); assert.equal((await stats()).revision, beforeUndo.revision + 1);
    assert.equal((await reading()).regions.length, 1); await closeReview();
    assert.equal(await page.getByRole("textbox", { name: "Note", exact: true }).inputValue(), "Keep my draft through cleanup Undo.");
    page.once("dialog", prompt => prompt.accept());
    await page.getByRole("region", { name: "Note", exact: true }).getByRole("button", { name: "Close", exact: true }).click();

    await load("remaining_regions"); await open(); await submit(); await dialog.getByText(/OCR cleanup completed/).waitFor(); await close();
    await review(); await panel.getByText("Source regions (2)", { exact: true }).waitFor();
    assert.equal((await reading()).regions.some(region => region.id === "r1"), false);
    await panel.getByRole("button", { name: "Original", exact: true }).first().click();
    await panel.getByRole("img").waitFor();
    await panel.getByRole("button", { name: "Hide original", exact: true }).click();
    assert.equal(await panel.getByRole("img").count(), 0);
    await configure("stale"); const concurrent = await reading();
    await panel.getByRole("button", { name: "Undo OCR cleanup", exact: true }).click();
    await panel.getByText(/Content changed elsewhere/).waitFor();
    assert.deepEqual(await reading(), concurrent);
    await panel.getByRole("button", { name: "Reload", exact: true }).click();
    await panel.getByText("Later changes prevent batch undo").waitFor();
    await page.screenshot({ path: `/tmp/smart-cleanup-recovery-conflict-${width}.png`, animations: "disabled" });
    const regionBefore = (await reading()).regions[0].id;
    await panel.getByRole("button", { name: "Original", exact: true }).first().click();
    await panel.getByRole("button", { name: "Ignore region", exact: true }).first().click();
    const ignore = page.getByRole("dialog", { name: "Ignore this OCR region?", exact: true });
    await ignore.getByRole("button", { name: "Ignore region", exact: true }).click();
    await ignore.waitFor({ state: "hidden" });
    await panel.getByText("Source regions (1)", { exact: true }).waitFor();
    assert.equal((await reading()).regions.some(region => region.id === regionBefore), false);
    assert.equal(await panel.getByRole("img").count(), 0);
    assert.equal((await stats()).calls, 1);
    await closeReview();

    await load(); await open(); lost = true; await submit();
    await dialog.getByRole("button", { name: "Check stored state" }).waitFor();
    holdLookup = true; release = null;
    await dialog.getByRole("button", { name: "Check stored state" }).click();
    for (let n = 0; n < 100 && !release; n++) await page.waitForTimeout(50);
    assert.ok(release);
    await page.clock.fastForward(120001);
    assert.equal(await dialog.getByText("Checking stored cleanup state...", { exact: true }).isVisible(), true);
    await page.evaluate(() => { history.pushState({}, "", "/"); dispatchEvent(new PopStateEvent("popstate")); });
    await dialog.waitFor({ state: "hidden" }); holdLookup = false; release();
    await page.waitForTimeout(150);
    assert.equal(await page.getByText(/OCR cleanup completed/).count(), 0);
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth), false);
    assert.deepEqual(errors, []);
    console.log(`PASS ${width}px recovery: auto failure/partial -> confirmed standalone, same-operation GET/restart, unknown, read retry, layout resume, stale, Undo/supersession, review, drafts, stable IDs and late-response isolation`);
    await page.close();
  }
} finally { await browser.close(); }
