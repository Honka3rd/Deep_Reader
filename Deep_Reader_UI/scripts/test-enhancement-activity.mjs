// Actual App lifecycle with controlled HTTP; no generation or user-book writes.
import assert from "node:assert/strict";
const { chromium } = await import(process.env.PLAYWRIGHT_MODULE || "playwright");
const browser = await chromium.launch({ headless: true,
  ...(process.env.CHROMIUM_PATH ? { executablePath: process.env.CHROMIUM_PATH } : {}) });
try {
  for (const width of [1280, 390]) {
    const page = await browser.newPage({ viewport: { width, height: 900 } });
    const calls = [], errors = [];
    let activity = "idle", unavailable = false, hold = false, release;
    page.on("pageerror", error => errors.push(error.message));
    await page.addInitScript(() => {
      sessionStorage.setItem("deep-reader:enhancement:v1:Book", JSON.stringify({
        hint: "auto", data: { confirmation_token: "obsolete", expires_at: 1 },
      }));
      // Browser storage being disabled for Enhance must not affect this workflow.
      for (const method of ["getItem", "setItem", "removeItem"]) {
        const original = Storage.prototype[method];
        Storage.prototype[method] = function(key, ...args) {
          if (key.startsWith("deep-reader:enhancement:")) throw new Error("Enhance used browser persistence");
          return original.call(this, key, ...args);
        };
      }
    });
    await page.route("**/api/**", async route => {
      const url = new URL(route.request().url()), path = url.pathname;
      calls.push(path);
      if (path.endsWith("/reparse-structure/status")) {
        const name = url.searchParams.get("doc_name"), status = activity;
        if (hold) await new Promise(resolve => { release = resolve; });
        return route.fulfill({ status: unavailable ? 503 : 200,
          json: unavailable ? { detail: "enhancement_status_unavailable" } : { doc_name: name, status } });
      }
      if (path.endsWith("/reparse-structure/preflight")) return route.fulfill({ json: {
        ready: true, execution_available: true, structure_version: 3,
        interaction_count: 0, includes_user_notes: false,
        cleanup_scope: "all_document_interactions_and_local_drafts", cleanup_condition: "structure_changed_only",
        capacity: { status: "fits", full_source: true }, confirmation_token: "fresh-consent",
        expires_at: Date.now() / 1000 + 300,
      } });
      if (path.endsWith("/task-layout")) return route.fulfill({ json: {
        document_id: route.request().postDataJSON().doc_name, chapters: [
          { chapter_id: "c", title: "Chapter", sections: [{ section_id: "s", title: "Section", task_units: [] }] },
        ],
      } });
      if (path.endsWith("/reading-content")) return route.fulfill({ json: {
        revision: 3, source_token: "current", disposition: "ready", can_undo: false, max_chars: 10000, regions: [], blocks: [],
      } });
      if (path.endsWith("/documents") || path.endsWith("/management")) return route.fulfill({ json: { items: [], total: 0 } });
      throw new Error(`Unexpected request: ${path}`);
    });
    const base = process.env.UI_URL || "http://127.0.0.1:5173";
    await page.goto(`${base}/documents/Book`);
    await page.waitForResponse(response => response.url().includes("/reparse-structure/status"));
    assert.ok(calls.some(path => path.endsWith("/status")), "Mount asks backend without opening a dialog");
    assert.ok(!calls.some(path => path.endsWith("/preflight")), "Mount does not create consent");
    const dialog = page.getByRole("dialog"), submit = page.getByRole("button", { name: "Enhance", exact: true });
    async function open() {
      await page.locator("#repair-menu-button").click();
      await page.getByRole("menuitem", { name: "LLM enhance", exact: true }).click();
      await page.getByLabel("Checking readiness").waitFor({ state: "hidden" });
    }
    const close = () => dialog.getByRole("button", { name: "Cancel", exact: true }).click();
    await open();
    assert.equal(await submit.isEnabled(), true, "Stale session storage cannot block new OCR structure");
    await close();
    activity = "running";
    await open();
    await dialog.getByText(/still running/).waitFor();
    assert.equal(await submit.isDisabled(), true);
    activity = "idle";
    await dialog.getByRole("button", { name: "Check stored state" }).click();
    await dialog.getByText("Ready for smart structuring.", { exact: true }).waitFor();
    assert.equal(await submit.isEnabled(), true);
    await close();
    unavailable = true;
    await open();
    await dialog.getByText(/Could not check readiness/).waitFor();
    assert.equal(await submit.isDisabled(), true);
    unavailable = false;
    await dialog.getByRole("button", { name: "Check again", exact: true }).click();
    await dialog.getByText("Ready for smart structuring.", { exact: true }).waitFor();
    await close();
    activity = "invalid";
    await open();
    await dialog.getByText(/Could not check readiness/).waitFor();
    await close();
    activity = "idle";
    await page.reload();
    await open();
    await dialog.getByText("Ready for smart structuring.", { exact: true }).waitFor();
    await page.screenshot({ path: `/tmp/enhancement-activity-${width}.png` });
    await close();
    hold = true;
    const opening = open();
    for (let n = 0; n < 100 && !release; n++) await page.waitForTimeout(20);
    assert.ok(release);
    await close();
    hold = false; release();
    await opening;
    await page.waitForTimeout(100);
    assert.equal(await dialog.count(), 0, "Late status response does not reopen dialog");
    assert.equal(calls.filter(path => path.endsWith("/reparse-structure")).length, 0);
    assert.equal(calls.filter(path => path.endsWith("/result")).length, 0);
    assert.deepEqual(errors, []);
    console.log(`PASS ${width}px: startup/reload, no browser persistence, active-to-idle, HTTP/malformed failures, late response`);
    await page.close();
  }
} finally { await browser.close(); }
