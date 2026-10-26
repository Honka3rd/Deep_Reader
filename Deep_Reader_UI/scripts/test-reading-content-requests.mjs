// Production App in StrictMode, with intercepted APIs and no real document writes.
import assert from "node:assert/strict";
const { chromium } = await import(process.env.PLAYWRIGHT_MODULE || "playwright");
const browser = await chromium.launch({ headless: true,
  ...(process.env.CHROMIUM_PATH ? { executablePath: process.env.CHROMIUM_PATH } : {}) });
const layout = name => ({ document_id: name, title: name, chapters: [{
  chapter_id: `${name}-c`, title: `${name} chapter`, sections: [{
    section_id: `${name}-s`, title: `${name} section`, task_units: [{ unit_id: `${name}-u` }],
  }],
}] });
const readingState = (name, canUndo = false) => ({ revision: 1, source_token: name,
  disposition: "ready", can_undo: canUndo, max_chars: 10000, regions: [], blocks: [],
  cleanup_audit: { total: 0, offset: 0, limit: 10, items: [] } });
try {
  for (const width of [1280, 390]) {
    const page = await browser.newPage({ viewport: { width, height: 900 } });
    const errors = [], unexpected = [];
    let hold = false, held, failReview = false;
    page.on("pageerror", error => errors.push(error.message));
    // Count fetch invocations, including requests aborted before Playwright routing.
    await page.addInitScript(() => {
      const original = window.fetch;
      window.readingRequests = [];
      window.fetch = function(input, init) {
        const url = new URL(String(input), location.href);
        if (url.pathname.endsWith("/reading-content")) {
          const request = { path: url.pathname, query: url.search, method: init?.method || "GET", aborted: false };
          window.readingRequests.push(request);
          init?.signal?.addEventListener("abort", () => { request.aborted = true; }, { once: true });
        }
        return original.apply(this, arguments);
      };
    });
    await page.route("**/api/**", async route => {
      const url = new URL(route.request().url());
      if (url.pathname === "/api/documents/reparse-structure/status") return route.fulfill({ json: { doc_name: url.searchParams.get("doc_name"), status: "idle" } });
      if (url.pathname === "/api/documents") return route.fulfill({ json: {
        items: ["Alpha", "Beta"].map(name => ({ doc_name: name, title: name, source: "structured" })), total: 2,
      } });
      if (url.pathname === "/api/documents/task-layout")
        return route.fulfill({ json: layout(route.request().postDataJSON().doc_name) });
      if (url.pathname.endsWith("/reading-content") && route.request().method() === "GET") {
        const name = decodeURIComponent(url.pathname.split("/")[3]);
        if (hold && name === "Beta") { held = route; return; }
        if (failReview && url.searchParams.has("include_cleanup_audit"))
          return route.fulfill({ status: 503, json: { detail: "Review unavailable" } });
        return route.fulfill({ json: readingState(name) });
      }
      unexpected.push(`${route.request().method()} ${url.pathname}`);
      return route.fulfill({ status: 500, json: { detail: "Unexpected API" } });
    });
    const select = async name => {
      await page.getByRole("combobox").fill(name);
      await page.getByRole("option", { name, exact: true }).click();
    };
    const review = page.getByRole("button", { name: "Review source", exact: true });
    const panel = page.locator('[aria-labelledby="reading-content-dialog-title"]');
    const settled = () => page.waitForFunction(() =>
      document.querySelector('button[aria-label="Review source"]')?.disabled === false);
    const requests = () => page.evaluate(() => window.readingRequests);
    const expectCount = async (count, message) => {
      await page.waitForTimeout(150);
      assert.equal((await requests()).length, count, message);
    };
    await page.goto(process.env.UI_URL || "http://127.0.0.1:5173");
    await select("Alpha");
    await settled();
    await expectCount(1, "StrictMode document selection must issue one overview request");
    await page.getByRole("combobox").fill("uncommitted search");
    await page.getByRole("combobox").press("Escape");
    await expectCount(1, "Unrelated parent renders must not reload reading state");
    await review.click();
    await panel.getByText("No unresolved OCR regions recorded.").waitFor();
    await expectCount(2, "Opening review loads audit once");
    assert.equal(new URLSearchParams((await requests())[1].query).get("include_cleanup_audit"), "true");
    await page.screenshot({ path: `/tmp/reading-requests-${width}.png`, fullPage: true });
    await panel.getByRole("button", { name: "Cancel", exact: true }).click();
    await settled();
    await expectCount(3, "Closing review refreshes overview once");
    failReview = true;
    await review.click();
    await panel.getByText("Review unavailable", { exact: true }).waitFor();
    await expectCount(4, "Failure must not cause automatic duplicate retry");
    failReview = false;
    await panel.getByRole("button", { name: "Reload", exact: true }).click();
    await panel.getByText("No unresolved OCR regions recorded.").waitFor();
    await expectCount(5, "Explicit retry must fetch fresh state once");
    await panel.getByRole("button", { name: "Cancel", exact: true }).click();
    await settled();
    await expectCount(6, "Review close remains bounded after retry");
    hold = true;
    await select("Beta");
    await page.waitForFunction(() => window.readingRequests.some(r => r.path.includes("/Beta/")));
    await expectCount(7, "A newly mounted document also loads once");
    await select("Alpha");
    await settled();
    await expectCount(8, "Returning to a document must revalidate, not reuse a stale cache");
    assert.equal((await requests()).find(r => r.path.includes("/Beta/")).aborted, true,
      "Switching documents must abort the obsolete request");
    assert.ok(held, "The slow document request reached the transport");
    await held.fulfill({ json: readingState("Beta", true) }).catch(() => {});
    await page.waitForTimeout(150);
    assert.equal(await page.getByRole("button", { name: "Undo last content change", exact: true }).isDisabled(), true,
      "Late response must not overwrite the current document");
    await page.reload();
    await settled();
    await expectCount(1, "Direct-route reload must issue one request under StrictMode");
    assert.deepEqual(errors, []);
    assert.deepEqual(unexpected, [], "Reading state loads must not start mutations or preparation");
    await page.close();
    console.log(`PASS ${width}px: selector/route StrictMode counts, parent renders, review, retry, switch/abort, stale isolation and fresh remount`);
  }
} finally { await browser.close(); }
