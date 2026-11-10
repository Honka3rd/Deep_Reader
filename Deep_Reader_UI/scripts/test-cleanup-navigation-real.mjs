// Actual App -> FastAPI -> disposable SQL backup of hidden block 20303:content:2.
import assert from "node:assert/strict";
import { cleanupAction } from "./cleanup-menu-driver.mjs";
import { writeFile } from "node:fs/promises";
const { chromium } = await import(process.env.PLAYWRIGHT_MODULE || "playwright");
const api = process.env.RCR_API, evidence = process.env.RCR_EVIDENCE;
assert.ok(api && evidence, "Run through accept-repeatable-cleanup.py with RCR_SCENARIO=hidden-navigation");
const browser = await chromium.launch({ headless: true,
  ...(process.env.CHROMIUM_PATH ? { executablePath: process.env.CHROMIUM_PATH } : {}) });
const report = [];
const call = async (path, body) => {
  const response = await fetch(api + path, body ? { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) } : {});
  assert.ok(response.ok, `${path}: ${response.status}`);
  return response.json();
};
try {
  for (const width of [1280, 390]) {
    const fixture = await call("/fixture", {});
    const { name, block_id: blockId, record_id: recordId } = fixture;
    assert.equal(fixture.initial.block.deleted, true);
    assert.equal(fixture.initial.record.links[0].can_navigate, false);
    const page = await browser.newPage({ viewport: { width, height: 900 } });
    page.setDefaultTimeout(45000);
    const errors = [], requests = [];
    page.on("pageerror", error => errors.push(error.message));
    await page.route("**/api/**", async route => {
      const request = route.request(), url = new URL(request.url()), path = url.pathname.replace(/^\/api/, "");
      if (["/documents", "/documents/management"].includes(path)) return route.fulfill({ json: { items: [], total: 0 } });
      if (path.endsWith("/reparse-structure/status")) return route.fulfill({ json: { doc_name: name, status: "idle" } });
      requests.push({ path, query: url.search, method: request.method() });
      const response = await route.fetch({ url: api + path + url.search, timeout: 0 });
      if (!response.ok()) errors.push(`${path}: ${response.status()} ${await response.text()}`);
      await route.fulfill({ response });
    });
    const review = page.locator('[aria-labelledby="reading-content-dialog-title"]');
    const row = review.locator(`.cleanup-history-record[data-record-id="${recordId}"]`);
    const ready = async () => {
      await review.getByLabel("Loading content changes").waitFor({ state: "hidden" });
      await page.waitForFunction(() => document.querySelector('.cleanup-history')?.getAttribute('aria-busy') === 'false');
    };
    const open = async () => {
      await page.getByRole("button", { name: "Review extracted text", exact: true }).click();
      await ready();
      await review.getByRole("button", { name: `Expand cleanup ${fixture.batch_revision}`, exact: true }).click();
      for (let i = 0; i < Math.floor(fixture.record_index / 10); i++) {
        await review.getByRole("button", { name: "Next cleanup items", exact: true }).click();
      }
      await row.waitFor();
    };
    const bodyReads = () => requests.filter(r => r.path.endsWith("/task-layout") || r.path.endsWith("/task-units/content")).length;
    try {
      await page.goto((process.env.UI_URL || "http://127.0.0.1:5173") + "/documents/" + encodeURIComponent(name));
      await open();
      const navigation = row.getByRole("button", { name: /^Go to content for block / });
      assert.equal(await navigation.isDisabled(), true);
      const baseline = bodyReads(), rowNode = await row.elementHandle(), dialogNode = await review.elementHandle();
      await cleanupAction(page, row, "Keep it");
      await ready();
      const restored = await call("/stats");
      assert.equal(restored.record.status, "restored");
      assert.equal(restored.record.links[0].can_navigate, true);
      assert.equal(restored.block.deleted, false);
      assert.equal(restored.block.manually_edited, false);
      assert.ok(restored.block.content.includes(fixture.initial.record.text));
      assert.equal(await navigation.isEnabled(), true);
      assert.equal(await rowNode.evaluate(el => el.isConnected), true);
      assert.equal(await dialogNode.evaluate(el => el.isConnected), true);
      assert.equal(bodyReads(), baseline, "Restore refreshes only affected blocks");
      await row.scrollIntoViewIfNeeded();
      await page.screenshot({ path: `${evidence}/navigation-history-${width}.png`, animations: "disabled" });
      await navigation.click();
      await review.waitFor({ state: "hidden" });
      const body = page.locator(`.reader-content-block-list [data-block-id="${blockId}"]`);
      await body.waitFor({ state: "visible" });
      assert.ok((await body.locator('.reader-content-block-body').innerText()).includes(fixture.initial.record.text));
      const highlight = body.locator('.reader-content-navigation-highlight');
      await highlight.waitFor();
      assert.equal(await highlight.innerText(), restored.block.content);
      await highlight.click();
      assert.equal(await highlight.count(), 1, "Clicking restored highlighted text retains it");
      await page.screenshot({ path: `${evidence}/navigation-body-${width}.png`, animations: "disabled" });
      await page.locator('.reader-content-title').click();
      assert.equal(await highlight.count(), 0, "Click away clears restored-record highlighting");
      await open();
      assert.equal(await row.getByRole("button", { name: /^Go to content for block / }).isEnabled(), true);
      await row.getByRole("button", { name: /^Go to content for block / }).click();
      await review.waitFor({ state: "hidden" });
      await highlight.waitFor();
      await page.getByRole("button", { name: "Next", exact: true }).click();
      assert.equal(await page.locator('.reader-content-navigation-highlight').count(), 0, "Paging clears the highlight");
      await page.getByRole("button", { name: "Previous", exact: true }).click();
      assert.equal(await page.locator('.reader-content-navigation-highlight').count(), 0, "Returning does not revive the highlight");
      assert.deepEqual(errors, []);
      report.push({ width, restored_revision: restored.revision, requests, block_id: blockId,
        status: "passed", source: "unmodified SQL incident backup" });
      console.log(`Hidden cleanup navigation ${width}: restore, local refresh, navigation and reopen passed`);
    } catch (error) {
      await page.screenshot({ path: `${evidence}/navigation-failure-${width}.png`, animations: "disabled" }).catch(() => {});
      console.error({ errors, requests: requests.slice(-10) });
      throw error;
    } finally {
      await page.unrouteAll({ behavior: "wait" });
      await page.close();
    }
  }
} finally {
  await writeFile(`${evidence}/browser-report.json`, JSON.stringify(report, null, 2), { mode: 0o600 });
  await browser.close();
}
