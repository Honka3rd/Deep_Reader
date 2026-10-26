// Actual App -> real FastAPI -> disposable SQL clone of incident 19652:content:2.
import assert from "node:assert/strict";
import { cleanupAction, cleanupDisabled } from "./cleanup-menu-driver.mjs";
import { writeFile } from "node:fs/promises";
const { chromium } = await import(process.env.PLAYWRIGHT_MODULE || "playwright");
const api = process.env.RCR_API, evidence = process.env.RCR_EVIDENCE;
assert.ok(api && evidence, "Run through accept-repeatable-cleanup.py");
const browser = await chromium.launch({ headless: true,
  ...(process.env.CHROMIUM_PATH ? { executablePath: process.env.CHROMIUM_PATH } : {}) });
const report = [];
const call = async (path, body) => {
  const response = await fetch(api + path, body ? { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) } : {});
  assert.ok(response.ok, `${path}: ${response.status} ${response.ok ? "" : await response.text()}`);
  return response.json();
};
try {
  for (const width of [1280, 390]) {
    const fixture = await call("/fixture", {});
    const { name, block_id: blockId, record_id: recordId } = fixture;
    const page = await browser.newPage({ viewport: { width, height: 900 } });
    page.setDefaultTimeout(45000);
    const errors = [], requests = [], mutations = [], steps = [];
    page.on("pageerror", error => errors.push(error.message));
    await page.route("**/api/**", async route => {
      const request = route.request(), url = new URL(request.url()), path = url.pathname.replace(/^\/api/, "");
      // Unrelated library/status reads are UI fixtures, never routed to production.
      if (["/documents", "/documents/management"].includes(path)) return route.fulfill({ json: { items: [], total: 0 } });
      if (path.endsWith("/reparse-structure/status")) return route.fulfill({ json: { doc_name: name, status: "idle" } });
      requests.push({ path, query: url.search, method: request.method() });
      if (path.endsWith("/reading-content") && request.method() === "POST") mutations.push(request.postDataJSON());
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
      await page.getByRole("button", { name: "Review source", exact: true }).click();
      await ready();
      await review.getByRole("button", { name: `Expand cleanup ${fixture.batch_revision}`, exact: true }).click();
      for (let i = 0; i < Math.floor(fixture.record_index / 10); i++) {
        await review.getByRole("button", { name: "Next cleanup items", exact: true }).click();
      }
      await row.waitFor();
    };
    const counts = () => ({ layouts: requests.filter(r => r.path.endsWith("/task-layout")).length,
      contents: requests.filter(r => r.path.endsWith("/task-units/content")).length,
      reads: requests.filter(r => r.path.endsWith("/reading-content") && r.method === "GET").length,
      refreshes: requests.filter(r => r.query.includes("refresh_mutation_id")).length });
    try {
      await page.goto((process.env.UI_URL || "http://127.0.0.1:5173") + "/documents/" + encodeURIComponent(name));
      await open();
      let historyPage = Math.floor(fixture.record_index / 10);
      const goHistoryPage = async target => {
        while (historyPage !== target) {
          const next = historyPage < target;
          await review.getByRole("button", { name: next ? "Next cleanup items" : "Previous cleanup items", exact: true }).click();
          historyPage += next ? 1 : -1;
        }
      };
      assert.ok(fixture.non_navigable.length, "Incident includes naturally non-navigable history");
      for (const retained of fixture.non_navigable) {
        await goHistoryPage(Math.floor(retained.index / 10));
        const retainedRow = review.locator(`.cleanup-history-record[data-record-id="${retained.record_id}"]`);
        await retainedRow.waitFor();
        for (const button of await retainedRow.getByRole("button", { name: /^Go to content for block / }).all()) {
          assert.equal(await button.isDisabled(), true);
        }
      }
      await goHistoryPage(Math.floor(fixture.record_index / 10));
      await row.getByRole("button", { name: /^Go to content for block / }).click();
      await review.waitFor({ state: "hidden" });
      const body = page.locator(`.reader-content-block-list [data-block-id="${blockId}"]`);
      await body.waitFor();
      await open();
      const baseline = counts();
      let previous = fixture.initial.block.content;
      for (const text of [fixture.initial.record.text, "RCR second replacement", "RCR third replacement"]) {
        if (text !== fixture.initial.record.text) {
          await cleanupAction(page, row, "Edit");
          await row.getByRole("textbox", { name: "Restoration text", exact: true }).fill(text);
          const reads = counts().reads, node = await row.elementHandle();
          await row.getByRole("button", { name: "Save restoration text" }).click();
          await ready();
          assert.equal(counts().reads, reads, "History Save does not GET the review again");
          assert.equal(await node.evaluate(el => el.isConnected), true);
          const saved = await call("/stats");
          assert.equal(saved.record.text, text);
          assert.equal(saved.block.content, previous, "Draft Save leaves applied body untouched");
          assert.equal(saved.block.manually_edited, false);
        }
        await cleanupAction(page, row, "Keep it");
        await ready();
        const state = await call("/stats");
        assert.equal(state.block.manually_edited, false);
        assert.equal(state.record.can_restore, true);
        assert.ok(state.block.content.includes(text));
        assert.equal(state.block.content.split(text).length - 1, 1, "Replacement appears once");
        assert.ok(state.applications[recordId]);
        assert.equal(state.sql_copies_equal, true);
        assert.equal(counts().layouts, baseline.layouts, "Restore does not reload layout");
        assert.equal(counts().contents, baseline.contents, "Restore does not reload section content");
        await page.waitForFunction(({ blockId, text }) => [...document.querySelectorAll('[data-block-id]')]
          .some(el => el.getAttribute('data-block-id') === blockId && el.querySelector('.reader-content-block-body')?.textContent.includes(text)), { blockId, text });
        if (text === "RCR third replacement") assert.ok(!state.block.content.includes("RCR second replacement"));
        previous = state.block.content;
        steps.push({ text, revision: state.revision, counts: counts(), manually_edited: state.block.manually_edited });
      }
      assert.equal(counts().refreshes - baseline.refreshes, 3);
      assert.equal(await review.evaluate(el => el.scrollWidth > el.clientWidth), false);
      await row.scrollIntoViewIfNeeded();
      await page.screenshot({ path: `${evidence}/history-${width}.png`, animations: "disabled" });
      await row.getByRole("button", { name: /^Go to content for block / }).click();
      await review.waitFor({ state: "hidden" });
      await body.waitFor();
      await page.screenshot({ path: `${evidence}/body-${width}.png`, animations: "disabled" });
      await body.getByRole("button", { name: "Content-block reading interactions for " + blockId, exact: true }).click();
      await page.getByRole("menuitem", { name: "Edit", exact: true }).click();
      await page.getByLabel("Block text").fill(previous + " Manual body edit.");
      await page.getByRole("button", { name: "Save", exact: true }).click();
      await page.getByRole("button", { name: "Keep and save", exact: true }).click();
      await review.waitFor({ state: "hidden" });
      await open();
      await row.getByText("The block was manually edited.").waitFor();
      assert.equal(await cleanupDisabled(page, row, "Keep it"), true);
      const protectedState = await call("/stats");
      assert.equal(protectedState.block.manually_edited, true);
      assert.equal(protectedState.record.can_restore, false);
      const current = await call(`/documents/${encodeURIComponent(name)}/reading-content?block_id=${encodeURIComponent(blockId)}`);
      const rejected = await fetch(`${api}/documents/${encodeURIComponent(name)}/reading-content`, {
        method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({
          action: "cleanup_restore", cleanup_record_id: recordId, expected_revision: current.revision,
          source_token: current.source_token, mutation_id: `protected-${width}`,
        }),
      });
      assert.equal(rejected.status, 409);
      assert.match(await rejected.text(), /cleanup_block_edited/);
      assert.equal((await call("/stats")).sql_sha256, protectedState.sql_sha256, "Rejected restore writes nothing");
      const beforeDelete = counts(), reviewNode = await review.elementHandle();
      await cleanupAction(page, row, "Delete");
      const confirmation = page.getByRole("dialog", { name: "Delete cleanup record?", exact: true });
      await page.screenshot({ path: `${evidence}/delete-${width}.png`, animations: "disabled" });
      await confirmation.getByRole("button", { name: "Delete record", exact: true }).click();
      await confirmation.waitFor({ state: "hidden" });
      await row.waitFor({ state: "detached" });
      const deleted = await call("/stats");
      assert.equal(deleted.record, null);
      assert.deepEqual(deleted.block, protectedState.block, "Deleting history leaves edited content untouched");
      assert.equal(deleted.sql_copies_equal, true);
      assert.deepEqual(counts(), beforeDelete, "Delete uses only the mutation response, no review/layout/content GET");
      assert.equal(await reviewNode.evaluate(el => el.isConnected), true);
      await review.getByRole("button", { name: "Close content editor" }).click();
      await page.getByRole("button", { name: "Review source", exact: true }).click();
      await ready();
      await review.getByRole("button", { name: `Expand cleanup ${fixture.batch_revision}`, exact: true }).click();
      assert.equal(await row.count(), 0, "Removed history stays absent when reopened");
      assert.deepEqual(errors, []);
      report.push({ width, steps, protected_revision: protectedState.revision, requests, mutations,
        retained_non_navigable_records: fixture.non_navigable.length,
        blocked_restore_status: rejected.status, source: "unmodified SQL incident backup",
        synthetic: ["library listing", "structuring status", "generated availability disabled"] });
      console.log(`RCR-05 ${width}: three replacements, draft isolation, targeted refresh, SQL/manual-edit protection passed`);
    } catch (error) {
      await page.screenshot({ path: `${evidence}/failure-${width}.png`, animations: "disabled" }).catch(() => {});
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
