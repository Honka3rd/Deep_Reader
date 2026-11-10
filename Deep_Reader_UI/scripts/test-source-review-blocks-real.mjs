// Actual App/API/SQL, natural Madame Bovary Page 3 multi-block review, backup only.
import assert from "node:assert/strict";
import { writeFile } from "node:fs/promises";
const { chromium } = await import(process.env.PLAYWRIGHT_MODULE || "playwright");
const api = process.env.SRREPAIR_API, evidence = process.env.SRREPAIR_EVIDENCE;
assert.ok(api && evidence);
const browser = await chromium.launch({ headless: true, channel: process.env.PLAYWRIGHT_CHANNEL || "chrome" });
const call = async (path, body) => {
  const response = await fetch(api + path, body ? { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) } : {});
  assert.ok(response.ok, `${path}: ${response.status}`);
  return response.json();
};
const reports = [];
try {
  for (const width of [1280, 390]) {
    const fixture = await call("/fixture", {});
    const { name, region_id: regionId, block_id: blockId } = fixture;
    const page = await browser.newPage({ viewport: { width, height: 900 } });
    page.setDefaultTimeout(60000);
    const errors = [], requests = [], mutations = [];
    let loseResponse = false;
    page.on("pageerror", e => errors.push(e.message));
    await page.route("**/api/**", async route => {
      const request = route.request(), url = new URL(request.url()), path = url.pathname.replace(/^\/api/, "");
      if (["/documents", "/documents/management"].includes(path)) return route.fulfill({ json: { items: [], total: 0 } });
      if (path.endsWith("/reparse-structure/status")) return route.fulfill({ json: { doc_name: name, status: "idle" } });
      requests.push({ path, query: url.search, method: request.method() });
      const mutation = path.endsWith("/reading-content") && request.method() === "POST";
      if (mutation) mutations.push(request.postDataJSON());
      const response = await route.fetch({ url: api + path + url.search, timeout: 0 });
      if (!response.ok()) errors.push(`${path}: ${response.status()}`);
      if (mutation && loseResponse) { loseResponse = false; return route.abort("failed"); }
      await route.fulfill({ response });
    });
    const review = page.locator('[aria-labelledby="reading-content-dialog-title"]');
    const editor = page.locator('[aria-labelledby="source-review-block-editor-title"]');
    const region = review.locator(`.source-review-region[data-region-id="${regionId}"]`);
    const row = region.locator(`.source-review-block[data-block-id="${blockId}"]`);
    const open = async () => {
      await page.getByRole("button", { name: "Review extracted text", exact: true }).click();
      await review.getByLabel("Loading content changes").waitFor({ state: "hidden" });
      await review.getByRole("tab", { name: "Flagged areas", exact: true }).click();
      while (!await region.count()) await review.getByRole("button", { name: "Next flagged areas", exact: true }).click();
      if (!await region.locator(".source-review-blocks").count()) await region.getByRole("button", { name: /^Related blocks/ }).click();
      await region.getByLabel("Loading region blocks").waitFor({ state: "hidden" });
    };
    const select = async (target, action) => {
      await target.locator(".source-review-block-menu-trigger").click();
      await page.getByRole("menuitem", { name: action, exact: true }).click();
      await editor.waitFor();
    };
    const save = async (text, target = row) => {
      await select(target, "Edit");
      await editor.getByRole("textbox", { name: "Block text", exact: true }).fill(text);
      await editor.getByRole("button", { name: "Save", exact: true }).click();
      await editor.getByRole("button", { name: "Keep and save", exact: true }).click();
      await editor.waitFor({ state: "hidden" });
      await region.getByLabel("Loading region blocks").waitFor({ state: "hidden" });
    };
    const counts = () => ({ layouts: requests.filter(r => r.path.endsWith("/task-layout")).length,
      content: requests.filter(r => r.path.endsWith("/task-units/content")).length,
      reads: requests.filter(r => r.path.endsWith("/reading-content") && r.method === "GET" && !r.query.includes("refresh_mutation_id")).length });
    try {
      await page.goto((process.env.UI_URL || "http://127.0.0.1:5173") + "/documents/" + encodeURIComponent(name));
      await open();
      assert.equal(await review.locator(".source-region-repair-menu-trigger").count(), 0);
      assert.ok(await region.locator(".source-review-block-menu-trigger").count() >= 2);
      assert.ok(await region.locator(".source-review-block").count() <= 20);
      const original = await call("/stats"), baseline = counts();
      const node = await row.elementHandle();
      for (const text of ["Block-specific first correction", "Block-specific second correction"]) {
        await save(text);
        assert.equal(await node.evaluate(el => el.isConnected), true);
        assert.equal(await row.locator(".source-review-block-content").innerText(), text);
        const state = await call("/stats");
        assert.equal(state.blocks.find(b => b.block_id === blockId).manually_edited, false);
        assert.deepEqual(state.blocks.filter(b => b.block_id !== blockId), original.blocks.filter(b => b.block_id !== blockId));
        assert.deepEqual(counts(), baseline, "Edits must not reload layout, section or review lists");
      }
      // Cancel Keep is read-only; commit Keep survives reopening and preserves body.
      await select(row, "Keep it");
      assert.match(await editor.innerText(), /cannot be edited here again/);
      await page.screenshot({ path: `${evidence}/block-keep-${width}.png`, animations: "disabled" });
      await editor.getByRole("button", { name: "Cancel", exact: true }).click();
      const beforeKeep = await call("/stats");
      await select(row, "Keep it");
      loseResponse = true;
      await editor.getByRole("button", { name: "Keep it", exact: true }).click();
      await editor.getByRole("alert").waitFor();
      await editor.getByRole("button", { name: "Keep it", exact: true }).click();
      await editor.waitFor({ state: "hidden" });
      await row.waitFor({ state: "detached" });
      assert.equal(mutations.at(-1).mutation_id, mutations.at(-2).mutation_id);
      const afterKeep = await call("/stats");
      assert.equal(afterKeep.revision, beforeKeep.revision + 1);
      assert.equal(afterKeep.blocks.find(b => b.block_id === blockId).content, "Block-specific second correction");
      assert.equal(afterKeep.blocks.find(b => b.block_id === blockId).deleted, false);
      assert.ok(afterKeep.kept[blockId]);
      assert.equal(afterKeep.region.block_ids.includes(blockId), false);
      await review.getByRole("button", { name: "Close content editor", exact: true }).click();
      await open();
      assert.equal(await row.count(), 0);
      // Delete is scoped to one remaining block. The adjacent DOM stays connected.
      const remaining = region.locator(".source-review-block");
      const deleteRow = remaining.first(), nextNode = await remaining.nth(1).elementHandle();
      const deleteId = await deleteRow.getAttribute("data-block-id");
      await select(deleteRow, "Delete");
      await editor.getByRole("button", { name: "Delete block", exact: true }).click();
      await editor.waitFor({ state: "hidden" });
      await region.locator(`.source-review-block[data-block-id="${deleteId}"]`).waitFor({ state: "detached" });
      assert.equal(await nextNode.evaluate(el => el.isConnected), true);
      const afterDelete = await call("/stats");
      assert.equal(afterDelete.blocks.find(b => b.block_id === deleteId).deleted, true);
      assert.deepEqual(afterDelete.blocks.filter(b => b.block_id !== deleteId), afterKeep.blocks.filter(b => b.block_id !== deleteId));
      assert.equal(counts().layouts, baseline.layouts);
      assert.equal(counts().content, baseline.content);
      await page.screenshot({ path: `${evidence}/block-list-${width}.png`, animations: "disabled" });
      assert.equal(await review.evaluate(el => el.scrollWidth > el.clientWidth), false);
      assert.deepEqual(errors, []);
      reports.push({ width, initialBlockCount: original.blocks.length, counts: counts(), status: "passed" });
      console.log(`PASS ${width}: Page 3 ${original.blocks.length} blocks, repeat single-block Edit, Keep/cancel/replay/reopen, Delete, local DOM`);
    } catch (error) {
      await page.screenshot({ path: `${evidence}/block-failure-${width}.png` });
      throw error;
    } finally { await page.close(); }
  }
  await writeFile(`${evidence}/block-browser-report.json`, JSON.stringify(reports, null, 2));
} finally { await browser.close(); }
