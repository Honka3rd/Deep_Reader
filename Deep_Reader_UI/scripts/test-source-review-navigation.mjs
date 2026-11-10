// Actual App in StrictMode; all APIs intercepted, no production writes.
import assert from "node:assert/strict";
import { book as readingBook } from "./learning-controller-fixtures.mjs";
const { chromium } = await import(process.env.PLAYWRIGHT_MODULE || "playwright");
const browser = await chromium.launch({ headless: true,
  ...(process.env.CHROMIUM_PATH ? { executablePath: process.env.CHROMIUM_PATH } : {}) });
try {
  for (const width of [1280, 390]) {
    const page = await browser.newPage({ viewport: { width, height: 900 } });
    page.setDefaultTimeout(12000);
    const errors = [], unexpected = [], reads = [];
    const units = Array.from({ length: 24 }, (_, i) => ({ unit_id: "u" + i }));
    const blocks = units.map((unit, i) => ({ block_id: "b" + i, task_unit_id: unit.unit_id,
      chapter_id: "c", section_id: "s", deleted: false, modified: false,
      content: "Paragraph " + i + ". " + "A stable paragraph with content to measure. ".repeat(15) }));
    const regions = [
      { id: "r1", page_index: 98, status: "unresolved", block_ids: ["b19", "b20"], mapping_status: "partial", body_status: "linked", mapping_reason: "partial_source_coverage" },
      { id: "r2", page_index: 101, status: "unresolved", block_ids: ["b19"], mapping_status: "complete", body_status: "linked" },
      { id: "unknown", page_index: 102, status: "unresolved", block_ids: [], mapping_status: "unavailable", body_status: "unknown" },
      { id: "many", page_index: 103, status: "unresolved", block_ids: blocks.map(b => b.block_id), mapping_status: "complete", body_status: "linked" },
    ];
    let revision = 0, token = "source", version = "1", fail = false, hold = false, held = null, legacy = false, dropTarget = false;
    let holdContent = false, heldContent = null;
    const audit = { total: 1, offset: 0, limit: 10, items: [{ revision: 0, at: "2026-10-25T09:00:00Z",
      applied_count: 1, skipped_count: 0, remaining_count: 1, undoable: true,
      deletions: [{ quote: "OCR noise", reason: "ocr_garbage", page_indices: [98] }] }] };
    const snapshot = query => {
      const region = regions.find(r => r.id === query.get("region_id"));
      const matching = region ? blocks.filter(b => region.block_ids.includes(b.block_id))
        : query.has("block_id") ? blocks.filter(b => b.block_id === query.get("block_id")) : [];
      const offset = Number(query.get("block_offset") || 0);
      return { revision, source_token: token, ...(legacy ? {} : { structure_version: version }),
        disposition: "needs_review", ocr_review_available: true, can_undo: false, max_chars: 10000,
        regions: regions.map(r => legacy ? { id: r.id, page_index: r.page_index, status: r.status, block_ids: r.block_ids } : r),
        blocks: matching.slice(offset, offset + 20).map(b => legacy ? { block_id: b.block_id, content: b.content, deleted: b.deleted } : b),
        block_page: { total: matching.length, offset, limit: 20 },
        ...(query.has("include_cleanup_audit") ? { cleanup_audit: audit } : {}) };
    };
    await page.addInitScript(() => {
      const original = window.fetch;
      window.reviewFetches = [];
      window.fetch = function(input, init) {
        const url = new URL(String(input), location.href);
        if (url.searchParams.has("region_id")) {
          const row = { region: url.searchParams.get("region_id"), aborted: false };
          window.reviewFetches.push(row);
          init?.signal?.addEventListener("abort", () => { row.aborted = true; }, { once: true });
        }
        return original.apply(this, arguments);
      };
    });
    page.on("pageerror", e => errors.push(e.message));
    await page.route("**/api/**", async route => {
      const url = new URL(route.request().url());
      if (url.pathname.endsWith("/reading-preferences/read")) return route.fulfill({ json: readingBook(route.request().postDataJSON().doc_name, "normal") });
      if (url.pathname === "/api/documents/reparse-structure/status") return route.fulfill({ json: { doc_name: url.searchParams.get("doc_name"), status: "idle" } });
      if (url.pathname === "/api/documents/task-layout") return route.fulfill({ json: {
        document_id: "Book", title: "Book", chapters: [{ chapter_id: "c", title: "Chapter",
          sections: [{ section_id: "s", title: "Body", task_units: units },
            { section_id: "other", title: "Other", task_units: [{ unit_id: "other-u" }] }] }] } });
      if (url.pathname === "/api/documents/Book/reading-content" && route.request().method() === "GET") {
        reads.push(url.search);
        if (fail && url.searchParams.has("region_id")) { fail = false; return route.abort(); }
        if (hold && url.searchParams.has("region_id")) {
          held = { route, json: structuredClone(snapshot(url.searchParams)) }; return;
        }
        return route.fulfill({ json: snapshot(url.searchParams) });
      }
      if (url.pathname === "/api/documents/Book/task-units/content") {
        const ids = route.request().postDataJSON().task_unit_ids;
        assert.ok(ids.every(id => id.startsWith("u")), "Only the selected section is read");
        reads.push("content:" + ids.join(","));
        const json = { contents: ids.map(id => ({
          document_id: "Book", chapter_id: "c", section_id: "s", task_unit_id: id,
          content_blocks: blocks.filter(b => b.task_unit_id === id && !b.deleted && !(dropTarget && b.block_id === "b19"))
            .map(b => ({ block_id: b.block_id, block_type: "paragraph", content: b.content })),
        })) };
        if (holdContent) { heldContent = { route, json }; return; }
        return route.fulfill({ json });
      }
      unexpected.push(route.request().method() + " " + url.pathname);
      return route.fulfill({ status: 500, json: { detail: "Unexpected API" } });
    });
    await page.goto((process.env.UI_URL || "http://127.0.0.1:5176") + "/documents/Book");
    const open = () => page.getByRole("button", { name: "Review extracted text", exact: true }).click();
    const review = page.locator('div[aria-labelledby="reading-content-dialog-title"]');
    const row = id => review.locator('.source-review-region[data-region-id="' + id + '"]');
    const select = async id => {
      await review.getByRole("tab", { name: "Flagged areas", exact: true }).click();
      await row(id).getByRole("button", { name: /^Related blocks/ }).click();
    };
    const ready = () => page.locator('.source-review-blocks[aria-busy="false"]').waitFor();
    await open();
    const before = reads.length;
    await select("r1"); await ready();
    assert.equal(reads.length - before, 1, "Expansion dispatches one region read in StrictMode");
    await review.getByText("Undo available", { exact: true }).waitFor({ state: "attached" });
    assert.equal(await review.getByRole("tabpanel", { name: "OCR cleanup history", exact: true }).count(), 0);
    assert.equal(await row("r1").evaluate(el => getComputedStyle(el).borderTopStyle), "solid");
    await page.screenshot({ path: `/tmp/source-review-regions-tabs-${width}.png`, animations: "disabled" });
    assert.equal(await review.getByRole("menuitem").count(), 0);
    assert.equal(await review.getByRole("button", { name: /Actions for block/ }).count(), 0);
    assert.equal(await row("r1").locator(".source-review-block").count(), 2);
    await row("r1").getByText("Some source text has no verified content link.").waitFor();
    const navBefore = reads.filter(s => s.startsWith("content:")).length;
    await row("r1").getByRole("button", { name: "Go to content", exact: true }).evaluate(el => { el.click(); el.click(); });
    await review.waitFor({ state: "hidden" });
    const target = page.locator('.reader-content-block-list [data-block-id="b19"]');
    await target.waitFor();
    await page.waitForFunction(() => document.activeElement?.getAttribute("data-block-id") === "b19");
    const highlight = target.locator('.reader-content-navigation-highlight');
    await highlight.waitFor();
    assert.equal(await highlight.textContent(), blocks[19].content);
    assert.equal(await page.locator('.reader-content-pagination-measure').count(), 0);
    await highlight.click();
    assert.equal(await highlight.count(), 1, "Clicking the highlighted text retains it");
    const numbers = (await page.locator(".reader-content-pagination-bar").innerText()).match(/Page (\d+) \/ (\d+)/);
    assert.ok(Number(numbers[1]) > 1 && Number(numbers[1]) < 99, "Measured reader page differs from PDF page 99");
    assert.equal(reads.filter(s => s.startsWith("content:")).length - navBefore, 2, "One section selection, bounded batches of 20");
    await page.setViewportSize({ width, height: 740 });
    await target.waitFor();
    await page.waitForFunction(() => document.activeElement?.getAttribute("data-block-id") === "b19");
    await page.screenshot({ path: "/tmp/source-review-navigation-" + width + ".png" });
    await page.locator('.reader-content-title').click();
    assert.equal(await highlight.count(), 0, "Click away clears the highlight");
    await page.setViewportSize({ width, height: 760 });
    assert.equal(await highlight.count(), 0, "Resize cannot revive a dismissed highlight");
    await open(); await ready();
    await row("r1").getByRole("button", { name: "Go to content", exact: true }).click();
    await review.waitFor({ state: "hidden" });
    await highlight.waitFor();
    await page.keyboard.press("Escape");
    assert.equal(await highlight.count(), 0, "Escape also clears a new navigation highlight");
    await open(); await ready();
    assert.equal(await row("r1").getByRole("button", { name: /^Related blocks/ }).getAttribute("aria-expanded"), "true");
    revision++; blocks[19].content = "Fresh manual text"; blocks[19].modified = true;
    regions[0].manual_change = { status: "modified", at: "2026-10-25T10:00:00Z" };
    const reselectBefore = reads.length;
    await select("r1"); await ready();
    await row("r1").getByText("Fresh manual text", { exact: true }).waitFor();
    assert.equal(reads.length - reselectBefore, 1, "Same-row reselection reads once");
    await row("r1").getByText("Manually modified", { exact: true }).waitFor();
    assert.equal(await row("r1").locator("time").getAttribute("datetime"), "2026-10-25T10:00:00Z");
    await review.getByText("Cleanup history is unavailable.").waitFor({ state: "attached" });
    assert.equal(await review.getByText("Undo available", { exact: true }).count(), 0, "New revision never inherits prior cleanup audit");
    audit.items[0].remaining_count = 0; audit.items[0].undoable = false;
    await review.getByRole("button", { name: "Refresh text review" }).click(); await ready();
    await review.getByText("Undone", { exact: true }).waitFor({ state: "attached" });
    await select("r2"); await ready();
    await row("r2").getByText("Fresh manual text", { exact: true }).waitFor();
    blocks[19].chapter_id = "";
    await select("r2"); await ready();
    assert.equal(await row("r2").getByRole("button", { name: "Go to content", exact: true }).isDisabled(), true);
    blocks[19].chapter_id = "missing-chapter";
    await select("r2"); await ready();
    const invalidBefore = reads.filter(value => value.startsWith("content:")).length;
    await row("r2").getByRole("button", { name: "Go to content", exact: true }).click(); await ready();
    await row("r2").getByText("This content belongs to an outdated structure. Reload the document.").waitFor();
    assert.equal(reads.filter(value => value.startsWith("content:")).length, invalidBefore, "No guessed hierarchy request");
    blocks[19].chapter_id = "c";
    for (const fault of ["revision", "source", "structure", "stale-batch", "deleted-target"]) {
      await select("r2"); await ready();
      holdContent = true; heldContent = null;
      await row("r2").getByRole("button", { name: "Go to content", exact: true }).click();
      while (!heldContent) await page.waitForTimeout(10);
      if (fault === "revision") revision++;
      if (fault === "source") token += "-changed";
      if (fault === "structure") version += "-changed";
      if (fault === "stale-batch") heldContent.json.contents.find(item => item.task_unit_id === "u19").content_blocks[0].content = "Old cached text under same IDs";
      if (fault === "deleted-target") blocks[19].deleted = true;
      holdContent = false;
      await heldContent.route.fulfill({ json: heldContent.json }); heldContent = null;
      await ready();
      await row("r2").getByText("Content changed. Refresh text review.").waitFor();
      assert.equal(await review.isVisible(), true, fault + " keeps review open");
      assert.equal(await target.count(), 0, fault + " cannot publish stale content");
      assert.equal(await page.locator('.reader-content-navigation-highlight').count(), 0, fault + " cannot highlight stale content");
      assert.notEqual(await page.evaluate(() => document.activeElement?.getAttribute("data-block-id")), "b19");
      blocks[19].deleted = false;
      await review.getByRole("button", { name: "Refresh text review" }).click(); await ready();
    }
    await select("r2"); await ready();
    holdContent = true;
    await row("r2").getByRole("button", { name: "Go to content", exact: true }).click();
    await page.waitForFunction(() => document.querySelector('.source-review-blocks[aria-busy="true"]'));
    while (!heldContent) await page.waitForTimeout(10);
    await select("unknown"); await ready();
    holdContent = false;
    await heldContent.route.fulfill({ json: heldContent.json }); heldContent = null;
    await page.waitForTimeout(50);
    assert.equal(await review.isVisible(), true, "Superseded content response cannot close review or focus a block");
    assert.equal(await row("unknown").getByRole("button", { name: "Go to content", exact: true }).isDisabled(), true);
    await select("many"); await ready();
    assert.equal(await row("many").locator(".source-review-block").count(), 20);
    await row("many").getByRole("button", { name: "Next region blocks" }).click(); await ready();
    assert.equal(await row("many").locator(".source-review-block").count(), 4);
    regions[3].block_ids = ["b0", "b1"];
    revision++;
    await row("many").getByRole("button", { name: "Refresh related blocks" }).click();
    await row("many").locator('.source-review-block[data-block-id="b0"]').waitFor(); await ready();
    assert.equal(await row("many").locator(".source-review-block").count(), 2, "Shrinking total clamps the old final page");
    assert.equal(await row("many").locator(".source-review-block-pagination").count(), 0);
    regions[3].block_ids = [];
    revision++;
    await row("many").getByRole("button", { name: "Refresh related blocks" }).click(); await ready();
    await row("many").getByText("No current related blocks.").waitFor();
    assert.equal(await row("many").getByRole("button", { name: "Go to content", exact: true }).isDisabled(), true);
    await select("r1"); await ready();
    fail = true;
    await row("r1").getByRole("button", { name: "Refresh related blocks" }).click();
    await row("r1").getByRole("status").waitFor();
    assert.equal(await row("r1").locator(".source-review-block").count(), 0, "Failed refresh clears old text");
    await row("r1").getByRole("button", { name: "Refresh related blocks" }).click(); await ready();
    // A surviving mapping that disappeared from the reader must never focus another block.
    dropTarget = true;
    await row("r1").getByRole("button", { name: "Go to content", exact: true }).click(); await ready();
    await row("r1").getByText("This block is no longer in the current content. Refresh text review.").waitFor();
    dropTarget = false;
    revision++; blocks[19].deleted = true; blocks[20].deleted = true;
    regions[0].manual_change = { status: "deleted", at: "2026-10-25T11:00:00Z" };
    regions[0].related_deleted_count = 2;
    await row("r1").getByRole("button", { name: "Go to content", exact: true }).click(); await ready();
    await row("r1").getByText("Manually deleted", { exact: true }).waitFor();
    assert.equal(await row("r1").getByRole("button", { name: "Go to content", exact: true }).isDisabled(), true);
    assert.equal(await review.locator(".source-review-region").count(), 4, "All-deleted and unknown rows remain");
    assert.equal(await row("r1").getByRole("button", { name: "Ignore region" }).isDisabled(), true, "Original still required");
    await page.screenshot({ path: "/tmp/source-review-readonly-" + width + ".png" });
    // Null optional fields may be omitted by FastAPI, and old servers cannot authorize navigation.
    legacy = true;
    await review.getByRole("button", { name: "Refresh text review" }).click(); await ready();
    assert.equal(await row("r1").locator(".source-review-manual-change").count(), 0);
    assert.equal(await row("r1").getByRole("button", { name: "Go to content", exact: true }).isDisabled(), true);
    legacy = false;
    await review.getByRole("button", { name: "Refresh text review" }).click(); await ready();
    version = "2"; token = "replacement-source";
    await row("r1").getByRole("button", { name: "Refresh related blocks" }).click(); await ready();
    await row("r1").getByText("Content changed. Refresh text review.").waitFor();
    assert.equal(await row("r1").locator(".source-review-block").count(), 0);
    await review.getByRole("button", { name: "Refresh text review" }).click(); await ready();
    assert.equal(await row("r1").locator(".source-review-block").count(), 2, "Explicit refresh re-resolves the replaced snapshot");
    hold = true; held = null;
    await select("r1");
    while (!held) await page.waitForTimeout(10);
    await review.getByRole("tab", { name: "OCR cleanup history", exact: true }).click();
    assert.equal(await review.locator('.source-review-blocks').count(), 0, "Inactive region reader is unmounted");
    assert.ok(await page.evaluate(() => window.reviewFetches.at(-1).aborted), "Tab switch cancels region requests");
    hold = false;
    await held.route.fulfill({ json: held.json }); held = null;
    assert.equal(await review.getByRole("tab", { name: "OCR cleanup history", exact: true }).getAttribute('aria-selected'), 'true');
    await select("r1"); await ready();
    hold = true; held = null;
    await select("r1");
    while (!held) await page.waitForTimeout(10);
    hold = false;
    await select("r2"); await ready();
    await held.route.fulfill({ json: held.json }); held = null;
    assert.equal(await row("r1").locator(".source-review-blocks").count(), 0, "Late response cannot reselect prior row");
    assert.ok(await page.evaluate(() => window.reviewFetches.some(row => row.region === "r1" && row.aborted)));
    hold = true;
    await select("r2");
    while (!held) await page.waitForTimeout(10);
    await review.getByRole("button", { name: "Cancel", exact: true }).click();
    hold = false;
    await held.route.fulfill({ json: held.json }); held = null;
    await review.waitFor({ state: "hidden" });
    await open(); await ready();
    await page.keyboard.press("Escape");
    await review.waitFor({ state: "hidden" });
    await open(); await ready();
    hold = true;
    await select("r2");
    while (!held) await page.waitForTimeout(10);
    await page.evaluate(() => { history.pushState({}, "", "/"); dispatchEvent(new PopStateEvent("popstate")); });
    hold = false;
    await held.route.fulfill({ json: held.json }); held = null;
    await review.waitFor({ state: "hidden" });
    assert.equal(await page.getByRole("button", { name: "Review extracted text", exact: true }).count(), 0, "Unmount discards review state");
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth), false);
    assert.deepEqual(errors, []); assert.deepEqual(unexpected, []);
    await page.close();
    console.log("PASS " + width + ": readonly navigation, measured page/focus/resize, fresh reselection, badges, shared/partial/unknown/deleted/legacy mappings, shrinking pagination, snapshot-bound content/text validation, audit invalidation, close/switch/abort/dedup");
  }
} finally { await browser.close(); }
