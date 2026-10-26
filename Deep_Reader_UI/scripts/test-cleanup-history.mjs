// Actual App and reader pagination with controlled transport; no production writes.
import assert from "node:assert/strict";
import { cleanupAction, cleanupDisabled, cleanupMenu } from "./cleanup-menu-driver.mjs";
const { chromium } = await import(process.env.PLAYWRIGHT_MODULE || "playwright");
const browser = await chromium.launch({ headless: true,
  ...(process.env.CHROMIUM_PATH ? { executablePath: process.env.CHROMIUM_PATH } : {}) });
try {
  for (const width of [1280, 390]) {
    const page = await browser.newPage({ viewport: { width, height: 900 } });
    page.setDefaultTimeout(12000);
    const errors = [], commands = [], receipts = new Set(), refreshes = new Map();
    let revision = 1, layouts = 0, failRestore = false, conflict = false, legacy = false;
    let hold = false, held;
    let failRefresh = false, refreshReads = 0, contentReads = 0, reviewReads = 0;
    const original = "Readable text remains. " + "A paragraph for reading. ".repeat(30);
    let content = "Start. " + original;
    const link = { block_id: "b", chapter_id: "c", section_id: "s", task_unit_id: "u", char_from: 7, char_to: 11, can_navigate: true };
    const rows = Array.from({ length: 23 }, (_, i) => ({ record_id: i.toString(16).padStart(64, "a"),
      quote: "@@?? " + i, text: "@@?? " + i, reason: "ocr_garbage", page_indices: [98],
      links: [link], status: "cleaned", can_restore: true, can_delete: true, restore_reason: null }));
    rows[2].links = [{ ...link, can_navigate: false }];
    const state = query => ({ document_id: "Book", revision, structure_version: "1", source_token: "source",
      disposition: "completed", can_undo: false, max_chars: 10000, regions: [],
      blocks: query?.has("block_id") ? [{ ...link, content, deleted: false, modified: false, manually_edited: false }] : [],
      ...(query?.has("include_cleanup_audit") ? { cleanup_audit: { total: 1, offset: 0, limit: 10, items: [{
        revision: 1, at: "2026-10-25T10:00:00Z", applied_count: rows.length, skipped_count: 0,
        remaining_count: rows.filter(r => r.can_restore).length, undoable: false,
        deletions: legacy ? rows.map(r => ({ quote: r.quote, reason: r.reason, page_indices: r.page_indices })) : rows,
      }] } } : {}) });
    page.on("pageerror", error => errors.push(error.message));
    page.on("dialog", dialog => dialog.accept());
    await page.route("**/api/**", async route => {
      const url = new URL(route.request().url()), path = url.pathname;
      if (path.endsWith("/reading-interactions/note/read")) return route.fulfill({ json: {
        target: route.request().postDataJSON().target, target_token: "note-fixture", max_chars: 10000, note: null,
      } });
      if (path.endsWith("/reparse-structure/status")) return route.fulfill({ json: { doc_name: "Book", status: "idle" } });
      if (path.endsWith("/task-layout")) { layouts++; return route.fulfill({ json: { document_id: "Book", chapters: [
        { chapter_id: "c", title: "Chapter", sections: [{ section_id: "s", title: "Body", task_units: ["u", "later1", "later2"].map(unit_id => ({ unit_id })) }] },
      ] } }); }
      if (path.endsWith("/reading-content")) {
        if (url.searchParams.has("refresh_mutation_id")) {
          refreshReads++;
          if (failRefresh) { failRefresh = false; return route.fulfill({ status: 503, json: { detail: "Temporary read failure" } }); }
          return route.fulfill({ json: refreshes.get(url.searchParams.get("refresh_mutation_id")) });
        }
        if (route.request().method() === "POST") {
          const command = route.request().postDataJSON(); commands.push(command);
          if (conflict) { conflict = false; return route.fulfill({ status: 409, json: { detail: "reading_content_changed_reload_required" } }); }
          if (!receipts.has(command.mutation_id)) {
            assert.equal(command.expected_revision, revision);
            const row = rows.find(item => item.record_id === command.cleanup_record_id);
            assert.ok(row); assert.equal(command.block_id, undefined); assert.equal(command.char_from, undefined);
            if (command.text !== undefined) row.text = command.text;
            if (command.action === "cleanup_restore") {
              assert.equal(row.can_restore, true); row.status = "restored";
              content = "Start. " + row.text + " " + original;
            } else if (command.action === "cleanup_delete") {
              assert.equal(command.text, undefined);
              rows.splice(rows.indexOf(row), 1);
            } else assert.equal(command.action, "cleanup_edit");
            row.edited_at = "2026-10-25T11:00:00Z"; revision++; receipts.add(command.mutation_id);
            refreshes.set(command.mutation_id, { document_id: "Book", structure_version: "1", revision,
              mutation_revision: revision, source_token: "source", mutation_id: command.mutation_id,
              cleanup_record_id: row.record_id, scope_current: true, generated_invalidated: command.action === "cleanup_restore",
              ...(command.action !== "cleanup_restore" ? { cleanup_change: { batch_revision: 1, record_id: row.record_id,
                deleted: command.action === "cleanup_delete", record: command.action === "cleanup_delete" ? null : structuredClone(row),
                applied_count: rows.length, remaining_count: rows.filter(r => r.status === "cleaned").length } } : {}),
              block_page: { total: command.action === "cleanup_restore" ? 1 : 0, offset: 0, limit: 20 },
              blocks: command.action === "cleanup_restore" ? [{ ...link, block_order: 0, unit_source_hash: "hash" + revision,
                manually_edited: false, deleted: false, content_block: { block_id: "b", block_type: "paragraph", content } }] : [] });
          }
          if (failRestore) { failRestore = false; return route.abort(); }
          return route.fulfill({ json: { ...state(), refresh: refreshes.get(command.mutation_id) } });
        }
        if (hold && url.searchParams.has("block_id")) { held = { route, json: structuredClone(state(url.searchParams)) }; return; }
        reviewReads++;
        return route.fulfill({ json: state(url.searchParams) });
      }
      if (path.endsWith("/task-units/content")) { contentReads++; return route.fulfill({ json: { contents: [{
        document_id: "Book", chapter_id: "c", section_id: "s", task_unit_id: "u",
        content_blocks: [{ block_id: "b", block_type: "paragraph", content }],
      }, ...["later1", "later2"].map(id => ({ document_id: "Book", chapter_id: "c", section_id: "s", task_unit_id: id,
        content_blocks: [{ block_id: id, block_type: "paragraph", content: "Later passage. ".repeat(150) }] }))] } }); }
      if (path.endsWith("/documents") || path.endsWith("/management")) return route.fulfill({ json: { items: [], total: 0 } });
      throw new Error(`Unexpected request: ${path}`);
    });
    await page.goto((process.env.UI_URL || "http://127.0.0.1:5173") + "/documents/Book");
    await page.locator(".hierarchy-section-button").first().click();
    await page.locator(".reader-content-block-body").first().waitFor();
    await page.waitForFunction(() => !document.querySelector('.reader-content-page-last')?.disabled);
    await page.getByRole("button", { name: "Last page", exact: true }).click();
    const pageBefore = await page.locator(".reader-content-pagination-bar p").innerText();
    const review = page.locator('[aria-labelledby="reading-content-dialog-title"]');
    const expand = async () => {
      await review.getByRole("button", { name: /^(Expand|Collapse) cleanup 1$/ }).waitFor();
      await review.getByLabel("Loading content changes").waitFor({ state: "hidden" });
      const button = review.getByRole("button", { name: "Expand cleanup 1", exact: true });
      if (await button.count()) await button.click();
      await review.locator(".cleanup-history-record").first().waitFor();
    };
    const row = index => review.locator(`.cleanup-history-record[data-record-id="${rows[index].record_id}"]`);
    const edit = index => cleanupAction(page, row(index), "Edit");
    await page.locator(".hierarchy-chapter-interaction-trigger").first().click();
    await page.getByRole("menuitem", { name: "Note", exact: true }).click();
    await page.getByRole("textbox", { name: "Note", exact: true }).fill("Keep my unsaved Note");
    await page.getByRole("button", { name: "Review source", exact: true, includeHidden: true }).evaluate(el => el.click()); await expand();
    assert.equal(await review.locator(".cleanup-history-record").count(), 10);
    assert.equal(await row(2).getByRole("button", { name: "Go to content for block b" }).isDisabled(), true, "Non-navigable history remains visible");
    assert.equal(await row(0).locator('.source-review-navigate-action [data-testid="ArrowForwardIcon"]').count(), 1);
    assert.equal(await row(0).locator('.source-review-item-menu-trigger [data-testid="MoreVertIcon"]').count(), 1);
    await cleanupMenu(page, row(0));
    await page.screenshot({ path: `/tmp/cleanup-history-menu-${width}.png`, animations: "disabled" });
    await page.keyboard.press("Escape");
    await edit(0); await row(0).getByRole("textbox", { name: "Restoration text", exact: true }).fill("Restored passage");
    const historyTab = review.getByRole("tab", { name: "OCR cleanup history", exact: true });
    const regionsTab = review.getByRole("tab", { name: "Source regions", exact: true });
    assert.equal(await historyTab.getAttribute("aria-selected"), "true");
    const draftNode = await row(0).elementHandle(), tabReads = reviewReads;
    const scroll = review.locator('.source-review-dialog-content');
    await scroll.evaluate(el => { el.scrollTop = 120; });
    const savedScroll = await scroll.evaluate(el => el.scrollTop);
    await regionsTab.click();
    await review.getByText("No unresolved OCR regions recorded.").waitFor();
    assert.equal(await row(0).isVisible(), false, "Only the selected panel is visible");
    await regionsTab.focus();
    await page.keyboard.press("ArrowLeft");
    await page.keyboard.press("Enter");
    await row(0).waitFor();
    assert.equal(await draftNode.evaluate(el => el.isConnected), true, "Tab switching preserves record DOM");
    assert.equal(await scroll.evaluate(el => el.scrollTop), savedScroll, "History scroll is retained");
    assert.equal(await row(0).getByRole("textbox", { name: "Restoration text", exact: true }).inputValue(), "Restored passage");
    assert.equal(reviewReads, tabReads, "Switching tabs does not refetch history");
    assert.equal(await row(0).evaluate(el => getComputedStyle(el).borderTopStyle), "solid");
    await page.screenshot({ path: `/tmp/source-review-history-tabs-${width}.png`, animations: "disabled" });
    await review.getByRole("button", { name: "Next cleanup items" }).click();
    await page.getByRole("button", { name: "Keep editing" }).click();
    assert.equal(await row(0).getByRole("textbox", { name: "Restoration text", exact: true }).inputValue(), "Restored passage");
    const beforeSave = layouts, readsBeforeSave = reviewReads;
    const modalNode = await review.elementHandle(), otherRow = await row(1).elementHandle();
    await row(0).getByRole("button", { name: "Save restoration text" }).click(); await expand();
    assert.equal(layouts, beforeSave, "Saving a history draft does not reload layout");
    assert.equal(reviewReads, readsBeforeSave, "Saving patches the response, without a review GET");
    assert.equal(await modalNode.evaluate(el => el === document.querySelector('[aria-labelledby="reading-content-dialog-title"]')), true);
    assert.equal(await otherRow.evaluate(el => el.isConnected), true, "Other rows are not recreated");
    await row(0).getByText("Restored passage", { exact: true }).waitFor();
    await row(0).getByText("@@?? 0", { exact: true }).waitFor();
    failRestore = true;
    await cleanupAction(page, row(0), "Keep it");
    await review.getByText(/could not be confirmed/).waitFor();
    const beforeRestore = layouts;
    const retryMenu = await cleanupMenu(page, row(0));
    await retryMenu.getByRole("menuitem", { name: "Keep it", exact: true }).evaluate(el => { el.click(); el.click(); });
    await expand();
    assert.equal(commands.at(-1).mutation_id, commands.at(-2).mutation_id, "Lost response replays same mutation ID");
    assert.equal(layouts - beforeRestore, 0, "No layout refresh after confirmed restore");
    assert.equal(contentReads, 1, "No section content reload");
    assert.equal(await page.locator(".reader-content-pagination-bar p").innerText(), pageBefore, "Refresh preserves current page");
    assert.equal(await cleanupDisabled(page, row(0), "Keep it"), false);
    await edit(0); await row(0).getByRole("textbox", { name: "Restoration text", exact: true }).fill("History remains editable");
    await row(0).getByRole("button", { name: "Save restoration text" }).click(); await expand();
    assert.ok(content.includes("Restored passage") && !content.includes("History remains editable"));
    const mutationsBeforeApply = commands.length, readsBeforeApply = refreshReads;
    failRefresh = true;
    await cleanupAction(page, row(0), "Keep it");
    await review.getByText("Changes saved. Reading refresh failed.").waitFor();
    await review.getByRole("button", { name: "Retry reading" }).click();
    await review.getByText("Changes saved. Reading refresh failed.").waitFor({ state: "hidden" });
    assert.equal(commands.length, mutationsBeforeApply + 1, "Read retry does not repeat the committed mutation");
    assert.equal(refreshReads, readsBeforeApply + 2);
    assert.equal(layouts, beforeRestore); assert.equal(contentReads, 1);
    assert.equal(await page.locator('.reader-content-block-list [data-block-id="b"]').count(), 0,
      "Targeted refresh must not mount an off-page block");
    assert.equal(await page.locator(".reader-content-pagination-bar p").innerText(), pageBefore);
    assert.equal(await page.getByRole("textbox", { name: "Note", exact: true, includeHidden: true }).inputValue(), "Keep my unsaved Note");
    await row(0).getByRole("button", { name: "Go to content for block b" }).click();
    await review.waitFor({ state: "hidden" });
    await page.waitForFunction(() => document.activeElement?.getAttribute("data-block-id") === "b");
    const updatedBody = await page.locator('.reader-content-block-list [data-block-id="b"] .reader-content-block-body').textContent();
    assert.equal(updatedBody.includes("History remains editable"), true);
    assert.equal(updatedBody.includes("Restored passage"), false);
    await page.getByRole("button", { name: "Review source", exact: true }).click(); await expand();
    await edit(0); await row(0).getByRole("textbox", { name: "Restoration text", exact: true }).fill("Third replacement");
    await cleanupAction(page, row(0), "Keep it"); await expand();
    await page.waitForFunction(() => document.querySelector('[aria-labelledby="reading-content-dialog-title"]')?.contains(document.activeElement)).catch(async error => {
      await page.screenshot({ path: `/tmp/review-tabs-focus-failure-${width}.png` });
      console.error(await page.evaluate(() => ({ active: document.activeElement?.outerHTML.slice(0, 500),
        busy: document.querySelector('.cleanup-history')?.getAttribute('aria-busy') })));
      throw error;
    });
    assert.equal(layouts, beforeRestore, "Repeat after navigation still avoids layout refresh");
    assert.ok(content.includes("Third replacement") && !content.includes("History remains editable"));
    await edit(1); await row(1).getByRole("textbox", { name: "Restoration text", exact: true }).fill("Unsubmitted draft");
    conflict = true;
    await cleanupAction(page, row(1), "Keep it");
    await review.getByText(/changed elsewhere/).waitFor();
    assert.equal(await row(1).getByRole("textbox", { name: "Restoration text", exact: true }).inputValue(), "Unsubmitted draft");
    await row(1).getByRole("button", { name: "Cancel record edit" }).click();
    await page.getByRole("button", { name: "Discard changes" }).click();
    rows[1].can_restore = false; rows[1].restore_reason = "block_edited";
    await review.getByRole("button", { name: "Reload", exact: true }).click(); await expand();
    await row(1).getByText("The block was manually edited.").waitFor();
    assert.equal(await cleanupDisabled(page, row(1), "Keep it"), true);
    await review.getByRole("button", { name: "Next cleanup items" }).click();
    assert.equal(await review.locator(".cleanup-history-record").count(), 10);
    await review.getByRole("button", { name: "Next cleanup items" }).click();
    assert.equal(await review.locator(".cleanup-history-record").count(), 3);
    await review.getByRole("button", { name: "Previous cleanup items" }).click();
    await review.getByRole("button", { name: "Previous cleanup items" }).click();
    await row(0).scrollIntoViewIfNeeded();
    await edit(0);
    await page.screenshot({ path: `/tmp/cleanup-history-${width}.png`, animations: "disabled" });
    await row(0).getByRole("button", { name: "Cancel record edit" }).click();
    assert.equal(await review.evaluate(el => el.scrollWidth > el.clientWidth), false);
    const deleteModalNode = await review.elementHandle();
    await edit(4); await row(4).getByRole("textbox", { name: "Restoration text", exact: true }).fill("Position-preserving draft");
    await row(4).getByRole("button", { name: "Save restoration text" }).scrollIntoViewIfNeeded();
    const scrollBeforeSave = await review.locator(".source-review-dialog-content").evaluate(el => el.scrollTop);
    const readsBeforeLocalSave = reviewReads;
    await row(4).getByRole("button", { name: "Save restoration text" }).click(); await expand();
    await page.waitForTimeout(100);
    assert.equal(reviewReads, readsBeforeLocalSave);
    const scrollAfterSave = await review.locator(".source-review-dialog-content").evaluate(el => el.scrollTop);
    assert.ok(Math.abs(scrollAfterSave - scrollBeforeSave) < 2,
      `Save preserves modal scroll offset: ${scrollBeforeSave} -> ${scrollAfterSave}`);
    const beforeDelete = { reads: reviewReads, layouts, contentReads, content, commands: commands.length };
    await cleanupAction(page, row(2), "Delete");
    const confirmation = page.getByRole("dialog", { name: "Delete cleanup record?", exact: true });
    await confirmation.getByRole("button", { name: "Cancel", exact: true }).click();
    await page.waitForFunction(() => document.activeElement?.classList.contains("cleanup-history-record-menu-trigger"));
    assert.equal(commands.length, beforeDelete.commands, "Cancel never sends a mutation");
    const deletedId = rows[2].record_id, retainedRow = await row(3).elementHandle();
    await cleanupAction(page, row(2), "Delete");
    failRestore = true;
    await confirmation.getByRole("button", { name: "Delete record", exact: true }).click();
    await confirmation.waitFor({ state: "hidden" });
    await review.getByText(/could not be confirmed/).waitFor();
    await cleanupAction(page, review.locator(`[data-record-id="${deletedId}"]`), "Delete");
    await confirmation.getByRole("button", { name: "Delete record", exact: true }).click();
    await confirmation.waitFor({ state: "hidden" });
    await review.locator(`[data-record-id="${deletedId}"]`).waitFor({ state: "detached" });
    await page.waitForFunction(() => document.querySelector('.cleanup-history')?.contains(document.activeElement));
    assert.equal(commands.at(-1).mutation_id, commands.at(-2).mutation_id, "Delete lost response reuses receipt");
    assert.equal(await retainedRow.evaluate(el => el.isConnected), true);
    assert.equal(await deleteModalNode.evaluate(el => el.isConnected), true);
    assert.equal(reviewReads, beforeDelete.reads);
    assert.equal(layouts, beforeDelete.layouts); assert.equal(contentReads, beforeDelete.contentReads);
    assert.equal(content, beforeDelete.content, "Delete history does not change reading text");
    await review.getByRole("button", { name: "Next cleanup items" }).click();
    await review.getByRole("button", { name: "Next cleanup items" }).click();
    for (let n = 0; n < 2; n++) {
      await cleanupAction(page, review.locator(".cleanup-history-record").last(), "Delete");
      await confirmation.getByRole("button", { name: "Delete record", exact: true }).click();
      await confirmation.waitFor({ state: "hidden" });
    }
    assert.equal(await review.locator(".cleanup-history-record").count(), 10, "Empty final page returns to nearest page");
    assert.equal(await review.locator(".cleanup-history-record-pagination").innerText(), "11-20 / 20");
    assert.equal(reviewReads, beforeDelete.reads, "Delete pagination uses the committed local delta");
    await review.getByRole("button", { name: "Previous cleanup items" }).click();
    hold = true; held = null;
    await row(0).getByRole("button", { name: "Go to content for block b" }).click();
    for (let n = 0; n < 100 && !held; n++) await page.waitForTimeout(10);
    assert.ok(held);
    await regionsTab.click();
    hold = false; await held.route.fulfill({ json: held.json }); held = null;
    await page.waitForTimeout(100);
    assert.equal(await review.isVisible(), true, "Inactive history cannot navigate after a late response");
    assert.equal(await regionsTab.getAttribute('aria-selected'), 'true');
    await historyTab.click();
    hold = true;
    await row(0).getByRole("button", { name: "Go to content for block b" }).click();
    for (let n = 0; n < 100 && !held; n++) await page.waitForTimeout(10);
    assert.ok(held);
    await review.getByRole("button", { name: "Close content editor" }).click();
    hold = false; await held.route.fulfill({ json: held.json });
    await page.waitForTimeout(250);
    assert.equal(await review.isVisible(), false);
    legacy = true;
    await page.getByRole("button", { name: "Review source", exact: true }).click(); await expand();
    for (const action of ["Keep it", "Edit", "Delete"]) {
      assert.equal(await cleanupDisabled(page, review.locator(".cleanup-history-record").first(), action), true);
    }
    await review.getByRole("button", { name: "Close content editor" }).click();
    legacy = false; rows.splice(1);
    await page.getByRole("button", { name: "Review source", exact: true }).click(); await expand();
    const lastReads = reviewReads;
    await cleanupAction(page, review.locator(".cleanup-history-record"), "Delete");
    await confirmation.getByRole("button", { name: "Delete record", exact: true }).click();
    await confirmation.waitFor({ state: "hidden" });
    await review.getByText("No retained cleanup records.").waitFor();
    await review.getByText("History entries removed").waitFor();
    assert.equal(reviewReads, lastReads);
    assert.equal(await review.locator(".cleanup-history-record").count(), 0);
    await page.waitForFunction(() => document.querySelector('.cleanup-history')?.contains(document.activeElement));
    assert.deepEqual(errors, []);
    console.log(`PASS ${width}px: draft/save/restore/replay/navigation, flags, conflicts, pagination, close and legacy`);
    await page.close();
  }
} finally { await browser.close(); }
