// Actual App with controlled HTTP; session preferences and exclusive focus destinations.
import assert from "node:assert/strict";
import { book as readingBook } from "./learning-controller-fixtures.mjs";
import { mkdir } from "node:fs/promises";
const { chromium } = await import(process.env.PLAYWRIGHT_MODULE || "playwright");
const output = process.env.SCREENSHOT_DIR || "/tmp/deep-reader-translation-session";
await mkdir(output, { recursive: true });
const browser = await chromium.launch({ headless: true, channel: "chrome" });
try {
  for (const width of [1280, 390]) {
    const page = await browser.newPage({ viewport: { width, height: 900 } });
    const errors = [], unexpected = [], reads = [], writes = [];
    const saved = new Map(); let failNext = false;
    page.on("pageerror", e => errors.push(e.message));
    const record = (doc, id, code, mutation = `saved-${code}`) => ({ source_language: "en", target_language: code,
      status: code === "fr" ? "partial" : "completed", translated_text: `${doc} ${id} ${code} translation.\n\n` + "Readable text. ".repeat(22),
      effective_content_hash: "hash", generated_at: "2026-10-26T12:00:00Z", updated_at: "2026-10-26T12:00:00Z",
      revision: 1, mutation_id: mutation });
    const records = body => {
      const key = `${body.doc_name}/${body.content_block_id}`;
      if (!saved.has(key)) saved.set(key, (body.content_block_id.startsWith("u2") ? ["de"] : ["fr", "de"])
        .map(code => record(body.doc_name, body.content_block_id, code)));
      return saved.get(key);
    };
    await page.route("**/api/**", async route => {
      const path = new URL(route.request().url()).pathname;
      if (path.endsWith("/reading-preferences/read")) return route.fulfill({ json: readingBook(route.request().postDataJSON().doc_name, "normal") });
      const body = route.request().method() === "POST" ? route.request().postDataJSON() : null;
      const doc = path.split("/")[3];
      if (path === "/api/documents") return route.fulfill({ json: { total: 2,
        items: ["Book", "Other"].map(doc_name => ({ doc_name, title: doc_name, source: "structured", preparation_status: "ready" })) } });
      if (path.endsWith("/reparse-structure/status")) return route.fulfill({ json: { status: "idle" } });
      if (path.endsWith("/task-layout")) return route.fulfill({ json: {
        document_id: body.doc_name, title: body.doc_name, chapters: [{ chapter_id: "c", title: "Chapter", sections: [
          { section_id: "s", title: "Section", task_units: [{ unit_id: "u1" }, { unit_id: "u2" }] },
          { section_id: "other", title: "Other section", task_units: [{ unit_id: "u3" }] },
        ] }],
      } });
      if (path.endsWith("/reading-content")) return route.fulfill({ json: { revision: 0, source_token: "fixture",
        disposition: "accepted", can_undo: false, max_chars: 10000, regions: [], blocks: [] } });
      if (path.endsWith("/task-units/content")) return route.fulfill({ json: { contents: body.task_unit_ids.map(id => ({
        document_id: doc, task_unit_id: id, chapter_id: "c", section_id: id === "u3" ? "other" : "s",
        content_blocks: [{ block_id: `${id}:content:0`, block_type: "content", content: `${doc} original ${id}.`, metadata: { source_hash: "source" } }],
      })) } });
      const target = body && { ...body, document_id: body.doc_name, target_id: body.content_block_id, target_type: "content_block" };
      if (path.endsWith("/translate/read")) {
        reads.push(body);
        if (failNext) { failNext = false; return route.fulfill({ status: 503, json: { detail: "unavailable" } }); }
        return route.fulfill({ json: { target, language_codes: ["en", "fr", "de"], default_source_language: "en",
          effective_content_hash: "hash", reading_revision: 0, translations: records(body) } });
      }
      if (path.endsWith("/translate/generate")) {
        writes.push(body);
        const next = record(body.doc_name, body.content_block_id, body.target_language, body.mutation_id);
        records(body).push(next);
        return route.fulfill({ json: { envelope: { target, status: next.status, persisted: true, reading_revision: 0, effective_content_hash: "hash" },
          payload: { source_language: next.source_language, target_language: next.target_language, translated_text: next.translated_text }, record: next } });
      }
      if (path.endsWith("/insight/read")) return route.fulfill({ json: {
        envelope: { target: { ...body.target, target_id: body.target.content_block_id, document_id: body.target.doc_name },
          interaction_type: "analysis", status: "completed", schema_version: "analysis_interaction_v1" },
        payload: { summary: "Saved insight evidence.", interpretation: "Interpretation.", explanation: "Explanation." },
      } });
      unexpected.push(path); return route.fulfill({ status: 500, json: { detail: "Unexpected request" } });
    });
    await page.goto(`${process.env.UI_URL || "http://127.0.0.1:5173"}/documents/Book`);
    const select = async id => {
      const item = page.locator(`[data-item-id="section:${id}"] .hierarchy-section-button`);
      if (!await item.isVisible()) await page.locator('[data-item-id="chapter:c"] .hierarchy-tree-icon').click();
      await item.click();
      await page.locator(`.reader-content-block[data-block-id="${id === "s" ? "u1" : "u3"}:content:0"]`).waitFor();
    };
    const block = id => page.locator(`.reader-content-block[data-block-id="${id}:content:0"]`);
    const countInline = async count => page.waitForFunction(count => document.querySelectorAll(".reader-inline-translation").length === count, count);
    const open = async (mode = "View translations") => {
      await page.locator(".reader-content-block-interaction-trigger").click();
      await page.getByRole("menuitem", { name: mode, exact: true }).click();
      if (mode === "View translations") await page.waitForFunction(() => {
        const drawer = document.querySelector(".translation-drawer");
        return drawer && !drawer.querySelector(".translation-loading") && drawer.querySelector(".translation-saved-section,.translation-empty,.translation-all-inline");
      });
      else await page.getByRole("combobox", { name: "Translate to" }).waitFor();
    };
    const display = () => page.getByRole("tab", { name: "Reading display", exact: true }).click();
    const close = async () => { await page.getByRole("button", { name: "Close translations", exact: true }).click();
      await page.locator(".translation-drawer").waitFor({ state: "detached" }); };
    const checkbox = name => page.getByRole("checkbox", { name, exact: true });
    await select("s"); assert.equal(reads.length, 0, "No preference means no background translation reads");
    await open(); await display();
    await checkbox("French (fr)").check(); await checkbox("German (de)").check();
    await page.getByRole("tab", { name: "Translations", exact: true }).click();
    await page.locator(".translation-all-inline").waitFor();
    assert.equal(await page.locator(".translation-saved-section").count(), 0, "Inline languages must not duplicate in drawer");
    await display(); await page.screenshot({ path: `${output}/${width}-session-options.png` });
    await close(); await countInline(2);
    await page.waitForFunction(() => document.activeElement?.classList.contains("reader-inline-translation-active"));
    assert.equal(await block("u1").locator(".reader-content-block-inline-translations").evaluate(el => el.previousElementSibling.className.includes("reader-content-block-body")), true);
    await open("Translate");
    await page.getByRole("combobox", { name: "Translate to" }).fill("French (fr)");
    await page.getByRole("option", { name: "French (fr)", exact: true }).click();
    await page.locator(".translation-submit").click();
    await page.locator(".translation-language-dialog").waitFor({ state: "detached" });
    assert.equal(await page.locator(".translation-drawer").count(), 0);
    await page.waitForFunction(() => document.activeElement?.getAttribute("data-language-code") === "fr");
    assert.equal(writes.length, 0, "Saved target reuse focuses inline without generation");
    await block("u1").locator(".reader-content-block-interaction-trigger").click();
    await page.locator(".reader-content-block-interaction-option-insight").click();
    await block("u1").locator(".inline-insight-content").waitFor();
    await page.locator(".reader-content-block-interaction-menu").waitFor({ state: "detached" });
    assert.equal(await block("u1").locator(".reader-content-block-inline-translations").evaluate(el => el.previousElementSibling.classList.contains("reader-content-block-inline-insight")), true);
    await page.locator(".reader-inline-translation-active").evaluate(el => el.scrollIntoView({ block: "start" }));
    await page.screenshot({ path: `${output}/${width}-inline-highlight.png` });
    await open(); await display();
    // Make the target active, then move its highlight between exclusive locations.
    await checkbox("French (fr)").uncheck(); await checkbox("French (fr)").check(); await checkbox("French (fr)").uncheck();
    await page.getByRole("tab", { name: "Translations", exact: true }).click();
    await page.locator('.translation-saved-section-active[data-language-code="fr"]').waitFor();
    assert.equal(await page.locator('.reader-inline-translation[data-language-code="fr"]').count(), 0);
    await page.getByRole("button", { name: "Reload translations", exact: true }).click();
    await page.locator('.translation-saved-section-active[data-language-code="fr"]').waitFor();
    await display(); await checkbox("French (fr)").check(); await close();
    await page.locator(".reader-content-page-next").click(); await block("u2").waitFor(); await countInline(1);
    assert.match(await block("u2").innerText(), /Book u2:content:0 de translation/);
    assert.equal(reads.filter(item => item.content_block_id === "u2:content:0").length, 1);
    await open("Translate");
    await page.getByRole("combobox", { name: "Translate to" }).fill("French (fr)");
    await page.getByRole("option", { name: "French (fr)", exact: true }).click(); await page.locator(".translation-submit").click();
    await countInline(2);
    await page.waitForFunction(() => document.activeElement?.getAttribute("data-language-code") === "fr");
    assert.equal(await page.locator(".translation-drawer").count(), 0, "New matching translation is focused inline");
    assert.equal(writes.length, 1);
    failNext = true; await select("other");
    await page.getByRole("button", { name: "Retry translations", exact: true }).click(); await countInline(2);
    await open(); await display(); await checkbox("German (de)").uncheck(); await close(); await countInline(1);
    await select("s"); await countInline(1);
    assert.equal(await page.locator('.reader-inline-translation[data-language-code="de"]').count(), 0, "Deselection applies across sections");
    await page.getByRole("combobox", { name: "Document name", exact: true }).fill("Other");
    await page.getByRole("option", { name: "Other", exact: true }).click();
    await select("s"); await countInline(1);
    assert.match(await block("u1").innerText(), /Other u1:content:0 fr translation/);
    assert.doesNotMatch(await block("u1").innerText(), /Book u1:content:0/);
    const beforeReload = reads.length;
    await page.reload(); await select("s");
    assert.equal(await page.locator(".reader-inline-translation").count(), 0);
    assert.equal(reads.length, beforeReload, "Reload resets session preference and stops implicit reads");
    await open(); await display(); assert.equal(await checkbox("French (fr)").isChecked(), false);
    assert.equal(writes.length, 1); assert.deepEqual(errors, []); assert.deepEqual(unexpected, []);
    console.log(`PASS ${width}: session scope, page/section/book reads, exclusive placement, inline/drawer focus transfer, future generation, failure retry, global deselection and reload reset`);
    await page.close();
  }
} finally { await browser.close(); }
