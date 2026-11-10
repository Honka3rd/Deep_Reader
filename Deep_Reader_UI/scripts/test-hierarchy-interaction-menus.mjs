// Real layout, fixture content/interaction reads; no production mutations or LLM calls.
import assert from "node:assert/strict";
const { chromium } = await import(process.env.PLAYWRIGHT_MODULE || "playwright");
const name = "Madame_Bovary_261021_155934";
const title = "Brief Bibliography of Gustave Flaubert";
const response = await fetch(`${process.env.READER_API || "http://127.0.0.1:8000"}/documents/task-layout`, {
  method: "POST", headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ doc_name: name, refresh_task_units: false,
    task_unit_split_mode: "progressive", include_anchor_page_evidence: false }),
});
assert.ok(response.ok, `Real layout read failed: ${response.status} ${response.ok ? "" : await response.text()}`);
const realLayout = await response.json();
const realChapter = realLayout.chapters.find(chapter => chapter.title === title);
assert.equal(realChapter?.sections.length, 1);
assert.equal(realChapter.sections[0].task_units.length, 451);
const section = (id, title) => ({ section_id: id, title, task_units: [{ unit_id: `${id}-u` }] });
const cases = [
  { chapter: realChapter, merged: true },
  { chapter: { chapter_id: "different", title: "Different chapter title", sections: [section("different-s", "Different section title")] }, merged: true },
  { chapter: { chapter_id: "multiple", title: "Multiple sections", sections: [section("s1", "First section"), section("s2", "Second section")] }, merged: false },
  { chapter: { chapter_id: "empty", title: "Empty chapter", sections: [] }, merged: false },
];
const browser = await chromium.launch({ headless: true });
try {
  for (const width of [1280, 390]) {
    const page = await browser.newPage({ viewport: { width, height: 900 } });
    const errors = [], unexpected = [], reads = [], contentReads = [];
    page.on("pageerror", error => errors.push(error.message));
    await page.route("**/api/**", route => {
      const request = route.request(), url = new URL(request.url());
      const path = decodeURIComponent(url.pathname);
      if (path === "/api/documents/task-layout") return route.fulfill({ json: {
        ...realLayout, chapters: cases.map(item => item.chapter),
      } });
      if (path === "/api/documents/reparse-structure/status") return route.fulfill({ json: { doc_name: name, status: "idle" } });
      if (path.endsWith("/reading-content")) return route.fulfill({ json: {
        revision: 0, source_token: "fixture", disposition: "accepted", can_undo: false, max_chars: 10000, regions: [], blocks: [],
      } });
      if (path.endsWith("/task-units/content")) {
        const ids = request.postDataJSON().task_unit_ids;
        contentReads.push(...ids);
        return route.fulfill({ json: { contents: ids.map(id => ({
          document_id: name, task_unit_id: id, chapter_id: realChapter.chapter_id,
          section_id: realChapter.sections[0].section_id,
          content_blocks: [{ block_id: `${id}:content:0`, block_type: "paragraph", content: "Reading selection fixture." }],
        })) } });
      }
      if (path === "/api/documents/reading-interactions/insight/read") {
        const target = request.postDataJSON().target;
        reads.push(target);
        return route.fulfill({ json: { envelope: {
          target: { ...target, target_id: target.section_id || target.chapter_id, document_id: name },
          interaction_type: "analysis", status: "completed", schema_version: "analysis_interaction_v1",
        }, payload: { summary: "Saved insight", reasoning: "Reason", interpretation: "Interpretation" } } });
      }
      unexpected.push(`${request.method()} ${path}`);
      return route.fulfill({ status: 500, json: { detail: "Unexpected API" } });
    });
    await page.goto(`${process.env.UI_URL || "http://127.0.0.1:5173"}/documents/${encodeURIComponent(name)}`);
    await page.locator(".hierarchy-chapter-item").first().waitFor();
    for (const [index, item] of cases.entries()) {
      const row = page.locator(".hierarchy-chapter-item").nth(index);
      if (!item.merged && item.chapter.sections.length > 0) {
        await row.locator(':scope > .hierarchy-tree-content .hierarchy-tree-icon').click();
        await row.locator('.hierarchy-section-item').first().waitFor();
      }
      assert.equal(await row.locator(".hierarchy-chapter-merged-row").count(), item.merged ? 1 : 0);
      assert.equal(await row.locator(".hierarchy-chapter-interaction-trigger").count(), 1);
      assert.equal(await row.locator(".hierarchy-section-interaction-trigger").count(), item.merged ? 0 : item.chapter.sections.length);
      const trigger = row.getByRole("button", { name: `Chapter reading interactions for ${item.chapter.title}`, exact: true });
      await trigger.click();
      for (const label of ["Summary", "Note", "Insights", "Quiz", "Critical thinking"])
        assert.equal(await page.getByRole("menuitem", { name: label, exact: true }).count(), 1);
      await page.getByRole("menuitem", { name: "Insights", exact: true }).click();
      await row.locator(".inline-insight-status-completed").waitFor();
      assert.equal(reads.at(-1).target_type, "chapter");
      assert.equal(reads.at(-1).chapter_id, item.chapter.chapter_id);
      assert.ok(!reads.at(-1).section_id, "Chapter action must not become a section action");
    }
    const multi = page.locator(".hierarchy-chapter-item").nth(2);
    for (const child of cases[2].chapter.sections) {
      await multi.getByRole("button", { name: `Section reading interactions for ${child.title}`, exact: true }).click();
      await page.getByRole("menuitem", { name: "Insights", exact: true }).click();
      await multi.locator(".hierarchy-section-item").filter({ hasText: child.title }).locator(".inline-insight-status-completed").waitFor();
      assert.equal(reads.at(-1).target_type, "section");
      assert.equal(reads.at(-1).chapter_id, "multiple");
      assert.equal(reads.at(-1).section_id, child.section_id);
    }
    const merged = page.locator(".hierarchy-chapter-item").first();
    await merged.locator(".hierarchy-section-button").click();
    await page.locator(".reader-content-block-list").waitFor();
    assert.equal(await merged.getAttribute("aria-selected"), "true");
    assert.deepEqual(contentReads, realChapter.sections[0].task_units.map(unit => unit.unit_id));
    assert.match(await page.locator(".reader-content-pagination-bar").innerText(), /Page 1 \/ 451/);
    await page.screenshot({ path: `/tmp/hierarchy-interaction-menus-${width}.png`, fullPage: true });
    assert.deepEqual(errors, []);
    assert.deepEqual(unexpected, []);
    await page.close();
    console.log(`PASS ${width}px: real 451-unit merged chapter has one menu; distinct titles, multiple/empty sections, chapter/section API targets and body selection`);
  }
} finally { await browser.close(); }
