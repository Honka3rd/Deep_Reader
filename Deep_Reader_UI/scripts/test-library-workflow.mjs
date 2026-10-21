// Mocked API acceptance only: never upload, delete, OCR or generate on the real backend.
import assert from "node:assert/strict";
const { chromium } = await import(process.env.PLAYWRIGHT_MODULE || "playwright");
const browser = await chromium.launch({ headless: true,
  ...(process.env.CHROMIUM_PATH ? { executablePath: process.env.CHROMIUM_PATH } : {}) });
const base = process.env.UI_URL || "http://127.0.0.1:5173";
async function waitUntil(predicate) {
  const deadline = Date.now() + 5000;
  while (!predicate()) {
    assert.ok(Date.now() < deadline, "Expected intercepted request within 5 seconds");
    await new Promise(resolve => setTimeout(resolve, 10));
  }
}
const name = "測試 Book (2)";
const layout = doc => ({ document_id: doc, title: doc, chapters: [{ chapter_id: "c", title: "Chapter",
  sections: [{ section_id: "s", title: "Section", task_units: [{ unit_id: "u" }] },
    { section_id: "s2", title: "Second section", task_units: [{ unit_id: "u2" }] }] }] });
try {
  for (const width of [1280, 390]) {
    const page = await browser.newPage({ viewport: { width, height: 900 } });
    page.setDefaultTimeout(15_000);
    const errors = [], unexpected = [], prepares = [], contents = [], interactions = [], mutations = [];
    let uploads = 0, prepared = false, prepareRoute, firstContent = true, victim = "Victim", pendingRead, holdRead = false;
    page.on("pageerror", error => errors.push(error.message));
    await page.route("**/api/**", async route => {
      const request = route.request();
      const path = decodeURIComponent(new URL(request.url()).pathname);
      if (path === "/api/documents") return route.fulfill({ json: { items: [{ doc_name: "Old", title: "Old" }], total: 1 } });
      if (path === "/api/documents/management") return route.fulfill({ json: {
        items: [victim, ...(uploads >= 2 ? [name] : [])].filter(Boolean).map(doc => ({
          doc_name: doc, title: doc, source_type: "pdf", source_status: "available",
          preparation_status: doc === name && !prepared ? "not_prepared" : "ready",
        })), total: victim ? 2 : 1,
      } });
      if (path === "/api/documents/upload") {
        uploads++;
        assert.match(request.postData(), new RegExp(uploads === 1 ? "reject" : "create_distinct"));
        if (uploads === 1) return route.fulfill({ status: 409, json: {
          source_status: "failed", reason: "duplicate_conflict", source_filename: "Book.pdf",
        } });
        return route.fulfill({ json: { doc_name: name, source_filename: "Book (2).pdf", source_type: "pdf", source_status: "uploaded" } });
      }
      const body = request.method() === "POST" ? request.postDataJSON() : null;
      if (path === "/api/documents/prepare-task-layout") {
        prepares.push(body); prepareRoute = route; return;
      }
      if (path === "/api/documents/task-layout") {
        assert.ok(body.doc_name?.trim());
        assert.equal(body.refresh_task_units, false);
        if (holdRead) { pendingRead = route; return; }
        return route.fulfill({ json: layout(body.doc_name) });
      }
      if (path.endsWith("/task-units/content")) {
        contents.push({ path, body });
        assert.equal(path, `/api/documents/${name}/task-units/content`);
        assert.deepEqual(body, { task_unit_ids: ["u"], segmented: true });
        if (firstContent) { firstContent = false; return route.fulfill({ status: 404, json: { detail: "Fixture content unavailable" } }); }
        return route.fulfill({ json: { contents: [{ task_unit_id: "u", document_id: name, chapter_id: "c", section_id: "s",
          content_blocks: [{ block_id: "u:content:0", block_type: "content", content: "Recovered fixture content." }] }] } });
      }
      if (/reading-interactions\/(insight|quiz|critical-thinking)\/read$/.test(path)) {
        assert.equal(body.target.doc_name, name);
        assert.equal(body.target.chapter_id, "c");
        assert.equal(body.target.section_id, "s");
        interactions.push(path);
        const kind = path.includes("/quiz/") ? "quiz" : path.includes("/insight/") ? "analysis" : "critical_thinking_session";
        return route.fulfill({ json: { envelope: { target: { ...body.target, target_id: "s", document_id: name, title: "Section" },
          interaction_type: kind, status: "not_generated", schema_version: `${kind}_v1` }, payload: null } });
      }
      if (path.endsWith("/lifecycle/rename")) {
        mutations.push(body); victim = body.new_doc_name;
        return route.fulfill({ json: { status: "accepted", new_doc_name: victim, doc_name: "Victim", action: "rename" } });
      }
      if (path.endsWith("/lifecycle/delete")) {
        assert.equal(body.confirm, true);
        mutations.push(body); victim = null;
        return route.fulfill({ json: { status: "accepted", deleted: true, doc_name: body.doc_name, action: "delete" } });
      }
      unexpected.push(path);
      return route.fulfill({ status: 500, json: { detail: "Unexpected request" } });
    });
    await page.goto(base);
    await page.getByRole("button", { name: "Library", exact: true }).click();
    await page.locator('input[type="file"]').setInputFiles({ name: "Book.pdf", mimeType: "application/pdf", buffer: Buffer.from("fixture") });
    await page.locator(".file-management-upload-action").click();
    await page.locator(".file-management-confirm-duplicate-upload-action").waitFor();
    assert.equal(prepares.length, 0);
    await page.locator(".file-management-confirm-duplicate-upload-action").click();
    await page.getByText("Preparing document.", { exact: true }).waitFor();
    assert.equal(await page.locator(".file-management-open-uploaded-action").isDisabled(), true);
    for (const selector of [".file-management-rename-action", ".file-management-delete-action"]) {
      for (const button of await page.locator(selector).all()) assert.equal(await button.isDisabled(), true);
    }
    await waitUntil(() => prepareRoute);
    assert.equal(prepares.length, 1);
    assert.equal(prepares[0].doc_name, name);
    assert.equal(prepares[0].pdf_extraction_mode, "fast");
    prepared = true;
    await prepareRoute.fulfill({ json: { ...layout(name), pdf_extraction: {
      requested_mode: "fast", effective_mode: "fast", provenance: "applied" } } });
    await page.getByText("Document prepared. You can open it now.", { exact: true }).waitFor();
    await page.locator(".file-management-open-uploaded-action").click();
    await page.locator(".hierarchy-section-button").first().click();
    await page.getByRole("button", { name: "Retry content", exact: true }).click();
    await page.getByText("Recovered fixture content.", { exact: true }).first().waitFor();
    assert.equal(decodeURIComponent(new URL(page.url()).pathname), `/documents/${name}`);
    assert.equal(contents.length, 2);
    for (const [label, state] of [[/insights/i, ".inline-insight-status-not_generated"], [/^quiz$/i, ".quiz-drawer-status-not_generated"], [/critical/i, ".critical-thinking-status-not_generated"]]) {
      await page.locator(".hierarchy-section-interaction-trigger").first().click();
      await page.getByRole("menuitem", { name: label }).click();
      await page.locator(state).waitFor();
      if (!state.startsWith(".inline")) await page.locator(".reading-interaction-drawer-close").click();
    }
    assert.equal(interactions.length, 3);
    await page.getByRole("button", { name: "Library", exact: true }).click();
    for (const action of ["rename", "delete"]) {
      holdRead = true; pendingRead = undefined;
      const row = page.locator(".file-management-row").filter({
        has: page.locator(".file-management-document-title").filter({ hasText: new RegExp(`^${victim}$`) }),
      });
      await row.locator(".file-management-open-action").click();
      await page.getByRole("status").filter({ hasText: "Checking reader readiness." }).waitFor();
      await waitUntil(() => pendingRead);
      await row.locator(`.file-management-${action}-action`).click();
      const dialog = page.locator(action === "rename" ? ".file-management-rename-dialog" : ".file-management-delete-confirmation-dialog");
      await dialog.waitFor();
      assert.equal(mutations.length, action === "rename" ? 0 : 1, "Lifecycle writes require confirmation");
      if (action === "rename") await dialog.getByRole("textbox").fill("Renamed");
      await dialog.locator(`.file-management-confirm-${action}-action`).click();
      await dialog.waitFor({ state: "hidden" });
      holdRead = false;
      await pendingRead?.fulfill({ json: layout("Victim") }).catch(() => {});
      assert.equal(decodeURIComponent(new URL(page.url()).pathname), `/documents/${name}`);
    }
    assert.equal(mutations.length, 2);
    assert.equal(prepares.length, 1);
    await page.screenshot({ path: `/tmp/library-workflow-${width}.png`, fullPage: true });
    assert.deepEqual(errors, []);
    assert.deepEqual(unexpected, []);
    await page.close();
    console.log(`PASS ${width}px: duplicate upload -> prepare -> reader -> 404 recovery -> three interactions; pending rename/delete isolation`);
  }
} finally { await browser.close(); }
