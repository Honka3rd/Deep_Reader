// All API requests are intercepted. No real prepare, generation or data writes.
import assert from "node:assert/strict";
const { chromium } = await import(process.env.PLAYWRIGHT_MODULE || "playwright");
const browser = await chromium.launch({ headless: true,
  ...(process.env.CHROMIUM_PATH ? { executablePath: process.env.CHROMIUM_PATH } : {}),
});
const base = process.env.UI_URL || "http://127.0.0.1:5173";
const layout = (name) => ({ document_id: name, title: name, chapters: [{
  chapter_id: `${name}-c`, title: "Chapter", sections: ["one", "two"].map((part) => ({
    section_id: `${name}-${part}`, title: part, task_units: [{ unit_id: `${name}-${part}-u` }],
  })),
}] });
try {
  const page = await browser.newPage();
  const errors = [];
  const unexpected = [];
  const interactions = [];
  let fault = "404";
  let pendingContent;
  let deferredSeen;
  page.on("pageerror", (error) => errors.push(error.message));
  await page.route("**/api/**", async (route) => {
    const request = route.request();
    const path = decodeURIComponent(new URL(request.url()).pathname);
    const body = request.postDataJSON();
    if (path === "/api/documents/task-layout") return route.fulfill({ json: layout(body.doc_name) });
    if (path === "/api/documents" || path === "/api/documents/management") return route.fulfill({ json: {
      total: 2, items: ["Book", "Other"].map((name) => ({ doc_name: name, title: name, source: "structured", preparation_status: "ready" })),
    } });
    if (path.endsWith("/task-units/content")) {
      assert(!path.includes("/documents//"));
      assert(body.task_unit_ids.every((id) => typeof id === "string" && id.trim()));
      const currentFault = fault;
      fault = "none";
      if (currentFault === "404") return route.fulfill({ status: 404, json: { detail: "Missing content fixture" } });
      if (currentFault === "network") return route.abort("failed");
      if (currentFault === "deferred") {
        await new Promise((resolve) => { pendingContent = resolve; deferredSeen(); });
      }
      const name = path.split("/")[3];
      const unit = body.task_unit_ids[0];
      return route.fulfill({ json: { contents: [{
        task_unit_id: currentFault === "wrong-unit" ? "unrequested" : unit,
        document_id: currentFault === "wrong-document" ? "unrelated" : name,
        chapter_id: `${name}-c`, section_id: unit.slice(0, -2),
        content_blocks: [{ block_id: `${unit}:content:0`, block_type: "content", content: `Visible ${unit}` }],
      }] } });
    }
    if (path === "/api/documents/reading-interactions/insight/read") {
      interactions.push(body.target);
      return route.fulfill({ json: { envelope: {
        target: { ...body.target, target_id: body.target.section_id, document_id: body.target.doc_name, title: "one" },
        interaction_type: "analysis", status: "not_generated", schema_version: "analysis_interaction_v1",
      }, payload: null } });
    }
    unexpected.push(path);
    return route.fulfill({ status: 500, json: { detail: "Unexpected API call" } });
  });
  await page.goto(`${base}/documents/Book`);
  const section = (name) => page.locator(".hierarchy-section-button").filter({ hasText: name });
  await section("one").click();
  await page.getByText("Missing content fixture", { exact: true }).first().waitFor();
  await page.getByRole("button", { name: "Retry content", exact: true }).click();
  await page.locator(".reader-content-block-body").filter({ hasText: "Visible Book-one-u" }).first().waitFor();

  for (const failure of ["network", "wrong-unit", "wrong-document"]) {
    fault = failure;
    await section("two").click();
    await page.getByRole("button", { name: "Retry content", exact: true }).waitFor();
    await page.getByRole("button", { name: "Retry content", exact: true }).click();
    await page.locator(".reader-content-block-body").filter({ hasText: "Visible Book-two-u" }).first().waitFor();
  }

  // A late response from a previous section must not overwrite the selected section.
  fault = "deferred";
  const firstDeferred = new Promise((resolve) => { deferredSeen = resolve; });
  await section("one").click();
  await firstDeferred;
  await section("two").click();
  await page.locator(".reader-content-block-body").filter({ hasText: "Visible Book-two-u" }).first().waitFor();
  const staleSectionResponse = page.waitForResponse((r) => r.url().endsWith("/task-units/content") && r.request().postDataJSON().task_unit_ids[0] === "Book-one-u");
  pendingContent();
  await staleSectionResponse;
  await page.waitForTimeout(100);
  assert.equal(await page.locator(".reader-content-block-body").filter({ hasText: "Visible Book-one-u" }).count(), 0);

  // Section interactions still work after content failures, independently of content state.
  await page.locator(".hierarchy-section-interaction-trigger").first().click();
  await page.getByRole("menuitem", { name: "Insights", exact: true }).click();
  await page.locator(".inline-insight-status-not_generated").waitFor();
  assert.equal(interactions.at(-1).doc_name, "Book");
  assert.equal(interactions.at(-1).section_id, "Book-one");

  // Late content from a different document must also be ignored.
  fault = "deferred";
  const secondDeferred = new Promise((resolve) => { deferredSeen = resolve; });
  await section("one").click();
  await secondDeferred;
  await page.getByRole("button", { name: "Library", exact: true }).click();
  await page.locator(".file-management-row").filter({ hasText: "Other" })
    .getByRole("button", { name: "Open", exact: true }).click();
  await section("two").click();
  await page.locator(".reader-content-block-body").filter({ hasText: "Visible Other-two-u" }).first().waitFor();
  const staleDocumentResponse = page.waitForResponse((r) => r.url().endsWith("/Book/task-units/content"));
  pendingContent();
  await staleDocumentResponse;
  await page.waitForTimeout(100);
  assert.equal(await page.locator(".reader-content-block-body").filter({ hasText: "Visible Book-one-u" }).count(), 0);

  const checks = await page.evaluate(async () => {
    const { TaskUnitContentService } = await import("/src/services/TaskUnitContentService.ts");
    const { ReadingInteractionService } = await import("/src/services/ReadingInteractionService.ts");
    const { assertSectionInLayout, assertTargetInLayout } = await import("/src/shared/readerIdentity.ts");
    let calls = 0;
    let rejected = 0;
    const fake = { requestJson: async () => { calls++; throw new Error("Unexpected transport"); } };
    const contents = new TaskUnitContentService(fake);
    const interaction = new ReadingInteractionService(fake);
    async function rejects(fn) {
      let failed = false;
      try { await fn(); } catch { failed = true; rejected++; }
      if (!failed) throw new Error("Expected identity rejection");
    }
    for (const id of ["", " ", null, undefined, 42]) {
      await rejects(() => contents.fetchTaskUnitContent("Book", id));
      await rejects(() => contents.fetchTaskUnitContents(id, ["u"]));
      await rejects(() => contents.fetchTaskUnitContents("Book", [id]));
    }
    await rejects(() => contents.fetchTaskUnitContents("Book", ["u", "u"]));
    const empty = await contents.fetchTaskUnitContents("Book", []);
    if (empty.length) throw new Error("Expected empty batch to skip transport");
    for (const method of ["readInsight", "generateInsight", "refreshInsight", "readQuiz", "generateQuiz", "refreshQuiz", "readCriticalThinkingSession", "generateCriticalThinkingQuestion", "submitCriticalThinkingAnswer", "retryCriticalThinkingEvaluation"]) {
      await rejects(() => interaction[method]({ doc_name: "", target_type: "document" }, {}));
      await rejects(() => interaction[method]({ doc_name: "Book", target_type: "task_unit", task_unit_id: " " }, {}));
    }
    const unit = { unit_id: "u" };
    const sec = { section_id: "s", task_units: [unit] };
    const chapter = { chapter_id: "c", sections: [sec] };
    const doc = { chapters: [chapter] };
    const valid = { chapter, section: sec, taskUnits: [unit] };
    assertSectionInLayout("Book", doc, valid);
    await rejects(() => assertSectionInLayout("", doc, valid));
    await rejects(() => assertSectionInLayout("Book", structuredClone(doc), valid));
    await rejects(() => assertSectionInLayout("Book", doc, { ...valid, taskUnits: [{ unit_id: "other" }] }));
    const target = { doc_name: "Book", target_type: "task_unit", chapter_id: "c", section_id: "s", task_unit_id: "u" };
    assertTargetInLayout(target, "Book", doc);
    await rejects(() => assertTargetInLayout(target, "Other", doc));
    await rejects(() => assertTargetInLayout({ ...target, section_id: "absent" }, "Book", doc));
    await rejects(() => assertTargetInLayout({ ...target, task_unit_id: "absent" }, "Book", doc));
    if (calls) throw new Error(`${calls} invalid calls reached transport`);
    return rejected;
  });
  assert.deepEqual(errors, []);
  assert.deepEqual(unexpected, []);
  console.log(`PASS ${checks} identity guard cases; 404/network/mismatched responses + retry, stale section/document responses, interaction recovery`);
} finally {
  await browser.close();
}
