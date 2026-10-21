// Mocked transport only: no real preparation, generation, or document writes.
import assert from "node:assert/strict";
const { chromium } = await import(process.env.PLAYWRIGHT_MODULE || "playwright");
const browser = await chromium.launch({ headless: true,
  ...(process.env.CHROMIUM_PATH ? { executablePath: process.env.CHROMIUM_PATH } : {}) });
try {
  for (const width of [1280, 390]) {
    const page = await browser.newPage({ viewport: { width, height: 900 } });
    const errors = [], unexpected = [], repairs = [], generations = [];
    let revision = 0, failRepair = true, fallback = false, intent = "restore_source", questionReady = false, realToc = false;
    const layout = () => ({ document_id: "Book", title: "Book", parse_provenance: {
      requested_parser_mode: revision ? "llm_enhanced" : "common",
      effective_parser_mode: revision && !fallback ? "llm_enhanced" : "common",
      fallback_used: fallback, fallback_reason: fallback ? "fixture_fallback" : null,
      implicit_single_body: !revision && !realToc, declaration_unconfirmed: !realToc, enhancement_intent: intent,
      toc_observation: realToc ? "validated_printed_toc" : "not_detected",
    }, chapters: [{ chapter_id: `c${revision}`, title: realToc ? "Preface" : revision ? "A generated heading" : "Main Body",
      metadata: { title_origin: revision ? "generated" : "implicit" }, sections: [{
        section_id: `s${revision}`, title: null, is_implicit_section: !revision,
        title_origin: realToc ? "source" : revision ? "generated" : "implicit",
        task_units: [{ unit_id: `u${revision}` }],
      }] }] });
    page.on("pageerror", (error) => errors.push(error.message));
    await page.route("**/api/**", async route => {
      const path = new URL(route.request().url()).pathname;
      const body = route.request().postDataJSON();
      if (["/api/documents", "/api/documents/management"].includes(path)) return route.fulfill({ json: { items: [], total: 0 } });
      if (["/api/documents/task-layout", "/api/documents/prepare-task-layout"].includes(path)) {
        if (path.includes("prepare-")) assert.equal(body.layout_only, true);
        return route.fulfill({ json: layout() });
      }
      if (path.endsWith("/task-units/content")) return route.fulfill({ json: { contents: [{
        task_unit_id: `u${revision}`, document_id: "Book", chapter_id: `c${revision}`, section_id: `s${revision}`,
        content_blocks: [{ block_id: `u${revision}:content:0`, block_type: "content", content: `Readable source ${revision}` }],
      }] } });
      if (path === "/api/documents/reparse-structure") {
        repairs.push(body);
        if (failRepair) return route.fulfill({ status: 502, json: { success: false, retryable: false, doc_name: "Book", error: "enhancement_candidate_rejected" } });
        revision++;
        intent = body.enhancement_intent;
        return route.fulfill({ json: { success: true, doc_name: "Book", parser_mode: "llm_enhanced",
          change_kind: "structure_changed", committed: true, interactions_cleared: true, reading_layout_required: true } });
      }
      if (path === "/api/documents/reparse-structure/preflight") return route.fulfill({ json: {
        ready: true, execution_available: true, reason: null, structure_version: revision + 1,
        interaction_count: null, includes_user_notes: true,
        cleanup_scope: "all_document_interactions_and_local_drafts",
        cleanup_condition: "structure_changed_only", capacity: { status: "fits", full_source: true },
        confirmation_token: "fixture", expires_at: Math.floor(Date.now() / 1000) + 300,
      } });
      if (path.includes("/reading-interactions/")) {
        const thinking = path.includes("critical-thinking");
        const reading = path.endsWith("/read");
        if (!reading) generations.push(body);
        const success = questionReady && path.endsWith("/generate-question");
        return route.fulfill({ status: success ? 200 : reading ? 409 : 503, json: { envelope: {
          target: { ...body.target, target_id: body.target.chapter_id, document_id: "Book", title: "Main Body" },
          interaction_type: thinking ? "critical_thinking_session" : "analysis",
          status: success ? "question_generated" : reading ? "stale_target" : "generation_failed",
          reason: success ? null : reading ? "interaction_policy_stale" : "provider_unavailable",
          schema_version: thinking ? "critical_thinking_session_v1" : "analysis_interaction_v1",
          session_id: thinking ? "saved-session" : null,
        }, payload: success ? { question: "What evidence supports this interpretation?" } : null } });
      }
      unexpected.push(path);
      return route.fulfill({ status: 500, json: { detail: "Unexpected request" } });
    });
    await page.goto(`${process.env.UI_URL || "http://127.0.0.1:5173"}/documents/Book`);
    await page.getByText("Main Body · No validated heading boundaries").waitFor();
    await page.locator(".hierarchy-section-button").first().click();
    await page.getByText("Readable source 0", { exact: true }).first().waitFor();
    for (let i = 0; i < 2; i++) {
      await page.locator(".hierarchy-chapter-interaction-trigger").first().click();
      await page.getByRole("menuitem", { name: "Critical thinking", exact: true }).click();
      await page.getByText("Outdated session", { exact: true }).waitFor();
      assert.equal(generations.length, 0);
      if (i === 1) {
        await page.getByRole("button", { name: "Generate question", exact: true }).click();
        await page.locator(".reading-interaction-drawer").getByText("provider_unavailable", { exact: true }).first().waitFor();
        assert.equal(generations.length, 1);
        assert.equal(await page.getByText("Critical-thinking question generated.", { exact: true }).count(), 0);
        questionReady = true;
        await page.getByRole("button", { name: "Generate question", exact: true }).click();
        await page.getByRole("textbox", { name: "Draft answer" }).fill("A locally saved argument with evidence.");
        await page.getByRole("button", { name: "Submit answer", exact: true }).click();
        await page.locator(".critical-thinking-error").getByText("provider_unavailable", { exact: true }).waitFor();
        assert.equal(await page.getByRole("textbox", { name: "Draft answer" }).inputValue(), "A locally saved argument with evidence.");
      }
      await page.locator(".reading-interaction-drawer").getByRole("button", { name: "Close", exact: true }).click();
    }
    async function repair(label, hint) {
      await page.locator("#repair-menu-button").click();
      assert.equal(await page.getByRole("menuitem", { name: /LLM:/ }).count(), 0);
      assert.equal(await page.getByRole("menuitem", { name: "LLM enhance", exact: true }).count(), 1);
      await page.getByRole("combobox", { name: "Printed table of contents" }).click();
      await page.getByRole("option", { name: hint, exact: true }).click();
      await page.getByRole("menuitem", { name: label, exact: true }).click();
      await page.getByRole("button", { name: "Enhance", exact: true }).click();
    }
    await repair("LLM enhance", "Present");
    await page.getByText(/Enhancement was not applied/).waitFor();
    assert.ok(await page.getByText("Readable source 0", { exact: true }).count() > 0);
    await page.getByRole("button", { name: "Cancel", exact: true }).click();
    assert.equal(repairs[0].toc_hint, "present");
    assert.equal(repairs[0].enhancement_intent, "restore_source");
    failRepair = false;
    await repair("LLM enhance", "Absent");
    await page.getByText("Generated reading outline, not an original table of contents.").waitFor();
    await page.getByText("Generated heading", { exact: true }).waitFor();
    assert.equal(repairs[1].toc_hint, "absent");
    assert.equal(repairs[1].enhancement_intent, "restore_source");
    assert.equal(await page.getByText("Readable source 0", { exact: true }).count(), 0);
    await page.locator(".hierarchy-section-button").first().click();
    await page.getByText("Readable source 1", { exact: true }).first().waitFor();
    fallback = true;
    await repair("LLM enhance", "Auto");
    await page.getByText(/Requested llm_enhanced; using common/).waitFor();
    await page.locator("#repair-menu").waitFor({ state: "hidden" });
    await page.screenshot({ path: `/tmp/reader-recovery-${width}.png`, fullPage: true });
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > window.innerWidth), false);
    realToc = true;
    revision = 0;
    fallback = false;
    intent = "restore_source";
    await page.reload();
    await page.locator(".hierarchy-section-button").filter({ hasText: "Preface" }).click();
    await page.getByText("Readable source 0", { exact: true }).first().waitFor();
    assert.equal(await page.getByText("Main Body · No validated heading boundaries").count(), 0);
    // Service contracts retain non-2xx states and never reinterpret plain errors as envelopes.
    await page.evaluate(async () => {
      const { ReadingInteractionService } = await import("/src/services/ReadingInteractionService.ts");
      const target = { doc_name: "Book", target_type: "document" };
      for (const [http, status] of [[409, "stale_target"], [422, "insufficient_content"], [502, "generation_failed"], [503, "evaluation_failed"]]) {
        const payload = { envelope: { status, reason: "fixture" }, payload: null };
        const service = new ReadingInteractionService({ requestJsonWithStatus: async () => ({ ok: false, status: http, payload }) });
        if (await service.readInsight(target) !== payload) throw new Error("Envelope lost");
      }
      const service = new ReadingInteractionService({ requestJsonWithStatus: async () => ({ ok: false, status: 500, payload: { detail: "plain failure" } }) });
      try { await service.readInsight(target); } catch (error) {
        if (error.message === "plain failure") return;
        throw error;
      }
      throw new Error("Plain error incorrectly accepted");
    });
    assert.deepEqual(errors, []);
    assert.deepEqual(unexpected, []);
    console.log(`PASS ${width}px: policy stale/read-only reopening, explicit failure, preserved content, TOC/intent snapshots, accepted ID replacement, fallback`);
    await page.close();
  }
} finally {
  await browser.close();
}
