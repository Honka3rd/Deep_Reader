// Local browser fixtures only; no paid generation or user-document writes.
import assert from "node:assert/strict";
const { chromium } = await import(process.env.PLAYWRIGHT_MODULE || "playwright");
const browser = await chromium.launch({ headless: true,
  ...(process.env.CHROMIUM_PATH ? { executablePath: process.env.CHROMIUM_PATH } : {}) });
try {
  for (const width of [1280, 390]) {
    const page = await browser.newPage({ viewport: { width, height: 900 } });
    const errors = [], calls = [], lookups = [];
    let kind = "unchanged", transport = "ok", lookup = "completed", expired = false;
    let labels = "Original", prepares = 0, reads = 0, contentReads = 0, failLabels = false;
    let conditional = true;
    const outcome = () => ({ success: true, doc_name: "Book", parser_mode: "llm_enhanced",
      change_kind: kind, committed: kind !== "unchanged", interactions_cleared: kind === "structure_changed",
      reading_layout_required: kind === "structure_changed", structure_version: kind === "structure_changed" ? 2 : 1 });
    page.on("pageerror", error => errors.push(error.message));
    await page.route("**/api/**", async route => {
      const path = new URL(route.request().url()).pathname;
      const body = route.request().postDataJSON();
      if (["/api/documents", "/api/documents/management"].includes(path)) return route.fulfill({ json: { items: [], total: 0 } });
      if (path.endsWith("/task-layout")) {
        reads++;
        if (failLabels) return route.fulfill({ status: 503, json: { detail: "test" } });
        return route.fulfill({ json: { document_id: "Book", title: "Book", chapters: [{ chapter_id: "c1", title: labels,
          sections: [{ section_id: "s1", title: `${labels} section`, task_units: [{ unit_id: "u1" }] }] }] } });
      }
      if (path.endsWith("/prepare-task-layout")) {
        prepares++;
        return route.fulfill({ json: { document_id: "Book", chapters: [] } });
      }
      if (path.endsWith("/task-units/content")) {
        contentReads++;
        return route.fulfill({ json: { contents: [{ document_id: "Book", chapter_id: "c1", section_id: "s1", task_unit_id: "u1",
          content_blocks: [{ block_id: "u1:content:0", block_type: "content", content: "Source evidence retained exactly." }] }] } });
      }
      if (path.endsWith("/preflight")) return route.fulfill({ json: {
        ready: true, execution_available: true, reason: null, structure_version: 1,
        interaction_count: 3, includes_user_notes: true, cleanup_scope: "all_document_interactions_and_local_drafts",
        cleanup_condition: conditional ? "structure_changed_only" : undefined,
        capacity: { status: "fits", full_source: true }, confirmation_token: "same-operation",
        expires_at: Date.now() / 1000 + (expired ? -1 : 300),
      } });
      if (path.endsWith("/reparse-structure/result")) {
        lookups.push(body);
        return route.fulfill({ json: { status: lookup, result: lookup === "completed" ? outcome() : null,
          retryable: lookup === "not_started", durability: "process_local" } });
      }
      if (path.endsWith("/reparse-structure")) {
        calls.push(body);
        if (transport === "transient") return route.fulfill({ status: 502, json: {
          success: false, doc_name: "Book", retryable: true, error: "enhancement_provider_transient" } });
        if (transport === "running") return route.fulfill({ status: 409, json: { detail: "enhancement_in_progress" } });
        if (kind === "titles_only") labels = "Revised";
        if (transport === "lost") return route.abort();
        if (transport === "legacy") return route.fulfill({ json: { success: true, doc_name: "Book" } });
        if (transport === "contradictory") return route.fulfill({ json: { ...outcome(), interactions_cleared: true } });
        return route.fulfill({ json: outcome() });
      }
      if (path.includes("/reading-interactions/")) {
        const thinking = path.includes("critical-thinking");
        return route.fulfill({ json: { envelope: {
          target: { ...body.target, target_id: "c1", document_id: "Book", title: "Original" },
          interaction_type: thinking ? "critical_thinking_session" : "analysis",
          status: thinking ? "question_generated" : "completed",
          schema_version: thinking ? "critical_thinking_session_v1" : "analysis_interaction_v1",
          session_id: thinking ? "saved-session" : null,
        }, payload: thinking ? { question: "What evidence supports this interpretation?" }
          : { summary: "Saved insight evidence.", interpretation: "A saved interpretation.", explanation: "Evidence explained." } } });
      }
      throw new Error(`Unexpected request ${path}`);
    });
    await page.goto(`${process.env.UI_URL || "http://127.0.0.1:5173"}/documents/Book`);
    await page.locator(".hierarchy-section-button").first().click();
    await page.getByText("Source evidence retained exactly.", { exact: true }).first().waitFor();
    const dialog = page.getByRole("dialog");
    const enhance = () => page.getByRole("button", { name: "Enhance", exact: true });
    async function open() {
      await page.locator("#repair-menu-button").click();
      await page.getByRole("menuitem", { name: "LLM enhance", exact: true }).click();
      await page.getByLabel("Checking readiness").waitFor({ state: "hidden" });
    }
    async function thinking() {
      await page.locator(".hierarchy-chapter-interaction-trigger").first().click();
      await page.getByRole("menuitem", { name: "Critical thinking", exact: true }).click();
      await page.getByRole("textbox", { name: "Draft answer" }).waitFor();
    }
    const closeDrawer = () => page.locator(".reading-interaction-drawer").getByRole("button", { name: "Close", exact: true }).click();
    await page.evaluate(async () => {
      const { confirmedChange } = await import("/src/features/structure-enhancement/outcome.ts");
      for (const kind of ["unchanged", "titles_only", "structure_changed"]) {
        const result = { success: true, doc_name: "Book", parser_mode: "llm_enhanced", change_kind: kind,
          committed: kind !== "unchanged", interactions_cleared: kind === "structure_changed",
          reading_layout_required: kind === "structure_changed" };
        if (confirmedChange(result, "Book") !== kind) throw new Error("Valid result rejected");
        for (const field of ["change_kind", "committed", "interactions_cleared", "reading_layout_required"]) {
          if (confirmedChange({ ...result, [field]: undefined }, "Book") !== null) throw new Error(`Missing ${field} accepted`);
        }
        if (confirmedChange(result, "Other") !== null) throw new Error("Wrong document accepted");
      }
    });
    await page.locator(".hierarchy-chapter-interaction-trigger").first().click();
    await page.getByRole("menuitem", { name: "Insights", exact: true }).click();
    await page.locator(".inline-insight-content").filter({ hasText: "Saved insight evidence." }).waitFor();
    await thinking();
    await page.getByRole("textbox", { name: "Draft answer" }).fill("My unsaved argument, with evidence.");
    await closeDrawer();
    async function retained() {
      assert.ok(await page.getByText("Source evidence retained exactly.", { exact: true }).count() > 0);
      assert.equal(await page.locator(".inline-insight-content").filter({ hasText: "Saved insight evidence." }).count(), 1);
      await thinking();
      assert.equal(await page.getByRole("textbox", { name: "Draft answer" }).inputValue(), "My unsaved argument, with evidence.");
      await closeDrawer();
      assert.equal(prepares, 0);
      assert.equal(contentReads, 1);
    }
    conditional = false;
    await open();
    await dialog.getByText(/service needs an update/).waitFor();
    assert.equal(await enhance().isDisabled(), true);
    await page.getByRole("button", { name: "Cancel", exact: true }).click();
    conditional = true;
    await open();
    await enhance().click();
    await dialog.getByText(/No structure changes/).waitFor();
    await page.screenshot({ path: `/tmp/enhancement-unchanged-${width}.png`, animations: "disabled" });
    await dialog.getByRole("button", { name: "Close", exact: true }).click();
    assert.equal(reads, 1);
    await retained();
    kind = "titles_only";
    failLabels = true;
    await open();
    await enhance().click();
    await dialog.getByText(/Titles saved, but label refresh failed/).waitFor();
    failLabels = false;
    await dialog.getByRole("button", { name: "Reload document" }).click();
    await dialog.waitFor({ state: "hidden" });
    await page.locator(".hierarchy-section-button").filter({ hasText: "Revised section" }).waitFor();
    await retained();
    // Temporary provider failure retries the same consent, without an extra preflight.
    transport = "transient";
    kind = "unchanged";
    await open();
    await enhance().click();
    await page.getByRole("button", { name: "Retry smart structuring" }).waitFor();
    transport = "ok";
    await page.getByRole("button", { name: "Retry smart structuring" }).click();
    await dialog.getByText(/No structure changes/).waitFor();
    assert.deepEqual(calls.at(-1), calls.at(-2));
    await dialog.getByRole("button", { name: "Close", exact: true }).click();
    await retained();
    for (const failure of ["lost", "legacy", "contradictory", "running"]) {
      transport = failure;
      lookup = "running";
      await open();
      await enhance().click();
      const reconcile = page.getByRole("button", { name: "Check stored state" });
      await reconcile.waitFor();
      const count = calls.length;
      await reconcile.click();
      await dialog.getByText(/still running/).waitFor();
      lookup = "unknown";
      await reconcile.click();
      await dialog.getByText(/earlier outcome cannot be confirmed/).waitFor();
      assert.equal(await enhance().isDisabled(), true);
      lookup = "completed";
      await reconcile.click();
      await dialog.getByText(/No structure changes/).waitFor();
      assert.equal(calls.length, count);
      assert.equal(lookups.at(-1).confirmation_token, calls.at(-1).confirmation_token);
      await dialog.getByRole("button", { name: "Close", exact: true }).click();
      await retained();
    }
    expired = true;
    await open();
    assert.equal(await enhance().isDisabled(), true);
    await page.getByRole("button", { name: "Cancel", exact: true }).click();
    expired = false;
    transport = "lost";
    lookup = "not_started";
    await open();
    await enhance().click();
    await page.getByRole("button", { name: "Check stored state" }).click();
    await page.getByRole("button", { name: "Retry smart structuring" }).waitFor();
    transport = "ok";
    await page.getByRole("button", { name: "Retry smart structuring" }).click();
    await dialog.getByText(/No structure changes/).waitFor();
    assert.deepEqual(calls.at(-1), calls.at(-2));
    await dialog.getByRole("button", { name: "Close", exact: true }).click();
    lookup = "completed";
    kind = "structure_changed";
    transport = "lost";
    await open();
    await enhance().click();
    await page.getByRole("button", { name: "Check stored state" }).click();
    await dialog.waitFor({ state: "hidden" });
    assert.equal(prepares, 1);
    assert.equal(await page.locator(".inline-insight-content").filter({ hasText: "Saved insight evidence." }).count(), 0);
    assert.equal(await page.getByText("Source evidence retained exactly.", { exact: true }).count(), 0);
    await thinking();
    assert.equal(await page.getByRole("textbox", { name: "Draft answer" }).inputValue(), "");
    await closeDrawer();
    await page.screenshot({ path: `/tmp/enhancement-outcomes-${width}.png`, animations: "disabled" });
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth), false);
    assert.deepEqual(errors, []);
    console.log(`PASS ${width}px: typed outcomes, saved insight/draft retention, label-only retry, conditional consent, transient retry, authoritative lost/legacy/inconsistent/concurrent reconciliation, expiry, structural reset`);
    await page.close();
  }
} finally { await browser.close(); }
