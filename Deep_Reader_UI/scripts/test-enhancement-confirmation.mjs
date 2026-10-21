// All network fixtures are local mocks; no LLM calls or user-document writes.
import assert from "node:assert/strict";
const { chromium } = await import(process.env.PLAYWRIGHT_MODULE || "playwright");
const browser = await chromium.launch({ headless: true,
  ...(process.env.CHROMIUM_PATH ? { executablePath: process.env.CHROMIUM_PATH } : {}) });
try {
  for (const width of [1280, 390]) {
    const page = await browser.newPage({ viewport: { width, height: 900 } });
    let available = false, ready = true, count = null, expired = false, version = 1;
    let sourceReason = "source_not_ready";
    const fits = { status: "fits", full_source: true, model: "fixture-model",
      required_input_tokens: 12000, available_input_tokens: 24000,
      output_reserve_tokens: 4000, max_output_tokens: 8000, count_method: "conservative_estimate" };
    let capacity = { ...fits }, preflightFails = false;
    let mode = "stale", reloadFails = false, layoutMissing = false, prepareFails = false, preparations = 0, release;
    let holdPreparation = false, releasePreparation;
    let confirmed = false;
    const success = name => ({ success: true, doc_name: name, parser_mode: "llm_enhanced",
      change_kind: "structure_changed", committed: true, interactions_cleared: true, reading_layout_required: true });
    const calls = [], errors = [], unexpected = [];
    page.on("pageerror", error => errors.push(error.message));
    await page.route("**/api/**", async route => {
      const path = new URL(route.request().url()).pathname;
      const body = route.request().postDataJSON();
      if (["/api/documents", "/api/documents/management"].includes(path)) return route.fulfill({ json: { items: [], total: 0 } });
      if (path === "/api/documents/task-layout") {
        if (layoutMissing) return route.fulfill({ status: 409, json: { detail: "Task layout is not ready (missing_task_units)." } });
        if (reloadFails) return route.fulfill({ status: 503, json: { detail: "layout_unavailable" } });
        return route.fulfill({ json: { document_id: body.doc_name, title: body.doc_name,
          chapters: [{ chapter_id: `c${version}`, title: "Main Body", sections: [{ section_id: `s${version}`,
            title: "Body", task_units: [{ unit_id: `u${version}` }] }] }] } });
      }
      if (path === "/api/documents/prepare-task-layout") {
        assert.equal(body.layout_only, true);
        assert.equal(body.force_rebuild, false);
        assert.equal(body.include_anchor_page_evidence, false);
        preparations++;
        if (holdPreparation) await new Promise(resolve => { releasePreparation = resolve; });
        if (prepareFails) return route.fulfill({ status: 503, json: { detail: "preparation_unavailable" } });
        layoutMissing = false;
        return route.fulfill({ json: { document_id: "Book", chapters: [] } });
      }
      if (path.endsWith("/task-units/content")) return route.fulfill({ json: { contents: [{
        document_id: "Book", task_unit_id: `u${version}`, chapter_id: `c${version}`, section_id: `s${version}`,
        content_blocks: [{ block_id: `u${version}:content:0`, block_type: "content", content: `Source ${version}` }],
      }] } });
      if (path.endsWith("/preflight") && preflightFails) return route.abort();
      if (path.endsWith("/preflight")) return route.fulfill({ json: {
        capacity,
        ready, execution_available: available, reason: ready ? null : sourceReason,
        structure_version: version, interaction_count: count, includes_user_notes: null,
        cleanup_scope: "all_document_interactions_and_local_drafts",
        cleanup_condition: "structure_changed_only",
        confirmation_token: `consent-${version}`, expires_at: Math.floor(Date.now() / 1000) + (expired ? -1 : 300),
      } });
      if (path.endsWith("/result")) return route.fulfill({ json: { status: confirmed ? "completed" : "unknown",
        result: confirmed ? success(body.doc_name) : null, retryable: false, durability: "process_local" } });
      if (path === "/api/documents/reparse-structure") {
        calls.push(body);
        if (mode === "capacity") return route.fulfill({ status: 503, json: {
          detail: "capacity_unavailable", capacity: { status: "unavailable" },
        } });
        if (mode === "stale") return route.fulfill({ status: 409, json: { detail: "confirmation_stale" } });
        if (mode === "reject") return route.fulfill({ status: 502, json: { success: false, retryable: false, error: "candidate_rejected", doc_name: body.doc_name } });
        if (mode === "lost") return route.abort();
        if (mode === "pending") await new Promise(resolve => { release = resolve; });
        version++;
        confirmed = true;
        return route.fulfill({ json: success(body.doc_name) });
      }
      unexpected.push(path);
      return route.fulfill({ status: 500, json: { detail: "unexpected" } });
    });
    await page.goto(`${process.env.UI_URL || "http://127.0.0.1:5173"}/documents/Book`);
    await page.locator(".hierarchy-section-button").first().click();
    await page.getByText("Source 1", { exact: true }).first().waitFor();
    const dialog = page.getByRole("dialog");
    const submit = page.getByRole("button", { name: "Enhance", exact: true });
    async function assertNoCapacityDetails() {
      assert.equal(await dialog.locator('[aria-label="Model capacity"]').count(), 0);
      assert.doesNotMatch(await dialog.innerHTML(),
        /fixture-model|LongModelName|Required input tokens|Available input tokens|Reserved output tokens|Maximum supported output tokens|12,000|24,000|4,000|8,000/);
      assert.doesNotMatch(await dialog.ariaSnapshot(), /Model capacity|input tokens|output tokens|fixture-model/);
    }
    async function open() {
      await page.locator("#repair-menu-button").click();
      await page.getByRole("menuitem", { name: "LLM enhance", exact: true }).click();
      await dialog.waitFor();
      await page.getByLabel("Checking readiness").waitFor({ state: "hidden" });
      await page.waitForFunction(() => {
        const paper = document.querySelector('[role="dialog"]');
        return paper && getComputedStyle(paper).opacity === "1";
      });
    }
    async function cancel() {
      await page.getByRole("button", { name: "Cancel", exact: true }).click();
      await dialog.waitFor({ state: "hidden" });
    }
    await open();
    await page.getByText("Smart structuring is currently unavailable. Try checking again later.").waitFor();
    await assertNoCapacityDetails();
    assert.equal(await submit.isDisabled(), true);
    assert.match(await dialog.innerText(), /Unknown \(not zero\)/);
    await page.waitForFunction(() => document.activeElement?.textContent === "Cancel");
    await page.screenshot({ path: `/tmp/enhancement-confirmation-${width}.png`, fullPage: true });
    const bounds = await dialog.boundingBox();
    assert.ok(bounds.x >= 0 && bounds.x + bounds.width <= width);
    await page.keyboard.press("Escape");
    await dialog.waitFor({ state: "hidden" });
    assert.equal(calls.length, 0);
    available = true;
    for (const [value, message] of [
      [undefined, /service update is required/],
      [{ status: "fits", full_source: false }, /Complete document text could not be confirmed/],
      [{ status: "unavailable" }, /service is not configured/],
      [{ status: "resource_limited" }, /service's processing limits/],
      [{ ...fits, status: "exceeded", model: "LongModelName".repeat(30) }, /too long for smart structuring/],
    ]) {
      capacity = value;
      await open();
      await dialog.getByText(message).waitFor();
      assert.equal(await submit.isDisabled(), true);
      await assertNoCapacityDetails();
      assert.equal(await dialog.getByText("Ready for smart structuring.", { exact: true }).count(), 0);
      if (value?.status === "exceeded") {
        await page.screenshot({ path: `/tmp/enhancement-capacity-${width}.png`, fullPage: true, animations: "disabled" });
        assert.equal(await dialog.evaluate(el => el.scrollWidth <= el.clientWidth), true);
        await dialog.getByText(message).scrollIntoViewIfNeeded();
        const alertBounds = await dialog.getByText(message).boundingBox();
        const actionBounds = await submit.boundingBox();
        assert.ok(alertBounds.y + alertBounds.height <= actionBounds.y);
        await page.screenshot({ path: `/tmp/enhancement-capacity-details-${width}.png`, fullPage: true, animations: "disabled" });
      }
      await cancel();
      assert.equal(calls.length, 0);
    }
    preflightFails = true;
    await open();
    await dialog.getByText(/Could not check readiness/).waitFor();
    await assertNoCapacityDetails();
    assert.equal(await submit.isDisabled(), true);
    assert.equal(await dialog.getByText(/too long for smart structuring/).count(), 0);
    await cancel();
    preflightFails = false;
    capacity = { ...fits };
    await open();
    await dialog.getByRole("status").getByText("Ready for smart structuring.", { exact: true }).waitFor();
    await assertNoCapacityDetails();
    await page.screenshot({ path: `/tmp/enhancement-ready-${width}.png`, animations: "disabled" });
    mode = "capacity";
    await submit.click();
    await dialog.getByText(/service is not configured/).waitFor();
    await assertNoCapacityDetails();
    assert.equal(await submit.isDisabled(), true);
    assert.equal(await page.getByRole("button", { name: "Check stored state" }).count(), 0);
    assert.equal(calls.length, 1);
    // A fresh preflight observes the changed policy and cannot reuse prior consent.
    capacity = { ...fits, status: "exceeded" };
    await page.getByRole("button", { name: "Check again", exact: true }).click();
    await dialog.getByText(/too long for smart structuring/).waitFor();
    assert.equal(await submit.isDisabled(), true);
    assert.equal(calls.length, 1);
    await cancel();
    calls.length = 0;
    mode = "stale";
    capacity = { ...fits };
    available = true; ready = false;
    await open();
    assert.equal(await submit.isDisabled(), true);
    await page.getByRole("button", { name: "Open Library" }).waitFor();
    await cancel();
    for (const [reason, message] of [
      ["extraction_identity_unknown", /saved source needs verification/],
      ["source_changed", /source file has changed/],
      ["source_missing", /source file is missing/],
      ["source_ambiguous", /More than one source file/],
    ]) {
      sourceReason = reason;
      await open();
      await dialog.getByText(message).waitFor();
      assert.equal(await submit.isDisabled(), true);
      assert.equal(await dialog.getByText(/OCR/).count(), 0);
      await cancel();
    }
    sourceReason = "source_not_ready";
    ready = true; expired = true;
    await open();
    assert.equal(await submit.isDisabled(), true);
    assert.equal(await dialog.getByText("Ready for smart structuring.", { exact: true }).count(), 0);
    expired = false; count = 0;
    await page.getByRole("button", { name: "Check again", exact: true }).click();
    await submit.click();
    await page.getByText(/Enhancement was not applied/).waitFor();
    assert.equal(calls.length, 1);
    assert.equal(calls[0].discard_interactions, true);
    assert.equal(calls[0].confirmation_token, "consent-1");
    assert.ok(await page.getByText("Source 1", { exact: true }).count());
    mode = "reject";
    await page.getByRole("button", { name: "Check again", exact: true }).click();
    await submit.click();
    await page.getByText(/Enhancement was not applied/).waitFor();
    mode = "lost";
    await page.getByRole("button", { name: "Check again", exact: true }).click();
    await submit.click();
    await page.getByText(/Connection lost or timed out/).waitFor();
    await cancel();
    await open();
    assert.equal(await submit.isDisabled(), true);
    await page.getByRole("button", { name: "Check stored state" }).click();
    await page.getByText(/earlier outcome cannot be confirmed/).waitFor();
    assert.equal(calls.length, 3);
    version++;
    confirmed = true;
    prepareFails = true;
    await page.getByRole("button", { name: "Check stored state" }).click();
    await page.getByText(/Structure saved, but reading preparation failed/).waitFor();
    assert.equal(calls.length, 3);
    prepareFails = false;
    reloadFails = true;
    await page.getByRole("button", { name: "Reload document" }).click();
    await page.getByText(/Structure saved, but reading reload failed/).waitFor();
    assert.equal(preparations, 2);
    reloadFails = false;
    await page.getByRole("button", { name: "Reload document" }).evaluate(el => { el.click(); el.click(); });
    await dialog.waitFor({ state: "hidden" });
    assert.equal(preparations, 2, "Read retry must reuse completed preparation");
    assert.equal(calls.length, 3);
    mode = "pending";
    await open();
    await submit.evaluate(el => { el.click(); el.click(); });
    await page.getByLabel("Enhancing", { exact: true }).waitFor();
    assert.equal(calls.length, 4);
    assert.equal(await page.getByRole("button", { name: "Cancel", exact: true }).isDisabled(), true);
    // Navigate without reloading the app while a server response is pending.
    await page.evaluate(() => { history.pushState({}, "", "/documents/Other"); dispatchEvent(new PopStateEvent("popstate")); });
    await dialog.waitFor({ state: "hidden" });
    release();
    await page.waitForTimeout(150);
    assert.equal(await page.locator(".document-search-input input").inputValue(), "Other");
    await page.evaluate(() => { history.pushState({}, "", "/documents/Book"); dispatchEvent(new PopStateEvent("popstate")); });
    await open();
    await page.getByRole("button", { name: "Reload document" }).click();
    await dialog.waitFor({ state: "hidden" });
    assert.equal(calls.length, 4);
    await page.clock.install();
    await open();
    await submit.click();
    await page.getByLabel("Enhancing", { exact: true }).waitFor();
    await page.clock.fastForward(120_001);
    await page.getByText(/Connection lost or timed out/).waitFor();
    assert.equal(calls.length, 5);
    assert.equal(await submit.isDisabled(), true);
    release();
    await page.waitForTimeout(150);
    await page.getByRole("button", { name: "Check stored state" }).click();
    await dialog.waitFor({ state: "hidden" });
    assert.equal(calls.length, 5);
    mode = "success";
    holdPreparation = true;
    const previousPreparations = preparations;
    await open();
    await submit.click();
    await page.getByLabel("Preparing reading layout", { exact: true }).waitFor();
    assert.equal(await submit.isDisabled(), true);
    assert.equal(await page.getByRole("button", { name: "Cancel", exact: true }).isDisabled(), true);
    await page.screenshot({ path: `/tmp/enhancement-preparing-${width}.png`, animations: "disabled" });
    await page.evaluate(() => { history.pushState({}, "", "/documents/Other"); dispatchEvent(new PopStateEvent("popstate")); });
    await dialog.waitFor({ state: "hidden" });
    holdPreparation = false;
    releasePreparation();
    await page.waitForTimeout(200);
    assert.equal(await page.locator(".document-search-input input").inputValue(), "Other");
    await page.evaluate(() => { history.pushState({}, "", "/documents/Book"); dispatchEvent(new PopStateEvent("popstate")); });
    await open();
    await page.getByRole("button", { name: "Reload document" }).click();
    await dialog.waitFor({ state: "hidden" });
    assert.equal(preparations, previousPreparations + 1, "Returning to document must reuse completed preparation");
    assert.equal(calls.length, 6, "Reading recovery must not regenerate");
    assert.deepEqual(errors, []);
    assert.deepEqual(unexpected, []);
    console.log(`PASS ${width}px: readiness-only UI, no model/token DOM or accessibility output, capacity/legacy/network gates, changed policy, consent, focus/cancel, stale/rejected, uncertainty, reload-only recovery, duplicate submit, document switch`);
    await page.close();
  }
} finally { await browser.close(); }
