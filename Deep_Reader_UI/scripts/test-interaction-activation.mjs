// Actual React app with isolated API fixtures: no production writes or model calls.
import assert from "node:assert/strict";
const { chromium } = await import(process.env.PLAYWRIGHT_MODULE || "playwright");
const browser = await chromium.launch({ headless: true });
const labels = { insight: "Insights", summary: "Summary", quiz: "Quiz", "critical-thinking": "Critical thinking" };
const interactionKinds = process.env.INTERACTION_KIND ? [process.env.INTERACTION_KIND] : Object.keys(labels);
const types = { insight: "analysis", summary: "summary", quiz: "quiz", "critical-thinking": "critical_thinking_session" };
const payloads = {
  insight: { summary: "Saved insight", reasoning: "Evidence reasoning", interpretation: "Interpretation" },
  summary: { headline: "Saved summary", body: "Long readable summary. ".repeat(200), key_points: ["First point", "Second point"] },
  quiz: { items: [{ item_id: "q1", item_type: "short_answer", prompt: "Saved quiz question", answer: "Answer" }] },
  "critical-thinking": { question: "Saved critical question", submitted_answer: "My saved answer", evaluation: { feedback: "Saved feedback", score: 8 }, retry_eligible: false },
};
const layout = { document_id: "Book", title: "Book", chapters: [{ chapter_id: "c1", title: "Chapter", sections: [{ section_id: "s1", title: "Section", task_units: [{ unit_id: "u1" }] }] }] };
const pause = ms => new Promise(resolve => setTimeout(resolve, ms));
const until = async fn => { for (let i = 0; i < 100; i++) { if (await fn()) return; await pause(30); } assert.fail("Timed out waiting for fixture condition"); };
try {
  for (const width of [1280, 390]) {
    const page = await browser.newPage({ viewport: { width, height: 900 } });
    const errors = [], unexpected = [], calls = [];
    let mode = "missing", holdPhase = null, release = null;
    page.on("pageerror", error => errors.push(error.message));
    await page.route("**/api/**", async route => {
      const path = new URL(route.request().url()).pathname;
      if (path === "/api/documents" || path === "/api/documents/management") return route.fulfill({ json: { total: 2,
        items: ["Book", "Other"].map(doc_name => ({ doc_name, title: doc_name, source: "structured", preparation_status: "ready" })) } });
      if (path.endsWith("/task-layout")) return route.fulfill({ json: { ...layout, document_id: route.request().postDataJSON().doc_name } });
      if (path.endsWith("/reparse-structure/status")) return route.fulfill({ json: { status: "idle" } });
      if (path.endsWith("/reading-content")) return route.fulfill({ json: {
        revision: 0, source_token: "fixture", disposition: "accepted", can_undo: false, max_chars: 10000, regions: [], blocks: [],
      } });
      const match = path.match(/reading-interactions\/([^/]+)\/([^/]+)$/);
      if (!match) { unexpected.push(path); return route.fulfill({ status: 500, json: { detail: "Unexpected API" } }); }
      const [, kind, operation] = match, target = route.request().postDataJSON().target;
      const currentMode = mode;
      calls.push({ kind, operation, target });
      if (holdPhase === operation) {
        holdPhase = null;
        await new Promise(resolve => { release = resolve; });
      }
      if (kind === "note") return route.fulfill({ json: { target: { ...target, target_id: "Book" }, target_token: "t1", max_chars: 10000, note: null } });
      const isRead = operation === "read";
      if (isRead && currentMode === "http-error") return route.fulfill({ status: 503, json: { detail: "Unavailable" } });
      if (isRead && currentMode === "malformed") return route.fulfill({ json: { unexpected: true } });
      const saved = currentMode === "saved" || !isRead;
      const status = isRead && ["stale_target", "insufficient_content", "validation_failed", "generation_failed"].includes(currentMode) ? currentMode
        : currentMode === "generation-error" && !isRead ? "generation_failed"
        : saved ? (kind === "critical-thinking" && !isRead ? "question_generated" : "completed") : "not_generated";
      const response = { envelope: { target: { ...target, target_id: target.chapter_id || "Book" }, interaction_type: types[kind], status,
        ...(kind === "critical-thinking" && ["completed", "question_generated"].includes(status) ? { session_id: "session-1" } : {}) },
        payload: status === "question_generated" ? { question: "Saved critical question", retry_eligible: false } : status === "completed" ? payloads[kind] : null };
      if (isRead && currentMode === "mismatch") response.envelope.target.doc_name = "Other";
      if (isRead && currentMode === "wrong-kind") response.envelope.interaction_type = "unrelated";
      if (isRead && currentMode === "contradictory") response.payload = payloads[kind];
      if (isRead && currentMode === "missing-payload") delete response.payload;
      return route.fulfill({ status: isRead && currentMode === "http-missing" ? 503 : 200, json: response });
    });
    const open = async (kind, chapter = false) => {
      await page.getByRole("button", { name: chapter ? "Chapter reading interactions for Chapter" : "Document reading interactions", exact: true }).click();
      await page.getByRole("menuitem", { name: labels[kind] || "Note", exact: true }).click();
    };
    const surface = kind => kind === "insight" ? page.locator(".inline-insight-region").last() : page.getByRole("region", { name: labels[kind] || "Note", exact: true });
    const close = async kind => {
      await surface(kind).getByRole("button", { name: kind === "insight" ? "Dismiss" : "Close", exact: true }).click();
      await surface(kind).waitFor({ state: "hidden" });
    };
    const idle = async kind => {
      await surface(kind).waitFor();
      if (kind === "summary") await surface(kind).getByLabel("Loading", { exact: true }).waitFor({ state: "hidden" });
      else await until(async () => !/^(Loading|Generating)$/m.test(await surface(kind).innerText()));
    };
    const visibleContent = async kind => {
      const text = { insight: "Saved insight", summary: "Saved summary", quiz: "Saved quiz question", "critical-thinking": "Saved critical question" }[kind];
      await surface(kind).getByText(text, { exact: kind !== "insight" }).first().waitFor();
    };
    await page.goto(`${process.env.UI_URL || "http://127.0.0.1:5173"}/documents/Book`);
    await page.getByRole("button", { name: "Document reading interactions", exact: true }).waitFor().catch(async error => { console.log(await page.locator("body").innerText(), errors, unexpected); throw error; });
    assert.equal(calls.length, 0, "Navigation is passive");
    await page.getByRole("button", { name: "Document reading interactions", exact: true }).click();
    await page.getByRole("menuitem", { name: "Insights", exact: true }).waitFor();
    assert.equal(calls.length, 0, "Opening menu is passive");
    await page.keyboard.press("Escape");
    for (const kind of interactionKinds) {
      mode = "missing"; calls.length = 0;
      await open(kind); await visibleContent(kind); await idle(kind);
      assert.deepEqual(calls.map(call => call.operation), ["read", kind === "critical-thinking" ? "generate-question" : "generate"], `${kind}: one selection generates once`);
      assert.deepEqual(calls[0].target, calls[1].target);
      if (kind === "summary" || kind === "critical-thinking") {
        await page.getByRole("menu").waitFor({ state: "hidden" });
        await pause(350); // Let MUI drawer transition settle before visual inspection.
        await page.screenshot({ path: `/tmp/interaction-activation-${kind}-${width}.png` });
        const size = await surface(kind).evaluate(el => ({ width: el.clientWidth, scroll: el.scrollWidth }));
        assert.ok(size.scroll <= size.width + 1, "No horizontal overflow");
      }
      await close(kind);
      mode = "saved"; calls.length = 0;
      await open(kind); await visibleContent(kind); await idle(kind);
      assert.deepEqual(calls.map(call => call.operation), ["read"], `${kind}: existing content is never regenerated`);
      if (kind === "critical-thinking") {
        await surface(kind).getByText("My saved answer", { exact: true }).waitFor();
        await surface(kind).getByText("Saved feedback", { exact: true }).waitFor();
      }
      await close(kind);
      for (const fault of ["http-error", "http-missing", "malformed", "mismatch", "wrong-kind", "contradictory", "missing-payload", "stale_target", "insufficient_content", "validation_failed", "generation_failed"]) {
        mode = fault; calls.length = 0;
        await open(kind); await until(() => calls.length > 0); await idle(kind);
        assert.deepEqual(calls.map(call => call.operation), ["read"], `${kind}/${fault} must not generate`);
        await close(kind);
      }
      if (kind === "summary") {
        mode = "stale_target"; calls.length = 0;
        await open(kind); await idle(kind);
        assert.deepEqual(calls.map(call => call.operation), ["read"]);
        await surface(kind).getByRole("button", { name: "Refresh", exact: true }).click();
        await visibleContent(kind); await idle(kind);
        assert.deepEqual(calls.map(call => call.operation), ["read", "refresh"], "Stale summary requires explicit refresh");
        await close(kind);
      }
      // Closing while read is pending must suppress the subsequent mutation.
      mode = "missing"; calls.length = 0; holdPhase = "read"; release = null;
      await open(kind); await until(() => release); await close(kind); release();
      await pause(100);
      assert.deepEqual(calls.map(call => call.operation), ["read"], `${kind}: close cancels chain`);
      assert.equal(await surface(kind).isVisible(), false);
      // A dispatched generation may finish server-side after close; never reopen stale UI.
      calls.length = 0; holdPhase = kind === "critical-thinking" ? "generate-question" : "generate"; release = null;
      await open(kind); await until(() => release); await close(kind); release(); await pause(100);
      assert.equal(calls.length, 2); assert.equal(await surface(kind).isVisible(), false);
      // Failed generation does not retry. Explicit retry reconciles a saved outcome.
      mode = "generation-error"; calls.length = 0;
      await open(kind); await until(() => calls.length === 2); await idle(kind); await pause(100);
      assert.equal(calls.length, 2);
      mode = "saved";
      const retry = kind === "summary" ? surface(kind).getByRole("button", { name: "Retry summary", exact: true })
        : surface(kind).locator(kind === "insight" ? ".inline-insight-generate" : kind === "quiz" ? ".quiz-drawer-generate" : ".critical-thinking-generate");
      await retry.click(); await visibleContent(kind); await idle(kind);
      assert.deepEqual(calls.map(call => call.operation), ["read", kind === "critical-thinking" ? "generate-question" : "generate", "read"]);
      await close(kind);
      console.log(`PASS ${width}px ${kind}: one-click, saved, 11 read faults, close, explicit retry reconciliation`);
    }
    // Same inline action repeated during its read is deduplicated.
    mode = "missing"; calls.length = 0; holdPhase = "read"; release = null;
    await open("insight"); await until(() => release);
    await open("insight"); assert.equal(calls.length, 1);
    release(); await visibleContent("insight"); await idle("insight"); assert.equal(calls.length, 2);
    await close("insight");
    // A superseding target must not generate for the old target.
    calls.length = 0; holdPhase = "read"; release = null;
    await open("insight"); await until(() => release); const releaseOld = release;
    await open("insight", true); await visibleContent("insight"); releaseOld(); await pause(100);
    assert.equal(calls.filter(call => call.operation === "generate").length, 1);
    assert.equal(calls.find(call => call.operation === "generate").target.chapter_id, "c1");
    // Note remains read-only and unsaved draft close stays protected.
    calls.length = 0; await open("note");
    const note = surface("note"); await note.getByLabel("Loading", { exact: true }).waitFor({ state: "hidden" });
    assert.deepEqual(calls.map(call => call.operation), ["read"]);
    await note.getByRole("textbox", { name: "Note", exact: true }).fill("Unsaved note");
    page.once("dialog", dialog => dialog.dismiss());
    await note.getByRole("button", { name: "Close", exact: true }).click();
    assert.equal(await note.getByRole("textbox", { name: "Note", exact: true }).inputValue(), "Unsaved note");
    page.once("dialog", dialog => dialog.accept()); await close("note");
    // Route changes reset both controller activation and the keyed Summary drawer.
    for (const kind of interactionKinds) {
      await page.goto(`${process.env.UI_URL || "http://127.0.0.1:5173"}/documents/Book`);
      mode = "missing"; calls.length = 0; holdPhase = "read"; release = null;
      await open(kind); await until(() => release);
      // Native click simulates application navigation while a modal is pending.
      await page.getByRole("button", { name: "Library", exact: true, includeHidden: true }).evaluate(button => button.click());
      await page.locator(".file-management-row").filter({ hasText: "Other" }).getByRole("button", { name: "Open", exact: true }).click();
      await page.waitForURL("**/documents/Other"); release(); await pause(100);
      assert.deepEqual(calls.map(call => call.operation), ["read"], `${kind}: navigation cancels activation`);
      assert.equal(await surface(kind).isVisible(), false);
    }
    assert.deepEqual(unexpected, []); assert.deepEqual(errors, []);
    await page.close();
  }
} finally { await browser.close(); }
