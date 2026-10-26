// Real App/FastAPI/disposable SQL; controlled provider, explicit transport fault fixtures.
import assert from "node:assert/strict";
const api = process.env.CLEANUP_TEST_API;
assert.match(api || "", /^http:\/\/127\.0\.0\.1:\d+$/);
const { chromium } = await import(process.env.PLAYWRIGHT_MODULE || "playwright");
const browser = await chromium.launch({ headless: true,
  ...(process.env.CHROMIUM_PATH ? { executablePath: process.env.CHROMIUM_PATH } : {}) });
try {
  for (const width of [1280, 390]) {
    const page = await browser.newPage({ viewport: { width, height: 900 } });
    const errors = [], requests = [];
    let fault = null, release, hold = false, lost = false, heldPreflight = false;
    page.on("pageerror", error => errors.push(error.message));
    await page.route("**/api/**", async route => {
      const path = new URL(route.request().url()).pathname.replace(/^\/api/, "");
      requests.push(path);
      assert.ok(!path.includes("reparse-structure") && !path.includes("prepare-task-layout") && !path.includes("/prepare"), path);
      if (["/documents", "/documents/management"].includes(path)) return route.fulfill({ json: { items: [], total: 0 } });
      if (path.endsWith("/note/read")) return route.fulfill({ json: {
        target: route.request().postDataJSON().target, target_token: "note-fixture", max_chars: 10000, note: null } });
      if (path.endsWith("/smart-cleanup/preflight")) {
        assert.deepEqual(route.request().postDataJSON(), {});
        if (heldPreflight) await new Promise(resolve => { release = resolve; });
        if (fault === "unsupported") return route.fulfill({ status: 404, json: { detail: "Not Found" } });
        if (fault === "invalid") return route.fulfill({ json: { execution_available: true } });
        if (fault && fault !== "expired") return route.fulfill({ json: { execution_available: false, reason: fault,
          operation_id: null, confirmation_token: null, expires_in_seconds: null, impact: {
            generated_invalidation: "only_if_reading_content_changed", preserves_notes: true,
            preserves_original: true, preserves_manual_edits: true, preserves_structure: true } } });
      }
      if (path.endsWith("/smart-cleanup")) {
        assert.deepEqual(Object.keys(route.request().postDataJSON()).sort(), ["confirmation_token", "operation_id"]);
        if (hold) await new Promise(resolve => { release = resolve; });
      }
      const response = await route.fetch({ url: api + path + new URL(route.request().url()).search });
      if (path.endsWith("/smart-cleanup/preflight") && fault === "expired") {
        return route.fulfill({ json: { ...await response.json(), expires_in_seconds: 1 } });
      }
      if (path.endsWith("/smart-cleanup") && lost) { lost = false; return route.abort(); }
      await route.fulfill({ response });
    });
    const dialog = page.getByRole("dialog", { name: "Smart cleanup", exact: true });
    const stats = async () => (await page.request.get(api + "/stats")).json();
    const load = async (scenario = "completed", select = true) => {
      assert.equal((await page.request.post(api + "/fixture", { data: { scenario } })).status(), 204);
      await page.goto(`${process.env.UI_URL || "http://127.0.0.1:5174"}/documents/book`);
      await page.evaluate(() => sessionStorage.clear());
      if (select) {
        await page.locator(".hierarchy-section-button").first().click();
        await page.locator(".reader-content-block-body").first().waitFor();
        assert.match(await page.locator("#repair-menu-button").innerText(), /Common/);
      }
    };
    const open = async () => {
      await page.locator("#repair-menu-button").click();
      await page.getByRole("menuitem", { name: "Smart cleanup", exact: true }).click();
      await dialog.waitFor();
    };
    const close = async () => { await dialog.locator(".smart-cleanup-close").click(); await dialog.waitFor({ state: "hidden" }); };
    for (const [scenario, message, changed] of [
      ["completed", /OCR cleanup completed/, true], ["partial", /Some OCR fragments were cleaned/, true],
      ["failed", /Cleanup could not finish/, false], ["no_changes", /No OCR cleanup changes/, false],
      ["unsafe", /Cleanup was skipped/, false],
    ]) {
      await load(scenario); await open();
      await dialog.getByText("Ready for smart cleanup.", { exact: true }).waitFor();
      assert.equal((await stats()).calls, 0);
      assert.match(await dialog.innerText(), /saved notes and manual edits are preserved/);
      assert.doesNotMatch(await dialog.innerText(), /fixture-model|tokens|confirmation_token|reading_revision/);
      const reads = requests.filter(p => p.endsWith("/task-units/content")).length;
      await dialog.getByRole("button", { name: "Clean text" }).evaluate(el => { el.click(); el.click(); });
      await dialog.getByText(message).waitFor();
      assert.equal((await stats()).calls, 1);
      assert.equal((await stats()).builds, 0);
      assert.equal((await stats()).revision, changed ? 1 : 0);
      assert.equal(requests.filter(p => p.endsWith("/task-units/content")).length - reads, changed ? 1 : 0);
      await page.screenshot({ path: `/tmp/smart-cleanup-${scenario}-${width}.png`, animations: "disabled" });
      const bounds = await dialog.boundingBox();
      assert.ok(bounds.x >= 0 && bounds.x + bounds.width <= width && bounds.height <= 900);
      await close();
    }
    await load("missing_layout", false); await open();
    await dialog.getByText("Ready for smart cleanup.", { exact: true }).waitFor();
    await dialog.getByRole("button", { name: "Clean text" }).click();
    await dialog.getByText(/OCR cleanup completed/).waitFor(); await close();
    await page.locator(".hierarchy-section-button").first().click();
    await page.locator(".reader-content-block-body").first().waitFor();
    assert.equal((await stats()).builds, 0);

    for (const [reason, message] of [["native_source", /does not contain OCR/], ["ocr_evidence_missing", /cannot be verified/],
      ["capacity_exceeded", /exceeds the current model/], ["cleanup_service_unavailable", /currently unavailable/],
      ["unsupported", /does not support/], ["invalid", /does not support/]]) {
      await load(); fault = reason; await open(); await dialog.getByText(message).waitFor();
      assert.equal(await dialog.getByRole("button", { name: "Clean text" }).count(), 0);
      assert.equal((await stats()).calls, 0); await close(); fault = null;
    }
    await load(); fault = "expired"; await open();
    await dialog.getByText(/Confirmation expired/).waitFor();
    assert.equal(await dialog.getByRole("button", { name: "Clean text" }).isDisabled(), true);
    assert.equal((await stats()).calls, 0); await close(); fault = null;
    await load(); await open();
    await dialog.getByText("Ready for smart cleanup.", { exact: true }).waitFor();
    await page.screenshot({ path: `/tmp/smart-cleanup-consent-${width}.png`, animations: "disabled" });
    await page.keyboard.press("Escape"); await dialog.waitFor({ state: "hidden" });
    assert.equal((await stats()).calls, 0);

    await page.locator(".reader-content-block-interaction-trigger").first().click();
    await page.getByRole("menuitem", { name: "Edit", exact: true }).click();
    await page.getByRole("textbox", { name: "Block text" }).fill("Preserve this unsaved block edit.");
    await page.locator("#repair-menu-button").evaluate(el => el.click());
    assert.equal(await page.getByRole("menuitem", { name: "Smart cleanup", exact: true }).getAttribute("aria-disabled"), "true");
    await page.keyboard.press("Escape");
    assert.equal(await page.getByRole("textbox", { name: "Block text" }).inputValue(), "Preserve this unsaved block edit.");
    page.once("dialog", prompt => prompt.accept());
    await page.getByRole("button", { name: "Close content editor" }).click();

    // Keep a mounted note draft while cleanup refreshes generated content.
    await page.locator(".hierarchy-chapter-interaction-trigger").first().click();
    await page.getByRole("menuitem", { name: "Note", exact: true }).click();
    await page.getByRole("textbox", { name: "Note", exact: true }).fill("Retain this unsaved note.");
    await page.locator("#repair-menu-button").evaluate(el => el.click());
    await page.getByRole("menuitem", { name: "Smart cleanup", exact: true }).click();
    await dialog.getByText("Ready for smart cleanup.", { exact: true }).waitFor();
    await dialog.getByRole("button", { name: "Clean text" }).click();
    await dialog.getByText(/OCR cleanup completed/).waitFor(); await close();
    assert.equal(await page.getByRole("textbox", { name: "Note", exact: true }).inputValue(), "Retain this unsaved note.");
    page.once("dialog", prompt => prompt.accept());
    await page.getByRole("region", { name: "Note", exact: true }).getByRole("button", { name: "Close", exact: true }).click();

    await load(); await page.clock.install(); heldPreflight = true; release = null; await open();
    await page.getByLabel("Checking cleanup readiness").waitFor();
    for (let n = 0; n < 100 && !release; n++) await page.waitForTimeout(20);
    assert.ok(release);
    await page.clock.fastForward(120001);
    assert.equal(await page.getByLabel("Checking cleanup readiness").isVisible(), true);
    heldPreflight = false; release();
    await dialog.getByText("Ready for smart cleanup.", { exact: true }).waitFor();
    hold = true; release = null;
    await dialog.getByRole("button", { name: "Clean text" }).evaluate(el => { el.click(); el.click(); });
    for (let n = 0; n < 100 && !release; n++) await page.waitForTimeout(20);
    assert.ok(release);
    await page.clock.fastForward(120001);
    assert.equal(await page.getByLabel("Cleanup in progress").isVisible(), true);
    assert.equal(requests.filter(p => p.endsWith("/smart-cleanup")).length > 0, true);
    await page.screenshot({ path: `/tmp/smart-cleanup-long-wait-${width}.png`, animations: "disabled" });
    hold = false; release();
    await dialog.getByText(/OCR cleanup completed/).waitFor();
    assert.equal((await stats()).calls, 1); await close();
    await page.clock.resume();

    await load(); heldPreflight = true; await open();
    await page.getByLabel("Checking cleanup readiness").waitFor();
    await close(); heldPreflight = false; release();
    await open(); await dialog.getByText("Ready for smart cleanup.", { exact: true }).waitFor();
    hold = true; lost = true;
    await dialog.getByRole("button", { name: "Clean text" }).click();
    await page.getByLabel("Cleanup in progress").waitFor();
    assert.equal(await dialog.getByRole("button", { name: "Clean text" }).count(), 0);
    hold = false; release();
    await dialog.getByText(/result is not confirmed/).waitFor();
    const before = requests.filter(p => p.endsWith("/smart-cleanup")).length;
    await close(); await open();
    await dialog.getByText(/OCR cleanup completed/).waitFor();
    assert.equal(requests.filter(p => p.endsWith("/smart-cleanup")).length, before);
    await close(); await page.reload(); await open();
    await dialog.getByText("Ready for smart cleanup.", { exact: true }).waitFor();
    assert.equal((await stats()).calls, 1); await close();

    await load(); await open();
    await dialog.getByText("Ready for smart cleanup.", { exact: true }).waitFor();
    hold = true;
    await dialog.getByRole("button", { name: "Clean text" }).click();
    await page.getByLabel("Cleanup in progress").waitFor();
    await page.evaluate(() => { history.pushState({}, "", "/"); dispatchEvent(new PopStateEvent("popstate")); });
    await dialog.waitFor({ state: "hidden" });
    hold = false; release();
    for (let attempt = 0; attempt < 100 && (await stats()).calls === 0; attempt++) await page.waitForTimeout(50);
    assert.equal((await stats()).calls, 1);
    assert.equal(await page.getByText(/OCR cleanup completed/).count(), 0);
    assert.equal(await dialog.count(), 0);
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth), false);
    assert.deepEqual(errors, []);
    console.log(`PASS ${width}px standalone cleanup: actual API/SQL outcomes, missing layout, consent, duplicate submit, notes, unavailable contracts, cancellation and unknown protection`);
    await page.close();
  }
} finally { await browser.close(); }
