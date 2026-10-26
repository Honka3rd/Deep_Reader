// Production UI -> actual FastAPI -> disposable PostgreSQL. No live model calls.
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
    let lost = false, unknown = false, failContent = false, failResume = false;
    page.on("pageerror", error => errors.push(error.message));
    await page.route("**/api/**", async route => {
      const path = new URL(route.request().url()).pathname.replace(/^\/api/, "");
      if (["/documents", "/documents/management"].includes(path)) return route.fulfill({ json: { items: [], total: 0 } });
      // A UI-only draft fixture; enhancement/storage still use the actual API.
      if (path === "/documents/reading-interactions/note/read") return route.fulfill({ json: {
        target: route.request().postDataJSON().target, target_token: "note-fixture", max_chars: 10000, note: null,
      } });
      assert.notEqual(path, "/documents/prepare-task-layout", "Modern flow must not send manual Prepare");
      requests.push(path);
      if (path.endsWith("/task-units/content") && failContent) {
        failContent = false;
        return route.fulfill({ status: 503, json: { detail: "Temporary read failure" } });
      }
      if (path === "/documents/reparse-structure" && failResume) {
        failResume = false;
        return route.fulfill({ status: 409, json: { detail: "confirmation_stale" } });
      }
      if (path.endsWith("/result") && unknown) return route.fulfill({ json: {
        status: "unknown", result: null, reason: "enhancement_outcome_unknown", retryable: false, durability: "process_local" } });
      const response = await route.fetch({ url: api + path + new URL(route.request().url()).search });
      if (path === "/documents/reparse-structure") {
        assert.equal(route.request().postDataJSON().allow_ocr_cleanup, true);
        if (lost) { lost = false; return route.abort(); }
      }
      await route.fulfill({ response });
    });
    const configure = async scenario => assert.equal((await page.request.post(api + "/fixture", { data: { scenario } })).status(), 204);
    const stats = async () => (await page.request.get(api + "/stats")).json();
    const dialog = page.getByRole("dialog", { name: "Smart structuring", exact: true });
    const open = async () => {
      await page.locator("#repair-menu-button").click();
      await page.getByRole("menuitem", { name: "LLM enhance", exact: true }).click();
      await dialog.waitFor();
      await page.getByLabel("Checking readiness").waitFor({ state: "hidden" });
    };
    const close = async () => {
      await dialog.getByRole("button", { name: "Close", exact: true }).click();
      await dialog.waitFor({ state: "hidden" });
    };
    const load = async scenario => {
      await configure(scenario);
      await page.goto(`${process.env.UI_URL || "http://127.0.0.1:5174"}/documents/book`);
      await page.evaluate(() => sessionStorage.clear());
      await page.locator(".hierarchy-section-button").first().click();
      await page.locator(".reader-content-block-body").first().waitFor({ timeout: 5000 }).catch(async error => {
        console.error(await page.locator("body").innerText());
        throw error;
      });
      await open();
      await dialog.getByText("Ready for smart structuring.", { exact: true }).waitFor();
      await dialog.getByText(/Cleanup alone invalidates generated interactions/).waitFor();
      assert.doesNotMatch(await dialog.innerText(), /Required input tokens|policy_v1|fixture-model|text_sha256/);
    };
    for (const [scenario, message, changed] of [
      ["completed", /OCR cleanup completed/, true], ["partial", /Some OCR fragments were cleaned/, true],
      ["skipped", /OCR cleanup was skipped/, false], ["failed", /OCR cleanup could not finish/, false],
      ["no_changes", /No OCR cleanup changes were needed/, false],
    ]) {
      await load(scenario);
      const before = requests.filter(p => p.endsWith("/task-units/content")).length;
      await dialog.getByRole("button", { name: "Enhance", exact: true }).evaluate(el => { el.click(); el.click(); });
      await dialog.getByText(message).waitFor();
      const state = await stats();
      assert.equal(state.builds, 1);
      assert.equal(state.revision, changed ? 1 : 0);
      assert.equal(requests.filter(p => p.endsWith("/task-units/content")).length - before, changed ? 1 : 0);
      await page.screenshot({ path: `/tmp/cleanup-${scenario}-${width}.png`, animations: "disabled" });
      await close();
      if (changed) assert.equal(await page.getByText("@@??", { exact: true }).count(), 0);
    }
    await load("layout_failed");
    await dialog.getByRole("button", { name: "Enhance", exact: true }).click();
    await dialog.getByRole("button", { name: "Resume reading" }).waitFor();
    const before = await stats();
    await configure("restart");
    await page.reload();
    await open();
    await dialog.getByRole("button", { name: "Resume reading" }).waitFor();
    failResume = true;
    await dialog.getByRole("button", { name: "Resume reading" }).click();
    await dialog.getByText(/Changes were already saved/).waitFor();
    await dialog.getByRole("button", { name: "Check stored state" }).click();
    await dialog.getByRole("button", { name: "Resume reading" }).waitFor();
    await configure("layout_ready");
    await dialog.getByRole("button", { name: "Resume reading" }).click();
    await dialog.getByText(/OCR cleanup completed/).waitFor();
    assert.deepEqual(await stats(), before);
    await close();
    await page.locator(".hierarchy-section-button").first().click();
    await page.locator(".reader-content-block-body").first().waitFor();
    assert.equal(await page.locator(".app-notification").count(), 0);

    await load("completed");
    await dialog.getByRole("button", { name: "Cancel", exact: true }).click();
    await page.locator(".hierarchy-chapter-interaction-trigger").first().click();
    await page.getByRole("menuitem", { name: "Note", exact: true }).click();
    await page.getByRole("textbox", { name: "Note", exact: true }).fill("My unsaved note survives cleanup.");
    // Dispatch the entry action while the draft drawer remains mounted to exercise
    // concurrent view retention, without pretending a pointer can cross its backdrop.
    await page.locator("#repair-menu-button").evaluate(el => el.click());
    await page.getByRole("menuitem", { name: "LLM enhance", exact: true }).click();
    await dialog.getByText("Ready for smart structuring.", { exact: true }).waitFor();
    failContent = true;
    await dialog.getByRole("button", { name: "Enhance", exact: true }).click();
    await dialog.getByText(/reading refresh failed/).waitFor();
    const savedBeforeRefresh = await stats();
    await dialog.getByRole("button", { name: "Reload document" }).click();
    await dialog.getByText(/OCR cleanup completed/).waitFor();
    assert.deepEqual(await stats(), savedBeforeRefresh);
    await close();
    assert.equal(await page.getByRole("textbox", { name: "Note", exact: true }).inputValue(), "My unsaved note survives cleanup.");
    page.once("dialog", prompt => prompt.dismiss());
    await page.getByRole("region", { name: "Note", exact: true }).getByRole("button", { name: "Close", exact: true }).click();
    assert.equal(await page.getByRole("textbox", { name: "Note", exact: true }).inputValue(), "My unsaved note survives cleanup.");
    page.once("dialog", prompt => prompt.accept());
    await page.getByRole("region", { name: "Note", exact: true }).getByRole("button", { name: "Close", exact: true }).click();

    await load("completed");
    lost = true;
    await dialog.getByRole("button", { name: "Enhance", exact: true }).click();
    await dialog.getByRole("button", { name: "Check stored state" }).waitFor();
    const committed = await stats();
    unknown = true;
    await dialog.getByRole("button", { name: "Check stored state" }).click();
    await dialog.getByText(/earlier outcome cannot be confirmed/).waitFor();
    assert.equal(await dialog.getByRole("button", { name: "Enhance", exact: true }).isDisabled(), true);
    unknown = false;
    await configure("restart");
    await page.reload();
    await open();
    await dialog.getByText(/OCR cleanup completed/).waitFor();
    assert.deepEqual(await stats(), committed);
    await close();

    await load("completed");
    await configure("stale");
    await dialog.getByRole("button", { name: "Enhance", exact: true }).click();
    await dialog.getByText(/Enhancement was not applied/).waitFor();
    assert.equal((await stats()).calls, 0);
    await dialog.getByRole("button", { name: "Cancel", exact: true }).click();

    await load("rollback");
    await dialog.getByRole("button", { name: "Enhance", exact: true }).click();
    await dialog.getByRole("button", { name: "Check stored state" }).waitFor();
    assert.equal((await stats()).receipts, 0);
    assert.equal((await stats()).revision, 0);
    await dialog.getByRole("button", { name: "Cancel", exact: true }).click();

    await load("completed");
    lost = true;
    await dialog.getByRole("button", { name: "Enhance", exact: true }).click();
    await dialog.getByRole("button", { name: "Check stored state" }).waitFor();
    await configure("undo");
    await dialog.getByRole("button", { name: "Check stored state" }).click();
    await dialog.getByText(/earlier reading version/).waitFor();
    assert.equal((await stats()).revision, 2);
    await close();

    await load("structure_changed");
    await dialog.getByRole("button", { name: "Enhance", exact: true }).click();
    await dialog.getByText(/Structure saved/).waitFor();
    assert.equal((await stats()).builds, 1);
    await close();
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth), false);
    assert.deepEqual(errors, []);
    console.log(`PASS ${width}px actual API/SQL: consent, outcomes, changed-only reload, duplicate submit, refresh/restart, layout resume, unknown, stale, rollback, Undo supersession`);
    await page.close();
  }
} finally { await browser.close(); }
