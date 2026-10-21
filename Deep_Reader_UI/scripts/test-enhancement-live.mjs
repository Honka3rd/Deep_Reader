// Run through accept_enhancement_browser.py, never against a production API.
import assert from "node:assert/strict";
const api = process.env.ENHANCEMENT_TEST_API;
assert.ok(api?.startsWith("http://127.0.0.1:"), "Disposable API harness required");
const { chromium } = await import(process.env.PLAYWRIGHT_MODULE || "playwright");
const browser = await chromium.launch({ headless: true,
  ...(process.env.CHROMIUM_PATH ? { executablePath: process.env.CHROMIUM_PATH } : {}) });
try {
  for (const width of [1280, 390]) {
    const page = await browser.newPage({ viewport: { width, height: 900 } });
    let commits = 0, prepares = 0, attempts = 0, expectedStatus = 200;
    let changeKind;
    const errors = [];
    page.on("pageerror", error => errors.push(error.message));
    await page.route("**/api/**", async route => {
      const path = new URL(route.request().url()).pathname.replace(/^\/api/, "");
      if (["/documents", "/documents/management"].includes(path)) return route.fulfill({ json: { items: [], total: 0 } });
      assert.ok(["/documents/reparse-structure", "/documents/reparse-structure/preflight",
        "/documents/task-layout", "/documents/prepare-task-layout"].includes(path), `Unexpected ${path}`);
      if (path === "/documents/reparse-structure") {
        attempts++;
        assert.equal(route.request().postDataJSON().discard_interactions, true);
      }
      if (path === "/documents/prepare-task-layout") {
        prepares++;
        assert.equal(route.request().postDataJSON().layout_only, true);
      }
      const response = await route.fetch({ url: api + path });
      if (path === "/documents/task-layout") assert.equal(response.status(), 200, "Layout must be prepared before reading");
      if (path === "/documents/reparse-structure") {
        assert.equal(response.status(), expectedStatus);
        if (response.status() === 200) {
          const result = await response.json();
          changeKind = result.change_kind;
          if (result.committed) commits++;
          assert.ok(result.structure_version > 1);
        }
      }
      await route.fulfill({ response });
    });
    await page.goto(`${process.env.UI_URL || "http://127.0.0.1:5173"}/documents/doc`);
    const open = async () => {
      await page.locator("#repair-menu-button").click();
      await page.getByRole("menuitem", { name: "LLM enhance", exact: true }).click();
      await page.getByRole("button", { name: "Enhance", exact: true }).waitFor();
      await page.getByLabel("Checking readiness").waitFor({ state: "hidden" });
    };
    const configure = async scenario => {
      const response = await page.request.post(api + "/fixture", { data: { scenario } });
      assert.equal(response.status(), 204);
    };
    const cancel = async () => {
      await page.getByRole("button", { name: "Cancel", exact: true }).click();
      await page.getByRole("dialog").waitFor({ state: "hidden" });
    };
    const submit = page.getByRole("button", { name: "Enhance", exact: true });
    for (const [scenario, message] of [
      ["exceeded", /too long for smart structuring/],
      ["unavailable", /service is not configured/],
      ["resource_limited", /service's processing limits/],
    ]) {
      await configure(scenario);
      await open();
      await page.getByRole("dialog").getByText(message).waitFor();
      assert.equal(await submit.isDisabled(), true);
      assert.equal(attempts, 0);
      await page.getByRole("dialog").getByText(message).scrollIntoViewIfNeeded();
      await page.screenshot({ path: `/tmp/enhancement-live-${scenario}-${width}.png`,
        fullPage: true, animations: "disabled" });
      await cancel();
    }
    for (const [scenario, status, message] of [
      ["exceeded", 413, /too long for smart structuring/],
      ["unavailable", 503, /service is not configured/],
      ["resource_limited", 413, /service's processing limits/],
      ["changed_model", 409, /Enhancement was not applied/],
      ["changed_policy", 409, /Enhancement was not applied/],
      ["changed_source", 409, /Enhancement was not applied/],
    ]) {
      await configure("fits");
      await open();
      assert.equal(await submit.isEnabled(), true);
      await configure(scenario);
      expectedStatus = status;
      const previous = attempts;
      await submit.click();
      await page.getByRole("dialog").getByText(message).waitFor();
      assert.equal(await submit.isDisabled(), true);
      assert.equal(attempts, previous + 1);
      assert.equal(commits, 0);
      await cancel();
    }
    await configure("fits");
    expectedStatus = 200;
    await open();
    await page.getByRole("button", { name: "Cancel", exact: true }).click();
    assert.equal(commits, 0);
    await open();
    await submit.click();
    if (width === 390) {
      await page.getByRole("dialog").getByText(/No structure changes/).waitFor();
      await page.getByRole("dialog").getByRole("button", { name: "Close", exact: true }).click();
    }
    await page.getByRole("dialog").waitFor({ state: "hidden" });
    await page.screenshot({ path: `/tmp/enhancement-live-${width}.png`, fullPage: true, animations: "disabled" });
    assert.equal(changeKind, width === 1280 ? "structure_changed" : "unchanged");
    assert.equal(prepares, width === 1280 ? 1 : 0);
    assert.equal(commits, width === 1280 ? 1 : 0);
    assert.equal(await page.locator("#repair-menu-button").isEnabled(), true);
    assert.deepEqual(errors, []);
    assert.equal(attempts, 7);
    console.log(`PASS ${width}px real API/DB: capacity/resource/unavailable rejection, model/policy/source invalidation, cancel -> ${changeKind}; conditional automatic prepare; zero layout 409`);
    await page.close();
  }
} finally { await browser.close(); }
