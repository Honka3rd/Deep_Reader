// Served exclusively by the disposable API/PostgreSQL harness.
import assert from "node:assert/strict";
const api = process.env.ENHANCEMENT_TEST_API;
assert.ok(api?.startsWith("http://127.0.0.1:"), "Disposable API required");
const { chromium } = await import(process.env.PLAYWRIGHT_MODULE || "playwright");
const browser = await chromium.launch({ headless: true,
  ...(process.env.CHROMIUM_PATH ? { executablePath: process.env.CHROMIUM_PATH } : {}) });
try {
  for (const width of [1280, 390]) {
    const page = await browser.newPage({ viewport: { width, height: 900 } });
    const errors = [], outcomes = [], requests = [], sequence = [];
    let loseResponse = false, failPrepare = false;
    page.on("pageerror", error => errors.push(error.message));
    await page.route("**/api/**", async route => {
      const path = new URL(route.request().url()).pathname.replace(/^\/api/, "");
      if (["/documents", "/documents/management"].includes(path)) return route.fulfill({ json: { items: [], total: 0 } });
      assert.ok(["/documents/reparse-structure", "/documents/reparse-structure/preflight",
        "/documents/reparse-structure/result", "/documents/task-layout", "/documents/prepare-task-layout"].includes(path));
      sequence.push(path);
      if (path === "/documents/reparse-structure") requests.push(route.request().postDataJSON());
      if (path === "/documents/prepare-task-layout" && failPrepare) {
        failPrepare = false;
        return route.fulfill({ status: 503, json: { detail: "synthetic recovery failure" } });
      }
      const response = await route.fetch({ url: api + path });
      if (path === "/documents/task-layout") assert.equal(response.status(), 200);
      if (path === "/documents/reparse-structure" && response.ok()) {
        outcomes.push(await response.json());
        if (loseResponse) { loseResponse = false; return route.abort(); }
      }
      await route.fulfill({ response });
    });
    const configure = async scenario => {
      assert.equal((await page.request.post(api + "/fixture", { data: { scenario } })).status(), 204);
    };
    await configure("fits");
    await configure("seed");
    await page.goto(`${process.env.UI_URL || "http://127.0.0.1:5173"}/documents/doc`);
    const dialog = page.getByRole("dialog");
    const enhance = page.getByRole("button", { name: "Enhance", exact: true });
    const open = async () => {
      await page.locator("#repair-menu-button").click();
      await page.getByRole("menuitem", { name: "LLM enhance", exact: true }).click();
      await page.getByText("Ready for smart structuring.", { exact: true }).waitFor();
    };
    const closeNoop = async () => {
      await dialog.getByText(/No structure changes/).waitFor();
      await dialog.getByRole("button", { name: "Close", exact: true }).click();
      await dialog.waitFor({ state: "hidden" });
    };
    await configure(`titles_${width}`);
    await open();
    await enhance.click();
    await dialog.waitFor({ state: "hidden" });
    assert.equal(outcomes.at(-1).change_kind, "titles_only");
    await page.locator(".hierarchy-section-button").filter({ hasText: `titles_${width}` }).first().waitFor();
    const reads = sequence.filter(path => path === "/documents/task-layout").length;
    await open();
    await enhance.click();
    await closeNoop();
    assert.equal(outcomes.at(-1).change_kind, "unchanged");
    assert.equal(sequence.filter(path => path === "/documents/task-layout").length, reads);
    await configure("transient");
    await open();
    await enhance.click();
    await page.getByRole("button", { name: "Retry smart structuring" }).click();
    await closeNoop();
    assert.deepEqual(requests.at(-1), requests.at(-2));
    await configure(`titles_recovered_${width}`);
    await open();
    loseResponse = true;
    await enhance.click();
    await page.getByRole("button", { name: "Check stored state" }).click();
    await dialog.waitFor({ state: "hidden" });
    assert.equal(outcomes.at(-1).change_kind, "titles_only");
    assert.equal(sequence.filter(path => path === "/documents/prepare-task-layout").length, 0);
    await configure("structural");
    await open();
    loseResponse = true;
    failPrepare = true;
    await enhance.click();
    await page.getByRole("button", { name: "Check stored state" }).click();
    await dialog.getByText(/Structure saved, but reading preparation failed/).waitFor();
    const dispatched = requests.length;
    await page.getByRole("button", { name: "Reload document" }).click();
    await dialog.waitFor({ state: "hidden" });
    assert.equal(requests.length, dispatched, "Recovery must not regenerate");
    assert.equal(outcomes.at(-1).change_kind, "structure_changed");
    assert.deepEqual(sequence.slice(-2), ["/documents/prepare-task-layout", "/documents/task-layout"]);
    assert.equal(requests.length, 6);
    assert.deepEqual(errors, []);
    await page.screenshot({ path: `/tmp/enhancement-lifecycle-live-${width}.png`, animations: "disabled" });
    console.log(`PASS ${width}px real API/DB: all three outcomes, transient retry, lost title/structural response, preparation-only recovery`);
    await page.close();
  }
} finally { await browser.close(); }
