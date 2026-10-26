// Actual backed-up OCR book, live-model result and disposable production API/SQL.
import assert from "node:assert/strict";
import path from "node:path";
import { readFile } from "node:fs/promises";
const api = process.env.CLEANUP_TEST_API;
const output = process.env.CLEANUP_EVIDENCE_DIR;
const recovery = process.env.CLEANUP_RECOVERY ? JSON.parse(process.env.CLEANUP_RECOVERY) : null;
assert.match(api || "", /^http:\/\/127\.0\.0\.1:\d+$/);
assert.ok(output);
const { chromium } = await import(process.env.PLAYWRIGHT_MODULE || "playwright");
const browser = await chromium.launch({ headless: true,
  ...(process.env.CHROMIUM_PATH ? { executablePath: process.env.CHROMIUM_PATH } : {}) });
try {
  for (const width of [1280, 390]) {
    const page = await browser.newPage({ viewport: { width, height: 900 } });
    if (process.env.CLEANUP_PREFLIGHT_ONLY === "1") {
      await acceptPreflight(page, width);
      await page.close();
      continue;
    }
    const errors = [];
    if (recovery) await page.addInitScript(handle => {
      sessionStorage.setItem("deep-reader.smart-cleanup:book", JSON.stringify(handle));
    }, recovery);
    page.on("pageerror", error => errors.push(error.message));
    await page.route("**/api/**", async route => {
      const url = new URL(route.request().url());
      const endpoint = url.pathname.replace(/^\/api/, "");
      if (["/documents", "/documents/management"].includes(endpoint)) {
        return route.fulfill({ json: { items: [], total: 0 } });
      }
      assert.notEqual(endpoint, "/documents/prepare-task-layout");
      assert.notEqual(endpoint, "/documents/reparse-structure");
      const response = await route.fetch({ url: api + endpoint + url.search });
      assert.ok(response.status() < 400, `${endpoint}: ${response.status()}`);
      return route.fulfill({ response });
    });
    await page.goto(`${process.env.UI_URL || "http://127.0.0.1:5174"}/documents/book`);
    await page.locator(".hierarchy-section-button").first().click();
    await page.locator(".reader-content-block-body").first().waitFor();
    if (recovery) {
      await page.locator("#repair-menu-button").click();
      await page.locator(".smart-cleanup-menu-item").click();
      const cleanup = page.getByRole("dialog", { name: "Smart cleanup", exact: true });
      await cleanup.getByLabel(/Checking/).waitFor({ state: "hidden" });
      await cleanup.getByRole("button", { name: /Check again|Retry cleanup/ }).waitFor();
      await page.screenshot({ path: path.join(output, `standalone-${width}.png`), animations: "disabled" });
      await cleanup.getByRole("button", { name: /Check again|Retry cleanup/ }).click();
      await cleanup.getByRole("button", { name: "Clean text", exact: true }).waitFor();
      if (width === 1280) {
        await cleanup.getByRole("button", { name: "Clean text", exact: true }).click();
        await cleanup.getByRole("button", { name: /Check again|Retry cleanup/ }).waitFor({ timeout: 120000 });
        await page.screenshot({ path: path.join(output, `standalone-executed-${width}.png`), animations: "disabled" });
      }
      await cleanup.getByRole("button", { name: /Cancel|Close/, exact: true }).click();
    }
    await page.getByRole("button", { name: "Review source", exact: true }).click();
    const panel = page.locator('div[aria-labelledby="reading-content-dialog-title"]');
    await panel.getByLabel("Loading content changes").waitFor({ state: "hidden" });
    const state = async () => (await page.request.get(api + "/documents/book/reading-content?include_cleanup_audit=true")).json();
    const before = await state();
    if (before.cleanup_audit.total) {
      await panel.getByRole("button", { name: /^Expand cleanup/ }).first().click();
      await panel.locator("blockquote").first().waitFor();
      assert.ok(await panel.getByText(/Page \d/).count());
      await page.screenshot({ path: path.join(output, `audit-${width}.png`), animations: "disabled" });
      await panel.getByRole("button", { name: /^Collapse cleanup/ }).first().click();
      await panel.locator("blockquote").waitFor({ state: "hidden" });
      if (width === 390) {
        await panel.getByRole("button", { name: "Undo OCR cleanup", exact: true }).click();
        await panel.getByText("Undone", { exact: true }).waitFor();
        assert.equal((await state()).revision, before.revision + 1);
      }
    } else {
      await panel.getByText("No applied OCR cleanup recorded.").waitFor();
      await page.screenshot({ path: path.join(output, `audit-${width}.png`), animations: "disabled" });
    }
    assert.equal(await panel.evaluate(el => el.scrollWidth > el.clientWidth), false);
    await panel.getByRole("button", { name: "Close content editor" }).click();
    await page.reload();
    await page.locator(".hierarchy-section-button").first().click();
    await page.locator(".reader-content-block-body").first().waitFor();
    assert.deepEqual(errors, []);
    console.log(`PASS real-book ${width}px: reading, audit, collapse, reload${width === 390 && before.cleanup_audit.total ? ", Undo" : ""}`);
    await page.close();
  }
} finally { await browser.close(); }

async function acceptPreflight(page, width) {
  const document = JSON.parse(await readFile(process.env.CLEANUP_BOOK_DOCUMENT, "utf8"));
  const name = document.document_id;
  const errors = [];
  let release, pending;
  let requests = 0;
  page.on("pageerror", error => errors.push(error.message));
  // Navigation is a read-only projection fixture from the immutable book. The
  // preflight itself traverses the real local HTTP route/adapter/token counter.
  await page.route("**/api/**", async route => {
    const endpoint = new URL(route.request().url()).pathname.replace(/^\/api/, "");
    if (["/documents", "/documents/management"].includes(endpoint)) {
      return route.fulfill({ json: { items: [], total: 0 } });
    }
    if (endpoint === "/documents/task-layout") return route.fulfill({ json: {
      document_id: name, title: document.title, chapters: document.chapters,
    } });
    if (endpoint.endsWith("/reading-content")) {
      const response = await route.fetch({ url: api + endpoint });
      assert.equal(response.status(), 200);
      return route.fulfill({ response });
    }
    if (endpoint.endsWith("/smart-cleanup/preflight")) {
      assert.equal(decodeURIComponent(endpoint), `/documents/${name}/smart-cleanup/preflight`);
      requests++;
      const response = await route.fetch({ url: api + endpoint });
      assert.equal(response.status(), 200);
      assert.equal((await response.json()).execution_available, true);
      pending = new Promise(resolve => { release = resolve; });
      await pending;
      await route.fulfill({ response }).catch(() => {}); // A closed page may discard the held response.
      return;
    }
    throw new Error(`Unexpected request in read-only acceptance: ${endpoint}`);
  });
  await page.goto(`${process.env.UI_URL || "http://127.0.0.1:5173"}/documents/${encodeURIComponent(name)}`);
  await page.locator(".hierarchy-section-button").first().waitFor();
  const dialog = page.getByRole("dialog", { name: "Smart cleanup", exact: true });
  const open = async () => {
    release = null;
    await page.locator("#repair-menu-button").click();
    await page.getByRole("menuitem", { name: "Smart cleanup", exact: true }).click();
    await dialog.getByLabel("Checking cleanup readiness").waitFor();
    for (let i = 0; i < 500 && !release; i++) await page.waitForTimeout(20);
    assert.ok(release, "Actual preflight did not return");
  };
  await page.clock.install();
  await open();
  await page.clock.fastForward(120001);
  assert.equal(await dialog.getByLabel("Checking cleanup readiness").isVisible(), true);
  assert.equal(await dialog.locator(".smart-cleanup-close").isEnabled(), true);
  await page.screenshot({ path: path.join(output, `full-book-pending-${width}.png`), animations: "disabled" });
  release(); await pending;
  await dialog.getByText("Ready for smart cleanup.", { exact: true }).waitFor();
  assert.equal(await dialog.getByRole("button", { name: "Clean text", exact: true }).isEnabled(), true);
  await page.screenshot({ path: path.join(output, `full-book-ready-${width}.png`), animations: "disabled" });
  const bounds = await dialog.boundingBox();
  assert.ok(bounds.x >= 0 && bounds.x + bounds.width <= width && bounds.height <= 900);
  await dialog.locator(".smart-cleanup-close").click();
  await open();
  await page.clock.fastForward(120001);
  await dialog.locator(".smart-cleanup-close").click();
  release(); await pending;
  await dialog.waitFor({ state: "hidden" });
  await page.waitForTimeout(100);
  assert.equal(await dialog.count(), 0);
  assert.equal(requests, 2);
  assert.deepEqual(errors, []);
  console.log(`PASS full-book preflight ${width}px: 120s pending, success, cancel/late response, no execution`);
}
