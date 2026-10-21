import assert from "node:assert/strict";
import { mkdir } from "node:fs/promises";
import path from "node:path";
const { chromium } = await import(process.env.PLAYWRIGHT_MODULE || "playwright");
const browser = await chromium.launch({ headless: true,
  ...(process.env.CHROMIUM_PATH ? { executablePath: process.env.CHROMIUM_PATH } : {}) });
const output = path.resolve("../Deep_Reflective_Reader/data/test-books/summary-note-20261021/browser");
await mkdir(output, { recursive: true });
try {
  for (const width of [1280, 390]) for (const name of ["Madame Bovary", "红色赌盘_261014_145614"]) {
    const page = await browser.newPage({ viewport: { width, height: 900 } });
    const errors = [], requests = [], mutations = [];
    page.on("pageerror", error => errors.push(error.message));
    page.on("request", request => {
      if (request.url().endsWith("/task-units/content")) requests.push(request.postDataJSON());
      if (/\/prepare|\/generate|\/save|\/refresh/.test(request.url())) mutations.push(request.url());
    });
    await page.goto(`http://127.0.0.1:5173/documents/${encodeURIComponent(name)}`);
    const buttons = page.locator(".hierarchy-section-button");
    await buttons.first().waitFor();
    const counts = await buttons.locator(".hierarchy-section-unit-count").allTextContents();
    const numbers = counts.map(value => Number.parseInt(value, 10));
    const index = numbers.indexOf(Math.max(...numbers));
    const start = Date.now();
    await buttons.nth(index).click();
    await page.waitForFunction(expected => document.querySelectorAll("[data-measure-group]").length === expected,
      numbers[index], { timeout: 30000 });
    const milliseconds = Date.now() - start;
    assert.ok(milliseconds < 20000, `UI content too slow: ${milliseconds}`);
    assert.ok(requests.length >= Math.ceil(numbers[index] / 20));
    assert.ok(requests.every(value => value.task_unit_ids.length <= 20));
    const block = page.locator(".reader-content-block-interaction-trigger:visible").first();
    await block.click();
    for (const label of ["Insights", "Quiz", "Critical thinking", "Summary", "Note"])
      assert.equal(await page.getByRole("menuitem", { name: new RegExp(`^${label}(?:\\s|$)`) }).count(), 1);
    await page.getByRole("menuitem", { name: "Note", exact: true }).click();
    const note = page.getByRole("region", { name: "Note", exact: true });
    await note.getByLabel("Note", { exact: true }).waitFor();
    await note.getByRole("button", { name: "Close", exact: true }).click();
    if (name === "Madame Bovary") {
      // Hold a large-section request, navigate away, then allow its late result.
      let release;
      const gate = new Promise(resolve => { release = resolve; });
      await page.route("**/task-units/content", async route => { await gate; await route.continue().catch(() => {}); }, { times: 1 });
      await buttons.nth(index).click();
      await buttons.first().click();
      release();
      await page.waitForFunction(expected => document.querySelectorAll("[data-measure-group]").length === expected,
        numbers[0], { timeout: 15000 });
      assert.equal(await buttons.first().getAttribute("aria-selected"), "true");
    }
    assert.deepEqual(errors, []);
    assert.deepEqual(mutations, [], "Reading must not prepare or generate");
    await page.screenshot({ path: path.join(output, `content-${name === "Madame Bovary" ? "madame" : "red"}-${width}.png`) });
    console.log(JSON.stringify({ book: name, width, milliseconds, batches: requests.length, passed: true }));
    await page.close();
  }
} finally { await browser.close(); }
