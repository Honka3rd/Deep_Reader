// Real App/API/LLM/SQL on immutable book excerpts. No production data writes.
import assert from "node:assert/strict";
import { writeFile } from "node:fs/promises";
const { chromium } = await import(process.env.PLAYWRIGHT_MODULE || "playwright");
const api = process.env.TRANSLATION_API, evidence = process.env.TRANSLATION_EVIDENCE;
assert.ok(api && evidence);
const browser = await chromium.launch({ headless: true, channel: process.env.PLAYWRIGHT_CHANNEL || "chrome" });
const reports = [];
const call = async (path, body) => {
  const response = await fetch(api + path, body ? { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) } : {});
  assert.ok(response.ok, `${path}: ${response.status}`);
  return response.json();
};
try {
  for (const [width, caseId] of [[1280, "fa794acef055f235"], [390, "085747b65094d52f"]]) {
    const fixture = await call("/fixture", { case: caseId });
    const page = await browser.newPage({ viewport: { width, height: 900 } });
    page.setDefaultTimeout(120000);
    const errors = [], requests = [];
    let holdGenerate = false, release;
    page.on("pageerror", error => errors.push(error.message));
    await page.route("**/api/**", async route => {
      const request = route.request(), url = new URL(request.url()), path = url.pathname.replace(/^\/api/, "");
      if (["/documents", "/documents/management"].includes(path)) return route.fulfill({ json: { items: [], total: 0 } });
      if (path.endsWith("/reparse-structure/status")) return route.fulfill({ json: { doc_name: "book", status: "idle" } });
      requests.push({ path, method: request.method(), body: request.postDataJSON() });
      if (path.endsWith("/translate/generate") && holdGenerate) await new Promise(resolve => { release = resolve; });
      try { const response = await route.fetch({ url: api + path + url.search, timeout: 0 }); await route.fulfill({ response }); }
      catch { await route.abort().catch(() => {}); }
    });
    const dialog = page.locator('[aria-labelledby="translation-language-title"]');
    const drawer = page.locator('[aria-labelledby="translation-saved-title"]');
    const block = page.locator(`.reader-content-block[data-block-id="${fixture.block_id}"]`);
    const menu = async name => {
      await block.locator('.reader-content-block-interaction-trigger').click();
      await page.getByRole("menuitem", { name, exact: true }).click();
    };
    const choose = async (label, code) => {
      await dialog.getByRole("combobox", { name: label, exact: true }).fill(code);
      await page.getByRole("option").filter({ hasText: `(${code})` }).click();
    };
    const navigate = async () => {
      await page.goto(`${process.env.UI_URL || "http://127.0.0.1:5173"}/documents/book`);
      await page.locator(".hierarchy-section-button").filter({ hasText: "Excerpt 1" }).click();
      await block.waitFor();
    };
    try {
      await navigate();
      await menu("Translate");
      await dialog.getByRole("combobox", { name: "Translate from", exact: true }).waitFor();
      await page.waitForFunction(() => !document.querySelector("#translation-from")?.disabled);
      assert.match(await dialog.getByRole("combobox", { name: "Translate from", exact: true }).inputValue(), new RegExp(`\\(${fixture.language}\\)`));
      assert.equal(await dialog.getByRole("button", { name: "Translate", exact: true }).isDisabled(), true);
      await dialog.getByRole("combobox", { name: "Translate to", exact: true }).fill("invented-language");
      assert.equal(await dialog.getByRole("button", { name: "Translate", exact: true }).isDisabled(), true);
      await page.keyboard.press("Escape");
      await choose("Translate to", "fr");
      await page.screenshot({ path: `${evidence}/languages-${width}.png`, animations: "disabled" });
      holdGenerate = true;
      await dialog.getByRole("button", { name: "Translate", exact: true }).dblclick();
      await dialog.waitFor({ state: "hidden" });
      assert.equal(await drawer.isVisible(), false);
      while (!release) await page.waitForTimeout(20);
      assert.equal(requests.filter(r => r.path.endsWith("/translate/generate")).length, 1);
      holdGenerate = false; release(); release = undefined;
      await drawer.waitFor();
      await drawer.locator('[data-language-code="fr"]').waitFor();
      await drawer.getByRole("button", { name: "Add translation", exact: true }).click();
      await page.waitForFunction(() => !document.querySelector("#translation-to")?.disabled);
      await choose("Translate to", "de");
      await dialog.getByRole("button", { name: "Translate", exact: true }).click();
      await drawer.locator('[data-language-code="de"]').waitFor();
      assert.equal(await drawer.locator(".translation-saved-section").count(), 2);
      await page.screenshot({ path: `${evidence}/translations-${width}.png`, animations: "disabled" });
      assert.equal(await drawer.evaluate(el => el.scrollWidth > el.clientWidth), false);
      const target = { doc_name: "book", content_block_id: fixture.block_id };
      const initial = await call("/documents/reading-interactions/translate/read", target);
      assert.equal(initial.translations.length, 2);
      await drawer.getByRole("button", { name: "Add translation", exact: true }).click();
      await page.waitForFunction(() => !document.querySelector("#translation-to")?.disabled);
      await choose("Translate to", "fr");
      await dialog.getByRole("button", { name: "Translate", exact: true }).click();
      await drawer.locator('[data-language-code="fr"]').waitFor();
      const reused = await call("/documents/reading-interactions/translate/read", target);
      assert.deepEqual(reused.translations, initial.translations);
      assert.equal(requests.filter(r => r.path.endsWith("/translate/generate")).length, 2);
      const directReuse = await call("/documents/reading-interactions/translate/generate", {
        ...target, target_language: "fr", source_language: "de", mutation_id: "new-request-existing-language",
      });
      assert.deepEqual(directReuse.record, initial.translations.find(r => r.target_language === "fr"));
      if (width < 500) assert.equal(await drawer.locator(".translation-saved-sections").evaluate(el => el.scrollHeight > el.clientHeight), true);
      const generateCount = requests.filter(r => r.path.endsWith("/translate/generate")).length;
      await navigate();
      await menu("View translations");
      await drawer.locator('[data-language-code="de"]').waitFor();
      assert.equal(await drawer.locator(".translation-saved-section").count(), 2);
      assert.equal(requests.filter(r => r.path.endsWith("/translate/generate")).length, generateCount);
      await drawer.getByRole("button", { name: "Close translations", exact: true }).click();
      await menu("Translate");
      await page.keyboard.press("Escape");
      await dialog.waitFor({ state: "hidden" });
      assert.equal(requests.filter(r => r.path.endsWith("/translate/generate")).length, generateCount);
      const editor = page.locator('[aria-labelledby="reading-content-dialog-title"]');
      const confirm = page.locator('[aria-labelledby="reading-content-interactions-title"]');
      const save = async changed => {
        await menu("Edit");
        const input = editor.getByRole("textbox", { name: "Block text", exact: true });
        await input.waitFor();
        await page.waitForFunction(() => !document.querySelector('textarea[aria-invalid]')?.disabled);
        if (changed) await input.fill((await input.inputValue()) + " Updated.");
        await editor.getByRole("button", { name: "Save", exact: true }).click();
        await confirm.getByRole("button", { name: "Keep and save", exact: true }).click();
        await editor.waitFor({ state: "hidden" });
      };
      const layoutCount = () => requests.filter(r => r.path.endsWith("/task-layout")).length;
      const beforeLayout = layoutCount(), beforeStats = await call("/stats");
      await save(false);
      assert.equal((await call("/stats")).revision, beforeStats.revision);
      assert.equal((await call("/documents/reading-interactions/translate/read", target)).translations.length, 2);
      await save(true);
      assert.equal((await call("/documents/reading-interactions/translate/read", target)).translations.length, 0);
      assert.equal(layoutCount(), beforeLayout, "Body edit must refresh only affected blocks");
      // A late model response must not reopen the drawer after section navigation.
      await menu("Translate");
      await page.waitForFunction(() => !document.querySelector("#translation-to")?.disabled);
      await choose("Translate to", "fr");
      holdGenerate = true;
      await dialog.getByRole("button", { name: "Translate", exact: true }).click();
      while (!release) await page.waitForTimeout(20);
      await page.locator(".hierarchy-section-button").filter({ hasText: "Excerpt 2" }).click();
      holdGenerate = false; release(); release = undefined;
      await page.waitForTimeout(1000);
      assert.equal(await drawer.isVisible(), false);
      assert.deepEqual(errors, []);
      reports.push({ width, case: caseId, status: "passed", modelSubmissions: requests.filter(r => r.path.endsWith("/translate/generate")).length });
      console.log(`PASS ${width}: registry/defaults, two persisted languages/reload, no-op/edit Keep, bounded refresh, stale-response guard`);
    } catch (error) {
      await page.screenshot({ path: `${evidence}/failure-${width}.png` });
      throw error;
    } finally { release?.(); await page.close(); }
  }
  await writeFile(`${evidence}/browser-report.json`, JSON.stringify(reports, null, 2));
} finally { await browser.close(); }
