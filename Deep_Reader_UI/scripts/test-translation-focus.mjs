// Actual feature controller/view with a controlled service; no production writes or LLM calls.
import assert from "node:assert/strict";
import { mkdir } from "node:fs/promises";

const { chromium } = await import(process.env.PLAYWRIGHT_MODULE || "playwright");
const output = process.env.SCREENSHOT_DIR || "/tmp/deep-reader-translation-focus";
await mkdir(output, { recursive: true });
const browser = await chromium.launch({ headless: true, channel: process.env.PLAYWRIGHT_CHANNEL || "chrome" });
try {
  for (const width of [1280, 390]) {
    const page = await browser.newPage({ viewport: { width, height: 800 } });
    const errors = [];
    page.on("pageerror", error => errors.push(error.message));
    await page.route("**/api/**", route => route.fulfill({ json: { documents: [] } }));
    await page.goto(process.env.UI_URL || "http://127.0.0.1:5173");
    await page.evaluate(async () => {
      const { default: React } = await import("/node_modules/.vite/deps/react.js");
      const { default: ReactDOMClient } = await import("/node_modules/.vite/deps/react-dom_client.js");
      const { useBlockTranslation } = await import("/src/features/block-translation/controller.ts");
      const { BlockTranslationView } = await import("/src/features/block-translation/view.tsx");
      const { savedSections } = await import("/src/features/block-translation/model.ts");
      const t = window.translationFocusTest = { writes: 0, reads: 0 };
      const block = { block_id: "focus:content:0", content: "Original text", ownerContext: { sourceHash: "source" } };
      const blocks = [block];
      const record = code => ({ source_language: "en", target_language: code, status: "completed",
        translated_text: (`${code}: A long saved translation for checking drawer scrolling and language selection.\n\n`).repeat(80),
        effective_content_hash: "hash", generated_at: "2026-10-26T12:00:00Z", updated_at: "2026-10-26T12:00:00Z",
        revision: 1, mutation_id: `saved-${code}` });
      const records = [record("fr"), record("de")];
      t.remove = code => records.splice(records.findIndex(item => item.target_language === code), 1);
      const codes = items => items.map(item => item.target_language).join(",");
      t.modelChecks = [codes(savedSections(records)) === "de,fr",
        codes(savedSections(records, "fr")) === "fr,de", codes(records) === "fr,de",
        codes(savedSections(records, "missing")) === "de,fr", savedSections([], "fr").length === 0];
      const response = target => ({ target: { ...target, document_id: "book", target_id: target.content_block_id,
        target_type: "content_block" }, language_codes: ["en", "fr", "de", "es"], default_source_language: "en",
        effective_content_hash: "hash", reading_revision: 0, translations: [...records] });
      const service = {
        read: async target => {
          t.reads += 1;
          if (t.failRead) { t.failRead = false; throw new Error("Read failed"); }
          if (t.holdRead) {
            t.holdRead = false;
            const snapshot = response(target);
            return new Promise(resolve => { t.finishRead = () => resolve(snapshot); });
          }
          return response(target);
        },
        generate: async command => {
          t.writes += 1;
          const saved = record(command.target_language);
          records.push(saved);
          return { envelope: { target: response(command).target, status: "completed", persisted: true,
            reading_revision: 0, effective_content_hash: "hash" }, record: saved };
        },
      };
      const container = document.createElement("div"); document.body.append(container);
      const root = ReactDOMClient.createRoot(container);
      function Probe() {
        t.c = useBlockTranslation({ docName: "book", blocks, scopeKey: "book", service });
        return React.createElement(BlockTranslationView, { controller: t.c });
      }
      root.render(React.createElement(React.StrictMode, null, React.createElement(Probe)));
      t.open = () => t.c.open(block, "translate");
      t.view = () => t.c.open(block, "view");
      t.dispose = () => root.unmount();
    });
    await page.waitForFunction(() => Boolean(window.translationFocusTest.c));
    assert.ok(await page.evaluate(() => window.translationFocusTest.modelChecks.every(Boolean)));
    const choose = async label => {
      await page.getByRole("combobox", { name: "Translate to" }).fill(label);
      await page.getByRole("option", { name: label, exact: true }).click();
      await page.locator(".translation-submit").click();
    };
    const verify = async (code, writes) => {
      await page.waitForFunction(code => {
        const first = document.querySelector(".translation-saved-section");
        return first?.getAttribute("data-language-code") === code && first.getAttribute("aria-current") === "true";
      }, code);
      await page.getByRole("dialog", { name: "Translations", exact: true }).waitFor();
      await page.getByRole("dialog", { name: "Translate block", exact: true }).waitFor({ state: "hidden" });
      await page.waitForFunction(() => {
        const drawer = document.querySelector('.translation-drawer [role="dialog"]');
        return drawer && getComputedStyle(drawer).transform === "none";
      });
      const result = await page.locator(".translation-saved-sections").evaluate(container => {
        const first = container.querySelector(".translation-saved-section");
        const bounds = container.getBoundingClientRect(), item = first.getBoundingClientRect();
        return { scrollTop: container.scrollTop, topVisible: item.top >= bounds.top && item.top < bounds.bottom,
          overflow: container.scrollWidth > container.clientWidth,
          background: getComputedStyle(first).backgroundColor,
          active: container.querySelectorAll(".translation-saved-section-active").length };
      });
      assert.equal(result.scrollTop, 0);
      assert.equal(result.topVisible, true);
      assert.equal(result.overflow, false);
      assert.equal(result.active, 1);
      assert.notEqual(result.background, "rgba(0, 0, 0, 0)");
      assert.equal(await page.evaluate(() => window.translationFocusTest.writes), writes);
      await page.screenshot({ path: `${output}/${width}-${code}.png` });
    };
    await page.evaluate(() => window.translationFocusTest.open());
    await choose("French (fr)");
    await verify("fr", 0);
    await page.getByRole("button", { name: "Reload translations", exact: true }).click();
    await page.waitForFunction(() => window.translationFocusTest.c.status === "ready");
    assert.equal(await page.locator(".translation-saved-section-active").getAttribute("data-language-code", { timeout: 2000 }), "fr",
      "Reload must retain the selected translation highlight");
    await verify("fr", 0);
    assert.equal(await page.evaluate(() => window.translationFocusTest.reads), 2, "Reload issues one read");
    await page.evaluate(() => { window.translationFocusTest.failRead = true; });
    await page.getByRole("button", { name: "Reload translations", exact: true }).click();
    await page.waitForFunction(() => window.translationFocusTest.c.status === "failed");
    await page.getByRole("button", { name: "Retry", exact: true }).click();
    await verify("fr", 0);
    await page.locator(".translation-saved-sections").evaluate(el => { el.scrollTop = el.scrollHeight; });
    await page.getByRole("button", { name: "Add translation" }).click();
    await choose("German (de)");
    await verify("de", 0);
    await page.getByRole("button", { name: "Add translation" }).click();
    await choose("Spanish (es)");
    await verify("es", 1);
    assert.equal(await page.locator(".translation-saved-section").count(), 3);
    await page.getByRole("button", { name: "Reload translations", exact: true }).focus();
    await page.keyboard.press("Enter");
    await verify("es", 1);
    assert.equal(await page.locator(".translation-saved-section").count(), 3);
    assert.equal(await page.evaluate(() => document.activeElement?.getAttribute("data-language-code")), "es");
    await page.evaluate(() => window.translationFocusTest.remove("es"));
    await page.getByRole("button", { name: "Reload translations", exact: true }).click();
    await page.waitForFunction(() => window.translationFocusTest.c.status === "ready");
    assert.equal(await page.locator(".translation-saved-section-active").count(), 0);
    assert.equal(await page.evaluate(() => window.translationFocusTest.c.focusLanguage), null,
      "A removed translation must not retain stale focus");
    await page.getByRole("button", { name: "Add translation" }).click();
    await choose("French (fr)");
    await verify("fr", 1);
    await page.evaluate(() => { window.translationFocusTest.holdRead = true; });
    await page.getByRole("button", { name: "Reload translations", exact: true }).click();
    await page.waitForFunction(() => window.translationFocusTest.c.status === "loading");
    assert.equal(await page.getByRole("button", { name: "Reload translations", exact: true }).isDisabled(), true);
    await page.keyboard.press("Escape");
    await page.getByRole("dialog", { name: "Translations", exact: true }).waitFor({ state: "hidden" });
    await page.evaluate(() => window.translationFocusTest.view());
    await page.waitForFunction(() => window.translationFocusTest.c.status === "ready");
    await page.evaluate(async () => { window.translationFocusTest.finishRead(); await new Promise(resolve => setTimeout(resolve, 0)); });
    assert.equal(await page.locator(".translation-saved-section-active").count(), 0);
    assert.equal(await page.locator(".translation-saved-section").first().getAttribute("data-language-code"), "de");
    await page.evaluate(() => window.translationFocusTest.dispose());
    assert.deepEqual(errors, []);
    await page.close();
    console.log(`PASS ${width}: cached/new target highlight, reload/retry, removed language, keyboard focus, long-text scroll, sibling retention, close/reopen stale-read guard`);
  }
} finally { await browser.close(); }
