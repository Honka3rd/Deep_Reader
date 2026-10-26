// Actual feature + DTO + fetch with controlled HTTP outcomes; no backend writes.
import assert from "node:assert/strict";
import { mkdir } from "node:fs/promises";

const { chromium } = await import(process.env.PLAYWRIGHT_MODULE || "playwright");
const output = process.env.SCREENSHOT_DIR || "/tmp/deep-reader-translation-recovery";
await mkdir(output, { recursive: true });
const browser = await chromium.launch({ headless: true, channel: process.env.PLAYWRIGHT_CHANNEL || "chrome" });
try {
  for (const width of [1280, 390]) {
    const page = await browser.newPage({ viewport: { width, height: 800 } });
    const errors = [], writes = [], records = [];
    let outcome = "provider", failRead = false;
    page.on("pageerror", error => errors.push(error.message));
    const target = { doc_name: "book", content_block_id: "block", target_id: "block",
      target_type: "content_block", document_id: "book" };
    const snapshot = () => ({ target, language_codes: ["en", "de", "fr"], default_source_language: "en",
      effective_content_hash: "hash", reading_revision: 1, translations: [...records] });
    await page.route("**/api/**", async route => {
      const path = new URL(route.request().url()).pathname;
      if (path.endsWith("/translate/read")) {
        if (failRead) { failRead = false; return route.fulfill({ status: 503, json: { detail: "unavailable" } }); }
        return route.fulfill({ json: snapshot() });
      }
      if (!path.endsWith("/translate/generate")) return route.fulfill({ json: { documents: [] } });
      const command = route.request().postDataJSON();
      writes.push(command);
      if (outcome === "provider") return route.fulfill({ status: 502, json: { detail: "translation_provider_failed" } });
      if (outcome === "unknown") return route.fulfill({ status: 504, body: "Gateway timeout" });
      const record = { source_language: command.source_language, target_language: command.target_language,
        status: "completed", translated_text: "Saved translation", effective_content_hash: "hash",
        mutation_id: command.mutation_id, revision: 1,
        generated_at: "2026-10-26T12:00:00Z", updated_at: "2026-10-26T12:00:00Z" };
      records.push(record);
      return route.fulfill({ json: { envelope: { target, status: "completed", persisted: true,
        effective_content_hash: "hash", reading_revision: 1 },
        payload: { source_language: record.source_language, target_language: record.target_language,
          translated_text: record.translated_text }, record } });
    });
    await page.goto(process.env.UI_URL || "http://127.0.0.1:5173");
    await page.evaluate(async () => {
      const { default: React } = await import("/node_modules/.vite/deps/react.js");
      const { default: ReactDOMClient } = await import("/node_modules/.vite/deps/react-dom_client.js");
      const { useBlockTranslation } = await import("/src/features/block-translation/controller.ts");
      const { BlockTranslationView } = await import("/src/features/block-translation/view.tsx");
      const blocks = [{ block_id: "block", content: "Original text", ownerContext: { sourceHash: "source" } }];
      const t = window.translationRecovery = {};
      const container = document.createElement("div"); document.body.append(container);
      const root = ReactDOMClient.createRoot(container);
      function Probe() {
        t.c = useBlockTranslation({ docName: "book", scopeKey: "book", blocks });
        return React.createElement(BlockTranslationView, { controller: t.c });
      }
      root.render(React.createElement(React.StrictMode, null, React.createElement(Probe)));
      t.open = () => t.c.open(blocks[0], "translate");
      t.dispose = () => root.unmount();
    });
    await page.waitForFunction(() => Boolean(window.translationRecovery.c));
    const ready = () => page.waitForFunction(() => window.translationRecovery.c.status === "ready");
    const choose = async (direction, language) => {
      const input = page.getByRole("combobox", { name: `Translate ${direction}`, exact: true });
      assert.equal(await input.isEnabled(), true, "Failed commands must not lock language selectors");
      if (await input.inputValue() === language) return;
      await input.fill(language);
      await page.getByRole("option", { name: language, exact: true }).click();
    };
    const submit = () => page.locator(".translation-submit").click();
    const recover = async () => {
      await page.getByRole("button", { name: "Try again", exact: true }).click();
      await ready();
    };
    await page.evaluate(() => window.translationRecovery.open());
    await ready();
    const toInput = page.getByRole("combobox", { name: "Translate to", exact: true });
    await toInput.click();
    assert.equal(await page.getByRole("option", { name: "English (en)", exact: true }).count(), 0);
    await page.keyboard.press("Escape");
    await choose("to", "German (de)");
    await choose("from", "German (de)");
    await page.waitForFunction(() => window.translationRecovery.c.from === "de" && window.translationRecovery.c.to === null
      && document.querySelector("#translation-to").value === "");
    assert.equal(await toInput.inputValue(), "", "Changing From clears an equal To");
    assert.equal(await page.locator(".translation-submit").isDisabled(), true);
    await toInput.click();
    assert.equal(await page.getByRole("option", { name: "German (de)", exact: true }).count(), 0);
    assert.equal(await page.getByRole("option", { name: "English (en)", exact: true }).count(), 1);
    await page.screenshot({ path: `${output}/${width}-distinct-language-options.png` });
    await page.keyboard.press("Escape");
    await choose("from", "English (en)");
    await choose("to", "German (de)");
    await submit();
    await page.waitForFunction(() => window.translationRecovery.c.status === "failed");
    assert.match(await page.getByRole("alert").innerText(), /failed to return a usable result/);
    await recover();
    assert.equal(writes.length, 1, "Recovery only reads saved translations");

    outcome = "unknown";
    await choose("to", "German (de)");
    await submit();
    await page.waitForFunction(() => window.translationRecovery.c.status === "failed");
    assert.notEqual(writes[0].mutation_id, writes[1].mutation_id, "A definite rejection releases the old command");
    const uncertainId = writes[1].mutation_id;
    failRead = true;
    await page.getByRole("button", { name: "Try again", exact: true }).click();
    await page.locator(".translation-language-dialog .translation-failed").waitFor();
    assert.equal(await page.getByRole("combobox", { name: "Translate to" }).isDisabled(), true);
    await page.getByRole("button", { name: "Retry", exact: true }).click();
    await ready();
    assert.equal(writes.length, 2);
    await choose("from", "German (de)");
    await choose("to", "French (fr)");
    await page.screenshot({ path: `${output}/${width}-editable-after-failure.png` });
    outcome = "success";
    await submit();
    await page.locator('.translation-saved-section-active[data-language-code="fr"]').waitFor();
    assert.equal(writes.length, 3);
    assert.notEqual(writes[2].mutation_id, uncertainId, "Different language commands get separate identities");

    await page.getByRole("button", { name: "Add translation" }).click();
    await ready();
    await choose("to", "French (fr)");
    await submit();
    await page.locator('.translation-saved-section-active[data-language-code="fr"]').waitFor();
    assert.equal(writes.length, 3, "Saved translations remain accessible while another command is unconfirmed");
    await page.getByRole("button", { name: "Add translation" }).click();
    await ready();
    await page.getByRole("button", { name: "Cancel", exact: true }).click();
    await page.evaluate(() => window.translationRecovery.open());
    await ready();
    await choose("from", "English (en)");
    await choose("to", "German (de)");
    await submit();
    await page.locator('.translation-saved-section-active[data-language-code="de"]').waitFor();
    assert.equal(writes.length, 4);
    assert.equal(writes[3].mutation_id, uncertainId, "Returning to the uncertain command retains its retry identity");
    assert.equal(await page.locator(".translation-saved-section").count(), 2);
    await page.waitForFunction(() => {
      const drawer = document.querySelector('.translation-drawer [role="dialog"]');
      return drawer && getComputedStyle(drawer).transform === "none";
    });
    await page.screenshot({ path: `${output}/${width}-recovered.png` });
    await page.evaluate(() => window.translationRecovery.dispose());
    assert.deepEqual(errors, []);
    await page.close();
    console.log(`PASS ${width}: provider failure/retry, unknown outcome, read failure, editable languages, independent commands, saved reuse, cancel/reopen, original retry identity`);
  }
} finally { await browser.close(); }
