// Mount the production editor against the backup-book disposable API, never Library storage.
import assert from "node:assert/strict";
const api = process.env.READING_TEST_API;
assert.ok(api?.startsWith("http://127.0.0.1:"));
const { chromium } = await import(process.env.PLAYWRIGHT_MODULE || "playwright");
const browser = await chromium.launch({ headless: true,
  ...(process.env.CHROMIUM_PATH ? { executablePath: process.env.CHROMIUM_PATH } : {}) });
try {
  for (const width of [1280, 390]) {
    const page = await browser.newPage({ viewport: { width, height: 900 } });
    const errors = [];
    page.on("pageerror", e => { errors.push(e.message); console.error(e.message); });
    let failNext = false;
    await page.route("**/api/**", async route => {
      const url = new URL(route.request().url());
      if (!url.pathname.includes("/reading-content") && !url.pathname.includes("/source-review/"))
        return route.fulfill({ json: { items: [], total: 0 } });
      if (failNext && route.request().method() === "POST") { failNext = false; return route.abort(); }
      await route.fulfill({ response: await route.fetch({ url: api + url.pathname.replace(/^\/api/, "") + url.search }) });
    });
    await page.goto(process.env.UI_URL || "http://127.0.0.1:5173");
    const initial = await (await page.request.get(api + "/documents/book/reading-content?include_blocks=true")).json();
    const blockId = initial.blocks[0].block_id;
    await page.evaluate(async blockId => {
      const source = await (await fetch("/src/features/reader-content/ReadingContentActions.tsx")).text();
      const reactUrl = source.match(/from "([^\"]+\/react\.js\?[^\"]+)"/)[1];
      const { default: React } = await import(reactUrl);
      const entry = await (await fetch("/src/main.tsx")).text();
      const domUrl = entry.match(/from "([^\"]+\/react-dom_client\.js\?[^\"]+)"/)[1];
      const { default: ReactDOM } = await import(domUrl);
      const { ReadingContentActions } = await import("/src/features/reader-content/ReadingContentActions.tsx");
      const host = document.createElement("div");
      document.body.replaceChildren(host);
      function Harness() {
        const [selection, setSelection] = React.useState(null);
        window.selectBlock = action => setSelection({ blockId, action });
        return React.createElement(ReadingContentActions, { docName: "book", selection,
          onCloseSelection: () => setSelection(null), onChanged: () => {} });
      }
      ReactDOM.createRoot(host).render(React.createElement(Harness));
    }, blockId);
    await page.getByRole("button", { name: "Review source", exact: true }).waitFor();
    await page.evaluate(() => window.selectBlock("edit"));
    const field = page.getByLabel("Block text");
    await field.waitFor();
    await field.fill(`Reviewed text ${width}`);
    failNext = true;
    await page.getByRole("button", { name: "Save", exact: true }).click();
    await page.getByRole("alert").last().waitFor();
    assert.equal(await field.inputValue(), `Reviewed text ${width}`);
    await page.getByRole("button", { name: "Save", exact: true }).click();
    await page.getByRole("dialog").waitFor({ state: "hidden" });
    await page.evaluate(() => window.selectBlock("edit"));
    await field.waitFor();
    await field.fill("Uncommitted draft");
    page.once("dialog", dialog => dialog.dismiss());
    await page.getByRole("button", { name: "Cancel", exact: true }).click();
    assert.equal(await field.inputValue(), "Uncommitted draft");
    const concurrent = await (await page.request.get(api + "/documents/book/reading-content")).json();
    assert.equal((await page.request.post(api + "/documents/book/reading-content", { data: {
      action: "edit", block_id: blockId, text: "Other editor", source_token: concurrent.source_token,
      expected_revision: concurrent.revision, mutation_id: `concurrent-${width}-${Date.now()}`,
    } })).status(), 200);
    await page.getByRole("button", { name: "Save", exact: true }).click();
    await page.getByRole("alert").last().waitFor();
    assert.equal(await field.inputValue(), "Uncommitted draft");
    page.once("dialog", dialog => dialog.accept());
    await page.getByRole("button", { name: "Cancel", exact: true }).click();
    await page.getByRole("dialog").waitFor({ state: "hidden" });
    await page.evaluate(() => window.selectBlock("delete"));
    await page.getByText("Delete this block from the reading content?", { exact: false }).waitFor();
    await page.screenshot({ path: `/tmp/reading-delete-${width}.png`, animations: "disabled" });
    await page.getByRole("button", { name: "Delete block", exact: true }).click();
    await page.getByRole("dialog").waitFor({ state: "hidden" });
    let state = await (await page.request.get(api + "/documents/book/reading-content?include_blocks=true")).json();
    assert.equal(state.blocks[0].deleted, true);
    await page.getByRole("button", { name: "Undo last content change" }).click();
    await page.waitForResponse(r => r.request().method() === "POST" && r.url().includes("reading-content"));
    await page.getByRole("button", { name: "Review source", exact: true }).click();
    await page.getByRole("button", { name: "Original", exact: true }).first().click();
    await page.getByRole("img").waitFor();
    await page.waitForFunction(() => [...document.images].some(i => i.complete && i.naturalWidth > 0));
    await page.screenshot({ path: `/tmp/reading-review-${width}.png`, animations: "disabled" });
    const unresolved = page.getByRole("button", { name: "Ignore region", exact: true }).first();
    if (await unresolved.count()) {
      await unresolved.click();
      await page.getByRole("dialog").last().getByRole("button", { name: "Ignore region", exact: true }).click();
      await page.getByRole("dialog").last().getByText("Ignored", { exact: false }).waitFor();
    }
    await page.getByRole("button", { name: "Close content editor" }).click();
    // Undo restores the unresolved review state for the next viewport.
    await page.getByRole("button", { name: "Undo last content change" }).click();
    await page.waitForResponse(r => r.request().method() === "POST" && r.url().includes("reading-content"));
    state = await (await page.request.get(api + "/documents/book/reading-content?include_blocks=true")).json();
    assert.equal(state.blocks[0].deleted, false);
    assert.equal(state.disposition, "needs_review");
    assert.deepEqual(errors, []);
    console.log(`PASS ${width}: production editor, network failure retry, confirmation, delete/undo, real source preview, explicit review/undo`);
    await page.close();
  }
} finally { await browser.close(); }
