// Mounted editor lifecycle, Unicode limits and destructive consent. Controlled callbacks.
import assert from "node:assert/strict";
const { chromium } = await import(process.env.PLAYWRIGHT_MODULE || "playwright");
const browser = await chromium.launch({ headless: true, channel: process.env.PLAYWRIGHT_CHANNEL || "chrome" });
try {
  for (const width of [1280, 390]) {
    const page = await browser.newPage({ viewport: { width, height: 900 } });
    const errors = [];
    page.on("pageerror", error => errors.push(error.message));
    await page.route("**/api/**", route => route.fulfill({ json: { documents: [] } }));
    await page.goto(process.env.UI_URL || "http://127.0.0.1:5173");
    await page.evaluate(async () => {
      const { default: React } = await import("/node_modules/.vite/deps/react.js");
      const { default: ReactDOMClient } = await import("/node_modules/.vite/deps/react-dom_client.js");
      const { default: ReactDOM } = await import("/node_modules/.vite/deps/react-dom.js");
      const { SourceReviewBlockEditor } = await import("/src/features/reader-content/SourceReviewBlockRepair.tsx");
      const container = document.createElement("div"); document.body.append(container);
      const root = ReactDOMClient.createRoot(container);
      const test = window.regionTest = { commands: [], dirty: false, closed: 0, disposed: false, lateDirty: 0 };
      const onDirty = value => { if (test.disposed) test.lateDirty++; test.dirty = value; };
      const block = { block_id: "u:content:0", content: "Original block", review_repair: {
        can_edit: true, can_keep: true, can_delete: true } };
      test.render = (id = "u:content:0", mode = "edit") => ReactDOM.flushSync(() => root.render(React.createElement(SourceReviewBlockEditor, {
        key: id, block: { ...block, block_id: id }, regionId: "r", mode, busy: false, error: "", onDirty,
        onClose: () => { test.closed++; }, onMutate: (action, command) => {
          test.commands.push({ action, ...command }); return new Promise(resolve => { test.release = resolve; });
        },
      })));
      test.dispose = () => { root.unmount(); test.disposed = true; };
      test.render();
    });
    const editor = page.locator('[aria-labelledby="source-review-block-editor-title"]');
    const field = editor.getByRole("textbox", { name: "Block text", exact: true });
    await field.fill("\u{1f600}".repeat(10000));
    assert.equal(await editor.getByRole("button", { name: "Save", exact: true }).isEnabled(), true);
    await field.fill("x".repeat(10001));
    assert.equal(await editor.getByRole("button", { name: "Save", exact: true }).isDisabled(), true);
    await field.fill("  ");
    assert.equal(await editor.getByRole("button", { name: "Save", exact: true }).isDisabled(), true);
    await field.fill("Draft");
    page.once("dialog", dialog => dialog.dismiss());
    await editor.getByRole("button", { name: "Cancel", exact: true }).click();
    assert.equal(await page.evaluate(() => window.regionTest.closed), 0);
    await editor.getByRole("button", { name: "Save", exact: true }).click();
    await editor.getByRole("button", { name: "Keep and save", exact: true }).evaluate(el => { el.click(); el.click(); });
    assert.equal(await page.evaluate(() => window.regionTest.commands.length), 1);
    assert.deepEqual(await page.evaluate(() => window.regionTest.commands.at(-1)), {
      action: "review_block_edit", region_id: "r", block_id: "u:content:0", text: "Draft", discard_interactions: false });
    // Change owner while its callback is pending; old completion must not close the new editor.
    await page.evaluate(() => window.regionTest.render("other"));
    await page.evaluate(() => window.regionTest.release(true));
    assert.equal(await field.inputValue(), "Original block");
    assert.equal(await page.evaluate(() => window.regionTest.closed), 0);
    await field.fill("Other draft");
    await editor.getByRole("button", { name: "Save", exact: true }).click();
    await editor.getByRole("button", { name: "Keep and save", exact: true }).click();
    await page.evaluate(() => window.regionTest.dispose());
    await page.evaluate(() => window.regionTest.release(true));
    assert.equal(await page.evaluate(() => window.regionTest.lateDirty), 0);
    assert.deepEqual(errors, []);
    await page.close();
    console.log(`PASS ${width}: limits, draft protection, duplicate guard, consent, target switch and unmount`);
  }
} finally { await browser.close(); }
