// Shared region/history presentation: keyboard, protection and owner lifecycle.
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
      const { SourceReviewBlockMenu } = await import("/src/features/reader-content/SourceReviewBlockRepair.tsx");
      const { SourceReviewNavigateButton } = await import("/src/features/reader-content/SourceReviewControls.tsx");
      const container = document.createElement("div"); document.body.append(container);
      const root = ReactDOMClient.createRoot(container);
      const test = window.controlsTest = { commands: [], navigated: 0 };
      test.render = (id = "b", disabled = false, protectedBlock = false) => ReactDOM.flushSync(() => root.render(
        React.createElement("section", { className: "review-controls-test" },
          React.createElement(SourceReviewBlockMenu, { block: { block_id: id, review_repair: {
            can_edit: !protectedBlock, can_delete: !protectedBlock, can_keep: !protectedBlock,
            blocked_reason: protectedBlock ? "block_manually_edited" : null } }, disabled,
            onSelect: (action, trigger) => test.commands.push({ action, trigger: trigger.getAttribute("aria-label") }) }),
          React.createElement(SourceReviewNavigateButton, { blockId: id, disabled, onClick: () => test.navigated++ }))));
      test.dispose = () => root.unmount();
      test.render();
    });
    const trigger = page.getByRole("button", { name: "Text actions for block b", exact: true });
    const menu = page.locator(".source-review-block-menu").getByRole("menu");
    await trigger.focus();
    await page.keyboard.press("Enter");
    await menu.waitFor();
    assert.deepEqual(await menu.getByRole("menuitem").allTextContents(), ["Edit", "Delete", "Keep it"]);
    assert.equal(await menu.locator('[data-testid="CheckIcon"]').count(), 1);
    await page.screenshot({ path: `/tmp/source-review-controls-${width}.png`, animations: "disabled" });
    await page.keyboard.press("Escape");
    await menu.waitFor({ state: "hidden" });
    assert.equal(await trigger.evaluate(el => el === document.activeElement), true);
    for (const name of ["Edit", "Delete", "Keep it"]) {
      await trigger.click();
      await menu.getByRole("menuitem", { name, exact: true }).click();
      await menu.waitFor({ state: "hidden" });
    }
    assert.deepEqual(await page.evaluate(() => window.controlsTest.commands), ["edit", "delete", "keep"].map(action => ({ action, trigger: "Text actions for block b" })));
    const navigation = page.getByRole("button", { name: "Go to content for block b", exact: true });
    assert.equal(await navigation.locator('[data-testid="ArrowForwardIcon"]').count(), 1);
    await navigation.click();
    assert.equal(await page.evaluate(() => window.controlsTest.navigated), 1);
    await page.evaluate(() => window.controlsTest.render("b", false, true));
    await trigger.click();
    for (const item of await menu.getByRole("menuitem").all()) assert.equal(await item.isDisabled(), true);
    await page.evaluate(() => window.controlsTest.render("b", true));
    await menu.waitFor({ state: "hidden" });
    assert.equal(await trigger.isDisabled(), true);
    await page.evaluate(() => window.controlsTest.render());
    await trigger.click();
    await page.evaluate(() => window.controlsTest.render("other"));
    await menu.waitFor({ state: "hidden" });
    await page.getByRole("button", { name: "Text actions for block other" }).click();
    await page.evaluate(() => window.controlsTest.dispose());
    assert.equal(await page.locator(".source-review-item-menu").count(), 0);
    assert.deepEqual(errors, []);
    console.log(`PASS ${width}: shared menu/actions/icons, keyboard focus, protection, identity change and unmount`);
    await page.close();
  }
} finally { await browser.close(); }
