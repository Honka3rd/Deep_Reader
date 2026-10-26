import assert from "node:assert/strict";

export async function cleanupMenu(page, row) {
  await row.locator(".cleanup-history-record-menu-trigger").click();
  const menu = page.locator(".cleanup-history-record-menu").getByRole("menu");
  await menu.waitFor();
  assert.deepEqual(await menu.getByRole("menuitem").allTextContents(), ["Edit", "Delete", "Keep it"]);
  return menu;
}

export async function cleanupAction(page, row, name) {
  const menu = await cleanupMenu(page, row);
  await menu.getByRole("menuitem", { name, exact: true }).click();
  await menu.waitFor({ state: "hidden" });
}

export async function cleanupDisabled(page, row, name) {
  const menu = await cleanupMenu(page, row);
  const disabled = await menu.getByRole("menuitem", { name, exact: true }).isDisabled();
  await page.keyboard.press("Escape");
  await menu.waitFor({ state: "hidden" });
  return disabled;
}
