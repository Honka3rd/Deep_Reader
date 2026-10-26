// Actual App with fixture API reads; dragging must never trigger backend work or storage writes.
import assert from "node:assert/strict";
const { chromium } = await import(process.env.PLAYWRIGHT_MODULE || "playwright");
const browser = await chromium.launch({ headless: true });
try {
  const page = await browser.newPage({ viewport: { width: 1280, height: 900 } });
  const errors = [], unexpected = [], requests = [];
  page.on("pageerror", error => errors.push(error.message));
  await page.addInitScript(() => {
    window.storageWrites = [];
    const original = Storage.prototype.setItem;
    Storage.prototype.setItem = function(key, value) {
      window.storageWrites.push(key); return original.call(this, key, value);
    };
  });
  await page.route("**/api/**", route => {
    const path = new URL(route.request().url()).pathname;
    requests.push(path);
    if (path === "/api/documents/task-layout") return route.fulfill({ json: {
      document_id: "Book", title: "Madame_Bovary_261021_155934",
      chapters: Array.from({ length: 40 }, (_, i) => ({ chapter_id: `c${i}`,
        title: i === 1 ? "Unbroken".repeat(45) : `Chapter ${i + 1}: Brief Bibliography of Gustave Flaubert`,
        sections: [{ section_id: `s${i}`, title: "Body", task_units: [{ unit_id: `u${i}` }] }],
      })),
    } });
    if (path.endsWith("/reparse-structure/status")) return route.fulfill({ json: { doc_name: "Book", status: "idle" } });
    if (path.endsWith("/reading-content")) return route.fulfill({ json: {
      revision: 0, source_token: "fixture", disposition: "accepted", can_undo: false, max_chars: 10000, regions: [], blocks: [],
    } });
    unexpected.push(path);
    return route.fulfill({ status: 500, json: { detail: "Unexpected API" } });
  });
  await page.goto(`${process.env.UI_URL || "http://127.0.0.1:5173"}/documents/Book`);
  const pane = page.getByRole("complementary", { name: "Document hierarchy" });
  const divider = page.getByRole("separator", { name: "Resize document hierarchy" });
  const width = async () => (await pane.boundingBox()).width;
  await page.locator(".hierarchy-chapter-item").last().waitFor({ state: "attached" });
  await page.waitForTimeout(150);
  const readsBefore = requests.length;
  const storageBefore = await page.evaluate(() => window.storageWrites.length);
  const originalWidth = await width();
  const startDrag = async () => {
    const r = await divider.boundingBox();
    const point = { x: r.x + r.width / 2, y: r.y + 60 };
    await page.mouse.move(point.x, point.y); await page.mouse.down(); return point;
  };
  let point = await startDrag();
  await page.mouse.move(point.x + 210, point.y, { steps: 8 });
  await page.mouse.up();
  assert.ok(await width() > originalWidth + 190, "Dragging right expands the hierarchy");
  assert.equal(await page.locator(".reader-workspace-resizing").count(), 0);
  const releasedWidth = await width();
  await page.mouse.move(point.x + 300, point.y);
  assert.equal(await width(), releasedWidth, "Released pointer cannot keep resizing");
  await divider.focus(); await page.keyboard.press("ArrowLeft");
  assert.ok(Math.abs(await width() - releasedWidth + 16) < 2);
  await page.keyboard.press("Home"); assert.equal(Math.round(await width()), 300);
  await page.keyboard.press("End");
  assert.ok((await page.locator(".content-pane").boundingBox()).width >= 319, "Reading pane keeps usable width");
  await divider.dblclick(); assert.ok(Math.abs(await width() - originalWidth) < 2, "Double click resets default width");
  point = await startDrag();
  await page.mouse.move(point.x + 100, point.y);
  await divider.dispatchEvent("pointercancel", { pointerId: 1 });
  const cancelledWidth = await width();
  await page.mouse.move(point.x + 150, point.y); await page.mouse.up();
  assert.equal(await width(), cancelledWidth, "Cancelled drag leaves no stale active gesture");
  assert.equal(await page.locator(".reader-workspace-resizing").count(), 0);
  const overflow = await pane.evaluate(el => {
    const s = getComputedStyle(el); return [s.overflowX, s.overflowY];
  });
  assert.deepEqual(overflow, ["auto", "auto"]);
  const list = page.locator(".hierarchy-chapter-list");
  const scroll = await list.evaluate(el => {
    el.scrollTop = el.scrollHeight; el.scrollLeft = el.scrollWidth;
    return { top: el.scrollTop, left: el.scrollLeft };
  });
  assert.ok(scroll.top > 0, "Long hierarchy remains vertically scrollable");
  await list.evaluate(el => { el.scrollTop = 0; el.scrollLeft = 0; });
  const titleFits = await page.locator(".hierarchy-section-label").nth(1).evaluate(el => {
    const title = el.querySelector(".hierarchy-section-title");
    const range = document.createRange(); range.selectNodeContents(title);
    const bounds = el.getBoundingClientRect();
    return Array.from(range.getClientRects()).every(r => r.left >= bounds.left - 1 && r.right <= bounds.right + 1);
  });
  assert.ok(titleFits, "Unbroken title must not cover adjacent controls");
  const fallback = await pane.evaluate(el => {
    const probe = document.createElement("div");
    probe.style.cssText = "min-width:1800px;height:1200px;flex-shrink:0";
    el.append(probe);
    el.scrollLeft = el.scrollWidth; el.scrollTop = el.scrollHeight;
    const result = { left: el.scrollLeft, top: el.scrollTop };
    probe.remove(); el.scrollLeft = 0; el.scrollTop = 0;
    return result;
  });
  assert.ok(fallback.left > 0 && fallback.top > 0, "Aside itself can scroll both axes for oversized content");
  await page.screenshot({ path: "/tmp/reader-pane-resize-desktop.png" });
  await divider.focus(); await page.keyboard.press("End");
  await page.setViewportSize({ width: 841, height: 900 });
  await page.waitForTimeout(100);
  assert.ok((await page.locator(".content-pane").boundingBox()).width >= 319, "Viewport shrink clamps preferred width");
  point = await startDrag();
  await page.setViewportSize({ width: 390, height: 900 });
  await page.waitForTimeout(100); await page.mouse.up();
  assert.equal(await divider.isVisible(), false);
  assert.equal(await page.locator(".reader-workspace-resizing").count(), 0);
  assert.ok(await width() <= 366, "Mobile uses full available width without desktop preference overflow");
  assert.equal(await pane.evaluate(el => getComputedStyle(el).overflow), "auto");
  await page.screenshot({ path: "/tmp/reader-pane-resize-mobile.png" });
  await page.setViewportSize({ width: 1280, height: 900 });
  await divider.waitFor();
  assert.equal(requests.length, readsBefore, "Resizing is presentation-only");
  assert.equal(await page.evaluate(() => window.storageWrites.length), storageBefore, "No persistent width preference");
  assert.deepEqual(errors, []); assert.deepEqual(unexpected, []);
  await page.reload(); await divider.waitFor(); await page.waitForTimeout(100);
  assert.ok(Math.abs(await width() - originalWidth) < 2, "Reload starts at the default width");
  console.log("PASS pointer/keyboard/reset/cancel, viewport clamp, mobile switch during drag, dual-axis overflow, no requests/storage writes, reload reset");
} finally { await browser.close(); }
