// Actual App, fixture API only; verifies wrapping with real-world and extreme titles.
import assert from "node:assert/strict";
const { chromium } = await import(process.env.PLAYWRIGHT_MODULE || "playwright");
const browser = await chromium.launch({ headless: true });
try {
  for (const title of ["Madame_Bovary_261021_155934", "Book", "Unbroken".repeat(16)]) {
    const page = await browser.newPage();
    const errors = [], unexpected = [];
    page.on("pageerror", error => errors.push(error.message));
    await page.route("**/api/**", route => {
      const path = new URL(route.request().url()).pathname;
      if (path === "/api/documents/task-layout") return route.fulfill({ json: {
        document_id: "Book", title, chapters: [{ chapter_id: "c", title: "Chapter", sections: [{
          section_id: "s", title: "Body", task_units: Array.from({ length: 451 }, (_, i) => ({ unit_id: `u${i}` })),
        }] }],
      } });
      if (path.endsWith("/reparse-structure/status")) return route.fulfill({ json: { doc_name: "Book", status: "idle" } });
      if (path.endsWith("/reading-content")) return route.fulfill({ json: {
        revision: 0, source_token: "fixture", disposition: "accepted", can_undo: false, max_chars: 10000, regions: [], blocks: [],
      } });
      unexpected.push(path);
      return route.fulfill({ status: 500, json: { detail: "Unexpected request" } });
    });
    await page.goto(`${process.env.UI_URL || "http://127.0.0.1:5173"}/documents/Book`);
    const header = page.locator(".hierarchy-navigation-header");
    await header.waitFor();
    for (const width of [1440, 1280, 841, 840, 390, 320, 1440]) {
      await page.setViewportSize({ width, height: 900 });
      const metrics = await header.evaluate(el => {
        const rect = node => {
          const r = node.getBoundingClientRect();
          return { left: r.left, right: r.right, top: r.top, bottom: r.bottom };
        };
        const heading = el.querySelector(".hierarchy-document-title");
        const text = document.createRange(); text.selectNodeContents(heading);
        const count = el.querySelector(".hierarchy-document-unit-count");
        return { parent: rect(el), heading: rect(heading), text: Array.from(text.getClientRects(), r => ({ left: r.left, right: r.right })),
          children: Array.from(el.children).filter(node => node.getBoundingClientRect().width > 0).map(rect),
          count: rect(count), countFits: count.scrollWidth <= count.clientWidth + 1 };
      });
      const context = `${title.slice(0, 30)} at ${width}px`;
      for (const r of metrics.children) assert.ok(r.left >= metrics.parent.left - 1 && r.right <= metrics.parent.right + 1
        && r.top >= metrics.parent.top - 1 && r.bottom <= metrics.parent.bottom + 1, `${context}: children fit header`);
      for (const r of metrics.text) assert.ok(r.left >= metrics.heading.left - 1 && r.right <= metrics.heading.right + 1,
        `${context}: title text must not overflow into count`);
      metrics.children.forEach((a, i) => metrics.children.slice(i + 1).forEach(b => {
        assert.ok(Math.min(a.right, b.right) - Math.max(a.left, b.left) <= 1
          || Math.min(a.bottom, b.bottom) - Math.max(a.top, b.top) <= 1, `${context}: header items must not overlap`);
      }));
      assert.ok(metrics.countFits, `${context}: count stays readable`);
      if (title.startsWith("Madame") && width <= 390)
        assert.ok(metrics.count.top >= metrics.heading.bottom, `${context}: count flows below long title`);
      if (title === "Book" && width === 1440)
        assert.ok(metrics.count.top < metrics.heading.bottom, "Short title and count remain on the same row when space allows");
      if (title.startsWith("Madame") && [1280, 390, 320].includes(width))
        await page.screenshot({ path: `/tmp/hierarchy-header-${width}.png` });
    }
    await page.getByRole("button", { name: "Document reading interactions", exact: true }).click();
    await page.getByRole("menuitem", { name: "Insights", exact: true }).waitFor();
    await page.keyboard.press("Escape");
    assert.deepEqual(errors, []); assert.deepEqual(unexpected, []);
    await page.close();
    console.log(`PASS ${title.slice(0, 30)}: 320-1440px, resize both ways, text bounds, non-overlap, count wrapping and menu`);
  }
} finally { await browser.close(); }
