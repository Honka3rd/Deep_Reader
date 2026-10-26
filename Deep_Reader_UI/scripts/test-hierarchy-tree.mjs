// Actual App with fixture APIs: tree navigation, narrow/long labels and read-only expansion.
import assert from 'node:assert/strict';
const { chromium } = await import(process.env.PLAYWRIGHT_MODULE || 'playwright');
const section = (id, title, units = 1) => ({ section_id: id, title,
  task_units: Array.from({ length: units }, (_, i) => ({ unit_id: `${id}-u${i}` })) });
const chapters = [
  { chapter_id: 'c1', title: '第一章：中國文化的深層結構與社會生活中的歷史傳承', sections: [
    section('s1', 'A long section heading about cultural structures and their influence on everyday life'),
    section('s2', 'UnbrokenHeading'.repeat(16)), section('empty', 'Empty section', 0),
  ] },
  { chapter_id: 'merged', title: 'Brief Bibliography of Gustave Flaubert', sections: [section('merged-s', 'Sources and references', 451)] },
  { chapter_id: 'empty-chapter', title: 'Empty chapter', sections: [] },
];
const browser = await chromium.launch({ headless: true });
try {
  for (const width of [1280, 390, 320]) {
    const page = await browser.newPage({ viewport: { width, height: 900 } });
    const errors = [], unexpected = [], contentReads = [], requests = [];
    page.on('pageerror', e => errors.push(e.message));
    await page.addInitScript(() => {
      window.storageWrites = [];
      const original = Storage.prototype.setItem;
      Storage.prototype.setItem = function(key, value) { window.storageWrites.push(key); return original.call(this, key, value); };
    });
    await page.route('**/api/**', route => {
      const request = route.request(), path = new URL(request.url()).pathname;
      requests.push(path);
      if (path === '/api/documents/task-layout') {
        const doc = request.postDataJSON().doc_name;
        const large = doc === 'Large';
        return route.fulfill({ json: { document_id: doc, title: doc,
          chapters: large ? Array.from({ length: 100 }, (_, i) => ({ chapter_id: `large-${i}`, title: `Chapter ${i}`,
            sections: Array.from({ length: 10 }, (_, j) => section(`large-${i}-${j}`, `Section ${i}.${j}`)),
          })) : chapters } });
      }
      if (path.endsWith('/reparse-structure/status')) return route.fulfill({ json: { status: 'idle' } });
      if (path.endsWith('/reading-content')) return route.fulfill({ json: {
        revision: 0, source_token: 'fixture', disposition: 'accepted', can_undo: false, max_chars: 10000, regions: [], blocks: [],
      } });
      if (path.endsWith('/task-units/content')) {
        const ids = request.postDataJSON().task_unit_ids;
        contentReads.push(...ids);
        return route.fulfill({ json: { contents: ids.map(id => ({ document_id: 'Book', task_unit_id: id,
          chapter_id: 'c1', section_id: 's1', content_blocks: [{ block_id: `${id}:content:0`, block_type: 'paragraph', content: 'Tree selection fixture.' }],
        })) } });
      }
      unexpected.push(path); return route.fulfill({ status: 500, json: { detail: 'Unexpected API' } });
    });
    const open = doc => page.goto(`${process.env.UI_URL || 'http://127.0.0.1:5173'}/documents/${doc}`);
    await open('Book');
    const tree = page.getByRole('tree', { name: 'Chapters and sections' });
    await tree.waitFor();
    const chapter = page.locator('[data-item-id="chapter:c1"]');
    const first = page.locator('[data-item-id="section:s1"]');
    const empty = page.locator('[data-item-id="section:empty"]');
    assert.equal(await chapter.getAttribute('aria-expanded'), 'false');
    await chapter.focus(); await page.keyboard.press('ArrowRight');
    await first.waitFor();
    assert.equal(await chapter.getAttribute('aria-expanded'), 'true');
    assert.equal(await empty.getAttribute('aria-disabled'), 'true');
    // The tree has a single keyboard entry; arrows change focus, Enter selects a section.
    assert.equal(await tree.locator('[role="treeitem"][tabindex="0"]').count(), 1);
    await chapter.focus(); await page.keyboard.press('ArrowLeft');
    assert.equal(await chapter.getAttribute('aria-expanded'), 'false');
    await first.waitFor({ state: 'hidden' });
    await page.keyboard.press('ArrowRight');
    await first.waitFor();
    await page.keyboard.press('ArrowRight');
    assert.equal(await first.evaluate(el => el === document.activeElement), true);
    assert.deepEqual(contentReads, [], 'Expansion and focus must not read content');
    await page.keyboard.press('Enter');
    await page.locator('.reader-content-block-body').first().waitFor();
    assert.equal(await first.getAttribute('aria-selected'), 'true');
    assert.deepEqual(contentReads, ['s1-u0']);
    await page.keyboard.press('ArrowDown');
    await page.keyboard.press('ArrowDown');
    assert.equal(await empty.evaluate(el => el === document.activeElement), false, 'Empty sections are skipped');
    // Menu click/keyboard must not select a section or toggle its parent.
    const readsBeforeMenu = requests.length;
    const menu = chapter.locator('.hierarchy-chapter-interaction-trigger');
    await menu.focus(); await page.keyboard.press('Enter');
    await page.getByRole('menuitem', { name: 'Insights', exact: true }).waitFor();
    assert.equal(await chapter.getAttribute('aria-expanded'), 'true');
    await page.keyboard.press('Escape');
    assert.equal(await menu.evaluate(el => el === document.activeElement), true);
    assert.equal(requests.length, readsBeforeMenu);
    // Collapse/reopen via mouse preserves section selection without new API requests.
    await chapter.locator(':scope > .hierarchy-tree-content .hierarchy-tree-icon').click();
    assert.equal(await chapter.getAttribute('aria-expanded'), 'false');
    await chapter.locator(':scope > .hierarchy-tree-content .hierarchy-tree-icon').click();
    await first.waitFor();
    assert.equal(await first.getAttribute('aria-selected'), 'true');
    assert.deepEqual(contentReads, ['s1-u0']);
    const bounds = await tree.evaluate(el => {
      const rect = node => { const r = node.getBoundingClientRect(); return { left: r.left, right: r.right, width: r.width }; };
      return { horizontalOverflow: el.scrollWidth > el.clientWidth + 1,
        pageOverflow: document.documentElement.scrollWidth > innerWidth || document.documentElement.scrollHeight > innerHeight,
        rows: [...el.querySelectorAll('.hierarchy-tree-content')].map(row => {
          const label = row.querySelector('.hierarchy-tree-label'), action = row.querySelector('.hierarchy-tree-action');
          const texts = [...label.querySelectorAll('span')].flatMap(span => {
            const range = document.createRange(); range.selectNodeContents(span); return [...range.getClientRects()].map(r => ({ left: r.left, right: r.right }));
          });
          return { row: rect(row), label: rect(label), action: rect(action), texts };
        }) };
    });
    assert.equal(bounds.horizontalOverflow, false, `${width}: no horizontal tree scrolling`);
    assert.equal(bounds.pageOverflow, false, `${width}: no page overflow`);
    for (const { row, label, action, texts } of bounds.rows) {
      assert.ok(label.width >= 90, `${width}: usable text width ${label.width}`);
      assert.ok(action.width >= 28 && action.right <= row.right + 1, `${width}: fixed action fits`);
      assert.ok(label.right <= action.left + 1, `${width}: label does not overlap action`);
      for (const text of texts) assert.ok(text.left >= label.left - 1 && text.right <= label.right + 1, `${width}: full text wraps inside label`);
    }
    await tree.evaluate(el => { el.scrollTop = 0; });
    await page.screenshot({ path: `/tmp/hierarchy-tree-${width}.png` });
    // Fresh documents must not inherit a previous document's collapsed state.
    await chapter.locator(':scope > .hierarchy-tree-content .hierarchy-tree-icon').click();
    await open('Other'); await chapter.waitFor();
    assert.equal(await chapter.getAttribute('aria-expanded'), 'false');
    await chapter.focus(); await page.keyboard.press('ArrowRight');
    await first.waitFor();
    assert.equal(await first.getAttribute('aria-selected'), 'false');
    if (width === 1280) {
      const start = performance.now(); await open('Large');
      await page.locator('[data-item-id="chapter:large-99"]').waitFor({ state: 'attached' });
      const loadMs = Math.round(performance.now() - start);
      assert.equal(await tree.locator('[role="treeitem"]').count(), 100, 'Collapsed sections are not mounted');
      const last = page.locator('[data-item-id="chapter:large-99"]');
      await last.scrollIntoViewIfNeeded();
      const interactionStart = performance.now();
      await last.locator(':scope > .hierarchy-tree-content .hierarchy-tree-icon').click({ timeout: 2500 });
      await page.locator('[data-item-id="section:large-99-9"]').waitFor();
      assert.equal(await last.getAttribute('aria-expanded'), 'true');
      assert.equal(await tree.locator('[role="treeitem"]').count(), 110);
      console.log(`MEASURE 1100-node layout (100 chapters mounted): initial navigation/render ${loadMs}ms, scroll-end expansion ${Math.round(performance.now() - interactionStart)}ms`);
      const metrics = await tree.evaluate(el => ({ scrolls: el.scrollHeight > el.clientHeight,
        overflow: document.documentElement.scrollHeight > innerHeight }));
      assert.equal(metrics.scrolls, true); assert.equal(metrics.overflow, false);
    }
    assert.deepEqual(await page.evaluate(() => window.storageWrites), []);
    assert.deepEqual(errors, []); assert.deepEqual(unexpected, []);
    await page.close();
    console.log(`PASS ${width}px: tree keyboard/mouse navigation, selection, disabled rows, menu focus, long titles, scroll and document reset`);
  }
} finally { await browser.close(); }
