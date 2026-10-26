// Read-only real-book capture; browser playback separates render cost from API latency.
import assert from "node:assert/strict";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
const { chromium } = await import(process.env.PLAYWRIGHT_MODULE || "playwright");
const folder = fileURLToPath(new URL('../../Deep_Reflective_Reader/data/test-books/reader-large-section/', import.meta.url));
await mkdir(folder, { recursive: true });
const captureFile = `${folder}/madame-bovary-response.json`;
let fixture;
try { fixture = JSON.parse(await readFile(captureFile, 'utf8')); }
catch (error) {
  if (error.code !== 'ENOENT') throw error;
  const name = 'Madame_Bovary_261021_155934';
  const call = async (path, body) => {
    const response = await fetch((process.env.READER_API || 'http://127.0.0.1:8000') + path,
      body ? { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) } : {});
    assert.ok(response.ok, `${path}: ${response.status}`);
    return response.json();
  };
  const layout = await call('/documents/task-layout', { doc_name: name, refresh_task_units: false,
    task_unit_split_mode: 'progressive', include_anchor_page_evidence: false });
  const section = layout.chapters.flatMap(c => c.sections).find(s => s.title === 'Brief Bibliography of Gustave Flaubert');
  assert.ok(section);
  const contents = [], units = section.task_units.map(u => u.unit_id);
  assert.equal(units.length, 451);
  for (let offset = 0; offset < units.length; offset += 20) {
    const next = await call(`/documents/${encodeURIComponent(name)}/task-units/content`,
      { task_unit_ids: units.slice(offset, offset + 20), segmented: true });
    contents.push(...next.contents);
    console.log(`Captured ${contents.length}/${units.length} real units`);
  }
  fixture = { name, layout, section, contents,
    reading: await call(`/documents/${encodeURIComponent(name)}/reading-content`) };
  await writeFile(captureFile, JSON.stringify(fixture), { flag: 'wx', mode: 0o600 });
}
const baseline = process.env.RENDER_BASELINE === '1';
const browser = await chromium.launch({ headless: true });
const report = [];
try {
  for (const width of baseline ? [1280] : [1280, 390]) {
    const page = await browser.newPage({ viewport: { width, height: 900 } });
    page.setDefaultTimeout(120000);
    const errors = [], requests = [];
    page.on('pageerror', error => errors.push(error.message));
    await page.addInitScript(() => {
      window.renderMetrics = { maxGroups: 0, longTasks: [] };
      new MutationObserver(() => {
        window.renderMetrics.maxGroups = Math.max(window.renderMetrics.maxGroups,
          document.querySelectorAll('.reader-content-task-unit-group').length);
      }).observe(document, { childList: true, subtree: true });
      new PerformanceObserver(list => window.renderMetrics.longTasks.push(...list.getEntries().map(e => e.duration)))
        .observe({ type: 'longtask', buffered: true });
    });
    await page.route('**/api/**', route => {
      const request = route.request(), url = new URL(request.url());
      const path = decodeURIComponent(url.pathname.replace(/^\/api/, ''));
      if (path === '/documents/task-layout') return route.fulfill({ json: fixture.layout });
      if (path === `/documents/${fixture.name}/reading-content` && request.method() === 'GET') return route.fulfill({ json: fixture.reading });
      if (path === `/documents/${fixture.name}/task-units/content`) {
        const ids = request.postDataJSON().task_unit_ids; requests.push(ids);
        return route.fulfill({ json: { contents: ids.map(id => {
          const unit = fixture.contents.find(u => u.task_unit_id === id);
          assert.ok(unit, `Unexpected unit ${id}`); return unit;
        }) } });
      }
      if (path.endsWith('/reparse-structure/status')) return route.fulfill({ json: { status: 'idle', doc_name: fixture.name } });
      if (path === '/documents' || path === '/documents/management') return route.fulfill({ json: { items: [], total: 0 } });
      throw new Error(`Unexpected request ${request.method()} ${path}`);
    });
    await page.goto((process.env.UI_URL || 'http://127.0.0.1:5173') + '/documents/' + encodeURIComponent(fixture.name));
    const start = performance.now();
    await page.locator('.hierarchy-section-button').filter({ hasText: fixture.section.title }).click();
    await page.locator('.reader-content-block-list .reader-content-task-unit-group').first().waitFor();
    await page.waitForTimeout(300);
    const initialMs = performance.now() - start;
    const initial = await page.evaluate(() => ({ ...window.renderMetrics,
      groups: document.querySelectorAll('.reader-content-task-unit-group').length,
      blocks: document.querySelectorAll('.reader-content-block').length,
      nodes: document.querySelectorAll('*').length,
      page: document.querySelector('.reader-content-pagination-bar').textContent }));
    if (!baseline) {
      assert.equal(initial.groups, 1);
      assert.equal(initial.maxGroups, 1, 'No initial all-units flash or hidden measurement tree');
      assert.match(initial.page, /Page 1 \/ 451/);
      assert.equal(await page.locator('.reader-content-pagination-measure').count(), 0);
    }
    const steps = [];
    for (const [control, index] of [['Last page', 450], ['Previous', 449], ['First page', 0], ['Next', 1]]) {
      const before = performance.now();
      await page.getByRole('button', { name: control, exact: true }).click();
      const id = fixture.section.task_units[index].unit_id;
      await page.locator(`.reader-content-block-list [data-task-unit-id="${id}"]`).first().waitFor();
      steps.push({ control, durationMs: performance.now() - before });
      if (!baseline) {
        assert.equal(await page.locator('.reader-content-task-unit-group').count(), 1);
        const expected = fixture.contents.find(unit => unit.task_unit_id === id).content_blocks.map(block => block.block_id);
        assert.deepEqual(await page.locator('.reader-content-block-list [data-block-id]').evaluateAll(
          blocks => blocks.map(block => block.dataset.blockId)), expected, 'Only the current unit blocks are mounted, without dropping text');
      }
    }
    if (!baseline) {
      await page.setViewportSize({ width, height: 760 });
      await page.waitForTimeout(100);
      assert.match(await page.locator('.reader-content-pagination-bar').innerText(), /Page 2 \/ 451/);
      assert.equal(await page.locator('.reader-content-task-unit-group').count(), 1);
      const scroll = await page.locator('.reader-content-block-list').evaluate(el => {
        el.scrollTop = el.scrollHeight;
        return { height: el.clientHeight, content: el.scrollHeight, top: el.scrollTop,
          overflow: getComputedStyle(el).overflowY };
      });
      assert.ok(scroll.height > 0, 'Current page retains a usable viewport');
      assert.equal(scroll.overflow, 'auto');
      if (scroll.content > scroll.height) assert.ok(scroll.top > 0, 'Long unit remains scrollable');
      await page.getByRole('button', { name: 'First page', exact: true }).click();
      await page.getByRole('button', { name: 'Next', exact: true }).click();
      assert.equal(await page.locator('.reader-content-block-list').evaluate(el => el.scrollTop), 0,
        'Changing pages resets the previous page scroll offset');
      await page.locator('.reader-content-block-list').scrollIntoViewIfNeeded();
    }
    await page.screenshot({ path: `${folder}/${baseline ? 'before' : 'after'}-${width}.png`, animations: 'disabled' });
    if (!baseline) {
      const readsBefore = requests.length;
      await page.getByRole('button', { name: 'Fullscreen reading', exact: true }).click();
      await page.waitForFunction(() => document.documentElement.classList.contains('reader-fullscreen'));
      assert.equal(await page.locator('.reader-content-task-unit-group').count(), 1);
      assert.match(await page.locator('.reader-content-pagination-bar').innerText(), /Page 2 \/ 451/);
      await page.screenshot({ path: `${folder}/fullscreen-${width}.png`, animations: 'disabled' });
      await page.getByRole('button', { name: 'Exit fullscreen reading', exact: true }).click();
      await page.waitForFunction(() => !document.documentElement.classList.contains('reader-fullscreen'));
      assert.match(await page.locator('.reader-content-pagination-bar').innerText(), /Page 2 \/ 451/);
      assert.equal(requests.length, readsBefore, 'Real section fullscreen toggling must not reload content');
    }
    assert.deepEqual(errors, []);
    report.push({ width, initialMs, initial, steps, contentRequests: requests.length,
      unitsFetched: requests.flat().length, source: captureFile, productionWrites: 0 });
    await page.close();
  }
} finally {
  await writeFile(`${folder}/${baseline ? 'before' : 'after'}.json`, JSON.stringify(report, null, 2), { mode: 0o600 });
  await browser.close();
}
console.log(JSON.stringify(report, null, 2));
