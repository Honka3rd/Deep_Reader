// Focused Library rename UI acceptance. All API traffic is intercepted.
import assert from 'node:assert/strict';
const { chromium } = await import(process.env.PLAYWRIGHT_MODULE || 'playwright');
const browser = await chromium.launch({ headless: true,
  ...(process.env.CHROMIUM_PATH ? { executablePath: process.env.CHROMIUM_PATH } : {}) });
try {
  for (const width of [1280,390]) {
    const page = await browser.newPage({ viewport: { width, height: 900 } });
    const errors = [], requests = []; let contexts = 0, name = 'A', fail = true;
    page.on('pageerror', e => errors.push(e.message));
    await page.route('**/api/**', async route => {
      const path = new URL(route.request().url()).pathname;
      if (path.endsWith('/documents') || path.endsWith('/documents/management')) {
        return route.fulfill({ json: { items: [{ doc_name: name, title: name, source_type: 'txt',
          source_status: 'available', preparation_status: 'ready' }], total: 1 } });
      }
      if (path.endsWith('/rename-context')) {
        contexts++; return route.fulfill({ json: { doc_name: name, source_token: String(contexts).repeat(64) } });
      }
      if (path.endsWith('/lifecycle/rename')) {
        const body = route.request().postDataJSON(); requests.push(body);
        if (fail) { fail = false; return route.fulfill({ status: 503, json: {
          doc_name: body.doc_name, action: 'rename', status: 'failed', reason: 'storage_failure', recovery_required: true } }); }
        name = body.new_doc_name;
        return route.fulfill({ json: { doc_name: body.doc_name, action: 'rename', status: 'accepted', new_doc_name: name } });
      }
      return route.fulfill({ json: {} });
    });
    await page.goto(process.env.UI_URL || 'http://127.0.0.1:5173');
    await page.getByRole('button', { name: 'Library', exact: true }).click();
    const row = page.locator('.file-management-row').filter({ has: page.locator('.file-management-document-title', { hasText: /^A$/ }) });
    await row.locator('.file-management-rename-action').click();
    const dialog = page.locator('.file-management-rename-dialog');
    await dialog.getByRole('textbox').fill('B');
    assert.equal(requests.length,0);
    await dialog.locator('.file-management-confirm-rename-action').click();
    await page.getByText('storage_failure', { exact: true }).waitFor();
    await page.screenshot({ path: `/tmp/audit-main-ac544288/rename-retry-${width}.png`, fullPage:true });
    await dialog.locator('.file-management-confirm-rename-action').click();
    await dialog.waitFor({state:'hidden'});
    assert.equal(contexts,1); assert.equal(requests.length,2);
    assert.deepEqual(requests[0],requests[1]);
    await page.locator('.file-management-row').filter({ has: page.locator('.file-management-document-title', {hasText:/^B$/}) })
      .locator('.file-management-rename-action').click();
    await dialog.getByRole('textbox').fill('A');
    await dialog.locator('.file-management-confirm-rename-action').click();
    await dialog.waitFor({state:'hidden'});
    assert.equal(contexts,2); assert.equal(requests.length,3);
    assert.notEqual(requests[0].command_id,requests[2].command_id);
    assert.deepEqual(errors,[]);
    await page.screenshot({ path: `/tmp/audit-main-ac544288/rename-complete-${width}.png`, fullPage:true });
    await page.close();
    console.log(`PASS ${width}px: explicit confirmation, unknown-outcome retry, reverse rename new command, no page errors`);
  }
} finally { await browser.close(); }
