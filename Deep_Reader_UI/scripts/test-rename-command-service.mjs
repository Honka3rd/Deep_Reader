// Real lifecycle service + RestClient in Chromium; isolated mocked HTTP only.
import assert from 'node:assert/strict';
const { chromium } = await import(process.env.PLAYWRIGHT_MODULE || 'playwright');
const browser = await chromium.launch({ headless: true,
  ...(process.env.CHROMIUM_PATH ? { executablePath: process.env.CHROMIUM_PATH } : {}) });
try {
  const page = await browser.newPage();
  await page.route('**/api/**', route => route.fulfill({ json: { items: [], total: 0 } }));
  await page.goto(process.env.UI_URL || 'http://127.0.0.1:5173');
  const result = await page.evaluate(async () => {
    const { DocumentLifecycleService } = await import('/src/services/DocumentLifecycleService.ts');
    const { RestClient } = await import('/src/services/RestClient.ts');
    const service = new DocumentLifecycleService(new RestClient('/rename-test'));
    const requests = []; let contexts = 0, mode = 'unknown';
    const original = window.fetch;
    window.fetch = async (url, options) => {
      if (!String(url).startsWith('/rename-test/')) return original(url, options);
      if (String(url).includes('rename-context')) {
        contexts++;
        return new Response(JSON.stringify({ doc_name: 'A', source_token: String(contexts).repeat(64) }),
          { headers: { 'content-type': 'application/json' } });
      }
      requests.push(JSON.parse(options.body));
      // Let concurrent callers overlap before resolving the single dispatch.
      await new Promise(resolve => setTimeout(resolve, 20));
      if (mode === 'transport') throw new TypeError('lost response');
      const status = mode === 'unknown' ? 503 : mode === 'conflict' ? 409 : 200;
      return new Response(JSON.stringify({ doc_name: 'A', action: 'rename', new_doc_name: 'B',
        status: status === 200 ? 'accepted' : status === 409 ? 'conflict' : 'failed',
        reason: status === 409 ? 'conflict' : 'storage_failure' }),
        { status, headers: { 'content-type': 'application/json' } });
    };
    try {
      await Promise.allSettled([service.renameDocument('A','B'),service.renameDocument('A','B')]);
      mode = 'transport'; await service.renameDocument('A','B').catch(() => {});
      mode = 'ok'; await service.renameDocument('A','B');
      const afterRetry = { contexts, requests: requests.slice() };
      // A later explicit command represents a new incarnation/round trip.
      mode = 'conflict'; await service.renameDocument('A','B').catch(() => {});
      mode = 'ok'; await service.renameDocument('A','B');
      return { afterRetry, contexts, requests };
    } finally { window.fetch = original; }
  });
  assert.equal(result.afterRetry.contexts, 1);
  assert.equal(result.afterRetry.requests.length, 3);
  assert.deepEqual(result.afterRetry.requests[0], result.afterRetry.requests[1]);
  assert.deepEqual(result.afterRetry.requests[0], result.afterRetry.requests[2]);
  assert.equal(result.contexts, 3);
  assert.equal(result.requests.length, 5);
  assert.equal(new Set(result.requests.map(r => r.command_id)).size, 3);
  assert.equal(result.requests[3].expected_source_token, '2'.repeat(64));
  assert.equal(result.requests[4].expected_source_token, '3'.repeat(64));
  console.log('PASS: duplicate dispatch, 503/transport retry, completed/new command, definitive rejection, new context');
} finally { await browser.close(); }
