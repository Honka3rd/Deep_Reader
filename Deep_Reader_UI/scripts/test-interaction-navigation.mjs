// R5: actual App navigation, controlled API transport; no production writes/model calls.
import assert from 'node:assert/strict';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { book } from './learning-controller-fixtures.mjs';

const { chromium } = await import(process.env.PLAYWRIGHT_MODULE || 'playwright');
const browser = await chromium.launch({ headless: true, ...(process.env.CHROMIUM_PATH ? { executablePath: process.env.CHROMIUM_PATH } : {}) });
const evidence = process.env.INTERACTION_EVIDENCE_DIR || '/tmp/audit-6f6e75d';
await mkdir(evidence, { recursive: true });
const sourceText = process.env.INTERACTION_SOURCE_FILE ? await readFile(process.env.INTERACTION_SOURCE_FILE, 'utf8') : 'Original paragraph.';
const labels = { insight: 'Insights', summary: 'Summary', quiz: 'Quiz', 'critical-thinking': 'Critical thinking' };
const types = { insight: 'analysis', summary: 'summary', quiz: 'quiz', 'critical-thinking': 'critical_thinking_session' };
const pause = ms => new Promise(resolve => setTimeout(resolve, ms));
const until = async predicate => {
  for (let i = 0; i < 200; i++) { if (await predicate()) return; await pause(20); }
  throw Error('Navigation condition timed out');
};
const payload = (kind, marker) => ({
  insight: { summary: marker, reasoning: 'Evidence', interpretation: 'Meaning' },
  summary: { headline: marker, body: 'Readable summary. '.repeat(100), key_points: ['Evidence'] },
  quiz: { items: [{ item_id: 'q1', item_type: 'short_answer', prompt: marker, answer: 'Answer' }] },
  'critical-thinking': { question: marker, retry_eligible: false },
})[kind];
const results = [];
try {
  for (const width of [1280, 390]) for (const kind of Object.keys(labels)) {
    for (const scenario of ['pending', 'completed', 'unknown', 'rejected', 'other-source-edit']) {
      const page = await browser.newPage({ viewport: { width, height: 900 } });
      page.setDefaultTimeout(10000);
      const errors = [], unexpected = [], calls = [], releases = [], delivered = [];
      const saved = new Map(), revisions = { Book: 0, Other: 0 };
      let sourceMutations = 0;
      page.on('pageerror', error => errors.push(error.message));
      await page.route('**/api/**', async route => {
        const request = route.request(), url = new URL(request.url()), path = url.pathname;
        const body = request.postData() ? request.postDataJSON() : null;
        const pathDocument = path.match(/^\/api\/documents\/([^/]+)\/(?:reading-content|task-units\/content)$/)?.[1];
        const doc = body?.doc_name || (pathDocument ? decodeURIComponent(pathDocument) : url.searchParams.get('doc_name'));
        if (path === '/api/documents' || path === '/api/documents/management') return route.fulfill({ json: {
          total: 2, items: ['Book', 'Other'].map(doc_name => ({ doc_name, title: doc_name, source: 'structured', preparation_status: 'ready' })),
        } });
        if (path.endsWith('/reading-preferences/read')) return route.fulfill({ json: book(doc, 'normal') });
        if (path.endsWith('/task-layout')) return route.fulfill({ json: {
          document_id: doc, title: doc, chapters: [{ chapter_id: 'c', title: 'Chapter', sections: [{ section_id: 's', title: 'Section', task_units: [{ unit_id: 'u1' }] }] }],
        } });
        if (path.endsWith('/task-units/content')) return route.fulfill({ json: { contents: [{
          document_id: doc, chapter_id: 'c', section_id: 's', task_unit_id: 'u1',
          content_blocks: [{ block_id: 'u1:content:0', block_type: 'content', content: sourceText }],
        }] } });
        if (path.endsWith('/reparse-structure/status')) return route.fulfill({ json: { status: 'idle' } });
        if (path.endsWith('/reading-content')) {
          const state = () => ({ document_id: doc, revision: revisions[doc], source_token: `${doc}-${revisions[doc]}`,
            disposition: 'accepted', ocr_review_available: true, can_undo: false, max_chars: 10000, regions: [],
            blocks: [{ block_id: 'u1:content:0', content: sourceText, deleted: false, chapter_id: 'c', section_id: 's', task_unit_id: 'u1' }],
          });
          if (request.method() === 'GET') return route.fulfill({ json: state() });
          assert.equal(doc, 'Other'); assert.equal(body.action, 'edit');
          assert.equal(body.expected_revision, revisions[doc]); assert.equal(body.source_token, state().source_token);
          revisions[doc]++; sourceMutations++;
          return route.fulfill({ json: { ...state(), refresh: {
            document_id: doc, structure_version: null, revision: revisions[doc], source_token: state().source_token,
            mutation_id: body.mutation_id, mutation_revision: revisions[doc], scope_current: true,
            generated_invalidated: true, body_changed: true, summaries_invalidated: true, translation_block_ids: ['u1:content:0'],
            blocks: [{ block_id: 'u1:content:0', chapter_id: 'c', section_id: 's', task_unit_id: 'u1', block_order: 0,
              unit_source_hash: 'other-updated', deleted: false, manually_edited: true,
              content_block: { block_id: 'u1:content:0', block_type: 'content', content: body.text } }],
            block_page: { total: 1, offset: 0, limit: 20 },
          } } });
        }
        const match = path.match(/reading-interactions\/([^/]+)\/([^/]+)$/);
        if (!match) { unexpected.push(path); return route.fulfill({ status: 500, json: {} }); }
        const [, interaction, operation] = match, target = body.target;
        assert.equal(interaction, kind);
        calls.push({ doc: target.doc_name, operation });
        const read = operation === 'read';
        if (!read) await new Promise(resolve => releases.push({ doc: target.doc_name, resolve }));
        if (!read && scenario === 'unknown') {
          await route.fulfill({ status: 503, json: { detail: 'Unknown generation outcome' } });
          delivered.push(target.doc_name); return;
        }
        const failed = scenario === 'rejected' && (!read || saved.has(target.doc_name));
        const status = failed ? 'generation_failed' : read && !saved.has(target.doc_name) ? 'not_generated'
          : kind === 'critical-thinking' ? 'question_generated' : 'completed';
        const marker = `${target.doc_name} ${read ? 'SAVED' : 'GENERATED'} RESULT`;
        if (!read) saved.set(target.doc_name, true);
        await route.fulfill({ json: { envelope: { target: { ...target, target_id: target.doc_name }, interaction_type: types[kind], status,
          ...(status === 'question_generated' ? { session_id: `session-${target.doc_name}` } : {}) },
          payload: ['completed', 'question_generated'].includes(status) ? payload(kind, marker) : null,
        } });
        if (!read) delivered.push(target.doc_name);
      });
      const open = async () => {
        await page.getByRole('button', { name: 'Document reading interactions', exact: true }).click();
        await page.getByRole('menuitem', { name: labels[kind], exact: true }).click();
      };
      const surface = () => kind === 'insight' ? page.locator('.inline-insight-region').last() : page.getByRole('region', { name: labels[kind], exact: true });
      const close = async () => {
        await surface().getByRole('button', { name: kind === 'insight' ? 'Dismiss' : 'Close', exact: true }).click();
        await surface().waitFor({ state: 'hidden' });
      };
      const navigate = async doc => {
        await page.getByRole('button', { name: 'Library', exact: true }).click();
        await page.locator('.file-management-row').filter({ hasText: doc }).getByRole('button', { name: 'Open', exact: true }).click();
        await page.waitForURL(`**/documents/${doc}`);
        await page.getByRole('button', { name: 'Document reading interactions', exact: true }).waitFor();
      };
      const settled = async () => {
        await surface().waitFor();
        await until(async () => !(await surface().innerText()).match(/^(Loading|Generating)$/m)
          && !await surface().getByLabel('Loading', { exact: true }).count());
      };
      try {
        await page.goto(`${process.env.UI_URL || 'http://127.0.0.1:5173'}/documents/Book`);
        await open(); await until(() => releases.length === 1); await close();
        await navigate('Other');
        if (scenario === 'other-source-edit') {
          await page.locator('.hierarchy-section-button').filter({ hasText: 'Section' }).click();
          await page.locator('.reader-content-block-interaction-trigger').click();
          await page.getByRole('menuitem', { name: 'Edit', exact: true }).click();
          const editor = page.locator('[aria-labelledby="reading-content-dialog-title"]');
          await editor.getByLabel('Block text').fill(`${sourceText} Updated paragraph.`);
          await editor.getByRole('button', { name: 'Save', exact: true }).click();
          await page.getByRole('button', { name: 'Keep and save', exact: true }).click();
          await editor.waitFor({ state: 'hidden' });
        } else if (scenario !== 'pending') {
          releases[0].resolve(); await until(() => delivered.length === 1); await pause(100);
          assert(!/Book (GENERATED|SAVED) RESULT/.test(await page.locator('body').innerText()), 'Detached response cannot render on B');
          assert.equal(await page.getByRole('alert').count(), 0, 'Detached failure cannot notify B');
        }
        await navigate('Book'); await open(); await pause(150);
        assert.equal(calls.filter(c => c.operation !== 'read').length, 1, 'A→B→A must not dispatch a second generation');
        if (scenario === 'pending' || scenario === 'other-source-edit') {
          assert.equal(calls.length, 2, 'Reopen joins pending work without another read');
          releases[0].resolve(); await until(async () => (await surface().innerText()).includes('Book GENERATED RESULT'));
        } else {
          await settled();
          if (scenario === 'completed') {
            assert.equal(calls.length, 3, 'Completed work must read current saved state');
            assert((await surface().innerText()).includes('Book SAVED RESULT'));
            assert(!(await surface().innerText()).includes('Book GENERATED RESULT'));
          } else if (scenario === 'unknown') {
            assert.equal(calls.length, 2, 'Unknown failure is retained without automatic retry');
            await close(); await open(); await settled(); assert.equal(calls.length, 2);
            // Server committed despite lost response: explicit retry must reconcile, not regenerate.
            saved.set('Book', true);
            const retry = kind === 'summary' ? surface().getByRole('button', { name: 'Retry summary', exact: true })
              : surface().locator(kind === 'insight' ? '.inline-insight-generate' : kind === 'quiz' ? '.quiz-drawer-generate' : '.critical-thinking-generate');
            await retry.click();
            await until(async () => (await surface().innerText()).includes('Book SAVED RESULT'));
            assert.equal(calls.length, 3);
          } else {
            assert.equal(calls.length, 3, 'Known failed result is read but not automatically regenerated');
            assert(!(await surface().innerText()).includes('RESULT'));
          }
        }
        assert.equal(sourceMutations, scenario === 'other-source-edit' ? 1 : 0);
        assert.deepEqual(errors, []); assert.deepEqual(unexpected, []);
        await page.screenshot({ path: `${evidence}/navigation-${kind}-${scenario}-${width}.png`, animations: 'disabled' });
        results.push({ width, kind, scenario, pass: true, calls, sourceMutations });
      } catch (error) {
        results.push({ width, kind, scenario, pass: false, error: String(error), calls, errors, unexpected });
      } finally {
        console.log(`${results.at(-1)?.pass ? 'PASS' : 'FAIL'} ${width}px ${kind} ${scenario}`);
        releases.forEach(item => item.resolve()); await page.close();
      }
    }
  }
  await writeFile(`${evidence}/navigation-results.json`, JSON.stringify(results, null, 2));
  console.log(JSON.stringify(results, null, 2));
  assert(results.every(result => result.pass), 'R5 navigation regressions failed');
} finally { await browser.close(); }
