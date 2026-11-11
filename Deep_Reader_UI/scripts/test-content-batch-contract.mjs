// Production TypeScript service in Chromium; isolated transport, no book writes.
import assert from 'node:assert/strict';
const { chromium } = await import(process.env.PLAYWRIGHT_MODULE || 'playwright');
const browser = await chromium.launch({headless:true,
  ...(process.env.CHROMIUM_PATH ? {executablePath:process.env.CHROMIUM_PATH} : {})});
try {
  const page = await browser.newPage();
  await page.route('**/api/**', route => route.fulfill({json:{items:[],total:0}}));
  await page.goto(process.env.UI_URL || 'http://127.0.0.1:5173');
  const results = await page.evaluate(async () => {
    const {TaskUnitContentService} = await import('/src/services/TaskUnitContentService.ts');
    const {RestClient} = await import('/src/services/RestClient.ts');
    const service = new TaskUnitContentService(new RestClient('/batch-contract'));
    const original = window.fetch;
    const ids = Array.from({length:45}, (_, n) => `unit-${n}`);
    let mode = 'ok', calls = [], active = 0, peak = 0, controller;
    window.fetch = async (url, options) => {
      if (!String(url).startsWith('/batch-contract/')) return original(url, options);
      const body = JSON.parse(options.body);
      calls.push({url:String(url), ids:body.task_unit_ids, signal:options.signal===controller?.signal});
      peak = Math.max(peak, ++active);
      await new Promise(resolve => setTimeout(resolve, 10)); active--;
      if (mode==='abort') controller.abort();
      if (mode==='missing') return new Response(JSON.stringify({detail:'task unit not found'}), {status:404});
      const contents = body.task_unit_ids.map(task_unit_id => ({task_unit_id}));
      if (mode==='wrong-order') contents.reverse();
      if (mode==='wrong-count') contents.pop();
      return new Response(JSON.stringify({contents}), {headers:{'content-type':'application/json'}});
    };
    const results=[];
    try {
      for (const count of [20,21,45]) {
        mode='ok'; calls=[]; peak=0; controller=new AbortController();
        const response = await service.fetchTaskUnitContents('中國文化【新版】.pdf', ids.slice(0,count), controller.signal);
        results.push({kind:'success',count,sizes:calls.map(c=>c.ids.length),ids:response.map(r=>r.task_unit_id),
          sent: calls.flatMap(c=>c.ids), peak,signals:calls.every(c=>c.signal),
          encoded:calls.every(c=>c.url.includes(encodeURIComponent('中國文化【新版】.pdf')))});
      }
      for (mode of ['abort','missing','wrong-order','wrong-count','duplicates','pre-aborted']) {
        calls=[]; controller=new AbortController();
        if (mode==='pre-aborted') controller.abort();
        let rejected=false;
        try { await service.fetchTaskUnitContents('book', mode==='duplicates' ? ['u','u'] : ids, controller.signal); }
        catch { rejected=true; }
        results.push({kind:mode,rejected,calls:calls.length});
      }
    } finally {window.fetch=original;}
    return results;
  });
  for (const r of results) {
    if(r.kind==='success') {
      assert.deepEqual(r.sizes,r.count===20?[20]:r.count===21?[20,1]:[20,20,5]);
      assert.deepEqual(r.ids,r.sent);assert.equal(r.ids.length,r.count);
      assert.equal(r.peak,1);assert(r.signals && r.encoded);
    } else {assert(r.rejected);assert.equal(r.calls,['duplicates','pre-aborted'].includes(r.kind)?0:1);}
  }
  console.log(JSON.stringify(results,null,2));
  console.log('PASS: nine batch-contract cases; 20-ID limit, sequential order, cancellation, unknown ID, response guards.');
} finally {await browser.close();}
