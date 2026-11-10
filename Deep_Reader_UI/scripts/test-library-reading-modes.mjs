// Actual App, controlled read-only APIs: no user data or model calls.
import assert from "node:assert/strict";
import { book } from "./learning-controller-fixtures.mjs";
const { chromium } = await import(process.env.PLAYWRIGHT_MODULE || "playwright");
const browser = await chromium.launch({ headless: true, ...(process.env.CHROMIUM_PATH ? { executablePath: process.env.CHROMIUM_PATH } : {}) });
const pause = ms => new Promise(r => setTimeout(r, ms));
const until = async fn => { for (let i = 0; i < 250; i++) { if (await fn()) return; await pause(20); } assert.fail("Library mode condition timed out"); };
try { for (const width of [1280, 390]) {
  const page = await browser.newPage({ viewport: { width, height: 900 } });
  const calls = [], unexpected = [], errors = [];
  let names = ["Book", "Other", ...Array.from({length:198}, (_,i) => `Unread ${String(i).padStart(3,"0")}`)];
  const modes = new Map([["Book", "normal"], ["Other", "language_learning"]]);
  let fail = null, malformed = false, hold = null, release = null, inFlight = 0, maxInFlight = 0;
  function projection(doc) {
    const value = book(doc, modes.get(doc) || "normal");
    value.modes.forEach(m => { m.label = m.mode_id === "normal" ? "Normal reading" : "Language learning"; });
    return value;
  }
  page.on("pageerror", e => errors.push(e.message));
  await page.route("**/api/**", async route => {
    const request = route.request(), path = new URL(request.url()).pathname;
    if (["/api/documents", "/api/documents/management"].includes(path)) return route.fulfill({json:{total:names.length,
      items:names.map(doc_name=>({doc_name,title:doc_name,source:"structured",preparation_status:"ready"}))}});
    if (path === "/api/documents/reading-preferences/read") {
      const doc = request.postDataJSON().doc_name, response = projection(doc);
      calls.push(doc); inFlight++; maxInFlight = Math.max(maxInFlight,inFlight);
      try {
        if (hold === doc) { hold = null; await new Promise(r=>release=r); }
        else await pause(60);
        if (fail === doc) { fail = null; return await route.fulfill({status:503,json:{detail:{kind:"unavailable",code:"preference_storage_unavailable"}}}); }
        return await route.fulfill({json:malformed ? {...response,doc_name:"wrong-document"} : response});
      } finally { inFlight--; }
    }
    unexpected.push(path); return route.fulfill({status:500,json:{}});
  });
  page.setDefaultTimeout(10000);
  const row = doc => page.locator('.file-management-row').filter({has:page.locator('.file-management-document-name').filter({hasText:new RegExp(`^${doc}$`)})});
  const label = doc => row(doc).locator('.file-management-mode-label');
  const show = async doc => { await label(doc).scrollIntoViewIfNeeded(); };
  const expectMode = async (doc, text) => { await show(doc); try { await until(async()=>await label(doc).innerText()===text); }
    catch { throw Error(`${width}: ${doc} expected ${text}, got ${await label(doc).innerText()} (${calls.filter(d=>d===doc).length} reads)`); } };
  const open = () => page.getByRole('button',{name:'Library',exact:true}).click();
  const close = () => page.locator('.file-management-dialog').getByRole('button',{name:'Close',exact:true}).click();
  const refresh = () => page.locator('.file-management-refresh-action').click();
  await page.goto(process.env.UI_URL || 'http://127.0.0.1:5173'); await open();
  await expectMode('Book','Normal reading'); await expectMode('Other','Language learning');
  assert.equal(await page.locator('.file-management-row').count(),200);
  assert(calls.length<20,`visible reads must not fan out to all 200 rows (${calls.length})`);
  assert(maxInFlight<=2,`initial visible request concurrency ${maxInFlight}`);
  const initialReads=calls.length;
  await page.screenshot({path:`/tmp/library-mode-first-${width}.png`,fullPage:true});
  await expectMode('Unread 197','Normal reading'); assert(calls.includes('Unread 197'),'scroll loads newly visible row');
  await close(); modes.set('Book','language_learning'); await open(); await expectMode('Book','Language learning');
  modes.set('Book','normal'); await refresh(); await expectMode('Book','Normal reading');
  // A failed label is explicit and retry is a read, never a settings mutation.
  fail='Book'; await refresh(); await expectMode('Book','Mode unavailable');
  await row('Book').getByRole('button',{name:'Retry reading mode for Book',exact:true}).click();
  await expectMode('Book','Normal reading');
  // Old close/reopen result cannot overwrite the fresh mode, even if it arrives last.
  hold='Book'; await refresh(); await show('Book'); await until(()=>release!==null);
  const oldRead=release; release=null; await close(); modes.set('Book','language_learning'); await open();
  await expectMode('Book','Language learning'); oldRead(); await pause(100); assert.equal(await label('Book').innerText(),'Language learning');
  // Identity replacement gets its own read; the old label must not migrate by row position.
  names=names.map(n=>n==='Book'?'Renamed':n); modes.set('Renamed','normal'); await refresh();
  await expectMode('Renamed','Normal reading'); assert.equal(await row('Book').count(),0);
  // Unknown saved modes stay unavailable; response-identity errors are never labelled Normal.
  modes.set('Renamed','future'); await refresh(); await expectMode('Renamed','Unsupported saved mode · Unavailable');
  malformed=true; await refresh(); await expectMode('Renamed','Mode unavailable'); malformed=false;
  await row('Renamed').getByRole('button',{name:'Retry reading mode for Renamed',exact:true}).click();
  await expectMode('Renamed','Unsupported saved mode · Unavailable');
  // Filtering out a held row suppresses its late result; returning rereads it.
  hold='Other'; await refresh(); await show('Other'); await until(()=>release!==null);
  const filteredRead=release;release=null;
  await page.locator('.file-management-filter-summary').click(); await page.getByLabel('Search',{exact:true}).fill('Renamed');
  await until(async()=>await row('Other').count()===0); filteredRead(); modes.set('Other','normal');
  await page.getByLabel('Search',{exact:true}).fill(''); await expectMode('Other','Normal reading');
  assert.deepEqual(errors,[]);assert.deepEqual(unexpected,[]);
  console.log(JSON.stringify({width,firstOpen:true,reopenRefresh:true,lateReadIgnored:true,identityFilter:true,retry:true,
    initialVisibleReads:initialReads,totalBooks:200,pass:true})); await page.close();
}} finally {await browser.close();}
