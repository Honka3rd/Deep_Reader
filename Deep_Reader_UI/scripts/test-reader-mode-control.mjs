// Actual App + temporary production preference/learning routes, deterministic provider.
import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { fileURLToPath } from 'node:url';
const backend=fileURLToPath(new URL('../../Deep_Reflective_Reader/',import.meta.url));
const server=spawn(`${backend}.venv/bin/python`,[fileURLToPath(new URL('learning-controller-api-fixture.py',import.meta.url))],{cwd:backend,env:{...process.env,PYTHONPATH:backend},stdio:['ignore','pipe','pipe']});
let log=''; server.stdout.on('data',v=>log+=v);server.stderr.on('data',v=>log+=v);
const pause=ms=>new Promise(r=>setTimeout(r,ms));
const until=async fn=>{for(let i=0;i<300;i++){if(await fn())return;await pause(30);}throw Error('Condition timed out');};
const base='http://127.0.0.1:18718';
async function api(path,body){const r=await fetch(base+path,{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify(body)});const d=await r.json();assert(r.ok,JSON.stringify(d));return d;}
async function normal(){const b=await api('/documents/reading-preferences/read',{doc_name:'book'});return api('/documents/reading-preferences/apply',{doc_name:'book',expected_revision:b.revision,expected_workspace_revision:b.workspace.revision,mutation_id:crypto.randomUUID(),selection:{mode_id:'normal',version:'1'}});}
let browser;
try{
 await until(async()=>{if(server.exitCode!==null)throw Error(log);try{return(await fetch(base+'/fixture')).ok;}catch{return false;}});
 const fixture=await(await fetch(base+'/fixture')).json();
 const {chromium}=await import(process.env.PLAYWRIGHT_MODULE||'playwright');
 browser=await chromium.launch({headless:true,...(process.env.CHROMIUM_PATH?{executablePath:process.env.CHROMIUM_PATH}:{})});
 for(const width of [1280,390]){
  await normal();
  const expectedGenerations=width===1280?2:0;
  const page=await browser.newPage({viewport:{width,height:900}}),errors=[],calls=[];
  page.setDefaultTimeout(12000);page.on('pageerror',e=>errors.push(e.message));
  let failRead=false,failApply=null,holdRead=false,holdApply=false,release=null,longLabel=false;
  await page.route('**/api/**',async route=>{
   const req=route.request(),path=new URL(req.url()).pathname,body=req.postData()?req.postDataJSON():null;
   calls.push({path,body});
   if(path.includes('/reading-preferences/')||path.includes('/reading-interactions/learning/')||path.includes('/reading-interactions/translate/')){
    if(path.endsWith('/reading-preferences/read')&&failRead){failRead=false;return route.fulfill({status:503,json:{detail:{kind:'unavailable',code:'preference_storage_unavailable'}}});}
    if(path.endsWith('/reading-preferences/apply')&&failApply==='rejected'){failApply=null;return route.fulfill({status:422,json:{detail:{kind:'ineligible',code:'preference_settings_invalid'}}});}
    const response=await fetch(base+path.slice(4),{method:req.method(),headers:{'content-type':'application/json'},...(body?{body:JSON.stringify(body)}:{})});
    const json=await response.json();
    if((path.endsWith('/reading-preferences/read')&&holdRead)||(path.endsWith('/reading-preferences/apply')&&holdApply)){holdRead=false;holdApply=false;await new Promise(r=>release=r);}
    if(path.endsWith('/reading-preferences/read')&&longLabel)json.modes.forEach(m=>m.label+=' · '+ 'VeryLongModeLabel'.repeat(8));
    if(path.endsWith('/reading-preferences/apply')&&failApply==='unknown'){failApply=null;return route.fulfill({status:503,json:{detail:{kind:'unknown',code:'preference_commit_unknown',recovery:'read_preferences'}}});}
    if(page.isClosed())return;
    return route.fulfill({status:response.status,json});
   }
   if(path==='/api/documents'||path==='/api/documents/management')return route.fulfill({json:{total:1,items:[{doc_name:'book',title:'Book',source:'structured',preparation_status:'ready'}]}});
   if(path.endsWith('/task-layout'))return route.fulfill({json:{document_id:'book',title:'Book',chapters:[{chapter_id:'chapter',title:'Chapter',sections:[{section_id:'section',title:'Section',task_units:fixture.sources.map(s=>({unit_id:s.id.split(':')[0]}))}]}]}});
   if(path.endsWith('/task-units/content'))return route.fulfill({json:{contents:body.task_unit_ids.map(id=>({task_unit_id:id,document_id:'book',chapter_id:'chapter',section_id:'section',content_blocks:fixture.sources.filter(s=>s.id.startsWith(`${id}:`)).map(s=>({block_id:s.id,block_type:'content',content:s.text}))}))}});
   if(path.endsWith('/reading-content'))return route.fulfill({json:{revision:0,source_token:'fixture',disposition:'accepted',can_undo:false,max_chars:10000,regions:[],blocks:[]}});
   if(path.endsWith('/reparse-structure/status'))return route.fulfill({json:{status:'idle'}});
   throw Error(`Unexpected request ${path}`);
  });
  const button=page.locator('.reader-mode-switch'),dialog=page.locator('.reading-settings-dialog');
  const applied=label=>until(async()=> (await button.innerText()).includes(label));
  const choose=async mode=>{await page.locator(`.reading-mode-option[data-mode-id="${mode}"]`).click();};
  const open=async()=>{await button.click();await page.locator('.reading-mode-option').first().waitFor();};
  const save=()=>page.locator('.reading-settings-save').click();
  const countApply=()=>calls.filter(c=>c.path.endsWith('/reading-preferences/apply')).length;
  const countGenerate=()=>calls.filter(c=>c.path.endsWith('/generate')).length;
  await page.goto(`${process.env.UI_URL||'http://127.0.0.1:5173'}/documents/book`);
  await page.locator('.hierarchy-section-button').filter({hasText:'Section'}).click();await applied('Normal reading');
  assert.equal(await button.count(),1);assert.equal(countApply(),0);
  await button.focus();await page.keyboard.press('Enter');await page.locator('.reading-mode-option').first().waitFor();
  assert.equal(await page.locator('.file-management-dialog').count(),0,'no Library detour');
  await choose('language_learning');assert.equal(countGenerate(),0,'preview does not generate');
  await page.keyboard.press('Escape');await page.locator('.reading-settings-discard').waitFor();
  await page.getByRole('button',{name:'Keep editing',exact:true}).click();
  holdApply=true;await save();await until(()=>release!==null);assert(await page.locator('.reading-settings-save').isDisabled());
  await page.locator('.reading-settings-save').evaluate(e=>e.click());release();release=null;await dialog.waitFor({state:'hidden'});await applied('Language learning');
  await until(async()=>await page.locator('.learning-progress-state').filter({hasText:/^Saved$/}).count()===2);
  assert.equal(countApply(),1);assert.equal(countGenerate(),expectedGenerations,'translation and vocabulary once');
  await until(async()=>await button.evaluate(e=>e===document.activeElement));
  const firstId=await page.locator('.reader-content-block').first().getAttribute('data-block-id');
  await page.screenshot({path:`/tmp/reader-mode-control/learning-${width}.png`,fullPage:true,animations:'disabled'});
  await open();await choose('normal');await save();await dialog.waitFor({state:'hidden'});await applied('Normal reading');
  assert.equal(await page.locator('.learning-sentence-pairs').count(),0);
  assert.equal(await page.locator('.reader-content-block').first().getAttribute('data-block-id'),firstId);
  await page.screenshot({path:`/tmp/reader-mode-control/normal-${width}.png`,fullPage:true,animations:'disabled'});
  await open();await choose('language_learning');await save();await dialog.waitFor({state:'hidden'});await applied('Language learning');
  await until(async()=>await page.locator('.learning-translated-sentence').count()>0);assert.equal(countGenerate(),expectedGenerations);
  // Known rejection keeps applied mode; unknown commit must reconcile, never auto-replay.
  await open();await choose('normal');failApply='rejected';const rejected=countApply();await save();await until(()=>countApply()===rejected+1);
  await until(async()=>await page.locator('.reading-settings-save').isEnabled());assert(await dialog.isVisible());
  assert.equal((await api('/documents/reading-preferences/read',{doc_name:'book'})).active_mode,'language_learning');
  failApply='unknown';await save();await page.locator('.reading-settings-recovery').waitFor();
  const unknown=countApply();assert(await page.locator('.reading-settings-save').isDisabled());
  await page.locator('.reading-settings-reload').click();await until(async()=>await page.locator('.reading-settings-save').isEnabled());
  assert.equal(countApply(),unknown);await page.getByRole('button',{name:'Back to reader',exact:true}).click();await dialog.waitFor({state:'hidden'});await applied('Normal reading');
  // Settings read failure and explicit retry.
  failRead=true;await button.click();await page.locator('.reading-settings-retry').click();await page.locator('.reading-mode-option').first().waitFor();
  await page.getByRole('button',{name:'Back to reader',exact:true}).click();await dialog.waitFor({state:'hidden'});
  // Cancel in-flight read; late result cannot reopen a dialog or change the mode.
  holdRead=true;await button.click();await until(()=>release!==null);await page.getByRole('button',{name:'Close reading settings'}).click();
  release();release=null;await dialog.waitFor({state:'hidden'});await applied('Normal reading');
  // Reader mode read failure stays explicit, and retry is read-only.
  await open();failRead=true;await page.getByRole('button',{name:'Back to reader',exact:true}).click();
  await applied('Could not load');await page.locator('.reader-mode-recheck').click();await applied('Normal reading');
  // Saving from a later page preserves the selected page and content DOM.
  await page.locator('.reader-content-page-next').click();await applied('Normal reading');
  const source=page.locator('.reader-content-block').first();await source.evaluate(e=>window.modeKeptSource=e);
  const pageText=await page.locator('.reader-content-pagination-bar').innerText();
  await open();await choose('normal');await save();await dialog.waitFor({state:'hidden'});await applied('Normal reading');
  assert.equal(await page.locator('.reader-content-pagination-bar').innerText(),pageText);
  assert(await source.evaluate(e=>e===window.modeKeptSource));
  // Fullscreen keeps the same control; long authoritative labels wrap on mobile.
  await page.locator('.reader-content-fullscreen-toggle').click();await page.waitForFunction(()=>document.documentElement.classList.contains('reader-fullscreen'));
  assert(await button.isVisible());longLabel=true;await open();await page.getByRole('button',{name:'Back to reader',exact:true}).click();await dialog.waitFor({state:'hidden'});
  await applied('VeryLongModeLabel');
  assert(await button.evaluate(e=>e.getBoundingClientRect().right<=innerWidth+1&&e.scrollWidth<=e.clientWidth+1));
  await page.screenshot({path:`/tmp/reader-mode-control/fullscreen-long-${width}.png`,fullPage:true,animations:'disabled'});
  // The same settings controller retires a late save when its reader document changes.
  await page.evaluate(async()=>{
   const {default:React}=await import('/node_modules/.vite/deps/react.js');
   const {default:Client}=await import('/node_modules/.vite/deps/react-dom_client.js');
   const {default:DOM}=await import('/node_modules/.vite/deps/react-dom.js');
   const {useReadingSettingsController}=await import('/src/features/reading-settings/controller.ts');
   const {ReadingSettingsView}=await import('/src/features/reading-settings/view.tsx');
   const host=document.createElement('div');document.body.append(host);const root=Client.createRoot(host);
   const state={};window.readerSettingsHarness=state;
   function Harness({owner}){const c=useReadingSettingsController(owner,{closeOnBookSave:true});state.c=c;return React.createElement(ReadingSettingsView,{controller:c,origin:'reader'});}
   state.render=owner=>DOM.flushSync(()=>root.render(React.createElement(Harness,{owner})));
   state.dispose=()=>{root.unmount();host.remove();delete window.readerSettingsHarness;};
   state.render('book');
  });
  await page.evaluate(()=>window.readerSettingsHarness.c.openBook('book','Book'));
  await page.locator('.reading-mode-option').first().waitFor();await choose('normal');
  holdApply=true;await save();await until(()=>release!==null);
  await page.evaluate(()=>window.readerSettingsHarness.render('other-book'));
  await dialog.waitFor({state:'hidden'});release();release=null;
  await until(async()=>await page.evaluate(()=>window.readerSettingsHarness.c.target===null&&!window.readerSettingsHarness.c.busy));
  assert.equal(await dialog.count(),0);await page.evaluate(()=>window.readerSettingsHarness.dispose());
  assert.deepEqual(errors,[]);assert.equal(countGenerate(),expectedGenerations,'no re-generation during settings observations');
  await page.unrouteAll({behavior:'wait'});await page.close();
  console.log(JSON.stringify({width,actualPreferenceAndLearningAPI:true,modeRoundtrip:true,failuresAndRecovery:true,keyboardFocus:true,fullscreen:true,provider:'offline fixture',pass:true}));
 }
}finally{await browser?.close();server.kill('SIGINT');await new Promise(r=>server.exitCode!==null?r():server.once('exit',r));console.log(log.trim());}
