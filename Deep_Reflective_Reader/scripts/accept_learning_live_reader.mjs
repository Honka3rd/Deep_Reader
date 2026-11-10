// Actual App renders saved live-provider SQL results through production learning HTTP.
// Ancillary Library/hierarchy transport is projected from the isolated fixture;
// this does not claim acceptance of the production Library/content endpoints.
import assert from 'node:assert/strict';
import {writeFile} from 'node:fs/promises';
const api=process.env.LEARNING_LIVE_API, doc=process.env.LEARNING_LIVE_BOOK;
const output=process.env.LEARNING_LIVE_REPORT_DIR;
assert(api && doc && output, 'requires isolated acceptance fixture');
const cases=await(await fetch(`${api}/fixture`)).json();
const fixture=cases.find(c=>c.doc_name===doc);
const pageResult=fixture.outputs[0].page.blocks[0];
const {chromium}=await import(process.env.PLAYWRIGHT_MODULE||'playwright');
const browser=await chromium.launch({headless:true,...(process.env.CHROMIUM_PATH?{executablePath:process.env.CHROMIUM_PATH}:{})});
const evidence=[];
try{
 for(const width of [1280,390]){
  const context=await browser.newContext({viewport:{width,height:900}});
  await context.tracing.start({screenshots:true,snapshots:true});
  const page=await context.newPage(), errors=[], calls=[];
  page.on('pageerror',e=>errors.push(e.message));
  await page.route('**/api/**',async route=>{
   const req=route.request(),path=new URL(req.url()).pathname,body=req.postData()?req.postDataJSON():null;
   calls.push({path,method:req.method()});
   if(path.includes('/reading-preferences/')||path.includes('/reading-interactions/learning/')){
    // A cached Reader visit must never need a paid command.
    assert(!path.endsWith('/generate'),'unexpected generation on saved data');
    const response=await fetch(`${api}${path.slice(4)}`,{method:req.method(),headers:{'content-type':'application/json'},...(body?{body:JSON.stringify(body)}:{})});
    return route.fulfill({status:response.status,contentType:'application/json',body:await response.text()});
   }
   if(path==='/api/documents'||path==='/api/documents/management')return route.fulfill({json:{total:1,items:[{doc_name:doc,title:'Acceptance book',source:'structured',preparation_status:'ready'}]}});
   const unit=fixture.block_id.split(':')[0];
   if(path.endsWith('/task-layout'))return route.fulfill({json:{document_id:doc,title:'Acceptance book',chapters:[{chapter_id:'chapter',title:'Chapter',sections:[{section_id:'section',title:'Section',task_units:[{unit_id:unit}]}]}]}});
   if(path.endsWith('/task-units/content'))return route.fulfill({json:{contents:[{task_unit_id:unit,document_id:doc,chapter_id:'chapter',section_id:'section',content_blocks:[{block_id:fixture.block_id,block_type:'content',content:fixture.text}]}]}});
   if(path.endsWith('/reading-content'))return route.fulfill({json:{revision:0,source_token:'fixture',disposition:'accepted',can_undo:false,max_chars:10000,regions:[],blocks:[]}});
   if(path.endsWith('/reparse-structure/status'))return route.fulfill({json:{status:'idle'}});
   throw Error(`Unexpected request ${path}`);
  });
  const open=async()=>{
   await page.goto(`${process.env.UI_URL||'http://127.0.0.1:5173'}/documents/${doc}`);
   await page.locator('.hierarchy-section-button').filter({hasText:'Section'}).click();
   await page.locator('.learning-translated-sentence').first().waitFor();
   await page.locator('.learning-vocabulary').waitFor();
  };
  await open();
  const translations=pageResult.components.find(c=>c.component==='translation').saved.rows;
  assert.deepEqual(await page.locator('.learning-translated-sentence').allTextContents(),translations.map(r=>r.translated_text));
  assert.deepEqual(await page.locator('.learning-source-sentence').allTextContents(),pageResult.sentences.filter(s=>s.sentence_id).map(s=>s.text));
  assert.equal(await page.locator('.learning-hint').count(),pageResult.components.find(c=>c.component==='hints').saved.items.length);
  await page.waitForFunction(()=>Array.from(document.querySelectorAll('.learning-component-status')).length===2
    && Array.from(document.querySelectorAll('.learning-component-status')).every(e=>e.textContent.endsWith(': Saved')));
  assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1));
  await page.screenshot({path:`${output}/${fixture.case_id}-${width}.png`,fullPage:true});
  await page.locator('.learning-vocabulary').scrollIntoViewIfNeeded();
  await page.screenshot({path:`${output}/${fixture.case_id}-${width}-vocabulary.png`,fullPage:true});
  await open();
  assert.deepEqual(errors,[]);
  assert.equal(calls.filter(c=>c.path.endsWith('/generate')).length,0);
  evidence.push({width,pass:true,rows:translations.length,requests:calls});
  await page.unrouteAll({behavior:'wait'});
  await context.tracing.stop({path:`${output}/${fixture.case_id}-${width}-trace.zip`});
  await context.close();
 }
 await writeFile(`${output}/${fixture.case_id}-browser.json`,JSON.stringify(evidence,null,2),{mode:0o600});
 console.log(JSON.stringify({pass:true,actualLearningAPI:true,storage:'disposable SQL',
  provider:process.env.LEARNING_LIVE_REPLAY==='1'?'recorded output replay':'live saved results',reloadGeneration:0}));
}finally{await browser.close();}
