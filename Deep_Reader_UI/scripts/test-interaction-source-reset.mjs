// R4: actual App source editing and actual React controllers with cancellation-ignoring transport.
import assert from 'node:assert/strict';
import { mkdir, readFile } from 'node:fs/promises';
import { book } from './learning-controller-fixtures.mjs';
const { chromium } = await import(process.env.PLAYWRIGHT_MODULE || 'playwright');
const browser = await chromium.launch({ headless: true, ...(process.env.CHROMIUM_PATH ? { executablePath: process.env.CHROMIUM_PATH } : {}) });
const evidence = process.env.INTERACTION_EVIDENCE_DIR || '/tmp/audit-b2517c2';
await mkdir(evidence, { recursive: true });
const sourceText=process.env.INTERACTION_SOURCE_FILE?await readFile(process.env.INTERACTION_SOURCE_FILE,'utf8'):'Original paragraph.';
const pause = ms => new Promise(r => setTimeout(r, ms));
const until = async fn => { for (let i=0;i<200;i++) { if(await fn())return; await pause(20); } throw Error('Source reset condition timed out'); };
const labels = { insight:'Insights', summary:'Summary', quiz:'Quiz', 'critical-thinking':'Critical thinking' };
const types = { insight:'analysis', summary:'summary', quiz:'quiz', 'critical-thinking':'critical_thinking_session' };
const results = [];
try {
 for (const width of (process.env.CONTROLLER_ONLY ? [] : [1280,390])) for (const kind of Object.keys(labels)) {
  const page=await browser.newPage({viewport:{width,height:900}}), errors=[], unexpected=[], calls=[], releases=[];
  page.setDefaultTimeout(10000);page.on('pageerror',e=>errors.push(e.message));
  let revision=0, text=sourceText, lastRefresh=null;
  const state=()=>({document_id:'Book',revision,source_token:`content-${revision}`,disposition:'accepted',ocr_review_available:true,can_undo:false,max_chars:10000,regions:[],
   blocks:[{block_id:'u1:content:0',content:text,deleted:false,chapter_id:'c',section_id:'s',task_unit_id:'u1'}]});
  await page.route('**/api/**',async route=>{
   const request=route.request(),url=new URL(request.url()),path=url.pathname,body=request.postData()?request.postDataJSON():null;
   if(path==='/api/documents'||path==='/api/documents/management')return route.fulfill({json:{total:1,items:[{doc_name:'Book',title:'Book',source:'structured',preparation_status:'ready'}]}});
   if(path.endsWith('/reading-preferences/read'))return route.fulfill({json:book('Book','normal')});
   if(path.endsWith('/task-layout'))return route.fulfill({json:{document_id:'Book',title:'Book',chapters:[{chapter_id:'c',title:'Chapter',sections:[{section_id:'s',title:'Section',task_units:[{unit_id:'u1'}]}]}]}});
   if(path.endsWith('/task-units/content'))return route.fulfill({json:{contents:[{task_unit_id:'u1',document_id:'Book',chapter_id:'c',section_id:'s',content_blocks:[{block_id:'u1:content:0',block_type:'content',content:text}]}]}});
   if(path.endsWith('/reparse-structure/status'))return route.fulfill({json:{status:'idle'}});
   if(path.endsWith('/reading-content')) {
    if(request.method()==='GET')return route.fulfill({json:url.searchParams.has('refresh_mutation_id')?lastRefresh:state()});
    assert.equal(body.action,'edit');assert.equal(body.expected_revision,revision);assert.equal(body.source_token,state().source_token);
    revision++;text=body.text;
    lastRefresh={document_id:'Book',structure_version:null,revision,source_token:state().source_token,mutation_id:body.mutation_id,mutation_revision:revision,
     scope_current:true,generated_invalidated:true,body_changed:true,summaries_invalidated:true,translation_block_ids:['u1:content:0'],
     blocks:[{block_id:'u1:content:0',chapter_id:'c',section_id:'s',task_unit_id:'u1',block_order:0,unit_source_hash:`hash-${revision}`,deleted:false,manually_edited:true,
      content_block:{block_id:'u1:content:0',block_type:'content',content:text}}],block_page:{total:1,offset:0,limit:20}};
    return route.fulfill({json:{...state(),refresh:lastRefresh}});
   }
   const match=path.match(/reading-interactions\/([^/]+)\/([^/]+)$/);
   if(!match){unexpected.push(path);return route.fulfill({status:500,json:{}});}
   const [,interaction,operation]=match, captured=revision, target=body.target;
   assert.equal(interaction,kind);calls.push({operation,revision:captured});
   const read=operation==='read';
   if(!read)await new Promise(r=>releases.push(r));
   const marker=captured===0?'OLD SOURCE RESULT':'CURRENT SOURCE RESULT';
   const payload={insight:{summary:marker,reasoning:marker,interpretation:marker},summary:{headline:marker,body:marker,key_points:[marker]},
    quiz:{items:[{item_id:'q1',item_type:'short_answer',prompt:marker,answer:marker}]},'critical-thinking':{question:marker,retry_eligible:false}}[kind];
   return route.fulfill({json:{envelope:{target:{...target,target_id:'Book'},interaction_type:types[kind],status:read?'not_generated':kind==='critical-thinking'?'question_generated':'completed',
    ...(!read&&kind==='critical-thinking'?{session_id:'session-'+captured}:{})},payload:read?null:payload}});
  });
  const open=async()=>{await page.getByRole('button',{name:'Document reading interactions',exact:true}).click();await page.getByRole('menuitem',{name:labels[kind],exact:true}).click();};
  const surface=()=>kind==='insight'?page.locator('.inline-insight-region').last():page.getByRole('region',{name:labels[kind],exact:true});
  const close=async()=>{await surface().getByRole('button',{name:kind==='insight'?'Dismiss':'Close',exact:true}).click();await surface().waitFor({state:'hidden'});};
  try {
   await page.goto(`${process.env.UI_URL||'http://127.0.0.1:5173'}/documents/Book`);
   await page.locator('.hierarchy-section-button').filter({hasText:'Section'}).click();
   await page.locator('.reader-content-block-interaction-trigger').waitFor();
   await open();await until(()=>releases.length===1);await close();await open();await pause(80);
   assert.equal(calls.length,2,'R1: unchanged close/reopen joins the same generation');await close();
   await page.locator('.reader-content-block-interaction-trigger').click();await page.getByRole('menuitem',{name:'Edit',exact:true}).click();
   const editor=page.locator('[aria-labelledby="reading-content-dialog-title"]');
   await editor.getByLabel('Block text').fill(`${sourceText} Updated paragraph. 中文 citation.`);
   await editor.getByRole('button',{name:'Save',exact:true}).click();await page.getByRole('button',{name:'Keep and save',exact:true}).click();
   await editor.waitFor({state:'hidden'});assert.equal(revision,1);
   await open();await pause(100);releases[0]();await pause(100);
   assert.equal((await surface().innerText()).includes('OLD SOURCE RESULT'),false,'R4: old result must not render after source save');
   await until(()=>releases.length===2);
   assert.deepEqual(calls.map(c=>c.revision),[0,0,1,1],'new source must read before generating');
   // Old finally/cleanup cannot evict current work or re-enable a third generation.
   await close();await open();await pause(80);assert.equal(calls.length,4);
   releases[1]();await until(async()=>(await surface().innerText()).includes('CURRENT SOURCE RESULT'));
   assert(!((await surface().innerText()).includes('OLD SOURCE RESULT')));
   assert.deepEqual(errors,[]);assert.deepEqual(unexpected,[]);
   await page.screenshot({path:`${evidence}/source-reset-${kind}-${width}.png`,animations:'disabled'});
   results.push({width,kind,pass:true,calls});
  } catch(error) {results.push({width,kind,pass:false,error:String(error),calls});}
  finally {releases.forEach(r=>r());await page.close();}
 }
 // No reliance on fetch cancellation: all these services deliberately ignore AbortSignal.
 const page=await browser.newPage();await page.route('**/api/**',r=>r.fulfill({json:{total:0,items:[]}}));await page.goto(process.env.UI_URL||'http://127.0.0.1:5173');
 const controllerResults=await page.evaluate(async()=>{
  const {default:React}=await import('/node_modules/.vite/deps/react.js');
  const {default:{createRoot}}=await import('/node_modules/.vite/deps/react-dom_client.js');
  const {useReadingInteractionMenuController}=await import('/src/features/reading-interactions/controller.ts');
  const {SummaryNoteDrawer}=await import('/src/features/reading-interactions/SummaryNoteDrawer.tsx');
  const {SummaryNoteService}=await import('/src/services/SummaryNoteService.ts');
  const pause=ms=>new Promise(r=>setTimeout(r,ms));
  const until=async fn=>{for(let i=0;i<200;i++){if(fn())return;await pause(5);}throw Error('Controller condition timed out');};
  const results=[];
  for(const variant of ['insight','quiz','critical_thinking','summary','summary_mounted','summary_read','summary_failed']) {
   const kind=variant.startsWith('summary')?'summary':variant;
   const host=document.createElement('div');document.body.append(host);const root=createRoot(host);let c,release,version=0,show=true;
   const calls=[],errors=[];const target={targetLevel:'document',docName:'Book',displayTitle:'Book',breadcrumb:['Book']};
   const request={doc_name:'Book',target_type:'document'};
   const response=(status)=>({envelope:{target:{...request,target_id:'Book'},interaction_type:{insight:'analysis',quiz:'quiz',critical_thinking:'critical_thinking_session',summary:'summary'}[kind],status,
    ...(kind==='critical_thinking'&&status==='completed'?{session_id:'session'}:{})},payload:status==='not_generated'?null:kind==='insight'?{summary:version?'CURRENT':'OLD',reasoning:'Evidence',interpretation:'Meaning'}:
    kind==='quiz'?{items:[{item_id:'q',item_type:'short_answer',prompt:version?'CURRENT':'OLD',answer:'Answer'}]}:kind==='summary'?{headline:version?'CURRENT':'OLD',body:'Text',key_points:[]}:{question:version?'CURRENT':'OLD',retry_eligible:false}});
   const read=async()=>{calls.push('read');const saved=response(version?'completed':'not_generated');if(variant==='summary_read'&&!version)await new Promise(r=>release=r);return saved;};
   const generate=async()=>{calls.push('generate');const old=response('completed');await new Promise(r=>release=r);if(variant==='summary_failed')throw Error('OLD failure');return old;};
   const original=SummaryNoteService.prototype.summary;
   SummaryNoteService.prototype.summary=async(_,action)=>action==='read'?read():generate();
   function Harness(){c=useReadingInteractionMenuController();return kind==='summary'&&show?React.createElement(SummaryNoteDrawer,{selection:{kind:'summary',target:request,breadcrumb:['Book']},generationTasks:c.generationTasks,dirtyRef:{current:false},onClose:()=>{}}):null;}
   const render=()=>root.render(React.createElement(Harness));
   try {
    render();await until(()=>c);
    const activate=async()=>{const s=c.selectInteractionAction(kind,target);await pause(5);return c.activateInteraction(s,request,
     {readInsight:read,generateInsight:generate,readQuiz:read,generateQuiz:generate,readCriticalThinkingSession:read,generateCriticalThinkingQuestion:generate},e=>errors.push(e));};
    const first=kind==='summary'?null:activate();await until(()=>release);
    version=1;c.resetInteractions();
    let ignoredWhileMounted=true;
    if(variant.startsWith('summary_')){release();await pause(25);const region=document.querySelector('[aria-label="Summary"]');ignoredWhileMounted=!region?.textContent.includes('OLD')&&!region?.querySelector('[role="alert"]')&&calls.length===(variant==='summary_read'?1:2);}
    if(kind==='summary'){show=false;render();await pause(10);show=true;render();}else await pause(10);
    const second=kind==='summary'?null:activate();await pause(20);release();await Promise.all([first,second]);await pause(20);
    const visible=kind==='summary'?document.querySelector('[aria-label="Summary"]')?.textContent:JSON.stringify(kind==='insight'?c.openInlineInsightsByTarget:kind==='quiz'?c.quizStateByTarget:c.criticalThinkingStateByTarget);
    results.push({kind:variant,pass:ignoredWhileMounted&&calls.join(',')===(variant==='summary_read'?'read,read':'read,generate,read')&&visible.includes('CURRENT')&&!visible.includes('OLD')&&!errors.length,calls,errors,visible});
   }finally{release?.();root.unmount();host.remove();SummaryNoteService.prototype.summary=original;}
  }
  const {InteractionGenerationTasks}=await import('/src/features/reading-interactions/generation-tasks.ts');
  const tasks=new InteractionGenerationTasks();let oldRelease,newRelease;
  const old=tasks.start('same-target',()=>new Promise(r=>oldRelease=r));await until(()=>oldRelease);
  tasks.invalidateSources();const current=tasks.start('same-target',()=>new Promise(r=>newRelease=r));await until(()=>newRelease);
  oldRelease('OLD');await old;await pause(5);
  const preserved=tasks.find('same-target',false)===current;newRelease('CURRENT');await current;tasks.dispose();
  results.push({kind:'registry-late-cleanup',pass:preserved});
  return results;
 });
 results.push(...controllerResults);await page.close();
 console.log(JSON.stringify(results,null,2));assert(results.every(r=>r.pass),'Source invalidation regression failed');
}finally{await browser.close();}
