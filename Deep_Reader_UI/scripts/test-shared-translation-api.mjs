// Shared translation acceptance: actual App + production learning routes/lifespan + disposable file persistence.
// Provider is deterministic/offline; user books and configured databases are untouched.
import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { fileURLToPath } from "node:url";
const backend=fileURLToPath(new URL("../../Deep_Reflective_Reader/",import.meta.url));
const server=spawn(`${backend}.venv/bin/python`,[fileURLToPath(new URL("learning-controller-api-fixture.py",import.meta.url))],{cwd:backend,env:{...process.env,PYTHONPATH:backend},stdio:["ignore","pipe","pipe"]});
let log="";server.stdout.on("data",v=>log+=v);server.stderr.on("data",v=>log+=v);
const pause=ms=>new Promise(r=>setTimeout(r,ms));
const until=async fn=>{for(let i=0;i<250;i++){if(await fn())return;await pause(40);}throw Error("API fixture condition timed out");};
let browser;
try {
 await until(async()=>{if(server.exitCode!==null)throw Error(log);try{return(await fetch("http://127.0.0.1:18718/fixture")).ok;}catch{return false;}});
 const fixture=await(await fetch("http://127.0.0.1:18718/fixture")).json();assert.equal(fixture.provider_calls,0);
 const {chromium}=await import(process.env.PLAYWRIGHT_MODULE||"playwright");browser=await chromium.launch({headless:true,...(process.env.CHROMIUM_PATH?{executablePath:process.env.CHROMIUM_PATH}:{})});
 const calls=[];
 const api=async(path,body)=>{const r=await fetch(`http://127.0.0.1:18718${path}`,{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify(body)});const d=await r.json();assert(r.ok,JSON.stringify(d));return d;};
 const mode=async(learning)=>{const b=await api('/documents/reading-preferences/read',{doc_name:'book'});await api('/documents/reading-preferences/apply',{
  doc_name:'book',expected_revision:b.revision,expected_workspace_revision:b.workspace.revision,mutation_id:crypto.randomUUID(),
  selection:learning?{mode_id:'language_learning',version:'1',language_code:'en',scheme_id:'cefr_reading',scheme_version:'1',level_id:'B1'}:{mode_id:'normal',version:'1'}});};
 for(const width of [1280,390]){
  const page=await browser.newPage({viewport:{width,height:900}}),errors=[];page.on("pageerror",e=>errors.push(e.message));
  await page.route("**/api/**",async route=>{
   const req=route.request(),path=new URL(req.url()).pathname,body=req.postData()?req.postDataJSON():null;
   if(path.includes("/reading-preferences/")||(path.includes("/reading-interactions/learning/")||path.includes("/reading-interactions/translate/"))){
    calls.push({path,body});const response=await fetch(`http://127.0.0.1:18718${path.slice(4)}`,{method:req.method(),headers:{"content-type":"application/json"},...(body?{body:JSON.stringify(body)}:{})});
    const payload=await response.text();if(page.isClosed())return;return route.fulfill({status:response.status,contentType:"application/json",body:payload});
   }
   if(path==="/api/documents"||path==="/api/documents/management")return route.fulfill({json:{total:1,items:[{doc_name:"book",title:"Book",source:"structured",preparation_status:"ready"}]}});
   if(path.endsWith("/task-layout"))return route.fulfill({json:{document_id:"book",title:"Book",chapters:[{chapter_id:"chapter",title:"Chapter",sections:[{section_id:"section",title:"Section",task_units:fixture.sources.map(s=>({unit_id:s.id.split(":")[0]}))}]}]}});
   if(path.endsWith("/task-units/content"))return route.fulfill({json:{contents:body.task_unit_ids.map(id=>({task_unit_id:id,document_id:"book",chapter_id:"chapter",section_id:"section",content_blocks:fixture.sources.filter(s=>s.id.startsWith(`${id}:`)).map(s=>({block_id:s.id,block_type:"content",content:s.text}))}))}});
   if(path.endsWith("/reading-content"))return route.fulfill({json:{revision:0,source_token:"fixture",disposition:"accepted",can_undo:false,max_chars:10000,regions:[],blocks:[]}});
   if(path.endsWith("/reparse-structure/status"))return route.fulfill({json:{status:"idle"}});
   throw Error(`Unexpected request ${path}`);
  });
  const open=async()=>{await page.goto(`${process.env.UI_URL||"http://127.0.0.1:5173"}/documents/book`);await page.locator(".hierarchy-section-button").filter({hasText:"Section"}).click();};
  const ready=()=>until(async()=>await page.locator(".learning-progress-state").filter({hasText:/^Saved$/}).count()===2).catch(async e=>{console.error(await page.locator("body").innerText());throw e;});
  const translationMenu=async(name)=>{await page.locator('.reader-content-block-interaction-trigger').click();await page.getByRole('menuitem',{name,exact:true}).click();};
  const text=()=>page.locator('.translation-saved-text');
  await mode(false);await open();
  await translationMenu('Translate');
  await page.getByRole('combobox',{name:'Translate to'}).fill('Chinese (zh)');
  await page.getByRole('option',{name:'Chinese (zh)',exact:true}).click();
  await page.locator('.translation-submit').click();await text().waitFor();
  await page.locator('.translation-language-dialog').waitFor({state:'detached'});
  assert.equal(await text().innerText(),'這是一句完整譯文。 這是一句完整譯文。');
  assert.equal(await page.locator('.learning-translated-sentence').count(),0);
  await page.screenshot({path:`/tmp/shared-translation-normal-${width}.png`,fullPage:true,animations:"disabled"});
  await page.getByRole('button',{name:'Close translations',exact:true}).click();
  const beforeLearning=calls.length;
  await mode(true);await open();await ready();
  assert.equal(await page.locator('.learning-translated-sentence').count(),2);
  assert.equal(calls.slice(beforeLearning).filter(c=>c.path.endsWith('/learning/generate')&&c.body.component==='translation').length,0,'ordinary result reused in learning');
  await page.screenshot({path:`/tmp/shared-translation-learning-${width}.png`,fullPage:true,animations:"disabled"});
  await page.locator('.reader-content-page-next').click();await ready();
  await mode(false);await open();await page.locator('.reader-content-page-next').click();
  await translationMenu('View translations');await text().waitFor();
  assert.equal(await text().innerText(),'這是一句完整譯文。 這是一句完整譯文。','learning result aggregated in ordinary mode');
  assert.deepEqual(errors,[]);await page.unrouteAll({behavior:"wait"});await page.close();
  console.log(JSON.stringify({width,actualAPI:true,bidirectionalReuse:true,provider:"offline fixture",pass:true}));
 }
 const after=await(await fetch("http://127.0.0.1:18718/fixture")).json();assert.equal(after.provider_calls,4,"two blocks, one translation and one hints call each across both modes");
 assert.equal(calls.filter(c=>c.path.endsWith("/generate")).length,4);assert(calls.filter(c=>c.path.endsWith("/generate")).every(c=>!c.body.content_block_id.startsWith("unit-2:")),"no unseen page generation");
 console.log(JSON.stringify({providerCalls:after.provider_calls,generatedCommands:4,pass:true}));
}finally{await browser?.close();server.kill("SIGINT");await new Promise(r=>server.exitCode!==null?r():server.once("exit",r));console.log(log.trim());}
