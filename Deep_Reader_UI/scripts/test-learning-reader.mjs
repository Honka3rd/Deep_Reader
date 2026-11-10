// Actual StrictMode App with controlled HTTP; no production data/model calls.
import assert from "node:assert/strict";
import * as f from "./learning-controller-fixtures.mjs";
const { chromium } = await import(process.env.PLAYWRIGHT_MODULE || "playwright");
const browser = await chromium.launch({ headless: true, ...(process.env.CHROMIUM_PATH ? { executablePath: process.env.CHROMIUM_PATH } : {}) });
const pause=ms=>new Promise(r=>setTimeout(r,ms));
const until=async fn=>{for(let i=0;i<200;i++){if(await fn())return;await pause(20);}throw Error("Fixture condition timed out");};
try {for(const width of [1280,390]) {
 const page=await browser.newPage({viewport:{width,height:900}}),calls=[],errors=[],unexpected=[];
 let mode="language_learning",fault=null,hold=null,release=null,version=1;
 const blocks=new Map(),ops=new Map(),receipts=new Map();
 const block=id=>{if(!blocks.has(id))blocks.set(id,f.block(id,`Hello world. ${id}. `+"A long readable sentence. ".repeat(10)));return blocks.get(id);};
 const project=doc=>({...f.book(doc,mode),revision:version});
 const layout=doc=>({document_id:doc,title:doc,chapters:[{chapter_id:`${doc}-c`,title:"Chapter",sections:[{section_id:`${doc}-s`,title:"Section",task_units:[{unit_id:`${doc}-u1`},{unit_id:`${doc}-u2`}]}]}]});
 page.setDefaultTimeout(10000);
 page.on("pageerror",e=>errors.push(e.message));
 await page.route("**/api/**",async route=>{
  const req=route.request(),path=decodeURIComponent(new URL(req.url()).pathname),body=req.postData()?req.postDataJSON():null;
  if(path==="/api/documents"||path==="/api/documents/management")return route.fulfill({json:{total:2,items:["Book","Other"].map(doc_name=>({doc_name,title:doc_name,source:"structured",preparation_status:"ready"}))}});
  if(path.endsWith("/task-layout"))return route.fulfill({json:layout(body.doc_name)});
  if(path.endsWith("/reparse-structure/status"))return route.fulfill({json:{status:"idle"}});
  if(path.endsWith("/reading-content"))return route.fulfill({json:{revision:0,source_token:"fixture",disposition:"accepted",can_undo:false,max_chars:10000,regions:[],blocks:[]}});
  if(path.endsWith("/task-units/content")){const doc=path.split("/")[3];return route.fulfill({json:{contents:body.task_unit_ids.map(id=>({task_unit_id:id,document_id:doc,chapter_id:`${doc}-c`,section_id:`${doc}-s`,content_blocks:Array.from({length:8},(_,i)=>{const b=block(`${id}:content:${i}`);return{block_id:b.content_block_id,block_type:"paragraph",content:b.sentences.map(s=>s.text).join("")};})}))}});}
  if(path.endsWith("/reading-preferences/read")){const saved=project(body.doc_name);calls.push({endpoint:"preferences",body});if(hold==="preferences"){hold=null;await new Promise(r=>release=r);}return route.fulfill({json:saved});}
  const endpoint=path.split("/reading-interactions/learning/")[1];
  if(!endpoint){unexpected.push(path);return route.fulfill({status:500,json:{detail:"Unexpected"}});}
  calls.push({endpoint,body});
  if(hold===endpoint){hold=null;await new Promise(r=>release=r);}
  if(endpoint==="read"){
   if(fault==="read" && body.content_block_ids.length===8 && body.content_block_ids[0].startsWith("Book-u2:")){fault=null;return route.fulfill({status:503,json:{detail:{kind:"unavailable",code:"storage_unavailable"}}});}
   return route.fulfill({json:{doc_name:body.doc_name,settings:project(body.doc_name),blocks:body.content_block_ids.map(id=>block(id))}});
  }
  if(endpoint==="generate"){
   const b=block(body.content_block_id),op=f.operation(b,body.component,"succeeded",body.command_id);op.handle.doc_name=body.doc_name;ops.set(body.command_id,op);f.save(b,body.component);f.attach(b,op);const receipt=f.submission(body);receipts.set(body.command_id,receipt);
   if(fault==="lost"){fault=null;return route.abort("failed");}
   return route.fulfill({json:receipt});
  }
  if(["operations/read","operations/wait"].includes(endpoint))return route.fulfill({json:ops.get(body.handle.operation_id)});
  if(endpoint==="commands/read")return route.fulfill({json:{doc_name:body.doc_name,command_id:body.command_id,found:receipts.has(body.command_id),receipt:receipts.get(body.command_id)||null}});
  unexpected.push(path);return route.fulfill({status:500,json:{detail:"Unexpected learning endpoint"}});
 });
 const count=e=>calls.filter(c=>c.endpoint===e).length;
 const openSection=async()=>{await page.locator(".hierarchy-section-button").filter({hasText:"Section"}).click();};
 const ready=()=>until(async()=>await page.locator(".learning-progress-state").filter({hasText:/^Saved$/}).count()===16);
 const library=()=>page.getByRole("button",{name:"Library",exact:true}).click();
 const closeLibrary=()=>page.getByRole("button",{name:"Close",exact:true}).click();
 hold="preferences";
 await page.goto(`${process.env.UI_URL||"http://127.0.0.1:5173"}/documents/Book`);await openSection();await until(()=>release!==null);
 assert(await page.locator(".reader-content-page-next").isDisabled(),"mode check guards rapid forward navigation");
 const releaseMode=release;release=null;hold="operations/wait";releaseMode();await until(()=>release!==null);
 await until(async()=>await page.locator(".learning-progress-state").filter({hasText:/^Saved$/}).count()===15);
 assert.equal(await page.locator(".learning-progress-overview").evaluate(e=>e.open),false,"details initially collapsed");
 const bar=page.getByRole("progressbar",{name:"Current-page learning completion"});
 assert(await bar.isVisible(),"collapsed overview retains overall progress");
 assert.equal(await bar.getAttribute("aria-valuenow"),"94");
 assert(await page.locator(".reader-content-page-next").isDisabled(),"active work blocks Next");
 assert(await page.locator(".reader-content-page-last").isDisabled(),"Last page cannot bypass guard");
 await page.locator(".learning-progress-toggle").focus();await page.keyboard.press("Enter");await page.keyboard.press("Enter");
 assert(await bar.isVisible(),"keyboard collapse retains progress");
 await page.locator(".reader-content-page-next").evaluate(e=>e.click());await page.locator(".reader-content-page-last").evaluate(e=>e.click());
 await pause(80);assert.equal(count("generate"),16,"disabled controls queue no next-page requests");
 await page.locator(".reader-content-block-list").evaluate(e=>e.scrollTop=0);
 await page.screenshot({path:`/tmp/learning-progress-collapsed-${width}.png`,fullPage:true});
 const releaseWait=release;release=null;releaseWait();await ready();await until(async()=>!await page.locator(".reader-content-page-next").isDisabled());
 assert.equal(await bar.getAttribute("aria-valuenow"),"100");
 assert.equal(count("generate"),16,"StrictMode one command per component/current-page block");
 assert(calls.filter(c=>c.endpoint==="generate").every(c=>c.body.content_block_id.startsWith("Book-u1:")),"no next page work");
 assert.equal(await page.locator(".reader-content-block-body").count(),8,"original readable");
 assert.equal(await page.locator(".learning-translated-sentence").count(),8,"paragraph sources render paired translations");
 assert.equal(await page.locator(".learning-vocabulary").count(),8,"paragraph sources render vocabulary");
 await page.locator(".learning-progress-toggle").click();
 assert(await page.locator(".learning-progress-list").evaluate(e=>e.scrollHeight>e.clientHeight&&e.clientHeight<=180),"bounded long progress list");
 await page.locator(".learning-progress-list").focus();await page.keyboard.press("ArrowDown");await until(async()=>await page.locator(".learning-progress-list").evaluate(e=>e.scrollTop>0));
 await page.locator(".learning-progress-list").evaluate(e=>e.scrollTop=0);
 await page.screenshot({path:`/tmp/llrn18-reader-${width}.png`,fullPage:true});
 await page.locator(".learning-translated-sentence").first().scrollIntoViewIfNeeded();
 await page.screenshot({path:`/tmp/llrn18-paragraph-pairs-${width}.png`,fullPage:true});
 await page.locator(".reader-content-block-interaction-trigger").first().click();
 assert.equal(await page.getByRole("menuitem",{name:"Translate",exact:true}).count(),0,"learning paragraph hides ordinary Translate");
 assert.equal(await page.getByRole("menuitem",{name:"View translations",exact:true}).count(),0,"learning paragraph hides ordinary translations");
 await page.keyboard.press("Escape");
 // Pause before next-page read; navigation cancels and obsolete data cannot start generation.
 hold="read";await page.locator(".reader-content-page-next").click();await until(()=>release!==null);const held=release;release=null;
 await page.locator(".reader-content-page-previous").click();await ready();held();await pause(150);assert.equal(count("generate"),16,"cancelled page did not dispatch");
 // Failed page read is not missing. Keyboard recovery starts exactly this page.
 fault="read";await page.locator(".reader-content-page-next").click();const again=page.getByRole("button",{name:"Read page again",exact:true});await again.waitFor();assert.equal(count("generate"),16);
 await again.focus();await page.keyboard.press("Enter");await ready();assert.equal(count("generate"),32);
 // Library suspends the mode controller; apply changed mode externally then close and reread.
 await library();assert.equal(await page.locator(".learning-page-status").count(),0);
 mode="normal";version++;await closeLibrary();await until(()=>calls.at(-1)?.endpoint==="preferences");await pause(100);assert.equal(await page.locator(".learning-page-status").count(),0);assert.equal(count("generate"),32);
 mode="future";version++;await library();await closeLibrary();await page.locator(".reader-mode-unavailable").waitFor();assert.equal(count("generate"),32);
 mode="language_learning";version++;await library();await closeLibrary();await ready();assert.equal(count("generate"),32,"reentry reuses saved");
 // Same-ID changed source remounts and reads before starting new work; old saved block is discarded.
 for(let i=0;i<8;i++){const id=`Book-u2:content:${i}`,b=f.block(id,"An edited source. "+"Readable source. ".repeat(20));b.source_token="b".repeat(64);blocks.set(id,b);}
 await page.reload();await openSection();await ready();await page.locator(".reader-content-page-next").click();await ready();assert.equal(count("generate"),48,"source replacement has new work");
 // Abort after a generation request was sent: reopening resolves receipt, never sends it again.
 blocks.delete("Book-u1:content:0");await page.reload();hold="generate";await openSection();await until(()=>release!==null);const releaseGenerate=release;release=null;
 await library();releaseGenerate();await pause(100);await closeLibrary();await ready();assert.equal(count("generate"),50,"no duplicate command after abort/reopen");assert(count("commands/read")>0,"uncertain command receipt consulted");
 // Out-of-order book preference response is discarded across book switch.
 await library();hold="preferences";await closeLibrary();await until(()=>release!==null);const releasePreferences=release;release=null;
 await library();await page.locator('.file-management-row[data-doc-name="Other"]').getByRole("button",{name:"Open",exact:true}).click();await openSection("Other");await ready();releasePreferences();await pause(150);
 assert((await page.locator(".learning-progress-item").first().getAttribute("data-block-id")).startsWith("Other-"),"foreign late response");
 assert.deepEqual(errors,[]);assert.deepEqual(unexpected,[]);
 console.log(JSON.stringify({width,generations:count("generate"),reads:count("read"),receiptLookups:count("commands/read"),errors,pass:true}));await page.close();
}}finally{await browser.close();}
