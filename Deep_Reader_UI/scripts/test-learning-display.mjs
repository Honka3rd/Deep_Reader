// Actual App, controlled HTTP; no production/model writes.
import assert from "node:assert/strict";
import * as f from "./learning-controller-fixtures.mjs";
const {chromium}=await import(process.env.PLAYWRIGHT_MODULE||"playwright");
const browser=await chromium.launch({headless:true,...(process.env.CHROMIUM_PATH?{executablePath:process.env.CHROMIUM_PATH}:{})});
const pause=ms=>new Promise(r=>setTimeout(r,ms));
const until=async fn=>{for(let i=0;i<250;i++){if(await fn())return;await pause(20);}throw Error("Fixture timed out");};
try{for(const width of [1280,390]){
 const page=await browser.newPage({viewport:{width,height:900}}),errors=[],unexpected=[],calls=[];
 page.setDefaultTimeout(10000);page.on("pageerror",e=>errors.push(e.message));
 let mode="normal",version=1,failure=null,hold=false,release=null;
 const b1=f.block("u1:content:0"),b2=f.block("u1:content:1");
 function sentences(b,parts){let offset=0;b.sentences=parts.map((text,i)=>{const start=offset;offset+=Array.from(text).length;return{sentence_id:text.trim()?`${b.content_block_id}:s${i}`:null,start,end:offset,text};});b.source_length=offset;}
 sentences(b1,["😀 Hello, monde.","\n ","مرحبا بالعالم."]);
 sentences(b2,["A very long source: "+"LongUnbrokenWord".repeat(40)+" <img src=x onerror=alert(1)>."," ","Another source sentence."]);
 const blocks=[b1,b2],ops=new Map();
 function save(b,component,revision=1){const v=f.save(b,component,revision);v.saved.explanation_language="ar";
  if(component==="translation")v.saved.rows.forEach((r,i)=>r.translated_text=`الترجمة ${revision} · ${i} <b>plain text</b>`);
  else if(b===b1)v.saved.items=[{sentence_id:b.sentences[0].sentence_id,span:{start:0,end:1},source_form:"😀",language_code:"en",kind:"word",meaning:"معنى في هذا السياق",lemma:null,part_of_speech:null}];
 }
 for(const b of blocks)for(const c of ["translation","hints"])save(b,c);
 // No committed translation on block 2: one recoverable checkpoint out of two batches.
 b2.components[0].saved=null;const partial=f.operation(b2,"translation","unknown","partial-op");partial.completed_batches=1;partial.total_batches=2;ops.set(partial.handle.operation_id,partial);f.attach(b2,partial);
 const settings=()=>({...f.book("Book",mode),revision:version,workspace:{revision:1,native_language:"ar"}});
 await page.route("**/api/**",async route=>{
  const req=route.request(),path=new URL(req.url()).pathname,body=req.postData()?req.postDataJSON():null;
  if(path==="/api/documents"||path==="/api/documents/management")return route.fulfill({json:{total:1,items:[{doc_name:"Book",title:"Book",source:"structured",preparation_status:"ready"}]}});
  if(path.endsWith("/task-layout"))return route.fulfill({json:{document_id:"Book",title:"Book",chapters:[{chapter_id:"c",title:"Chapter",sections:[{section_id:"s",title:"Section",task_units:[{unit_id:"u1"}]}]}]}});
  if(path.endsWith("/task-units/content"))return route.fulfill({json:{contents:[{task_unit_id:"u1",document_id:"Book",chapter_id:"c",section_id:"s",content_blocks:blocks.map(b=>({block_id:b.content_block_id,block_type:"content",content:b.sentences.map(s=>s.text).join("")}))}]}});
  if(path.endsWith("/reading-content"))return route.fulfill({json:{revision:0,source_token:"fixture",disposition:"accepted",can_undo:false,max_chars:10000,regions:[],blocks:[]}});
  if(path.endsWith("/reparse-structure/status"))return route.fulfill({json:{status:"idle"}});
  if(path.endsWith("/reading-preferences/read"))return route.fulfill({json:settings()});
  if(path.endsWith("/translate/read")){calls.push({endpoint:"ordinary-read",body});return route.fulfill({json:{target:{...body,target_id:body.content_block_id,target_type:"content_block"},language_codes:["en","ar"],default_source_language:"en",effective_content_hash:"hash",reading_revision:0,
   translations:[{source_language:"en",target_language:"ar",status:"completed",translated_text:"Ordinary saved translation",generated_at:"2026-10-28T00:00:00Z",updated_at:"2026-10-28T00:00:00Z",revision:1,mutation_id:"saved",effective_content_hash:"hash"}]}});}
  if(path.endsWith("/insight/read"))return route.fulfill({json:{envelope:{target:{...body.target,target_id:body.target.content_block_id},interaction_type:"analysis",status:"completed"},payload:{summary:"Saved insight",reasoning:"Reason",interpretation:"Interpretation"}}});
  const endpoint=path.split("/learning/")[1];if(!endpoint){unexpected.push(path);return route.fulfill({status:500,json:{}});}calls.push({endpoint,body});
  if(endpoint==="read"){
   if(failure==="read"){failure=null;return route.fulfill({status:503,json:{}});}
   return route.fulfill({json:{doc_name:"Book",settings:settings(),blocks:body.content_block_ids.map(id=>blocks.find(b=>b.content_block_id===id))}});
  }
  if(["operations/read","operations/wait"].includes(endpoint))return route.fulfill({json:ops.get(body.handle.operation_id)});
  if(endpoint==="operations/partial"){const b=blocks.find(b=>ops.get(body.handle.operation_id).content_block_id===b.content_block_id);return route.fulfill({json:{handle:body.handle,component:"translation",source_token:b.source_token,offset:0,completed_batches:1,total_batches:2,next_offset:null,batches:[{component:"translation",batch_index:0,rows:[{sentence_id:b.sentences[0].sentence_id,translated_text:"ترجمة جزئية"}]}]}});}
  if(endpoint==="generate"){
   if(hold){hold=false;await new Promise(r=>release=r);}
   const b=blocks.find(b=>b.content_block_id===body.content_block_id),state=failure||"succeeded",op=f.operation(b,body.component,state,body.command_id);failure=null;
   if(state==="succeeded"){const old=b.components.find(v=>v.component===body.component).saved?.result_revision||0;save(b,body.component,old+1);op.result_revision=old+1;}
   ops.set(body.command_id,op);f.attach(b,op);return route.fulfill({json:f.submission(body)});
  }
  unexpected.push(path);return route.fulfill({status:500,json:{}});
 });
 const b=id=>page.locator(`.reader-content-block[data-block-id="u1:content:${id}"]`);
 const openMenu=()=>b(0).locator(".reader-content-block-interaction-trigger").click();
 const changeMode=async next=>{await page.getByRole("button",{name:"Library",exact:true}).click();mode=next;version++;await page.getByRole("button",{name:"Close",exact:true}).click();};
 const gen=()=>calls.filter(c=>c.endpoint==="generate");
 await page.goto(`${process.env.UI_URL||"http://127.0.0.1:5173"}/documents/Book`);await page.locator(".hierarchy-section-button").click();
 await until(()=>calls.length===0);await openMenu();await page.getByRole("menuitem",{name:"View translations",exact:true}).click();
 await page.getByRole("tab",{name:"Reading display",exact:true}).click();await page.locator('.translation-inline-language[data-language-code="ar"] input').check();
 await page.getByRole("button",{name:"Close translations",exact:true}).click();await b(0).locator(".reader-inline-translation-text").waitFor();
 // Preserve a saved Insight through mode changes.
 await openMenu();await page.getByRole("menuitem",{name:"Insights",exact:true}).click();await b(0).locator(".reader-content-block-inline-insight").waitFor();
 await changeMode("language_learning");await b(0).locator(".learning-translated-sentence").first().waitFor();await b(1).locator(".learning-partial-label").first().waitFor();
 const ordinaryCount=calls.filter(c=>c.endpoint==="ordinary-read").length;
 assert.equal(gen().length,0,"reuse/checkpoints do not generate");assert.equal(await page.locator(".reader-inline-translation-text").count(),0);
 await openMenu();assert.equal(await page.getByRole("menuitem",{name:"Translate",exact:true}).count(),0);assert.equal(await page.getByRole("menuitem",{name:"View translations",exact:true}).count(),0);await page.keyboard.press("Escape");
 assert.equal(await b(0).locator(".learning-source-sentence").count(),2);assert.equal(await b(0).locator(".learning-translated-sentence").count(),2);
 assert.equal(await b(1).locator(".learning-translated-sentence").count(),1,"partial row only");
 assert.equal(await b(0).locator(".learning-source-sentence").first().getAttribute("lang"),"en");assert.equal(await b(0).locator(".learning-translated-sentence").first().getAttribute("lang"),"ar");
 assert.equal(await b(0).locator(".learning-translated-sentence").first().evaluate(e=>getComputedStyle(e).direction),"rtl");
 assert.equal(await b(1).locator(".learning-vocabulary-empty").count(),1);assert.equal(await page.locator(".learning-sentence-pairs img, .learning-sentence-pairs b").count(),0,"LLM/source HTML is text");
 assert(await b(0).evaluate(e=>{const body=e.querySelector(".reader-content-block-body"),h=e.querySelector(".learning-vocabulary"),insight=e.querySelector(".reader-content-block-inline-insight");return !!(body.compareDocumentPosition(h)&Node.DOCUMENT_POSITION_FOLLOWING)&&!!(h.compareDocumentPosition(insight)&Node.DOCUMENT_POSITION_FOLLOWING);}),"source/translation then vocabulary then Insight");
 assert(await b(0).locator(".learning-sentence-pair").evaluateAll(es=>es.every(e=>e.children[0].classList.contains("learning-source-sentence")&&e.children[1].classList.contains("learning-translated-sentence"))),"alternating rows");
 const exact=b1.sentences.map(s=>s.text).join("");assert.equal(await b(0).evaluate(e=>[...e.querySelectorAll(".learning-source-sentence,.learning-source-gap")].map(e=>e.textContent).join("")),exact,"source gaps retained");
 assert(await b(1).locator(".learning-source-sentence").first().evaluate(e=>e.scrollWidth<=e.clientWidth+1),"long source wraps");
 await b(0).locator(".learning-source-sentence").first().evaluate(e=>window.keptSentence=e);
 const action=()=>b(0).getByRole("button",{name:"Refresh translation",exact:true});await action().waitFor();await until(async()=>await action().isEnabled());
 failure="failed";hold=true;await action().click();await until(()=>release!==null);assert.equal(await b(0).locator(".learning-translated-sentence").count(),2,"saved while generating");
 const scroller=page.locator(".reader-content-block-list"),scroll=await scroller.evaluate(e=>e.scrollTop);release();release=null;
 const retry=b(0).getByRole("button",{name:"Retry translation",exact:true});await retry.waitFor();await until(async()=>await retry.isEnabled());
 assert.equal(await b(0).locator(".learning-source-sentence").first().evaluate(e=>e===window.keptSentence),true,"source DOM survives refresh");
 assert(Math.abs(await scroller.evaluate(e=>e.scrollTop)-scroll)<3,"refresh retains scroll");assert((await b(0).innerText()).includes("saved result retained"));assert.equal(b1.components[1].saved.result_revision,1,"hints independent");
 await retry.focus();await page.keyboard.press("Enter");await action().waitFor();await until(async()=>await action().isEnabled());assert.equal(b1.components[0].saved.result_revision,2);assert.equal(gen().length,2);
 assert.equal(await b(0).locator('.learning-component-controls[data-component="translation"] .learning-component-action').evaluate(e=>document.activeElement===e),true,"action focus survives phase change");
 // Unknown refresh retains the saved text; Cancel-first modal, then one acknowledged retry.
 failure="unknown";await action().click();await retry.waitFor();await until(async()=>await retry.isEnabled());const unknownCommand=gen().at(-1).body.command_id;
 await retry.click();const dialog=page.getByRole("dialog",{name:"Retry translation?"});await dialog.waitFor();await until(async()=>await dialog.getByRole("button",{name:"Cancel",exact:true}).evaluate(e=>document.activeElement===e));
 await page.keyboard.press("Escape");await dialog.waitFor({state:"hidden"});assert.equal(gen().length,3);await retry.click();await dialog.getByRole("button",{name:"Retry with possible duplicate cost",exact:true}).click();await action().waitFor();await until(async()=>await action().isEnabled());
 assert.equal(gen().length,4);const cmd=gen().at(-1).body;assert.equal(cmd.intent,"retry");assert.equal(cmd.retry_operation.operation_id,unknownCommand);assert.notEqual(cmd.command_id,unknownCommand);assert.equal(cmd.acknowledge_possible_duplicate_cost,true);
 // Read failure retains results and requires a read check, never a paid replay.
 failure="read";await action().click();const check=b(0).getByRole("button",{name:"Check progress for translation",exact:true});await check.waitFor();await until(async()=>await check.isEnabled());assert.equal(gen().length,4);assert.equal(await b(0).locator(".learning-translated-sentence").count(),2);await check.click();await action().waitFor();
 await scroller.evaluate(e=>e.scrollTop=0);
 await page.screenshot({path:`/tmp/llrn19-display-${width}.png`,fullPage:true});
 await b(0).screenshot({path:`/tmp/llrn19-display-block-${width}.png`});
 assert.equal(calls.filter(c=>c.endpoint==="ordinary-read").length,ordinaryCount,"no ordinary translation reads in learning mode");
 await changeMode("normal");await b(0).locator(".reader-inline-translation-text").waitFor();assert.equal(await page.locator(".learning-sentence-pairs").count(),0,"no duplicated sources");
 await openMenu();await page.getByRole("menuitem",{name:"View translations",exact:true}).waitFor();await page.keyboard.press("Escape");
 await changeMode("language_learning");await retry.waitFor({state:"hidden"});await action().waitFor();assert.equal(await page.locator(".reader-inline-translation-text").count(),0);assert.equal(gen().length,4);
 const highlights=await page.evaluate(async ({book,b})=>{
  const {default:React}=await import("/node_modules/.vite/deps/react.js");
  const {default:Client}=await import("/node_modules/.vite/deps/react-dom_client.js");
  const {default:DOM}=await import("/node_modules/.vite/deps/react-dom.js");
  const {LearningContext}=await import("/src/features/learning-reader/context.ts");
  const {LearningBlockBody}=await import("/src/features/learning-reader/block-view.tsx");
  const {itemKey}=await import("/src/features/learning-reader/model.ts");
  const host=document.createElement("div");document.body.append(host);const root=Client.createRoot(host);
  const value={book,controller:{state:{blocks:{[b.content_block_id]:b},items:Object.fromEntries(b.components.map(view=>[itemKey(b.content_block_id,view.component),{view}]))}}};
  const block={block_id:b.content_block_id,block_type:"content",content:b.sentences.map(s=>s.text).join(""),ownerContext:{documentId:"Book",taskUnitId:"u1"}};
  const render=highlighted=>DOM.flushSync(()=>root.render(React.createElement(LearningContext.Provider,{value},React.createElement(LearningBlockBody,{block,highlighted}))));
  render(true);const original=[...host.querySelectorAll("mark")].map(e=>e.textContent).join("");const translated=host.querySelector(".learning-translated-sentence");
  const noTranslationMark=!host.querySelector(".learning-translated-sentence mark");render(false);
  const stable=translated===host.querySelector(".learning-translated-sentence")&&!host.querySelector("mark");root.unmount();host.remove();
  return {source:original===block.content,noTranslationMark,stable};
 },{book:settings(),b:b1});assert.deepEqual(highlights,{source:true,noTranslationMark:true,stable:true});
 // Filter assessments survive saved reads and distinguish no valid proposals from no needed hints.
 b1.components.find(v=>v.component==="hints").saved.assessment={candidate_count:3,accepted_count:1,filtered_count:1,rejected_count:1,omitted_sentences:0,issues:[{code:"invalid_source",count:1},{code:"known_level",count:1}]};
 b2.components.find(v=>v.component==="hints").saved.assessment={candidate_count:2,accepted_count:0,filtered_count:0,rejected_count:2,omitted_sentences:0,issues:[{code:"invalid_source",count:2}]};
 await changeMode("normal");await changeMode("language_learning");
 await until(async()=>await page.locator(".learning-vocabulary-omissions").count()===2);
 assert.equal(await b(0).locator(".learning-hint").count(),1,"valid hints retained");
 assert.equal(await page.locator(".learning-vocabulary-empty").count(),0,"invalid candidates are not a successful no-hints conclusion");
 const selection=b(1).locator(".learning-vocabulary-selection-toggle");await selection.focus();await page.keyboard.press("Enter");
 assert(await b(1).locator(".learning-vocabulary-selection-details").evaluate(e=>e.open),"keyboard filtering details");
 await selection.scrollIntoViewIfNeeded();await page.screenshot({path:`/tmp/vocabulary-filter-${width}.png`,fullPage:true});
 const beforeReload=gen().length;await page.reload();await page.locator(".hierarchy-section-button").filter({hasText:"Section"}).click();
 await until(async()=>await page.locator(".learning-vocabulary-omissions").count()===2);assert.equal(gen().length,beforeReload,"filtered results reuse without automatic regeneration");
 assert.deepEqual(errors,[]);assert.deepEqual(unexpected,[]);console.log(JSON.stringify({width,pass:true,generatedCommands:gen().length,normalPreferenceRestored:true}));await page.close();
}}finally{await browser.close();}
