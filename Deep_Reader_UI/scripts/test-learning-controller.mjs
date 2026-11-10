import assert from "node:assert/strict";
const { chromium } = await import(process.env.PLAYWRIGHT_MODULE || "playwright");
const browser = await chromium.launch({ headless: true, ...(process.env.CHROMIUM_PATH ? { executablePath: process.env.CHROMIUM_PATH } : {}) });
try {
 const page = await browser.newPage();
 await page.route("**/api/**", r => r.fulfill({ json: { total: 0, items: [] } }));
 await page.goto(process.env.UI_URL || "http://127.0.0.1:5173");
 const results = await page.evaluate(async () => {
  const { LearningPageSession } = await import("/src/features/learning-reader/session.ts");
  const { LearningSessionMemory, itemKey, pageSources, pageProgress, emptyPage } = await import("/src/features/learning-reader/model.ts");
  const { LearningService, LearningFailure, validRead, validOperation } = await import("/src/services/LearningService.ts");
  const f = await import("/scripts/learning-controller-fixtures.mjs");
  const check = (v, message) => { if (!v) throw Error(message); };
  const pause = ms => new Promise(r => setTimeout(r, ms));
  const until = async fn => { for (let i = 0; i < 300; i++) { if (fn()) return; await pause(2); } throw Error("condition timeout"); };
  const tests = [], sessions = [];
  const test = async (name, fn) => { try { await fn(); tests.push({ name, pass: true }); } catch(e) { tests.push({ name, error: e.stack }); }
   finally { sessions.splice(0).forEach(s => s.stop()); } };
  const copy = v => structuredClone(v);
  function harness(blocks = [f.block()], opts = {}) {
   let state; const calls = [], ops = new Map(), commands = new Map(); const settings = opts.book || f.book();
   const api = new LearningService({ requestJsonWithStatus: async (path, init) => {
    const body = JSON.parse(init.body), endpoint = path.split("/learning/")[1]; calls.push({ endpoint, body, signal: init.signal });
    const custom = await opts.respond?.(endpoint, body, { blocks, ops, commands, settings });
    if (custom !== undefined) return custom;
    let payload;
    if (endpoint === "read") payload = { doc_name: settings.doc_name, settings, blocks: body.content_block_ids.map(id => blocks.find(b => b.content_block_id === id)) };
    else if (endpoint === "generate") { const b = blocks.find(b => b.content_block_id === body.content_block_id); const op = f.operation(b, body.component, "succeeded", body.command_id);
     ops.set(body.command_id, op); f.save(b, body.component); f.attach(b, op); payload = f.submission(body); commands.set(body.command_id, payload); }
    else if (["operations/read", "operations/wait"].includes(endpoint)) payload = ops.get(body.handle.operation_id);
    else if (endpoint === "commands/read") payload = { doc_name: body.doc_name, command_id: body.command_id, found: commands.has(body.command_id), receipt: commands.get(body.command_id) || null };
    else throw Error(`Unexpected ${endpoint}`);
    return { ok: true, status: 200, payload: copy(payload) };
   } });
   const memory = opts.memory || new LearningSessionMemory();
   const session = new LearningPageSession(copy(settings), opts.sources || blocks.map(b => ({ id: b.content_block_id, text: b.sentences.map(s => s.text).join("") })), memory,
    value => { state = value; opts.publish?.(value); }, api, { concurrency: 2 }); sessions.push(session);
   return { session, calls, ops, commands, memory, settings, state: () => state, item: (id = "b1", c = "translation") => state?.items[itemKey(id,c)],
    count: endpoint => calls.filter(c => c.endpoint === endpoint).length };
  }
  await test("source identity and DTOs fail closed, code-point spans", async () => {
   const b = f.block("b1", "😀 Hello."); const read = { doc_name: "Book", settings: f.book(), blocks: [b] };
   check(validRead(read), "valid Unicode source"); b.sentences[0].end++; check(!validRead(read), "UTF16 mismatch");
   for (const change of [b => b.source_token = "bad", b => b.components[0].state = "ready", b => b.components.push(b.components[0])]) {
    const b = f.block(); change(b); check(!validRead({ ...read, blocks: [b] }), "invalid response accepted"); }
   let rejected = false; try { pageSources("Other", [{block_id:"b1",block_type:"content",content:"text",ownerContext:{documentId:"Book",taskUnitId:"u1"}}]); } catch {rejected=true;} check(rejected,"foreign page");
   const op=f.operation(f.block(),"translation"); check(validOperation(op),"operation"); check(!validOperation({...op,result_revision:1}),"uncommitted revision");
  });
  await test("backend block identities govern learning, not presentation types", async () => {
   const blocks=["paragraph","content","quote",null,undefined].map((block_type,i)=>({block_id:`b${i}`,block_type,content:`Source ${i}.`,ownerContext:{documentId:"Book",taskUnitId:"u1"}}));
   const sources=pageSources("Book",blocks);
   check(JSON.stringify(sources)===JSON.stringify(blocks.map(b=>({id:b.block_id,text:b.content}))),"all addressed source blocks retained");
   for(const invalid of [
    [...blocks,{...blocks[0]}],
    [{...blocks[0],block_id:""}],
    [{...blocks[0],ownerContext:{documentId:"Book",taskUnitId:""}}],
    [{...blocks[0],ownerContext:{documentId:"Other",taskUnitId:"u1"}}],
   ]) {
    let rejected=false;try{pageSources("Book",invalid);}catch{rejected=true;}
    check(rejected,"invalid paragraph identity must fail closed");
   }
  });
  await test("read before generate, independent outputs, committed readback, dedup", async () => {
   const h=harness(); await h.session.start(); h.session.start();
   h.session.enqueue(itemKey("b1","translation"),"refresh");
   await until(()=>h.item()?.phase==="ready" && h.item("b1","hints")?.phase==="ready");
   check(h.count("generate")===2,"duplicate dispatch"); check(h.calls[0].endpoint==="read","read first");
   check(h.item("b1","hints").view.saved.items.length===0,"empty hints ready"); check(h.memory.commands.size===0,"resolved commands released");
   const c=h.calls.find(c=>c.endpoint==="generate").body;
   check(c.expected_source_token==="a".repeat(64) && c.expected_book_revision===1 && c.expected_workspace_revision===1,"binding");
   check(!("text" in c)&&!("target_language" in c)&&c.intent==="get_or_generate","no UI source authority");
  });
  await test("saved results and blank source never generate", async () => {
   const b=f.block(); f.save(b,"translation"); f.save(b,"hints"); const h=harness([b,f.block("blank","  \n")]); await h.session.start(); await pause(20);
   check(h.count("generate")===0,"saved/blank generation"); check(h.item("blank")?.phase==="empty","blank state");
  });
  await test("malformed/read failure/settings/source mismatch never mean missing", async () => {
   for(const kind of ["http","malformed","settings","source","foreign"]) {
    const h=harness([f.block()],{respond:(e,b,{settings,blocks})=> e!=="read"?undefined:kind==="http"?{ok:false,status:503,payload:{}}:{ok:true,status:200,payload:
      kind==="malformed"?{}:{doc_name:"Book",settings:kind==="settings"?{...settings,revision:2}:settings,blocks:kind==="source"?[f.block("b1","Changed.")]:kind==="foreign"?[f.block("foreign")]:blocks}}});
    await h.session.start(); await pause(5); check(h.state().error && h.count("generate")===0,kind);
   }
  });
  await test("all read chunks validated before any generation", async () => {
   const blocks=Array.from({length:51},(_,i)=>f.block(`b${i}`)); blocks.forEach(b=>{f.save(b,"translation");f.save(b,"hints");});
   const h=harness(blocks); await h.session.start(); check(h.count("read")===2,"read chunk count"); check(h.calls[0].body.content_block_ids.length===50 && h.calls[1].body.content_block_ids.length===1,"read bounds");check(h.count("generate")===0,"saved batch");
   const bad=harness(blocks,{respond:(e,b)=>e==="read"&&b.content_block_ids[0]==="b50"?{ok:false,status:503,payload:{}}:undefined}); await bad.session.start();check(bad.count("generate")===0,"partial read should not start page");
  });
  await test("two workflow ceiling, abort stops queue and late state", async () => {
   const releases=[]; let published=0;
   const h=harness([f.block(),f.block("b2"),f.block("b3")],{publish:()=>published++, respond:async(e,b)=>{if(e==="generate"){await new Promise(r=>releases.push(r));return {ok:true,status:200,payload:f.submission(b,"unknown",null)};}}});
   await h.session.start(); await until(()=>releases.length===2);check(h.count("generate")===2,"pool limit");const before=published;h.session.stop();releases.forEach(r=>r());await pause(20);
   check(h.count("generate")===2&&published===before,"abort suppresses queue/late update");check(h.calls.filter(c=>c.endpoint==="generate").every(c=>c.signal.aborted),"owned abort");check(h.memory.commands.size===2,"retain unresolved identities");
  });
  await test("lost acknowledgment survives reopen; missing receipt is not resent", async () => {
   const b=f.block();f.save(b,"hints"); const memory=new LearningSessionMemory();
   const first=harness([b],{memory,respond:e=>e==="generate"?Promise.reject(new TypeError("lost connection")):undefined});
   await first.session.start();await until(()=>first.item()?.phase==="unknown"); first.session.stop();
   const next=harness([b],{memory});await next.session.start();await until(()=>next.item()?.message.includes("no confirmed receipt"));
   check(next.count("commands/read")===1 && next.count("generate")===0,"unknown auto retry");
   next.session.enqueue(itemKey("b1","translation"),"check");await until(()=>next.count("commands/read")===2);check(next.count("generate")===0,"manual check resends");
  });
  await test("receipt is acceptance only; actual operation and committed read decide ready", async () => {
   const b=f.block();f.save(b,"hints");const h=harness([b],{respond:e=>e==="generate"?Promise.reject(new Error("lost")):undefined});await h.session.start();await until(()=>h.item()?.phase==="unknown");
   const cmd=h.calls.find(c=>c.endpoint==="generate").body,op=f.operation(b,"translation","succeeded",cmd.command_id);h.ops.set(cmd.command_id,op);h.commands.set(cmd.command_id,f.submission(cmd));f.save(b,"translation");f.attach(b,op);
   h.session.enqueue(itemKey("b1","translation"),"check");await until(()=>h.item()?.phase==="ready");check(h.count("generate")===1&&h.count("commands/read")===1&&h.count("operations/read")===1,"receipt recovery path");
  });
  await test("one completion wait; partial is checkpoint only; manual reconcile commits", async () => {
   const b=f.block();f.save(b,"hints");let op=f.operation(b,"translation","partial");f.attach(b,op);
   const h=harness([b],{respond:(e,body)=>{
    if(e==="operations/partial")return{ok:true,status:200,payload:{handle:op.handle,component:"translation",source_token:b.source_token,offset:0,completed_batches:1,total_batches:1,next_offset:null,batches:[{component:"translation",batch_index:0,rows:[{sentence_id:"b1:s1",translated_text:"你好。"}]}]}};
    if(e==="operations/reconcile"){op=f.operation(b,"translation","succeeded",op.handle.operation_id);h.ops.set(op.handle.operation_id,op);f.save(b,"translation");f.attach(b,op);return{ok:true,status:200,payload:f.submission({doc_name:"Book",command_id:op.handle.operation_id},"succeeded",op.handle)};}
   }});h.ops.set(op.handle.operation_id,op);await h.session.start();await until(()=>h.item()?.message.includes("Still in progress"));
   check(h.count("operations/wait")===1&&h.count("operations/read")===0&&!h.item().view.saved&&h.item().partial.batches.length===1,"partial not saved");await pause(20);check(h.count("operations/wait")===1&&h.count("operations/read")===0,"no periodic reads");
   check(pageProgress(h.state(),2).blocked&&pageProgress(h.state(),2).checking===1,"wait timeout must retain forward guard");
   h.session.enqueue(itemKey("b1","translation"),"check");await until(()=>h.item()?.phase==="ready");check(h.count("operations/reconcile")===1&&h.count("generate")===0&&!h.item().partial,"safe commit");
  });
  await test("unknown retry requires consent, fresh command, bound operation", async () => {
   const b=f.block();f.save(b,"hints");const op=f.operation(b,"translation","unknown");f.attach(b,op);const h=harness([b]);h.ops.set(op.handle.operation_id,op);await h.session.start();await until(()=>h.count("operations/wait")===1);await pause(5);
   h.session.enqueue(itemKey("b1","translation"),"retry");await until(()=>h.item()?.message.includes("confirmation"));check(h.count("generate")===0,"no consent");await pause(5);
   h.session.enqueue(itemKey("b1","translation"),"retry",true);await until(()=>h.item()?.phase==="ready");const cmd=h.calls.find(c=>c.endpoint==="generate").body;
   check(cmd.intent==="retry"&&cmd.retry_operation.operation_id===op.handle.operation_id&&cmd.command_id!==op.handle.operation_id&&cmd.acknowledge_possible_duplicate_cost,"retry binding");
  });
  await test("one component failure preserves other success and prior saved refresh", async () => {
   const b=f.block();const h=harness([b],{respond:(e,body)=>e==="generate"&&body.component==="translation"?{ok:false,status:503,payload:{detail:{kind:"unavailable",code:"admission_closed"}}}:undefined});
   await h.session.start();await until(()=>h.item()?.phase==="error"&&h.item("b1","hints")?.phase==="ready");check(h.count("generate")===2,"component isolation");
   const b2=f.block();f.save(b2,"translation");f.save(b2,"hints");const op=f.operation(b2,"translation","failed");f.attach(b2,op);const old=harness([b2]);old.ops.set(op.handle.operation_id,op);await old.session.start();await until(()=>old.item()?.phase==="failed");check(old.item().view.saved.result_revision===1&&old.count("generate")===0,"retained saved");
  });
  await test("explicit refresh of succeeded operation sends one new regenerate", async () => {
   const b=f.block();f.save(b,"translation");f.save(b,"hints");const op=f.operation(b,"translation","succeeded");f.attach(b,op);const h=harness([b]);h.ops.set(op.handle.operation_id,op);await h.session.start();await until(()=>h.item()?.phase==="ready");await pause(10);
   h.session.enqueue(itemKey("b1","translation"),"refresh");h.session.enqueue(itemKey("b1","translation"),"refresh");await until(()=>h.count("generate")===1);await until(()=>h.item()?.phase==="ready");
   check(h.calls.find(c=>c.endpoint==="generate").body.intent==="regenerate","refresh intent");
  });
  await test("foreign operation and false commit cannot become ready", async () => {
   for(const kind of ["foreign","commit"]) {
    const b=f.block();f.save(b,"hints");const op=f.operation(b,"translation","succeeded");f.attach(b,op);if(kind==="foreign")op.content_block_id="Other";
    const h=harness([b]);h.ops.set(op.handle.operation_id,op);await h.session.start();await until(()=>h.item()?.phase==="error");check(!h.item().view.saved&&h.count("generate")===0,"operation/result verification");
   }
  });
  await test("saved result survives settings roundtrip and refresh uses current revisions", async () => {
   const b=f.block();f.save(b,"translation");f.save(b,"hints");
   const op=f.operation(b,"translation","succeeded");f.attach(b,op);
   // The saved read is current; only the completed operation's old settings are stale.
   op.state="stale";op.settings_current=false;
   const book=f.book();book.revision=3;book.workspace.revision=2;
   const h=harness([b],{book});h.ops.set(op.handle.operation_id,op);
   await h.session.start();await until(()=>!h.item()?.busy);
   check(h.item()?.phase==="ready"&&h.count("generate")===0&&h.count("operations/read")===0,"historical operation invalidated reusable saved result");
   h.session.enqueue(itemKey("b1","translation"),"refresh");
   await until(()=>h.count("generate")===1);await until(()=>h.item()?.phase==="ready"&&!h.item()?.busy);
   const command=h.calls.find(c=>c.endpoint==="generate").body;
   check(command.intent==="regenerate"&&command.expected_book_revision===3&&command.expected_workspace_revision===2,"refresh used historical settings");
  });
  await test("same-id source/settings changes prevent refresh, retain old saved", async () => {
   for(const kind of ["source","settings"]) {
    const b=f.block();f.save(b,"translation");f.save(b,"hints");const h=harness([b]);await h.session.start();await pause(10);
    if(kind==="source")b.source_token="b".repeat(64);else h.settings.revision++;
    h.session.enqueue(itemKey("b1","translation"),"refresh");await until(()=>h.item()?.phase==="error");check(h.count("generate")===0&&h.item().view.saved,"changed binding");
   }
  });
  await test("overall completion distinguishes active, terminal and uncertain work", async () => {
   check(pageProgress(emptyPage(),4).blocked,"initial read must guard navigation");
   const b=f.block(), view=b.components[0];
   const progress=(phase,extra={})=>pageProgress({loading:false,error:"",blocks:{},items:{one:{blockId:"b1",component:"translation",phase,view,message:"",...extra}}},1);
   for(const phase of ["queued","reading","pending","partial"])check(progress(phase).blocked,phase);
   for(const phase of ["failed","stale","unavailable","error"])check(!progress(phase).blocked&&progress(phase).percent===0,phase);
   check(progress("ready").percent===100&&progress("empty").percent===100,"saved/no-content completion");
   check(progress("ready",{busy:true}).blocked&&progress("ready",{busy:true}).percent===0,"refresh retains old saved, not completion");
   check(progress("error",{awaitingCompletion:true}).blocked,"wait failure cannot prove job stopped");
   check(!progress("error",{awaitingCompletion:false,view:{...view,operation_state:"running"}}).blocked,"confirmed completion releases guard even if result read fails");
   check(progress("unknown").blocked,"lost receipt needs confirmation");
   check(!progress("unknown",{view:{...view,operation:{doc_name:"Book",operation_id:"stopped"},operation_state:"unknown"}}).blocked,"confirmed terminal unknown is not running");
  });
  await test("failed completion observation guards navigation until explicit confirmation", async () => {
   let fail=true;
   const h=harness([f.block()],{respond:async(endpoint)=>{
    if(endpoint==="operations/wait"&&fail){fail=false;return{ok:false,status:503,payload:{}};}
   }});
   await h.session.start();await until(()=>Object.values(h.state().items).every(i=>!i.busy));
   check(pageProgress(h.state(),2).blocked&&h.item().phase==="error","accepted request lost observation guard");
   const before=h.count("generate");h.session.enqueue(itemKey("b1","translation"),"check");
   await until(()=>!h.item().busy&&h.item().phase==="ready");
   check(!pageProgress(h.state(),2).blocked&&h.count("generate")===before,"check should unlock without regeneration");
  });
  await test("refresh is incomplete until terminal status, failed refresh unlocks", async () => {
   const b=f.block();f.save(b,"translation");f.save(b,"hints");let release, held=false;
   const h=harness([b],{respond:async(e,body)=>{
    if(e==="operations/wait"&&held){await new Promise(r=>release=r);return{ok:true,status:200,payload:f.operation(b,"translation","failed",body.handle.operation_id)};}
   }});
   await h.session.start();await until(()=>Object.values(h.state().items).every(i=>!i.busy));
   held=true;h.session.enqueue(itemKey("b1","translation"),"refresh");await until(()=>!!release);
   check(pageProgress(h.state(),2).blocked&&pageProgress(h.state(),2).completed===1&&h.item().view.saved,"saved refresh cannot count complete");
   release();await until(()=>!h.item().busy&&h.item().phase==="failed");
   const done=pageProgress(h.state(),2);check(!done.blocked&&done.attention===1&&done.completed===1,"known failure unlocks without fake 100 percent");
  });
  return tests;
 });
 console.log(JSON.stringify(results,null,2));assert(results.every(r=>r.pass),"Controller acceptance failed");
} finally {await browser.close();}
