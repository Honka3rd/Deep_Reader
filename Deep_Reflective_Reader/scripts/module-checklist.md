# scripts Checklist

## RETRIEVAL-03 support (2026-11-19)

- [x] Align ordinary output allowance and retrieval budgeting (scripts ownership).
  Acceptance: Public provider contract, target-preserving resolver, cold/warm endpoint and capacity regressions in the affected owner.
  Evidence: `/tmp/retrieval-unit3-51813cd/report.txt`; new 10/10, cumulative 188/188 zero skips; original external red/green and bounded immutable-book controls.
  Status: Locally verified supporting integration; external review pending.
  Timestamp: 2026-11-19


## Retrieval audit unit 2 (2026-11-19)

- [x] Verify source-free direct/QA search diagnostics with full returned hits.
  Acceptance: INFO/ERROR stdout/stderr/file captures, marker absence, content/prompt/call-count preservation, no-hit/ranking/neighbor/dedupe controls.
  Evidence: `/tmp/retrieval-unit2-51813cd/report.txt`; 13/13 new, scoped 74/74 zero skips, external red/green and 8 immutable-book sample flows.
  Status: Unit 2 locally verified; no scripts independent audit claim.
  Timestamp: 2026-11-19

## Retrieval audit unit 1 (2026-11-19)

- [x] Exercise ID integrity through actual FAISS read, candidate publication and runtime readiness.
  Acceptance: Original external red/green and new malformed/non-contiguous/journal/recovery controls; preserve existing PREP assertions and isolated data.
  Evidence: `/tmp/retrieval-unit1-51813cd/report.txt`; new 24/24 file/SQL, cumulative 78/78 zero skips; external 5/5 and full immutable-book controls.
  Status: Unit 1 locally verified; scripts independent audit remains exempt.
  Timestamp: 2026-11-19

## LLM-02 (2026-11-19)

- [x] Verify retired nano admission and preserved model/history contracts.
  Evidence: `/tmp/llm-fix-fb7f9dc/report.txt`; 7/7 admission, external 4/4,
  cumulative 105/105 zero skips; real asset/full immutable snapshot counting.
  Status: Locally verified; external review pending; scripts audit remains exempt.
  Timestamp: 2026-11-19


## LLM-01 (2026-11-18)

- [x] Reject incomplete or unavailable completion before successful publication.
  Evidence: `/tmp/llm-01-fb7f9dc/report.txt`; external provider/QA 5/5 and file
  summary 2/2 red-to-green; new 13/13; scoped regression 82/82, zero skips.
  Two immutable books: 24 bounded deterministic SDK flows, preserved hashes.
  Status: Locally verified; cumulative exact-SHA external review pending.
  Timestamp: 2026-11-19


## EMBED-01 (2026-11-18)

- [x] Verify real SDK INFO/DEBUG privacy, dispatch and error contracts in isolated sinks.
  Evidence: `/tmp/embed-01-121cc96/report.txt`; new SDK/sink suite 13/13,
  scoped formal regression 72/72, zero skips. External DEBUG red -> green,
  INFO control and SDK adapter 8/8; numeric 17/17 and caller 16/16 named controls.
  Status: Locally verified; exact-SHA external review required. No deployment/push.
  Timestamp: 2026-11-18

## SESSION-01 (2026-11-18)

- [x] Reproduce SESSION-01 with actual QA/ASGI and lifecycle controls.
  Evidence: `scripts/test_session_lifecycle.py` (12/12); cumulative scoped regression
  70/70, no skips. External races: 7 baseline failed subassertions -> 5/5 methods;
  lifecycle controls 5/5 with documented snapshot-read fixture adaptation.
  `/tmp/session-01-3dab1fa/report.txt`; original capacity assertions retained.
  Status: Locally verified; exact-SHA external review required; no deployment/push.
  Timestamp: 2026-11-18

## Apt retry memory correction (2026-11-18)

- [x] BUILD-CACHE-04: Upgrade apt/libapt before OCR downloads and enforce fixed versions.
  Acceptance: Actual RUN ordering/failure/version guards, preserved caches and
  registry models; isolated real-package upgrade/install and retry-memory evidence.
  Status: Locally verified; exact-SHA external review and BUILD-CACHE-02 remain required.
  Evidence: Final 16-test recipe suite: six baseline failed sub-assertions -> 16/16;
  ten original methods AST-identical. Support 71 executed: 68 pass/3 rebuilt-image
  opt-in skips. Actual RUN in isolated base upgrades both packages to 3.0.4, installs
  21 packaged OCR models, peak apt RSS 117648 kB. Cached-index fallback only;
  no online/full-image/deployment claim. `/tmp/apt-upgrade-2ec31af/report.txt`.
  Timestamp: 2026-11-18

## Manual Docker startup diagnostics (2026-11-18)

- [x] BUILD-CACHE-03: Make manual startup network/cache failures actionable.
  Acceptance: Help works without Docker/env setup; default plain progress and explicit
  override; distinct build/runtime proxy guidance; original exit status, startup
  ordering and no UI after failure; no secrets, automatic proxy changes or retries.
  Status: Locally verified; new exact-SHA review required. Existing approved image trial remains separate.
  Evidence: Launcher 16/16 (five old-launcher failures become green); original 11 methods AST-identical.
  Final support 42 executed: 39 pass/3 rebuilt-image opt-in skips; copied local asset verified offline.
  Shell/diff checks pass. `/tmp/docker-deploy-c35299a/retry/launcher-*.log`.
  Timestamp: 2026-11-18

## Build dependency cache isolation (2026-11-18)

- [x] BUILD-CACHE-01: Isolate dependency/resource inputs and preserve package download caches.
  Acceptance: Real registry manifest invariance/change/fail-closed tests; recipe graph,
  cache mount and final asset/verification tests; existing setup/OCR/startup controls.
  Scope: Dockerfile, regression scripts and owned delivery documents; no runtime mutation.
  Status: Source verified; external review and BUILD-CACHE-02 remain required.
  Evidence: New recipe/registry suite: 7 baseline failures + 3 passing projection controls -> 10/10. Setup/OCR/launcher support: 50 executions, 47 pass and 3 rebuilt-image opt-in skips after fixture correction; real local tokenizer asset verification runs offline. Original fixture assertions are AST-identical. Docker RUN shell syntax and changed Python AST checks pass. `/tmp/docker-cache-b4c8418/report.txt`. No actual image build/cache-hit/deployment result is claimed.
  Timestamp: 2026-11-18

- [ ] BUILD-CACHE-02: Verify approved image builds, cache reuse and offline deployment.
  Depends on BUILD-CACHE-01 and external approval of the exact latest SHA.
  Acceptance: Existing startup script rebuild, warm/application-only/dependency-change
  cache evidence, final venv/package/OCR/tokenizer checks, offline startup, existing
  data preservation and matching deployed revision. Static tests are insufficient.
  Timestamp: 2026-11-18

## Context auxiliary-interface audit unit (2026-11-18)

- [x] CONTEXT-02/03: Verify retained evidence and artifact signals with real helpers/services.
  Acceptance: Red/green provenance matrices, injected quiz/critical-thinking metadata,
  prior capacity/question/interaction controls and immutable-book samples.
  Status: Locally verified; external exact-SHA context review pending.
  Evidence: `scripts/test_context_retained_provenance.py`; Exact 9234471 baseline: new formal suite 9 methods /15 failed sub-assertions -> 9/9; unchanged external auxiliary suite 6 methods /3 failures -> 6/6. CONTEXT-01 formal 9/9, existing focused context/capacity 26/26 and question 40/40 pass. Zero errors/skips. Two complete immutable book copies: 8 bounded helper checks pass with canonical/raw/manifest hashes preserved. Groups overlap. `/tmp/context-unit2-9234471/report.txt`.
  Timestamp: 2026-11-18

## Context audit unit 1 (2026-11-18)

- [x] CONTEXT-01: Verify zero-budget helpers and exhausted QA refusal.
  Acceptance: Failing-before/passing-after actual helpers and coordinator paths,
  retained positive/context/question controls and immutable-book sample evidence.
  Status: Locally verified; external context review required; no deployment or push.
  Evidence: Exact 38d00dd baseline: new formal suite 9 methods /11 failed sub-assertions -> 9/9; external QA 3 methods /3 failures -> 3/3 with the documented early-capacity-error diagnostic adaptation; builders 7 methods /2 failures -> 7/7. Existing focused context/capacity 26/26 and question 40/40 pass. Zero errors/skips. Two complete immutable book copies: 4 bounded helper checks, 2 baseline zero-budget failures -> 4/4 with canonical/raw/manifest hashes preserved. Groups overlap. `/tmp/context-unit1-38d00dd/report.txt`.
  Timestamp: 2026-11-18

## Quoted term-role correction (2026-11-18)

- [x] QUESTION-08: Preserve target-specific term and meaning roles.
  Status: Locally verified; external exact-SHA question review remains open.
  Evidence: Exact 4763650 baseline: core external 3 methods /2 failed sub-assertions; supporting 4 methods /4 failed sub-assertions; fixed 3/3 and 4/4, zero errors/skips. Expanded requested-scope suite 15 methods /33 baseline failures -> 15/15; cumulative 12 suites 99/99, zero skips. Prior quoted-positive/extent/envelope 4/4 each; QUESTION-01–04 probes 6/4/5/4 and current-contract smoke 11/11 pass. Twenty-one bounded immutable-book flows pass with canonical/raw/manifest hashes preserved. Groups overlap. `/tmp/question-terms-4763650/report.txt`.
  Timestamp: 2026-11-18

## Quoted range correction (2026-11-18)

- [x] QUESTION-07: QUESTION-07 paired quote-role acceptance.
  Status: Locally verified; external exact-SHA question review remains open.
  Evidence: At exact 56bbadd baseline, external quoted-positive suite: 4 methods /2 failed sub-assertions, zero errors/skips; fixed 4/4. Expanded formal scope suite: 12 methods /23 baseline failures -> 12/12; cumulative 12 suites: 96/96, zero skips. QUESTION-05 extent and QUESTION-06 envelope 4/4 each; original QUESTION-01–04 probes 6/4/5/4 pass and current-contract smoke 11/11. Sixteen bounded immutable-book flows pass with full backup hashes unchanged. Overlapping groups are not unique-case totals. `/tmp/question-quoted-56bbadd/report.txt`.
  Timestamp: 2026-11-18

## Question follow-up (2026-11-18)

- [x] QUESTION-05/06: Verify extent and complete-envelope regressions with isolated fixtures.
  Status: Locally verified; exact-SHA external question review remains open.
  Evidence: Exact 57a7d76 baseline: new formal suite 9 methods / 53 failed sub-assertions / zero errors or skips; fixed 9/9. External extent 4 methods / 6 failures and envelope 4 methods / 2 failures become 4/4 each. Cumulative 12 formal suites: 93/93 executions, zero skips; original external lexical/standardizer/context/diagnostic 6/4/5/4 pass; fixed-method smoke 11/11. Actual envelope context probe and twelve bounded immutable-book flows pass; full backup hashes unchanged. Groups overlap and are not unique-case totals. `/tmp/question-followup-57a7d76/report.txt`.
  Timestamp: 2026-11-18

## Question audit unit 2 (2026-11-18)

- [x] QUESTION-03/04: Verify strict output admission and safe diagnostics through context and HTTP.
  Status: Locally verified; cumulative exact-SHA external question review pending.
  Acceptance: Invalid model matrices, no downstream dispatch, safe logs/errors/HTTP, multilingual controls, no retry and cumulative unit 1 regression.
  Evidence: New output-boundary suite: exact baseline 15 methods / 85 failed sub-assertions / zero errors or skips; corrected 15/15. Cumulative 11 formal suites: 84/84, zero skips. Original external lexical 6/6, standardizer 4/4, context 5/5 and diagnostic/ASGI route 4/4 pass. Original offline smoke retains two failures in one method expecting free-form model reason; a separately recorded copy changes only the method expectation plus output directory and passes 11/11, retaining routing/call/budget assertions. Full immutable book hashes unchanged; ten bounded deterministic routing/admission/diagnostic flows pass. Evidence groups overlap; not unique-case totals. `/tmp/question-unit2-252db05/report.txt`.
  Timestamp: 2026-11-18

## Question audit unit 1 (2026-11-17)

- [x] QUESTION-01/02: Verify scope red/green and actual context routing.
  Status: Locally verified; external question review pending.
  Evidence: New routing suite: baseline 9 methods / 50 failed sub-assertions / zero errors or skips; fixed 10/10 (one new extent-registry interface test). External lexical 6/6; external context routing 4/4 methods, with the separate invalid-standardization method retaining five QUESTION-03 failures. Ten formal suites: 69/69 executions, zero skips; offline smoke 11/11 and standalone context-builder CLI pass. Two hash-verified complete book copies supply four bounded routing flows; backup hashes unchanged. Evidence groups overlap and are not summed as unique cases. `/tmp/question-unit1-252db05/report.txt`.
  Timestamp: 2026-11-18

## LANG-03 support (2026-11-17)

- [x] LANG-03: Verify script filtering, persistence and policy isolation on file and real PostgreSQL.
  Status: Locally verified; external language review pending.
  Evidence: New script boundary suite: baseline 10 methods / 7 failed assertions / zero errors or skips; fixed 10/10 including real file/PostgreSQL persistence. Cumulative 23 suites: 470/470 executions, zero skips (overlapping inherited scenarios are not unique business cases). External persistence probe and 11 controls pass with documented observation-only harness changes. Both immutable books retain hashes; bounded English negative / Chinese positive samples pass four file/SQL flows. `/tmp/lang-unit2-847f0d6/report.txt`. External exact-SHA language review remains pending.
  Timestamp: 2026-11-17

## Language audit unit 1 (2026-11-17)

- [x] LANG-01/02: Reject unknown detector authority and avoid content-bearing diagnostics.
  Status: Locally verified; language external review pending.
  Acceptance: Normalization/Prepare/retrieval and captured-log red/green; preserve accepted data and warm reuse, no automatic retry.
  Evidence: New boundary suite 13/13 (real file/PostgreSQL); external detector 4/4 and normalization 3/3; adjacent/focused total 207/207, zero skips after documented runner cwd correction. Two full hash-verified book copies across file/SQL retain accepted assets under three invalid rebuild replies. `/tmp/lang-unit1-847f0d6/report.txt`. Unit 1 evidence retained; LANG-03 now locally verified, external language approval pending.
  Timestamp: 2026-11-17


## Learning public recovery audit (2026-11-17)

- [x] LEARN-03: Preserve known recovery failures across the public contract.
  Status: Locally verified; exact-SHA external learning review pending.
  Acceptance: Both failure codes observable, strict invalid-response rejection, saved/history preserved, explicit retry only; real file/SQL HTTP and UI consumer evidence.
  Evidence: Public recovery 4/4 file/SQL methods; external 28 failing assertions now pass; cumulative backend 534/534, typed payloads 16/16, controller 22/22, desktop/mobile display/lifecycle and type/build pass. `/tmp/learn03-fix-cafc4ab/report.txt`.
  Timestamp: 2026-11-17


## Learning audit unit 2 (2026-11-15)

- [x] LEARN-02: Verify conditional publication of late replies after expiry observation.
  Status: Locally verified; exact-SHA external review pending.
  Acceptance: File/real SQL both-component exact-once checkpoint, no extra provider dispatch, current-source/fence/CAS and unknown-ack controls.
  Evidence: Expiry regression 26/26; cumulative 474/474; external expiry 4/4 and guards 8/8; two hash-verified book samples/8 file-SQL flows. `/tmp/learn02-fix-471db7b/report.txt`.
  Timestamp: 2026-11-16


## Learning audit (2026-11-15)

- [x] LEARN-01: Guard partial recovery provenance and provide explicit recovery for unassemblable complete checkpoints.
  Status: Locally verified; external learning review pending.
  Acceptance: Changed generation identity performs zero recovery dispatch; valid complete checkpoints finish offline; explicit retry preserves history and old saved; real file/PostgreSQL controls.
  Evidence: 22/22 new recovery tests, 370/370 adjacent executions, 56/56 routes, 8/8 external guards; corrected external matrix 4/4, two immutable-book excerpts/8 flows. `/tmp/learn01-fix-471db7b/report.txt`.
  Timestamp: 2026-11-15

## TASK-05 supporting contract (2026-11-15)

- [x] TASK-05: Validate legacy summary/quiz outputs without masking corruption or losing valid prior artifacts.
  Acceptance: Strict complete-result validation, compatible read-only caches, actual file/SQL HTTP and retained source guards.
  Evidence: New file/real SQL HTTP suite 22/22 (original 18-method red subset: 133 failed assertions, zero errors/skips); external five observational cases satisfy desired assertions after absence-safe diagnostic adaptation. Adjacent suites 114/114 after one fixture setup correction; TASK-04 20/20, prior focused 229 pass/one skip, app 529 pass/six skips, translation/storage 404/404. Two full immutable backups hash-verified; bounded first-2000-character section/chapter HTTP on both stores preserves valid outputs on rejected refresh. `/tmp/task05-fix-3f85191/report.txt`. Local verification only; exact-SHA external section_tasks review required.
  Timestamp: 2026-11-15

## Section tasks audit unit 4 (2026-11-15)

- [x] TASK-04: Support exact explicit compatibility translation targets without duplicate enumeration.
  Acceptance: file/real SQL HTTP exact source and saved reuse; source lifecycle/CAS protection, read-only behavior and unchanged primary pagination.
  Evidence: New exact-target suite 20/20 file/real SQL; original 12-method subset red 24 assertions/zero errors. External four default assertions red -> 2/2 methods green. Translation/storage 404/404; prior focused 229 pass/one skip; app 529 pass/six retained opt-in skips. Two hash-verified complete backups with bounded 2000-character HTTP samples pass. `/tmp/task04-fix-3f85191/report.txt`. Locally verified; section_tasks external gate remains open.
  Timestamp: 2026-11-15

## Section tasks audit unit 3 (2026-11-15)

- [x] TASK-03: Verify Summary/Note optional block evidence across file/real SQL HTTP workflows.
  Acceptance: absent/null/valid distinct/stale evidence, no provider or persistence on rejection, positive complete workflows and preserved source/lifecycle/editor guards.
  Evidence: New file/real SQL HTTP 14/14; eight baseline assertions red/zero errors; external four failures -> 2/2 green. Final focused 229 pass/one skip; app 529 pass/six skips; two immutable full-source copies with bounded block evidence pass, hashes unchanged. `/tmp/task03-fix-3f85191/report.txt`. Local unit verification only; section_tasks external gate remains open.
  Timestamp: 2026-11-15

## Section tasks audit unit 2 (2026-11-15)

- [x] TASK-02: Verify durable current-session selection independently of JSON key ordering.
  Acceptance: file/real SQL HTTP current/history/availability, old-answer retry, atomic creation/rollback, target isolation and legacy read-only behavior.
  Evidence: New current-session suite 18/18 file/real SQL; original 16-method subset red 12 assertions/zero errors; external SQL failure -> 2/2 file/SQL green. Focused 215 pass/one private opt-in skip; app 529 pass/six retained skips; full backup current/history checks pass with unchanged hashes. `/tmp/task02-fix-3f85191/report.txt`. Unit 2 locally verified; external module review remains open.
  Timestamp: 2026-11-15

## Section tasks audit unit 1 (2026-11-14)

- [x] TASK-01: Verify TASK-01 lifecycle matrices using real file and PostgreSQL persistence.
  Acceptance: exact source/incarnation/rename/revision checks at locked publication; positive reuse/new generation, notes/receipt/CAS preservation, rollback and delayed result matrices.
  Evidence: New lifecycle suite 28/28 file/SQL; archive matrix red 68 assertions/zero errors; external 10+3+3 assertions red -> 4+3+3 methods green. Final focused 197 pass/one skip; app 529 pass/six skips; interaction 178 pass/one skip; full immutable-source file/SQL checks. `/tmp/task01-fix-3f85191/report.txt`. Unit 1 locally verified, module external review remains open.
  Timestamp: 2026-11-14


- [x] Reconcile section_tasks audit fixtures without weakening product assertions.
  Acceptance: Safe evaluation error classification preserves answer/identity/metadata/provider count; private section-7 fixture has cwd-independent location and retains all split assertions; source bytes unchanged.
  Evidence: Output language 12/12; splitting 24/24 actual functions, zero-discovery integration entrypoints excluded. Private fixture hash unchanged; `/tmp/task01-fix-3f85191/split-fixture-after.json` and `report.txt`.
  Timestamp: 2026-11-14

## APP-06 correction (2026-11-14)

- [x] APP-06 regression: Verify default/segmented exact-source mapping and retained guards.
  Acceptance: file/real SQL single/batch HTTP exact source/hash, availability isolation, custom/heading/short paragraphs and retained lifecycle/CAS guards; original app regressions and explicit skips.
  Evidence: external mapping red four assertions / green eight subcases; new mapping 20/20 plus expanded custom rerun 2/2; archive matrix red 24 assertions/zero errors. Preserved targets 15/15, focused 77/77, interaction 96/96, mapping support 118/118, app 321/327 with six opt-in skips. Two immutable backups: bounded first-five-unit mapping and eight file/SQL default/segmented session checks. `/tmp/app06-fix-eaae891/report.txt`; locally verified, external app gate remains open.
  Timestamp: 2026-11-14

## App audit regression support (2026-11-14)

- [x] APP-05 regression: cover five HTTP target levels and optional block evidence across file/real PostgreSQL.
  Acceptance: reproduce the unchanged-source SQL rejection, verify correction, durable retry and locked source/CAS guards; retain 77 focused, 96 adjacent and 327 app executions with explicit skips.
  Evidence: test_critical_thinking_target_publication 15/15; ten initial methods reproduce eight SQL assertion failures without errors/skips. Final 77/77 focused, 96/96 interaction, app 321/327 with six existing private opt-in skips/zero failures or errors, external block HTTP 2/2 and original app 3/3. `/tmp/app05-fix-b8f5b94/report.txt`; external approval pending.
  Timestamp: 2026-11-14

- [x] Preserve app source/session publication regressions and reconcile current hierarchy/read-only fixtures.
  Acceptance: real file/SQL/HTTP races and failures, source/answer preservation, short mutations, fixture intent retained and explicit opt-in skip accounting.
  Evidence: focused 77/77; original app 327 executions with 321 passes/six private-source opt-in skips/zero failures or errors; related interaction 96/96. Seven formerly failing suites pass isolated reruns. Two complete immutable books pass source/session checks on file/SQL with unchanged backups and no live provider calls. `/tmp/app-fix-5b3291a3/final-report.txt`; support only, app external review pending.
  Timestamp: 2026-11-14

## DOCLOAD deployment fixture follow-up (2026-11-13)

- [ ] Align synthetic Outline fixtures and disabled-OCR tests with current contracts.
  Acceptance: Matching standalone native/scanned headings, unchanged strict Outline/transcript/span/cache assertions, old inline-heading rejection, and explicit disabled-OCR configuration independent of deployment environment. Real-engine synthetic acceptance must pass after exact-SHA review.
  Evidence: Approved `8569fb1` Docker acceptance fails at the mixed Outline parser assertion; diagnosis and preserved-data checks in `/tmp/deep-reader-deploy-8569fb1/`. New regression reproduces two old inline-heading failures; corrected 27 harness/lazy-layout/source-boundary and 214 PDF tests pass with zero skips. Standalone inspection passes with host OCR enabled. Implementation is locally verified; item remains open until external approval and corrected real-engine Docker acceptance.
  Timestamp: 2026-11-13


## Shared sentence translation (2026-11-09)

- [x] Verify shared translation on file/PostgreSQL/API and backed-up real sources.
  Owner: scripts. Status: Implemented and verified.
  Acceptance: Both modes reuse compatible aligned output; ordinary aggregation and learning sentence pairs retain their UI behavior. No automatic legacy migration or duplicate aggregate storage.
  Evidence: 470 backend tests (35 shared cases); actual API/React at 1280/390 uses four offline calls for two paragraphs across both modes. Existing display/inline suites and type/build pass; backed-up English/Chinese excerpts remain unchanged.
  Timestamp: 2026-11-09


## Language Learning Reading Mode Queue (2026-10-28)

**2/2 owned system tasks complete: LLRN-20/22 verified (2026-10-29).** [Shared dependency queue](../section_tasks/module-checklist.md#language-learning-reading-mode-queue-2026-10-28)；[canonical design](../section_tasks/module-detailed-design.md#language-learning-reading-mode-plan-2026-10-28)。本檔只持有 scripts 的唯一 checklist。

### LLRN-20

- [x] LLRN-20: Backend integration acceptance.
  Owner: scripts. Status: Implemented and verified with deterministic providers and disposable SQL/file storage.
  Depends on: LLRN-12, LLRN-16.
  Deliverable: 整合 SQL/file/API、切句/registry/生成服務與 document/settings lifecycle 的故障和並行驗收；各 owner 的單元證據不重複計數。
  Acceptance: 驗證 read purity、單次 admission、全部 cache 維度、設定失效、rollback/unknown/replay/worker fencing、全等級來源失效與正常翻譯回歸。
  Evidence required: 可重跑 isolated suites、provider call counters、before/after source/profile/setting hashes、disposable SQL 清理；缺前提記錄 skipped/Needs Confirmation。
  Evidence: `test_learning_backend_integration.py` adds 14 continuous ASGI/DI/file/SQL journeys. `accept_learning_backend.py` passes 738 distinct tests (14 new + 724 regressions), zero failures/skips; source/profile/settings phase hashes and dispatch counts recorded. Covers settings roundtrips, runtime restart/reuse, source/Undo/reparse, rename/delete/recreation, unknown consent and concurrent acknowledgement recovery; existing suites rerun cache/registry/rollback/fencing/migration/ordinary translation cases. All temporary schemas and the isolated PostgreSQL container removed. Private evidence: `data/test-books/learning-backend/llrn20-20261028/report.json`; missing-DSN control fails closed. At this historical checkpoint LLRN-21/22 were still open; current system acceptance is recorded below and in the shared queue.
  Timestamp: 2026-10-29

### LLRN-22

- [x] LLRN-22: Real-book and provider system acceptance.
  Owner: scripts. Status: Implemented and verified within the maintainer-confirmed system scope (2026-10-29).
  Depends on: LLRN-20, LLRN-21.
  Deliverable: 隔離資料、不可變真書有限段落、實際 API/SQL/配置 provider/Reader 的系統驗收；保留 migration/recovery 與 request evidence。
  Acceptance: 來源及逐句覆蓋、registry/級別規則、格式驗證、保存／重用、失敗分類及無自動付費重送。依最新指示，模型選擇、提示詞調整及語義評分移出此任務；歷史語義抽查不改判通過。
  Evidence required: 已捕捉的實際 provider 報告、backup/source hashes、API/SQL/browser 操作收據、故障控制及資源清理；不以 JSON 合法證明翻譯品質，不整書預生成或修改 production。
  Evidence: `accept_learning_live.py` now retains exact returned text and optional decoded diagnostics. Malformed output reaches production validation as known `invalid_output`; provider exceptions stay unknown. Raw replay preserves duplicate keys/formatting while remapping sentence references; decoded-only legacy replay is labelled, missing output is not fabricated, chained replay retains original request IDs. No provider, prompt, capacity or production schema change.
  Validation: `test_learning_live_recording.py` adds 15 regressions including actual SDK mock transport and file/SQL/HTTP failed refresh, saved preservation, explicit retry, command replay and unknown consent. Full backend runner passes 765 distinct tests, zero failures/errors/skips. `learning-live/llrn22e-legacy-replay/` and `llrn22e-raw-replay/` pass two-book/two-level/mixed API/SQL, runtime restart and actual Reader at 1280/390, with zero live calls or reload generation; backups unchanged, owned schemas removed. Final evidence: `data/test-books/learning-live/llrn22e-review/`.
  Historical evidence: the eight-call v4 live technical pass remains in `learning-live/llrn22c-20261029-v1/`; earlier semantic failures and v3/v4 follow-ups remain in the owner design/private reports. The raw-format replay derives from a legacy report and cannot recover original provider bytes; controlled raw/duplicate-key tests cover lossless preservation. The llrn22d malformed response was lost by the old recorder and cannot be reconstructed. Historical semantic observations are outside this system closure.
  Deployment follow-up (2026-10-29): corrected local startup to apply reviewed 003..005 in one transaction before API; fresh Compose mounts 001..005. Verified fresh initialization, restored-backup upgrade/reapply, failure stopping API and unchanged old rows. Current local 004/005 deployment restores the reported Vite preference request to 200; backup/private evidence: `data/maintenance/preference-schema-20261029/`.
  Timestamp: 2026-10-29

## OCR Registry Packaging (2026-10-27)

- [x] OCR-LANG-02: Package and verify all registry OCR dependencies.
  Acceptance: Registry-derived build installation, offline startup completeness gate, real model loading and missing/corrupt dependency tests; isolated real-book checks preserve immutable backups. No request-time downloads or duplicate language list in deployment configuration.
  Evidence: Dockerfile derives apt model packages from pdf_ocr_language_policy; verify_ocr_assets.py independently loads all 23 models at build/OCR-enabled startup. Compose/template/local configuration use registry. 108 focused and 214 PDF tests pass; eight generated multilingual OCR cases and three immutable-book cases pass. Real offline missing/corrupt French model controls prevent startup even with en hint; complete startup succeeds. No runtime acquisition or production book reparse.
  Timestamp: 2026-10-27

## Persisted Block Translation (2026-10-26)

Implemented and verified: 1/1 owned task. [Shared queue](../section_tasks/module-checklist.md#persisted-block-translation-queue-2026-10-26).

- [x] TRUI-06: Accept persistent translation and source-change semantics with real storage/books.
  Priority: P1. Depends on: TRUI-02..05. Acceptance: Real file/disposable PostgreSQL persistence/reload, concurrent same/different target languages, replay/response loss, rollback and stale-writer protection. Full registry/default/profile/no-op/change/delete/Keep/Clear/summary-dependency matrix, draft versus applied edits, Undo/rename/reparse and legacy writers. Immutable Madame Bovary/Red Roulette live-provider block tests with two stored target languages and preserved originals/profiles.
  Evidence required: Actual API/SQL results, before/after metadata/source hashes, safe fault injection and private reports; unchanged production data and removed disposable resources. Earlier stateless 92 tests are baseline, not this acceptance. No skipped live prerequisite or valid JSON alone certifies completion/literary accuracy.
  Evidence: 408 tests pass without skips, including test_translation_persistence(_postgres), existing route/service/reading/source-review/cleanup regressions and opt-in cleanup backups. accept_persisted_translation_books --live --browser exercises real model/API/SQL with immutable Madame Bovary/Red Roulette excerpts; report data/test-books/persisted-translation/20261026T070158Z-06e6f6e0. Backup hashes unchanged, disposable schemas removed; no production mutation or literary-accuracy certification.
  Timestamp: 2026-10-26

## Offline Preflight Follow-up (2026-10-24)

Owned WAIT tasks 2/2 (100%). [Shared order](../llm/module-checklist.md#offline-preflight-follow-up-2026-10-24).

- [x] WAIT-03: Package verified tokenizer assets for recreated containers and local development.
  Priority: P0. Depends on: WAIT-02. Owner: scripts; config supports wiring the asset location.
  Acceptance: Explicit setup/build command reuses provider encoding mapping, downloads with bounded setup-only timeouts, verifies hashes/version and installs runtime-readable assets outside ephemeral /tmp. Root Dockerfile/local-dev setup consume it; startup verifies offline without request/import-time network repair. No API keys/private books in assets/images. Missing/corrupt local setup is explicit, not hidden by a longer frontend timeout.
  Evidence required: Clean image build, offline process startup/token count, container recreation with empty ephemeral cache, missing/corrupt assets and failed acquisition tests. Local-dev setup smoke; no production restart required to mark packaging source verified. Full-book acceptance remains WAIT-05.
  Evidence: prepare_tokenizer_assets reuses the llm manifest, verifies size/SHA-256 and runtime model mapping, stages bounded setup-only downloads and atomically installs readable assets. Root Dockerfile packages /opt assets; tokenizer_startup verifies offline before exec. Local host setup uses the same command; start_local_dev waits for API health before UI. Private assets/env/venvs are excluded from image context. No runtime download or API/schema change.
  Validation: 145 tests pass, including 16 new setup/container cases, actual downloaded local assets and immutable-book regressions. Root image 62776af533ea builds with fresh verified tokenizer assets (unchanged dependency layers reused); rejected incomplete downloads remain failures, explicit retry succeeds. New/recreated non-root read-only containers with network denied and empty tmpfs count correctly; missing/corrupt assets block startup. Actual offline default Uvicorn /health returns 200, disposable container removed. Image counter/budget suite passes 20/21 with one private-book opt-in skipped, covered on host. Shell setup, acquisition/atomicity fault tests, private-safe logs and diff checks pass. Production data/services unchanged; full incident-book HTTP/browser gate remains WAIT-05.
  Timestamp: 2026-10-24

- [x] WAIT-05: Accept cold/offline preflight and long frontend waits with the full incident book.
  Priority: P1; final gate. Depends on: WAIT-01..04. Owner: scripts; UI supports browser harness.
  Acceptance: Hash-backed full 暗水幽灵 PDF plus accepted 176196-character OCR snapshot/evidence in private data/test-books, not just Sample. Disposable-storage real tokenizer/API runs in new/recreated containers with network denied and empty ephemeral cache; five cold/five warm runs under the documented 10-second test-environment gate, zero model/OCR/profile/cleanup mutation calls, unchanged source/profile/hierarchy/reading state. At 1280/390, held responses past 120 seconds remain pending/cancellable and then succeed; external failures/expiry/lifecycle/unknown outcomes still recover without duplicate work.
  Evidence required: Private manifest, environment/image/encoding identities, stage timing distributions, network/call counters, before-after hashes, fault results, browser screenshots and build checks. No production reparse/deletion/restart or new live-model cleanup needed. Missing real-book/runtime prerequisites remain Needs Confirmation; warm-only probes or timer removal alone do not close the task.
  Evidence: Immutable corpus 7270c4d2c581f525 preserves the full 176196-character incident source, 261 page mappings, 221 OCR evidence spans, hierarchy/reading state and separately hash-backed profile. accept_wait_preflight.py runs actual HTTP routes/accepted adapter/provider counter against disposable file storage. Five recreated offline/read-only containers (image 9c0a3963e9ff) with empty tmpfs pass five cold and five warm preflights; cold 413.7-664.6 ms, warm 125.2-380.2 ms, all below the 10-second test gate. Zero generation/OCR/profile/cleanup-write/registry/network calls and unchanged hashes. Production SQL remains untouched; SQL composition is covered separately by disposable regression/browser fixtures.
  Validation: 227 backend tests pass without skips, including missing/corrupt assets, concurrent initialization, capacity identity and file/SQL/API faults. Full-book actual HTTP preflight reaches App at 1280/390 after held responses exceeding 120 seconds, with cancel/late-response protection and zero execution. Five existing browser suites and 36 malformed transport cases pass; npm run check passes with the existing bundle warning. Screenshots inspected, disposable DB/containers removed. Evidence: data/test-books/7270c4d2c581f525/wait05-20261024/{offline,browser}.json and private logs/screenshots. No production deployment, restart, live LLM or original-book mutation.
  Timestamp: 2026-10-24

## Independent Smart Cleanup Backlog (2026-10-23)

Owned tasks 1/1 (100%). [Shared order](../document_structure/module-checklist.md#independent-smart-cleanup-backlog-2026-10-23).
Timestamp: 2026-10-24

- [x] SCLEAN-10: Accept both cleanup entry paths with real books and deployed-contract smoke tests.
  Priority: P1; mandatory final gate. Depends on: SCLEAN-01..09. Owner: scripts; UI supports browser harness.
  Acceptance: Hash-verified private Sample/Sample2/noise backups exercise live-model Common-only cleanup, enhance's in-memory cleanup and failed-auto-then-independent retry; subsequent cleanup excludes prior deletions/protects manual edits. Require at least one reviewed nonempty safe deletion, no meaningful-text loss, immutable raw/canonical/page/profile and stable standalone hierarchy/IDs. TXT/Fast controls make zero cleanup calls. Actual API/SQL and 1280/390 App prove one commit, truthful partial/failure, five-level effective content, audit/Undo, refresh/recovery and no hidden extraction/profile/split calls. Reuse existing fault suites; do not relax guards or count mocks/all-skipped as positive live acceptance.
  Evidence required: Private versioned manifests/reports, full local model evidence, applied-span manual review, stage timings/counts/rejection reasons, screenshots and isolated current-image route smoke. Original Library data untouched; production restart/deployment requires separate authorization. Missing live prerequisites remain Needs Confirmation, not checked.
  Evidence: accept_ocr_cleanup_books.py extends the immutable corpus with standalone/combined/retry entry modes, live gpt-4.1-mini, prompt v3 and unchanged guards. Seven final actual-book/API/SQL/browser runs pass. Sample2's sole applied span [386,389) was reviewed against original page 1 in every flow; subsequent cleanup excludes prior deletion, fresh browser execution does not delete twice, and Undo restores the projection. Sample/dark-water unsafe proposals remain rejected with retained noise, not false perfect-cleanup claims. Source/document hashes and page maps are unchanged; standalone hierarchy/unit/block IDs remain stable. Private review: data/test-books/cleanup-20261023/sclean10-review-20261024.json. Native TXT and Fast PDF controls make zero cleanup calls.
  Validation: 387 backend tests pass with private backups enabled; four actual App/API/SQL fault suites pass at 1280/390, plus all real-book browser flows. Type/build and adapter checks pass (existing bundle warning). Isolated current-runtime image 83cead1c68d3 passes 39 tests with two private-source opt-in skips; temporary Uvicorn HTTP smoke verifies exactly three routes and unsupported-storage 503/no consent. The image suite overlaps host coverage. Fixed test-only dynamic App state restoration exposed by combined image execution. No runtime policy change, Library mutation, OCR/profile regeneration or production restart. Fixture profiles are absent; SQL profile-row preservation is covered by regression, not fabricated live-book hashes. Full limitations and reproduction are in the scripts design.
  Timestamp: 2026-10-24

## Loader Boundary Governance

- [x] LOADER-04: Validate loader abstraction with third-format contracts, PDF regressions and backed-up real books.
  Acceptance: no production adapter claimed for synthetic format; exact source/profile preservation, existing API/identity regressions and real-book reports.
  Evidence: test_loader_boundary.py; accept_loader_boundary_books.py; private loader-boundary-20261021T133550.json and final read-only content-recovery-20261021T133544.json. Docker PDF 180 pass, focused 95 pass, PostgreSQL 88 pass/two skips (overlapping suites). Both books preserve canonical hashes and zero warm extraction/profile work. Initial unsafe-default live acceptance and explicit layout recovery recorded in design; read-only flag added.
  Timestamp: 2026-10-21

## Format-Neutral Extraction Backlog (2026-10-21)

Implemented; evidence and scope are recorded per item. Ordered design: [shared plan](../document_preparation/module-detailed-design.md#format-neutral-extraction-plan-2026-10-21).

- [x] EXTRACT-10: Accept real TXT and PDF enhancement using immutable backup books.
  Depends on: EXTRACT-01 through EXTRACT-09.
  Acceptance: Both named books complete actual LLM enhancement/atomic commit/automatic layout from backup-only isolated inputs; exact text and profile hashes preserved, negative matrix and PDF regressions pass; private reports required, mocks/Common parse alone do not close.
  Evidence: Madame Bovary backup fa794acef055f235/20261021T083239624177Z/api-live.json: direct real API/model, three calls, 5 chapters/31 sections/423 units, 652001 characters preserved and profile hashes unchanged. PDF backup 085747b65094d52f/20261021T083612482751Z: two real model calls, 11 chapters/30 sections; api-replay.json proves atomic commit/cleanup and 108 real-layout units with unchanged profiles. Final 95 focused tests pass; 180 PDF tests pass; 75 PostgreSQL regressions pass with two opt-in skips; desktop/mobile confirmation and build pass. Failed live run led to a located semantic-boundary regression/fix (prompt v16), not a weakened validator. All enhancement writes used disposable schemas; live Library TXT provenance was explicitly migrated without replacing hierarchy/profile.
  Timestamp: 2026-10-21

## Content Read Performance (2026-10-21)

- [x] CONTENT-05: Rebuild both authorized books from verified backups and accept full-section latency, constant read counts and profile immutability.
  Acceptance: See app detailed design Content Read Recovery; evidence required before completion.
  Evidence: accept_content_read_books.py and test-content-read-live.mjs pass after authorized deletion/reupload/Common Fast preparation. Private content-recovery-20261021T073403.json and final-deployment content-recovery-20261021T074014.json: largest sections, three HTTP runs each, unchanged profile hashes, unchanged repeated-prepare layout and five interaction reads. 51 focused tests and eight regression scripts pass.
  Timestamp: 2026-10-21

## Purpose

This checklist records completed, code-confirmed or design-confirmed tasks for the `scripts` module.

It is used to:
- preserve module-level implementation memory
- reduce hallucination in future Codex tasks
- prevent context-window compression from losing completed work
- track future task completion explicitly

## Source Documents

- `Deep_Reflective_Reader/scripts/module-detailed-design.md`
- `Deep_Reflective_Reader/proposal.md`
- `Deep_Reflective_Reader/high-level-design.md`
- `Deep_Reflective_Reader/scripts/`

## Rules

- Only completed work is listed as checked.
- Future work must not be added unless explicitly requested.
- If a new task is added later, it must first be added unchecked.
- Once completed, it must be checked in this file.
- Uncertain items must go to `Needs Confirmation`, not the completed checklist.

## Completed Checklist

- [x] Move real-book fixtures under data and run acceptance against immutable backups.
  Acceptance: Preserve the raw PDF, accepted source and original hierarchy with hash checks; never overwrite a different backup. Both live generation and API replay use backup case IDs without original Library reads. Verify deletion survival, corruption rejection and real-book replay.
  Evidence: `private_book_corpus.py`, migrated `data/test-books/085747b65094d52f`, four passing corpus tests (original deletion, immutable overwrite guard, corruption/missing backup, path containment). Backup-only live attempt `20261021T052028529811Z` passes with 18 chapters/35 sections and exact 147401-character preservation. Backup-only real API/disposable PostgreSQL/layout replay passes with 107 units; production source-read methods are explicitly blocked. Git/data Docker exclusions and diff check pass.
  Timestamp: 2026-10-21

- [x] Validate upload/preparation timing logs without real document parsing or LLM calls.
  Evidence: passing `test_upload_preparation_logging.py` (4 cases), `test_document_upload_route.py`, `test_prepare_structured_reuse_before_raw_load.py`, `test_document_preparation_raw_load_errors.py`, `test_pdf_document_loader_inspection.py`, and `test_main_request_logging.py`.
  Notes: Covers worker/async request-id isolation, stage boundaries, cache reuse, unchanged exception propagation, and exclusion of content/error-detail sentinels from added logs.
  Validation limitation: Additional `test_task_layout_chapters_response.py` fails its existing task-unit ordering assertion both in the working tree and an isolated `git archive HEAD` baseline. No unrelated task-layout fix is included. Live Docker rejected-upload smoke confirms stage records and request id are persisted without creating a document.
  Timestamp: 2026-10-14

- [x] Regress storage-aware deletion, residual database records, failures, and namespace isolation.
  Evidence: passing `test_document_delete_route.py`, `test_document_management_route.py`, `test_document_list_search_api.py`; Docker execution of `test_postgres_document_delete.py` on 2026-10-14.
  Notes: Covers confirmation, missing raw files, full-width slash names, database failures and retry, catalog readback, all document-owned PostgreSQL cascades, and namespace isolation without LLM calls.
  Timestamp: 2026-10-14

- [x] Maintains regression script suite for hierarchy, task-layout, artifact persistence, and profile metadata.
  Evidence: `Deep_Reflective_Reader/scripts/test_task_layout_hierarchy_first_read.py; Deep_Reflective_Reader/scripts/test_post_structure_metadata_enrichment.py; Deep_Reflective_Reader/scripts/module-detailed-design.md (Key Files)`
  Notes: Scripts folder acts as primary test entry surface in current repo state.

- [x] Includes real-document and REST smoke script coverage for end-to-end verification paths.
  Evidence: `Deep_Reflective_Reader/scripts/test_rest_structured_parser_modes.py; Deep_Reflective_Reader/scripts/test_rest_dynamic_context.sh; Deep_Reflective_Reader/scripts/module-detailed-design.md (Main Responsibilities)`
  Notes: Smoke scripts validate runtime behavior beyond unit-style checks.

- [x] Covers profile/metadata and language registry hardening through dedicated regression scripts.
  Evidence: `Deep_Reflective_Reader/scripts/test_document_profile_parser_metadata.py; Deep_Reflective_Reader/scripts/test_language_script_registry.py; Deep_Reflective_Reader/scripts/test_language_discourse_registry.py; Deep_Reflective_Reader/scripts/module-detailed-design.md (Main Responsibilities)`
  Notes: Metadata and registry semantics are now tracked by dedicated tests.

- [x] Covers structured-document reuse before raw-load preparation
  Evidence: `Deep_Reflective_Reader/scripts/test_prepare_structured_reuse_before_raw_load.py`; container execution of `PYTHONPATH=. python scripts/test_prepare_structured_reuse_before_raw_load.py`.
  Notes: The regression proves existing structured documents are validated and reused before raw PDF loading in base/common/non-force preparation, while force rebuild still loads raw text and rewrites structure.

- [x] Covers manual structure API request, validation/preview response, and commit reparse request schema validation
  Evidence: `Deep_Reflective_Reader/scripts/test_manual_structure_api_schemas.py`; `Deep_Reflective_Reader/.venv/bin/python Deep_Reflective_Reader/scripts/test_manual_structure_api_schemas.py`.
  Notes: Regression validates source-agnostic manual structure request schema, typed char/page anchors, title/doc normalization, two-level maximum depth, orphan-section rejection, invalid range rejection, lightweight preview response serialization, issue taxonomy fail-fast behavior, provenance preview normalization, response projected-range validation, and explicit manual reparse request schema gating.

- [x] Covers manual structure validation route behavior
  Evidence: `Deep_Reflective_Reader/scripts/test_manual_structure_validate_route.py`; `Deep_Reflective_Reader/.venv/bin/python Deep_Reflective_Reader/scripts/test_manual_structure_validate_route.py`.
  Notes: Regression validates `POST /documents/manual-structure/validate` returns a lightweight normalized preview, rejects invalid manual plans with 422, does not expose raw text, task content, or task-layout payload, and verifies schema-valid `parser_mode=manual_structure` commit requests route through the manual commit coordinator boundary without reaching the legacy common/LLM reparse coordinator.

- [x] Covers deterministic manual structure projection behavior
  Evidence: `Deep_Reflective_Reader/scripts/test_manual_structure_projection.py`; `Deep_Reflective_Reader/.venv/bin/python Deep_Reflective_Reader/scripts/test_manual_structure_projection.py`.
  Notes: Regression validates source-agnostic manual plan projection into a lightweight chapter/section preview, chapter-only same-name section preview, two-level maximum depth enforcement, orphan-section rejection, unsupported/mixed anchor rejection, char/page range validation, and sibling overlap detection without persistence or `StructuredDocument` creation.

- [x] Covers manual structure validation route delegation to projector
  Evidence: `Deep_Reflective_Reader/scripts/test_manual_structure_validate_route.py`; `Deep_Reflective_Reader/.venv/bin/python Deep_Reflective_Reader/scripts/test_manual_structure_validate_route.py`.
  Notes: Regression validates the validation route now surfaces deterministic projection errors such as overlapping sibling ranges as `valid=false` without raw text, task content, task-layout payload, hierarchy persistence, or reparse execution.

- [x] Covers manual structure commit validation gate before persistence
  Evidence: `Deep_Reflective_Reader/scripts/test_manual_structure_validate_route.py`; `Deep_Reflective_Reader/.venv/bin/python Deep_Reflective_Reader/scripts/test_manual_structure_validate_route.py`.
  Notes: Regression validates schema-valid but projection-invalid `parser_mode=manual_structure` commit requests return stable 422 with projector issue evidence and no structured document path or section count.

- [x] Covers manual structure commit source evidence gates
  Evidence: `Deep_Reflective_Reader/scripts/test_manual_structure_commit_source_evidence.py`; `Deep_Reflective_Reader/.venv/bin/python Deep_Reflective_Reader/scripts/test_manual_structure_commit_source_evidence.py`.
  Notes: Regression validates invalid projection avoids source loading, missing source returns 404, stale `source_hash` returns 409, and matching source evidence proceeds to draft/persistence only after validation succeeds.

- [x] Covers manual structure StructuredDocument draft building
  Evidence: `Deep_Reflective_Reader/scripts/test_manual_structure_document_builder.py`; `Deep_Reflective_Reader/.venv/bin/python Deep_Reflective_Reader/scripts/test_manual_structure_document_builder.py`.
  Notes: Regression validates char-range manual plans build hierarchy-only `StructuredDocument` drafts with `chapters[].sections[]`, no root `sections[]` or `structure_nodes`, chapter-only same-name implicit sections, advisory parse provenance, explicit page-range rejection until page-boundary mapping exists, and invalid range rejection without a partial document.

- [x] Covers manual structure commit draft-build gate
  Evidence: `Deep_Reflective_Reader/scripts/test_manual_structure_commit_source_evidence.py`; `Deep_Reflective_Reader/.venv/bin/python Deep_Reflective_Reader/scripts/test_manual_structure_commit_source_evidence.py`.
  Notes: Regression validates the coordinator builds a manual `StructuredDocument` draft only after source evidence passes and rejects raw-text out-of-range draft anchors with 422 before persistence.

- [x] Covers manual structure commit persistence gate
  Evidence: `Deep_Reflective_Reader/scripts/test_manual_structure_commit_source_evidence.py`; `Deep_Reflective_Reader/.venv/bin/python Deep_Reflective_Reader/scripts/test_manual_structure_commit_source_evidence.py`.
  Notes: Regression validates a source-hash-matched manual commit saves exactly one hierarchy-only `StructuredDocument` through the repository, while invalid projection, missing source, stale source, draft anchor failure, and repository save failure do not report a successful commit.

- [x] Covers manual structure commit route success mapping
  Evidence: `Deep_Reflective_Reader/scripts/test_manual_structure_validate_route.py`; `Deep_Reflective_Reader/.venv/bin/python Deep_Reflective_Reader/scripts/test_manual_structure_validate_route.py`.
  Notes: Regression validates `/documents/reparse-structure` maps a `parser_mode=manual_structure` coordinator success result to HTTP 200 with `success=true`, parser mode, section count, nullable backend path, and no legacy common/LLM reparse dispatch.

- [x] Covers page-backed manual structure route status mapping
  Evidence: `Deep_Reflective_Reader/scripts/test_manual_structure_validate_route.py`; `Deep_Reflective_Reader/.venv/bin/python Deep_Reflective_Reader/scripts/test_manual_structure_validate_route.py`.
  Notes: Regression validates schema-valid `page_range` commit routing for missing/unsupported page evidence (`422`), stale source evidence (`409`), out-of-range/unprojectable page anchors (`422`), and successful page-backed commit response mapping (`200`) without falling through to legacy common/LLM reparse.

- [x] Covers manual structure preparation source-evidence handoff
  Evidence: `Deep_Reflective_Reader/scripts/test_manual_structure_preparation_handoff.py`; `Deep_Reflective_Reader/.venv/bin/python Deep_Reflective_Reader/scripts/test_manual_structure_preparation_handoff.py`.
  Notes: Regression validates preparation provides normalized source identity, raw text, source hash, best-effort language, optional empty page-boundary evidence, and empty-source failure without structured artifact writes.

- [x] Covers manual structure commit provenance from preparation evidence
  Evidence: `Deep_Reflective_Reader/scripts/test_manual_structure_commit_source_evidence.py`; `Deep_Reflective_Reader/.venv/bin/python Deep_Reflective_Reader/scripts/test_manual_structure_commit_source_evidence.py`.
  Notes: Regression validates accepted manual commit preserves evidence language and actual raw-text source hash in saved `StructuredDocument.parse_provenance`.

- [x] Covers task-unit content endpoint cache-valid task-layout fixtures
  Evidence: `Deep_Reflective_Reader/scripts/test_task_unit_content_endpoint.py`; `PYTHONPATH=. .venv/bin/python scripts/test_task_unit_content_endpoint.py`.
  Notes: Regression fixtures now derive the expected task-layout resolver version from `SectionTaskCoordinator._TASK_LAYOUT_RESOLVER_VERSION`, so cache-valid smoke coverage verifies no resolver refresh and no persistence write under the current section-scoped metadata contract.

- [x] Covers task-layout cache-hit manual source-evidence regression
  Evidence: `Deep_Reflective_Reader/scripts/test_task_layout_persistence_cache.py`; `.venv` execution of `PYTHONPATH=Deep_Reflective_Reader Deep_Reflective_Reader/.venv/bin/python Deep_Reflective_Reader/scripts/test_task_layout_persistence_cache.py`.
  Notes: Regression simulates a task-layout cache hit and asserts the coordinator must not call `load_manual_structure_source_evidence(...)` on ordinary reads, because that path can trigger scanned-PDF OCR. The same script verifies an explicit `include_anchor_page_evidence=True` cache-hit read still loads page evidence for TOC edit-existing anchor prefill without recomputing task units.
  Timestamp: 2026-10-05

- [x] Covers native Apple PDF page-boundary loading without OCR
  Evidence: `Deep_Reflective_Reader/scripts/test_pdf_document_loader_inspection.py`; `.venv` execution of `PYTHONPATH=Deep_Reflective_Reader Deep_Reflective_Reader/.venv/bin/python Deep_Reflective_Reader/scripts/test_pdf_document_loader_inspection.py`; direct timing check for `APPLE.pdf`.
  Notes: Regression asserts native PDF text extraction happens once per page per caller, and `APPLE.pdf` page-boundary evidence returns one boundary per page with `ocr_started=False`. This protects simple born-digital PDFs from entering OCR or repeated native extraction during page-default evidence loading.
  Timestamp: 2026-10-05

- [x] Covers PostgreSQL manual reparse repository boundary separation
  Evidence: `Deep_Reflective_Reader/scripts/test_postgres_manual_reparse_repository_boundary.py`; `Deep_Reflective_Reader/.venv/bin/python Deep_Reflective_Reader/scripts/test_postgres_manual_reparse_repository_boundary.py`.
  Notes: Regression validates generic PostgreSQL structured document saves preserve current hierarchy/version through `replace_existing_hierarchy=False`, while parser-level manual reparse saves use the explicit replacement boundary with `replace_existing_hierarchy=True`.

- [x] Covers live PostgreSQL manual reparse duplicate chapter_order replacement
  Evidence: `Deep_Reflective_Reader/scripts/test_postgres_manual_reparse_duplicate_chapter_order.py`; rebuilt Docker API container validation with `docker compose exec -T api python scripts/test_postgres_manual_reparse_duplicate_chapter_order.py`.
  Notes: Live regression validates existing-document manual replacement over an existing `chapter_order=0` row, confirms the replacement hierarchy is current, confirms `initial_parse`/`hard_reparse` provenance, and confirms transaction rollback preserves the old hierarchy when replacement fails.

- [x] Covers artifact-aware interaction metadata schema validation
  Evidence: `Deep_Reflective_Reader/scripts/test_artifact_aware_interaction_metadata_schema.py`; `.venv` execution of `PYTHONPATH=Deep_Reflective_Reader Deep_Reflective_Reader/.venv/bin/python Deep_Reflective_Reader/scripts/test_artifact_aware_interaction_metadata_schema.py`; `python3 -m py_compile Deep_Reflective_Reader/api_schemas.py Deep_Reflective_Reader/scripts/test_artifact_aware_interaction_metadata_schema.py`.
  Notes: Regression validates `ArtifactAwareInteractionMetadataResponse` defaults, artifact context mode normalization, primary source evidence ids, referenced artifact ids/types/target levels, coverage counts, deduplication/abstraction flags, invalid shape rejection, and explicit rejection of nested child artifact payload expansion.
  Timestamp: 2026-10-08

- [x] Covers artifact-aware secondary context assembly
  Evidence: `Deep_Reflective_Reader/scripts/test_artifact_aware_context_builder.py`; `.venv` execution of `PYTHONPATH=Deep_Reflective_Reader Deep_Reflective_Reader/.venv/bin/python Deep_Reflective_Reader/scripts/test_artifact_aware_context_builder.py`; `python3 -m py_compile Deep_Reflective_Reader/context/artifact_aware_context.py Deep_Reflective_Reader/scripts/test_artifact_aware_context_builder.py`.
  Notes: Regression validates compact lower-level artifact summary assembly, empty/insufficient-content skipping, duplicate/max-count pruning, artifact context char-budget pruning, coverage counts, deduplication/abstraction hint flags, provenance metadata export, defensive metadata copies, and absence of raw child artifact payload expansion.
  Timestamp: 2026-10-08

- [x] Covers analysis interaction strict output validation
  Evidence: `Deep_Reflective_Reader/scripts/test_analysis_interaction_service.py`; `.venv` execution of `PYTHONPATH=Deep_Reflective_Reader Deep_Reflective_Reader/.venv/bin/python Deep_Reflective_Reader/scripts/test_analysis_interaction_service.py`; `python3 -m py_compile Deep_Reflective_Reader/section_tasks/analysis_interaction_service.py Deep_Reflective_Reader/scripts/test_analysis_interaction_service.py`.
  Notes: Regression validates completed analysis from strict JSON/dict payloads, required `summary`/`reasoning`/`explanation` fields, schema/prompt/context metadata, insufficient-content fast path without generator calls, invalid JSON and missing fields as `generation_failed`, and fail-fast rejection of non-analysis requests.
  Timestamp: 2026-10-08

- [x] Covers analysis interaction public route vertical slice
  Evidence: `Deep_Reflective_Reader/scripts/test_analysis_interaction_routes.py`; `.venv` execution of `PYTHONPATH=Deep_Reflective_Reader Deep_Reflective_Reader/.venv/bin/python Deep_Reflective_Reader/scripts/test_analysis_interaction_routes.py`; `python3 -m py_compile Deep_Reflective_Reader/main.py Deep_Reflective_Reader/scripts/test_analysis_interaction_routes.py`.
  Notes: Regression validates `POST /documents/reading-interactions/insight/read`, `/insight/generate`, and `/insight/refresh` route mapping, schema-normalized target handoff, shared envelope serialization, compact analysis payload mapping, reasoning-to-interpretation fallback, and read/write dispatch separation.
  Timestamp: 2026-10-10

- [x] Covers reading interaction route HTTP status policy for insight routes
  Evidence: `Deep_Reflective_Reader/scripts/test_analysis_interaction_routes.py`; `.venv` execution of `PYTHONPATH=Deep_Reflective_Reader Deep_Reflective_Reader/.venv/bin/python Deep_Reflective_Reader/scripts/test_analysis_interaction_routes.py`; `python3 -m py_compile Deep_Reflective_Reader/main.py Deep_Reflective_Reader/scripts/test_analysis_interaction_routes.py`.
  Notes: Regression validates malformed insight requests fail with `422` before app dispatch, missing hierarchy targets map to `404`, `insufficient_content` remains a `200` envelope state without payload, `stale_target` maps to `409`, `generation_failed` maps to `502`, `validation_failed` maps to `422`, and successful read/generate/refresh remain `200`.
  Timestamp: 2026-10-10

- [x] Covers no-auto-generation insight read route behavior
  Evidence: `Deep_Reflective_Reader/scripts/test_analysis_interaction_routes.py`; `.venv` execution of `PYTHONPATH=Deep_Reflective_Reader Deep_Reflective_Reader/.venv/bin/python Deep_Reflective_Reader/scripts/test_analysis_interaction_routes.py`; `python3 -m py_compile Deep_Reflective_Reader/main.py Deep_Reflective_Reader/scripts/test_analysis_interaction_routes.py`.
  Notes: Regression uses a read-only poison coordinator for `POST /documents/reading-interactions/insight/read` and proves absent artifact reads return `not_generated` without payload while route dispatch remains limited to `read_analysis_artifact`. Accidental non-read coordinator access, including generation/refresh/session/prepare/reparse/task-layout/profile diagnostic paths, fails the test.
  Timestamp: 2026-10-10

- [x] Covers quiz interaction public route vertical slice
  Evidence: `Deep_Reflective_Reader/scripts/test_quiz_interaction_routes.py`; `.venv` execution of `PYTHONPATH=Deep_Reflective_Reader Deep_Reflective_Reader/.venv/bin/python Deep_Reflective_Reader/scripts/test_quiz_interaction_routes.py`; `python3 -m py_compile Deep_Reflective_Reader/main.py Deep_Reflective_Reader/scripts/test_quiz_interaction_routes.py`.
  Notes: Regression validates `POST /documents/reading-interactions/quiz/read`, `/quiz/generate`, and `/quiz/refresh` route mapping, schema-normalized target handoff, shared envelope serialization, drawer quiz payload item mapping, prompt version handoff, request `max_items` validation, generated item type validation, and separation from legacy section/chapter quiz routes.
  Timestamp: 2026-10-10

- [x] Covers critical-thinking interaction public route vertical slice
  Evidence: `Deep_Reflective_Reader/scripts/test_critical_thinking_interaction_routes.py`; `.venv` execution of `PYTHONPATH=Deep_Reflective_Reader Deep_Reflective_Reader/.venv/bin/python Deep_Reflective_Reader/scripts/test_critical_thinking_interaction_routes.py`; `python3 -m py_compile Deep_Reflective_Reader/main.py Deep_Reflective_Reader/api_schemas.py Deep_Reflective_Reader/scripts/test_critical_thinking_interaction_routes.py`.
  Notes: Regression validates `POST /documents/reading-interactions/critical-thinking/read`, `/generate-question`, `/submit-answer`, and `/retry-evaluation` route mapping, no-generation missing reads, shared target handoff, generated session id mapping, answer preservation on evaluation failure, retry completion without regenerating the question, and malformed submit rejection before dispatch.
  Timestamp: 2026-10-10

- [x] Covers complete public reading interaction route exposure set
  Evidence: `Deep_Reflective_Reader/scripts/test_analysis_interaction_routes.py`; `Deep_Reflective_Reader/scripts/test_quiz_interaction_routes.py`; `Deep_Reflective_Reader/scripts/test_critical_thinking_interaction_routes.py`; `Deep_Reflective_Reader/scripts/test_reading_interaction_api_schemas.py`; `.venv` execution of those route/schema regressions.
  Notes: Regression coverage now spans insight read/generate/refresh, quiz read/generate/refresh, and critical-thinking read/generate-question/submit-answer/retry-evaluation with shared target/envelope schema usage, no-auto-generation read behavior, explicit write-path dispatch, and malformed payload rejection before app dispatch.
  Timestamp: 2026-10-10

- [x] Covers document-backed analysis interaction artifact store
  Evidence: `Deep_Reflective_Reader/scripts/test_reading_interaction_artifact_store.py`; `.venv` execution of `PYTHONPATH=Deep_Reflective_Reader Deep_Reflective_Reader/.venv/bin/python Deep_Reflective_Reader/scripts/test_reading_interaction_artifact_store.py`; `python3 -m py_compile Deep_Reflective_Reader/section_tasks/reading_interaction_artifact_store.py Deep_Reflective_Reader/scripts/test_reading_interaction_artifact_store.py`.
  Notes: Regression validates missing current analysis reads return `None`, saved analysis artifacts receive deterministic artifact ids, persisted metadata is stored under `reading_interaction_artifacts`, and readback round-trips through the shared `CommonArtifact` adapter without creating a separate persistence category.
  Timestamp: 2026-10-10

- [x] Covers explicit document upload route behavior
  Evidence: `Deep_Reflective_Reader/scripts/test_document_upload_route.py`; `Deep_Reflective_Reader/main.py`; `Deep_Reflective_Reader/api_schemas.py`; `.venv` execution of `PYTHONPATH=Deep_Reflective_Reader Deep_Reflective_Reader/.venv/bin/python Deep_Reflective_Reader/scripts/test_document_upload_route.py`; `.venv` execution of `PYTHONPATH=Deep_Reflective_Reader Deep_Reflective_Reader/.venv/bin/python Deep_Reflective_Reader/scripts/test_document_list_search_api.py`; `Deep_Reflective_Reader/.venv/bin/python -m py_compile Deep_Reflective_Reader/main.py Deep_Reflective_Reader/scripts/test_document_upload_route.py`.
  Notes: Regression validates `POST /documents/upload` stores raw `.txt` source candidates, returns lightweight typed upload metadata, exposes uploaded raw candidates through `GET /documents`, rejects duplicate uploads by default, supports explicit replace and create-distinct conflict policies, asserts create-distinct returns a unique backend-owned `source_filename` and `doc_name`, rejects unsupported file types with `unsupported_file_type`, rejects unsafe filenames, and does not prepare/reparse/OCR/profile/retrieval/task-layout/generate interactions.
  Timestamp: 2026-10-13

- [x] Covers document management status/list route behavior
  Evidence: `Deep_Reflective_Reader/scripts/test_document_management_route.py`; `Deep_Reflective_Reader/main.py`; `Deep_Reflective_Reader/api_schemas.py`; `.venv` execution of `PYTHONPATH=Deep_Reflective_Reader Deep_Reflective_Reader/.venv/bin/python Deep_Reflective_Reader/scripts/test_document_management_route.py`; `Deep_Reflective_Reader/.venv/bin/python -m py_compile Deep_Reflective_Reader/main.py Deep_Reflective_Reader/api_schemas.py Deep_Reflective_Reader/scripts/test_document_management_route.py`.
  Notes: Regression validates `GET /documents/management` merges structured readiness and raw source metadata, marks structured-only rows as ready, raw-only rows as not prepared, same-name raw+structured rows as ready while preserving raw source metadata, supports query filtering, and does not expose raw text, hierarchy, task-unit content, content blocks, profile payloads, retrieval records, or artifacts.
  Timestamp: 2026-10-13

- [x] Covers document lifecycle action schema behavior
  Evidence: `Deep_Reflective_Reader/scripts/test_document_lifecycle_api_schemas.py`; `Deep_Reflective_Reader/api_schemas.py`; `.venv` execution of `PYTHONPATH=Deep_Reflective_Reader Deep_Reflective_Reader/.venv/bin/python Deep_Reflective_Reader/scripts/test_document_lifecycle_api_schemas.py`; `Deep_Reflective_Reader/.venv/bin/python -m py_compile Deep_Reflective_Reader/api_schemas.py Deep_Reflective_Reader/scripts/test_document_lifecycle_api_schemas.py`.
  Notes: Regression validates explicit `archive`, `restore`, `delete`, and `rename` lifecycle DTO semantics, rename target rules, typed non-success reasons, accepted response shape, source/structured/derived-resource effect flags, and rejection of raw text, hierarchy, task-layout, content-block, and artifact payload fields.
  Timestamp: 2026-10-13

- [x] Covers explicit document delete route behavior
  Evidence: `Deep_Reflective_Reader/scripts/test_document_delete_route.py`; `Deep_Reflective_Reader/main.py`; `Deep_Reflective_Reader/api_schemas.py`; `.venv` execution of `PYTHONPATH=Deep_Reflective_Reader Deep_Reflective_Reader/.venv/bin/python Deep_Reflective_Reader/scripts/test_document_delete_route.py`; `Deep_Reflective_Reader/.venv/bin/python -m py_compile Deep_Reflective_Reader/main.py Deep_Reflective_Reader/scripts/test_document_delete_route.py`.
  Notes: Regression validates `POST /documents/lifecycle/delete` requires explicit confirmation, removes matching raw source, structured JSON, and derived FAISS/profile artifacts, returns typed missing/unconfirmed/wrong-action lifecycle responses, keeps response payload lightweight, and does not expose raw text, hierarchy, task-unit content, content blocks, profile payloads, retrieval records, or artifacts.
  Timestamp: 2026-10-13

- [x] Covers explicit document rename route behavior
  Evidence: `Deep_Reflective_Reader/scripts/test_document_rename_route.py`; `Deep_Reflective_Reader/main.py`; `Deep_Reflective_Reader/api_schemas.py`; `.venv` execution of `PYTHONPATH=Deep_Reflective_Reader Deep_Reflective_Reader/.venv/bin/python Deep_Reflective_Reader/scripts/test_document_rename_route.py`; `Deep_Reflective_Reader/.venv/bin/python -m py_compile Deep_Reflective_Reader/main.py Deep_Reflective_Reader/scripts/test_document_rename_route.py`.
  Notes: Regression validates `POST /documents/lifecycle/rename` moves matching raw source, structured JSON, and derived FAISS/profile artifacts, updates lightweight structured JSON identity metadata, rejects target conflicts without partial overwrite, returns typed missing/wrong-action lifecycle responses, keeps response payload lightweight, and does not expose raw text, hierarchy, task-unit content, content blocks, profile payloads, retrieval records, or artifacts.
  Timestamp: 2026-10-13

- [x] Supports local development env-file injection for private LLM keys
  Evidence: `scripts/load_local_env.sh`; `scripts/start_local_dev.sh`; `.env.local.example`; `.env.example`; validation with `bash -n scripts/load_local_env.sh scripts/start_local_dev.sh`; dry-run source validation against a temporary env file.
  Notes: Local development startup now sources a reusable env loader that imports `.env.local` into the shell before Docker Compose receives the same env file. The template documents `.env.local` as the private ignored file, uses provider-neutral `DEEP_READER_LLM_API_KEY` for developer-owned secrets, bridges that value to the current backend runtime variable for compatibility, and includes `DEEP_READER_UI_PORT`.
  Timestamp: 2026-10-13

## Needs Confirmation

No unresolved confirmation items identified in this pass.

## Future Task Policy

New future tasks for this module must be added here first as unchecked items:

- [x] Cover batch task-unit content route order and read-only semantics
  Evidence: `Deep_Reflective_Reader/scripts/test_task_unit_content_endpoint.py`; `.venv` execution of `PYTHONPATH=Deep_Reflective_Reader Deep_Reflective_Reader/.venv/bin/python Deep_Reflective_Reader/scripts/test_task_unit_content_endpoint.py`.
  Notes: Regression verifies batch content returns per-task-unit responses in request order, suppresses raw content by default, rejects duplicate request ids fail-fast, loads the document once for the batch, and does not mutate persistence.
  Timestamp: 2026-10-05
- [x] Cover reading target resolver hierarchy-only semantics
  Evidence: `Deep_Reflective_Reader/scripts/test_reading_target_resolver.py`; `.venv` execution of `PYTHONPATH=Deep_Reflective_Reader Deep_Reflective_Reader/.venv/bin/python Deep_Reflective_Reader/scripts/test_reading_target_resolver.py`; `python3 -m py_compile Deep_Reflective_Reader/section_tasks/reading_target_resolver.py Deep_Reflective_Reader/app/section_task_coordinator.py Deep_Reflective_Reader/scripts/test_reading_target_resolver.py`.
  Notes: Regression covers document/book, chapter, section, and task-unit target resolution, app-layer one-load orchestration, optional parent-id consistency checks, duplicate task-unit rejection, title-primary rejection, and no fallback to root `sections[]` legacy-only targets.
  Timestamp: 2026-10-08
- [x] Cover target-agnostic reading interaction service contracts
  Evidence: `Deep_Reflective_Reader/scripts/test_reading_interaction_service_contracts.py`; `.venv` execution of `PYTHONPATH=Deep_Reflective_Reader Deep_Reflective_Reader/.venv/bin/python Deep_Reflective_Reader/scripts/test_reading_interaction_service_contracts.py`; `python3 -m py_compile Deep_Reflective_Reader/section_tasks/reading_interaction_service_contracts.py Deep_Reflective_Reader/scripts/test_reading_interaction_service_contracts.py`.
  Notes: Regression validates resolved-target requests, metadata defensive copies, successful artifact shape, no raw target/source content serialization, reasoned insufficient-content status, invalid type/status rejection, empty completed artifact rejection, and critical-thinking-only status gating.
  Timestamp: 2026-10-08
- [x] Cover analysis read/generate split and strict output validation
  Evidence: `Deep_Reflective_Reader/scripts/test_analysis_interaction_service.py`; `Deep_Reflective_Reader/scripts/test_analysis_interaction_read_generate_split.py`; `.venv` execution of `PYTHONPATH=Deep_Reflective_Reader Deep_Reflective_Reader/.venv/bin/python Deep_Reflective_Reader/scripts/test_analysis_interaction_service.py`; `.venv` execution of `PYTHONPATH=Deep_Reflective_Reader Deep_Reflective_Reader/.venv/bin/python Deep_Reflective_Reader/scripts/test_analysis_interaction_read_generate_split.py`; `python3 -m py_compile Deep_Reflective_Reader/section_tasks/analysis_interaction_service.py Deep_Reflective_Reader/section_tasks/analysis_interaction_orchestrator.py Deep_Reflective_Reader/scripts/test_analysis_interaction_service.py Deep_Reflective_Reader/scripts/test_analysis_interaction_read_generate_split.py`.
  Notes: Regression verifies read returns persisted analysis or `not_generated` without generator calls or placeholder writes; explicit generate reuses existing artifacts unless refreshed, persists completed and insufficient-content results only, and leaves invalid JSON generation failures unpersisted while preserving strict output validation coverage.
  Timestamp: 2026-10-08
- [x] Cover quiz read/generate split and current-artifact semantics
  Evidence: `Deep_Reflective_Reader/scripts/test_quiz_interaction_service.py`; `Deep_Reflective_Reader/scripts/test_quiz_interaction_read_generate_split.py`; `.venv` execution of `PYTHONPATH=Deep_Reflective_Reader Deep_Reflective_Reader/.venv/bin/python Deep_Reflective_Reader/scripts/test_quiz_interaction_service.py`; `.venv` execution of `PYTHONPATH=Deep_Reflective_Reader Deep_Reflective_Reader/.venv/bin/python Deep_Reflective_Reader/scripts/test_quiz_interaction_read_generate_split.py`; `python3 -m py_compile Deep_Reflective_Reader/section_tasks/quiz_interaction_service.py Deep_Reflective_Reader/section_tasks/quiz_interaction_orchestrator.py Deep_Reflective_Reader/scripts/test_quiz_interaction_service.py Deep_Reflective_Reader/scripts/test_quiz_interaction_read_generate_split.py`.
  Notes: Regression verifies read returns persisted quiz artifacts or `not_generated` without generator calls or placeholder writes; explicit generate reuses existing artifacts unless refreshed, persists completed and insufficient-content results only, and leaves invalid JSON generation failures unpersisted while preserving configured-limit and answer validation coverage.
  Timestamp: 2026-10-08
- [x] Cover quiz validation and configured count limits
  Evidence: `Deep_Reflective_Reader/scripts/test_quiz_interaction_service.py`; `.venv` execution of `PYTHONPATH=Deep_Reflective_Reader Deep_Reflective_Reader/.venv/bin/python Deep_Reflective_Reader/scripts/test_quiz_interaction_service.py`; `python3 -m py_compile Deep_Reflective_Reader/section_tasks/quiz_interaction_service.py Deep_Reflective_Reader/scripts/test_quiz_interaction_service.py`.
  Notes: Regression verifies valid quiz type enum handoff, task-unit/section/chapter/document target-level max defaults, configured max override enforcement, fewer-than-max acceptance, answer payload validation for short-answer/multiple-choice/true-false items, invalid output as `generation_failed`, and insufficient-content fast path without generator calls.
  Timestamp: 2026-10-08
- [x] Cover critical-thinking session lifecycle
  Evidence: `Deep_Reflective_Reader/scripts/test_critical_thinking_session_service.py`; `.venv` execution of `PYTHONPATH=Deep_Reflective_Reader Deep_Reflective_Reader/.venv/bin/python Deep_Reflective_Reader/scripts/test_critical_thinking_session_service.py`; `python3 -m py_compile Deep_Reflective_Reader/section_tasks/critical_thinking_session_service.py Deep_Reflective_Reader/scripts/test_critical_thinking_session_service.py`.
  Notes: Regression verifies generated question status, fixed question/evaluation instruction handoff, generated-but-unanswered preservation, answer submission, strict evaluation validation, completed status, evaluation failure preservation with retry, insufficient-content fast path, invalid question generation failure, and invalid lifecycle transition rejection.
  Timestamp: 2026-10-08
- [x] Cover reading interaction artifact validity semantics
  Evidence: `Deep_Reflective_Reader/scripts/test_reading_interaction_artifact_validity.py`; `.venv` execution of `PYTHONPATH=Deep_Reflective_Reader Deep_Reflective_Reader/.venv/bin/python Deep_Reflective_Reader/scripts/test_reading_interaction_artifact_validity.py`; `python3 -m py_compile Deep_Reflective_Reader/section_tasks/artifact_validity.py Deep_Reflective_Reader/section_tasks/reading_interaction_service_contracts.py Deep_Reflective_Reader/section_tasks/analysis_interaction_service.py Deep_Reflective_Reader/section_tasks/quiz_interaction_service.py Deep_Reflective_Reader/section_tasks/critical_thinking_session_service.py Deep_Reflective_Reader/scripts/test_reading_interaction_artifact_validity.py`.
  Notes: Regression verifies shared validity preflight, stale target priority before insufficient-content checks, no generator call on stale target for analysis/quiz/critical-thinking services, insufficient-content reason preservation across all services, and required reasons for `stale_target` and `insufficient_content` artifacts.
  Timestamp: 2026-10-08
- [x] Cover adaptive full-context versus semantic compact context selection
  Evidence: `Deep_Reflective_Reader/scripts/test_reading_interaction_context_selection.py`; `.venv` execution of `PYTHONPATH=Deep_Reflective_Reader Deep_Reflective_Reader/.venv/bin/python Deep_Reflective_Reader/scripts/test_reading_interaction_context_selection.py`; `python3 -m py_compile Deep_Reflective_Reader/context/reading_interaction_context.py Deep_Reflective_Reader/scripts/test_reading_interaction_context_selection.py`.
  Notes: Regression verifies full target context when within configured/model capability budget, semantic compact fallback when the target exceeds capability, provenance metadata without source text, and fail-fast rejection of empty content without title fallback. No multi-call map-reduce occurs in the first version.
  Timestamp: 2026-10-08
- [x] Cover hard-reparse cleanup for reading interaction artifacts
  Evidence: `Deep_Reflective_Reader/scripts/test_hard_reparse_reading_interaction_cleanup.py`; `Deep_Reflective_Reader/scripts/test_postgres_manual_reparse_repository_boundary.py`; `.venv` execution of `PYTHONPATH=Deep_Reflective_Reader Deep_Reflective_Reader/.venv/bin/python Deep_Reflective_Reader/scripts/test_hard_reparse_reading_interaction_cleanup.py`; `.venv` execution of `PYTHONPATH=Deep_Reflective_Reader Deep_Reflective_Reader/.venv/bin/python Deep_Reflective_Reader/scripts/test_postgres_manual_reparse_repository_boundary.py`; `python3 -m py_compile Deep_Reflective_Reader/document_structure/structured_document_artifact_repository.py Deep_Reflective_Reader/db/postgres_structured_document_artifact_repository.py Deep_Reflective_Reader/scripts/test_hard_reparse_reading_interaction_cleanup.py Deep_Reflective_Reader/scripts/test_postgres_manual_reparse_repository_boundary.py`.
  Notes: Regression verifies parser-level hard reparse saves clear document/chapter/section/task-unit interaction artifacts, critical-thinking session metadata, and referenced-artifact metadata from replacement structured payloads. PostgreSQL replacement remains on `replace_existing_hierarchy=True` while generic saves remain non-replacement.
  Timestamp: 2026-10-08
- [x] Cover artifact-aware secondary context assembly
  Evidence: `Deep_Reflective_Reader/scripts/test_lower_level_artifact_reference_policy.py`; `.venv` execution of `PYTHONPATH=Deep_Reflective_Reader Deep_Reflective_Reader/.venv/bin/python Deep_Reflective_Reader/scripts/test_lower_level_artifact_reference_policy.py`; `python3 -m py_compile Deep_Reflective_Reader/section_tasks/reading_interaction_service_contracts.py Deep_Reflective_Reader/section_tasks/analysis_interaction_service.py Deep_Reflective_Reader/section_tasks/quiz_interaction_service.py Deep_Reflective_Reader/section_tasks/critical_thinking_session_service.py Deep_Reflective_Reader/scripts/test_lower_level_artifact_reference_policy.py`.
  Notes: Regression verifies analysis, quiz, and critical-thinking generation receive compact lower-level artifact summaries as `secondary_context`, preserve primary `target.content`, record artifact provenance metadata without context text or child payloads, and continue generation when no child artifacts exist.
  Timestamp: 2026-10-08
- [x] Cover quiz deduplication from lower-level artifacts
  Evidence: `Deep_Reflective_Reader/scripts/test_quiz_deduplication_from_lower_artifacts.py`; `.venv` execution of `PYTHONPATH=Deep_Reflective_Reader Deep_Reflective_Reader/.venv/bin/python Deep_Reflective_Reader/scripts/test_quiz_deduplication_from_lower_artifacts.py`; `python3 -m py_compile Deep_Reflective_Reader/section_tasks/quiz_interaction_service.py Deep_Reflective_Reader/scripts/test_quiz_deduplication_from_lower_artifacts.py`.
  Notes: Regression verifies higher-level quiz generation receives explicit deduplication guidance plus lower-level quiz coverage signals, preserves primary target content, does not concatenate lower-level quiz focus text into payload items, records bounded metadata only, and fails generation for exact lower-level focus repeats where backend validation can detect duplication.
  Timestamp: 2026-10-08
- [x] Cover critical-thinking abstraction from lower-level sessions
  Evidence: `Deep_Reflective_Reader/scripts/test_critical_thinking_abstraction_from_lower_sessions.py`; `.venv` execution of `PYTHONPATH=Deep_Reflective_Reader Deep_Reflective_Reader/.venv/bin/python Deep_Reflective_Reader/scripts/test_critical_thinking_abstraction_from_lower_sessions.py`; `python3 -m py_compile Deep_Reflective_Reader/section_tasks/critical_thinking_session_service.py Deep_Reflective_Reader/scripts/test_critical_thinking_abstraction_from_lower_sessions.py`.
  Notes: Regression verifies higher-level critical-thinking generation receives fixed learning-continuity guidance, child session focus/outcome signals, unchanged primary source content, predictable base question instruction, bounded provenance metadata without child payloads, and one generated question for the current target.
  Timestamp: 2026-10-08
- [x] Cover referenced-artifact metadata persistence
  Evidence: `Deep_Reflective_Reader/scripts/test_referenced_artifact_metadata_persistence.py`; `.venv` execution of `PYTHONPATH=Deep_Reflective_Reader Deep_Reflective_Reader/.venv/bin/python Deep_Reflective_Reader/scripts/test_referenced_artifact_metadata_persistence.py`; `python3 -m py_compile Deep_Reflective_Reader/section_tasks/reading_interaction_service_contracts.py Deep_Reflective_Reader/section_tasks/analysis_interaction_service.py Deep_Reflective_Reader/section_tasks/quiz_interaction_service.py Deep_Reflective_Reader/section_tasks/critical_thinking_session_service.py Deep_Reflective_Reader/scripts/test_referenced_artifact_metadata_persistence.py`.
  Notes: Regression verifies generated analysis, quiz, and critical-thinking artifacts store top-level referenced-artifact metadata with ids/types/target levels, coverage counts, and hint flags, without secondary context text or child payloads. Hard-reparse cleanup remains tracked by the separate hard-reparse cleanup checklist item.
  Timestamp: 2026-10-08
- [x] Cover reading interaction common artifact mapping
  Evidence: `Deep_Reflective_Reader/scripts/test_reading_interaction_common_artifact_model.py`; `.venv` execution of `PYTHONPATH=Deep_Reflective_Reader Deep_Reflective_Reader/.venv/bin/python Deep_Reflective_Reader/scripts/test_reading_interaction_common_artifact_model.py`; `python3 -m py_compile Deep_Reflective_Reader/shared/common_artifact_model.py Deep_Reflective_Reader/section_tasks/reading_interaction_common_artifact.py Deep_Reflective_Reader/scripts/test_reading_interaction_common_artifact_model.py`.
  Notes: Regression verifies analysis, quiz, and critical-thinking session artifacts all map to the same `CommonArtifact` entity class through `artifact_type`, preserve hierarchy-aware target ids, round-trip back to the service contract, reject unsupported artifact types/invalid targets, and avoid category-specific top-level serialization roots.
  Timestamp: 2026-10-08

After implementation, the task owner must update this checklist and mark the task as completed:

- [x] <completed task>

No coding task should be considered complete unless the corresponding module checklist is updated.

## Selective Per-Page OCR Track

OCR-01 supporting evidence (counted only in the loader checklist):
`test_pdf_ocr_decision.py` passed 7 contract tests on 2026-10-14; existing
`test_pdf_document_loader_inspection.py` and `test_pdf_page_evidence.py` passed.
Runtime routing and real-document performance acceptance remain pending.

OCR-02 supporting evidence (counted only in the loader checklist):
`test_pdf_page_inspection.py` passed 18 tests on 2026-10-14; OCR-01's 7 tests and
existing PDF loader inspection, page-evidence and Outline regressions also passed.
Evidence covers real synthetic PDF streams, single native extraction, clipped
image union, unknown geometry/errors, no image decoding/subprocess, metadata
content exclusion and source replacement. No extra delivery task is counted.
The 18 inspection tests also passed with pinned pypdf 6.17.0 in a disposable
Compose container, without rebuilding the API or preparing user documents.

OCR-03 supporting evidence (counted only in the loader checklist):
`test_pdf_ocr_policy.py` passed 22 tests locally and in an isolated Compose
container on 2026-10-14; existing OCR-01 (7), OCR-02 (18) and PDF loader inspection
also passed. Tests cover three-way decisions, numeric/Chinese/Latin fixtures,
conservative mixed/hidden/header handling, conflicts, threshold/config validation
and bounded probe plans. Visual checks are fixtures; no actual detector or routing
is implemented by this task.

OCR-04 supporting evidence (counted only in the loader checklist):
`test_pdf_layout_routing.py` passed 14 tests locally and in isolated Compose on
2026-10-14, alongside earlier decision/policy/inspection, loader, page-evidence,
Outline and OCR layout fixture regressions. Read-only local four-page `《歸來》`
layout smoke retained 8,187 native characters with zero render/OCR calls in
346.1 ms; this is not full preparation acceptance or a deployed timing claim.

OCR-05 supporting evidence (counted only in the loader checklist):
`test_pdf_raw_recovery.py` passed 25 tests locally and in isolated Compose on
2026-10-14. Earlier PDF and manual-structure regressions, including the updated
mixed-PDF Outline test, pass; upload/preparation logging tests pass. Read-only
`《歸來》` smoke retained exact native text and four-page raw-span parity with OCR
disabled. Raster cover/trailing recovery in the Outline test is mocked, not a
real OCR quality claim. Design section 15 and loader section 16.9 record limits.

OCR-06 supporting evidence (counted only in the preparation checklist):
`test_prepare_lazy_pdf_layout.py` passes 15 tests locally and in isolated Compose
on 2026-10-14; all 86 PDF unittest cases and standalone PDF/TOC, raw-error,
structured-reuse, profile-ordering, manual and logging regressions pass. Shared
raw/spans, lazy acceptance, source mismatch, required recovery, force/cache and
failure paths are covered with isolated fixtures. No full latency acceptance.

OCR-07 supporting evidence (counted only in the loader checklist):
`test_pdf_execution_pass.py` adds 25 tests; all 111 PDF unittest cases pass locally
and in isolated Compose on 2026-10-14. Fifteen lazy-layout tests and 11 standalone
PDF/TOC/preparation/manual/logging scripts pass locally. Coverage includes bounded
cache identity/lifecycle, worker isolation, actual attempts/deadlines, required
failure atomicity, optional fallback and content-free correlated diagnostics.
Earlier memory fixtures explicitly declare their shared scope; timeout fixtures
expect the new budget-specific reason. No live deployment or OCR-08 completion.

- [x] OCR-08: Validate selective OCR correctness and measured cost end to end.
  Acceptance: Run fixture matrix in design section 15; assert render/OCR call counts, native/scanned/mixed extraction parity, quality rejection, boundary order, Outline/fallback behavior, disabled OCR, budget and cache invalidation. Compare cold/warm stage timings on the reported four-page PDF and representative scans using disposable preparation storage; record unavailable cases explicitly.
  Evidence: `accept_selective_ocr.py`; `test_selective_ocr_acceptance.py` (3 tests); 111 PDF and 15 lazy-preparation tests, all passing locally and in isolated Compose. Eleven standalone PDF/TOC/preparation/manual/logging scripts pass locally. Real-engine native/scanned/mixed fixtures verify full known transcript, exact native/blank spans and zero layout OCR for accepted Outline. Design section 15 records measured cases, environment, commands and limitations.
  Results: Four-page `《歸來》` retains 8,200 exact canonical characters with zero render/OCR calls; controlled cold/warm preparation 789.9/1.9 ms. Real `暗水幽灵` page 8 completes and reuses structure; `國富論lite` page 1 is quality-rejected without persistence. Scanned-page timings/counts are recorded separately, not hidden behind overall latency.
  Limits: Real language/profile/LLM, HTTP/PostgreSQL costs, full scanned-book accuracy and real vertical/rotated transcription accuracy are unverified. Excerpt quality-gate success is not ground-truth accuracy. No live deployment, user-artifact changes or matched historical speedup claim; broader historical roadmap items remain open.
  Depends on: OCR-07 in `../doc_loaders/module-checklist.md`. Final task of eight; earlier contract/policy tests belong to their implementation task, not extra progress items.
  Timestamp: 2026-10-14

## Maintenance Notes

- This checklist is module memory for completed work.
- It does not replace the module detailed design document.
- It does not replace tests and test evidence.
- It does not replace proposal/HLD decisions and governance context.
