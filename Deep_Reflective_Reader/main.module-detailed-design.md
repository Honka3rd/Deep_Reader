# main.py Detailed Design

## Reading Content Routes (2026-10-21)

GET /documents/{doc_name}/reading-content returns revision/source token, review
regions/disposition and undo availability. Blocks are omitted by default; block_id
selects one, include_blocks=true opts into the inventory. POST accepts only edit,
delete, undo or discard_region. No add endpoint exists. The service, not the route,
owns atomic mutation/invalidation; 404 missing document/region, 409 conflict, 422
invalid request, 503 bounded infrastructure error. PNG original-region preview at
/documents/{doc_name}/source-review/{region_id}/image is no-store, server-bbox-only
and source-hash checked. Existing local namespace routing applies; no new auth layer.
Library exposes review_required; note responses expose source_stale. Neither reads
nor preview invoke OCR/profile/LLM. API tests run against isolated file/SQL storage.

## Summary And Note Routes (2026-10-21)

POST /documents/reading-interactions/summary/{read,generate,refresh} and
POST /documents/reading-interactions/note/{read,save,delete} are implemented.
Reads are pure. Summary has the shared envelope and strict payload; generation
failure returns 502 while stored output remains intact. Source hash/version and
timestamps are exposed when present. Note mutations require revision/source token;
conflicts return 409, malformed requests 422, missing documents 404.
No prompts or persistence logic belongs to the handlers. Routes/coordinator and
actual two-book generation/read/save/reuse acceptance pass.

Rename now includes the configured PostgreSQL record and new interaction identity
mapping, in addition to file-backed resources. If DB rename rejects, moved files
are rolled back. Disposable PostgreSQL tests cover conflicts and retained notes/
summaries; production acceptance does not rename unrelated user books.

## FULL-03 Capacity Transport (2026-10-20)

Wire the coordinator capacity reader into preflight. Capacity recheck failures
return 413/503 with existing string detail plus schema-filtered capacity; confirmed
configuration changes immediately before dispatch map to 409. Normal diagnostic
preflight remains 200. Existing generation/commit errors retain their envelopes.
No source work or budget calculation lives in routes. See app FULL-03 for verified
consent behavior and isolated API/PostgreSQL evidence; frontend work remains pending.

## Enhancement Rejection Mapping (2026-10-20)

Supporting ENH-PLAN-05, counted in document_structure: known prefixed enhancement
rejection results retain ReparseDocumentStructureResponse success=false/error shape.
Invalid JSON/schema/reference/application plan => 422; prompt budget exceeded => 413;
FULL-02 also maps local_resource_limit to 413 while retaining the existing response
shape. This is supporting error mapping only, not FULL-03 capacity preflight delivery.
provider failure => 502; provider/budget unavailable or invalid output allowance =>
503; unexpected validation/builder failure => 500. Existing EnhancementConflict
status/reason paths remain unchanged. Unknown exceptions in enhanced execution
return only enhancement_execution_failed; Common/manual error behavior is unchanged.
Tests call the route with mocked coordinator results to verify status/envelope and
exception redaction, not live HTTP/DB execution. No implicit retry or mutation added.

## Enhancement Consent Transport (2026-10-19)

ENH-APP-01 owns this supporting adapter work. Read-only
`POST /documents/reparse-structure/preflight` and explicit consent fields on enhanced
reparse are implemented. The coordinator's PostgreSQL reader is wired; preflight
returns validated readiness, impact and a token only when evidence is sufficient.
Unsupported storage returns ready=false with an explicit reason and no token.
PREP-03 wires execution_available to storage/profile capability checks. Coordinator
rejects missing/stale consent with 409 and missing guards with 503 before generation.
Supported execution returns committed structure_version; nullable path does not imply
failure for PostgreSQL. Browser/API/storage acceptance passed in a disposable served
fixture; the user's running API was not redeployed.
Main only validates/forwards contracts
and maps readiness/conflict results; it must not extract text, start OCR, construct
plans or delete interactions. Missing/stale consent must not reach paid generation.
See app and api_schemas dated designs. Existing legacy intent compatibility does not
authorize unconfirmed destructive execution once this new contract is activated.

## Enhancement Semantics Correction (2026-10-19)

Reparse retains its current endpoint and legacy enhancement_intent inputs, but the
UI has one LLM enhance operation optimizing Common output. Both enhanced wire
values use that same planner policy; generated titles need no second user choice.
Transport/mode validation and explicit-write boundaries are unchanged. Current
policy supersedes the two-purpose interpretation in historical NATIVE-04 notes;
see document_structure's unified enhancement correction and local validation rules.

## READER-01 Public Layout (2026-10-19)

The existing task-layout mapper forwards bounded TOC/intent/implicit provenance
and section title_origin into the additive schemas. Ordinary reads remain read-only;
prepare-task-layout reuses the same response projection after explicit preparation.
No endpoint or parser authority is added. Four `test_reader_provenance.py` cases
cover real API serialization and isolated unchanged-file evidence; readiness
regressions pass eleven cases. UI READER-01 owns this supporting delivery/count.

## NATIVE-05 Runtime Status (2026-10-19)

The existing enhancement-intent forwarding now reaches active v2 application.
Prepare and Reparse succeed in isolated real-route tests with fake providers and
an in-memory store; invalid candidates fail without rewriting the stored hierarchy.
The NATIVE-04 staged 400 guard is removed. No route/schema expansion or read-path
mutation is introduced by this activation; live service deployment is not claimed.

## NATIVE-04 Intent Forwarding (2026-10-19)

Three explicit preparation/reparse routes forward enhancement_intent without
selecting OCR or silently changing parser mode. Incompatible requests fail schema
validation; valid generated intent currently receives 400 from the staged pipeline
guard, with no paid plan or write. Restore defaults remain compatible. API tests
verify independent toc_hint and intent propagation and the explicit rejection.
Full v2 runtime activation belongs to NATIVE-05, not this transport change.

## NATIVE-03 Explicit TOC Intent (2026-10-19)

Prepare and prepare-task-layout forward `toc_hint` to preparation; reparse forwards
it through the coordinator. The layout-only path rejects non-auto hints with 400,
since it cannot apply source-parser settings. Defaults preserve existing callers'
request shape while explicit preparation now checks Common policy identity.
GET/task-layout reads do not rebuild. Contract tests cover defaults, all three
hints, PDF-mode independence, forwarding and layout-only rejection. LLM prompt
consumption belongs to NATIVE-04, not route behavior.

## 1. Module Purpose

### Task-Layout Readiness Separation (2026-10-16)

App LAYOUT-02 owns this implementation. Ordinary `/documents/task-layout` reads
persisted units only; missing/stale/invalid layouts and refresh requests return
409 with actionable string `detail`. `/documents/prepare-task-layout` explicitly
prepares structure, ensures persisted units and validates repository readback before
returning the lightweight projection. Management `ready` remains structured-ready.
Additive `layout_only=true` skips base preparation and only ensures units on existing
structure; force_rebuild/page-evidence combinations return 400, missing structure
409, PDF effective mode null/provenance unknown. UI uses this after explicit repair
or TOC commit, while post-upload Prepare uses the full flow. Explicit page evidence
on ordinary reads now uses source-validated persisted page spans (LAYOUT-03), with
character-anchor fallback when unavailable; no read-time extraction/language LLM.
Evidence: readiness, page-evidence and PDF-mode route tests.

### PDF Extraction Mode API Contract (2026-10-15)

Implemented contract (PDF-MODE-01): `pdf_extraction_mode: fast | ocr | auto`, omitted => fast,
on explicit `/documents/prepare` and `/documents/prepare-task-layout` requests.
This is separate from `mode=base/free_qa` and `structured_parser_mode=common/...`.
Both routes forward the enum through the existing coordinator's preparation pipeline;
preparation owns execution/reuse. Unknown/null values fail validation before work.
TXT omitted/fast retains normal preparation; explicit ocr/auto returns 422.
Successful prepare responses add `pdf_extraction` with `requested_mode`, nullable
`effective_mode`, and `provenance` (unknown/not_applicable/applied/reused).
Only applied/reused permits an effective mode, which must match requested mode.
Prepare-task-layout uses a dedicated response subclass; ordinary layout reads are unchanged.
Typed failures contain `doc_name`, `reason`, `detail`, and `pdf_extraction`.
HTTP mappings: mode_unavailable=501; mode_not_applicable/no_text/quality_failed/
budget_exceeded=422; engine_unavailable=503; extraction_failed=502. Standard request
validation remains a separate 422 detail-array shape. Engine failure mappings are
contract-tested with injected errors, not evidence of implemented mode executors.
No new cancellation/job endpoint is implied. Raw upload stores only the source;
UI subsequently starts a visible explicit preparation using the returned unique
name. Management/Open reads cannot change mode or trigger preparation. Existing
artifacts lacking mode provenance remain readable and unknown, never backfilled
as fast by a read. Mode-changing rebuild is an explicit preparation operation.

PDF-MODE-05 removes the temporary 501 gate: both routes now dispatch fast/ocr/auto
through source-snapshot and identity-aware preparation. Auto retains existing
selective-policy/configuration limits. Engine availability and quality/budget
errors remain explicit; this is not a real-document accuracy acceptance claim.
TXT success reports not_applicable with no effective mode. Internal callers that
omit the pipeline argument retain legacy execution and unknown provenance; they
are not advertised as fast. No live deployment was performed. UI LIB-06 still owns
the selector; ordinary layout reads are unchanged. Evidence:
`test_pdf_extraction_mode_contract.py` and `test_pdf_mode_preparation.py`.

`main.py` 是 FastAPI 入口，負責 route 註冊、request 映射到 coordinator，以及 coordinator 結果映射回 API response schema。 **[Code-Confirmed]**

## 2. Position in Overall Architecture

- API Layer

## 3. Key Files

| File | Responsibility | Notes |
|---|---|---|
| `main.py` | FastAPI app 建立、routes、HTTP error mapping | 綁定 `QACoordinator` / `SectionTaskCoordinator` **[Code-Confirmed]** |

## 4. Main Responsibilities

1. 建立 app 與健康檢查端點。 **[Code-Confirmed]**
2. `/documents/prepare`、`/documents/ask` 等 route dispatch。 **[Code-Confirmed]**
3. section/chapter summary/quiz route 映射。 **[Code-Confirmed]**
4. task-layout route 將 internal layout DTO 映射為 public chapters-first response。 **[Code-Confirmed]**
5. 統一 exception -> HTTP status translation（部分路徑）。 **[Code-Confirmed]**
6. 暴露 lightweight document list/search route，供 UI 取得可選 `doc_name` 候選；不返回 hierarchy/content heavy payload。 **[Code-Confirmed]**

## 5. Non-Responsibilities

1. 不應承擔 parser/task business logic。 **[From HLD]**
2. 不應直接操作 structured persistence。 **[From HLD]**
3. 不應在 route 內實作深層 cache/recommendation 決策。 **[From HLD]**

## 6. Important Data Structures / Contracts

- FastAPI route contracts via `api_schemas.py`
- `/documents/task-layout` response 的 chapters-first + diagnostics 映射

## 7. Route-to-Coordinator Mapping

| Endpoint | Coordinator / Service Entry | Path Type |
|---|---|---|
| `GET /health` | direct route handler | stateless read |
| `GET /documents` | `DocumentArtifactRepository.list_documents` | lightweight read/search |
| `GET /documents/management` | structured list + raw source metadata merge | lightweight management status read |
| `POST /documents/upload` | route-local raw source registration | explicit raw-source write |
| `POST /documents/lifecycle/delete` | active structured repository plus raw/derived file cleanup | explicit delete mutation |
| `POST /documents/lifecycle/rename` | file-backed document candidate rename | explicit rename mutation |
| `POST /documents/prepare` | prepare pipeline via coordinator path | write-capable orchestration |
| `POST /documents/ask` | `QACoordinator` | runtime QA |
| `POST /documents/task-layout` | `SectionTaskCoordinator.get_document_task_layout` | projection/read-centric |
| `POST /documents/prepare-task-layout` | prepare pipeline + task-layout projection | prepare-then-read orchestration |
| `POST /documents/section-summary` | `SectionTaskCoordinator.summarize_section` | write path |
| `POST /documents/section-quiz` | `SectionTaskCoordinator.generate_section_quiz` | write path |
| `POST /documents/summarize-chapter` | chapter summary path | write path |
| `POST /documents/chapter-quiz` | chapter quiz path | write path |
| `POST /documents/reparse-structure` | explicit reparse path | explicit mutation |
| `POST /documents/reading-interactions/insight/read` | `SectionTaskCoordinator.read_analysis_artifact` | read persisted current artifact |
| `POST /documents/reading-interactions/insight/generate` | `SectionTaskCoordinator.generate_analysis_artifact` | explicit interaction write path |
| `POST /documents/reading-interactions/insight/refresh` | `SectionTaskCoordinator.refresh_analysis_artifact` | explicit replacement write path |
| `POST /documents/reading-interactions/quiz/read` | `SectionTaskCoordinator.read_quiz_artifact` | read persisted current quiz artifact |
| `POST /documents/reading-interactions/quiz/generate` | `SectionTaskCoordinator.generate_quiz_artifact` | explicit target-agnostic quiz write path |
| `POST /documents/reading-interactions/quiz/refresh` | `SectionTaskCoordinator.refresh_quiz_artifact` | explicit quiz replacement write path |
| `POST /documents/reading-interactions/critical-thinking/read` | `SectionTaskCoordinator.read_critical_thinking_session` | read persisted session state |
| `POST /documents/reading-interactions/critical-thinking/generate-question` | `SectionTaskCoordinator.generate_critical_thinking_question` | explicit question/session write path |
| `POST /documents/reading-interactions/critical-thinking/submit-answer` | `SectionTaskCoordinator.submit_critical_thinking_answer` | explicit answer/evaluation write path |
| `POST /documents/reading-interactions/critical-thinking/retry-evaluation` | `SectionTaskCoordinator.retry_critical_thinking_evaluation` | explicit evaluation retry write path |

## 8. Projection-Only and Mutation Boundary

1. `/documents/task-layout`：projection/read path，不應 hidden mutation。 **[Code-Confirmed] + [From HLD]**
2. diagnostics 是 runtime projection，不應在 route 層回寫 profile。 **[Code-Confirmed] + [From HLD]**
3. summary/quiz/reparse 端點屬明確 mutation path。 **[Code-Confirmed]**
4. `GET /documents`：lightweight discovery/read path，只返回 document metadata candidates；不讀取或返回 chapter/section/task-unit content。 **[Code-Confirmed]**

## 8.1 Document List API Contract

`GET /documents` supports the Reader UI document combo box without changing task-layout. The UI opens the combo box by calling the list form without `q`:

```http
GET /documents?limit=200
```

Query parameters:

- `q`: optional case-insensitive substring query against `doc_name` and title; not required for the Reader UI combo-box open behavior.
- `limit`: optional bounded result count.

Response shape:

- `items[]`: ordered document candidates.
- each item includes `doc_name`, optional `title`, and `source`.
- `query`: normalized query string used for filtering.
- `total`: number of items returned after filtering and limit.

Boundary:

- read-only
- no hidden prepare/reparse
- no hierarchy mutation
- no profile diagnostics write-back
- no task-layout heavy payload
- no `content_blocks`
- backend storage implementation remains behind repository/container selection

## 9. Manual Reparse Policy

1. 目前不自動觸發 parser mode 切換。 **[From Proposal] + [From HLD]**
2. 建議由 recommendation 提示使用者手動 reparse。 **[From Proposal]**
3. force refresh 仍是必要機制，用於清除 cache 相關問題。 **[From Proposal]**

## 10. Main Flows Involving This Module

1. prepare endpoint flow
2. ask endpoint flow
3. section/chapter summary & quiz endpoint flow
4. task-layout endpoint flow
5. reparse endpoint flow

（此模組僅負責 mapping/dispatch，不負責內部演算法） **[Code-Confirmed]**

## 11. Persistence / Side Effects

- read persistence：否（由 coordinator/pipeline/repository 處理）
- write persistence：否（由 coordinator/repository 處理）
- mutate structured document：否（間接觸發，不在本檔落盤）
- generate runtime projection：否（僅映射現有 DTO）
- call LLM：否（透過 coordinator/service）
- diagnostics only：否（只是傳遞 diagnostics payload）

## 12. Known Legacy / Compatibility Behavior

No known legacy compatibility responsibility（route 層不直接管理 sections/structure_nodes 兼容）。 **[Code-Confirmed]**

## 13. Current Risks

1. risk：route 映射邏輯與 coordinator DTO 漂移
- why：可能出現欄位缺失或語義不一致
- guardrail：integration tests + schema contract checks

2. risk：HTTP status mapping 不一致
- why：客戶端難以穩定處理失敗分支
- guardrail：集中化 failure reason mapping 規則

3. risk：public response 不慎暴露 heavy payload
- why：性能與隱私風險
- guardrail：持續 no-heavy-payload regression tests

4. risk：prepare-then-read route 被 UI 當成普通 read route 使用
- why：`/documents/prepare-task-layout` may first execute document preparation before returning task-layout. For already prepared documents, repeated UI selection can therefore enter OCR/language/profile stages and indirectly call LLM even when the client only expects to read an existing layout. **[Code-Observed] + [Inferred]**
- guardrail：document selection/read flows should prefer `/documents/task-layout` when an active structured document/layout exists. `/documents/prepare-task-layout` should be reserved for explicit prepare, first-time load, repair, or fallback flows with clear observability. **[Future Direction]**

## 14. Open Questions for Maintainer

1. 是否要把 endpoint failure reason code 系統化（尤其 cache invalidation 可觀測性）？
2. `profile_diagnostics` 在 API 層是否要有專屬版本號或 contract policy？
3. 是否需要 route-level observability doc（request id / correlation id）？

## 15. Suggested Next Documentation Improvements

1. 增加 endpoint flow mapping 圖（route -> coordinator -> response）。
2. 補 API error handling policy。
3. 補 task-layout response contract appendix（含 diagnostics）。

## 16. Future Direction Note: Manual Structure Override Route Boundary

> 本節記錄 source-agnostic manual structure override 的 route-layer boundary；不代表目前 implementation。 **[Maintainer-Provided] + [Future Direction]**

1. manual structure override routes should support any document type, not only OCR/PDF. **[Maintainer-Provided]**
2. route design should separate validation/preview from commit reparse. **[Future Direction]**
3. validation/preview route should be read/analysis-oriented: it may load raw text or page evidence through the coordinator, but must not persist hierarchy, task-layout metadata, profile diagnostics, or artifacts. **[Future Direction]**
4. commit route should be an explicit mutation path, either by extending `/documents/reparse-structure` with `parser_mode=manual_structure` or by adding a clearly named manual-structure reparse endpoint. **[Future Direction]**
5. `/documents/task-layout` must not accept manual structure edits and must not trigger hidden reparse; it continues to project the current active structured hierarchy. **[From HLD] + [Future Direction]**
6. route-level validation should reject unknown manual parser modes, malformed manual plans, unsupported anchors, and requests that try to combine preview-only and commit-only semantics. **[Future Direction]**
7. HTTP mapping should distinguish 400 malformed request, 404 missing document/source, 409 stale source evidence or conflicting active structure version, and 422 unprojectable manual structure plan. **[Future Direction]**
8. route response should not expose raw text or heavy content payload; preview may expose normalized hierarchy labels/ranges and validation errors only. **[Future Direction]**
9. This note does not add endpoints, source code, schema fields, or runtime behavior in this pass. **[Doc-Confirmed]**

## 16.1 Document Management, Upload, And Lifecycle Routes

> 本節記錄 file-management / upload 的 route-layer boundary. The explicit raw-source
> upload route, lightweight management status route, explicit delete lifecycle route, and
> explicit rename lifecycle route are implemented. **[Code-Confirmed]**

1. `GET /documents` remains the current lightweight discovery route for compact reader
   document selection. **[Code-Confirmed]**
2. `GET /documents/management` is the richer lightweight management/status read route.
   It merges structured document candidates with raw source metadata and returns source
   status plus preparation status without raw text, hierarchy, content blocks, profile
   payloads, retrieval records, or artifacts. **[Code-Confirmed]**
3. Management rows mark structured artifacts as `preparation_status=ready`, raw-only
   candidates as `preparation_status=not_prepared`, and same-name raw+structured
   candidates as one merged row preserving raw source metadata while keeping structured
   readiness. **[Code-Confirmed]**
4. `POST /documents/upload` is an explicit multipart raw-source write route. It stores a
   supported `.txt` or `.pdf` raw source candidate under the raw document directory and
   returns `DocumentUploadResponse` with lightweight source metadata. **[Code-Confirmed]**
5. The upload route accepts optional `title` and `conflict_policy` fields. Conflict
   policy supports `reject`, `replace`, and `create_distinct`; default duplicate handling
   is reject with a typed `duplicate_conflict` response. **[Code-Confirmed]**
   `create_distinct` preserves the existing raw source and resolves the upload to a
   backend-owned unique filename/doc_name, for example `Book (1).txt` and `Book (1)`.
   **[Code-Confirmed]**
6. Upload route failures return the typed upload response shape with stable failure
   reasons such as unsupported file type, unsafe filename, duplicate conflict, missing
   source, validation failure, file too large, or storage failure. **[Code-Confirmed]**
7. Upload routes must not silently call `/documents/prepare`, `/documents/task-layout`,
   OCR, profile generation, reparse, retrieval build, or reading-interaction generation.
   The current route only writes raw source bytes and lets `GET /documents` discover the
   raw candidate. **[Code-Confirmed] + [From HLD]**
8. Explicit prepare/open flows remain separate: upload success does not equate upload
   with prepared layout readiness and returns `preparation_status=not_prepared`.
   **[Code-Confirmed]**
9. Document management lifecycle routes such as archive/delete/rename must be explicit
   mutation routes with stable HTTP status mapping and confirmation-oriented frontend
   affordances. They must use explicit lifecycle request/response schemas rather than
   extending list/read/open/upload routes with hidden destructive behavior.
   **[Code-Confirmed] + [Future Direction]**
10. Current upload storage is a minimal route-level raw-file registration slice over the
    existing file-backed raw directory. A future storage service should own validation,
    storage, conflict handling, user scope, and cleanup policy when document management
    evolves beyond this raw-source slice. **[Code-Confirmed] + [Future Direction]**
11. Route responses must preserve user-scope and raw-file ownership boundaries; no list or
    upload response should imply cross-user document sharing. **[From Proposal] + [Future Direction]**
12. Lifecycle route handlers validate requests and map lightweight responses.
    Structured deletion delegates to the active repository; raw source and derived
    directory cleanup currently remain route-local helpers. Moving those file helpers
    into a dedicated storage/service layer remains future work.
    **[Code-Confirmed] + [Future Direction]**
13. `POST /documents/lifecycle/delete` is an explicit destructive lifecycle mutation
    route. It requires `action=delete` and `confirm=true`, removes matching raw source
    and derived FAISS/profile files, then calls the active repository's
    `delete_document(doc_name)` for structured JSON or PostgreSQL data. It returns a
    lightweight `DocumentLifecycleActionResponse` with effect flags only.
    It does not call prepare, reparse, task-layout, OCR/profile generation,
    retrieval build, or reading-interaction generation. **[Code-Confirmed]**
14. Maintainer-scoped lifecycle MVP is limited to `delete` and `rename`. `archive` and
    `restore` remain reserved lifecycle vocabulary only and should not be implemented or
    exposed in the file-management UI until explicitly reprioritized. **[Maintainer-Provided] + [Future Direction]**
15. Delete routing is now explicit and confirmation-gated at the API boundary. The
    frontend must still ask the user for confirmation before issuing the delete request;
    backend `confirm=true` is a route-level guard, not a replacement for user-facing UI
    confirmation. **[Code-Confirmed] + [Maintainer-Provided]**
16. Future rename routing must be an explicit mutation that changes backend-owned
    document identity or display identity according to storage policy, rejects conflicts
    deterministically, and must not be simulated through upload replace/create-distinct
    flows. **[Maintainer-Provided] + [Future Direction]**
17. `POST /documents/lifecycle/rename` is an explicit lifecycle mutation route. It
    requires `action=rename` and `new_doc_name`, renames matching file-backed raw source,
    structured JSON artifact, and derived FAISS/profile directory when present, updates
    lightweight top-level structured JSON identity metadata, rejects target conflicts
    deterministically, and returns `DocumentLifecycleActionResponse` effect flags only.
    It does not call prepare, reparse, task-layout, OCR/profile generation, retrieval
    build, reading-interaction generation, or upload conflict-policy behavior.
    **[Code-Confirmed]**

### Delete Consistency And Recovery (2026-10-14)

- Root cause: deleting files alone left an active PostgreSQL document visible through
  both `GET /documents` and `GET /documents/management`. Raw-file absence does not prove
  document deletion. **[Code-Confirmed]**
- Structured cleanup follows the selected repository, without falling back to inactive
  structured JSON storage. `structured_artifacts_affected` includes DB document deletion
  and its owned rows; `derived_resources_affected` reports file-directory cleanup.
  **[Code-Confirmed]**
- File cleanup precedes repository deletion so a file failure retains the structured
  discovery record. Filesystem and PostgreSQL writes do not share an atomic transaction;
  a storage failure may leave files already removed, but must never return accepted.
  Retry removes remaining resources, including a database-only document. A fully absent
  target returns `404 / missing_document`. **[Code-Confirmed]**
- Live verification: the reported `《歸來》白事／喪葬／鄉村儀式素材整理稿` residual
  document (former DB id 18) was deleted through the corrected route. Both catalogs
  exclude it, the FAISS directory is absent, and other catalog names are unchanged.
  Evidence: `scripts/test_document_delete_route.py`, `scripts/test_postgres_document_delete.py`,
  local Docker API/SQL verification on 2026-10-14. **[Test-Confirmed]**

### Upload And Preparation Timing Logs (2026-10-14)

Each HTTP request receives a server-generated `request_id`, propagated to synchronous
worker logs and returned as `X-Request-ID`. Upload and prepare remain separate requests;
correlate them by the final backend-owned `doc_name` and timestamp, then filter each
request id for its stages. Health success probes remain suppressed. **[Code-Confirmed]**

`stage_started`, `stage_finished`, and `stage_failed` include a unique `stage_id`;
finish/failure includes monotonic `duration_ms`. Stages cover `upload_receive`,
`upload_multipart_parse`, `upload_write`, `upload_hash`, the preparation pipeline,
PDF/OCR substeps, and `task_layout_projection`. `upload_stored` records the actual
filename/doc_name and size, explicitly marking `preparation_started=false`.
Upload rejections log stable reason codes. **[Code-Confirmed]**

Default output is stderr/container logs plus rotating `data/logs/deep-reader.log`
at INFO level. From repository root, follow
`tail -f Deep_Reflective_Reader/data/logs/deep-reader.log` or
`docker compose logs -f api`. Filter by `request_id=<X-Request-ID>` for one request.
**[Code-Confirmed]**

Timing is server-observed: `upload_receive` measures application body reading, not
browser/proxy time before the request reaches the API. Nested stage durations are
inclusive and must not be summed. `stage_finished` means the call returned, not that
all assets succeeded; check `prepare_result` readiness/error codes and HTTP status.
A start without a finish identifies in-progress/interrupted work, not a diagnosed
performance cause. Added logs exclude document bodies, prompts, credentials, and
exception text; this change does not investigate latency or alter processing policy.
**[Code-Confirmed]**

## 17. Route Mapping For TOC Anchor Evidence

> 本節支援 UI TOC editor 的 page-first anchor UX。
> 目前 `/documents/task-layout` 已可 pass through coordinator DTO anchor evidence；manual page-backed validation/commit routing 仍屬後續 checkpoint。 **[Code-Confirmed] + [Future Direction]**

1. `/documents/task-layout` maps optional chapter/section anchor evidence from coordinator DTOs into public schema for UI prefill, while keeping the endpoint projection-only and lightweight. **[Code-Confirmed]**
2. Route mapping does not compute page boundaries or char offsets itself; it pass-through maps validated DTO evidence from `section_tasks/` / `app/`. **[Code-Confirmed] + [From HLD]**
3. `POST /documents/manual-structure/validate` should map `page_range` validation failures separately from malformed payload once backend page evidence is available. **[Future Direction]**
4. `POST /documents/reparse-structure` with `parser_mode=manual_structure` should continue to be the explicit mutation path; page-backed commit must fail before persistence when page evidence is missing, stale, ambiguous, or unprojectable. **[Future Direction]**
5. No route may accept manual TOC edits through `/documents/task-layout`, and no route may trigger hidden reparse from task-layout read. **[From HLD] + [Maintainer-Provided]**
6. Anchor evidence route mapping does not expose raw text, page text, OCR geometry, task-unit content, or content blocks. **[Code-Confirmed]**

## 18. Future Direction Note: Task-Layout Read Versus Prepare Boundary

> 本節記錄 task-layout route governance；不代表目前 implementation 已改變。 **[Code-Observed] + [Future Direction]**

1. `/documents/task-layout` is the read-centric projection route for the current active structured hierarchy. It must not accept manual TOC edits and must not trigger hidden reparse. **[Code-Confirmed] + [From HLD]**
2. `/documents/prepare-task-layout` is a prepare-then-read convenience route. It may be appropriate for first-time preparation or explicit repair/fallback, but it is not equivalent to a pure task-layout read. **[Code-Observed] + [Inferred]**
3. UI document selection for an already known backend document should prefer the pure task-layout route and only use prepare-then-read when the read path reports missing/unavailable layout or when the user explicitly starts a prepare/repair flow. **[Future Direction]**
4. Route-level observability should expose whether a prepare-then-read request actually reused existing structured artifacts or entered raw/OCR/language/profile work. **[Future Direction]**
5. This boundary preserves the existing architecture rule that API routes dispatch to coordinator/service behavior and do not implement parser/cache authority directly. **[From HLD]**

## 19. Batch Task-Unit Content Route Boundary

> 本節記錄 route-layer batch content boundary。Batch route 已落地，single content route 仍保留。 **[Code-Confirmed]**

1. `POST /documents/{doc_name}/task-units/content` loads all requested task units for one selected section with one HTTP request. **[Code-Confirmed]**
2. The route maps public request schema fields `task_unit_ids`, `segmented`, and `include_raw_content` to coordinator batch read plus response serialization. **[Code-Confirmed]**
3. The response mapping preserves the existing per-task-unit content response shape and returns items in request order. **[Code-Confirmed]**
4. The existing `GET /documents/{doc_name}/task-units/{task_unit_id}/content` endpoint remains compatibility/fallback behavior. **[Code-Confirmed] + [Future Direction]**
5. The route remains a read-only on-demand content path and does not expand `/documents/task-layout`, trigger prepare/reparse, mutate profile diagnostics, write artifacts, or compute parser semantics in route code. **[Code-Confirmed] + [From HLD]**

## 20. Reading Interaction Route Exposure Audit

> 本節記錄 analysis / quiz / critical-thinking route planning 與目前 REST exposure audit。Analysis/insight, target-agnostic quiz, and critical-thinking session route families are now exposed from `main.py`. **[Code-Confirmed]**

### 20.1 Current Route Exposure

1. `main.py` currently exposes the legacy task routes `POST /documents/section-quiz` and `POST /documents/chapter-quiz`; these routes generate the older section/chapter quiz payloads through `SectionTaskCoordinator.generate_section_quiz(...)` and `generate_chapter_quiz(...)`. **[Code-Confirmed]**
2. `main.py` exposes generic analysis/insight routes for reading persisted analysis artifacts, explicitly generating analysis artifacts, and explicitly refreshing/replacing the current analysis artifact. These routes map shared reading target request schemas to app orchestration and return `AnalysisInteractionResponse`. **[Code-Confirmed]**
3. `main.py` exposes generic target-agnostic quiz routes for reading persisted quiz artifacts, explicitly generating quiz artifacts, and explicitly refreshing/replacing the current quiz artifact. These routes map shared reading target request schemas to app orchestration and return `QuizInteractionResponse`; they do not use the legacy `/section-quiz` or `/chapter-quiz` endpoints. **[Code-Confirmed]**
4. `main.py` exposes generic critical-thinking session routes for reading existing session state, generating one question session, submitting an answer with evaluation, and retrying a failed evaluation. These routes map shared reading target request schemas plus session ids to app orchestration and return `CriticalThinkingSessionResponse`. **[Code-Confirmed]**
5. `main.py` imports and maps the analysis, quiz, and critical-thinking public response schemas, but it does not directly import or orchestrate `AnalysisInteractionOrchestrator`, `QuizInteractionOrchestrator`, or `CriticalThinkingSessionService`; service/orchestrator access remains behind `SectionTaskCoordinator` and DI assembly. **[Code-Confirmed]**
6. Existing task-layout and task-unit content routes must not be treated as substitutes for reading-interaction artifact routes. `/documents/task-layout` remains lightweight projection, and task-unit content routes remain on-demand content reads. **[From HLD] + [Code-Confirmed]**
7. Reading-interaction routes preserve `content_block_id` from schema-validated `content_block` target requests when dispatching to `SectionTaskCoordinator` and when mapping shared response envelopes. Route regressions cover insight read/generate/refresh, quiz read/generate/refresh, and critical-thinking read/generate-question/submit-answer/retry-evaluation. Route mapping itself still does not resolve block source text, perform hierarchy search, construct prompts, choose context, or own artifact persistence. **[Code-Confirmed]**
8. For UI per-block menus, route mapping expects the frontend to submit `target_type=content_block`, parent `task_unit_id`, optional parent `chapter_id` / `section_id`, and `content_block_id` copied from `content_blocks[].block_id` in the task-unit content response. The same target object is valid across insight, quiz, and critical-thinking generic route families. Route code must still avoid legacy quiz dispatch, UI state fields, hidden task-layout mutation, and route-local source slicing. **[Code-Confirmed]**

### 20.2 Future Route Boundary

1. Route families should be organized around explicit read versus write semantics: read persisted artifact/session state, generate or refresh artifact, generate critical-thinking session/question, submit answer, and retry evaluation. **[Maintainer-Provided] + [Future Direction]**
2. Read routes must never call LLM, generate artifacts, refresh artifacts, mutate task-layout, or prepare/reparse documents. Missing artifacts/sessions return a stable `not_generated` response. The exposed insight, quiz, and critical-thinking read routes follow this policy and are covered by route regressions. **[Code-Confirmed] + [From HLD]**
3. Generate/refresh routes are explicit mutation paths and should be the insertion point for future cost, quota, and permission checks. Insight and quiz generate/refresh currently preserve this route split. **[Code-Confirmed] + [Future Direction]**
4. Route mapping passes a validated reading target to app orchestration and avoids route-level hierarchy search, prompt assembly, context selection, parser decisions, or artifact persistence internals. **[Code-Confirmed] + [From HLD]**
5. Critical-thinking route mapping keeps three first-version write operations clear: generate question session, submit answer for evaluation, and retry failed evaluation. **[Code-Confirmed]**
6. Route responses expose structured statuses including missing/not-generated, insufficient-content, generation failed, validation failed, evaluation failed, and completed where appropriate. **[Code-Confirmed] + [Future Direction]**
7. No interaction route should expand `/documents/task-layout` or use task-layout as artifact truth; task-layout remains a lightweight hierarchy projection. **[From HLD] + [Future Direction]**

### 20.2.1 Reading Interaction HTTP Status Policy

The route-owned status policy is currently applied to analysis/insight, quiz, and critical-thinking route families. **[Code-Confirmed]**

| Condition / Envelope Status | HTTP Status | Route Meaning |
|---|---:|---|
| malformed request schema | `422` | FastAPI/Pydantic rejected the request before app orchestration. |
| missing document or hierarchy target | `404` | The request is well-formed, but the document or requested target id is absent. |
| `not_generated` | `200` | Read succeeded and no current artifact/session exists. |
| `completed` | `200` | Operation succeeded with completed artifact/session payload. |
| `insufficient_content` | `200` | Operation reached a terminal recoverable state without payload generation. |
| `question_generated` | `200` | Planned critical-thinking generation succeeded with a saved/generated question state. |
| `answer_submitted` | `200` | Planned critical-thinking answer submission state is accepted. |
| `evaluation_failed` | `200` | Planned critical-thinking evaluation failed as a retryable interaction state while preserving answer context. |
| `stale_target` | `409` | Target/source context is stale or conflicts with the active hierarchy. |
| `validation_failed` | `422` | Backend accepted the request but produced/received an invalid interaction payload state. |
| `generation_failed` | `502` | Upstream generation/model output failed validation or could not produce the requested artifact. |
| unexpected route exception | `500` | Route-level fallback for unclassified server failures. |

This policy does not move hierarchy search, prompt construction, context selection, or artifact persistence into `main.py`; it only converts app DTO status or route exceptions into stable HTTP status codes. **[Code-Confirmed] + [From HLD]**

### 20.3 Proposed Frontend-Facing Route Family

> 本節記錄 route exposure plan and status。Analysis/insight, quiz, and critical-thinking endpoints are code-confirmed. **[Code-Confirmed]**

1. Read-first insight/analysis routes are exposed:
   - `POST /documents/reading-interactions/insight/read`
   - `POST /documents/reading-interactions/insight/generate`
   - `POST /documents/reading-interactions/insight/refresh`
2. Read-first quiz routes are exposed:
   - `POST /documents/reading-interactions/quiz/read`
   - `POST /documents/reading-interactions/quiz/generate`
   - `POST /documents/reading-interactions/quiz/refresh`
3. Critical-thinking session routes are exposed:
   - `POST /documents/reading-interactions/critical-thinking/read`
   - `POST /documents/reading-interactions/critical-thinking/generate-question`
   - `POST /documents/reading-interactions/critical-thinking/submit-answer`
   - `POST /documents/reading-interactions/critical-thinking/retry-evaluation`
4. All request bodies carry the shared reading target request object from `api_schemas.py`. Critical-thinking submit/retry requests carry the same target object plus `session_id`, preserving hierarchy-aware session lookup through app orchestration instead of route-local global session lookup. Route paths identify interaction kind and operation; target identity stays in the request body so the same route shape works for book/chapter/section/task-unit UI menus. **[Code-Confirmed]**
5. The `read` operation is what the UI should call when opening inline insight or drawer panels. Insight, quiz, and critical-thinking read routes return persisted state or `not_generated`; they do not synthesize content, call LLM, create sessions, or mutate artifacts. **[Code-Confirmed]**
6. `generate`, `refresh`, `generate-question`, `submit-answer`, and `retry-evaluation` are explicit write paths. `refresh` is distinct from read and keeps a clear future insertion point for permission/cost confirmation before replacing a current artifact. **[Code-Confirmed] + [Future Direction]**
7. The legacy `POST /documents/section-quiz` and `POST /documents/chapter-quiz` routes remain compatibility/task endpoints until replaced. They should not be documented to the frontend as the generic quiz drawer API. **[Code-Confirmed] + [Future Direction]**
8. Route handlers should only validate schema, call app orchestration, and map result/status to HTTP response. They must not perform hierarchy search, prompt construction, context selection, parser decisions, or hidden task-layout persistence mutation. **[From HLD] + [Future Direction]**

### 20.3.1 Future Summary And Note Route Families

> 本節規劃新增 `summary` / `note` route family 的 route-layer 邊界；不代表目前
> route 已實作。 **[Future Direction]**

1. Summary routes should follow the existing generic reading-interaction route family:
   read returns current persisted summary or `not_generated`; generate is an explicit
   write path; refresh is an explicit replacement path. **[Future Direction]**
2. Summary route handlers should perform schema mapping and HTTP status mapping only.
   Prompt construction, higher-level summary preparation, generation, validation, and
   persistence must remain in app/section_tasks boundaries. **[Future Direction]**
3. Note routes should be explicit user mutation/read routes, such as save/list/delete
   or create/update/delete/read. They must not call LLM-backed summary generation,
   insight generation, quiz generation, critical-thinking session creation, prepare,
   reparse, retrieval build, or task-layout mutation. **[Future Direction]**
4. Route responses for both actions should preserve the no-heavy-payload rule and must
   not expose raw target text, raw ancestor summaries, frontend drawer/menu state,
   draft editing state, or selected-row state. **[Future Direction]**

### 20.4 Frontend Exposure Completion Plan: Route/Test-Owned Tasks

> 本節把 frontend exposure 12-task plan 中由 `main.py` / route regression 擁有的任務固定為 implementation backlog。These tasks turn app orchestration into a stable frontend API surface without treating legacy quiz routes as the new drawer API. **[Code-Confirmed] + [Future Direction]**

| Task ID | Task | Scope | Completion Evidence |
|---|---|---|---|
| FE-INT-07 | Define route family and HTTP status mapping | Finalize request/response mapping and HTTP status policy for read, generate, refresh, submit-answer, and retry-evaluation operations. | Implemented for insight, quiz, and critical-thinking route families by `main.py` helpers and route regressions in `scripts/test_analysis_interaction_routes.py`, `scripts/test_quiz_interaction_routes.py`, and `scripts/test_critical_thinking_interaction_routes.py`. |
| FE-INT-08 | Add no-auto-generation read regressions | Protect read routes from LLM calls, artifact writes, session creation, prepare, reparse, task-layout mutation, and profile diagnostics write-back. | Implemented across the exposed read route family: insight read uses a read-only poison coordinator, quiz read proves absent artifacts return `not_generated` without write dispatch, and critical-thinking read proves missing sessions do not generate questions or submit/retry evaluation. |
| FE-INT-09 | Add insight vertical slice route tests | Cover insight read/generate/refresh over at least one hierarchy target. | Implemented by `scripts/test_analysis_interaction_routes.py`; tests prove inline insight responses use shared envelope plus insight payload and preserve read/write split. |
| FE-INT-10 | Add quiz vertical slice route tests | Cover quiz read/generate/refresh with strict item validation. | Implemented by `scripts/test_quiz_interaction_routes.py`; tests prove quiz routes use the new generic API, not legacy section/chapter quiz endpoints, map strict quiz items into drawer payloads, and reject invalid item/count/type shapes. |
| FE-INT-11 | Add critical-thinking session route tests | Cover read, generate-question, submit-answer, evaluation failure preservation, retry-evaluation, and completed evaluation. | Implemented by `scripts/test_critical_thinking_interaction_routes.py`; tests prove missing reads do not generate sessions, generated-but-unanswered sessions persist, answer submission is preserved on evaluation failure, retry completes evaluation, and retry does not regenerate the question. |
| FE-INT-12 | Synchronize implementation documentation and checklists after API exposure | After coding completes, update detailed design/checklists with actual route names, schema names, orchestration methods, tests, and any adjusted boundaries. | Implemented by synchronizing `main`, `api_schemas`, `app`, and `scripts` module memory with the code-confirmed insight, quiz, and critical-thinking route families; `progress.md` remains untouched because no progress-sync request/skill is active. |

Route/test-owned tasks must preserve `/documents/task-layout` as lightweight projection, keep task-unit content on the on-demand content route, and avoid route-level hierarchy search, prompt assembly, parser decisions, or artifact persistence internals. **[From HLD] + [Maintainer-Provided]**
