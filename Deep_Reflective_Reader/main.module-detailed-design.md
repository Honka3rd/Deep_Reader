# main.py Detailed Design

## 1. Module Purpose

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
| `POST /documents/prepare` | prepare pipeline via coordinator path | write-capable orchestration |
| `POST /documents/ask` | `QACoordinator` | runtime QA |
| `POST /documents/task-layout` | `SectionTaskCoordinator.get_document_task_layout` | projection/read-centric |
| `POST /documents/prepare-task-layout` | prepare pipeline + task-layout projection | prepare-then-read orchestration |
| `POST /documents/section-summary` | `SectionTaskCoordinator.summarize_section` | write path |
| `POST /documents/section-quiz` | `SectionTaskCoordinator.generate_section_quiz` | write path |
| `POST /documents/summarize-chapter` | chapter summary path | write path |
| `POST /documents/chapter-quiz` | chapter quiz path | write path |
| `POST /documents/reparse-structure` | explicit reparse path | explicit mutation |

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
