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
| `POST /documents/prepare` | prepare pipeline via coordinator path | write-capable orchestration |
| `POST /documents/ask` | `QACoordinator` | runtime QA |
| `POST /documents/task-layout` | `SectionTaskCoordinator.get_document_task_layout` | projection/read-centric |
| `POST /documents/section-summary` | `SectionTaskCoordinator.summarize_section` | write path |
| `POST /documents/section-quiz` | `SectionTaskCoordinator.generate_section_quiz` | write path |
| `POST /documents/summarize-chapter` | chapter summary path | write path |
| `POST /documents/chapter-quiz` | chapter quiz path | write path |
| `POST /documents/reparse-structure` | explicit reparse path | explicit mutation |

## 8. Projection-Only and Mutation Boundary

1. `/documents/task-layout`：projection/read path，不應 hidden mutation。 **[Code-Confirmed] + [From HLD]**
2. diagnostics 是 runtime projection，不應在 route 層回寫 profile。 **[Code-Confirmed] + [From HLD]**
3. summary/quiz/reparse 端點屬明確 mutation path。 **[Code-Confirmed]**

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

## 14. Open Questions for Maintainer

1. 是否要把 endpoint failure reason code 系統化（尤其 cache invalidation 可觀測性）？
2. `profile_diagnostics` 在 API 層是否要有專屬版本號或 contract policy？
3. 是否需要 route-level observability doc（request id / correlation id）？

## 15. Suggested Next Documentation Improvements

1. 增加 endpoint flow mapping 圖（route -> coordinator -> response）。
2. 補 API error handling policy。
3. 補 task-layout response contract appendix（含 diagnostics）。
