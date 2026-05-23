# api_schemas.py Detailed Design

## 1. Module Purpose

`api_schemas.py` 定義 REST API 的 request/response schema contract（Pydantic models），是 API 層與 orchestration 層之間的資料邊界。 **[Code-Confirmed]**

## 2. Position in Overall Architecture

- API Layer

## 3. Key Files

| File | Responsibility | Notes |
|---|---|---|
| `api_schemas.py` | REST DTO 定義與輸入校驗 | 含 prepare/ask/task-layout/summary/quiz/reparse schemas **[Code-Confirmed]** |

## 4. Main Responsibilities

1. 定義 endpoint payload contract。 **[Code-Confirmed]**
2. 對關鍵請求做基礎 validation（例如 chapter_id/chapter_title at-least-one）。 **[Code-Confirmed]**
3. 約束 task-layout public response 為 chapters-first 結構。 **[Code-Confirmed]**
4. 定義 `ProfileStructureDiagnosticsResponse` mixed-source 語義說明。 **[Code-Confirmed]**

## 5. Non-Responsibilities

1. 不負責 business logic。 **[From HLD]**
2. 不負責 persistence 寫入。 **[From HLD]**
3. 不應承擔 parser strategy 決策。 **[From HLD]**

## 6. Important Data Structures / Contracts

- `PrepareDocumentRequest/Response`
- `GetDocumentTaskLayoutRequest`
- `DocumentTaskLayoutResponse`
- `SummarizeChapterRequest`, `ChapterQuizRequest`
- `ProfileStructureDiagnosticsResponse`

## 7. Public API vs Internal DTO Boundary

| Layer | Scope |
|---|---|
| Public API Schema (`api_schemas.py`) | 對外契約，應穩定、可版本化 |
| Internal DTO (`section_tasks/document_task_layout.py`) | 協調層與 service 層中間模型，可保留 transitional fields |

原則：internal transitional fields 不應直接暴露為 public API。 **[Code-Confirmed] + [From HLD]**

## 8. Main Flows Involving This Module

1. API request parse/validate flow。
2. API response serialization flow。

（此模組不直接參與 prepare/parse/artifact 算法流程） **[Code-Confirmed]**

## 9. No-Heavy-Payload Contract

`task-layout` 與 diagnostics 相關 schema 不應暴露 heavy payload，包括但不限於：
- `raw_text`
- `section.content`
- `summary.content`
- `quiz.items`
- 問答逐題全文內容

**[Code-Confirmed] + [From HLD]**

## 10. Cache/Validity Reason Code Surface (Current)

目前 schema 已支持 validity/reason 類欄位，但完整 reason-code taxonomy 仍待文件化統一。 **[Code-Confirmed] + [Needs Confirmation]**

方向：保留 API 可觀測 invalidation reason code。 **[From Proposal] + [Needs Confirmation]**

## 11. Persistence / Side Effects

- read persistence：否
- write persistence：否
- mutate structured document：否
- generate runtime projection：否（僅 schema）
- call LLM：否
- diagnostics only：否（只定義 diagnostics payload 形狀）

## 12. Known Legacy / Compatibility Behavior

No known legacy compatibility responsibility（僅 DTO 契約層）。 **[Code-Confirmed]**

## 13. Current Risks

1. risk：schema docstring 與實作行為漂移
- why：使用者理解成本上升
- guardrail：schema regression + doc sync

2. risk：internal transitional fields 誤暴露為 public API
- why：可能破壞 hierarchy-first外部契約
- guardrail：保持 task-layout response 僅 chapters-first映射

3. risk：validation 規則與 coordinator 語義不一致
- why：出現 400/500 邊界錯亂
- guardrail：request validator + integration tests

## 14. Open Questions for Maintainer

1. `profile_diagnostics` 是否要升級成獨立 API contract 文檔章節？
2. cache invalidation reason code 若 API 化，是否放在 diagnostics 或 recommendation metadata？
3. 是否要在 schema 層標記更多 compatibility/deprecation 註釋？

## 15. Suggested Next Documentation Improvements

1. 增加 endpoint-to-schema mapping 表。
2. 增加 error status mapping 表（validation/business/runtime）。
3. 補 no-heavy-payload 保證清單。

## 16. Future Direction Note: Rich Task-Unit Content Schema Evolution

> 本節同時記錄「已落地的 schema normalization」與「後續 future-direction」。  
> 已落地項目為 backward-compatible additive evolution，不引入 API breaking change。 **[Code-Confirmed]**

1. 已正式定義 top-level `TaskUnitContentBlockResponse`，作為 task-unit on-demand rich-content API 的官方 block contract。 **[Code-Confirmed]**
2. `TaskUnitContentResponse` 已正式收斂為 `content`（compatibility/simple rendering）+ `content_blocks`（preferred future interaction payload）雙軌 additive 形狀。 **[Code-Confirmed]**
3. `content` 仍保留，未標記 deprecated，既有 clients 可持續使用 simple rendering path。 **[Code-Confirmed]**
4. block optional 欄位（`block_type` / `artifact_ids` / `metadata`）在目前 adapter 路徑下可為 `null`，且 response serialization 形狀固定。 **[Code-Confirmed]**
5. 上述 schema normalization 不包含 persistence migration、不包含 task-layout heavy payload 擴張、不包含 retrieval/LLM integration。 **[Code-Confirmed]**

6. future `TaskUnitContentResponse` 可由 simple `content: str` 演進為更完整 structured content blocks / segments。 **[From Proposal] + [Inferred]**
7. future rich-content schema 可引入 `content_block_id` / `content_segment_id` 細分互動語義（若與現有 `block_id` 需要分層命名再決策）。 **[From Proposal] + [Needs Confirmation]**
8. future block-level metadata 可擴展為 artifact metadata（discovery/availability）與 evidence/quote/annotation target metadata。 **[Inferred]**
9. future content blocks 可作 QA evidence target、quote target、annotation target、retrieval grounding target。 **[From Proposal] + [Inferred]**
10. content blocks 是 API interaction target，不是 persisted hierarchy level；不取代 hierarchy-first contract `chapters[].sections[].task_units[]`。 **[Code-Confirmed] + [From HLD]**
11. evidence/grounding metadata 不等同 parser authority，亦不得成為 persistence truth source。 **[From HLD] + [Inferred]**
12. `task-layout` response 邊界不變：保持 lightweight metadata/projection，不返回 heavy content payload；rich content 仍走 on-demand content API path。 **[Code-Confirmed] + [From HLD]**
13. content-block artifact target metadata 已以 `ArtifactTargetRefResponse` 正式化為 response-only pass-through contract（metadata only，不是 artifact persistence truth / existence proof）。 **[Code-Confirmed]**
14. task-unit content response 的 artifact target metadata glossary 目前僅允許：`source_hash`、`content_block_id`、`quote_span_start`、`quote_span_end`、`schema_version`。 **[Code-Confirmed] + [Maintainer-Confirmed]**
15. content endpoint context 下的最小 target constraints：`content_block` level 必須含 `task_unit_id + content_block_id`；`task_unit` level 必須含 `task_unit_id`。其他 level 保留 enum 表達能力，但不宣稱本 endpoint 支援 artifact write semantics。 **[Code-Confirmed] + [Maintainer-Confirmed]**
16. `target_level` 使用 shared single-source enum（`shared.artifact_target_model.ArtifactTargetLevel`）驗證，非法值在 schema/shared boundary fail-fast；不做 silent coercion/silent fallback。 **[Code-Confirmed] + [Maintainer-Confirmed]**
