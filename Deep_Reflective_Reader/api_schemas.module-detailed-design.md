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
