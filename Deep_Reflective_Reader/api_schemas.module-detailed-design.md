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
- `DocumentListItemResponse`
- `DocumentListResponse`
- `GetDocumentTaskLayoutRequest`
- `DocumentTaskLayoutResponse`
- `SummarizeChapterRequest`, `ChapterQuizRequest`
- `ManualStructureAnchorRequest`
- `ManualStructureEntryRequest`
- `ManualStructurePlanRequest`
- `ManualStructureValidationRequest`
- `ManualStructureValidationResponse`
- `ReparseDocumentStructureRequest`
- `ProfileStructureDiagnosticsResponse`
- `ReadingInteractionTargetRequest`
- `ReadingInteractionTargetResponse`
- `ReadingInteractionResponseEnvelope`
- `AnalysisInteractionReadRequest`
- `AnalysisInteractionGenerateRequest`
- `AnalysisInteractionRefreshRequest`
- `AnalysisArtifactPayloadResponse`
- `AnalysisInteractionResponse`
- `QuizInteractionReadRequest`
- `QuizInteractionGenerateRequest`
- `QuizInteractionRefreshRequest`
- `QuizArtifactItemResponse`
- `QuizArtifactPayloadResponse`
- `QuizInteractionResponse`
- `CriticalThinkingSessionReadRequest`
- `CriticalThinkingQuestionGenerateRequest`
- `CriticalThinkingAnswerSubmitRequest`
- `CriticalThinkingEvaluationRetryRequest`
- `CriticalThinkingEvaluationResponse`
- `CriticalThinkingSessionPayloadResponse`
- `CriticalThinkingSessionResponse`
- `ArtifactAwareInteractionMetadataResponse`

Critical-thinking submit and retry requests carry the shared `ReadingInteractionTargetRequest` plus `session_id` so route mapping can preserve hierarchy-aware session-store lookup through app orchestration. They do not define a route-local global session lookup contract. **[Code-Confirmed]**

### 6.1 Reading Interaction Route Integration Contract

The shared reading interaction schema family is now verified against the public REST route set for insight, quiz, and critical-thinking sessions. **[Code-Confirmed]**

1. Insight routes use `AnalysisInteractionReadRequest`, `AnalysisInteractionGenerateRequest`, `AnalysisInteractionRefreshRequest`, `AnalysisArtifactPayloadResponse`, and `AnalysisInteractionResponse`. **[Code-Confirmed]**
2. Quiz routes use `QuizInteractionReadRequest`, `QuizInteractionGenerateRequest`, `QuizInteractionRefreshRequest`, `QuizArtifactItemResponse`, `QuizArtifactPayloadResponse`, and `QuizInteractionResponse`. **[Code-Confirmed]**
3. Critical-thinking routes use `CriticalThinkingSessionReadRequest`, `CriticalThinkingQuestionGenerateRequest`, `CriticalThinkingAnswerSubmitRequest`, `CriticalThinkingEvaluationRetryRequest`, `CriticalThinkingEvaluationResponse`, `CriticalThinkingSessionPayloadResponse`, and `CriticalThinkingSessionResponse`. **[Code-Confirmed]**
4. All route families share `ReadingInteractionTargetRequest`, `ReadingInteractionTargetResponse`, `ReadingInteractionResponseEnvelope`, and `ArtifactAwareInteractionMetadataResponse` instead of defining route-local target, status, provenance, or UI-state variants. **[Code-Confirmed]**
5. Schema validation remains a boundary check only. It does not perform hierarchy lookup, prompt construction, persistence mutation, LLM calls, or frontend drawer/inline state management. **[Code-Confirmed] + [From HLD]**

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

`DocumentListResponse` follows the same lightweight API principle for document discovery:

- returns document candidates only
- does not expose raw text
- does not expose chapter/section/task-unit hierarchy
- does not expose task-unit content or `content_blocks`
- does not include diagnostics/profile payload

**[Code-Confirmed]**

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
2. `TaskUnitContentResponse` 已正式收斂為 `content_blocks`（preferred future interaction payload）+ nullable `content`（compatibility/debug payload）雙軌 additive 形狀。 **[Code-Confirmed]**
3. `content` 仍保留，未標記 deprecated；但預設回應可為 `null`，需透過顯式 `include_raw_content=true` 取得 legacy raw content。 **[Code-Confirmed]**
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
17. metadata glossary keys 由 `api_schemas.ARTIFACT_TARGET_METADATA_GLOSSARY_KEYS` 單一常量維護，schema validation 與 endpoint mapping 共用同一來源，避免雙份定義漂移。 **[Code-Confirmed]**

## 17. Future Direction Note: Manual Structure Override API Schema

> 本節記錄 source-agnostic manual structure override 的 public schema direction；不代表目前 implementation。 **[Maintainer-Provided] + [Future Direction]**

1. manual structure schemas should apply to any document type, not only OCR/PDF. **[Maintainer-Provided]**
2. public request naming should avoid OCR-only terms; prefer `manual_structure`, `manual_entries`, `manual_structure_mode`, or `manual_structure_plan` over OCR-specific vocabulary. **[Future Direction]**
3. schema should support validation/preview request and commit reparse request separately. **[Future Direction]**
4. manual entry schema should minimally include `title`, `level`, and one supported anchor form. **[Future Direction]**
5. anchor schema should be explicit and typed, for example `char_range`, `page_range`, or later `printed_page_number`; clients must not rely on implicit title matching as the sole locator. **[Future Direction]**
6. response schema should expose validation result, normalized entries, warnings, errors, preview hierarchy shape, and parse provenance preview without raw text/heavy content payload. **[Future Direction]**
7. commit reparse response may reuse/extend `ReparseDocumentStructureResponse`, but must identify `parser_mode=manual_structure` or equivalent source-agnostic mode. **[Future Direction]**
8. task-layout response schema should not be expanded with manual structure edit payload; task-layout remains lightweight projection. **[From HLD] + [Future Direction]**
9. user-defined manual structure is limited to two levels for now: `level=1` chapter and `level=2` section. `level>2` must be rejected by validation instead of folded. **[Maintainer-Provided] + [Future Direction]**
10. schema validation should distinguish malformed payload, unsupported anchor type, out-of-range anchor, overlapping range, empty projected range, invalid level sequence, unsupported depth, and stale source evidence. **[Future Direction]**
11. Implemented checkpoint: request-side manual structure schema now exists as `ManualStructureAnchorRequest`, `ManualStructureEntryRequest`, `ManualStructurePlanRequest`, and `ManualStructureValidationRequest`. It is source-agnostic, supports typed `char_range` / `page_range` anchors, enforces current two-level `chapter -> section` maximum depth, and does not add endpoint/runtime behavior. **[Code-Confirmed]**
12. Implemented checkpoint: validation/preview response schema now exists as `ManualStructureValidationResponse` with normalized entry, issue, preview chapter/section, and provenance preview DTOs. It is source-agnostic, keeps preview payload lightweight, validates the planned issue taxonomy, and does not add endpoint/runtime behavior. **[Code-Confirmed]**
13. Implemented checkpoint: commit reparse request schema can now represent `parser_mode=manual_structure` with a required `manual_structure` plan. Existing `common` / `llm_enhanced` modes remain accepted and reject manual plans; this checkpoint is schema-only and does not add endpoint/runtime behavior. **[Code-Confirmed]**
14. Manual structure commit route orchestration exists for explicit manual reparse, but page-backed manual commit remains incomplete until backend page-boundary evidence can validate and materialize `page_range` into hierarchy content. **[Code-Confirmed] + [Future Direction]**

## 18. Page-First Manual TOC API Support

> 本節支援 UI TOC editor 的 page-first anchor UX。
> 目前已完成 task-layout existing-structure anchor evidence response schema；page-backed manual validation/commit semantics 仍屬後續 checkpoint。 **[Code-Confirmed] + [Future Direction]**

1. Public manual-structure schemas should keep `char_range` as the universal fallback anchor and treat `page_range` as available only when backend page-boundary evidence is explicit and validated. **[Future Direction]**
2. `ManualStructureAnchorRequest` already has typed `page_range` vocabulary, but schema presence alone must not imply commit support. Validation/commit responses must distinguish "schema-valid but unsupported by available source evidence" from malformed payload. **[Code-Confirmed] + [Future Direction]**
3. Task-layout response schema now exposes lightweight existing-structure anchor evidence for UI prefill via `AnchorEvidenceResponse` on chapter/section nodes, without raw text, full page text, OCR boxes, or heavy layout payload. **[Code-Confirmed]**
4. Existing-structure anchor evidence should be optional per chapter/section and should support at least:
   - anchor type (`page_range` or `char_range`)
   - page indices or display page labels when validated
   - character offsets when no reliable page evidence exists
   - source/evidence availability state and warning code
5. The schema boundary must preserve hierarchy-first response shape: anchor evidence is metadata on `chapters[]` / `chapters[].sections[]`, not a second hierarchy source. **[From HLD] + [Future Direction]**
6. Any future `page_range` request/response schema must remain source-agnostic; PDFs are the first expected pageable source, not an OCR-only special case. **[Maintainer-Provided] + [Future Direction]**
7. `AnchorEvidenceResponse` currently supports `anchor_type`, `status`, `reason`, char offsets, page indices, and page labels as optional lightweight metadata. **[Code-Confirmed]**

## 19. Batched Task-Unit Content API

> 本節記錄 frontend content fan-out optimization 的 schema boundary。Batch schema 已落地為 backward-compatible additive API evolution；single task-unit content schema 仍保留。 **[Code-Confirmed]**

1. Batched content request allows the UI to request one selected section's ordered `task_unit_ids` through a single on-demand content API call instead of issuing one request per task unit. **[Code-Confirmed]**
2. `BatchTaskUnitContentRequest` keeps explicit render options `segmented` and `include_raw_content`, matching the existing single task-unit content endpoint semantics. **[Code-Confirmed]**
3. `BatchTaskUnitContentResponse.contents[]` returns ordered per-task-unit content responses that preserve the existing `TaskUnitContentResponse` shape rather than inventing a second content-block contract. **[Code-Confirmed]**
4. Request validation rejects empty task-unit id lists and duplicate ids at schema boundary; unknown ids and hierarchy-incompatible legacy documents fail fast through the same coordinator lookup semantics as the single endpoint. **[Code-Confirmed]**
5. Initial batch semantics are all-or-nothing; there is no partial-success item schema. **[Code-Confirmed]**
6. The existing single task-unit content endpoint remains the compatibility and fallback path. **[Code-Confirmed] + [Future Direction]**
7. Batch content is still on-demand rich content, not task-layout payload; `/documents/task-layout` must remain lightweight and must not return task-unit content or `content_blocks`. **[Code-Confirmed] + [From HLD]**
8. Batch schema evolution must not introduce persistence mutation, profile diagnostics write-back, parser authority, artifact write semantics, root `sections[]` flow, or `structure_nodes` as main flow. **[From HLD]**

## 20. Reading Target Interaction API Schema Exposure Audit

> 本節記錄 reading interaction API schema planning 與目前 schema exposure audit。Shared target/envelope schemas and artifact-aware metadata exist, while payload-specific schema families for generic reading-interaction endpoints are not yet complete. **[Code-Confirmed] + [Future Direction]**

### 20.1 Current Schema Exposure

1. `api_schemas.py` currently exposes `ArtifactAwareInteractionMetadataResponse` as metadata-only provenance for future artifact-aware interaction responses. It intentionally does not embed lower-level artifact payloads. **[Code-Confirmed]**
2. `api_schemas.py` now defines `ReadingInteractionTargetRequest` for generic `document|chapter|section|task_unit` target identity. It trims ids, normalizes `book` to `document`, rejects title-primary locator fields via `extra="forbid"`, and validates required/forbidden id combinations per target level. **[Code-Confirmed]**
3. `api_schemas.py` now defines `ReadingInteractionTargetResponse` and `ReadingInteractionResponseEnvelope` as shared response metadata for future reading-interaction route payloads. The envelope validates interaction type, status, artifact/session ids, provenance fields, reason-required failure statuses, critical-thinking-only statuses, and artifact-aware metadata without embedding UI behavior state. **[Code-Confirmed]**
4. `api_schemas.py` now defines public payload-specific request/response schemas for `analysis` artifacts, target-agnostic `quiz` artifacts, and `critical_thinking_session` lifecycle states, including read/generate/submit/retry requests, completed payloads, missing state, insufficient-content, and failure-state response validation. **[Code-Confirmed]**
5. Existing `SectionQuizResponse` and `ChapterQuizResponse` belong to the legacy section/chapter quiz endpoints and should not be treated as the full schema surface for the new target-agnostic quiz interaction API. **[Code-Confirmed]**

### 20.2 Future Schema Boundary

1. Public schemas should represent a generic reading target with `target_type=document|chapter|section|task_unit` and the required id for that level: `doc_name`, `chapter_id`, `section_id`, or `task_unit_id`. **[Maintainer-Provided] + [Future Direction]**
2. Optional parent ids may be included only for consistency validation and must not become title-based fallback. **[Maintainer-Provided] + [Future Direction]**
3. Read schemas and generate schemas must be separate. Read responses return persisted artifact/session state or a missing/not-generated status without implying generation. **[Maintainer-Provided] + [Future Direction]**
4. Analysis schemas should expose validated structured fields for summary, reasoning/interpretation, parsing/explanation, status, target metadata, artifact metadata, and validation/generation failure information when applicable. **[Maintainer-Provided] + [Future Direction]**
5. Quiz schemas should expose `short_answer`, `multiple_choice`, and `true_false` item types only, include answer payloads, enforce item count limits, and support insufficient-content status. **[Maintainer-Provided] + [Future Direction]**
6. Critical-thinking schemas should cover session generation, answer submission, and evaluation retry with statuses `question_generated`, `insufficient_content`, `answer_submitted`, `evaluation_failed`, and `completed`. **[Maintainer-Provided] + [Future Direction]**
7. All successful generated-artifact response schemas should carry schema version, prompt instruction version, context mode, source structure version, source hash or equivalent provenance where available, and target metadata. **[Maintainer-Provided] + [Future Direction]**
8. Validation should reject malformed target ids, unsupported quiz types, excessive quiz count, missing critical-thinking answer where required, and invalid model-output shapes. **[Maintainer-Provided] + [Future Direction]**
9. Interaction schemas must not expand `/documents/task-layout` with heavy artifact payloads and must not expose raw text by default. **[From HLD] + [Future Direction]**

### 20.3 Frontend-Required Reading Interaction Data Shape

> 本節是 frontend UI 對接所需的 public DTO planning。Shared target/envelope schemas are implemented; payload-specific insight/quiz/critical-thinking schemas remain future work. **[Code-Confirmed] + [Future Direction]**

1. `ReadingInteractionTargetRequest` carries a hierarchy identity object rather than loose route/body fields. Minimal fields are `doc_name`, `target_type`, and the one required id for the selected level (`chapter_id`, `section_id`, or `task_unit_id` where applicable). **[Code-Confirmed]**
2. Target titles are not accepted in request schema as locators. Response-side display title metadata is allowed through `ReadingInteractionTargetResponse`, but it is not target authority. **[Code-Confirmed] + [From HLD]**
3. `ReadingInteractionResponseEnvelope` includes `target`, `interaction_type`, `status`, optional `artifact_id` or `session_id`, `generated_at`, `updated_at`, optional `schema_version`, optional `prompt_instruction_version`, optional `source_structure_version`, optional `source_hash`, `reason`, and optional `artifact_context_metadata`. **[Code-Confirmed]**
4. Common read statuses include `not_generated`, `completed`, `insufficient_content`, `stale_target`, `generation_failed`, and `validation_failed`. Critical-thinking session responses additionally support `question_generated`, `answer_submitted`, and `evaluation_failed`. **[Code-Confirmed]**
5. Insight/analysis response is represented by `AnalysisInteractionResponse`, which composes `ReadingInteractionResponseEnvelope` with compact `AnalysisArtifactPayloadResponse` fields: `summary`, `reasoning`, `interpretation`, optional `explanation`, and optional `key_points`. Completed analysis responses require payload; missing, insufficient-content, stale-target, generation-failed, and validation-failed responses must not carry payload. **[Code-Confirmed]**
6. Quiz response is represented by `QuizInteractionResponse`, which composes `ReadingInteractionResponseEnvelope` with bounded drawer-oriented `QuizArtifactPayloadResponse.items[]`. Each `QuizArtifactItemResponse` carries `item_id`, `item_type`, `prompt`, optional `options[]`, `answer`, and optional `explanation`; allowed item types are `short_answer`, `multiple_choice`, and `true_false`. Completed quiz responses require payload; missing, insufficient-content, stale-target, generation-failed, and validation-failed responses must not carry payload. **[Code-Confirmed]**
7. Critical-thinking response is represented by `CriticalThinkingSessionResponse`, which composes `ReadingInteractionResponseEnvelope` with `CriticalThinkingSessionPayloadResponse`. It validates `question_generated`, `answer_submitted`, `evaluation_failed`, and `completed` lifecycle shapes; failed evaluation preserves `question` and `submitted_answer` and marks retry eligibility; completed sessions require evaluation feedback and may carry numeric score, suggested refinement, strengths, and improvements. **[Code-Confirmed]**
8. Frontend-local UI state such as drawer open state, selected menu item, pending textarea text, optimistic answer draft, or inline expanded/collapsed state must not be represented as backend truth schemas. **[From HLD] + [Maintainer-Provided]**

### 20.4 Artifact-Aware Interaction Metadata Schemas

1. Generated artifact response schemas should optionally expose artifact-aware context metadata: `artifact_context_mode`, `referenced_artifact_ids`, referenced artifact target levels/types, coverage counts, and deduplication/abstraction hint flags. **[Maintainer-Provided] + [Future Direction]**
2. Metadata should distinguish primary source evidence from secondary lower-level artifact references. **[Maintainer-Provided] + [Future Direction]**
3. Missing lower-level artifacts should serialize as empty/omitted secondary context metadata, not as an error. **[Maintainer-Provided] + [Future Direction]**
4. API schemas should not expose full lower-level artifact payloads inside higher-level artifact responses by default. **[Maintainer-Provided] + [Future Direction]**
5. Referenced artifact ids are observability/provenance metadata and must not become client-side hierarchy authority. **[From HLD] + [Future Direction]**

### 20.5 Frontend Exposure Completion Plan: Schema-Owned Tasks

> 本節把 frontend exposure 12-task plan 中由 `api_schemas.py` 擁有的任務固定為 implementation backlog。Shared target/envelope schema has an initial implementation, but the task remains open until it is verified against the first public REST route integration and payload-specific schemas. **[Code-Confirmed] + [Future Direction]**

| Task ID | Task | Scope | Completion Evidence |
|---|---|---|---|
| FE-INT-01 | Freeze shared reading interaction contract for route integration | Confirm `ReadingInteractionTargetRequest`, `ReadingInteractionTargetResponse`, `ReadingInteractionResponseEnvelope`, status vocabulary, and artifact metadata work as the single public schema foundation for all three route families. | Route-level request/response tests for insight, quiz, and critical-thinking all use the shared schemas without adding route-local target/status variants. |
| FE-INT-02 | Define insight/analysis payload schema | Add payload-specific request/response schemas for compact inline insight/analysis rendering. | Implemented by `AnalysisInteractionReadRequest`, `AnalysisInteractionGenerateRequest`, `AnalysisInteractionRefreshRequest`, `AnalysisArtifactPayloadResponse`, and `AnalysisInteractionResponse`; schema tests cover `summary`, `reasoning`, `interpretation`, optional `key_points`, shared envelope composition, missing state, insufficient content, and generation/validation failure. |
| FE-INT-03 | Define quiz artifact payload schema | Add drawer-oriented quiz request/response schemas with strict item type/count validation. | Implemented by `QuizInteractionReadRequest`, `QuizInteractionGenerateRequest`, `QuizInteractionRefreshRequest`, `QuizArtifactItemResponse`, `QuizArtifactPayloadResponse`, and `QuizInteractionResponse`; schema tests cover `short_answer`, `multiple_choice`, `true_false`, answers/explanations, max count, insufficient content, and invalid item shapes. |
| FE-INT-04 | Define critical-thinking session payload schema | Add session-oriented request/response schemas for read, question generation, answer submission, failed evaluation, completed evaluation, and retry eligibility. | Implemented by `CriticalThinkingSessionReadRequest`, `CriticalThinkingQuestionGenerateRequest`, `CriticalThinkingAnswerSubmitRequest`, `CriticalThinkingEvaluationRetryRequest`, `CriticalThinkingEvaluationResponse`, `CriticalThinkingSessionPayloadResponse`, and `CriticalThinkingSessionResponse`; schema tests cover `session_id`, question, submitted answer, evaluation feedback/score/refinement, status transitions, answer preservation on evaluation failure, and invalid missing-answer payloads. |

Schema-owned tasks must not expand `/documents/task-layout`, expose raw target content by default, accept frontend UI state as backend truth, or use title fields as target locators. **[From HLD] + [Maintainer-Provided]**
