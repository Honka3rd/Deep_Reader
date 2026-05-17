# question Detailed Design

## 1. Module Purpose

`question/` 負責問題標準化、範圍（global/local）解析，以及相關枚舉契約，支援上下文選擇與 prompt routing。 **[Code-Confirmed]**

## 2. Position in Overall Architecture

- Task Layout / Task Unit Layer（QA query understanding sub-layer）

## 3. Key Files

| File | Responsibility | Notes |
|---|---|---|
| `question/qa_enums.py` | QA 相關 enum 契約 | `AnswerLevel/QuestionScope/PromptMode/ContextMode` **[Code-Confirmed]** |
| `question/standardized/standardized_question.py` | 標準化問題 DTO | user/doc language fields **[Code-Confirmed]** |
| `question/standardized/question_standardizer.py` | LLM 問題語言檢測/轉寫標準化 | strict JSON parse **[Code-Confirmed]** |
| `question/question_scope_keywords_provider.py` | scope keywords provider | 取自 language registry **[Code-Confirmed]** |
| `question/question_scope_resolver.py` | lexical+semantic(+LLM fallback) scope resolver | 含 resolution diagnostics **[Code-Confirmed]** |

## 4. Main Responsibilities

1. 定義問題處理必要 enum 與 DTO。 **[Code-Confirmed]**
2. 產生標準化問題（跨語言一致化）。 **[Code-Confirmed]**
3. 解析 global/local scope，並輸出方法/相似度等診斷資訊。 **[Code-Confirmed]**

## 5. Non-Responsibilities

1. 不直接做 retrieval。 **[Code-Confirmed]**
2. 不直接建構最終 context。 **[Code-Confirmed]**
3. 不負責 summary/quiz artifacts。 **[Code-Confirmed]**

## 6. Important Data Structures / Contracts

- `StandardizedQuestion`
- `QuestionScopeResolution`
- `LocalReferenceSignalResolution`
- `QuestionScope`
- `PromptMode`
- `ContextMode`

## 7. Module Relationships

- depends on: `language/`, `embeddings/`, `llm/`
- used by: `app/qa_coordinator.py`, `context/context_orchestrator.py`

## 8. Main Flows Involving This Module

1. QA ask flow 的 question standardize。 **[Code-Confirmed]**
2. scope resolution（lexical/semantic/llm fallback）。 **[Code-Confirmed]**

## 9. Persistence / Side Effects

- read/write persistence：否
- call LLM：是（standardizer + scope fallback）
- runtime cache：`QuestionScopeResolver.text_embedding_cache`（in-memory） **[Code-Confirmed]**

## 10. Known Legacy / Compatibility Behavior

1. `LanguageProfileRegistry` 上存在 backward-compatible alias methods 供 local signal 取得。 **[Code-Confirmed]**

## 11. Current Risks

1. risk：scope fallback 邏輯複雜，行為可解釋性下降
- why：使用者可能難理解為何判 global/local
- guardrail：保留 method/similarity diagnostics

2. risk：LLM fallback JSON parse 失敗
- why：scope 決策退回 local
- guardrail：明確 warn log + deterministic fallback

## 12. Open Questions for Maintainer

1. scope resolver 診斷是否要對外 API 暴露？ **[Needs Confirmation]**
2. standardizer 是否允許在某些語言直接 bypass LLM？ **[Needs Confirmation]**

## 13. Suggested Next Documentation Improvements

1. 補 scope decision flowchart（lexical -> semantic -> llm fallback -> local）。

## 14. Future Direction Note: Rich Content Question Interaction Preparation

> 本節屬 future-direction documentation/preparation，非當前 implementation。 **[Doc-Confirmed]**

1. future question interaction 可能擴展到 content-block-level references（含 sentence/paragraph-level evidence 或 quote target），不再僅限 section/task_unit 粒度。 **[Inferred]**
2. content block reference 是 interaction targeting semantics，不等於 hierarchy ownership；content block 不是新的 persisted hierarchy level。 **[Code-Confirmed] + [Inferred]**
3. question targeting 應維持 id-based hierarchy context（chapter/section/task_unit/content_block ids），不得以 title-only path 取代 deterministic targeting。 **[Code-Confirmed] + [Inferred]**
4. question module 不擁有 content persistence，也不應成為 document truth source 或 artifact persistence owner。 **[Code-Confirmed] + [Inferred]**
5. question/evidence references 不得隱式 override source document truth，亦不得隱式觸發 artifact/persistence write-back。 **[Inferred]**
6. 後續若實作 rich-content QA targeting，需與 `section_tasks`、`document_structure`、`evaluated_answer`、`prompts` 協調 reference contract 與 fail-fast error semantics。 **[Inferred]**
