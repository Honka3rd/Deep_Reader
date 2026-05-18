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

1. future question interaction 可能擴展到 content-block-level references，不再僅限 section/task_unit 粒度。 **[Maintainer-Confirmed] + [Inferred]**
2. future target model 可包含 `task_unit_id`、`content_block_id`、sentence range、paragraph range、quote span、evidence reference；但這些都屬 interaction semantics，不是 hierarchy ownership。 **[Maintainer-Confirmed] + [Inferred]**
3. content block 不是新的 persisted hierarchy level；`TaskUnit` 仍是主要 interaction container，且需保留 chapter/section/task_unit hierarchy context。 **[Code-Confirmed] + [Maintainer-Confirmed]**
4. question targeting 應維持 id-based hierarchy context（chapter/section/task_unit/content_block ids），不得以 title-only path 取代 deterministic targeting。 **[Code-Confirmed] + [Maintainer-Confirmed]**
5. future reference resolution direction 應採 fail-fast：reference 缺失、歧義或 hierarchy path 不一致時直接報錯，不應隱式修正。 **[Inferred]**
6. question module 不擁有 content persistence，也不應成為 document truth source 或 artifact persistence owner。 **[Code-Confirmed] + [Inferred]**
7. question/evidence/source-span references 不得隱式 override source document truth，不得隱式觸發 artifact write-back 或其他 hidden persistence。 **[Maintainer-Confirmed] + [Inferred]**
8. future question layer 不得回退 flat task-unit truth model，不得重引 root sections mirror 或 structure_nodes 主流程語義。 **[Code-Confirmed] + [Maintainer-Confirmed]**
9. 後續若實作 rich-content QA targeting，需與 `section_tasks`、`shared`、`document_structure`、`evaluated_answer`、`prompts` 協調 quote/evidence/source-span reference contract。 **[Inferred]**

### 14.1 Needs Confirmation

1. sentence range / paragraph range / quote span 的 canonical reference format（offset、token index、或 block-local span）尚待統一。 **[Needs Confirmation]**
2. evidence reference 的最小跨模組 compatibility contract（`question` 與 `evaluated_answer` 共享欄位）尚待定義。 **[Needs Confirmation]**
