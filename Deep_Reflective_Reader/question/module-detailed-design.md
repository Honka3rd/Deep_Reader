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
