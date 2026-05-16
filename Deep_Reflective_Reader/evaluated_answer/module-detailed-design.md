# evaluated_answer Detailed Design

## 1. Module Purpose

`evaluated_answer/` 將 retrieval score 映射為 answer strictness（strict/cautious/reject），供 prompt rule 選擇。 **[Code-Confirmed]**

## 2. Position in Overall Architecture

- Peripheral / Support Layer（QA quality gate）

## 3. Key Files

| File | Responsibility | Notes |
|---|---|---|
| `evaluated_answer/answer_mode.py` | `AnswerMode` DTO | level + reason **[Code-Confirmed]** |
| `evaluated_answer/question_relevance.py` | relevance evaluator | 以距離閾值映射 answer mode **[Code-Confirmed]** |

## 4. Main Responsibilities

1. 根據 retrieval 結果做回答嚴謹度分級。 **[Code-Confirmed]**
2. 提供 prompt rule 分支依據。 **[Code-Confirmed]**

## 5. Non-Responsibilities

1. 不執行 retrieval。 **[Code-Confirmed]**
2. 不生成答案內容。 **[Code-Confirmed]**
3. 不做 artifact persistence。 **[Code-Confirmed]**

## 6. Important Data Structures / Contracts

- `AnswerMode`
- `QuestionRelevanceEvaluator`
- `AnswerLevel`

## 7. Module Relationships

- used by: `app/qa_coordinator.py`, `prompts/prompt_assembler.py`
- depends on: `retrieval/search_metadata.py`, `question/qa_enums.py`

## 8. Main Flows Involving This Module

1. QA ask flow：search 後評估 strictness。 **[Code-Confirmed]**

## 9. Persistence / Side Effects

- read/write persistence：否
- side effect：console debug log（best_score）

## 10. Known Legacy / Compatibility Behavior

No known legacy compatibility responsibility.

## 11. Current Risks

1. risk：閾值固定，跨文檔泛化不佳
- why：可能過度 reject 或過度寬鬆
- guardrail：把閾值文件化並可配置化 **[Needs Confirmation]**

## 12. Open Questions for Maintainer

1. relevance threshold 是否要進 AppDIConfig？ **[Needs Confirmation]**

## 13. Suggested Next Documentation Improvements

1. 補 answer level calibration 指南。
