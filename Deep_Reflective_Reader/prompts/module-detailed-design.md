# prompts Detailed Design

## 1. Module Purpose

`prompts/` 負責把 profile、context、question、answer mode 組裝為最終 LLM prompt。 **[Code-Confirmed]**

## 2. Position in Overall Architecture

- Prompt / LLM Interaction Layer

## 3. Key Files

| File | Responsibility | Notes |
|---|---|---|
| `prompts/prompt_assembler.py` | QA 回答 prompt 組裝器 | 包含 rules/mode guidance/profile render **[Code-Confirmed]** |

## 4. Main Responsibilities

1. 產生不同 answer strictness 的 instruction rules。 **[Code-Confirmed]**
2. 根據 `PromptMode` 注入 local/retrieval/full-text guidance。 **[Code-Confirmed]**
3. 將 `DocumentProfile` 渲染為 prompt profile 區塊。 **[Code-Confirmed]**

## 5. Non-Responsibilities

1. 不執行模型呼叫。 **[Code-Confirmed]**
2. 不做 context selection。 **[Code-Confirmed]**
3. 不做 persistence。 **[Code-Confirmed]**

## 6. Important Data Structures / Contracts

- `PromptAssembler`
- `PromptMode`
- `AnswerMode`
- `StandardizedQuestion`

## 7. Module Relationships

- depends on: `profile/`, `question/`, `evaluated_answer/`
- used by: `context/token_budget_manager.py`, `app/qa_coordinator.py`

## 8. Main Flows Involving This Module

1. QA ask flow prompt build。 **[Code-Confirmed]**
2. token budget estimation（non-context prompt tokens）時使用同一 assembler。 **[Code-Confirmed]**

## 9. Persistence / Side Effects

- read/write persistence：否
- side effects：無

## 10. Known Legacy / Compatibility Behavior

No known legacy compatibility responsibility.

## 11. Current Risks

1. risk：prompt rules 擴充時與 API 行為語義脫鉤
- why：使用者可觀測結果不一致
- guardrail：維持 prompt mode/answer mode 文件與測試

## 12. Open Questions for Maintainer

1. prompt contract 是否要對外文件化（例如 readme/API docs）？ **[Needs Confirmation]**

## 13. Suggested Next Documentation Improvements

1. 補 prompt mode matrix（local/retrieval/fulltext）。
