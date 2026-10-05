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

## 14. Future Direction Note: Reading Interaction Prompt Contracts

> 本節記錄 analysis / quiz / critical-thinking prompt governance；不代表目前 implementation。 **[Maintainer-Provided] + [Future Direction]**

1. Reading interaction prompts should use fixed, versioned instruction templates per interaction type. Dynamic input should be limited to target metadata, selected context, session state, and allowed generation options. **[Maintainer-Provided] + [Future Direction]**
2. `analysis` prompt instruction should clearly ask for summary, reasoning/interpretation, and parsing/explanation over the resolved reading target, then require strict JSON output. **[Maintainer-Provided] + [Future Direction]**
3. `quiz` prompt instruction should provide only the valid quiz type enum, target-level max count, and context; the model may decide the type mix and generate fewer than max. **[Maintainer-Provided] + [Future Direction]**
4. Critical-thinking question prompt instruction must explicitly tell the model this is critical-thinking training and request one focused question for the resolved target. **[Maintainer-Provided] + [Future Direction]**
5. Critical-thinking evaluation prompt instruction should evaluate the user's answer against the original question and context, returning structured feedback and status without starting a new question. **[Maintainer-Provided] + [Future Direction]**
6. All interaction prompts must request strict structured JSON and must be paired with server-side schema validation. Prompt wording alone is not a correctness boundary. **[Maintainer-Provided] + [Future Direction]**
7. Prompt instructions should include an insufficient-content escape hatch so noisy OCR fragments or symbol-only units can produce a valid empty/insufficient result instead of hallucinated content. **[Maintainer-Provided] + [Future Direction]**
8. Prompt templates should record instruction version in artifact metadata to support future regeneration and validation policy changes. **[Future Direction]**

### 14.1 Artifact-Aware Prompt Inputs

1. Prompt templates for higher-level targets should clearly separate `Primary source context` from `Secondary lower-level artifact context`. **[Maintainer-Provided] + [Future Direction]**
2. Fixed instructions should tell the model to use lower-level artifacts for avoiding repetition, identifying coverage, and raising abstraction, while grounding final output in the primary source context. **[Maintainer-Provided] + [Future Direction]**
3. Quiz prompts should explicitly avoid repeating lower-level quiz questions unless repetition is pedagogically necessary and justified by the output schema. **[Maintainer-Provided] + [Future Direction]**
4. Critical-thinking prompts should ask the model to build on lower-level critical-thinking focus areas when available, moving from local comprehension toward synthesis, transfer, and critique. **[Maintainer-Provided] + [Future Direction]**
5. Prompt instructions must forbid treating lower-level artifacts as source truth or parser authority. **[From HLD] + [Maintainer-Provided]**
6. Structured output should include enough fields for the server to validate referenced-artifact metadata and deduplication/abstraction hints when those fields are part of the schema. **[Future Direction]**
