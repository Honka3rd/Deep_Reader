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

## 14. Future Direction Note: Rich Content-Aware Evaluation Preparation

> 本節屬 future-direction documentation/preparation，非當前 implementation。 **[Doc-Confirmed]**

1. evaluated_answer 未來可能擴展為 content-block-aware answer evaluation，支援 content-block reference、quote target、sentence/paragraph-level evidence mapping。 **[Inferred]**
2. `content_block_id` 僅作 interaction/evidence targeting，不等於 hierarchy ownership；content block 不是新的 persisted hierarchy level。 **[Code-Confirmed] + [Inferred]**
3. evaluation result 是 downstream quality signal，不得成為 document hierarchy truth source，也不得覆蓋 chapter/section/task_unit identity。 **[Code-Confirmed] + [Inferred]**
4. evidence linkage 或 quote validation 不得隱式觸發 artifact persistence mutation；artifact-linked evaluation 不等於 artifact persistence ownership。 **[Code-Confirmed] + [Inferred]**
5. future evidence targeting 應保持 hierarchy-aware（chapter/section/task_unit context）並優先 id-based reference semantics。 **[Inferred]**
6. evaluated_answer 未來若與 question/section_tasks/shared 協作 rich content reference，需保持 deterministic boundary：evaluation 不得成為 parser authority、不得重引入 root sections mirror 或 structure_nodes 主流程語意。 **[Code-Confirmed] + [Inferred]**

### 14.1 Answer Evidence Linkage Direction

1. future answer evidence linkage 可擴展為 sentence/paragraph-level evidence mapping，並支持 content-block-linked answer evidence。 **[Maintainer-Confirmed] + [Inferred]**
2. evidence-target references（含 quote/span）屬 interaction semantics，不等於 hierarchy ownership。 **[Maintainer-Confirmed] + [Inferred]**
3. evaluation output 不能作為 document truth source，亦不得覆蓋 source hierarchy identity。 **[Code-Confirmed] + [Inferred]**
4. evidence linkage 不得 hidden mutate artifacts；artifact write/read authority 仍在明確 persistence 邊界外。 **[Code-Confirmed] + [Inferred]**

### 14.2 Content-Block Reference Semantics

1. content-block references 應以 id-based targeting（例如 `content_block_id`）為主，不回退 title-only reference path。 **[Code-Confirmed] + [Inferred]**
2. reference resolution 應保持 hierarchy-aware 行為（chapter/section/task_unit context），避免跨層錯配。 **[Inferred]**
3. content block 作為 evidence/quote target 時，不得被解讀為 persisted hierarchy level。 **[Code-Confirmed] + [Inferred]**

### 14.3 Quote/Source Validation Direction

1. future quote/source validation 方向是 source/evidence alignment，而非由 evaluation 層重寫來源文本。 **[Inferred]**
2. deterministic validation contract 尚未 finalized；本節僅記錄方向，不代表規則已定案或已實作。 **[Doc-Confirmed] + [Needs Confirmation]**
3. quote validation 不能被描述為現行 runtime capability；目前仍是 future-direction planning。 **[Doc-Confirmed]**

### 14.4 Artifact-Linked Evaluation Boundary

1. future artifact-linked evaluation flow 僅代表可觀測與關聯語義，不代表 evaluated_answer 擁有 artifact persistence authority。 **[Inferred]**
2. evaluation layer 不得成為 parser authority，不得回退 flat task-unit truth model，不得重引 root sections mirror/structure_nodes 主流程語義。 **[Code-Confirmed] + [Inferred]**
3. 後續若實作，需與 `question`、`section_tasks`、`shared` 協調跨模組 evidence/reference contract。 **[Inferred]**

### 14.5 Needs Confirmation

1. quote/evidence validation 的最小 deterministic contract（例如 strict span match vs semantic match）是否需先固定？ **[Needs Confirmation]**
2. content-block-level evidence 是否需要 persisted trace metadata，或僅 runtime projection 即可？ **[Needs Confirmation]**
