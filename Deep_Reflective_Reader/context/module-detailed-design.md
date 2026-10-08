# context Detailed Design

## 1. Module Purpose

`context/` 負責在 QA 執行時組裝可用上下文，支援 local window、retrieval 與 full-text 三種模式，並透過 token budget 控制輸入大小。 **[Code-Confirmed]**

## 2. Position in Overall Architecture

- Task Layout / Task Unit Layer（QA context sub-layer）

## 3. Key Files

| File | Responsibility | Notes |
|---|---|---|
| `context/context_orchestrator.py` | mode 決策與 context 組裝協調 | 產出 `ContextBuildResult` **[Code-Confirmed]** |
| `context/document_context_builder.py` | local/retrieval/full-text context build | 核心拼裝邏輯 **[Code-Confirmed]** |
| `context/coverage_oriented_context_builder.py` | global retrieval coverage 去密集化 | 提升跨段覆蓋 **[Code-Confirmed]** |
| `context/artifact_aware_context.py` | lower-level artifact secondary context projection | metadata-only, no payload expansion **[Code-Confirmed]** |
| `context/reading_interaction_context.py` | reading interaction primary source context selection | resolved target only; full target vs semantic compact **[Code-Confirmed]** |
| `context/token_budget_manager.py` | token 估算、截斷、budget 計算 | prompt-aware context budget **[Code-Confirmed]** |

## 4. Main Responsibilities

1. 根據 scope/session 選擇 context mode。 **[Code-Confirmed]**
2. 依 budget 拼裝上下文且可截斷。 **[Code-Confirmed]**
3. 對 global retrieval 做 coverage-oriented 選擇。 **[Code-Confirmed]**

## 5. Non-Responsibilities

1. 不負責最終 answer generation。 **[Code-Confirmed]**
2. 不負責 artifact persistence。 **[Code-Confirmed]**
3. 不應直接改寫 structured hierarchy。 **[From HLD]**

## 6. Important Data Structures / Contracts

- `ContextBuildResult`
- `CoverageSelection`
- `PromptMode` / `ContextMode`
- `ReadingInteractionContextResult`

## 7. Module Relationships

- depends on: `retrieval/`, `session/`, `question/`, `prompts/`, `profile/`
- used by: `app/qa_coordinator.py`
- reads from: runtime `FaissIndexBundle`
- writes to: 無持久化

## 8. Main Flows Involving This Module

1. QA ask flow 的 context selection/build。 **[Code-Confirmed]**
2. local reading window 擴張與 budget 裁切。 **[Code-Confirmed]**
3. global scope coverage selection。 **[Code-Confirmed]**

## 9. Persistence / Side Effects

- read persistence：否（讀 runtime bundle）
- write persistence：否
- projection-only：是
- call LLM：否（由其他層處理）

## 10. Known Legacy / Compatibility Behavior

No known legacy compatibility responsibility.

## 11. Current Risks

1. risk：token estimation heuristic 偏差
- why：可能導致 context 過長或過短
- guardrail：保留可觀測 budget fields

2. risk：mode 決策與 scope resolver 結果不一致
- why：回答品質波動
- guardrail：維持 diagnostics/reason fields

3. risk：local window fallback 過度擴張
- why：弱化 local reading 目的
- guardrail：near chunk threshold policy

## 12. Open Questions for Maintainer

1. 是否需要把 context mode decision 的 reason 對外 API 化？ **[Needs Confirmation]**
2. 是否需要固定 full-text mode 在某些問句類型下禁用？ **[Needs Confirmation]**

## 13. Suggested Next Documentation Improvements

1. 補 context mode state diagram。

## 14. Reading Target Interaction Context

> 本節記錄 book/chapter/section/task-unit interaction 的 context strategy。第一版 implementation 已落地在 `context/reading_interaction_context.py`。 **[Maintainer-Provided] + [Code-Confirmed]**

1. Context selection for reading interactions operates on a resolved target shape, not on free-form titles or unresolved ids. Empty content fails fast instead of falling back to target title. **[Code-Confirmed]**
2. If the resolved target content fits the active model capability and configured budget, the context layer provides full target context with a target-level evidence id. **[Code-Confirmed]**
3. If the target is too large, the context layer produces a deterministic semantic/coverage compact context for the fixed prompt instruction. **[Code-Confirmed]**
4. First version avoids multi-call map-reduce; it selects between full target context and compacted single-call context only. **[Code-Confirmed]**
5. Context build metadata exposes `context_mode`, token estimate, used context tokens, effective budget, selected evidence/chunk ids, truncation/compaction reason, target identity, and model capability source without serializing source text. **[Code-Confirmed]**
6. The context layer does not call LLM, persist artifacts, mutate sessions, or decide generation status. **[Code-Confirmed]**
7. Interaction context remains compatible with existing token-budget utilities while keeping interaction prompt instructions caller-provided and predictable. **[Code-Confirmed]**

### 14.1 Artifact-Aware Context Policy

1. Higher-level interaction generation should optionally include lower-level artifacts as secondary context when generating section, chapter, or document/book artifacts. **[Maintainer-Provided] + [Future Direction]**
2. The current target source text remains primary context. Lower-level artifacts are learning-memory signals for deduplication, coverage awareness, abstraction hints, and difficulty escalation. **[Maintainer-Provided] + [Future Direction]**
3. Lower-level artifacts must not replace target source text, become hierarchy truth, become parser authority, or become the only evidence for factual claims. **[From HLD] + [Maintainer-Provided]**
4. Artifact context should be summarized/projection-based, not raw payload dumping. It should prefer compact fields such as artifact type, target id, prompt/topic summary, quiz concepts, critical-thinking question focus, answer/evaluation outcome, status, and generation metadata. **[Maintainer-Provided] + [Future Direction]**
5. Budget priority should be: fixed prompt instruction, current target source context, then lower-level artifact summaries. If over budget, artifact context is pruned or compacted before dropping source context. **[Maintainer-Provided] + [Future Direction]**
6. For section generation, candidate lower-level artifacts come from task units in that section; for chapter generation, from sections/task units in that chapter; for document/book generation, from chapter/section/task-unit artifacts under the document. **[Maintainer-Provided] + [Future Direction]**
7. Context metadata should record `artifact_context_mode`, `referenced_artifact_ids`, artifact target levels, coverage counts, deduplication hints, and pruning/compaction reasons. **[Maintainer-Provided] + [Future Direction]**
8. Absence of lower-level artifacts must not block generation; it only means the secondary artifact context is empty. **[Maintainer-Provided] + [Future Direction]**
