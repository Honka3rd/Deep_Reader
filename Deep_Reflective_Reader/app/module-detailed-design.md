# app Detailed Design

## 1. Module Purpose

`app/` 是 runtime orchestration 層，承接 API 請求後的跨模組協調：
- QA 協調（`QACoordinator`）
- section/chapter task 協調（`SectionTaskCoordinator`）

**[Code-Confirmed]**

## 2. Position in Overall Architecture

- API Layer 與 Core Layers 之間的 orchestration layer（Application service 層）

## 3. Key Files

| File | Responsibility | Notes |
|---|---|---|
| `app/qa_coordinator.py` | QA 主協調（prepare+retrieve+prompt+LLM+session） | 使用 DI container 組裝依賴 **[Code-Confirmed]** |
| `app/section_task_coordinator.py` | section/chapter 任務協調與 task-layout projection | hierarchy-first + diagnostics projection **[Code-Confirmed]** |
| `app/coordinator.py` | backward-compatible alias module | naming migration compatibility **[Code-Confirmed]** |

## 4. Main Responsibilities

1. 協調 prepare/load 與 downstream module 使用順序。 **[Code-Confirmed]**
2. 對 task layout 進行 runtime projection 組裝。 **[Code-Confirmed]**
3. 協調 summary/quiz 任務執行與 artifact 更新路徑。 **[Code-Confirmed]**
4. 輸出 enhanced recommendation 與 profile diagnostics。 **[Code-Confirmed]**
5. 維持 fail-fast 邊界（如 hierarchy inconsistency / migration required）。 **[Code-Confirmed]**

## 5. Non-Responsibilities

1. 不應成為 parser rule 定義層。 **[From HLD]**
2. 不應自行改寫 structured model schema。 **[From HLD]**
3. 不應在 task-layout read path 做 profile 隱性回寫。 **[Code-Confirmed] + [From HLD]**

## 6. Important Data Structures / Contracts

- `AskExecutionResult`
- `SectionTaskResult`
- `DocumentTaskLayout`
- `EnhancedParseRecommendationDTO`
- `ProfileStructureDiagnosticsDTO`
- `ResolvedTaskUnit`（coordinator internal runtime contract）

## 7. Coordinator Responsibility Slices

| Slice | Coordinator Scope | Must Not Do |
|---|---|---|
| Prepare orchestration | 呼叫 pipeline，整合 readiness/error | 自己實作 parser/repository 邏輯 |
| Task layout projection | assemble DTO + diagnostics | hidden persistence write |
| Summary/Quiz write path | 呼叫 service + repository 寫入 | 直接繞過 repository contract |
| Recommendation/diagnostics | projection 與 advisory訊號輸出 | 直接控制 parser 行為 |

## 8. Module Relationships

- depends on:
  - `document_preparation/`
  - `document_structure/`
  - `section_tasks/`
  - `profile/`
  - `retrieval/`, `question/`, `prompts/`, `session/`
- used by:
  - `main.py`
- reads from:
  - structured/profile artifacts（透過 pipeline/repository）
- writes to:
  - task artifacts（透過 repository）
- projection relationship:
  - task-layout + diagnostics DTO projection

## 9. Main Flows Involving This Module

1. QA ask flow（prepare_and_load -> context build -> prompt -> LLM -> session update）。 **[Code-Confirmed]**
2. task-layout flow（load/refresh layout -> build chapters-first response）。 **[Code-Confirmed]**
3. section summary/quiz flow（cache check -> resolve -> generate -> persist）。 **[Code-Confirmed]**
4. chapter summary/quiz flow（chapter_id-first target resolution）。 **[Code-Confirmed]**
5. reparse flow（explicit parser mode re-run）。 **[Code-Confirmed]**

## 10. Error Semantics (High-Level)

| Scenario | Expected Behavior |
|---|---|
| missing/invalid section target | fail-fast with explicit error |
| ambiguous chapter title (title-only) | fail-fast; encourage id-based target |
| legacy sections-only document at hierarchy-required runtime path | fail-fast migration-required semantics |
| severe hierarchy inconsistency | fail-fast; no legacy runtime mask |

**[Code-Confirmed]**

## 11. Persistence / Side Effects

- read persistence：是（透過 pipeline/store/repository）
- write persistence：是（summary/quiz/task-layout metadata 更新）
- mutate structured document：是（透過 repository update methods）
- generate runtime projection：是（task-layout + diagnostics）
- call LLM：間接（透過 task services / QA path）
- diagnostics only：部分（profile_diagnostics 為 projection）

## 12. Known Legacy / Compatibility Behavior

1. `app/coordinator.py` 保留 alias compatibility。 **[Code-Confirmed]**
2. `_find_chapter_or_raise` 仍保留 title path 下的 legacy section fallback 分支。 **[Code-Confirmed] + [Needs Confirmation]**
3. `_resolve_task_layout_sections` 已對 legacy sections-only / severe inconsistency 做 fail-fast。 **[Code-Confirmed]**

## 13. Current Risks

1. risk：coordinator 職責過重
- why：跨層邏輯集中，易造成耦合增長
- guardrail：保持 module contract 文檔與測試邊界

2. risk：runtime fallback 收斂不完全
- why：局部 legacy compatibility path 仍存在
- guardrail：標註 compatibility-only + 退場計畫

3. risk：diagnostics 與 recommendation 混用
- why：可能誤把 advisory 當 control signal
- guardrail：維持 projection-only 契約與欄位語義註釋

4. risk：chapter title ambiguity 若回退 title-only
- why：重複標題文檔（如 part/chapter 作品）會失敗或不穩
- guardrail：優先 chapter_id targeting contract

## 14. Open Questions for Maintainer

1. `section_task_coordinator` 是否需要拆分成 read/write/application service 子層文檔（先文檔化，不改程式）？
2. `_find_chapter_or_raise` 的 legacy title fallback 是否要列入下一步退場？
3. recommendation 與 diagnostics 是否要在 API 語義層分離文件？

## 15. Suggested Next Documentation Improvements

1. 增加 coordinator flow sequence diagrams（QA ask / task-layout / summary）。
2. 增加 fail-fast error taxonomy（migration required / inconsistency / cache stale）。
3. 補 runtime read-path vs write-path 邊界圖。

## 16. Future Direction Note: Rich Content Interaction API Preparation

> 本節為 future-task documentation/preparation，非當前 implementation。 **[Doc-Confirmed]**

1. app layer 未來可暴露 content-block lookup/read API orchestration，但本輪僅記錄 future tasks。 **[Inferred]**
2. `task-layout` API 仍應保持 lightweight metadata/projection read path，不承載完整 rich content body。 **[Code-Confirmed] + [Inferred]**
3. rich content body 應維持 on-demand read model；task-layout 與 rich-content read path 必須分離。 **[Code-Confirmed] + [Maintainer-Confirmed]**
4. sentence/paragraph/content-block interaction targeting 應以 stable ids（如 chapter_id/section_id/task_unit_id/content_block_id）解析；找不到或歧義時應 fail-fast。 **[Inferred]**
5. app layer 不應自行實作 rich content construction、artifact persistence internals、parser authority、或 diagnostics write-back。 **[Code-Confirmed] + [Inferred]**
6. future interaction targeting 不得回退為 title-only path 取代 id-based deterministic path。 **[Code-Confirmed] + [Maintainer-Confirmed]**
