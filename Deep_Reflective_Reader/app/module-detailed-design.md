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
2. `_find_chapter_or_raise` 為 hierarchy-only chapter target resolver：`chapter_id` 優先，`chapter_title` 僅作 hierarchy 內 secondary lookup；缺失/歧義皆 fail-fast。 **[Code-Confirmed]**
3. runtime path 不再回退 root `sections`、不再合成 legacy chapter、不再使用 legacy title/section fallback 掩蓋 hierarchy 問題。 **[Code-Confirmed]**
4. `_resolve_task_layout_sections` 對 legacy sections-only / severe inconsistency 皆採 fail-fast。 **[Code-Confirmed]**

## 13. Current Risks

1. risk：coordinator 職責過重
- why：跨層邏輯集中，易造成耦合增長
- guardrail：保持 module contract 文檔與測試邊界

2. risk：runtime fallback 收斂不完全
- why：若未持續維持 hierarchy-only fail-fast，可能在後續維護時回退為隱性 compatibility path
- guardrail：保留 hierarchy-required 測試與文檔邊界，避免重新引入 root-sections/title fallback runtime 行為

3. risk：diagnostics 與 recommendation 混用
- why：可能誤把 advisory 當 control signal
- guardrail：維持 projection-only 契約與欄位語義註釋

4. risk：chapter title ambiguity 若回退 title-only
- why：重複標題文檔（如 part/chapter 作品）會失敗或不穩
- guardrail：優先 chapter_id targeting contract

## 14. Open Questions for Maintainer

1. `section_task_coordinator` 是否需要拆分成 read/write/application service 子層文檔（先文檔化，不改程式）？
2. recommendation 與 diagnostics 是否要在 API 語義層分離文件？

## 15. Suggested Next Documentation Improvements

1. 增加 coordinator flow sequence diagrams（QA ask / task-layout / summary）。
2. 增加 fail-fast error taxonomy（migration required / inconsistency / cache stale）。
3. 補 runtime read-path vs write-path 邊界圖。

## 16. Future Direction Note: Rich Content Interaction API Preparation

> 本節為 future-task documentation/preparation，非當前 implementation。 **[Doc-Confirmed]**

1. app coordinator 未來可擴展為 content-block lookup/read API 的 orchestration boundary（request normalize、identity resolve、response assemble），但本輪僅記錄方向，不代表已實作 endpoint。 **[Maintainer-Confirmed] + [Inferred]**
2. target resolution 順序的 future contract 應為：`chapter_id/section_id -> task_unit_id -> content_block_id`；id 缺失或不一致時 fail-fast。 **[Inferred]**
3. 若 `content_block_id` 缺失（且該 API 要求 block 粒度），應 fail-fast，不得隱式降級為 title/全文掃描。 **[Inferred]**
4. 若 `content_block_id` 重複或產生歧義，應 fail-fast，避免非決定性 target 綁定。 **[Inferred]**
5. 若 hierarchy path 無效（chapter/section/task-unit 與 block 關聯不成立），應 fail-fast，不得容忍跨層錯配。 **[Inferred]**
6. future targeting path 不得回退為 title-only lookup，也不得回退 root `sections` 或 synthetic legacy hierarchy。 **[Code-Confirmed] + [Maintainer-Confirmed]**
7. `task-layout` API 仍保持 lightweight metadata/projection read path，不承載 heavy rich content body。 **[Code-Confirmed] + [Maintainer-Confirmed]**
8. rich content body 應透過 on-demand content API path 讀取，並與 task-layout metadata path 分離。 **[Code-Confirmed] + [Maintainer-Confirmed]**
9. app layer 在 rich-content 方向中仍是 orchestration 層，不是 parser authority，不主導 parser strategy。 **[Code-Confirmed] + [From HLD]**
10. app layer 不應擁有 artifact persistence internals；write path 仍由 service/repository 邊界承擔。 **[Code-Confirmed] + [From HLD]**
11. rich-content interaction request path 不得 hidden mutation，不得觸發 profile write-back；diagnostics 仍為 projection-only。 **[Code-Confirmed] + [Maintainer-Confirmed]**
12. 上述內容均為 future-direction contract preparation；本輪不新增 runtime/API/schema 行為。 **[Doc-Confirmed]**

## 17. Future Direction Note: Manual Structure Override Orchestration

> 本節記錄 app-layer 對 source-agnostic manual structure override 的 orchestration 邊界；不代表目前 implementation。 **[Maintainer-Provided] + [Future Direction]**

1. app layer may orchestrate manual structure validation and explicit manual reparse for any document type, not only OCR/PDF. **[Maintainer-Provided]**
2. app layer should normalize request intent, call preparation to load required raw/page evidence, call document_structure validation/projection, and return validation/reparse results. **[Future Direction]**
3. app layer must not define parser rules or manual projection semantics; those remain owned by `document_structure/`. **[From HLD] + [Future Direction]**
4. manual validation/preview should be read/analysis-oriented and must not persist hierarchy, task-layout metadata, profile diagnostics, or artifacts. **[Future Direction]**
5. manual commit reparse is an explicit mutation path and should be separate from `/documents/task-layout`; task-layout remains projection-only. **[From HLD] + [Future Direction]**
6. successful manual commit should replace the active structured hierarchy only after validation succeeds; failure should preserve the current structured artifact. **[Future Direction]**
7. app-layer error semantics should distinguish malformed request, unresolved anchors, invalid hierarchy shape, empty ranges, and stale source evidence. **[Future Direction]**
8. This direction does not add runtime endpoints, schema fields, source code, or persistence behavior in this documentation pass. **[Doc-Confirmed]**

## 18. Page-Backed TOC Anchor Orchestration

> 本節支援 UI TOC editor 的 page-first anchor UX。
> 目前已完成 task-layout coordinator DTO prefill orchestration、public API schema/route mapping、read-only page-boundary handoff、以及 page-backed manual validation/commit orchestration。 **[Code-Confirmed]**

1. The app layer should orchestrate two related flows:
   - read/projection flow: expose current structure anchor evidence for UI edit-existing prefill **[Code-Confirmed at coordinator DTO level]**
   - explicit mutation flow: validate and commit manual `page_range` anchors when source page evidence is available **[Code-Confirmed]**
2. The app layer should not compute parser semantics, page boundary mapping, or hierarchy projection itself; those remain delegated to `document_preparation/` and `document_structure/`. **[From HLD] + [Future Direction]**
3. For task-layout, `SectionTaskCoordinator.get_document_task_layout(...)` now loads optional preparation-owned page boundary evidence, delegates existing hierarchy-span projection to `document_structure.project_structure_anchor_evidence(...)`, and passes lightweight `AnchorEvidenceDTO` into chapter/section DTOs. **[Code-Confirmed]**
4. For manual commit, app loads canonical source evidence including source hash and page boundaries, rejects stale source evidence, then delegates page-anchor projection/draft building to `document_structure/`. **[Code-Confirmed]**
5. Page-backed validation/commit failures preserve the current structured artifact and do not silently fallback to common parser as a successful manual reparse. **[Maintainer-Provided] + [Code-Confirmed]**
6. `char_range` remains the source-agnostic fallback for documents with no reliable page evidence or for explicit advanced override. **[Maintainer-Provided] + [Future Direction]**
7. Current task-layout coordinator projection may load preparation source evidence to access page boundaries, but only forwards compact page boundary metadata to `document_structure/`; it does not expose raw text, page text, OCR geometry, or content blocks. **[Code-Confirmed]**
8. Public `/documents/task-layout` response schema exposes lightweight `anchor_evidence` for chapters/sections, including `page_range` when page boundaries are available and `char_range` fallback otherwise. **[Code-Confirmed]**
9. The task-layout anchor evidence path remains read-only: it does not mutate hierarchy, profile diagnostics, task-layout metadata, or artifacts. **[Code-Confirmed]**

## 19. PostgreSQL Manual Reparse Replacement Boundary

Manual `page_range` reparse validates source evidence and builds a hierarchy-only draft before persistence. PostgreSQL-backed manual commit must then use the explicit parser-level replacement boundary rather than the artifact/task-layout save method that preserves `current_structure_version` and existing hierarchy rows. **[Code-Confirmed]**

Observed failure:

- API: `/documents/reparse-structure`
- parser mode: `manual_structure`
- affected document example: `暗水幽灵`
- original failure: `duplicate key value violates unique constraint "uq_chapters_document_order"` for `(document_id, chapter_order)=(13, 0)`

Root-cause boundary:

1. Manual commit is a hard reparse mutation and should use a parser-level hierarchy replacement save. **[From HLD] + [Code-Confirmed]**
2. PostgreSQL artifact/task-layout saves intentionally preserve the current structure version and should not clear hierarchy rows. **[Code-Confirmed]**
3. Manual reparse calls the explicit `save_reparsed_document(...)` repository boundary, which PostgreSQL maps to `replace_existing_hierarchy=True`. **[Code-Confirmed]**
4. Validation/source/draft failures must continue to preserve the current structured hierarchy and derived resources. **[From HLD] + [Code-Confirmed]**
5. Successful manual reparse must remain explicit, transactional, hierarchy-first, and separate from task-layout read/projection. **[From HLD] + [Code-Confirmed]**
