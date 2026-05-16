# Deep Reflective Reader Proposal

> 文件定位：本 proposal 是「基於目前 codebase 已完成狀態」的階段性提案，不是假設從零開始。
> 事實標註規則：
> - **[Code-Confirmed]**：已從程式碼直接確認
> - **[Maintainer-Provided]**：由維護者提供的實測背景
> - **[Inferred]**：依程式邏輯推論，仍需確認

## 1. Project Purpose

Deep_Reflective_Reader 的目標不是單純做 parser，而是提供一個可持久化、可回讀、可診斷的深度閱讀系統：

- 長文結構解析（Hierarchy-First）
- 結構化互動（Section/Chapter/TaskUnit）
- 任務化輸出（summary / quiz）
- Artifact persistence 與 cache validity
- parser quality diagnostics 與 recommendation

核心互動模型：

`Document -> Chapter -> Section -> TaskUnit`

## 2. Current Architecture Snapshot

### 2.1 資料流（當前）

```text
Raw Document
  -> Document Preparation Pipeline
  -> Pre-structure Profile (parser_metadata)
  -> Structured Hierarchy Build (chapters[].sections[])
  -> Post-structure Metadata Enrichment
  -> Task Layout Projection (read path)
  -> Summary / Quiz / TaskUnit Artifacts
  -> Diagnostics / Enhanced Parse Recommendation
```

### 2.2 主要模組責任（摘要）

- Preparation orchestration：`document_preparation/document_preparation_pipeline.py` **[Code-Confirmed]**
- Structured model / serialization：`document_structure/structured_document.py` **[Code-Confirmed]**
- Hierarchy helper/index：`document_structure/document_hierarchy_index.py` **[Code-Confirmed]**
- Hierarchy build：`document_structure/structured_document_builder.py`, `document_structure/structured_hierarchy_builder.py` **[Code-Confirmed]**
- Task layout + coordinator：`app/section_task_coordinator.py`, `section_tasks/document_task_layout.py` **[Code-Confirmed]**
- Artifact persistence：`document_structure/structured_document_artifact_repository.py` **[Code-Confirmed]**
- Profile + metadata：`profile/document_profile.py`, `profile/document_profile_builder.py`, `profile/parser_metadata_extractor.py`, `profile/post_structure_metadata_enricher.py` **[Code-Confirmed]**
- API surface：`main.py`, `api_schemas.py` **[Code-Confirmed]**

## 3. Confirmed Implemented Capabilities

以下僅列 codebase 可直接確認項目。

1. **Hierarchy-first StructuredDocument contract** **[Code-Confirmed]**
- `StructuredDocument` 保留 `chapters / sections / structure_nodes`，但 docstring 已明確 `chapters` 為 primary source。
- 路徑：`document_structure/structured_document.py`

2. **新 JSON 預設不輸出 root sections / structure_nodes** **[Code-Confirmed]**
- `to_dict()` 需顯式 `include_legacy_sections=True` 才輸出 `sections`。
- `to_dict()` 需顯式 `include_legacy_structure_nodes=True` 才輸出 `structure_nodes`。
- 路徑：`document_structure/structured_document.py`

3. **Legacy payload 仍可讀（相容讀取）** **[Code-Confirmed]**
- `from_dict()` 仍可讀 `sections` 與 `structure_nodes`。
- 路徑：`document_structure/structured_document.py`

4. **`get_effective_sections(document)` 已收斂為 hierarchy-only** **[Code-Confirmed]**
- 只 flatten `chapters[].sections[]`，不再 fallback root `document.sections`。
- 路徑：`document_structure/document_hierarchy_index.py`

5. **Artifact repository 已 hierarchy-aware，含 load-time migration** **[Code-Confirmed]**
- 寫入前 load 路徑：`chapters` 空且 `sections` 有值時，先 `migrate_legacy_sections_to_chapters(...)`。
- Section/chapter/task-layout 更新皆以 hierarchy 為主要寫入路徑。
- 路徑：`document_structure/structured_document_artifact_repository.py`

6. **Task-layout API 為 chapters-only public response** **[Code-Confirmed]**
- `/documents/task-layout` response model 僅對外 `chapters`（另有 recommendation/diagnostics），無 top-level public `sections/task_units/chapter_artifacts`。
- 路徑：`api_schemas.py`, `main.py`

7. **Chapter API 支援 `chapter_id`，解決 title ambiguity** **[Code-Confirmed]**
- `SummarizeChapterRequest` / `ChapterQuizRequest`：`chapter_id` 可選，與 `chapter_title` 同時提供時 `chapter_id` 優先。
- 路徑：`api_schemas.py`, `main.py`, `app/section_task_coordinator.py`

8. **Chapter artifact key 已有 id-based key** **[Code-Confirmed]**
- `chapter_id::` key 前綴存在，並有 legacy key candidate 機制。
- 路徑：`app/section_task_coordinator.py`

9. **Profile 已具 pre + post metadata 能力** **[Code-Confirmed]**
- pre：`parser_metadata`
- post：`post_structure_metadata`
- 並保留 `structure_profile` 相容欄位。
- 路徑：`profile/document_profile.py`

10. **Post-structure enrichment 已在 pipeline Step 4.5 接入** **[Code-Confirmed]**
- profile 在 structured build 後 enrichment；失敗不阻塞 prepare。
- 路徑：`document_preparation/document_preparation_pipeline.py`

11. **Diagnostics 已作為 task-layout runtime projection** **[Code-Confirmed]**
- DTO docstring 明確 mixed-source；task-unit coverage 來自「當前 layout sections」，不是 prepare-time snapshot。
- 路徑：`section_tasks/document_task_layout.py`, `app/section_task_coordinator.py`

12. **Enhanced parse recommendation evaluator 已存在** **[Code-Confirmed]**
- 有 deterministic score/reasons/metrics。
- 路徑：`document_structure/enhanced_parse_trigger_evaluator.py`

## 4. Deprecated / Legacy Compatibility

### 4.1 root `sections[]`
- **可讀**：是（`StructuredDocument.from_dict`）**[Code-Confirmed]**
- **預設會寫**：否（需 `include_legacy_sections=True`）**[Code-Confirmed]**
- **主流程使用**：
  - `get_effective_sections`：否（已移除 fallback）**[Code-Confirmed]**
  - 少數 helper 仍可選 legacy fallback（例如 `find_section_by_id_effective(..., allow_legacy_fallback=True)`）**[Code-Confirmed]**

### 4.2 `structure_nodes`
- **可讀**：是（legacy compatibility）**[Code-Confirmed]**
- **預設會寫**：否（需 `include_legacy_structure_nodes=True`）**[Code-Confirmed]**
- **主流程使用**：否（未見 runtime 主線依賴）**[Code-Confirmed]**

### 4.3 flat top-level `task_units` / `chapter_artifacts`
- 在 `DocumentTaskLayout` 內部 DTO 仍有 transitional 欄位註記（internal/backward compatible）。
- 對外 API response 已 chapters-first。**[Code-Confirmed]**
- 路徑：`section_tasks/document_task_layout.py`, `main.py`, `api_schemas.py`

### 4.4 legacy `structure_profile`
- `DocumentProfile` 仍保留欄位與舊 schema 反序列化能力。
- 新 builder 主流程不主動產生它（`structure_profile=None`）。**[Code-Confirmed]**
- 路徑：`profile/document_profile.py`, `profile/document_profile_builder.py`

## 5. Current Design Principles

以下原則已在 codebase 中可觀察：

1. **Hierarchy Purity（持久化來源單一化）** **[Code-Confirmed]**
- 新輸出避免 root mirror，主資料源在 `chapters[].sections[].task_units[]`。

2. **Metadata Advisory（metadata 不是 parser authority）** **[Code-Confirmed]**
- pipeline 仍有 TODO：尚未把 `parser_metadata` hints 接入 structured parser。
- 路徑：`document_preparation/document_preparation_pipeline.py`

3. **Diagnostics Read-Only Projection** **[Code-Confirmed]**
- task-layout diagnostics 使用 runtime state 計算，不回寫 profile。
- 路徑：`app/section_task_coordinator.py`

4. **LLM Helper, Not Authority** **[Code-Confirmed]**
- profile builder：deterministic + lightweight LLM classification + fallback。
- 路徑：`profile/document_profile_builder.py`, `profile/parser_metadata_extractor.py`

5. **Backward Compatibility Without New Legacy Writes** **[Code-Confirmed]**
- 可讀舊 payload，但新 artifact 預設不再寫回完整 legacy mirror。

## 6. Known Real-Document Findings

> 本節資料來源混合：程式碼可證 + 維護者提供之實測脈絡。

1. **《许三观卖血记》**
- 「common parser 對 chapter-only 表現穩定」：**[Maintainer-Provided]**
- codebase 可見 chapter-only shape 與 metadata/diagnostics 有相應規則：**[Code-Confirmed]**
  - 路徑：`profile/post_structure_metadata_enricher.py`, `document_structure/structured_hierarchy_builder.py`

2. **Madame Bovary**
- 「Chapter One 在不同 Part 重複，title target 會 ambiguous」：**[Maintainer-Provided]**
- chapter_id-first target resolution 與 ambiguous title fail 行為存在測試與實作：**[Code-Confirmed]**
  - 路徑：`app/section_task_coordinator.py`, `api_schemas.py`, `scripts/test_chapter_id_target_resolution.py`

3. **《中式思维》**
- 「common parser 可能偏 flat / fallback，llm_enhanced 可改善成 essay_sections」：**[Maintainer-Provided]**
- enhanced recommendation evaluator 與 profile shape 診斷機制存在：**[Code-Confirmed]**
  - 路徑：`document_structure/enhanced_parse_trigger_evaluator.py`, `profile/post_structure_metadata_enricher.py`

## 7. Current Gaps

1. **parser_metadata 尚未進入 parser strategy**
- 證據：pipeline TODO 註解。**[Code-Confirmed]**
- 影響：metadata 目前主要供 diagnostics/recommendation，不直接影響 parse。

2. **尚未 auto parser switching**
- recommendation 存在，但未見自動 reparse 切換主流程。**[Code-Confirmed]**

3. **post_structure_metadata 不反向修改 structured_document**
- enrichment 只更新 profile。**[Code-Confirmed]**

4. **文件形態正規化策略尚未文件化為固定契約**
- 目前實作已具備 common / llm_enhanced 與 recommendation，但「何時單章、何時多章、何時 chapter-only」仍需在設計文件明確寫死。**[Code-Confirmed]**

5. **runtime helper 仍有局部 legacy fallback 參數**
- 例：`find_section_by_id_effective(..., allow_legacy_fallback)`；`_find_chapter_or_raise` 內仍有 legacy title fallback 分支。
- 是否完全退場需政策決定。**[Code-Confirmed]**

6. **Diagnostics contract 雖已有實作，但外部文件化仍可加強**
- mixed-source 語義已在 DTO docstring，但缺統一維護文檔。**[Code-Confirmed]**

## 8. Proposed Next Phase

> 以下是「小步可執行」方向，不是大重構。

### A. Parser Strategy Recommendation Formalization
- **Goal**：把 recommendation 從「分散訊號」整理成明確決策契約（提示/人工確認/手動 rerun）。
- **Why now**：已有 evaluator + diagnostics，但產品決策規則未定。
- **Affected modules**：`document_structure/enhanced_parse_trigger_evaluator.py`, `app/section_task_coordinator.py`, `api_schemas.py`（若只補文檔可先不動 schema）。
- **Risk**：過早自動化可能誤切 parser。
- **First small task**：先寫 recommendation decision table（文檔 + 測試 fixture），不改 runtime。

### B. ParserHints Advisory DTO（Read-only）
- **Goal**：把 parser_metadata/post_metadata 的可用欄位整理成穩定 advisory DTO。
- **Why now**：目前 diagnostics 已 mixed-source，適合固定對外語義。
- **Affected modules**：`profile/document_profile.py`, `section_tasks/document_task_layout.py`, 文件。
- **Risk**：欄位語義鎖定後調整成本上升。
- **First small task**：新增「欄位語義矩陣」文件（來源、更新時機、是否可持久化）。

### C. Enhanced Parse Trigger Evaluator Documentation + Calibration
- **Goal**：把 score/reason/metrics 的門檻做可追蹤校準。
- **Why now**：實際已有多文檔案例，適合建立 calibration baseline。
- **Affected modules**：`document_structure/enhanced_parse_trigger_evaluator.py`, `scripts/*smoke*`。
- **Risk**：過度 overfit 少量文檔。
- **First small task**：增加 3~5 個固定 regression fixture 的評分期望。

### D. Document Shape Normalization Policy（固定兩層主模型）
- **Goal**：明確採用通用兩層結構 `chapter -> section`，不引入 `Part -> Chapter` persistence schema。
- **Why now**：目前維護方向已確認「優先覆蓋普遍文檔」，避免過早引入罕見多層級 schema 複雜度。
- **Affected modules**：`document_structure/`, `section_tasks/`, `profile/`（主要是 contract 與 metadata 語義）。
- **Risk**：若沒有明確規則，平面長文、章節文、多層碎片文在 parser 路徑會產生不一致。
- **First small task**：新增「Document Shape Normalization Contract」文件，明確如下策略（不改 API）：
  - flat 長文：優先允許 llm parser 產生多章；chapter title 可由 LLM 提供。
  - flat 短文：若僅足夠單一 task unit（或 LLM 判定不足以分章），固定單章。
  - chapter-only 文檔：尊重章級結構；每章仍持有唯一 section（同名、不同 id）。
  - 標準文檔：先 common，不佳再 llm_enhanced。
  - 多層細碎文檔：持續收斂為 `chapter-section`；更細粒度下沉到 task units，並把原層級線索記入 metadata 供未來擴展。

### E. Diagnostics Contract Documentation
- **Goal**：明文化「projection-only / no hidden mutation / no heavy payload」契約。
- **Why now**：避免後續開發把 diagnostics 當 persisted truth。
- **Affected modules**：`section_tasks/document_task_layout.py`, `api_schemas.py`, `main.py`, docs。
- **Risk**：低。
- **First small task**：補一份 `docs/diagnostics_contract.md`（或 proposal 附錄）。

## 9. Questions for Maintainer

1. 你確認目前階段把 `Part -> Chapter nested persistence` 定義為明確 non-goal，並固定兩層主模型（chapter-section）嗎？
2. `parser_metadata` 未來是否只做 recommendation，還是允許進入 common parser 的 heuristic path？
3. 若要 metadata-guided parsing，是否接受 feature flag 漸進啟用？
4. enhanced parser recommendation 是「只提示使用者」還是允許「一鍵/自動 rerun」？
5. `title_uniqueness_risk` 最終權威來源是否應固定為 post-structure metadata（而非 pre-structure LLM）？
6. `find_*_effective(... allow_legacy_fallback)` 這類 helper 是否要進入明確退場時程？
7. `structure_profile` legacy 相容欄位要保留到哪個版本節點？
8. proposal 的主要受眾是「維護者內部」還是「未來協作者/開源讀者」？
9. 是否需要在 proposal 加入 API contract 專章（含 no-heavy-payload 保證）？
10. 是否需要加 migration history（從 dual-representation 到 pure hierarchy 的階段記錄）？
11. `post_structure_metadata` 是否需要獨立 refresh 機制（非 prepare 路徑）？
12. 對於中式 essay 類文檔，是否要把 common→enhanced 推薦規則納入正式產品策略（而非僅 diagnostics）？

---

## Needs Maintainer Confirmation

以下事項目前只可推論，需 maintainer 明確拍板：

- 兩層主模型（chapter-section）是否作為本期穩定契約，且明確不做 Part->Chapter persistence-level schema。
- `allow_legacy_fallback` 參數族群的最終退場時點。
- recommendation 是否可在 UX 層觸發半自動 reparse。
- diagnostics 欄位是否要凍結為公開契約。
