# shared Detailed Design

## 1. Module Purpose

`shared/` 提供跨模組共用 DTO/抽象結果契約，包含 task artifacts、task units 與通用 result 基類。 **[Code-Confirmed]**

## 2. Position in Overall Architecture

- Shared Utility Layer

## 3. Key Files

| File | Responsibility | Notes |
|---|---|---|
| `shared/abstract_result.py` | Generic result 抽象基類 | success/payload/reason/cache_hit **[Code-Confirmed]** |
| `shared/artifact_target_model.py` | artifact target level/ref shared contract | single source of truth for target-level enum **[Code-Confirmed]** |
| `shared/task_artifacts.py` | summary/quiz/task artifacts schema | section/task-unit/document-level contracts **[Code-Confirmed]** |
| `shared/task_unit_model.py` | `TaskUnit` 與 `TaskUnitContentBlock` schema | parent_section_id、artifact nested fields、string->block adapter foundation **[Code-Confirmed]** |

## 4. Main Responsibilities

1. 定義可序列化的共用資料契約。 **[Code-Confirmed]**
2. 統一 summary/quiz artifact 的 metadata 欄位。 **[Code-Confirmed]**
3. 作為 document_structure/section_tasks/profile 等模組共用基礎。 **[Code-Confirmed]**

## 5. Non-Responsibilities

1. 不負責業務流程（解析、查找、API route）。 **[Code-Confirmed]**
2. 不負責 persistence store。 **[Code-Confirmed]**

## 6. Important Data Structures / Contracts

- `AbstractResult[PayloadT]`
- `SummaryArtifact`
- `QuizArtifact`
- `TaskArtifacts`
- `DocumentTaskArtifacts`
- `ArtifactTargetLevel`
- `ArtifactTargetRef`
- `TaskUnit`
- `TaskUnitContentBlock`

## 7. Module Relationships

- used by: `document_structure/`, `section_tasks/`, `app/`, `profile/`
- depends on: 無重依賴（基礎資料層）

## 8. Main Flows Involving This Module

1. artifact write/read flow 的 schema 載體。 **[Code-Confirmed]**
2. task-layout response 與 summary/quiz service payload 載體。 **[Code-Confirmed]**

## 9. Persistence / Side Effects

- persistence：無（DTO only）
- side effects：無

## 10. Known Legacy / Compatibility Behavior

1. `TaskUnit.is_fallback_generated` 欄位保留歷史 fallback 來源訊號。 **[Code-Confirmed]**
2. `DocumentTaskArtifacts` 保留 chapter_artifacts map compatibility。 **[Code-Confirmed]**

## 11. Current Risks

1. risk：schema 欄位增加後未同步所有 consumer
- why：序列化/反序列化不一致
- guardrail：保持 round-trip 測試

2. risk：artifact metadata version 演進缺少集中策略
- why：cache validity 判斷可能漂移
- guardrail：維持 version 欄位並文件化

## 12. Open Questions for Maintainer

1. `AbstractResult` 是否要擴充標準錯誤碼欄位？ **[Needs Confirmation]**

## 13. Suggested Next Documentation Improvements

1. 補 artifact metadata field glossary。

## 14. Rich Task-Unit Content Foundation and Future Direction

1. shared 層已新增 `TaskUnitContentBlock` 與 deterministic block-id helper（`<task_unit_id>:content:<index>`）作為 caller-neutral foundation。 **[Code-Confirmed]**
2. `TaskUnit` 現在含有 additive `content_blocks: list[TaskUnitContentBlock]` 內部表示；`TaskUnit.content` 仍維持 `str` 並保留 compatibility role。 **[Code-Confirmed]**
3. `TaskUnit` 初始化/反序列化時，若缺少 `content_blocks` 且 `content` 非空，會自動穩定化成單一 deterministic block；空字串內容則維持空 block 清單。 **[Code-Confirmed]**
4. 當 `content_blocks` 已存在時，`TaskUnit.to_content_blocks()` 回傳穩定化後的現有 blocks；`content` 仍可供 compatibility 使用。 **[Code-Confirmed]**
5. `TaskUnit.to_dict(include_content_blocks=False)` 預設維持既有 serialization 輸出；`include_content_blocks=True` 提供 additive rich-content round-trip。 **[Code-Confirmed]**
6. content block 定位為 task-unit 內部 render/interaction segmentation，不是 chapter/section/task_unit 之外的新 hierarchy level。 **[Maintainer-Confirmed] + [Code-Confirmed]**
7. content block 不得成為 parser authority；block-level artifact metadata/ids 僅作 interaction/evidence metadata，不是 hierarchy truth source。 **[Maintainer-Confirmed] + [Doc-Confirmed]**
8. 本輪僅 shared-layer internal stabilization，未在本任務中變更 task-layout DTO、API schema/route、或 persistence migration 機制。 **[Code-Confirmed]**
9. retrieval/LLM/artifact persistence integration 與更細粒度互動 targeting 仍屬 future direction。 **[Future Direction]**
10. shared 層已新增 `ArtifactTargetLevel`（`document/chapter/section/task_unit/content_block`）與 `ArtifactTargetRef` 作為 content-block artifact target foundation metadata。 **[Code-Confirmed]**
11. `TaskUnitContentBlock` 新增 additive `artifact_target_refs`（optional）欄位；舊 payload 缺少該欄位時仍可正常反序列化，`artifact_ids` compatibility 行為不變。 **[Code-Confirmed]**
12. `artifact_target_refs` 僅是 target metadata，不代表 artifact persistence write path，不具 parser authority，且不改變 hierarchy truth（仍以 `chapters[].sections[].task_units[]` 為準）。 **[Code-Confirmed] + [From HLD]**
13. 本輪未引入 artifact repository 依賴、未引入 task-layout/API/persistence migration，question/evaluated_answer/retrieval 的 block-level integration 仍是後續工作。 **[Code-Confirmed] + [Future Direction]**
14. `ArtifactTargetLevel` / `ArtifactTargetRef` 已抽取至 `shared/artifact_target_model.py`，`shared/task_unit_model.py` 與 `api_schemas.py` 共用同一 target-level contract，避免 enum duplicated-definition drift。 **[Code-Confirmed]**

## 15. Future Direction Note: Content Block Segmentation Design Preparation

> 本節是 segmentation algorithm 的 documentation/design preparation，非 implementation。 **[Doc-Confirmed]**

### 15.1 Segmentation Input/Output Contract

1. input baseline：`TaskUnit.content`（string）；可選 future inputs 僅限 deterministic metadata/context（例如 parser-originated boundary hints）。 **[Future Direction]**
2. output baseline：ordered deterministic `TaskUnitContentBlock[]`，需保留原文順序並保持可回溯到原始 `content`。 **[Future Direction]**
3. compatibility requirement：`TaskUnit.content` 仍保留 compatibility role；multi-block generation 不得破壞既有 simple string consumer。 **[Code-Confirmed] + [Future Direction]**
4. segmentation output 屬 task-unit internal rich-content representation，不是 hierarchy truth。 **[Maintainer-Confirmed] + [From HLD]**

### 15.2 Segmentation Priority Strategy (Deterministic-Only)

1. 建議優先級：paragraph-first -> heading-aware merge/split -> list-aware split -> sentence fallback。 **[Future Direction]**
2. table/code block 先記為 future consideration，需 deterministic parsing rule 才能納入。 **[Future Direction]**
3. segmentation 必須 deterministic，不依賴 LLM、不依賴 semantic hallucination-style splitting。 **[Maintainer-Confirmed] + [Future Direction]**

### 15.3 Deterministic Block ID Strategy

1. 候選策略：
   - positional id（可讀性高、但插入漂移敏感）
   - hash-assisted id（穩定性較好、可讀性較低）
   - span-assisted id（對 quote linkage 友善、需 offset policy）
2. block id 不得依賴 random uuid / LLM output / retrieval result。 **[Maintainer-Confirmed] + [Future Direction]**
3. 方向：保留人類可讀前綴 + deterministic segment token（平衡可讀性與重算穩定性）。 **[Future Direction]**

### 15.4 Source Hash Strategy Direction

1. source hash 建議以 normalized text hash 為核心，需先定義 whitespace normalization policy。 **[Future Direction]**
2. 可評估 document-level hash + block-local hash 雙層策略，以區分全局編輯與局部漂移。 **[Future Direction]**
3. hash 只作 reference/invalidation support，不作 hierarchy authority。 **[From HLD] + [Future Direction]**

### 15.5 Quote Span Semantics Direction

1. quote/span metadata 可評估：absolute character offsets、block-local offsets、paragraph-local span。 **[Future Direction]**
2. quote span 需有 deterministic extraction rule，且與 `content_block_id` 一起描述 evidence target。 **[Future Direction]**
3. quote spans 不等於 hierarchy position，也不等於 parser authority。 **[Maintainer-Confirmed] + [Future Direction]**

### 15.6 Empty Content Behavior

1. default direction：empty content -> empty block list。 **[Future Direction]**
2. 不建議 synthetic empty semantic block，避免 downstream 對空內容誤判為有效 evidence target。 **[Future Direction]**

### 15.7 Block Size Policy Direction

1. 需定義 min/max block size、heading merge/split policy、oversized paragraph deterministic fallback。 **[Future Direction]**
2. policy 必須 parser-safe、deterministic、non-LLM。 **[Maintainer-Confirmed] + [Future Direction]**

### 15.8 Reparse Stability Risks

1. 主要漂移來源：paragraph insertion、heading renumbering、OCR normalization、whitespace normalization、parser version drift。 **[Future Direction]**
2. artifact/evidence linkage 在 reparse 後可能失配；後續需明確 stale/unresolved 分類邊界。 **[Future Direction]**

### 15.9 Compatibility Strategy (String -> Multi-Block)

1. staging direction：
   - Stage A：保留 `content` + 單塊 adapter（已存在）
   - Stage B：引入 deterministic multi-block generation（future）
   - Stage C：維持 compatibility mirror，逐步讓 consumers 以 `content_blocks` 為主
2. old persisted payload（無 `content_blocks`）仍需可載入並自動得到安全 fallback。 **[Code-Confirmed] + [Future Direction]**
3. 本節不引入 persistence migration、API change、task-layout payload change。 **[Doc-Confirmed]**

### 15.10 Guardrails (This Preparation Pass)

1. content block != hierarchy level。
2. segmentation 不改 parser authority。
3. segmentation 不改 hierarchy truth（仍以 `chapters[].sections[].task_units[]` 為準）。
4. segmentation 不依賴 LLM/retrieval。
5. segmentation 不新增 artifact persistence behavior。
6. segmentation metadata 僅 advisory/runtime support。

以上均屬 design preparation，不代表 implementation-ready algorithm 已落地。 **[Doc-Confirmed] + [Future Direction]**

## 16. Deterministic Content Block Segmentation Foundation (Implemented)

> 本節描述已落地的 shared-layer segmentation foundation；僅 shared 內部能力，不代表 endpoint/API/task-layout 行為變更。 **[Code-Confirmed]**

1. 已新增 explicit opt-in segmentation path：`TaskUnit.segment_content_blocks()` 與 shared helper `segment_task_unit_content(task_unit)`。 **[Code-Confirmed]**
2. `TaskUnit.to_content_blocks()` 預設行為維持 compatibility-safe 單塊適配，不會因本次變更自動改成多塊輸出。 **[Code-Confirmed]**
3. 已實作 deterministic split rules（目前最小可用）：
   - paragraph-first（blank-line paragraph boundaries）
   - list-item split（僅當整段每個 non-empty line 都是 deterministic list item pattern）
   - 無安全分割時 fallback 為單塊
   - 不做 LLM/NLP 語義切句。 **[Code-Confirmed]**
4. block id 策略：沿用 `build_default_content_block_id`，格式 `<task_unit_id>:content:<index>`；不依賴 random uuid/LLM/retrieval。 **[Code-Confirmed]**
5. segmented block metadata（advisory-only）包含：`source_hash`、`content_block_id`、`quote_span_start`、`quote_span_end`、`schema_version`。 **[Code-Confirmed]**
6. `source_hash` 目前以原始 `TaskUnit.content` 原文（不做 normalization）計算 deterministic SHA-256；同一輸入字串輸出穩定。 **[Code-Confirmed]**
7. quote span 語義：absolute character offsets，`quote_span_start` inclusive、`quote_span_end` exclusive，對應原始 `TaskUnit.content` 切片。 **[Code-Confirmed]**
8. empty content 行為維持：`content == ""` 時 segmentation 回傳空 block list（不生成 synthetic empty semantic block）。 **[Code-Confirmed]**
9. repeated segmentation 呼叫具 idempotent 結果；在同一內容輸入下輸出 block id、span、hash 穩定。 **[Code-Confirmed] + [Test-Confirmed]**
10. 本實作不變更 hierarchy truth、parser authority、artifact persistence、task-layout/API/persistence schema，也不接入 retrieval/LLM/question/evaluated_answer。 **[Code-Confirmed] + [Doc-Confirmed]**
