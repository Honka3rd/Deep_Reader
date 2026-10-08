# document_structure Detailed Design

## 1. Module Purpose

`document_structure/` 是結構化閱讀核心，負責把原文切分並收斂為 hierarchy-first 的 `StructuredDocument`，並提供結構一致性與結構相關持久化邊界。 **[Code-Confirmed]**

## 2. Position in Overall Architecture

- Document Structure Core

## 3. Key Files

| File | Responsibility | Notes |
|---|---|---|
| `document_structure/structured_document.py` | 定義 `StructuredDocument/StructuredChapter/StructuredSection` 契約與序列化 | normal `from_dict/from_json` 已 strict 要求 chapters；legacy loader 僅 migration-only **[Code-Confirmed]** |
| `document_structure/structured_document_builder.py` | 以 splitter 建立 structured document，錯誤時 fallback doc | 支援 parser mode 選擇入口 **[Code-Confirmed]** |
| `document_structure/structured_hierarchy_builder.py` | 將 flat sections 收斂成 chapter->section hierarchy | 主流程不再生成 structure_nodes mirror **[Code-Confirmed]** |
| `document_structure/document_hierarchy_index.py` | hierarchy-first 查找/flatten/一致性校驗 helper | `get_effective_sections` 已 hierarchy-only **[Code-Confirmed]** |
| `document_structure/section_splitter.py` | common parser split 實作 | 受 language registry 支持 **[Code-Confirmed]** |
| `document_structure/llm_section_splitter.py` | llm enhanced parser split 實作 | 作為 selector 另一條路徑 **[Code-Confirmed]** |
| `document_structure/section_splitter_selector.py` | common/llm split mode 選擇 | parser mode contract 中樞 **[Code-Confirmed]** |
| `document_structure/structured_document_store.py` | structured JSON load/save | normal load 經 strict hierarchy contract 校驗 **[Code-Confirmed]** |
| `document_structure/document_artifact_repository.py` | artifact repository 抽象介面 | lightweight document discovery plus section/chapter/task-unit/document-level/parser-replacement methods **[Code-Confirmed]** |
| `document_structure/structured_document_artifact_repository.py` | 具體 artifact repository（hierarchy-aware） | normal read/write 路徑要求 chapters；legacy 讀取僅 explicit migration helper；parser-level replacement clears derived artifacts **[Code-Confirmed]** |
| `document_structure/enhanced_parse_trigger_evaluator.py` | enhanced parse recommendation 評估器 | recommendation only，非 auto-switch **[Code-Confirmed] + [From HLD]** |
| `document_structure/document_structure_language_registry.py` | parser/regional/profile-evidence 用語言標記 registry | multi-consumer registry **[Code-Confirmed]** |

## 4. Main Responsibilities

1. 定義 hierarchy 主契約（Document -> Chapter -> Section）。 **[Code-Confirmed]**
2. 實現 common/llm enhanced 結構切分入口與 fallback 產物。 **[Code-Confirmed]**
3. 提供 hierarchy-first 查找與一致性檢查 helper。 **[Code-Confirmed]**
4. 管理 structured artifact 讀寫與 hierarchy-aware artifact 更新。 **[Code-Confirmed]**
5. 提供 enhanced parse recommendation 訊號。 **[Code-Confirmed]**
6. 提供 lightweight structured document discovery contract，供 API list/search 使用；不返回 hierarchy/content heavy payload。 **[Code-Confirmed]**

## 5. Non-Responsibilities

1. 不應直接承擔 API request/response mapping。 **[From HLD]**
2. 不應讓 profile metadata 直接硬控制 parser 切分規則。 **[From HLD]**
3. 不應在 task-layout read path 做隱性寫回。 **[From HLD]**
4. 不承擔 task-layout projection contract 本身（該責任屬 `section_tasks/` + `app/`；此 module 僅提供 hierarchy helper 與 persistence primitives）。 **[Code-Confirmed] + [From HLD]**
5. 不承擔 profile diagnostics 組裝與 response 投影責任。 **[Code-Confirmed] + [From HLD]**

## 6. Important Data Structures / Contracts

- `StructuredDocument`
- `StructuredChapter`
- `StructuredSection`
- `SectionSplitterMode`
- `DocumentArtifactRepository` (interface)
- `DocumentListItem`
- `EnhancedParseTriggerDecision`

以上是本 module 的高層契約代表。 **[Code-Confirmed]**

## 7. Architecture Constraints

1. persisted source of truth 是 `chapters[].sections[].task_units[]`。 **[Code-Confirmed] + [From HLD]**
2. 不重新引入 root `sections` mirror 作新寫入來源。 **[Code-Confirmed] + [From HLD]**
3. `structure_nodes` 不作主流程結構來源。 **[Code-Confirmed] + [From HLD]**
4. parser metadata 屬 advisory，不是 parser authority。 **[From HLD]**
5. `StructuredDocument.to_dict()` 預設不輸出 root `sections[]`/`structure_nodes[]`，僅在 legacy include 旗標顯式開啟時輸出。 **[Code-Confirmed]**
6. artifact 是 interaction output/support material，不是 hierarchy truth source。 **[Code-Confirmed] + [From HLD]**
7. artifact write path 必須以 hierarchy-aware targeting 為前提，不得反向改寫 chapter/section/task-unit identity。 **[Code-Confirmed]**
8. artifact availability projection ownership 不在本 module（屬 task-layout/coordinator DTO 層）。 **[Code-Confirmed] + [From HLD]**

## 8. Module Relationships

- depends on:
  - `language/`（language code + structure language registry）
  - `shared/`（task artifacts / task unit model）
- used by:
  - `document_preparation/`
  - `app/section_task_coordinator.py`
  - `section_tasks/`
- reads from:
  - structured JSON artifact
- writes to:
  - structured JSON artifact（store/repository）
- advisory relationship:
  - 與 `profile/` 是弱耦合（metadata advisory，不是 parser authority） **[From HLD]**

## 9. Main Flows Involving This Module

1. prepare flow：`structured_document_builder` build + `structured_document_store` save。 **[Code-Confirmed]**
2. hierarchy build flow：flat sections -> hierarchy chapters/sections。 **[Code-Confirmed]**
3. artifact write flow：repository 更新 section/chapter/task-unit artifacts。 **[Code-Confirmed]**
4. hierarchy helper flow（被 task-layout/runtime 使用）：提供 `get_effective_sections` 與 find helpers；不等同於擁有 task-layout projection contract。 **[Code-Confirmed] + [From HLD]**
5. enhanced recommendation flow：evaluator 輸出 should_recommend/score/reasons。 **[Code-Confirmed]**
6. hard reparse replacement flow：`save_reparsed_document(...)` sanitizes the accepted replacement document by clearing document/chapter/section/task-unit derived task artifacts before persistence. **[Code-Confirmed]**

## 10. Failure Semantics Matrix

| Scenario | Current Behavior | Notes |
|---|---|---|
| `StructuredDocument` normal 載入僅含舊 `sections`/`structure_nodes` | fail-fast | legacy payload 需走 explicit migration-only loader **[Code-Confirmed] + [Test-Confirmed]** |
| 無 chapters 的 runtime 結構查找 | 上層多為 fail-fast | helper 層逐步收斂 **[Code-Confirmed]** |
| severe hierarchy inconsistency | 上層 coordinator fail-fast | 不以 legacy fallback 掩蓋 **[Code-Confirmed]** |
| parser mode invalid | selector/上層拋錯 | 依 route/coordinator 轉 HTTP **[Code-Confirmed]** |

## 11. Persistence / Side Effects

- read persistence：是（structured store/repository）
- write persistence：是（structured store/repository）
- mutate structured document：是（builder/repository update）
- generate runtime projection：否（主要由 coordinator/task_layout DTO 層完成）
- call LLM：部分（`llm_section_splitter`）
- diagnostics only：否（同時有寫入/建模責任）

## 12. Known Legacy / Compatibility Behavior

| Legacy Item | Can Read | New Write | Runtime Primary Path |
|---|---:|---:|---:|
| root `sections[]` | Migration-only helper | No (default) | No |
| `structure_nodes[]` | Migration-only helper | No (default) | No |
| sections-only payload migration | Explicit only | Not automatic in normal repository write path | No |
| root artifact mirror | N/A | No | No |

說明：`find_*_effective(...allow_legacy_fallback=...)` 已完成退場；普通 helper API surface 不再暴露該參數，effective lookup 現為 hierarchy-only，sections-only legacy 文檔不再經由 effective helper 解析。 **[Code-Confirmed] + [Test-Confirmed] + [Maintainer-Confirmed]**
補充：legacy payload 若需處理，需走 explicit migration-only loader（如 `StructuredDocument.from_legacy_dict_for_migration` / `from_legacy_json_for_migration` 與 repository internal migration helper），不經 normal `from_dict/from_json` 與 ordinary repository write/read path。 **[Code-Confirmed]**

## 13. Terminology Governance Audit

| Term | Current Meaning in This Module | Classification | Governance Decision |
|---|---|---|---|
| root `sections[]` | legacy compatibility input field；normal `from_dict/from_json` 不消費，僅 migration-only loader 可讀 | **[Code-Confirmed]** | 不得描述為 primary persistence source |
| top-level sections | 若指 root `sections[]`，僅 compatibility 語境可用；正式契約應改稱 `chapters[].sections[]` hierarchy | **[Doc-Confirmed] + [Code-Confirmed]** | 在 module 文檔中避免作 current contract 用語 |
| `structure_nodes[]` | legacy experimental hierarchy field；normal `from_dict/from_json` 不消費，僅 migration-only loader 可讀 | **[Code-Confirmed]** | 僅保留 old JSON / round-trip compatibility 語義 |
| `StructuredDocumentNode` | compatibility type，非主流程 hierarchy source | **[Code-Confirmed]** | 不得在架構圖描述為 active main flow |
| flat `task_units` | 非 persistence truth；task units 應掛載於 section (`chapters[].sections[].task_units[]`) | **[Code-Confirmed] + [From HLD]** | 禁止作新 write contract |
| mirror（deprecated terminology） | 僅 historical/compatibility 搜尋語境；正式術語改為 `legacy compatibility fields` / `compatibility-only fields` | **[Maintainer-Confirmed] + [Doc-Confirmed]** | 不得作為現行 architecture contract/persistence authority/runtime primary path 術語 |
| legacy | 指 migration-only compatibility，不代表 runtime primary path | **[Doc-Confirmed] + [Code-Confirmed]** | 文檔必須顯式區分 compatibility vs primary contract |

### Terminology Validation Notes

1. 本 module 文檔現已將 `root sections[]` 與 `structure_nodes[]` 固定為 compatibility 語義，不再暗示主流程來源。  
2. task-layout 與 diagnostics ownership 已明確放在 module boundary 外，避免責任漂移。  
3. metadata / LLM classification 被明確標記為 advisory，非 parser authority。  
4. `mirror` 已退出正式 contract wording；文檔統一以 `legacy compatibility fields` / `compatibility-only fields` 表述。 **[Maintainer-Confirmed]**  
5. `allow_legacy_fallback` 已從普通 helper API surface 移除；runtime primary lookup 已不依賴 legacy fallback。 **[Code-Confirmed] + [Test-Confirmed]**  

## 14. Artifact Governance Boundary

### 14.1 Hierarchy Truth vs Artifact Output

1. hierarchy truth source 固定為 `chapters[].sections[].task_units[]`；artifact payload 不得成為 hierarchy truth source。 **[Code-Confirmed] + [From HLD]**  
2. section/task-unit artifact 僅作 interaction output，依既有 hierarchy target 更新，不可反向創建/重命名 hierarchy identity。 **[Code-Confirmed]**  
3. chapter summary/quiz artifact 目前持久化於 `document_task_artifacts.chapter_artifacts`（id-first key + legacy key candidate），屬內容輸出索引，不是 hierarchy identity source。 **[Code-Confirmed]**

### 14.2 Persistence Ownership

1. `document_structure` 擁有 artifact persistence primitives（repository contract + atomic save + hierarchy consistency guard）。 **[Code-Confirmed]**  
2. `document_structure` 不擁有 artifact availability projection contract；availability/validity 展示屬 coordinator + task-layout DTO。 **[Code-Confirmed] + [From HLD]**  
3. `document_structure` 不擁有 profile diagnostics projection contract。 **[Code-Confirmed] + [From HLD]**
4. Parser-level hard reparse replacement clears derived artifact payloads and referenced-artifact metadata from the replacement structured document before save; ordinary artifact/task-layout updates do not use this cleanup path. **[Code-Confirmed]**

### 14.3 Runtime Projection Boundary

1. availability（`has_summary/has_quiz/cache_valid`）是 runtime projection concern，不是 persisted hierarchy truth。 **[Code-Confirmed]**  
2. task-layout 讀路徑應消費 repository 已持久化 artifact 與 hierarchy，並做當次可用性判斷；該投影邏輯不在本 module。 **[Code-Confirmed] + [Inferred]**

### 14.4 Compatibility Boundary

1. legacy 可讀/migration 可保留，但不得重新引入 root sections mirror 或 flat task_units 作新 artifact write contract。 **[Code-Confirmed] + [From HLD]**  
2. `structure_nodes` compatibility 不延伸到 artifact truth contract。 **[Code-Confirmed]**

## 15. Storage Ownership Boundary

### 15.1 Owned By `document_structure`

1. `document_structure` owns the hierarchy persistence contract: `chapters[].sections[].task_units[]` remains the structured document truth source. **[Code-Confirmed] + [From HLD]**
2. `document_structure` owns the structured document storage contract through `StructuredDocumentStore`, `DocumentArtifactRepository`, and the hierarchy-aware concrete repository. **[Code-Confirmed]**
3. `document_structure` owns hierarchy-aware artifact write primitives only insofar as artifacts are persisted against existing hierarchy targets. **[Code-Confirmed]**

### 15.2 Not Owned By `document_structure`

1. `document_structure` does not own profile persistence; profile artifacts and advisory metadata snapshots belong to `profile/`. **[Code-Confirmed] + [From HLD]**
2. `document_structure` does not own retrieval persistence; FAISS index, node records, and fingerprint metadata belong to `retrieval/`, `fingerprint_handler.py`, and bundle orchestration. **[Code-Confirmed] + [Inferred]**
3. `document_structure` does not own runtime cache persistence; runtime bundle cache belongs to `bundle_factory.py` / `bundle_provider.py` and is not authoritative. **[Code-Confirmed] + [Inferred]**
4. `document_structure` does not own raw uploaded file storage; canonical raw document loading belongs to `doc_loaders/`. **[Code-Confirmed] + [Inferred]**

### 15.3 Storage Abstraction Boundary

1. `document_structure` owns the `StructuredDocument` contract and hierarchy truth. **[Code-Confirmed]**
2. `document_structure` owns the domain semantics of structured persistence across file and future DB-backed representations. **[Maintainer-Provided] + [Future Direction]**
3. `document_structure` does not own storage backend selection; that belongs to future `config/` backend policy. **[Maintainer-Provided] + [Future Direction]**
4. `document_structure` does not own DB rollout strategy, migration rollout switches, or backend enablement policy. **[Maintainer-Provided] + [Future Direction]**
5. A future storage abstraction must preserve hierarchy-first `StructuredDocument` semantics and must not make the backend, schema, or file path a new hierarchy authority. **[Maintainer-Provided] + [Future Direction]**

## 16. Current Risks

1. risk：新 helper/feature 誤把 legacy sections 重新引入 runtime lookup
- why：會破壞 hierarchy-only 路徑一致性
- guardrail：維持 effective lookup hierarchy-only regression + coordinator fail-fast regression

2. risk：common/llm parser mode decision 無統一契約
- why：可能出現 recommendation 與實際行為落差
- guardrail：補 recommendation decision contract 文檔

3. risk：artifact 寫入若偏離 hierarchy source
- why：可能再度形成 dual representation drift
- guardrail：維持 repository hierarchy-aware write tests

4. risk：Part->Chapter 非目標若未清晰文件化
- why：後續開發者可能重引入多層 persistence
- guardrail：在 module docs/non-goals 明確固定 current scope **[From Proposal] + [From HLD]**

## 17. Open Questions for Maintainer

1. `llm_section_splitter` 的輸出契約是否要加入更明確 schema guard（僅文檔層）？
2. enhanced recommendation 的 score threshold 調整責任層級在哪（config 或 evaluator 固化）？
3. chapter summary/quiz artifact 長期是否固定以 `document_task_artifacts.chapter_artifacts` 為唯一寫入權威（chapter node `task_artifacts` 僅作投影輔助）？
4. `legacy compatibility fields`（原 historical mirror wording）是否需在全專案文檔補一份 alias 對照表以利遷移搜尋？

## 18. Suggested Next Documentation Improvements

1. 增加「common vs llm enhanced parser lifecycle」sequence diagram。
2. 增加 artifact repository write boundary 狀態圖（section/chapter/task-unit/document-level）。
3. 補 `document_structure_language_registry` 的 consumer matrix。

## 19. Future Direction Note: Rich Task-Unit Content Governance Preparation

> 本節屬 future-direction governance preparation，非當前 implementation。 **[Inferred]**

1. hierarchy source 固定為 `chapters[].sections[].task_units[]`；不得把 `Document -> Chapter -> Section -> TaskUnit -> ContentBlock` 描述成 persisted hierarchy contract。 **[Code-Confirmed] + [Inferred]**
2. rich content content-block/segment model 的定位是 `task_unit` 內部 render segmentation / interaction segmentation，不是 hierarchy node、不是 persisted structure authority、也不是 runtime navigation hierarchy。 **[Inferred]**
3. content block 不得替代 `task_unit`、`section`、`chapter` ownership；`TaskUnit` 仍是主要 interaction container。 **[Code-Confirmed] + [Inferred]**
4. `content_block_id` / `content_segment_id` 的定位是 interaction target id，不是 hierarchy node id。 **[From Proposal] + [Inferred]**
5. future content-block artifacts 可作 interaction/annotation/evidence target，但仍不是 hierarchy truth source，且不得反向決定 chapter/section ownership 或 task-unit identity。 **[Inferred]**
6. rich content governance 需避免 dual hierarchy representation（例如把 content block 漂移為 structure authority 或 retrieval authority）。 **[Inferred]**
7. 目前 `task_unit.content` 為 string；兼容方向可規劃 `string -> single content block` adapter，但該 adapter 屬 future compatibility strategy，不是 legacy fallback runtime path。 **[Code-Confirmed] + [Inferred]**
8. 本節僅做 governance/邊界對齊；不引入 persistence schema、migration algorithm、runtime API 或 execution model。 **[Doc-Confirmed]**

## 20. Future Direction Note: Artifact Target Validation Boundary Preparation

> 本節屬 validation-boundary governance preparation，非當前 repository/runtime implementation。 **[Future Direction] + [Maintainer-Confirmed]**

### 20.1 Current Boundary (What Is Already True)

1. `ArtifactTargetRef` 目前語義是 metadata + target intent，不是 validated persistence truth。 **[Code-Confirmed] + [Doc-Confirmed]**
2. hierarchy truth source 仍固定為 `chapters[].sections[].task_units[]`；content block 不是 hierarchy node。 **[Code-Confirmed]**
3. content endpoint 現階段僅 pass through target metadata，不查 artifact repository、不驗證 artifact existence、不創建 persistence target。 **[Doc-Confirmed]**

### 20.2 Future Validation Lifecycle (Design Boundary)

`ArtifactTargetRef` (request intent)
-> hierarchy-aware validation
-> resolved target identity
-> repository trust boundary

治理要求：repository 層不得直接 trust target ref；必須先完成 hierarchy-aware resolution。 **[Future Direction] + [Maintainer-Confirmed]**

### 20.3 Reparse / Restructure Stale-Ref Semantics

以下情況可使 target ref 失效：

1. task-unit split policy 改變
2. content-block id regeneration
3. section/chapter restructure
4. source text edit
5. `source_hash` mismatch

治理語義：stale ref != malformed payload；future validation layer 必須可區分 stale/unresolved/malformed。 **[Future Direction] + [Maintainer-Confirmed]**

### 20.4 Failure Boundary (Fail-Fast Direction)

future validation boundary 應 fail-fast 於以下類型：

1. invalid `target_level`
2. required id 缺失
3. hierarchy mismatch
4. `content_block_id` not found under selected `task_unit_id`
5. `source_hash` mismatch
6. stale ref after reparse/restructure

並區分 error class：malformed / unresolved / stale / hierarchy-mismatched。 **[Future Direction] + [Maintainer-Confirmed]**

### 20.5 Allowed Target Combinations (Endpoint Context Policy)

1. `target_level=content_block`：必須至少包含 `task_unit_id` + `content_block_id`。
2. `target_level=task_unit`：必須至少包含 `task_unit_id`。
3. `target_level=document/chapter/section`：可保留 enum vocabulary，但本階段不宣稱完整 artifact persistence semantics。

本政策用於避免 enum 漂移被誤解為完整 persistence 支援。 **[Doc-Confirmed] + [Maintainer-Confirmed]**

### 20.6 Minimum Metadata Glossary (Cross-Module Anti-Drift)

允許的最小 glossary key：

1. `source_hash`
2. `content_block_id`
3. `quote_span_start`
4. `quote_span_end`
5. `schema_version`

治理要求：metadata glossary 用於 interaction/reference pass-through，非 parser authority、非 hierarchy truth、非 artifact persistence truth。 **[Doc-Confirmed] + [Maintainer-Confirmed]**

### 20.7 Non-Goals In This Pass

1. 不實作 repository validation engine。
2. 不實作 artifact read/write flow。
3. 不實作 persistence schema migration。
4. 不接入 retrieval/question/evaluated_answer/LLM integration。

以上僅為 future design boundary preparation。 **[Doc-Confirmed]**

## 21. Future Direction Note: Content Block Segmentation Boundary Design Preparation

> 本節僅做 segmentation governance/boundary 設計準備，不代表 segmentation algorithm 已實作。 **[Doc-Confirmed]**

### 21.1 Segmentation vs Hierarchy Parser Boundary

1. hierarchy parser responsibility 仍是建立/維護 `chapters[].sections[].task_units[]` 主契約。 **[Code-Confirmed] + [From HLD]**
2. segmentation responsibility 定位於 task-unit 內部內容切分（render/interaction segmentation），不重新定義 hierarchy。 **[Maintainer-Confirmed] + [Future Direction]**
3. `task_unit` boundary 與 `content_block` boundary 必須分離：`task_unit` 是 hierarchy-interaction container；`content_block` 是其內部 interaction target。 **[Maintainer-Confirmed] + [Future Direction]**

### 21.2 Content-Block Persistence Semantics (Non-Authority Rule)

1. 即使 future `content_blocks` 有 persisted 表示，content block 仍不是 hierarchy truth、不是 structure authority、不是 runtime navigation hierarchy。 **[Maintainer-Confirmed] + [Future Direction]**
2. 禁止形成 `Document -> Chapter -> Section -> TaskUnit -> ContentBlock` persisted hierarchy model。 **[Maintainer-Confirmed] + [Future Direction]**
3. content block 不得覆蓋 chapter/section ownership，也不得替代 task-unit identity。 **[Maintainer-Confirmed] + [Future Direction]**

### 21.3 Stale-Target Semantics for Reparse/Resegmentation

1. reparse/resegmentation 後，`content_block_id` 可能 stale。 **[Maintainer-Confirmed] + [Future Direction]**
2. stale target != malformed payload；後續邊界需區分 malformed / unresolved / stale / source-mismatched。 **[Maintainer-Confirmed] + [Future Direction]**
3. stale detection 方向可依賴 `source_hash`、`quote_span`、`segmentation_version`、`content fingerprint`，但本輪不定義 executable detection engine。 **[Maintainer-Confirmed] + [Future Direction]**

### 21.4 content_block_id Dependency Semantics

1. `content_block_id` 必須依附 `task_unit_id` 語境，不可脫離 task unit 成為獨立 hierarchy node。 **[Maintainer-Confirmed] + [Future Direction]**
2. `content_block_id` 是 interaction target id，不是 hierarchy node id。 **[Code-Confirmed] + [Maintainer-Confirmed]**
3. segmentation 不能改變 `task_unit_id` identity，不得反向重寫 hierarchy ownership。 **[Maintainer-Confirmed] + [Future Direction]**

### 21.5 Legacy String Compatibility Direction

1. 現況 `task_unit.content`（string）仍是 compatibility field，`string -> block` 屬 adapter strategy。 **[Code-Confirmed]**
2. future multi-block strategy 不得變成 legacy fallback runtime path，也不得回退 root-sections/structure_nodes compatibility 主流程。 **[Maintainer-Confirmed] + [Future Direction]**
3. compatibility 演進需避免 dual hierarchy representation。 **[Maintainer-Confirmed] + [Future Direction]**

### 21.6 Segmentation Non-Authority and Artifact Boundary Guardrails

1. segmentation 不得改變 hierarchy truth，不得成為 parser authority 或 retrieval authority。 **[Maintainer-Confirmed] + [Future Direction]**
2. `ArtifactTargetRef` 與 content-block target metadata 不得漂移為 persistence truth 或 hierarchy authority。 **[Maintainer-Confirmed] + [Future Direction]**
3. 允許 resegmentation policy change 與 content block regeneration；不得假設 `content_block_id` 永久穩定。 **[Maintainer-Confirmed] + [Future Direction]**
4. 本節不引入 segmentation algorithm、persistence schema、repository logic、runtime API、stale-detection engine、retrieval integration。 **[Doc-Confirmed]**

## 22. Future Direction Note: DB-Centric Structured Persistence Migration

> 本節屬 DB-centric structured persistence migration 的 documentation/governance preparation，非目前 implementation。 **[Maintainer-Provided] + [Future Direction]**

### 22.1 Current Active Implementation Boundary

1. 目前 active structured persistence implementation 仍是 structured JSON file storage。 **[Code-Confirmed]**
2. 現有 `data/structured/*.structured.json` file path 在 DB migration 期間仍是有效 storage path，不得在本階段移除或破壞。 **[Maintainer-Provided]**
3. DB storage 是 future representation target，不是目前 runtime read/write behavior。 **[Maintainer-Provided] + [Future Direction]**

### 22.2 StructuredDocument Contract Preservation

1. DB storage 必須保留 `StructuredDocument` hierarchy contract。 **[Maintainer-Provided] + [Future Direction]**
2. hierarchy truth 仍固定為 `chapters[].sections[].task_units[]`。 **[Code-Confirmed] + [Maintainer-Provided]**
3. DB rows、JSONB documents、relational tables 都只是 persistence representations，不是新的 parser authority、structure authority、或 hierarchy identity authority。 **[Maintainer-Provided] + [Future Direction]**
4. DB schema 不得改變 chapter/section/task-unit identity semantics。 **[Maintainer-Provided] + [Future Direction]**

### 22.3 Legacy and Repository Boundary

1. DB migration 必須保留 explicit legacy migration-only boundary。 **[Maintainer-Provided] + [Future Direction]**
2. DB-backed repository 必須仍是 hierarchy-aware repository，不得信任 root `sections[]`、`structure_nodes[]`、或 flat `task_units` 作 primary flow。 **[Maintainer-Provided] + [Future Direction]**
3. DB read-path switch 前必須先定義 validation rules，確認 hierarchy parity、target resolution、artifact target consistency、以及 fail-fast error behavior。 **[Maintainer-Provided] + [Future Direction]**
4. file-backed structured JSON 在 DB readiness 被驗證前仍可作 compatibility / fallback / migration source。 **[Maintainer-Provided] + [Future Direction]**

### 22.4 Separate Persistence Concerns

1. content blocks 是 task-unit internal interaction/render concern，不是 structured hierarchy persistence authority。 **[Maintainer-Provided] + [Future Direction]**
2. artifacts 是 interaction output；artifact persistence migration 需要獨立 track，不應混入 hierarchy truth contract。 **[Code-Confirmed] + [Maintainer-Provided]**
3. profile metadata 是 advisory；profile DB migration 需要獨立 track，不得成為 parser authority。 **[Code-Confirmed] + [Maintainer-Provided]**
4. retrieval records / FAISS artifacts 是 retrieval persistence concern，需要獨立 DB migration track。 **[Maintainer-Provided] + [Future Direction]**
5. raw files / user-uploaded documents 是 user-scoped ownership/copyright concern；DB migration 不代表跨使用者共享文件內容。 **[Maintainer-Provided]**
6. 本節不引入 DB schema、DB dependency、repository implementation、runtime/API behavior change、data migration tooling、或 `data/` retirement execution。 **[Doc-Confirmed]**

## 23. Future Direction Note: Source-Agnostic Manual Structure Override

> 本節記錄「任何文檔解析失敗時」的人工結構修正方向，不限於 OCR/scanned PDF。這是 future-direction design，不代表目前已有 API 或 parser implementation。 **[Maintainer-Provided] + [Future Direction]**

### 23.1 Problem Scope

1. 任何 raw source 都可能產生錯誤 hierarchy：born-digital PDF、掃描 OCR PDF、TXT、EPUB-like text dump、LLM split plan、common parser、native outline、TOC projection 都可能失敗。 **[Maintainer-Provided] + [Inferred]**
2. manual structure override 的目的，是在自動 parser / outline / TOC / LLM fallback 失敗後，由使用者提供顯式目錄或結構邊界，觸發一次明確 reparse。 **[Maintainer-Provided] + [Future Direction]**
3. 這不是 OCR 專用補丁，也不是只為 `暗水幽靈` 設計；OCR 只是其中一個會暴露自動目錄失敗的來源。 **[Maintainer-Provided]**

### 23.2 Authority Boundary

1. 使用者提供的 manual TOC / manual structure plan 是 explicit parser input，不是第二套 persisted hierarchy。 **[Future Direction]**
2. reparse 成功後，唯一 runtime hierarchy truth 仍必須是 `chapters[].sections[].task_units[]`。 **[Code-Confirmed] + [Future Direction]**
3. manual plan 不得作為 task-layout read path 的 hidden mutation，也不得直接編輯 task-layout projection。 **[From HLD] + [Future Direction]**
4. manual plan 不得重新引入 root `sections[]`、`structure_nodes[]`、或 flat `task_units` 作 primary flow。 **[From HLD] + [Future Direction]**
5. manual plan provenance 可保存在 `parse_provenance`，但 provenance 是 observability，不是 parser authority 或 fallback hierarchy source。 **[Future Direction]**

### 23.3 Input Model Direction

最小 manual structure entry 應描述使用者意圖與可驗證定位：

```text
title
level
start anchor
optional end anchor
optional external/user id
optional notes
```

anchor 可分階段支持：

1. `char_start` / `char_end`：適用於任何已抽取 raw text 的文檔，最 source-agnostic。 **[Future Direction]**
2. `page_index` / `page_range`：適用於 PDF/OCR/native page-boundary 可用的文檔。 **[Future Direction]**
3. `printed_page_number + offset_hypothesis`：可作後續擴展，不應作 MVP 唯一定位方式。 **[Future Direction]**
4. `title_match_hint`：只能作 validation/evidence 輔助，不得單獨授權 projection。 **[Future Direction]**

### 23.4 Projection Rules

1. manual structure projection 必須收斂到現有 two-layer normalized hierarchy。 **[From HLD] + [Future Direction]**
2. `level=1` 可形成 chapter；`level=2` 可形成 section。 **[Future Direction]**
3. 只有 chapter entries 時，每個 chapter 應生成一個同名 section，以維持 `chapters[].sections[]` contract。 **[Future Direction]**
4. manual structure MVP 僅允許 `level=1` chapter 與 `level=2` section；`level>2` 必須 fail-fast，不得自動折疊或隱式合併。 **[Maintainer-Provided] + [Future Direction]**
5. task-unit generation 仍是 downstream concern，不由 manual TOC 直接持久化 flat task units。 **[Future Direction]**

### 23.5 Validation and Failure Semantics

manual plan 必須先 validation，再 commit reparse：

1. entry title 不得為空。 **[Future Direction]**
2. anchors 必須在 raw text/page boundary 範圍內。 **[Future Direction]**
3. ranges 必須單調、非重疊、非空。 **[Future Direction]**
4. hierarchy levels 必須只包含 chapter/section 兩層；任何 `level>2` 或跳層結構都應 fail-fast。 **[Maintainer-Provided] + [Future Direction]**
5. partial projection 不得落盤；validation 失敗時應保留現有 structured document。 **[Future Direction]**
6. validation failure 不應 silent fallback 到 common parser 並宣稱 manual reparse 成功。 **[Future Direction]**

### 23.6 Reparse Lifecycle Direction

建議 lifecycle：

```text
manual structure request
  -> validate / preview
  -> explicit commit reparse
  -> build StructuredDocument
  -> save single active hierarchy source
  -> downstream task-layout reads normal hierarchy
```

成功 reparse 的 `parse_provenance` 建議包含：

```text
requested_parser_mode=manual_structure
effective_parser_mode=manual_structure_projection
source=user_supplied_structure
fallback_used=false
manual_entry_count
anchor_type
validation_summary
```

本方向不實作 API、schema、builder、repository、或 UI；僅固定 future design boundary。 **[Doc-Confirmed]**

## 24. Page-Backed Manual Structure Anchors

> 本節支援 UI TOC editor 的 page-first anchor UX。Manual `page_range` draft building is implemented when preparation-provided page-boundary evidence is available; task-layout prefill projection remains a separate checkpoint. **[Maintainer-Provided] + [Code-Confirmed] + [Future Direction]**

1. `document_structure/` owns deterministic validation and hierarchy projection semantics for manual structure anchors. **[Code-Confirmed] + [From HLD]**
2. Manual projection can represent `page_range`, and manual draft building can resolve page-backed anchors into raw-text spans when validated page boundaries are supplied. **[Code-Confirmed]**
3. Page-backed manual commit resolves validated `page_range` anchors using preparation-provided page boundaries before building a hierarchy-only `StructuredDocument`. **[Code-Confirmed]**
4. Page-backed anchors are validated for page-boundary availability, unique page-index evidence, page existence/resolvability, sibling overlap, empty projected ranges, stale source evidence, and source hash consistency through the projection/source-evidence/draft-build gates. **[Code-Confirmed]**
5. Accepted manual page anchors record advisory parse provenance including anchor type, source hash, validation summary, entry count, chapter count, and section count. Provenance is observability only and must not become a second hierarchy source. **[Code-Confirmed]**
6. `project_structure_anchor_evidence(...)` can expose lightweight chapter/section anchor evidence from the existing parsed hierarchy for task-layout/UI prefill. This evidence is read-only projection metadata and does not rewrite hierarchy identity or artifacts. **[Code-Confirmed]**
7. If page evidence is missing or ambiguous, including duplicate `page_index` boundaries, validation fails for `page_range` and leaves `char_range` as the explicit fallback. **[Maintainer-Provided] + [Code-Confirmed]**
8. Existing-structure anchor evidence prefers `page_range` when validated page boundaries cover the current hierarchy spans, falls back to `char_range` when page evidence is absent, and marks missing/invalid spans as unavailable instead of inventing anchors from titles, OCR guesses, profile metadata, or task-layout state. **[Code-Confirmed]**

## 25. TOC Shape Classification and Two-Layer Projection

> 本節記錄 deterministic TOC projection 的 shape handling。 **[Code-Confirmed]**

1. `TableOfContentsDetector` classifies TOC evidence into `flat_chapter`, `chapter_section`, `deep_hierarchy`, or `unknown` without using metadata or LLM output as parser authority. **[Code-Confirmed]**
2. Validated TOC projection maps flat chapter entries to chapter-level sections; `structured_hierarchy_builder` then materializes one chapter with one same-name section for each entry. **[Code-Confirmed]**
3. Validated chapter/section TOCs map level-1 entries to `toc_chapter` and level-2 entries to `toc_subsection`, producing `chapters[].sections[]` without root `sections[]` or `structure_nodes` as persisted truth. **[Code-Confirmed]**
4. Deep TOC levels (`level > 2`) are collapsed into the nearest projected section content range rather than persisted as a third hierarchy level. **[Code-Confirmed]**
5. TOC projection provenance records original level, projected level, printed page number, resolved source page index, and merge reason for collapsed entries. Provenance is observability only and does not become a second hierarchy source. **[Code-Confirmed]**
6. Task-unit generation remains downstream of the resulting two-layer sections; TOC projection does not write flat task units or artifact targets. **[From HLD] + [Code-Confirmed]**

## 26. TOC Page/Character Boundary Validation and Atomic Fallback

> 本節記錄 validated TOC projection 的 page/char gate。 **[Code-Confirmed]**

1. TOC projection requires body title matches after the detected TOC region; insufficient body-title recall rejects projection without partially applying TOC sections. **[Code-Confirmed]**
2. Printed page numbers are validated against body title offsets through one explicit page-offset hypothesis. Mixed or incompatible offsets reject projection atomically. **[Code-Confirmed]**
3. Printed page anchors must resolve to available source page boundaries after applying the offset; missing page anchors reject projection. **[Code-Confirmed]**
4. Projected section ranges must be non-empty, monotonic, and non-overlapping before `StructuredSection` objects are materialized. **[Code-Confirmed]**
5. When any TOC projection gate fails, `StructuredDocumentBuilder` preserves the current parser result and does not write `toc_projection` provenance or claim `validated_toc_projection` as the effective parser. **[Code-Confirmed]**
6. This path does not introduce root `sections[]` persistence, `structure_nodes` main flow, hidden task-layout mutation, profile write-back, or LLM/parser-metadata authority. **[Code-Confirmed] + [From HLD]**

## 27. Universal Page-Level TOC Candidate Scoring

> 本節記錄 page-level TOC candidate scoring。Candidate detection identifies pages worth reconstructing or validating later; it does not by itself authorize structure splitting. **[Code-Confirmed]**

1. `TableOfContentsDetector.detect_page_candidates(...)` assigns deterministic scores from page-local evidence and returns candidate pages once the score reaches the conservative threshold. **[Code-Confirmed]**
2. Scoring covers document-relative early position, horizontal/vertical writing mode, left-to-right/right-to-left reading order, multiple text columns, coordinate OCR availability, short-title density, dotted leader lines, page-number evidence, separated page-number columns, circled/boxed page numbers, OCR-fragmented title density, and explicit TOC markers. **[Code-Confirmed]**
3. Candidate scoring can detect TOC-like pages even when TOC markers are missing or page numbers are visually separated from titles. Missing or unreliable anchors still require later entry reconstruction and global validation before projection. **[Code-Confirmed]**
4. `detected` and splitting usability remain separate decisions: page candidates may raise `detected=true`, while `validate_for_projection(...)` and `StructuredDocumentBuilder` still enforce entry count, page-number coverage, body-title matches, monotonic page mapping, and non-overlapping ranges before any hierarchy replacement. **[Code-Confirmed]**
5. This path uses OCR/layout text evidence only; it does not use metadata, diagnostics profiles, or LLM classification as parser authority and does not persist an alternate hierarchy source. **[Code-Confirmed] + [From HLD]**

## 28. Geometry-First TOC Entry Reconstruction

> 本節記錄 page-level TOC entry reconstruction。Reconstruction creates auditable title/page-number pairs from OCR geometry for later validation; it does not by itself authorize hierarchy replacement. **[Code-Confirmed]**

1. `PdfPageLayoutEvidence` can retain OCR word-level geometry from TSV input, including text, confidence, bounding box, and block/paragraph/line/word indices. This evidence is page-local parser evidence, not hierarchy truth. **[Code-Confirmed]**
2. `TableOfContentsDetector.reconstruct_page_entries(...)` reconstructs TOC candidate entries from geometry when word boxes are available. The output records title, page number, source page index, title region box, page-number region box, confidence, and evidence reasons. **[Code-Confirmed]**
3. Horizontal TOC reconstruction groups OCR words by row, orders title regions left-to-right, excludes leader tokens from titles, and pairs the title with the right-side page-number region by row alignment and leader-line endpoint evidence. **[Code-Confirmed]**
4. Vertical TOC reconstruction clusters OCR words into columns, sorts right-to-left pages by descending x-coordinate, orders title characters top-to-bottom within each column, and pairs the title with the lower page-number region. **[Code-Confirmed]**
5. Circled page numbers are recognized during geometry reconstruction for bounded page-number regions. The current implementation reconstructs candidate pairs only; later checkpoints still own OCR quality gates, global page-number validation, and projection authorization. **[Code-Confirmed]**
6. Geometry reconstruction is detection-only and must not replace raw text, page-boundary evidence, `chapters[].sections[]`, or parser validation. It does not write task-layout state, profile diagnostics, artifacts, root `sections[]`, or `structure_nodes`. **[Code-Confirmed] + [From HLD]**

## 29. Logical Reading-Order Normalization With Coordinate Preservation

> 本節記錄 TOC OCR geometry 的 logical reading-order normalization。The normalized stream exists only for deterministic detection and audit; it does not replace raw text or become a hierarchy source. **[Code-Confirmed]**

1. `TableOfContentsDetector.normalize_page_reading_order(...)` converts page OCR word boxes into `TocNormalizedToken` records in logical reading order. Each token retains page index, logical index, group index, normalized text, raw OCR text, original coordinate box, confidence, writing mode, reading order, rotation/orientation, and order hypothesis. **[Code-Confirmed]**
2. Horizontal pages use row grouping with left-to-right or right-to-left ordering according to the page reading order. Vertical pages use column grouping with top-to-bottom ordering within each column and right-to-left column sequencing when page evidence indicates RTL. **[Code-Confirmed]**
3. `TableOfContentsDetector.reconstruct_normalized_page_pairs(...)` exposes normalized title/page pairs with page index, title/page-number boxes, raw OCR sequence, writing mode, reading order, rotation, order hypothesis, evidence reasons, confidence, and confidence breakdown. **[Code-Confirmed]**
4. The normalized stream preserves raw OCR and coordinates for audit while giving detection code a stable logical order. It must not overwrite `PdfPageLayoutEvidence.ocr_text`, page-boundary evidence, source hashes, or extracted raw text. **[Code-Confirmed]**
5. Normalized pairs are detection-only and do not authorize projection. Later quality gates and global validation still decide whether entries are usable for splitting. **[Code-Confirmed] + [From HLD]**
6. This path does not introduce root `sections[]`, `structure_nodes`, hidden task-layout mutation, diagnostics profile write-back, metadata authority, or LLM parser authority. **[Code-Confirmed] + [From HLD]**

## 30. TOC-Specific OCR Quality Gates

> 本節記錄 TOC OCR quality gates。A page can be detected as TOC-like while still rejected as unusable for splitting when OCR entries are incomplete or unreliable. **[Code-Confirmed]**

1. `TableOfContentsDetector.evaluate_toc_ocr_quality(...)` evaluates page-local TOC OCR quality without mutating parser output or authorizing projection. It returns `detected` separately from `usable_for_splitting`. **[Code-Confirmed]**
2. The quality gate measures reconstructed entry count, title completeness, page-number coverage, geometric title/page pairing coverage, page-number ordering consistency, body-title recall, confidence, and explicit rejection reasons. **[Code-Confirmed]**
3. A high-scoring TOC-like page with missing or corrupt page numbers can remain `detected=true` while `usable_for_splitting=false`. The gate does not infer unreadable page numbers from sequence, row count, or title order. **[Code-Confirmed]**
4. Split usability requires enough reconstructed entries, complete titles, sufficient page-number coverage, paired title/page-number geometry, monotonic page numbers, and body-title recall against supplied body text. Missing body text rejects usability rather than silently trusting OCR layout. **[Code-Confirmed]**
5. Quality gate rejection preserves reasons such as missing reconstructed entries, incomplete titles, missing page numbers, missing geometric pairs, page-order failure, low body-title recall, and missing body text. **[Code-Confirmed]**
6. This path is detection/validation evidence only. It does not persist hierarchy, replace current parser output, write task-layout/profile/artifacts, or introduce root `sections[]`, `structure_nodes`, metadata authority, or LLM parser authority. **[Code-Confirmed] + [From HLD]**

## 31. Page-Number Region Recognition and Global Offset Validation

> 本節記錄 TOC page-number region recognition and explicit printed-page offset validation。Region recognition normalizes bounded page-number tokens; global validation decides whether one page-number-to-source-page mapping is reliable enough for later projection. **[Code-Confirmed]**

1. `TableOfContentsDetector.recognize_page_number_regions(...)` emits page-local `TocPageNumberRegion` records from reconstructed TOC title/page pairs. Each record keeps the source page index, raw page-number token, normalized text, parsed value, numeral system, region box, confidence, and evidence reasons. **[Code-Confirmed]**
2. Page-number recognition is region-scoped: only page-number regions paired by TOC geometry are normalized as anchors. The detector does not scan arbitrary body text or profile metadata for parser authority. **[Code-Confirmed] + [From HLD]**
3. Recognition supports Arabic numerals, multi-digit numbers, Chinese numerals, Roman numerals, circled numbers, boxed/bracketed numbers, and fragmented OCR digits such as spaced digit tokens. Raw OCR is preserved so symbols normalized into digits still retain their original evidence. **[Code-Confirmed]**
4. `TableOfContentsDetector.validate_page_number_anchor_offsets(...)` validates explicit printed-page-to-source-page offset hypotheses against supplied page text. A valid result requires monotonic page numbers, plausible source-page anchors, and at least two body-title matches under one unambiguous offset. **[Code-Confirmed]**
5. Ambiguous offsets are rejected atomically. If multiple offsets produce the same sufficient title-match evidence, or if matches are insufficient, page numbers are non-monotonic, page text is missing, or anchors fall out of range, validation returns rejection reasons and no selected offset. **[Code-Confirmed]**
6. This path produces validation evidence only. It does not persist hierarchy, replace current parser output, mutate task-layout/profile/artifacts, or introduce root `sections[]`, `structure_nodes`, metadata authority, or LLM parser authority. **[Code-Confirmed] + [From HLD]**
