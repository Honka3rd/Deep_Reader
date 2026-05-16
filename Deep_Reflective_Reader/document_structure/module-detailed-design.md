# document_structure Detailed Design

## 1. Module Purpose

`document_structure/` 是結構化閱讀核心，負責把原文切分並收斂為 hierarchy-first 的 `StructuredDocument`，並提供結構一致性與結構相關持久化邊界。 **[Code-Confirmed]**

## 2. Position in Overall Architecture

- Document Structure Core

## 3. Key Files

| File | Responsibility | Notes |
|---|---|---|
| `document_structure/structured_document.py` | 定義 `StructuredDocument/StructuredChapter/StructuredSection` 契約與序列化 | 新 JSON 預設不輸出 root `sections`/`structure_nodes` **[Code-Confirmed]** |
| `document_structure/structured_document_builder.py` | 以 splitter 建立 structured document，錯誤時 fallback doc | 支援 parser mode 選擇入口 **[Code-Confirmed]** |
| `document_structure/structured_hierarchy_builder.py` | 將 flat sections 收斂成 chapter->section hierarchy | 主流程不再生成 structure_nodes mirror **[Code-Confirmed]** |
| `document_structure/document_hierarchy_index.py` | hierarchy-first 查找/flatten/一致性校驗 helper | `get_effective_sections` 已 hierarchy-only **[Code-Confirmed]** |
| `document_structure/section_splitter.py` | common parser split 實作 | 受 language registry 支持 **[Code-Confirmed]** |
| `document_structure/llm_section_splitter.py` | llm enhanced parser split 實作 | 作為 selector 另一條路徑 **[Code-Confirmed]** |
| `document_structure/section_splitter_selector.py` | common/llm split mode 選擇 | parser mode contract 中樞 **[Code-Confirmed]** |
| `document_structure/structured_document_store.py` | structured JSON load/save | persistence store **[Code-Confirmed]** |
| `document_structure/document_artifact_repository.py` | artifact repository 抽象介面 | section/chapter/task-unit/document-level methods **[Code-Confirmed]** |
| `document_structure/structured_document_artifact_repository.py` | 具體 artifact repository（hierarchy-aware） | 含 legacy load-time migration **[Code-Confirmed]** |
| `document_structure/enhanced_parse_trigger_evaluator.py` | enhanced parse recommendation 評估器 | recommendation only，非 auto-switch **[Code-Confirmed] + [From HLD]** |
| `document_structure/document_structure_language_registry.py` | parser/regional/profile-evidence 用語言標記 registry | multi-consumer registry **[Code-Confirmed]** |

## 4. Main Responsibilities

1. 定義 hierarchy 主契約（Document -> Chapter -> Section）。 **[Code-Confirmed]**
2. 實現 common/llm enhanced 結構切分入口與 fallback 產物。 **[Code-Confirmed]**
3. 提供 hierarchy-first 查找與一致性檢查 helper。 **[Code-Confirmed]**
4. 管理 structured artifact 讀寫與 hierarchy-aware artifact 更新。 **[Code-Confirmed]**
5. 提供 enhanced parse recommendation 訊號。 **[Code-Confirmed]**

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

## 10. Failure Semantics Matrix

| Scenario | Current Behavior | Notes |
|---|---|---|
| `StructuredDocument` 載入含舊 `sections` | 可讀（compatibility） | 讀取相容，非新寫入來源 **[Code-Confirmed]** |
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
| root `sections[]` | Yes | No (default) | No |
| `structure_nodes[]` | Yes | No (default) | No |
| sections-only payload migration | Yes | Converted to hierarchy on save path | No |
| root artifact mirror | N/A | No | No |

說明：`find_*_effective(...allow_legacy_fallback=...)` 部分 helper 仍保留 compatibility 分支。 **[Code-Confirmed] + [Needs Confirmation]**

## 13. Terminology Governance Audit

| Term | Current Meaning in This Module | Classification | Governance Decision |
|---|---|---|---|
| root `sections[]` | legacy compatibility input field on `StructuredDocument`; not default write output | **[Code-Confirmed]** | 不得描述為 primary persistence source |
| top-level sections | 若指 root `sections[]`，僅 compatibility 語境可用；正式契約應改稱 `chapters[].sections[]` hierarchy | **[Doc-Confirmed] + [Code-Confirmed]** | 在 module 文檔中避免作 current contract 用語 |
| `structure_nodes[]` | legacy experimental hierarchy field，可讀；預設不寫 | **[Code-Confirmed]** | 僅保留 old JSON / round-trip compatibility 語義 |
| `StructuredDocumentNode` | compatibility type，非主流程 hierarchy source | **[Code-Confirmed]** | 不得在架構圖描述為 active main flow |
| flat `task_units` | 非 persistence truth；task units 應掛載於 section (`chapters[].sections[].task_units[]`) | **[Code-Confirmed] + [From HLD]** | 禁止作新 write contract |
| mirror | 若指 root legacy mirrors（`sections[]`/`structure_nodes[]`）僅 compatibility | **[Inferred] + [Needs Confirmation]** | 禁止重新引入 mirror 作主流程來源 |
| legacy | 指可讀相容，不代表 runtime primary path | **[Doc-Confirmed] + [Code-Confirmed]** | 文檔必須顯式區分 compatibility vs primary contract |

### Terminology Validation Notes

1. 本 module 文檔現已將 `root sections[]` 與 `structure_nodes[]` 固定為 compatibility 語義，不再暗示主流程來源。  
2. task-layout 與 diagnostics ownership 已明確放在 module boundary 外，避免責任漂移。  
3. metadata / LLM classification 被明確標記為 advisory，非 parser authority。  
4. `find_*_effective(...allow_legacy_fallback=...)` 是否最終完全退場仍屬 **[Needs Confirmation]**。  

## 14. Artifact Governance Boundary

### 14.1 Hierarchy Truth vs Artifact Output

1. hierarchy truth source 固定為 `chapters[].sections[].task_units[]`；artifact payload 不得成為 hierarchy truth source。 **[Code-Confirmed] + [From HLD]**  
2. section/task-unit artifact 僅作 interaction output，依既有 hierarchy target 更新，不可反向創建/重命名 hierarchy identity。 **[Code-Confirmed]**  
3. chapter summary/quiz artifact 目前持久化於 `document_task_artifacts.chapter_artifacts`（id-first key + legacy key candidate），屬內容輸出索引，不是 hierarchy identity source。 **[Code-Confirmed]**

### 14.2 Persistence Ownership

1. `document_structure` 擁有 artifact persistence primitives（repository contract + atomic save + hierarchy consistency guard）。 **[Code-Confirmed]**  
2. `document_structure` 不擁有 artifact availability projection contract；availability/validity 展示屬 coordinator + task-layout DTO。 **[Code-Confirmed] + [From HLD]**  
3. `document_structure` 不擁有 profile diagnostics projection contract。 **[Code-Confirmed] + [From HLD]**

### 14.3 Runtime Projection Boundary

1. availability（`has_summary/has_quiz/cache_valid`）是 runtime projection concern，不是 persisted hierarchy truth。 **[Code-Confirmed]**  
2. task-layout 讀路徑應消費 repository 已持久化 artifact 與 hierarchy，並做當次可用性判斷；該投影邏輯不在本 module。 **[Code-Confirmed] + [Inferred]**

### 14.4 Compatibility Boundary

1. legacy 可讀/migration 可保留，但不得重新引入 root sections mirror 或 flat task_units 作新 artifact write contract。 **[Code-Confirmed] + [From HLD]**  
2. `structure_nodes` compatibility 不延伸到 artifact truth contract。 **[Code-Confirmed]**

## 15. Current Risks

1. risk：helper 層 legacy fallback 若被 runtime 誤用
- why：會破壞 hierarchy-only 路徑一致性
- guardrail：維持 fail-fast regression + 明確 allow_legacy_fallback 使用邊界

2. risk：common/llm parser mode decision 無統一契約
- why：可能出現 recommendation 與實際行為落差
- guardrail：補 recommendation decision contract 文檔

3. risk：artifact 寫入若偏離 hierarchy source
- why：可能再度形成 dual representation drift
- guardrail：維持 repository hierarchy-aware write tests

4. risk：Part->Chapter 非目標若未清晰文件化
- why：後續開發者可能重引入多層 persistence
- guardrail：在 module docs/non-goals 明確固定 current scope **[From Proposal] + [From HLD]**

## 16. Open Questions for Maintainer

1. `find_*_effective` 的 legacy fallback 參數是否設定退場時間表？
2. `llm_section_splitter` 的輸出契約是否要加入更明確 schema guard（僅文檔層）？
3. enhanced recommendation 的 score threshold 調整責任層級在哪（config 或 evaluator 固化）？
4. chapter summary/quiz artifact 長期是否固定以 `document_task_artifacts.chapter_artifacts` 為唯一寫入權威（chapter node `task_artifacts` 僅作投影輔助）？
5. `root artifact mirror` 是否需在全專案文檔定義為明確 non-goal 詞彙？

## 17. Suggested Next Documentation Improvements

1. 增加「common vs llm enhanced parser lifecycle」sequence diagram。
2. 增加 artifact repository write boundary 狀態圖（section/chapter/task-unit/document-level）。
3. 補 `document_structure_language_registry` 的 consumer matrix。
