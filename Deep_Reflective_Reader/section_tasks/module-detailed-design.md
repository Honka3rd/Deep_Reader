# section_tasks Detailed Design

## 1. Module Purpose

`section_tasks/` 定義 section/chapter/task-unit 任務層：
- task unit resolve/split
- task layout DTO/projection model
- summary/quiz 任務 prompt/service
- cache validity 相關輔助
- artifact availability runtime projection（供 task-layout/read path 可觀測）

**[Code-Confirmed]**

## 2. Position in Overall Architecture

- Task Layout / Task Unit Layer

## 3. Key Files

| File | Responsibility | Notes |
|---|---|---|
| `section_tasks/document_task_layout.py` | task-layout DTO contract（chapters-first + diagnostics DTO） | public response 映射來源之一 **[Code-Confirmed]** |
| `section_tasks/task_unit_resolver.py` | task unit resolution 主入口 | 使用 `get_effective_sections`（hierarchy-first） **[Code-Confirmed]** |
| `section_tasks/task_unit_split_resolver_selector.py` | split mode resolver selector | semantic/progressive/llm resolvers **[Code-Confirmed]** |
| `section_tasks/heuristic_task_unit_split_resolver.py` | deterministic split path | semantic boundary scoring integration **[Code-Confirmed]** |
| `section_tasks/llm_task_unit_split_resolver.py` | llm split path | fallback to heuristic path **[Code-Confirmed]** |
| `section_tasks/task_unit_id_normalizer.py` | task unit id normalization and parent identity fill | hierarchy scope uniqueness support **[Code-Confirmed]** |
| `section_tasks/chapter_summary_service.py` | summary generation service | consume context/prompt factory + LLM **[Code-Confirmed]** |
| `section_tasks/chapter_quiz_service.py` | quiz generation service | consume context/prompt factory + LLM **[Code-Confirmed]** |
| `section_tasks/section_task_context_builder.py` | section/task unit context build | hierarchy-only lookup path **[Code-Confirmed]** |
| `section_tasks/artifact_validity.py` | artifact validity result contract | cache validity reason surface **[Code-Confirmed]** |

## 4. Main Responsibilities

1. 定義 task-layer DTO 與 task mode 表達（direct/split/merged）。 **[Code-Confirmed]**
2. 根據 document structure 解析 task units。 **[Code-Confirmed]**
3. 建立 summary/quiz prompt + context 並調用 LLM 服務。 **[Code-Confirmed]**
4. 提供 task artifact validity 判斷契約。 **[Code-Confirmed]**
5. 支援 hierarchy-first target resolution 所需 context。 **[Code-Confirmed]**

## 5. Non-Responsibilities

1. 不負責 API route mapping。 **[From HLD]**
2. 不應在 read-only task-layout endpoint 直接做 persistence write（由 coordinator/repository 管控）。 **[From HLD]**
3. 不應把 profile metadata 直接當 parser rule。 **[From HLD]**
4. 不應污染 `StructuredDocument` persistence contract（不得重新引入 root sections mirror / flat task_units 作 persisted truth）。 **[Code-Confirmed] + [From HLD]**
5. 不負責 artifact persistence truth 定義；artifact source-of-truth 與 hierarchy persistence 邊界由 `document_structure` + repository 層承載。 **[Code-Confirmed] + [From HLD]**

## 6. Important Data Structures / Contracts

- `DocumentTaskLayout`
- `DocumentTaskLayoutChapterDTO`
- `DocumentTaskLayoutSectionDTO`
- `TaskUnitDTO`
- `ProfileStructureDiagnosticsDTO`
- `TaskUnit`
- `ArtifactValidityResult`

## 7. Read/Write Boundary Matrix

| Path | Responsibility | Mutation Allowed |
|---|---|---:|
| task-layout projection | assemble chapters/sections/task_units DTO | No |
| artifact availability projection | expose has_summary/has_quiz/cache_valid/invalid_reason as runtime view | No |
| task-unit resolve/split | compute runtime/refresh units | Depends on caller path |
| section/chapter summary/quiz service | generate content payload | No direct store write |
| artifact persistence | handled by coordinator + repository (`document_structure` boundary) | Yes |

補充：`/documents/task-layout` 是 projection/read path，不應 hidden mutation。 **[From HLD] + [Code-Confirmed]**

## 8. Public API Boundary Clarification

1. `DocumentTaskLayout` 內含 `sections` / `task_units` / `chapter_artifacts` 屬 internal transitional DTO fields。 **[Code-Confirmed]**
2. `/documents/task-layout` public response 由 `main.py` 映射為 chapters-first，未返回 top-level `sections` / top-level `task_units` / top-level `chapter_artifacts`。 **[Code-Confirmed]**
3. `profile_diagnostics` 是 runtime projection；task-unit coverage 取自 current layout sections，不是 persisted profile snapshot overwrite。 **[Code-Confirmed]**
4. diagnostics projection 不應回寫 profile artifact。 **[Code-Confirmed] + [From HLD]**

## 9. Artifact Governance and Projection Boundary

1. persisted hierarchy truth 來自 `StructuredDocument` 的 hierarchy path（`chapters[].sections[].task_units[]`）；section_tasks 不定義 hierarchy truth。 **[Code-Confirmed] + [From HLD]**
2. section/chapter/task-unit artifacts 是 interaction output（summary/quiz 等），不是 hierarchy identity source。 **[Code-Confirmed] + [From HLD]**
3. section_tasks 負責的是 artifact availability 的 runtime projection，不是 artifact persistence authority。 **[Code-Confirmed]**
4. task-layout diagnostics/availability 不得寫回 profile，不得覆寫 persisted profile snapshot。 **[Code-Confirmed]**
5. API response DTO shape 是 transport contract，不等於 persistence schema。 **[Code-Confirmed]**

## 10. Target Resolution Priority

1. chapter-level target：`chapter_id` 優先。 **[Code-Confirmed]**
2. `chapter_title` 只作相容路徑；ambiguous 時應 fail-fast。 **[Code-Confirmed]**
3. section-level target：以 hierarchy section id 為核心。 **[Code-Confirmed]**

## 11. Module Relationships

- depends on:
  - `document_structure/`
  - `profile/`（作為 prompt/diagnostics 輸入）
  - `llm/`
  - `shared/`
- used by:
  - `app/section_task_coordinator.py`
- projection relationship:
  - `document_task_layout.py` 是 projection DTO contract（含 artifact availability/runtime diagnostics 投影）
- advisory relationship:
  - consume profile/document metadata，但非 parser authority

## 12. Main Flows Involving This Module

1. task-unit resolution flow（split mode dependent）。 **[Code-Confirmed]**
2. section/chapter summary flow。 **[Code-Confirmed]**
3. section/chapter quiz flow。 **[Code-Confirmed]**
4. task-layout projection data contract flow（供 coordinator/API mapping）。 **[Code-Confirmed]**
5. artifact validity flow（cache_valid + invalid_reason）。 **[Code-Confirmed]**

## 13. Persistence / Side Effects

- read persistence：間接（由 coordinator/repository 提供 document/artifacts）
- write persistence：否（本 package service 層本身通常不直接落盤）
- mutate structured document：否（主要由 repository/coordinator）
- generate runtime projection：是（DTO contract）
- call LLM：是（summary/quiz/some split resolvers）
- diagnostics only：部分（artifact validity / DTO）
- artifact availability state：runtime projection（非 persisted truth）

## 14. Known Legacy / Compatibility Behavior

1. `DocumentTaskLayout` 仍保留內部 transitional top-level fields（sections/task_units/chapter_artifacts）供 backward-compatible 測試/mapper。 **[Code-Confirmed]**
2. public API 已 chapters-first，不應依賴 transitional fields。 **[Code-Confirmed]**
3. `section_task_context_builder` 已 hierarchy-only，不再默認 legacy root sections fallback。 **[Code-Confirmed]**

## 15. Terminology Governance Audit

| Term | Current Meaning in This Module | Classification | Governance Decision |
|---|---|---|---|
| top-level sections | 僅 internal transitional DTO field (`DocumentTaskLayout.sections`)，非 public API contract | **[Code-Confirmed]** | 不得描述為 `/documents/task-layout` response source of truth |
| top-level task_units | 僅 internal transitional DTO field (`DocumentTaskLayout.task_units`) | **[Code-Confirmed]** | 不得描述為 persistence contract 或 public API main path |
| chapter_artifacts (top-level) | internal transitional map；chapter artifact availability public path 走 `chapters[].artifacts` | **[Code-Confirmed]** | 不得在文檔描述為 public response required field |
| artifact source of truth | 本模組不定義 artifact truth；僅消費 repository/hierarchy-aware 結果做可觀測投影 | **[Code-Confirmed] + [From HLD]** | 不得描述為由 task-layout DTO 決定 persisted truth |
| artifact persistence | write path 在 coordinator + repository，不在 task-layout projection DTO | **[Code-Confirmed]** | 不得把 projection path 描述為 write authority |
| artifact availability | runtime/projection state（has_summary/has_quiz/cache_valid/invalid_reason） | **[Code-Confirmed]** | 不得描述為 persistence snapshot overwrite |
| availability cache / artifact validity | 僅可觀測 cache-validity 訊號，不改 hierarchy identity | **[Code-Confirmed]** | 避免被誤讀為 hierarchy source |
| top-level legacy mirrors（deprecated wording） | 歷史/文檔語境用詞，正式術語已收斂為 `transitional internal field` / `compatibility-only field` | **[Maintainer-Confirmed] + [Doc-Confirmed]** | 僅允許 historical/deprecated/compatibility 說明，不得作現行 contract terminology |
| flat task_units | historical/compatibility語境，非 hierarchy persistence truth | **[Code-Confirmed] + [From HLD]** | 禁止重新引入 flat persisted semantics |
| artifact mirror（deprecated wording） | 僅 historical/migration 搜尋關鍵字；正式術語改為 `transitional internal field` | **[Maintainer-Confirmed] + [Doc-Confirmed]** | 不得作為現行 contract terminology、persistence authority 或 hierarchy truth 描述 |
| write-back / mutation | task-layout request path 應視為 projection/read；artifact write path 需顯式呼叫 repository update | **[Code-Confirmed] + [From HLD]** | 禁止 hidden mutation |
| profile diagnostics | runtime mixed-source projection，非 persisted profile body | **[Code-Confirmed]** | 禁止描述為 profile snapshot overwrite |
| root sections mirror | 不屬 section_tasks contract；不得在本模組文件暗示可回退為主路徑 | **[Code-Confirmed] + [From HLD]** | 維持 hierarchy-first contract 邊界 |
| API response shape | transport DTO（chapters-first） | **[Code-Confirmed]** | 不等於 persistence schema |

### Terminology Validation Notes

1. 本模組文檔已明確區分 internal transitional DTO fields 與 public chapters-first API contract。  
2. task-layout 與 diagnostics 皆定義為 runtime projection；不承擔 profile write-back。  
3. section/task-unit/chapter artifact availability 的讀取語義固定為 hierarchy-aware path（read-side observability）。  
4. `artifact mirror` 一詞已降級為 deprecated terminology；正式 contract 用語統一為 `transitional internal field`。 **[Maintainer-Confirmed]**  
5. API response shape 已與 persistence schema 語義分離，避免 DTO 漂移成 artifact truth source。  

## 16. Current Risks

1. risk：task unit split mode 行為差異造成結果波動
- why：同文檔在不同 mode 下結構變化可能大
- guardrail：mode-specific regression tests

2. risk：transitional DTO fields 被誤當 public API 契約
- why：可能重引 flat mirrors
- guardrail：在 docs/schema 明確 internal-only

3. risk：LLM split 與 heuristic split 邊界不清
- why：debug 成本高
- guardrail：維持 selector 與 fallback reason 可觀測

4. risk：task unit id normalization 若失效會破壞 artifact 對齊
- why：cache validity/target resolution 錯位
- guardrail：document-scope unique id tests + parent_section_id checks

## 17. Open Questions for Maintainer

1. transitional DTO fields 何時可正式標記為 deprecated-for-removal？
2. summary/quiz prompt builder contract 是否需要獨立文檔（非實作細節）？
3. task-unit split mode 對外可觀測性是否需統一 reason code？
4. `DocumentTaskLayout` internal transitional fields 是否需要在類註解中加 deprecation horizon（版本或階段）？

## 18. Suggested Next Documentation Improvements

1. 增加 task-unit split mode behavior matrix。
2. 增加 task-layout DTO public/internal field boundary表。
3. 補 artifact validity reason taxonomy 文檔。
