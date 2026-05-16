# section_tasks Detailed Design

## 1. Module Purpose

`section_tasks/` 定義 section/chapter/task-unit 任務層：
- task unit resolve/split
- task layout DTO/projection model
- summary/quiz 任務 prompt/service
- cache validity 相關輔助

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
| task-unit resolve/split | compute runtime/refresh units | Depends on caller path |
| section/chapter summary/quiz service | generate content payload | No direct store write |
| artifact persistence | handled by coordinator + repository | Yes |

補充：`/documents/task-layout` 是 projection/read path，不應 hidden mutation。 **[From HLD] + [Code-Confirmed]**

## 8. Target Resolution Priority

1. chapter-level target：`chapter_id` 優先。 **[Code-Confirmed]**
2. `chapter_title` 只作相容路徑；ambiguous 時應 fail-fast。 **[Code-Confirmed]**
3. section-level target：以 hierarchy section id 為核心。 **[Code-Confirmed]**

## 9. Module Relationships

- depends on:
  - `document_structure/`
  - `profile/`（作為 prompt/diagnostics 輸入）
  - `llm/`
  - `shared/`
- used by:
  - `app/section_task_coordinator.py`
- projection relationship:
  - `document_task_layout.py` 是 projection DTO contract
- advisory relationship:
  - consume profile/document metadata，但非 parser authority

## 10. Main Flows Involving This Module

1. task-unit resolution flow（split mode dependent）。 **[Code-Confirmed]**
2. section/chapter summary flow。 **[Code-Confirmed]**
3. section/chapter quiz flow。 **[Code-Confirmed]**
4. task-layout projection data contract flow（供 coordinator/API mapping）。 **[Code-Confirmed]**
5. artifact validity flow（cache_valid + invalid_reason）。 **[Code-Confirmed]**

## 11. Persistence / Side Effects

- read persistence：間接（由 coordinator/repository 提供 document/artifacts）
- write persistence：否（本 package service 層本身通常不直接落盤）
- mutate structured document：否（主要由 repository/coordinator）
- generate runtime projection：是（DTO contract）
- call LLM：是（summary/quiz/some split resolvers）
- diagnostics only：部分（artifact validity / DTO）

## 12. Known Legacy / Compatibility Behavior

1. `DocumentTaskLayout` 仍保留內部 transitional top-level fields（sections/task_units/chapter_artifacts）供 backward-compatible 測試/mapper。 **[Code-Confirmed]**
2. public API 已 chapters-first，不應依賴 transitional fields。 **[Code-Confirmed]**
3. `section_task_context_builder` 已 hierarchy-only，不再默認 legacy root sections fallback。 **[Code-Confirmed]**

## 13. Current Risks

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

## 14. Open Questions for Maintainer

1. transitional DTO fields 何時可正式標記為 deprecated-for-removal？
2. summary/quiz prompt builder contract 是否需要獨立文檔（非實作細節）？
3. task-unit split mode 對外可觀測性是否需統一 reason code？

## 15. Suggested Next Documentation Improvements

1. 增加 task-unit split mode behavior matrix。
2. 增加 task-layout DTO public/internal field boundary表。
3. 補 artifact validity reason taxonomy 文檔。
