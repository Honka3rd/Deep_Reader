# shared Detailed Design

## 1. Module Purpose

`shared/` 提供跨模組共用 DTO/抽象結果契約，包含 task artifacts、task units 與通用 result 基類。 **[Code-Confirmed]**

## 2. Position in Overall Architecture

- Shared Utility Layer

## 3. Key Files

| File | Responsibility | Notes |
|---|---|---|
| `shared/abstract_result.py` | Generic result 抽象基類 | success/payload/reason/cache_hit **[Code-Confirmed]** |
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
