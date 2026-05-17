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
| `shared/task_unit_model.py` | `TaskUnit` schema | parent_section_id 與 artifact nested fields **[Code-Confirmed]** |

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

## 14. Future Direction Note: Rich Task-Unit Content Model Preparation

> 本節屬未來方向說明，非當前 implementation。 **[Inferred]**

1. `TaskUnit.content` 未來可由 simple string 演進為 content block / content segment model；shared 層僅承擔 caller-neutral 的 low-level model/helper preparation。 **[Inferred]**
2. content block 的定位是 task-unit 內部的 content segmentation / evidence target，不是 chapter/section/task_unit 之外的新 hierarchy level。 **[Inferred]**
3. content block 不得成為 parser authority；artifact 掛載於 content block 也不得成為 hierarchy truth。 **[Inferred]**
4. `TaskUnit` 仍是 interaction container；hierarchy-first persistence 仍以 `chapters[].sections[].task_units[]` 為可信來源。 **[Code-Confirmed] + [Inferred]**
5. backward compatibility 方向可考慮 `string content -> single content block` adapter，但本輪不實作。 **[Inferred]**
