# Module Detailed Design Index

## Purpose

本目錄用於從**現有實作**反向整理 Deep_Reflective_Reader 的 module-level design，作為維護、交接、與後續演進的技術地圖。  
既有 inventory 描述現況；經 maintainer 明確確認的未實作模組另列 Planned Modules，不混列成已存在的 runtime package。

Global Progress: [progress.md](../../progress.md)

## Source of Truth

1. `proposal.md`
2. `high-level-design.md`
3. 實際 codebase 掃描結果（package 與 root python modules）

## Documentation Rules

- **[Code-Confirmed]**：從 code 直接確認
- **[From Proposal]**：來自 `proposal.md`
- **[From HLD]**：來自 `high-level-design.md`
- **[Inferred]**：根據 code relation 推斷
- **[Needs Confirmation]**：需要 maintainer 回答
- **[Planned]**：maintainer 已確認規劃方向，但尚未實作；不能當成 runtime 能力

## Module Inventory

| Module | Type | Role Summary | Detailed Design Doc | Checklist | Status |
|---|---|---|---|---|---|
| `app/` | package | QA/task orchestration and runtime coordination | [app/module-detailed-design.md](../../app/module-detailed-design.md) | [app/module-checklist.md](../../app/module-checklist.md) | Drafted |
| `auth/` | package | API key/provider abstraction | [auth/module-detailed-design.md](../../auth/module-detailed-design.md) | [auth/module-checklist.md](../../auth/module-checklist.md) | Drafted |
| `config/` | package | DI assembly and runtime config contracts | [config/module-detailed-design.md](../../config/module-detailed-design.md) | [config/module-checklist.md](../../config/module-checklist.md) | Drafted |
| `context/` | package | QA context composition/orchestration | [context/module-detailed-design.md](../../context/module-detailed-design.md) | [context/module-checklist.md](../../context/module-checklist.md) | Drafted |
| `doc_loaders/` | package | raw document loader selection (txt/pdf) | [doc_loaders/module-detailed-design.md](../../doc_loaders/module-detailed-design.md) | [doc_loaders/module-checklist.md](../../doc_loaders/module-checklist.md) | Drafted |
| `db/` | package + design module | DB identity/versioning/lifecycle design, PostgreSQL store adapters and explicit migrations, preference and learning SQL persistence | [db/module-detailed-design.md](../../db/module-detailed-design.md) | [db/module-checklist.md](../../db/module-checklist.md) | Drafted |
| `document_preparation/` | package | preparation pipeline and readiness snapshot | [document_preparation/module-detailed-design.md](../../document_preparation/module-detailed-design.md) | [document_preparation/module-checklist.md](../../document_preparation/module-checklist.md) | Drafted |
| `document_structure/` | package | hierarchy model, parser split, structure persistence contracts, shared sentence translation/learning repository behavior/file adapter | [document_structure/module-detailed-design.md](../../document_structure/module-detailed-design.md) | [document_structure/module-checklist.md](../../document_structure/module-checklist.md) | Drafted |
| `embeddings/` | package | embedding provider and similarity utilities | [embeddings/module-detailed-design.md](../../embeddings/module-detailed-design.md) | [embeddings/module-checklist.md](../../embeddings/module-checklist.md) | Drafted |
| `evaluated_answer/` | package | QA answer mode/relevance model | [evaluated_answer/module-detailed-design.md](../../evaluated_answer/module-detailed-design.md) | [evaluated_answer/module-checklist.md](../../evaluated_answer/module-checklist.md) | Drafted |
| `language/` | package | language codes/registries, proficiency definitions and learning sentence tailoring | [language/module-detailed-design.md](../../language/module-detailed-design.md) | [language/module-checklist.md](../../language/module-checklist.md) | Drafted |
| `learning/` | application package | Learning orchestration, worker and completion observation; extracted from app | [learning/module-detailed-design.md](../../learning/module-detailed-design.md) | [learning/module-checklist.md](../../learning/module-checklist.md) | Extracted and locally verified (2026-11-10); external review/deployment pending |
| `llm/` | package | LLM provider abstraction and capabilities | [llm/module-detailed-design.md](../../llm/module-detailed-design.md) | [llm/module-checklist.md](../../llm/module-checklist.md) | Drafted |
| `profile/` | package | profile building, parser metadata, post-structure metadata | [profile/module-detailed-design.md](../../profile/module-detailed-design.md) | [profile/module-checklist.md](../../profile/module-checklist.md) | Drafted |
| `prompts/` | package | prompt assembly for QA response generation | [prompts/module-detailed-design.md](../../prompts/module-detailed-design.md) | [prompts/module-checklist.md](../../prompts/module-checklist.md) | Drafted |
| `question/` | package | question standardization and scope resolution | [question/module-detailed-design.md](../../question/module-detailed-design.md) | [question/module-checklist.md](../../question/module-checklist.md) | Drafted |
| `reading_preferences/` | package | workspace 母語、book mode/source/proficiency contracts、catalog、file adapter、effective settings service；未來 user migration 邊界 | [design](../../reading_preferences/module-detailed-design.md) | [tasks](../../reading_preferences/module-checklist.md) | LLRN-02/09 foundation/service implemented; SQL adapter owned by db (LLRN-07 complete); runtime DI implemented (LLRN-15); production API/lifespan implemented (LLRN-16); settings UI implemented (LLRN-17) |
| `retrieval/` | package | FAISS bundle/index build/load/search metadata | [retrieval/module-detailed-design.md](../../retrieval/module-detailed-design.md) | [retrieval/module-checklist.md](../../retrieval/module-checklist.md) | Drafted |
| `scripts/` | package-like test folder | regression and smoke test scripts | [scripts/module-detailed-design.md](../../scripts/module-detailed-design.md) | [scripts/module-checklist.md](../../scripts/module-checklist.md) | Drafted |
| `section_tasks/` | package | task layout projection, task unit resolution, summary/quiz flow, sentence segmentation, shared ordinary/learning aligned translation and cache/generation policy, ability-aware hints services | [section_tasks/module-detailed-design.md](../../section_tasks/module-detailed-design.md) | [section_tasks/module-checklist.md](../../section_tasks/module-checklist.md) | Drafted |
| `session/` | package | reading session lifecycle/state | [session/module-detailed-design.md](../../session/module-detailed-design.md) | [session/module-checklist.md](../../session/module-checklist.md) | Drafted |
| `shared/` | package | shared task/artifact models and source-bound sentence/learning DTOs | [shared/module-detailed-design.md](../../shared/module-detailed-design.md) | [shared/module-checklist.md](../../shared/module-checklist.md) | Drafted |
| `api_schemas.py` | root-python-module | REST request/response contracts; learning DTO/projection helpers remain under this owner | [api_schemas.module-detailed-design.md](../../api_schemas.module-detailed-design.md) | [api_schemas.module-checklist.md](../../api_schemas.module-checklist.md) | Drafted |
| `bundle_factory.py` | root-python-module | bundle cache/rebuild orchestration helper | [bundle_factory.module-detailed-design.md](../../bundle_factory.module-detailed-design.md) | [bundle_factory.module-checklist.md](../../bundle_factory.module-checklist.md) | Drafted |
| `bundle_provider.py` | root-python-module | runtime bundle provider facade | [bundle_provider.module-detailed-design.md](../../bundle_provider.module-detailed-design.md) | [bundle_provider.module-checklist.md](../../bundle_provider.module-checklist.md) | Drafted |
| `fingerprint_handler.py` | root-python-module | fingerprint/hash comparison and cache reuse signals | [fingerprint_handler.module-detailed-design.md](../../fingerprint_handler.module-detailed-design.md) | [fingerprint_handler.module-checklist.md](../../fingerprint_handler.module-checklist.md) | Drafted |
| `main.py` | root-python-module | FastAPI entry and route mapping | [main.module-detailed-design.md](../../main.module-detailed-design.md) | [main.module-checklist.md](../../main.module-checklist.md) | Drafted |

## Language Learning Implementation Scope (2026-10-28)

`reading_preferences` 已從文檔規劃模組移入實作 inventory；package 已有 LLRN-02 foundation，db 已交付 LLRN-07 preference SQL/migration；LLRN-08 learning SQL 也已交付；LLRN-09 有效設定/eligibility 服務、LLRN-10 逐句翻譯生成及 LLRN-11 能力提示生成已交付；LLRN-12 source/document lifecycle ports 與 LLRN-13 learning orchestration/admission 已交付；LLRN-14 transport schemas/pure projections 已交付；LLRN-15 runtime composition 已交付；LLRN-16 production API/lifespan 已交付；LLRN-17 settings UI 及 LLRN-18 當頁 controller 已交付；LLRN-19 逐句呈現也已交付；LLRN-20 backend integration 與 LLRN-21 reader lifecycle/accessibility 已驗證，LLRN-22 已依 2026-10-29 maintainer 指示完成系統驗收，模型語義評分移出本任務。

語言學習跨模組唯一 [規格](../../section_tasks/module-detailed-design.md#language-learning-reading-mode-plan-2026-10-28) / [22 項任務 DAG](../../section_tasks/module-checklist.md#language-learning-reading-mode-queue-2026-10-28)。目前 LLRN-01 language proficiency、LLRN-02 preference foundation、LLRN-03 source-bound sentence/learning DTOs、LLRN-04 multilingual segmentation、LLRN-05 pure cache/generation policy、LLRN-06 hierarchy-aware learning repository/file adapter 、LLRN-07 preference SQL/migration 、LLRN-08 learning SQL/operation persistence 、LLRN-09 有效偏好服務、LLRN-10 逐句翻譯生成、LLRN-11 能力提示生成、LLRN-12 source/document lifecycle 、LLRN-13 learning orchestration/admission 、LLRN-14 learning/preference transport schemas 、LLRN-15 dependency composition 、LLRN-16 routes/lifespan 、LLRN-17 settings UI 、LLRN-18 current-page controller 、LLRN-19 sentence-pair display 、LLRN-20 backend integration acceptance 及 LLRN-21 reader lifecycle/accessibility acceptance 完成，加上 LLRN-22 系統驗收共 22/22；原始回應記錄、無損格式重播、錯誤分類及 API/SQL/Reader 證據見 scripts owner；歷史語義品質不改判為通過；UI checklist 仍單獨治理，不併入 backend progress。

## Recommended Reading Order

1. [document_structure/module-detailed-design.md](../../document_structure/module-detailed-design.md)
2. [document_preparation/module-detailed-design.md](../../document_preparation/module-detailed-design.md)
3. [profile/module-detailed-design.md](../../profile/module-detailed-design.md)
4. [section_tasks/module-detailed-design.md](../../section_tasks/module-detailed-design.md)
5. [app/module-detailed-design.md](../../app/module-detailed-design.md)
6. [api_schemas.module-detailed-design.md](../../api_schemas.module-detailed-design.md)
7. [main.module-detailed-design.md](../../main.module-detailed-design.md)
8. [config/module-detailed-design.md](../../config/module-detailed-design.md)
9. [retrieval/module-detailed-design.md](../../retrieval/module-detailed-design.md)
10. [db/module-detailed-design.md](../../db/module-detailed-design.md) for future DB-era identity/versioning/reparse and logical schema planning
11. 其餘 support/peripheral modules（`auth/`, `context/`, `language/`, `session/`, `shared/` 等）

## Open Questions

1. `scripts/` 是否視為正式 test layer 的唯一來源（目前未見 `tests/` 目錄）。 **[Code-Confirmed] + [Needs Confirmation]**
2. `language/language_profile_registry.py` 與其他 language registries 的長期邊界是否再收斂。 **[Code-Confirmed] + [Needs Confirmation]**
3. `bundle_factory.py` 與 `document_preparation/` 的責任邊界（索引快取 vs prepare pipeline）是否要文件化為固定 contract。 **[Inferred] + [Needs Confirmation]**
4. 已定案 policy（manual reparse、Part->Chapter non-goal、cache-first naming defer）是否要同步到 API-facing docs。 **[From HLD] + [Needs Confirmation]**
5. `structure_profile` 採短期保留（非當前阻塞）後，何時再進入退場評估。 **[From Proposal] + [Needs Confirmation]**
