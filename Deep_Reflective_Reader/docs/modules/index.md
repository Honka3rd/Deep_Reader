# Module Detailed Design Index

## Purpose

本目錄用於從**現有實作**反向整理 Deep_Reflective_Reader 的 module-level design，作為維護、交接、與後續演進的技術地圖。  
本目錄不是重構提案，也不是重新設計文件。

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

## Module Inventory

| Module | Type | Role Summary | Detailed Design Doc | Checklist | Status |
|---|---|---|---|---|---|
| `app/` | package | QA/task orchestration and runtime coordination | [app/module-detailed-design.md](../../app/module-detailed-design.md) | [app/module-checklist.md](../../app/module-checklist.md) | Drafted |
| `auth/` | package | API key/provider abstraction | [auth/module-detailed-design.md](../../auth/module-detailed-design.md) | [auth/module-checklist.md](../../auth/module-checklist.md) | Drafted |
| `config/` | package | DI assembly and runtime config contracts | [config/module-detailed-design.md](../../config/module-detailed-design.md) | [config/module-checklist.md](../../config/module-checklist.md) | Drafted |
| `context/` | package | QA context composition/orchestration | [context/module-detailed-design.md](../../context/module-detailed-design.md) | [context/module-checklist.md](../../context/module-checklist.md) | Drafted |
| `doc_loaders/` | package | raw document loader selection (txt/pdf) | [doc_loaders/module-detailed-design.md](../../doc_loaders/module-detailed-design.md) | [doc_loaders/module-checklist.md](../../doc_loaders/module-checklist.md) | Drafted |
| `document_preparation/` | package | preparation pipeline and readiness snapshot | [document_preparation/module-detailed-design.md](../../document_preparation/module-detailed-design.md) | [document_preparation/module-checklist.md](../../document_preparation/module-checklist.md) | Drafted |
| `document_structure/` | package | hierarchy model, parser split, structure persistence contracts | [document_structure/module-detailed-design.md](../../document_structure/module-detailed-design.md) | [document_structure/module-checklist.md](../../document_structure/module-checklist.md) | Drafted |
| `embeddings/` | package | embedding provider and similarity utilities | [embeddings/module-detailed-design.md](../../embeddings/module-detailed-design.md) | [embeddings/module-checklist.md](../../embeddings/module-checklist.md) | Drafted |
| `evaluated_answer/` | package | QA answer mode/relevance model | [evaluated_answer/module-detailed-design.md](../../evaluated_answer/module-detailed-design.md) | [evaluated_answer/module-checklist.md](../../evaluated_answer/module-checklist.md) | Drafted |
| `language/` | package | language code + language registries | [language/module-detailed-design.md](../../language/module-detailed-design.md) | [language/module-checklist.md](../../language/module-checklist.md) | Drafted |
| `llm/` | package | LLM provider abstraction and capabilities | [llm/module-detailed-design.md](../../llm/module-detailed-design.md) | [llm/module-checklist.md](../../llm/module-checklist.md) | Drafted |
| `profile/` | package | profile building, parser metadata, post-structure metadata | [profile/module-detailed-design.md](../../profile/module-detailed-design.md) | [profile/module-checklist.md](../../profile/module-checklist.md) | Drafted |
| `prompts/` | package | prompt assembly for QA response generation | [prompts/module-detailed-design.md](../../prompts/module-detailed-design.md) | [prompts/module-checklist.md](../../prompts/module-checklist.md) | Drafted |
| `question/` | package | question standardization and scope resolution | [question/module-detailed-design.md](../../question/module-detailed-design.md) | [question/module-checklist.md](../../question/module-checklist.md) | Drafted |
| `retrieval/` | package | FAISS bundle/index build/load/search metadata | [retrieval/module-detailed-design.md](../../retrieval/module-detailed-design.md) | [retrieval/module-checklist.md](../../retrieval/module-checklist.md) | Drafted |
| `scripts/` | package-like test folder | regression and smoke test scripts | [scripts/module-detailed-design.md](../../scripts/module-detailed-design.md) | [scripts/module-checklist.md](../../scripts/module-checklist.md) | Drafted |
| `section_tasks/` | package | task layout projection, task unit resolution, summary/quiz task flow | [section_tasks/module-detailed-design.md](../../section_tasks/module-detailed-design.md) | [section_tasks/module-checklist.md](../../section_tasks/module-checklist.md) | Drafted |
| `session/` | package | reading session lifecycle/state | [session/module-detailed-design.md](../../session/module-detailed-design.md) | [session/module-checklist.md](../../session/module-checklist.md) | Drafted |
| `shared/` | package | shared task artifacts and task unit models | [shared/module-detailed-design.md](../../shared/module-detailed-design.md) | [shared/module-checklist.md](../../shared/module-checklist.md) | Drafted |
| `api_schemas.py` | root-python-module | REST request/response contracts | [api_schemas.module-detailed-design.md](../../api_schemas.module-detailed-design.md) | [api_schemas.module-checklist.md](../../api_schemas.module-checklist.md) | Drafted |
| `bundle_factory.py` | root-python-module | bundle cache/rebuild orchestration helper | [bundle_factory.module-detailed-design.md](../../bundle_factory.module-detailed-design.md) | [bundle_factory.module-checklist.md](../../bundle_factory.module-checklist.md) | Drafted |
| `bundle_provider.py` | root-python-module | runtime bundle provider facade | [bundle_provider.module-detailed-design.md](../../bundle_provider.module-detailed-design.md) | [bundle_provider.module-checklist.md](../../bundle_provider.module-checklist.md) | Drafted |
| `fingerprint_handler.py` | root-python-module | fingerprint/hash comparison and cache reuse signals | [fingerprint_handler.module-detailed-design.md](../../fingerprint_handler.module-detailed-design.md) | [fingerprint_handler.module-checklist.md](../../fingerprint_handler.module-checklist.md) | Drafted |
| `main.py` | root-python-module | FastAPI entry and route mapping | [main.module-detailed-design.md](../../main.module-detailed-design.md) | [main.module-checklist.md](../../main.module-checklist.md) | Drafted |

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
10. 其餘 support/peripheral modules（`auth/`, `context/`, `language/`, `session/`, `shared/` 等）

## Open Questions

1. `scripts/` 是否視為正式 test layer 的唯一來源（目前未見 `tests/` 目錄）。 **[Code-Confirmed] + [Needs Confirmation]**
2. `language/language_profile_registry.py` 與其他 language registries 的長期邊界是否再收斂。 **[Code-Confirmed] + [Needs Confirmation]**
3. `bundle_factory.py` 與 `document_preparation/` 的責任邊界（索引快取 vs prepare pipeline）是否要文件化為固定 contract。 **[Inferred] + [Needs Confirmation]**
4. 已定案 policy（manual reparse、Part->Chapter non-goal、cache-first naming defer）是否要同步到 API-facing docs。 **[From HLD] + [Needs Confirmation]**
5. `structure_profile` 採短期保留（非當前阻塞）後，何時再進入退場評估。 **[From Proposal] + [Needs Confirmation]**
