# Deep Reflective Reader Progress

## 1. Purpose

This file aggregates module-level checklists into a global progress view.

It is used to:
- coordinate multi-agent work
- preserve completed implementation memory
- reduce hallucination
- avoid repeated full-context scans
- provide a clean entry point for future task planning

## 2. Source Documents

- `proposal.md`
- `high-level-design.md`
- `docs/modules/index.md`
- all module-level checklist files

## 3. Progress Rules

- Module checklist files are the source of truth for module-level task status.
- `progress.md` is an aggregate view, not the source of truth.
- Every future coding task must be added to the relevant module checklist first.
- When a task is completed, the module checklist must be checked.
- After module checklist changes, `progress.md` must be regenerated or updated.
- Uncertain work must remain under Needs Confirmation, not Completed.
- No task is globally complete unless the relevant module checklist reflects it.

## 4. Global Module Progress Summary

| Module | Type | Checklist | Completed Items | Needs Confirmation Items | Progress Status |
|---|---|---|---:|---:|---|
| `app/` | package | `app/module-checklist.md` | 3 | 0 | Completed Baseline Captured |
| `auth/` | package | `auth/module-checklist.md` | 3 | 0 | Completed Baseline Captured |
| `config/` | package | `config/module-checklist.md` | 3 | 0 | Completed Baseline Captured |
| `context/` | package | `context/module-checklist.md` | 3 | 0 | Completed Baseline Captured |
| `doc_loaders/` | package | `doc_loaders/module-checklist.md` | 3 | 0 | Completed Baseline Captured |
| `document_preparation/` | package | `document_preparation/module-checklist.md` | 3 | 0 | Completed Baseline Captured |
| `document_structure/` | package | `document_structure/module-checklist.md` | 6 | 2 | Needs Confirmation |
| `embeddings/` | package | `embeddings/module-checklist.md` | 3 | 0 | Completed Baseline Captured |
| `evaluated_answer/` | package | `evaluated_answer/module-checklist.md` | 3 | 0 | Completed Baseline Captured |
| `language/` | package | `language/module-checklist.md` | 3 | 0 | Completed Baseline Captured |
| `llm/` | package | `llm/module-checklist.md` | 3 | 0 | Completed Baseline Captured |
| `profile/` | package | `profile/module-checklist.md` | 5 | 0 | Completed Baseline Captured |
| `prompts/` | package | `prompts/module-checklist.md` | 3 | 0 | Completed Baseline Captured |
| `question/` | package | `question/module-checklist.md` | 3 | 0 | Completed Baseline Captured |
| `retrieval/` | package | `retrieval/module-checklist.md` | 3 | 0 | Completed Baseline Captured |
| `scripts/` | package | `scripts/module-checklist.md` | 3 | 0 | Completed Baseline Captured |
| `section_tasks/` | package | `section_tasks/module-checklist.md` | 5 | 1 | Needs Confirmation |
| `session/` | package | `session/module-checklist.md` | 3 | 0 | Completed Baseline Captured |
| `shared/` | package | `shared/module-checklist.md` | 3 | 0 | Completed Baseline Captured |
| `api_schemas.py` | root-python-module | `api_schemas.module-checklist.md` | 3 | 0 | Completed Baseline Captured |
| `bundle_factory.py` | root-python-module | `bundle_factory.module-checklist.md` | 3 | 0 | Completed Baseline Captured |
| `bundle_provider.py` | root-python-module | `bundle_provider.module-checklist.md` | 3 | 0 | Completed Baseline Captured |
| `fingerprint_handler.py` | root-python-module | `fingerprint_handler.module-checklist.md` | 3 | 0 | Completed Baseline Captured |
| `main.py` | root-python-module | `main.module-checklist.md` | 3 | 0 | Completed Baseline Captured |

## 5. Package Module Progress

### `app/`

- Checklist: `app/module-checklist.md`
- Detailed Design: `app/module-detailed-design.md`
- Status: `Completed Baseline Captured`
- Completed item count: `3`
- Needs confirmation count: `0`

#### Completed Work

- [x] Implements QA orchestration via `QACoordinator` across prepare, retrieval, prompt, and session update paths.
- [x] Implements section/chapter task orchestration via `SectionTaskCoordinator`, including task-layout projection assembly.
- [x] Maintains hierarchy-required fail-fast behavior for incompatible runtime structure states.

#### Needs Confirmation

No unresolved confirmation items identified in module checklist.

### `auth/`

- Checklist: `auth/module-checklist.md`
- Detailed Design: `auth/module-detailed-design.md`
- Status: `Completed Baseline Captured`
- Completed item count: `3`
- Needs confirmation count: `0`

#### Completed Work

- [x] Defines abstract API key contract via `APIKeyProvider`.
- [x] Provides environment-backed OpenAI API key loader via `OpenAIAPIKeyProvider`.
- [x] Uses fail-fast initialization when API key is missing.

#### Needs Confirmation

No unresolved confirmation items identified in module checklist.

### `config/`

- Checklist: `config/module-checklist.md`
- Detailed Design: `config/module-detailed-design.md`
- Status: `Completed Baseline Captured`
- Completed item count: `3`
- Needs confirmation count: `0`

#### Completed Work

- [x] Defines grouped runtime policy dataclasses in `AppDIConfig` and related config types.
- [x] Assembles core dependencies through `ApplicationLookupContainer`.
- [x] Implements namespace normalization and legacy namespace/file migration for artifact storage configs.

#### Needs Confirmation

No unresolved confirmation items identified in module checklist.

### `context/`

- Checklist: `context/module-checklist.md`
- Detailed Design: `context/module-detailed-design.md`
- Status: `Completed Baseline Captured`
- Completed item count: `3`
- Needs confirmation count: `0`

#### Completed Work

- [x] Implements context-mode orchestration for local window, retrieval, and full-text paths.
- [x] Builds ordered context chunks with budget controls via `DocumentContextBuilder`.
- [x] Provides prompt-aware token budgeting and truncation utilities through `TokenBudgetManager`.

#### Needs Confirmation

No unresolved confirmation items identified in module checklist.

### `doc_loaders/`

- Checklist: `doc_loaders/module-checklist.md`
- Detailed Design: `doc_loaders/module-detailed-design.md`
- Status: `Completed Baseline Captured`
- Completed item count: `3`
- Needs confirmation count: `0`

#### Completed Work

- [x] Defines loader abstraction via `AbstractDocumentLoader.load(doc_name) -> str`.
- [x] Implements TXT loader and PDF loader for canonical raw text extraction.
- [x] Implements loader selection through `DocumentLoaderFactory` with extension/path heuristics and historical TXT default.

#### Needs Confirmation

No unresolved confirmation items identified in module checklist.

### `document_preparation/`

- Checklist: `document_preparation/module-checklist.md`
- Detailed Design: `document_preparation/module-detailed-design.md`
- Status: `Completed Baseline Captured`
- Completed item count: `3`
- Needs confirmation count: `0`

#### Completed Work

- [x] Implements ordered prepare pipeline with profile-before-structured sequencing.
- [x] Supports `base` and `free_qa` preparation modes with explicit mode contract.
- [x] Collects non-blocking profile/enrichment errors while preserving structured readiness semantics.

#### Needs Confirmation

No unresolved confirmation items identified in module checklist.

### `document_structure/`

- Checklist: `document_structure/module-checklist.md`
- Detailed Design: `document_structure/module-detailed-design.md`
- Status: `Needs Confirmation`
- Completed item count: `6`
- Needs confirmation count: `2`

#### Completed Work

- [x] Implements hierarchy-first structured model contracts (`StructuredDocument`, chapter, section) with pure-hierarchy write defaults.
- [x] Implements hierarchy-first effective indexing helpers and section lookup paths.
- [x] Implements hierarchy-aware artifact repository with compatibility migration support on load paths.
- [x] Completed document governance cleanup for hierarchy-first persistence terminology and legacy wording boundary separation.
- [x] Synchronized unresolved confirmation status after governance cleanup across detailed-design/checklist/progress.
- [x] Clarified artifact governance and hierarchy persistence boundary (truth/output/projection ownership split).

#### Needs Confirmation

- [ ] `find_*_effective(...allow_legacy_fallback=...)` compatibility 分支是否有正式退場時程。
- [ ] `mirror` 用詞是否在 document_structure 文檔中全面替換為 `legacy compatibility fields`。

### `embeddings/`

- Checklist: `embeddings/module-checklist.md`
- Detailed Design: `embeddings/module-detailed-design.md`
- Status: `Completed Baseline Captured`
- Completed item count: `3`
- Needs confirmation count: `0`

#### Completed Work

- [x] Defines embedding backend interface (`Embedder`) with single/batch/dimension probes.
- [x] Implements OpenAI embedding provider integration via `OpenAIEmbedder`.
- [x] Implements vector normalization and nearest-similarity utility service.

#### Needs Confirmation

No unresolved confirmation items identified in module checklist.

### `evaluated_answer/`

- Checklist: `evaluated_answer/module-checklist.md`
- Detailed Design: `evaluated_answer/module-detailed-design.md`
- Status: `Completed Baseline Captured`
- Completed item count: `3`
- Needs confirmation count: `0`

#### Completed Work

- [x] Defines answer strictness result DTO via `AnswerMode`.
- [x] Implements retrieval-score to answer-mode mapping in `QuestionRelevanceEvaluator`.
- [x] Handles empty retrieval result as explicit reject mode.

#### Needs Confirmation

No unresolved confirmation items identified in module checklist.

### `language/`

- Checklist: `language/module-checklist.md`
- Detailed Design: `language/module-detailed-design.md`
- Status: `Completed Baseline Captured`
- Completed item count: `3`
- Needs confirmation count: `0`

#### Completed Work

- [x] Defines canonical language code enum and resolver/inference utilities.
- [x] Implements document language detection with profile/records reuse and LLM fallback.
- [x] Implements script/discourse/profile registries for language-scoped heuristics and cues.

#### Needs Confirmation

No unresolved confirmation items identified in module checklist.

### `llm/`

- Checklist: `llm/module-checklist.md`
- Detailed Design: `llm/module-detailed-design.md`
- Status: `Completed Baseline Captured`
- Completed item count: `3`
- Needs confirmation count: `0`

#### Completed Work

- [x] Defines LLM provider abstraction contract for completion and capability reporting.
- [x] Implements OpenAI-backed provider with model-capability mapping and endpoint routing.
- [x] Implements capability-aware prompt text normalization fallback helper.

#### Needs Confirmation

No unresolved confirmation items identified in module checklist.

### `profile/`

- Checklist: `profile/module-checklist.md`
- Detailed Design: `profile/module-detailed-design.md`
- Status: `Completed Baseline Captured`
- Completed item count: `5`
- Needs confirmation count: `0`

#### Completed Work

- [x] Defines `DocumentProfile` contract with `parser_metadata` and `post_structure_metadata` plus legacy compatibility fields.
- [x] Builds pre-structure profile via deterministic extraction plus lightweight LLM classification/fallback.
- [x] Implements post-structure metadata enrichment and profile persistence store operations.
- [x] Completed documentation governance cleanup for advisory-only profile semantics and diagnostics projection boundaries.
- [x] Clarified artifact governance vs profile metadata boundary and fixed snapshot/projection semantics wording.

#### Needs Confirmation

No unresolved confirmation items identified in module checklist.

### `prompts/`

- Checklist: `prompts/module-checklist.md`
- Detailed Design: `prompts/module-detailed-design.md`
- Status: `Completed Baseline Captured`
- Completed item count: `3`
- Needs confirmation count: `0`

#### Completed Work

- [x] Implements profile rendering block for answer prompts.
- [x] Implements answer-rule rendering by `AnswerMode` strictness levels.
- [x] Implements mode-specific guidance for local reading, retrieval, and full-text prompts.

#### Needs Confirmation

No unresolved confirmation items identified in module checklist.

### `question/`

- Checklist: `question/module-checklist.md`
- Detailed Design: `question/module-detailed-design.md`
- Status: `Completed Baseline Captured`
- Completed item count: `3`
- Needs confirmation count: `0`

#### Completed Work

- [x] Defines query-related enums and standardized question contract.
- [x] Implements LLM-based question standardization with strict JSON parsing and language normalization.
- [x] Implements scope resolution using lexical, semantic, and optional LLM fallback paths with diagnostics.

#### Needs Confirmation

No unresolved confirmation items identified in module checklist.

### `retrieval/`

- Checklist: `retrieval/module-checklist.md`
- Detailed Design: `retrieval/module-detailed-design.md`
- Status: `Completed Baseline Captured`
- Completed item count: `3`
- Needs confirmation count: `0`

#### Completed Work

- [x] Parses raw text into node sequence with positional metadata through `NodeProvider`.
- [x] Builds FAISS bundles from parsed nodes with capability-aware token budgets.
- [x] Persists and reloads FAISS artifacts with record-schema checks and rebuild guards.

#### Needs Confirmation

No unresolved confirmation items identified in module checklist.

### `scripts/`

- Checklist: `scripts/module-checklist.md`
- Detailed Design: `scripts/module-detailed-design.md`
- Status: `Completed Baseline Captured`
- Completed item count: `3`
- Needs confirmation count: `0`

#### Completed Work

- [x] Maintains regression script suite for hierarchy, task-layout, artifact persistence, and profile metadata.
- [x] Includes real-document and REST smoke script coverage for end-to-end verification paths.
- [x] Covers profile/metadata and language registry hardening through dedicated regression scripts.

#### Needs Confirmation

No unresolved confirmation items identified in module checklist.

### `section_tasks/`

- Checklist: `section_tasks/module-checklist.md`
- Detailed Design: `section_tasks/module-detailed-design.md`
- Status: `Needs Confirmation`
- Completed item count: `5`
- Needs confirmation count: `1`

#### Completed Work

- [x] Implements chapters-first task-layout DTO contracts and diagnostics DTO types.
- [x] Implements hierarchy-first task unit resolution entry using effective hierarchy sections.
- [x] Implements section task context lookup with hierarchy-only section resolution behavior.
- [x] Completed documentation governance cleanup for hierarchy-first task-layout semantics and projection/write boundary wording.
- [x] Clarified artifact availability projection vs artifact persistence truth boundary for task-layout/read path semantics.

#### Needs Confirmation

- [ ] 是否要在 section_tasks 文檔中全面淘汰 `artifact mirror` 用語，統一改為 `transitional internal field`。

### `session/`

- Checklist: `session/module-checklist.md`
- Detailed Design: `session/module-detailed-design.md`
- Status: `Completed Baseline Captured`
- Completed item count: `3`
- Needs confirmation count: `0`

#### Completed Work

- [x] Defines in-memory reading session state model (`ReadingSession`).
- [x] Implements session lifecycle operations (create, hit, reset) in `SessionManager`.
- [x] Implements session update from retrieval results to maintain active chunk continuity.

#### Needs Confirmation

No unresolved confirmation items identified in module checklist.

### `shared/`

- Checklist: `shared/module-checklist.md`
- Detailed Design: `shared/module-detailed-design.md`
- Status: `Completed Baseline Captured`
- Completed item count: `3`
- Needs confirmation count: `0`

#### Completed Work

- [x] Defines cross-module `TaskUnit` contract including parent identity and artifact payload fields.
- [x] Defines summary/quiz artifact schemas and document-level artifact container models.
- [x] Defines generic abstract result contract for service execution outputs.

#### Needs Confirmation

No unresolved confirmation items identified in module checklist.

## 6. Root Python Module Progress

### `api_schemas.py`

- Checklist: `api_schemas.module-checklist.md`
- Detailed Design: `api_schemas.module-detailed-design.md`
- Status: `Completed Baseline Captured`
- Completed item count: `3`
- Needs confirmation count: `0`

#### Completed Work

- [x] Defines external API schemas used by request and response boundaries.
- [x] Defines task-layout response contract with chapters-first projection fields and diagnostics response model.
- [x] Defines chapter summary/quiz request validation boundary for id/title target fields.

#### Needs Confirmation

No unresolved confirmation items identified in module checklist.

### `bundle_factory.py`

- Checklist: `bundle_factory.module-checklist.md`
- Detailed Design: `bundle_factory.module-detailed-design.md`
- Status: `Completed Baseline Captured`
- Completed item count: `3`
- Needs confirmation count: `0`

#### Completed Work

- [x] Implements bundle cache lifecycle (put, evict, invalidate, clear) for document-scoped runtime bundles.
- [x] Implements profile readiness logic with existing-load and rebuild fallback.
- [x] Implements index readiness with fingerprint matching and legacy-record-schema rebuild guard.

#### Needs Confirmation

No unresolved confirmation items identified in module checklist.

### `bundle_provider.py`

- Checklist: `bundle_provider.module-checklist.md`
- Detailed Design: `bundle_provider.module-detailed-design.md`
- Status: `Completed Baseline Captured`
- Completed item count: `3`
- Needs confirmation count: `0`

#### Completed Work

- [x] Implements runtime object assembly (`FaissStorageConfig`, `FingerprintHandler`, `BundleFactory`) for bundle requests.
- [x] Implements raw document loading path before bundle ensure-index flow.
- [x] Implements force-rebuild invalidation behavior before index readiness execution.

#### Needs Confirmation

No unresolved confirmation items identified in module checklist.

### `fingerprint_handler.py`

- Checklist: `fingerprint_handler.module-checklist.md`
- Detailed Design: `fingerprint_handler.module-detailed-design.md`
- Status: `Completed Baseline Captured`
- Completed item count: `3`
- Needs confirmation count: `0`

#### Completed Work

- [x] Implements stable text-hash generation and fingerprint payload construction.
- [x] Implements persisted fingerprint save/load/exists/clear file operations.
- [x] Implements current-vs-stored fingerprint matching used for cache reuse decisions.

#### Needs Confirmation

No unresolved confirmation items identified in module checklist.

### `main.py`

- Checklist: `main.module-checklist.md`
- Detailed Design: `main.module-detailed-design.md`
- Status: `Completed Baseline Captured`
- Completed item count: `3`
- Needs confirmation count: `0`

#### Completed Work

- [x] Defines FastAPI entrypoint and route registration for prepare/ask/task-layout/summary/quiz/reparse endpoints.
- [x] Maps API schemas to coordinator execution paths and response payload construction.
- [x] Implements explicit projection/mutation route boundary including manual reparse endpoint.

#### Needs Confirmation

No unresolved confirmation items identified in module checklist.

## 7. Cross-Module Completed Capabilities

### 7.1 Hierarchy and Persistence

- [x] Hierarchy-first structured document and chapter/section persistence baseline captured.
  Modules: `document_structure`, `document_preparation`, `section_tasks`
- [x] Hierarchy-aware artifact write boundary captured in design/checklists.
  Modules: `document_structure`, `app`, `section_tasks`, `shared`
- [x] Legacy compatibility exists as controlled read/migration behavior rather than new default writes.
  Modules: `document_structure`, `config`, `document_preparation`

### 7.2 Profile and Metadata

- [x] Parser metadata extraction and profile build pipeline baseline captured.
  Modules: `profile`, `language`, `llm`, `document_preparation`
- [x] Post-structure metadata enrichment path captured in prepare lifecycle.
  Modules: `profile`, `document_preparation`, `app`
- [x] Metadata advisory boundary (non-authority) documented across layers.
  Modules: `profile`, `document_structure`, `app`, `main.py`
- [x] Artifact governance boundary documented: profile metadata is advisory and not artifact persistence/availability authority.
  Modules: `profile`, `section_tasks`, `document_structure`, `app`

### 7.3 Task Layout and Artifacts

- [x] Task-layout chapters-first projection and diagnostics baseline captured.
  Modules: `section_tasks`, `app`, `api_schemas.py`, `main.py`
- [x] Task unit resolution and split orchestration baseline captured.
  Modules: `section_tasks`, `document_structure`, `app`
- [x] Summary/quiz generation and artifact persistence path captured.
  Modules: `section_tasks`, `app`, `document_structure`, `shared`
- [x] Artifact availability projection boundary documented as runtime/read-side observability (not persistence truth source).
  Modules: `section_tasks`, `app`, `document_structure`, `api_schemas.py`

### 7.4 API and Application Entry

- [x] API schema boundary and route dispatch entrypoint captured.
  Modules: `api_schemas.py`, `main.py`, `app`
- [x] Manual reparse route and non-auto-switch policy captured in current docs/checklists.
  Modules: `main.py`, `app`, `document_structure`

### 7.5 Support Modules

- [x] Language/script/discourse registries and language detection support baseline captured.
  Modules: `language`, `profile`, `question`
- [x] LLM wrapper and model capability normalization baseline captured.
  Modules: `llm`, `prompts`, `profile`, `question`
- [x] Retrieval/embedding/session/runtime bundle support baseline captured.
  Modules: `retrieval`, `embeddings`, `session`, `bundle_factory.py`, `bundle_provider.py`, `fingerprint_handler.py`

## 8. Cross-Module Needs Confirmation

| Module | Item | Reason | Needed Confirmation |
|---|---|---|---|
| `document_structure/` | `allow_legacy_fallback` 兼容分支是否有正式退場時程 | detailed design 已保留 `Needs Confirmation`，程式碼仍保留可選 fallback 參數 | maintainer 是否提供 deprecation phase/date |
| `document_structure/` | `mirror` 用詞是否全面替換為更精準術語 | terminology audit 仍標註 `Inferred + Needs Confirmation` | maintainer 是否啟動全局術語替換策略 |
| `section_tasks/` | 是否要全面淘汰 `artifact mirror` 用語 | 目前已標註為歷史語境，但全專案術語未完全統一 | maintainer 是否要啟動全局術語替換策略 |

## 9. Missing or Weak Checklists

- No checklist files are missing in this pass.
- No module checklist has insufficient completed item count in this pass.
- No completed checklist evidence formatting gaps were detected in this pass.

## 10. Multi-Agent Coordination Policy

- Main agent reads `progress.md` first.
- Module agent reads its own `module-detailed-design.md` and `module-checklist.md`.
- New task must be localized to one or a few modules.
- Before implementation, add unchecked task to the relevant module checklist.
- After implementation, update module checklist to checked.
- Then update global `progress.md`.
- If task spans modules, update all affected module checklists.

## 11. Next Update Policy

`progress.md` should be updated whenever:
- a module checklist is added
- a module checklist item is completed
- a new task is added to a module checklist
- a module detailed design document is materially changed
- a new module is added
- a module is deprecated

## 12. Current Global Status

- Documentation baseline has been captured across all listed package and root modules via checklist files.
- Most aggregated module checklists report completed baseline items; `document_structure/` and `section_tasks/` currently retain terminology/fallback governance confirmation items.
- Global status currently reflects checklist aggregation, not roadmap completion.
- Hierarchy-first and pure-hierarchy persistence direction is consistently represented across structure, preparation, and task modules.
- Profile metadata and post-structure enrichment boundaries are captured as advisory signals, not parser authority.
- Task-layout projection boundary is documented as read/projection-focused in current docs/checklists.
- API entry/schema and coordinator orchestration baselines are captured and linked in module-level documentation.
- Support modules (language, llm, retrieval, embedding, session, shared, bundle utilities) all have checklist memory established.
