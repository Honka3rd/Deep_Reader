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
| `app/` | package | `app/module-checklist.md` | 7 | 0 | Completed Baseline Captured |
| `auth/` | package | `auth/module-checklist.md` | 3 | 0 | Completed Baseline Captured |
| `config/` | package | `config/module-checklist.md` | 3 | 0 | Completed Baseline Captured |
| `context/` | package | `context/module-checklist.md` | 3 | 0 | Completed Baseline Captured |
| `doc_loaders/` | package | `doc_loaders/module-checklist.md` | 3 | 0 | Completed Baseline Captured |
| `document_preparation/` | package | `document_preparation/module-checklist.md` | 3 | 0 | Completed Baseline Captured |
| `document_structure/` | package | `document_structure/module-checklist.md` | 11 | 0 | Completed Baseline Captured |
| `embeddings/` | package | `embeddings/module-checklist.md` | 3 | 0 | Completed Baseline Captured |
| `evaluated_answer/` | package | `evaluated_answer/module-checklist.md` | 3 | 0 | Completed Baseline Captured |
| `language/` | package | `language/module-checklist.md` | 3 | 0 | Completed Baseline Captured |
| `llm/` | package | `llm/module-checklist.md` | 3 | 0 | Completed Baseline Captured |
| `profile/` | package | `profile/module-checklist.md` | 5 | 0 | Completed Baseline Captured |
| `prompts/` | package | `prompts/module-checklist.md` | 3 | 0 | Completed Baseline Captured |
| `question/` | package | `question/module-checklist.md` | 3 | 0 | Completed Baseline Captured |
| `retrieval/` | package | `retrieval/module-checklist.md` | 3 | 0 | Completed Baseline Captured |
| `scripts/` | package | `scripts/module-checklist.md` | 3 | 0 | Completed Baseline Captured |
| `section_tasks/` | package | `section_tasks/module-checklist.md` | 12 | 0 | Completed Baseline Captured |
| `session/` | package | `session/module-checklist.md` | 3 | 0 | Completed Baseline Captured |
| `shared/` | package | `shared/module-checklist.md` | 9 | 0 | Completed Baseline Captured |
| `api_schemas.py` | root-python-module | `api_schemas.module-checklist.md` | 10 | 0 | Completed Baseline Captured |
| `bundle_factory.py` | root-python-module | `bundle_factory.module-checklist.md` | 3 | 0 | Completed Baseline Captured |
| `bundle_provider.py` | root-python-module | `bundle_provider.module-checklist.md` | 3 | 0 | Completed Baseline Captured |
| `fingerprint_handler.py` | root-python-module | `fingerprint_handler.module-checklist.md` | 3 | 0 | Completed Baseline Captured |
| `main.py` | root-python-module | `main.module-checklist.md` | 9 | 0 | Completed Baseline Captured |

## 5. Package Module Progress

### `app/`

- Checklist: `app/module-checklist.md`
- Detailed Design: `app/module-detailed-design.md`
- Status: `Completed Baseline Captured`
- Completed item count: `7`
- Needs confirmation count: `0`

#### Completed Work

- [x] Implements QA orchestration via `QACoordinator` across prepare, retrieval, prompt, and session update paths.
- [x] Implements section/chapter task orchestration via `SectionTaskCoordinator`, including task-layout projection assembly.
- [x] Maintains hierarchy-required fail-fast behavior for incompatible runtime structure states.
- [x] Replaced legacy chapter-title fallback with fail-fast hierarchy lookup behavior.
- [x] Exposes task-unit content lookup through coordinator boundary with hierarchy-only id resolution.
- [x] Passes through additive `content_blocks` in task-unit content coordinator response via `TaskUnit.to_content_blocks()`.
- [x] Passes through explicit `segmented` option in task-unit content coordinator response (`segmented=true` uses shared segmentation helper; default/false keeps compatibility-safe block behavior).

#### Needs Confirmation

No unresolved confirmation items identified in module checklist.

#### Future Direction Preparation

- [ ] Added unchecked future tasks for app-layer rich content interaction API preparation in `app/module-checklist.md`.
- [ ] App-layer future direction keeps task-layout lightweight and separates on-demand rich-content read path from task-layout projection path.
- [ ] Future-direction items above remain planning-only; current additive content-block runtime integration is captured in completed checklist items.

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
- Status: `Completed Baseline Captured`
- Completed item count: `11`
- Needs confirmation count: `0`

#### Completed Work

- [x] Implements hierarchy-first structured model contracts (`StructuredDocument`, chapter, section) with pure-hierarchy write defaults.
- [x] Implements hierarchy-first effective indexing helpers and section lookup paths.
- [x] Implements hierarchy-aware artifact repository with strict hierarchy-required runtime load paths.
- [x] Completed document governance cleanup for hierarchy-first persistence terminology and legacy wording boundary separation.
- [x] Synchronized unresolved confirmation status after governance cleanup across detailed-design/checklist/progress.
- [x] Clarified artifact governance and hierarchy persistence boundary (truth/output/projection ownership split).
- [x] Closed governance terminology inconsistency by deprecating `mirror` contract wording and enforcing compatibility-only fallback wording.
- [x] Completed allow_legacy_fallback retirement audit with compatibility-only isolation (hierarchy-first runtime no longer depends on fallback success).
- [x] Removed `allow_legacy_fallback` API surface and enforced hierarchy-only runtime lookup.
- [x] Isolated legacy read compatibility from normal model/repository boundaries via strict hierarchy read contract + explicit migration-only loaders.
- [x] Defined hierarchy-aware artifact target validation boundary as future-direction governance contract (`ArtifactTargetRef` metadata-only semantics, validation lifecycle, stale-ref/error taxonomy, allowed target combinations, and metadata glossary alignment).

#### Needs Confirmation

No unresolved confirmation items identified in module checklist.

#### Future Direction Preparation

- [ ] Rich task-unit content governance preparation is captured as unchecked future work in `document_structure/module-checklist.md`.
- [ ] Future preparation includes hierarchy-boundary rules to keep content blocks out of persisted hierarchy truth.
- [ ] `document_structure/module-detailed-design.md` now includes `Future Direction Note: Rich Task-Unit Content Governance Preparation` for boundary clarification only.
- [ ] Segmentation boundary preparation is documented as future-direction governance only (no segmentation implementation/runtime behavior changes in this pass).
- [ ] Future unchecked tasks now include segmentation-vs-hierarchy persistence boundary, resegmentation stale-target semantics, and content-block persistence non-authority rule.

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

#### Future Direction Preparation

- [ ] Added unchecked future tasks for rich-content-aware evaluation preparation in `evaluated_answer/module-checklist.md`.
- [ ] Maintainer-confirmed design contract fixes deterministic evidence baseline to `content_block_id + quote_span_start + quote_span_end + source_hash` (strict deterministic-first).
- [ ] Maintainer-confirmed policy keeps content-block evidence trace runtime-projection-only (no persisted trace metadata in current phase).
- [ ] Reparse stale-evidence taxonomy (`malformed`/`unresolved`/`stale`/`source-mismatched`) is documented as future-direction boundary, not runtime implementation.
- [ ] This update is documentation/checklist preparation only; no runtime behavior, scoring logic, LLM evaluation implementation, or API/schema behavior was changed.

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

#### Future Direction Preparation

- [ ] Added unchecked future tasks for rich-content question targeting preparation in `question/module-checklist.md`.
- [ ] Future direction keeps question targeting id-based and hierarchy-context-aware while treating content-block references as interaction semantics (not hierarchy ownership).
- [ ] This synchronization is documentation/checklist preparation only; no runtime/retrieval/persistence behavior changed.

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
- Status: `Completed Baseline Captured`
- Completed item count: `12`
- Needs confirmation count: `0`

#### Completed Work

- [x] Implements chapters-first task-layout DTO contracts and diagnostics DTO types.
- [x] Implements hierarchy-first task unit resolution entry using effective hierarchy sections.
- [x] Implements section task context lookup with hierarchy-only section resolution behavior.
- [x] Completed documentation governance cleanup for hierarchy-first task-layout semantics and projection/write boundary wording.
- [x] Clarified artifact availability projection vs artifact persistence truth boundary for task-layout/read path semantics.
- [x] Closed terminology governance item by deprecating `artifact mirror` as formal contract wording.
- [x] Adds on-demand task-unit content lookup support via stable `task_unit_id` without expanding task-layout payload.
- [x] Adds backward-compatible rich content blocks to task-unit content lookup response while keeping task-layout payload lightweight.
- [x] Adds explicit `segmented` opt-in for task-unit content endpoint to return deterministic multi-block output while preserving default compatibility-safe block behavior.
- [x] Safely passes through content-block artifact target metadata in task-unit content response with metadata glossary filtering and no artifact repository integration.
- [x] Documents minimum artifact target metadata glossary and allowed target combinations for content endpoint pass-through boundary.
- [x] Prepares section_tasks segmentation design direction for future content-block projection behavior (endpoint projection semantics, failure/validation boundary, identity constraints, and artifact-target alignment guardrails) without runtime/API changes.

#### Needs Confirmation

No unresolved confirmation items identified in module checklist.

#### Future Direction Preparation

- [ ] Added unchecked future preparation tasks for rich content interaction planning in `section_tasks/module-checklist.md`.
- [ ] Task-layout remains lightweight metadata/projection contract; rich content is future-direction-only and remains on-demand API oriented.
- [ ] Advanced segmented content-block projection semantics (beyond current explicit opt-in endpoint integration) remain future implementation work.
- [ ] Duplicate/missing content-block validation behavior remains future implementation work.
- [ ] Segmented block artifact-target alignment behavior remains future implementation work.
- [ ] Future-direction items above remain planning-only; current additive content-block endpoint integration is captured in completed checklist items.

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
- Completed item count: `9`
- Needs confirmation count: `0`

#### Completed Work

- [x] Defines cross-module `TaskUnit` contract including parent identity and artifact payload fields.
- [x] Defines summary/quiz artifact schemas and document-level artifact container models.
- [x] Defines generic abstract result contract for service execution outputs.
- [x] Implements shared rich task-unit content block foundation with deterministic string-to-block adapter behavior.
- [x] Stabilizes TaskUnit rich content internal representation with additive `content_blocks`, auto-stabilization from `content`, compatibility-safe serialization, and no endpoint/task-layout/API/persistence migration.
- [x] Defines shared content-block artifact target foundation with additive metadata refs and serialization-safe backward compatibility.
- [x] Extracts artifact target level/ref contract into dedicated shared module to establish single-source cross-module governance.
- [x] Prepares shared segmentation design direction for future deterministic content-block generation (contract, deterministic id/hash/span policy direction, reparse risk model, and compatibility staging) without source-code/API/persistence changes.
- [x] Implements deterministic content block segmentation foundation as explicit opt-in shared-layer behavior with paragraph-first/list-safe rules, deterministic block ids, advisory source-hash/span metadata, and preserved default compatibility behavior.

#### Needs Confirmation

No unresolved confirmation items identified in module checklist.

#### Future Direction Preparation

- [ ] Finalize content-block identity and artifact attachment semantics beyond shared foundation metadata remains pending in `shared/module-checklist.md`.
- [ ] Hardening deterministic segmentation rules for heading/sentence/table-like structures remains future-direction work.
- [ ] Reparse-resilient block-id/source-hash/span evolution strategy remains future-direction work.
- [ ] Promotion strategy from explicit opt-in segmentation to default multi-block behavior remains future-direction work.
- [ ] Richer artifact targeting boundary for content blocks remains future-direction work and is not implemented in this slice.

## 6. Root Python Module Progress

### `api_schemas.py`

- Checklist: `api_schemas.module-checklist.md`
- Detailed Design: `api_schemas.module-detailed-design.md`
- Status: `Completed Baseline Captured`
- Completed item count: `10`
- Needs confirmation count: `0`

#### Completed Work

- [x] Defines external API schemas used by request and response boundaries.
- [x] Defines task-layout response contract with chapters-first projection fields and diagnostics response model.
- [x] Defines chapter summary/quiz request validation boundary for id/title target fields.
- [x] Defines dedicated task-unit content response schema for on-demand frontend rendering.
- [x] Adds backward-compatible `content_blocks` schema to task-unit content response while preserving compatibility path for legacy `content`.
- [x] Normalize official rich-content API response schema.
- [x] Preserves backward-compatible schema for segmented task-unit content response via additive `segmented` request flag and unchanged `content + content_blocks` response contract.
- [x] Reduces raw content exposure by making `content` compatibility/debug-oriented and introducing explicit `include_raw_content` opt-in while keeping `content_blocks` primary.
- [x] Validates content-block artifact target metadata response schema with explicit enum/constraint fail-fast behavior.
- [x] Reuses shared artifact target level contract to remove duplicated enum definitions across shared/API schema boundaries.

#### Needs Confirmation

No unresolved confirmation items identified in module checklist.

#### Future Direction Preparation

- [ ] Added unchecked future tasks for rich task-unit content schema evolution in `api_schemas.module-checklist.md`.
- [ ] API schema future direction keeps hierarchy-first contract and id-based targeting while preserving task-layout lightweight response boundary.
- [ ] Future-direction items above remain planning-only; current additive `content_blocks` schema integration is captured in completed checklist items.

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
- Completed item count: `9`
- Needs confirmation count: `0`

#### Completed Work

- [x] Defines FastAPI entrypoint and route registration for prepare/ask/task-layout/summary/quiz/reparse endpoints.
- [x] Maps API schemas to coordinator execution paths and response payload construction.
- [x] Implements explicit projection/mutation route boundary including manual reparse endpoint.
- [x] Adds read-only task-unit content endpoint resolved by `doc_name + task_unit_id`.
- [x] Exposes additive `content_blocks` in task-unit content endpoint response without changing route semantics.
- [x] Adds explicit `segmented` query option to task-unit content endpoint while preserving default backward-compatible response behavior.
- [x] Stops returning raw task-unit `content` by default and adds explicit `include_raw_content` compatibility flag.
- [x] Normalizes rich-content endpoint response mapping with official top-level content block schema usage.
- [x] Maps content-block artifact target metadata in task-unit content endpoint response with safe glossary-key filtering.

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
- [x] On-demand task-unit content read path exposed as separate API with backward-compatible additive `content_blocks`, preserving task-layout metadata-only contract.
  Modules: `section_tasks`, `app`, `api_schemas.py`, `main.py`

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

No unresolved cross-module confirmation items identified.
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
- Most aggregated module checklists report completed baseline items with no unresolved confirmation items in this aggregation pass.
- Global status currently reflects checklist aggregation, not roadmap completion.
- Hierarchy-first and pure-hierarchy persistence direction is consistently represented across structure, preparation, and task modules.
- Profile metadata and post-structure enrichment boundaries are captured as advisory signals, not parser authority.
- Task-layout projection boundary is documented as read/projection-focused in current docs/checklists.
- API entry/schema and coordinator orchestration baselines are captured and linked in module-level documentation.
- Support modules (language, llm, retrieval, embedding, session, shared, bundle utilities) all have checklist memory established.
