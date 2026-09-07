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
| `app/` | package | `app/module-checklist.md` | 9 | 0 | Task Layout Parse Provenance Verified |
| `auth/` | package | `auth/module-checklist.md` | 3 | 0 | Completed Baseline Captured |
| `config/` | package | `config/module-checklist.md` | 3 | 0 | Completed Baseline Captured |
| `context/` | package | `context/module-checklist.md` | 3 | 0 | Completed Baseline Captured |
| `doc_loaders/` | package | `doc_loaders/module-checklist.md` | 12 | 0 | OCR File Cache Removed |
| `db/` | package + design module | `db/module-checklist.md` | 45 | 0 | PostgreSQL Parse Provenance Runtime Verified |
| `document_preparation/` | package | `document_preparation/module-checklist.md` | 5 | 0 | OCR Layout Enhancement Design Captured |
| `document_structure/` | package | `document_structure/module-checklist.md` | 18 | 0 | Native PDF Outline Normalization Implemented |
| `embeddings/` | package | `embeddings/module-checklist.md` | 3 | 0 | Completed Baseline Captured |
| `evaluated_answer/` | package | `evaluated_answer/module-checklist.md` | 3 | 0 | Completed Baseline Captured |
| `language/` | package | `language/module-checklist.md` | 3 | 0 | Completed Baseline Captured |
| `llm/` | package | `llm/module-checklist.md` | 3 | 0 | Completed Baseline Captured |
| `profile/` | package | `profile/module-checklist.md` | 5 | 0 | Completed Baseline Captured |
| `prompts/` | package | `prompts/module-checklist.md` | 3 | 0 | Completed Baseline Captured |
| `question/` | package | `question/module-checklist.md` | 3 | 0 | Completed Baseline Captured |
| `retrieval/` | package | `retrieval/module-checklist.md` | 3 | 0 | Completed Baseline Captured |
| `scripts/` | package | `scripts/module-checklist.md` | 3 | 0 | Completed Baseline Captured |
| `section_tasks/` | package | `section_tasks/module-checklist.md` | 15 | 0 | Universal PDF Layout Evidence Governance Planned |
| `session/` | package | `session/module-checklist.md` | 3 | 0 | Completed Baseline Captured |
| `shared/` | package | `shared/module-checklist.md` | 9 | 0 | Completed Baseline Captured |
| `api_schemas.py` | root-python-module | `api_schemas.module-checklist.md` | 12 | 0 | Task Layout Parse Provenance Verified |
| `bundle_factory.py` | root-python-module | `bundle_factory.module-checklist.md` | 3 | 0 | Completed Baseline Captured |
| `bundle_provider.py` | root-python-module | `bundle_provider.module-checklist.md` | 3 | 0 | Completed Baseline Captured |
| `fingerprint_handler.py` | root-python-module | `fingerprint_handler.module-checklist.md` | 3 | 0 | Completed Baseline Captured |
| `main.py` | root-python-module | `main.module-checklist.md` | 12 | 0 | Health Request Logging Suppressed |

## 5. Package Module Progress

### `app/`

- Checklist: `app/module-checklist.md`
- Detailed Design: `app/module-detailed-design.md`
- Status: `Task Layout Parse Provenance Verified`
- Completed item count: `9`
- Needs confirmation count: `0`

#### Completed Work

- [x] Implements QA orchestration via `QACoordinator` across prepare, retrieval, prompt, and session update paths.
- [x] Implements section/chapter task orchestration via `SectionTaskCoordinator`, including task-layout projection assembly.
- [x] Maintains hierarchy-required fail-fast behavior for incompatible runtime structure states.
- [x] Replaced legacy chapter-title fallback with fail-fast hierarchy lookup behavior.
- [x] Exposes task-unit content lookup through coordinator boundary with hierarchy-only id resolution.
- [x] Passes through additive `content_blocks` in task-unit content coordinator response via `TaskUnit.to_content_blocks()`.
- [x] Passes through explicit `segmented` option in task-unit content coordinator response (`segmented=true` uses shared segmentation helper; default/false keeps compatibility-safe block behavior).
- [x] Suppresses duplicated leading hierarchy title in segmented render blocks while preserving quote-span traceability.
- [x] Projects accepted structured parse provenance through task-layout coordinator response without mutating hierarchy/profile state.

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

#### Future Direction Preparation

- [ ] Added unchecked future tasks for file / DB storage backend coexistence policy in `config/module-checklist.md`.
- [ ] Future direction keeps current file path behavior valid during DB migration rollout.
- [ ] Configuration-layer DB work remains planning-only; no database dependency or runtime backend selection behavior changed.
- [ ] Added unchecked future tasks for storage backend configuration contract, rollout policy contract, and file-to-db coexistence configuration model in `config/module-checklist.md`.
- [ ] Storage abstraction boundary remains future-direction planning only; `config/` owns backend selection and rollout policy, not domain persistence semantics.

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
- Status: `OCR File Cache Removed`
- Completed item count: `12`
- Needs confirmation count: `0`

#### Completed Work

- [x] Defines loader abstraction via `AbstractDocumentLoader.load(doc_name) -> str`.
- [x] Implements TXT loader and PDF loader for canonical raw text extraction.
- [x] Implements loader selection through `DocumentLoaderFactory` with extension/path heuristics and historical TXT default.
- [x] Detect scanned-image PDFs before prepare treats them as generic empty raw text.
- [x] Add explicit requires-OCR raw-load failure reason.
- [x] Design and implement optional OCR fallback behind explicit configuration or request option.
- [x] Deploy multilingual OCR language packages and align OCR language selection with project language-code strategy.
- [x] Retire OCR text file-cache persistence after OCR run storage.

- [x] Detect and expose native PDF Outline/bookmark structure before OCR, including validated destinations and nesting.
  Evidence: `Deep_Reflective_Reader/doc_loaders/pdf_outline.py`; `Deep_Reflective_Reader/doc_loaders/pdf_document_loader.py`; `Deep_Reflective_Reader/scripts/test_pdf_outline.py`; `许三观卖血记.pdf` container verification.

#### Needs Confirmation

No unresolved confirmation items identified in module checklist.

#### Future Direction Preparation

- [ ] Page-aware OCR provenance and stable page boundaries for future TOC detection are captured as unchecked future work in `doc_loaders/module-checklist.md`.
- [x] Native PDF Outline/bookmark inspection and the `Outline -> OCR TOC -> keyword matching` precedence chain are implemented and recorded in the corresponding module checklists.
- [ ] Universal PDF page-layout evidence, competing orientation hypotheses, region metadata, and tiered OCR cost policy remain future work in `doc_loaders/module-checklist.md`.
  Notes: The current implementation exposes limited page layout evidence and deterministic vertical/RTL selection; it does not yet preserve competing rotation hypotheses or full region geometry.
- [x] Renderer-first PDF page normalization for OCR is implemented and recorded in `doc_loaders/module-checklist.md`.
  Evidence: `Dockerfile`; `Deep_Reflective_Reader/doc_loaders/pdf_document_loader.py`; `docker-compose.yml`; `國富論.pdf` page 5 renderer/OCR verification.
- [x] OCR file-cache persistence is removed from PDF loading; OCR pages are reused only in memory during the active prepare pass and durable OCR output goes through structured-store OCR run persistence.

### `document_preparation/`

- Checklist: `document_preparation/module-checklist.md`
- Detailed Design: `document_preparation/module-detailed-design.md`
- Status: `Ordered PDF Structure Discovery Implemented`
- Completed item count: `5`
- Needs confirmation count: `0`

#### Completed Work

- [x] Implements ordered prepare pipeline with profile-before-structured sequencing.
- [x] Supports `base` and `free_qa` preparation modes with explicit mode contract.
- [x] Collects non-blocking profile/enrichment errors while preserving structured readiness semantics.
- [x] Discovers native PDF Outline before OCR/layout TOC analysis and preserves conservative fallback behavior.
  Evidence: `Deep_Reflective_Reader/document_preparation/document_preparation_pipeline.py`; `Deep_Reflective_Reader/document_structure/structured_document_builder.py`; `Deep_Reflective_Reader/scripts/test_pdf_outline.py`.

#### Needs Confirmation

No unresolved confirmation items identified in module checklist.

#### Future Direction Preparation

- [ ] Added unchecked future tasks for DB-backed structured persistence behavior in the preparation pipeline in `document_preparation/module-checklist.md`.
- [ ] Future direction preserves current file-based prepare outputs during migration.
- [ ] Future storage abstraction boundary must separately account for structured/profile/retrieval artifacts; this pass is documentation/checklist preparation only.
- [ ] Future storage independence note clarifies that preparation should prepare artifacts without permanently assuming file-path persistence as the only possible destination.
- [x] TOC-aware preparation orchestration and conservative fallback policy are implemented in `document_preparation/module-checklist.md`, with bounded page-evidence handoff and conservative parser fallback.
- [x] Universal page inventory, candidate-page analysis, cache reuse, cost budgets, and layout/TOC failure fallback are captured as unchecked future work in `document_preparation/module-checklist.md`.
  Notes: Documentation only; no new preparation runtime behavior is claimed complete.

### `db/`

- Checklist: `db/module-checklist.md`
- Detailed Design: `db/module-detailed-design.md`
- Status: `PostgreSQL Parse Provenance Runtime Verified`
- Completed item count: `45`
- Needs confirmation count: `0`

#### Completed Work

- [x] Captures DB-era identity strategy as documentation-only golden source.
- [x] Captures document-level `structure_version` policy.
- [x] Captures current-state-only hierarchy policy for early DB design.
- [x] Captures hard reparse transaction and cleanup policy.
- [x] Captures derived-resource provenance and validation policy.
- [x] Captures minimal parse event provenance policy.
- [x] Derives Phase 1 logical DB schema proposal from repository memory.
- [x] Confirms optional `StructuredDocument` JSONB snapshot boundary for Phase 1.
- [x] Confirms Phase 1 advisory profile snapshot placement and minimum metadata.
- [x] Confirms Phase 1 artifact payload grouping strategy.
- [x] Confirms Phase 1 public document identity requirement.
- [x] Confirms Phase 1 parse event retention policy.
- [x] Confirms Phase 1 raw document storage boundary.
- [x] Converts Phase 1 schema design into SQL/DDL migration plan.
- [x] Defines physical table, column, foreign-key, uniqueness, and index candidates for Phase 1.
- [x] Defines ORM/model mapping plan for Phase 1 entities without making ORM classes parser authority.
- [x] Defines DB repository/storage interfaces without making schema authority.
- [x] Defines document_structure storage integration plan for DB-backed hierarchy writes and reads.
- [x] Defines raw-source metadata persistence plan while keeping raw bytes file-backed/object-backed.
- [x] Defines advisory document_profile persistence plan without profile hierarchy authority.
- [x] Defines migration/evaluation fixtures that do not production-migrate existing JSON identity.
- [x] Defines empty-database / new-document ingestion migration path before any existing JSON production migration.
- [x] Defines transaction-level hard reparse implementation plan.
- [x] Defines parse event persistence implementation plan.
- [x] Defines content-block relational persistence implementation plan.
- [x] Defines artifact relational persistence implementation plan.
- [x] Defines application-level stale/invalid derived-row validation behavior.
- [x] Defines backend configuration integration with `config/`.
- [x] Defines optional `StructuredDocument` JSONB parity snapshot validation plan without runtime fallback authority.
- [x] Defines validation tests for current-state-only hierarchy replacement.
- [x] Defines DB parity validation fixtures against file-backed `StructuredDocument` golden outputs.
- [x] Defines runtime read/write switch plan behind explicit backend policy, without enabling it by default.
- [x] Defines rollback and failure-mode behavior for failed initial parse, failed hard reparse, and failed derived-resource cleanup.
- [x] Implements Phase 1 core DB hierarchy persistence slice for new-document isolated validation.
- [x] Enforces same-document parent consistency for PostgreSQL hierarchy and content-block rows.
- [x] Enforces PostgreSQL parse_event event-specific row-shape constraints.
- [x] Enforces conservative PostgreSQL numeric and span representation constraints.
- [x] Defines PostgreSQL `updated_at` ownership as application-managed.
- [x] Implements Docker PostgreSQL structured document runtime read/write switch for new documents.
- [x] Fixes PostgreSQL structured runtime read path for section-summary verification.
- [x] Persists structured parse provenance in PostgreSQL metadata and exposes effective parser mode through parse events for task-layout observability.

#### Needs Confirmation

No unresolved confirmation items identified in module checklist.

#### Future Direction Preparation

- [ ] Future production DB rollout remains pending for broader hard-reparse transaction behavior, relational artifact/content-block persistence beyond structured-document payload metadata, profile persistence wiring, existing JSON migration tooling if required, and broader API/path regression coverage.
- [ ] Future DB implementation must not production-migrate current JSON `unit_id` as stable identity.
- [ ] Future DB implementation must keep DB-generated IDs as default internal identity and add public/domain IDs only for concrete external stability requirements.
- [ ] Future hard reparse implementation must validate candidate hierarchy before transaction cutover and must explicitly delete all document-derived content blocks and artifacts after version advancement in the same transaction.
- [ ] Future production DB implementation remains pending for ORM mappings if needed, hard reparse transaction behavior, derived-resource persistence beyond the structured-document path, existing JSON migration, and broader tests; the current implementation is limited to new-document structured persistence behind explicit backend selection.

#### Latest Verification Detail

- [x] `Madame Bovary` `/documents/section-summary` now reads the PostgreSQL structured document through the storage abstraction and returns HTTP 200 for runtime section id `2`.
  Evidence: `Deep_Reflective_Reader/document_preparation/document_preparation_pipeline.py`; `Deep_Reflective_Reader/db/postgres_structured_document_store.py`; `Deep_Reflective_Reader/scripts/test_postgres_structured_uri_prepare_and_load.py`; `Deep_Reflective_Reader/docs/postgres-structured-runtime-verification.txt`; Docker API verification on `2026-08-08`.
  Notes: The fix prevents `postgres://structured/default/Madame Bovary` from being normalized to `postgres:/structured/default/Madame Bovary`, and makes DB task-unit writes idempotent by section/order during repeated runtime projections.

### `document_structure/`

- Checklist: `document_structure/module-checklist.md`
- Detailed Design: `document_structure/module-detailed-design.md`
- Status: `Native PDF Outline Normalization Implemented`
- Completed item count: `18`
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
- [x] Rejects LLM split plans that resolve main-body sections to TOC-only spans and falls back before hierarchy persistence.
- [x] Records requested/effective parser mode and LLM fallback reason as advisory `StructuredDocument` parse provenance.
- [x] Normalizes native PDF Outline into the current two-layer `chapters[].sections[]` hierarchy for scanned-image books, including level-0 outline chapters and single-root wrapper outlines.

#### Needs Confirmation

No unresolved confirmation items identified in module checklist.

#### Future Direction Preparation

- [ ] Rich task-unit content governance preparation is captured as unchecked future work in `document_structure/module-checklist.md`.
- [ ] Future preparation includes hierarchy-boundary rules to keep content blocks out of persisted hierarchy truth.
- [ ] `document_structure/module-detailed-design.md` now includes `Future Direction Note: Rich Task-Unit Content Governance Preparation` for boundary clarification only.
- [ ] Segmentation boundary preparation is documented as future-direction governance only (no segmentation implementation/runtime behavior changes in this pass).
- [ ] Future unchecked tasks now include segmentation-vs-hierarchy persistence boundary, resegmentation stale-target semantics, and content-block persistence non-authority rule.
- [ ] DB-centric structured persistence migration is captured as unchecked future work in `document_structure/module-checklist.md`.
- [ ] `document_structure/module-detailed-design.md` now includes `Future Direction Note: DB-Centric Structured Persistence Migration` for boundary clarification only.
- [ ] Future DB migration planning preserves hierarchy-first `StructuredDocument` semantics across file and DB storage.
- [ ] File-backed structured JSON remains valid compatibility/fallback/migration source until DB readiness validation; no runtime read/write behavior changed.
- [ ] Storage abstraction boundary planning clarifies that `document_structure` owns `StructuredDocument` hierarchy truth and domain persistence semantics, not backend selection or DB rollout strategy.
- [x] Deterministic TOC detection contract and atomic projection rejection are implemented in `document_structure/module-checklist.md`; shape projection, page boundary validation, multi-page grouping, and global validation remain staged future work.
- [ ] TOC-derived hierarchy must remain `chapters[].sections[]`; metadata and LLM classification remain advisory and cannot become parser authority.
- [x] Native PDF Outline normalization for scanned-image books is implemented: use normalized outline chapter level, flatten descendants into sections, map content by PDF page index plus OCR page boundaries, and reject incomplete projections atomically.
- [ ] Universal page-level TOC scoring, layout-hypothesis normalization, multi-page grouping, and global validation are captured as unchecked future work in `document_structure/module-checklist.md`.
- [ ] TOC projection implementation details are recorded as unchecked work: page-to-character boundary mapping, two-level hierarchy compression, multi-page termination, global validation, and reliable PDF fixture requirements.

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

- [ ] Support content-block-linked answer evidence.
- [ ] Define content-block reference validation semantics.
- [ ] Define artifact-linked evaluation flow.
- [ ] Define deterministic content-block evidence validation contract (baseline: `content_block_id + quote_span_start + quote_span_end + source_hash`; strict deterministic-first).
- [ ] Define runtime-only evidence trace projection flow (no persisted trace metadata in current phase).
- [ ] Define stale evidence classification semantics after reparse (`malformed` / `unresolved` / `stale` / `source-mismatched`).
- [ ] Define hierarchy-aware evidence targeting semantics.

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
- Status: `Task Layout Parse Provenance Verified`
- Completed item count: `15`
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
- [x] Hardens segmented task-unit content endpoint behavior with deterministic flag-matrix regression and multilingual segmentation fixtures (Chinese/Japanese paragraphs, mixed paragraph+list, heading-like, table-like) without task-layout/persistence/artifact/retrieval/LLM/evaluated_answer changes.
- [x] Suppresses duplicated section/chapter heading line in segmented content endpoint projection without changing hierarchy truth or task-layout payload.
- [x] Adds optional `ParseProvenanceDTO` to task-layout projection without adding heavy content or changing hierarchy nodes.

#### Needs Confirmation

No unresolved confirmation items identified in module checklist.

#### Future Direction Preparation

- [ ] Added unchecked future preparation tasks for rich content interaction planning in `section_tasks/module-checklist.md`.
- [ ] Task-layout remains lightweight metadata/projection contract; rich content is future-direction-only and remains on-demand API oriented.
- [ ] Advanced segmented content-block projection semantics (beyond current explicit opt-in endpoint integration) remain future implementation work.
- [ ] Duplicate/missing content-block validation behavior remains future implementation work.
- [ ] Segmented block artifact-target alignment behavior remains future implementation work.
- [ ] Future-direction items above remain planning-only; current additive content-block endpoint integration is captured in completed checklist items.
- [ ] Section-scoped task-layout ownership for TOC-derived hierarchy is captured as unchecked future work in `section_tasks/module-checklist.md`.
- [ ] Task-layout consumption of page/layout-derived hierarchy is captured as unchecked future work in `section_tasks/module-checklist.md`; task-layout remains projection-only and does not reconstruct TOC structure.

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
- Status: `Task Layout Parse Provenance Verified`
- Completed item count: `12`
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
- [x] Defines lightweight document list/search API schemas.
- [x] Defines optional task-layout `parse_provenance` response schema with requested/effective parser mode and fallback metadata.

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
- Status: `Health Request Logging Suppressed`
- Completed item count: `12`
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
- [x] Adds lightweight document list/search endpoint.
- [x] Maps optional task-layout parse provenance to public REST response and validates it through backend and UI proxy routes.
- [x] Suppresses successful `/health` request lifecycle logs while preserving non-health request completion logs and exception logs.

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
- [ ] DB-centric persistence migration direction captured as documentation/checklist preparation only; JSON/file and future DB storage may coexist during migration, with `data/` retirement gated by validation.
  Modules: `document_structure`, `config`, `document_preparation`
- [ ] Storage abstraction boundary direction captured as future planning only; domain modules own persistence semantics while `config` owns backend selection and rollout policy.
  Modules: `document_structure`, `config`, `document_preparation`, `profile`, `retrieval`, `doc_loaders`
- [ ] StructuredDocument JSONB-first Phase 1 evaluation plan is captured as documentation-only planning; PostgreSQL JSONB is an assumed evaluation backend, not a final backend commitment, and no schema, migration execution, repository interface, runtime read-path switch, or backend cutover is implemented.
  Modules: `document_structure`, `config`
- [ ] StructuredDocument JSONB-first Phase 1 readiness audit is captured as documentation-only planning; the audit inventories current structured files and identifies fixture gaps without implementing PostgreSQL, JSONB persistence, schema, migration, repository abstraction, runtime switching, or API behavior changes.
  Modules: `document_structure`
- [ ] Maintainer clarification for Phase 1 JSONB planning is captured as documentation-only future planning: `APPLE` is a valid no-task-unit `StructuredDocument` hierarchy reference file, current `unit_id` is reference/import identity evidence only, DB-era public/domain identity is introduced only for concrete external stability requirements, and lazy content-block persistence remains separate from `StructuredDocument` JSONB.
  Modules: `document_structure`, `shared`, `db`

- [x] DB-era identity/versioning/reparse golden source captured.
  Modules: `db`, `document_structure`, `shared`, `config`, `document_preparation`

- [x] Phase 1 logical DB schema proposal captured as documentation-only architecture memory.
  Modules: `db`, `document_structure`, `shared`, `config`, `document_preparation`

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

### 7.6 Storage Inventory Documentation

- [x] Storage Contract Inventory Documentation
  Status: `Completed`
  Type: `Documentation Only`
  Implementation Impact: `None`
  Evidence: `Deep_Reflective_Reader/db/storage-contract-inventory.md`; `Deep_Reflective_Reader/proposal.md (Storage Contract Inventory)`; `Deep_Reflective_Reader/document_structure/module-detailed-design.md (Storage Ownership Boundary)`; `Deep_Reflective_Reader/config/module-detailed-design.md (Storage Backend Governance)`

### 7.7 StructuredDocument JSONB-First Evaluation Planning

- [ ] StructuredDocument JSONB-First Evaluation Document
  Status: `Future Planning`
  Type: `Documentation Only`
  Implementation Impact: `None`
  Evidence: `Deep_Reflective_Reader/db/structured-document-jsonb-evaluation.md`; `Deep_Reflective_Reader/proposal.md (Phase 1 StructuredDocument JSONB-First Evaluation Plan)`; `Deep_Reflective_Reader/document_structure/module-checklist.md`; `Deep_Reflective_Reader/config/module-checklist.md`
  Notes: This planning note records the Phase 1 evaluation direction only. It does not approve DB implementation, schema design, migration execution, repository interface design, runtime read-path switching, backend cutover, or runtime behavior changes.

- [ ] StructuredDocument JSONB-First Evaluation Readiness Audit
  Status: `Future Planning`
  Type: `Documentation Only`
  Implementation Impact: `None`
  Evidence: `Deep_Reflective_Reader/db/structured-document-jsonb-evaluation-readiness.md`; `Deep_Reflective_Reader/document_structure/module-checklist.md`; `Deep_Reflective_Reader/data/structured/*.structured.json`
  Notes: This audit prepares future Phase 1 evaluation by inventorying current structured files and identifying fixture/readiness gaps. It does not mark Phase 1 evaluation complete and does not add schema, DB implementation, migration execution, runtime switch, API change, or source code change.

- [ ] Maintainer Identity and Lazy Content Persistence Clarification Sync
  Status: `Future Planning`
  Type: `Documentation Only`
  Implementation Impact: `None`
  Evidence: `Deep_Reflective_Reader/db/structured-document-jsonb-evaluation.md`; `Deep_Reflective_Reader/db/structured-document-jsonb-evaluation-readiness.md`; `Deep_Reflective_Reader/document_structure/module-checklist.md`; `Deep_Reflective_Reader/shared/module-checklist.md`; `Deep_Reflective_Reader/db/module-detailed-design.md`
  Notes: Clarifies that APPLE is valid no-task-unit hierarchy evaluation material, not legacy/pre-task-unit schema evidence, production migration constraint, or DB identity requirement; also clarifies task-unit identity strategy risk and lazy content-block persistence separation. Does not mark DB implementation, schema design, identity migration, fixture creation, content-block persistence, PostgreSQL, JSONB persistence, runtime behavior, or API changes complete.

### 7.8 DB-Era Identity and Reparse Golden Source

- [x] DB-Era Identity / Versioning / Reparse Golden Source
  Status: `Documentation Baseline Captured`
  Type: `Documentation Only`
  Implementation Impact: `None`
  Evidence: `Deep_Reflective_Reader/db/module-detailed-design.md`; `Deep_Reflective_Reader/db/module-checklist.md`; maintainer-confirmed grill-me decisions in current architecture task.
  Notes: Captures DB-generated IDs as default internal identity, public/domain IDs only for concrete external stability needs, no production dependency on current Python-generated `unit_id`, document-level monotonic `structure_version`, current-state-only hierarchy, hard reparse transaction order, physical deletion of all document artifacts/content blocks on successful hard reparse, and required minimal parse event provenance. Does not mark schema, ORM, migration, repository interface, fixture, runtime behavior, or API changes complete.

- [x] Phase 1 Logical DB Schema Proposal
  Status: `Documentation Baseline Captured`
  Type: `Documentation Only`
  Implementation Impact: `None`
  Evidence: `Deep_Reflective_Reader/db/module-detailed-design.md`; `Deep_Reflective_Reader/db/module-checklist.md`; `Deep_Reflective_Reader/db/storage-contract-design.md`; supporting module designs for `document_structure`, `shared`, `config`, and `document_preparation`.
  Notes: Derives logical persistence domains, proposed logical entities, entity responsibilities, relationships, ownership boundaries, JSONB-vs-relational placement rationale, and open design questions. Optional full `StructuredDocument` JSONB snapshot is confirmed only as a validation/parity/debug artifact, not runtime hierarchy authority or fallback. Does not mark SQL, DDL, ORM models, repository interfaces, migration scripts, fixtures, runtime behavior, API changes, PostgreSQL implementation, or backend selection complete.

- [x] Phase 1 DB Schema Design Preparation
  Status: `Documentation Planning Captured`
  Type: `Documentation Only`
  Implementation Impact: `None`
  Evidence: `Deep_Reflective_Reader/db/phase-1-schema-design.md`; `Deep_Reflective_Reader/db/module-detailed-design.md`; `Deep_Reflective_Reader/db/module-checklist.md`; supporting module designs for `document_structure`, `shared`, `profile`, `config`, and `document_preparation`.
  Notes: Defines logical tables/entities, candidate fields, relationships, ownership boundaries, constraint candidates, hard-reparse lifecycle, application-level validation, and governance guardrails for Phase 1. It does not mark SQL, DDL, ORM models, repository interfaces, migration scripts, fixtures, tests, runtime behavior, API changes, PostgreSQL implementation, or backend selection complete.

- [x] Phase 1 SQL/DDL Migration Plan
  Status: `Documentation Planning Captured`
  Type: `Documentation Only`
  Implementation Impact: `None`
  Evidence: `Deep_Reflective_Reader/db/phase-1-sql-ddl-migration-plan.md`; `Deep_Reflective_Reader/db/phase-1-schema-design.md`; `Deep_Reflective_Reader/db/module-checklist.md`.
  Notes: Converts the Phase 1 schema design into a migration planning reference covering DDL work units, dependency order, foreign-key direction, delete/retention strategy, rollback expectations, validation gates, future migration file shape, and governance guardrails. It does not mark executable SQL DDL, ORM models, repository interfaces, migration scripts, fixtures, tests, runtime behavior, API changes, PostgreSQL implementation, or backend selection complete.

- [x] Phase 1 Physical Schema Candidates
  Status: `Documentation Planning Captured`
  Type: `Documentation Only`
  Implementation Impact: `None`
  Evidence: `Deep_Reflective_Reader/db/phase-1-physical-schema-candidates.md`; `Deep_Reflective_Reader/db/phase-1-schema-design.md`; `Deep_Reflective_Reader/db/phase-1-sql-ddl-migration-plan.md`; `Deep_Reflective_Reader/db/module-checklist.md`.
  Notes: Defines physical table names, column/type candidates, required/nullability candidates, foreign-key directions, uniqueness candidates, index candidates, delete behavior candidates, cross-table consistency rules, and governance guardrails for Phase 1. It does not mark executable SQL DDL, ORM models, repository interfaces, migration scripts, fixtures, tests, runtime behavior, API changes, PostgreSQL implementation, or backend selection complete.

- [x] Phase 1 ORM/Model Mapping Plan
  Status: `Documentation Planning Captured`
  Type: `Documentation Only`
  Implementation Impact: `None`
  Evidence: `Deep_Reflective_Reader/db/phase-1-orm-model-mapping-plan.md`; `Deep_Reflective_Reader/db/phase-1-physical-schema-candidates.md`; `Deep_Reflective_Reader/db/phase-1-schema-design.md`; `Deep_Reflective_Reader/db/module-checklist.md`.
  Notes: Defines candidate ORM record classes, table mapping, relationship mapping, model responsibility boundaries, DTO/domain conversion boundaries, loading/session policy candidates, cascade/lifecycle boundaries, and validation responsibility split. It does not mark executable ORM model code, SQL DDL, repository interfaces, migration scripts, fixtures, tests, runtime behavior, API changes, PostgreSQL implementation, or backend selection complete.

- [x] Phase 1 Repository / Storage Interface Plan
  Status: `Documentation Planning Captured`
  Type: `Documentation Only`
  Implementation Impact: `None`
  Evidence: `Deep_Reflective_Reader/db/phase-1-repository-storage-interface-plan.md`; `Deep_Reflective_Reader/db/phase-1-orm-model-mapping-plan.md`; `Deep_Reflective_Reader/db/phase-1-physical-schema-candidates.md`; `Deep_Reflective_Reader/db/module-checklist.md`.
  Notes: Defines candidate storage ports, ownership boundaries, pseudo-interface operations, transaction/unit-of-work boundary, DTO boundary candidates, schema authority guardrails, and runtime integration boundaries. It does not mark Python repository interfaces, executable ORM model code, SQL DDL, migration scripts, fixtures, tests, runtime behavior, API changes, PostgreSQL implementation, or backend selection complete.

- [x] Phase 1 document_structure Storage Integration Plan
  Status: `Documentation Planning Captured`
  Type: `Documentation Only`
  Implementation Impact: `None`
  Evidence: `Deep_Reflective_Reader/db/phase-1-document-structure-storage-integration-plan.md`; `Deep_Reflective_Reader/db/phase-1-repository-storage-interface-plan.md`; `Deep_Reflective_Reader/db/phase-1-schema-design.md`; `Deep_Reflective_Reader/db/module-checklist.md`.
  Notes: Defines future `document_structure` integration boundaries for DB-backed initial hierarchy writes, current hierarchy reads, hard-reparse replacement, mapper/DTO boundaries, fail-fast behavior, and file/DB coexistence. It does not mark Python code, repository interfaces, ORM models, SQL DDL, migration scripts, fixtures, tests, runtime behavior, API changes, PostgreSQL implementation, or backend selection complete.

- [x] Phase 1 Remaining DB Planning Batch
  Status: `Documentation Planning Captured`
  Type: `Documentation Only`
  Implementation Impact: `None`
  Evidence: `Deep_Reflective_Reader/db/phase-1-raw-source-metadata-persistence-plan.md`; `Deep_Reflective_Reader/db/phase-1-document-profile-persistence-plan.md`; `Deep_Reflective_Reader/db/phase-1-migration-evaluation-fixtures-plan.md`; `Deep_Reflective_Reader/db/phase-1-empty-database-ingestion-path-plan.md`; `Deep_Reflective_Reader/db/phase-1-hard-reparse-transaction-plan.md`; `Deep_Reflective_Reader/db/phase-1-parse-event-persistence-plan.md`; `Deep_Reflective_Reader/db/phase-1-content-block-persistence-plan.md`; `Deep_Reflective_Reader/db/phase-1-artifact-persistence-plan.md`; `Deep_Reflective_Reader/db/phase-1-derived-row-validation-plan.md`; `Deep_Reflective_Reader/db/phase-1-config-backend-integration-plan.md`; `Deep_Reflective_Reader/db/phase-1-structured-document-jsonb-parity-snapshot-plan.md`; `Deep_Reflective_Reader/db/phase-1-current-state-hierarchy-validation-test-plan.md`; `Deep_Reflective_Reader/db/phase-1-db-parity-validation-fixtures-plan.md`; `Deep_Reflective_Reader/db/phase-1-runtime-read-write-switch-plan.md`; `Deep_Reflective_Reader/db/phase-1-db-failure-mode-rollback-plan.md`; `Deep_Reflective_Reader/db/module-checklist.md`.
  Notes: Completes the remaining DB checklist planning items for raw-source metadata, advisory profile persistence, fixture strategy, empty-database ingestion, hard reparse transaction behavior, parse events, content blocks, artifacts, derived-row freshness validation, config integration, JSONB parity snapshots, validation-test planning, parity-fixture planning, runtime switch policy, and rollback/failure modes. It does not mark Python code, SQL DDL, ORM models, repository/storage implementations, migration scripts, fixtures, tests, runtime behavior, API changes, PostgreSQL implementation, object storage integration, or backend selection complete.

- [x] Phase 1 Core DB Hierarchy Persistence Slice
  Status: `Implementation Slice Captured`
  Type: `Implementation + Validation`
  Implementation Impact: `Isolated DB validation only`
  Evidence: `Deep_Reflective_Reader/db/sqlite_validation/phase_1_core_hierarchy_schema.sql`; `Deep_Reflective_Reader/db/phase_1_core_schema.py`; `Deep_Reflective_Reader/db/sqlite_core_document_store.py`; `Deep_Reflective_Reader/scripts/test_db_phase_1_core_hierarchy_persistence.py`; `Deep_Reflective_Reader/db/module-checklist.md`.
  Notes: Keeps SQLite isolated under `db/sqlite_validation/` for new-document hierarchy validation only: schema application, accepted hierarchy write, required namespace/document-name identity, raw-source metadata row, initial parse event, no `documents.raw_text`, and current hierarchy readback with DB-generated document/chapter/section/task-unit IDs. It does not enable production runtime DB reads/writes, profile persistence, content-block persistence, artifact persistence, hard reparse transaction behavior, JSONB runtime fallback, public/domain IDs, existing JSON production migration, ORM mappings, or backend selection.

- [x] Phase 1 PostgreSQL DDL Surface Completion
  Status: `Implementation Slice Captured`
  Type: `Implementation + Static Validation`
  Implementation Impact: `PostgreSQL DDL shape only`
  Evidence: `Deep_Reflective_Reader/db/migrations/001_phase_1_core_hierarchy.sql`; `Deep_Reflective_Reader/scripts/test_db_phase_1_postgresql_migration_shape.py`; `Deep_Reflective_Reader/scripts/test_db_phase_1_postgresql_relational_consistency_smoke.py`; `Deep_Reflective_Reader/db/module-checklist.md`.
  Notes: Completes the PostgreSQL-targeted Phase 1 DDL surface for documents, raw-source metadata, advisory document profile, parse events, current hierarchy rows, lazy content blocks, common artifacts, and optional structured-document parity snapshots. It preserves raw-source metadata-only storage, one common artifact table, application-level polymorphic artifact target validation, and no runtime backend switch.

- [x] Phase 1 PostgreSQL Same-Document Parent Consistency
  Status: `Implementation Slice Captured`
  Type: `Implementation + Static Validation`
  Implementation Impact: `PostgreSQL DDL shape only`
  Evidence: `Deep_Reflective_Reader/db/migrations/001_phase_1_core_hierarchy.sql`; `Deep_Reflective_Reader/scripts/test_db_phase_1_postgresql_migration_shape.py`; `Deep_Reflective_Reader/scripts/test_db_phase_1_postgresql_relational_consistency_smoke.py`; `Deep_Reflective_Reader/db/module-checklist.md`.
  Notes: Adds parent-side `unique(id, document_id)` constraints and composite foreign keys for `sections`, `task_units`, and `content_blocks` so duplicated child `document_id` values cannot disagree with their static parent rows. It does not add composite artifact target foreign keys and does not enable runtime DB reads or writes.

- [x] Phase 1 PostgreSQL Parse Event Shape Constraints
  Status: `Implementation Slice Captured`
  Type: `Implementation + Static Validation`
  Implementation Impact: `PostgreSQL DDL shape only`
  Evidence: `Deep_Reflective_Reader/db/migrations/001_phase_1_core_hierarchy.sql`; `Deep_Reflective_Reader/scripts/test_db_phase_1_postgresql_migration_shape.py`; `Deep_Reflective_Reader/scripts/test_db_phase_1_postgresql_relational_consistency_smoke.py`; `Deep_Reflective_Reader/db/module-checklist.md`.
  Notes: Adds CHECK constraints for `initial_parse` and `hard_reparse` version semantics, reparse-only metadata on initial parse, and non-negative invalidation counts. It keeps parse events as provenance only and does not add triggers, event sourcing, hierarchy history, or parse-event-driven `documents.current_structure_version` updates.

- [x] Phase 1 PostgreSQL Numeric And Span Representation Constraints
  Status: `Implementation Slice Captured`
  Type: `Implementation + Static Validation`
  Implementation Impact: `PostgreSQL DDL shape only`
  Evidence: `Deep_Reflective_Reader/db/migrations/001_phase_1_core_hierarchy.sql`; `Deep_Reflective_Reader/scripts/test_db_phase_1_postgresql_migration_shape.py`; `Deep_Reflective_Reader/scripts/test_db_phase_1_postgresql_relational_consistency_smoke.py`; `Deep_Reflective_Reader/db/module-checklist.md`.
  Notes: Adds conservative CHECK constraints for non-negative order fields, file size, section offsets, invalidation counts, and quote-span endpoints, plus ordered section and quote-span ranges. One-sided quote spans remain allowed; the DB validates representation integrity only and does not infer, repair, or classify spans.

- [x] Phase 1 PostgreSQL Updated-At Ownership Policy
  Status: `Implementation Slice Captured`
  Type: `Architecture + Implementation + Static Validation`
  Implementation Impact: `PostgreSQL DDL shape and repository obligation only`
  Evidence: `Deep_Reflective_Reader/db/migrations/001_phase_1_core_hierarchy.sql`; `Deep_Reflective_Reader/scripts/test_db_phase_1_postgresql_migration_shape.py`; `Deep_Reflective_Reader/scripts/test_db_phase_1_postgresql_relational_consistency_smoke.py`; `Deep_Reflective_Reader/db/module-checklist.md`.
  Notes: Chooses application-managed `updated_at` ownership. PostgreSQL may initialize insert defaults, but document and artifact mutations must set `updated_at` explicitly in repository/service update statements. No timestamp trigger or lifecycle trigger was introduced.

- [x] Docker PostgreSQL Structured Document Runtime Switch
  Status: `Runtime Switch Captured`
  Type: `Implementation + Docker Validation`
  Implementation Impact: `New-document structured persistence can run through PostgreSQL in Docker`
  Evidence: `Deep_Reflective_Reader/db/postgres_structured_document_store.py`; `Deep_Reflective_Reader/db/postgres_structured_document_artifact_repository.py`; `Deep_Reflective_Reader/config/app_DI_config.py`; `Deep_Reflective_Reader/config/container.py`; `Deep_Reflective_Reader/document_preparation/document_preparation_pipeline.py`; `docker-compose.yml`; Docker smoke validation confirming rows in `documents`, `chapters`, `sections`, `task_units`, and `structured_document_snapshots`; container health check; `Deep_Reflective_Reader/db/module-checklist.md`.
  Notes: Adds explicit `file` / `postgres` structured storage backend selection. Docker defaults structured document read/write to PostgreSQL through `DEEP_READER_STRUCTURED_STORAGE_BACKEND=postgres` and `DEEP_READER_POSTGRES_DSN`; local default remains file-backed. Runtime hierarchy reads are rebuilt from relational current hierarchy rows, while optional `structured_document_snapshots` remain parity/debug evidence only and are not runtime hierarchy authority. Existing `data/structured` files are not migrated by this implementation.

## 8. Cross-Module Needs Confirmation

No unresolved cross-module confirmation items identified.

## 9. Missing or Weak Checklists

- No checklist files are missing in this pass.
- `db/` now has DB-era planning memory, an isolated Phase 1 core hierarchy validation implementation slice, and a Docker PostgreSQL structured-document runtime switch for new documents.
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
- DB-centric persistence now has PostgreSQL DDL shape validation and a Docker structured-document read/write switch for new documents; existing structured JSON migration remains out of scope.
- Existing `data/` file storage remains valid for local/default compatibility and migration/reference use; Docker structured document read/write can use PostgreSQL for new documents without production-migrating existing `data/structured` files.
- Storage Contract Inventory Documentation is completed as documentation-only work with no implementation impact.
- Storage abstraction boundary remains guarded: backend selection is explicit, Docker structured read/write can target PostgreSQL, and no hidden dual-write or silent fallback is introduced.
- DB-era identity/versioning/reparse policy remains governed by `db/module-detailed-design.md`; the current runtime switch covers new-document structured persistence only and does not complete hard-reparse or existing JSON migration.
- Phase 1 logical DB schema proposal is captured in `db/module-detailed-design.md`; it is documentation-only and has no unresolved Phase 1 logical schema confirmation items in `db/module-checklist.md`.
- Profile metadata and post-structure enrichment boundaries are captured as advisory signals, not parser authority.
- Task-layout projection boundary is documented as read/projection-focused in current docs/checklists.
- API entry/schema and coordinator orchestration baselines are captured and linked in module-level documentation.
- Support modules (language, llm, retrieval, embedding, session, shared, bundle utilities) all have checklist memory established.
