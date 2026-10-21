# Deep Reflective Reader Progress

## Reviewable OCR Recovery (2026-10-21)

Backend-owned shared queue: OCR-REVIEW-01..06 complete: 6/6 (100%), zero pending.
Explicit OCR supports source-linked reviewable partial imports with conservative
document-wide generation guards. Versioned block Edit/Delete actions and single-level
undo preserve source/profile/notes; manual addition is not supported. File/SQL
mutation tests, backup-only Sample/Sample2 actual OCR, fault-injected partial review,
and desktop/mobile production-editor/API checks pass. Fast/Auto are unchanged.
This does not claim arbitrary-scan accuracy or production Library reparse/deployment.
Source: doc_loaders/module-checklist.md#reviewable-ocr-recovery.
Timestamp: 2026-10-22

## Direction-Aware OCR Remediation (2026-10-21)

doc_loaders OCR-DIR-01..07: 7/7 complete (100%) within the scoped acceptance;
zero pending in this track. Chinese-local routing, pinned models, column/label
assembly, quality diagnostics and model identity are implemented. Private backed-up
six-page sample passes explicit OCR title/order/label acceptance: 12/17 columns,
4.82%/1.85% CJK title CER below 8%; real-image horizontal-header/vertical-body passes.
Decorative headings remain imperfect and are excluded from title CER. Auto remains
a verified native_coverage_unresolved rejection, not success. Fast uses zero
subprocesses. Tests: 15 Chinese, 101 overlapping focused, 180 PDF (178 pass/two
skips), 18/18 synthetic real-engine mode cases. No universal rotation/artistic-layout
guarantee, live Library reparse, profile mutation or running-container deployment.
Source: [checklist](doc_loaders/module-checklist.md#direction-aware-ocr-remediation)
and [design](doc_loaders/module-detailed-design.md#direction-aware-ocr-remediation-2026-10-21).
Timestamp: 2026-10-21

## Loader Boundary Governance (2026-10-21)

LOADER-01..04: 4/4 completed (100%), zero pending in this backend track.
Owners: doc_loaders 2/2, document_preparation 1/1, scripts 1/1. SourceSession and
SourceArtifact now separate extraction implementations from preparation control;
legacy API/provenance compatibility remains explicit. PDF 180 tests pass, focused
95 pass, PostgreSQL 88 pass/two skips; suites overlap. Both backed-up real books
pass isolated cold/warm preparation and final live read-only acceptance. The earlier
default-Common acceptance changed Red-book hierarchy; layout-only recovery restored
93 units. See [preparation governance](document_preparation/module-detailed-design.md#loader-boundary-governance-2026-10-21)
for evidence and limitations. No EPUB/DOCX production support or new live LLM
acceptance is claimed; unrelated historical counts remain unchanged.
Timestamp: 2026-10-21

## Common Multilingual Correctness Backlog (2026-10-21)

document_structure COMMON-LANG-01: candidate/final heading evidence now shares
the selected language registry, with source-local guards retained. Track status:
1/1 completed (100%), no pending item in this track. Evidence: 63 unittest cases,
four standalone regressions and both real books deleted/rebuilt from verified
backups; final content reads and profile immutability pass. Multilingual fixtures
cover 19 registry languages; real-book coverage is English/Chinese only.
Unrelated historical completion counts are unchanged.
Timestamp: 2026-10-21
Source: [module checklist](document_structure/module-checklist.md#common-multilingual-correctness-backlog)
and [design](document_structure/module-detailed-design.md#common-multilingual-validation-mismatch-2026-10-21).

## Format-Neutral Extraction Backlog (2026-10-21)

EXTRACT-01..08 and EXTRACT-10 implemented and accepted. Authoritative order and
acceptance: [preparation design](document_preparation/module-detailed-design.md#format-neutral-extraction-plan-2026-10-21).
This pending track does not invalidate earlier Common/content/PDF-mode acceptance,
which did not establish TXT LLM-enhance readiness.

| Backend owner | Pending IDs | Completed / total |
|---|---|---|
| doc_loaders | none | 3 / 3 |
| document_preparation | none | 3 / 3 |
| db | none | 1 / 1 |
| app | none | 1 / 1 |
| scripts | none | 1 / 1 |

Backend track: 9/9 (100%), no unresolved tasks in this track. UI EXTRACT-09 is tracked
only in Deep_Reader_UI/module-checklist.md and excluded from this aggregate.
Counts apply only to this new track, not unrelated historical backlog.

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
| `app/` | package | `app/module-checklist.md` | 45 | 0 | Format-Neutral Enhancement Accepted |
| `auth/` | package | `auth/module-checklist.md` | 3 | 0 | Completed Baseline Captured |
| `config/` | package | `config/module-checklist.md` | 3 | 0 | Completed Baseline Captured |
| `context/` | package | `context/module-checklist.md` | 3 | 0 | Completed Baseline Captured |
| `doc_loaders/` | package | `doc_loaders/module-checklist.md` | 57 | 0 | Reviewable OCR Recovery 6/6 (2026-10-22) |
| `db/` | package + design module | `db/module-checklist.md` | 55 | 0 | Generic Extraction Commit Accepted |
| `document_preparation/` | package | `document_preparation/module-checklist.md` | 25 | 0 | Generic Accepted Sources Delivered |
| `document_structure/` | package | `document_structure/module-checklist.md` | 60 | 0 | ID-Free Semantic Plan Accepted On Real Book |
| `embeddings/` | package | `embeddings/module-checklist.md` | 3 | 0 | Completed Baseline Captured |
| `evaluated_answer/` | package | `evaluated_answer/module-checklist.md` | 3 | 0 | Completed Baseline Captured |
| `language/` | package | `language/module-checklist.md` | 3 | 0 | Completed Baseline Captured |
| `llm/` | package | `llm/module-checklist.md` | 4 | 0 | Full-Source FULL-01 Capacity Migration Delivered |
| `profile/` | package | `profile/module-checklist.md` | 5 | 0 | Completed Baseline Captured |
| `prompts/` | package | `prompts/module-checklist.md` | 3 | 0 | Completed Baseline Captured |
| `question/` | package | `question/module-checklist.md` | 3 | 0 | Completed Baseline Captured |
| `retrieval/` | package | `retrieval/module-checklist.md` | 3 | 0 | Completed Baseline Captured |
| `scripts/` | package | `scripts/module-checklist.md` | 59 | 0 | Real TXT/PDF Enhancement Accepted |
| `section_tasks/` | package | `section_tasks/module-checklist.md` | 92 | 0 | Summary And Note Real-Book Acceptance Complete |
| `session/` | package | `session/module-checklist.md` | 3 | 0 | Completed Baseline Captured |
| `shared/` | package | `shared/module-checklist.md` | 9 | 0 | Completed Baseline Captured |
| `api_schemas.py` | root-python-module | `api_schemas.module-checklist.md` | 34 | 0 | Summary And Note Schemas Delivered |
| `bundle_factory.py` | root-python-module | `bundle_factory.module-checklist.md` | 4 | 0 | Runtime Profile Reuse Only |
| `bundle_provider.py` | root-python-module | `bundle_provider.module-checklist.md` | 3 | 0 | Completed Baseline Captured |
| `fingerprint_handler.py` | root-python-module | `fingerprint_handler.module-checklist.md` | 3 | 0 | Completed Baseline Captured |
| `main.py` | root-python-module | `main.module-checklist.md` | 40 | 0 | Summary And Note Routes Exposed |

## 5. Package Module Progress

Summary/Note resumed backend track (2026-10-21): SN-01..16 complete, 16/16 (100%).
Owner counts above were recounted from their actual checklists, including correction
of the formerly stale api_schemas aggregate. This is not completion of unrelated
future tasks. UI ownership/progress remains in Deep_Reader_UI/module-checklist.md,
not aggregated here.

Implemented independent summary read/generate/refresh and single-note read/save/
delete APIs, full primary prompt plus lazy ancestor summaries, capacity/language/
abstention validation, atomic note revision/token/idempotency guards and lifecycle
cleanup/retention. 32 initial contract/service/disposable-PostgreSQL tests, four
conditional enhancement outcome tests and a further real rename-route test pass.
Existing interaction schemas/store/read-generate and rename regressions pass.
User-authorized live acceptance backs up raw/DB/derived files with immutable hashes,
deletes both named books via product API, verifies absence, uploads only backups,
and reparses. Both books pass all five Summary/Note levels using the actual configured
provider; final deployed read/generate reuse report is accept-20261021T064438.json
under ignored data/test-books/summary-note-20261021. No private book/prompt/response
files are committed. File/PostgreSQL failure tests remain isolated.

Semantic-plan ownership refactor (2026-10-21): SEM-01..04 complete, 4/4 (100%).
The model emits semantic titles/levels/quotes, while exact positions and IDs remain
backend-owned. Two independent final v15 real-book runs pass (147401 characters,
exact preservation, 3/33 and 4/27 chapters/sections). Accepted output also passes
real API/disposable PostgreSQL commit/cleanup and real layout with 107 task units;
original book/interactions unchanged. Backend regressions: 179 pass, one private
opt-in skip; live harnesses separately executed. This supersedes historical ID-copy
mitigation evidence, not unrelated future tasks. Module completed count is 60,
excluding the literal checklist template; 18 historical future items remain open.
Two successful runs are incident acceptance, not guaranteed model/editorial quality.

Enhancement autonomy/conditional lifecycle revision (updated 2026-10-21): backend 5/5
accepted (100%). document_structure owns ENH-REV-01/04/06; db owns ENH-REV-02;
app owns ENH-REV-03. ENH-REV-01 pure comparator is complete (20 focused tests and
42 plan regressions). ENH-REV-02 conditional persistence is complete (42 passing
PostgreSQL cases, one private-source opt-in skipped). ENH-REV-03 delivers public
three-way outcomes, conditional consent and bounded process-local result/retry handling;
seven-suite acceptance: 123 pass, one private-source skip. ENH-REV-04 delivers v8
autonomous semantic planning instructions with unchanged body guards/Common/model
limits; ten-suite acceptance: 144 pass, one private-source skip. ENH-REV-06 adds a
single lifecycle acceptance entry: 154 backend tests pass, one private-source skip;
real API/disposable PostgreSQL browser checks cover three outcomes, preserved row
payloads, retry, lost responses and explicit reading recovery. Live-model semantic
quality is unverified, separate from contract tests. No production deployment.
No-op does not commit/clear; title-only retains
interactions; boundary/hierarchy/parent change uses full-document cleanup. Common
conservatism must not govern LLM semantic planning. UI readiness/recovery work is
tracked only in its owned checklist and is not aggregated here.
This new track does not reuse historical FULL completion as acceptance.

Full-source enhancement follow-up (2026-10-20): backend 4/4 completed (100%).
Owners: llm FULL-01, document_structure FULL-02/FULL-05, app FULL-03.
FULL-01/02/03/05 are checked; no backend task remains in this follow-up. Frontend work
is tracked only in its own module documents, not aggregated here. Official mini
context/output values, capability migration and full-source dispatch are verified
offline. Capacity preflight/consent passed isolated API/PostgreSQL validation;
FULL-05 passed 139 backend tests (138 passed, one private-source opt-in skipped)
and desktop/mobile real API/disposable PostgreSQL acceptance with deterministic
generation. Live model semantic quality remains unverified; no production deployment.
Evidence and dependencies are in owner designs/checklists.

### `app/`

- Checklist: `app/module-checklist.md`
- Detailed Design: `app/module-detailed-design.md`
- Status: `ENH-REV-03 Outcome Contract And Process-Local Recovery Delivered`
- Completed item count: `41` (including FULL-03 and ENH-REV-03).
- Needs confirmation count: `0`
- Unchecked item count: `13` (historical items; ENH-REV-03 complete).
- 2026-10-21 ENH-REV-03: typed outcome/cleanup/layout flags, conditional preflight,
  result lookup/replay and transient provider retry. 124 tests: 123 pass, one private
  opt-in skip. Process-local registry explicitly returns unknown after restart/expiry;
  no cross-worker durability, automatic LLM retry loop or frontend completion claimed.
- 2026-10-20 FULL-03 delivered: capacity-aware preflight, consent v2, under-lease
  revalidation and dispatch guard. 139 tests: 138 passed, one private-source skip;
  isolated PostgreSQL schema created/migrated/removed. No paid model or deployment.
- 2026-10-19: ENH-APP-01 delivered: PostgreSQL read-only source/impact snapshot,
  token guards and advisory single-flight handoff. Eight disposable-schema tests,
  nine consent tests and fourteen PDF-mode regressions pass. Actual enhancement
  PREP-03 now enables guarded execution; file hierarchy storage remains unsupported.
  New backend enhancement track is 5/5; UI is excluded
  from this backend progress aggregation.
- 2026-10-16 follow-up: LAYOUT-01..03 are `3/3`; explicit preparation owns writes
  and persisted readback. Ordinary reads reject missing/stale layouts without work.
  Explicit page evidence reuses source-validated persisted spans without extraction/LLM.

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
- [x] Defines manual structure commit coordinator DTO/result boundary before persistence orchestration.
- [x] Gates manual structure commit path with document_structure projection validation before persistence orchestration.
- [x] Gates manual structure commit path with raw source evidence loading and stale source-hash rejection.
- [x] Builds manual structure draft hierarchy after source evidence gates before repository persistence.
- [x] Orchestrates explicit manual structure reparse commit through the structured-document repository save boundary.
- [x] Audits app-layer reading interaction exposure and confirms target resolution exists while REST-facing read/generate/submit/retry orchestration remains unwired.

#### Needs Confirmation

No unresolved confirmation items identified in module checklist.

#### Future Direction Preparation

- [ ] Added unchecked future tasks for app-layer rich content interaction API preparation in `app/module-checklist.md`.
- [ ] App-layer future direction keeps task-layout lightweight and separates on-demand rich-content read path from task-layout projection path.
- [ ] Future-direction items above remain planning-only; current additive content-block runtime integration is captured in completed checklist items.
- [x] Explicit manual structure commit orchestration now validates and saves a normal hierarchy-only `StructuredDocument` through the repository boundary.
- [x] App-layer manual structure path keeps validation/preview non-mutating and commit reparse as an explicit mutation path separate from task-layout.
- [x] App-layer source-agnostic manual structure validation now delegates projection semantics to `document_structure/` and does not mutate task-layout/profile/artifacts.
- [x] Manual structure source evidence gate now rejects missing source evidence and stale `source_hash` before persistence orchestration.
- [x] Manual structure draft-build gate now validates raw-text-backed anchors before repository save.

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
- Status: `Reviewable OCR Recovery 2/6; Earlier Three-Mode Acceptance Complete`
- Completed item count: `53`
- Needs confirmation count: `0`
- Unchecked item count: `4`.
- 2026-10-17 final acceptance: PDF-FIX-03 complete; loader follow-up `5/5`.
  All 21 real-engine cases, 177 PDF regressions and 18 preparation/support tests
  pass. Actual-source fast/Auto use zero render/OCR; explicit OCR processes all
  13 pages. Ordered spans, warm reuse and atomic failure controls pass. Design
  18.8 records timings and runtime; no production deployment. Earlier entries
  below describe intermediate states, superseded by this final acceptance.
- 2026-10-16 routing delivery: loader follow-up `4/5`. PDF-AUTO-01 is complete:
  scoped native visibility proof, versioned Auto identity and bounded fontTools
  glyph evidence. 177 PDF tests pass; actual-source Auto preparation succeeds
  with zero subprocess calls, 13 ordered pages and warm reuse. See design 18.7.
  Final PDF-FIX-03 matrix remains pending; no deployment or extra task counted.
- 2026-10-16 acceptance follow-up: PDF-FIX-03 remains unchecked. Real-engine matrix
  passes 20/21 cases after observed-script-v3 CJK fragmentation correction; actual
  13-page PDF succeeds in fast and OCR, but auto fails native coverage. Failure
  preserves the prior hierarchy. 165 local PDF regression tests pass. See loader
  design 18.5 and `scripts/accept_pdf_modes.py`; no completion
  count change, deployment or PDF-AUTO-01 implementation.
- 2026-10-16 fail-fast follow-up: Auto coverage failure now stops at the first
  conflicting page and reports page/retry guidance. Actual-source regression:
  1 render/4 OCR calls, 21.2 seconds, saved fast hierarchy unchanged. All 169 PDF
  tests pass. PDF-FIX-03 stays pending; this is not a new completed task or a full
  matrix rerun. See loader design 18.6.
- 2026-10-15 revision: `3/5`; PDF-MODE-02/03 fast/full-OCR executors and PDF-FIX-02
  quality correction complete. Seven full-OCR isolation tests and fast, quality,
  raw recovery, execution pass and API contract regressions pass.
  PDF-MODE-05 now integrates auto entry and mode-aware preparation; the temporary
  501 gate is removed in code. Auto policy limitations remain; no deployment.
  PDF-FIX-01 is superseded by deferred PDF-AUTO-01, not completed.
  Timestamp: 2026-10-15
- Selective OCR track: `8/8 (100%)` across three owners; see section 13.

#### Completed Work

- [x] Defines loader abstraction via `AbstractDocumentLoader.load(doc_name) -> str`.
- [x] Implements TXT loader and PDF loader for canonical raw text extraction.
- [x] Implements loader selection through `DocumentLoaderFactory` with extension/path heuristics and historical TXT default.
- [x] Detect scanned-image PDFs before prepare treats them as generic empty raw text.
- [x] Add explicit requires-OCR raw-load failure reason.
- [x] Design and implement optional OCR fallback behind explicit configuration or request option.
- [x] Deploy multilingual OCR language packages and align OCR language selection with project language-code strategy.
- [x] Retire OCR text file-cache persistence after OCR run storage.
- [x] Stabilizes raw data directory resolution so default `data/raw` loading is package/project-root anchored rather than process-cwd dependent.
- [x] Preserves page-aware OCR/page-boundary provenance for future TOC detection while keeping the public raw-text loading contract unchanged.
  Evidence: `Deep_Reflective_Reader/doc_loaders/pdf_page_evidence.py`; `Deep_Reflective_Reader/doc_loaders/pdf_document_loader.py`; `Deep_Reflective_Reader/scripts/test_pdf_document_loader_inspection.py`; `Deep_Reflective_Reader/scripts/test_manual_structure_preparation_handoff.py`; `Deep_Reflective_Reader/scripts/test_manual_structure_commit_source_evidence.py`.
- [x] Defines universal PDF page-layout evidence metadata for source identity, page metrics, layout hypotheses, OCR confidence, and schema version.
  Evidence: `Deep_Reflective_Reader/doc_loaders/pdf_page_evidence.py`; `Deep_Reflective_Reader/doc_loaders/pdf_document_loader.py`; `Deep_Reflective_Reader/scripts/test_pdf_document_loader_inspection.py`; `Deep_Reflective_Reader/scripts/test_pdf_page_evidence.py`; `Deep_Reflective_Reader/scripts/test_toc_projection.py`.
- [x] Defines deterministic orientation and reading-order evidence with rotation hypotheses, writing-mode confidence, reading-order confidence, and ambiguous-orientation retention.
  Evidence: `Deep_Reflective_Reader/doc_loaders/pdf_page_evidence.py`; `Deep_Reflective_Reader/scripts/test_pdf_page_evidence.py`.
- [x] Defines page-level orientation and reading-order evidence contract with region count, OCR confidence, schema version, and serialized layout hypotheses.
  Evidence: `Deep_Reflective_Reader/doc_loaders/pdf_page_evidence.py`; `Deep_Reflective_Reader/scripts/test_pdf_page_evidence.py`.
- [x] Defines OCR quality and character provenance contract for OCR token traceability and bounded quality flags.
  Evidence: `Deep_Reflective_Reader/doc_loaders/pdf_page_evidence.py`; `Deep_Reflective_Reader/scripts/test_pdf_page_evidence.py`.
- [x] Defines tiered PDF inspection and OCR cost policy for cheap inventory, coordinate OCR candidates, cache keys, and bounded high-cost analysis policy.
  Evidence: `Deep_Reflective_Reader/doc_loaders/pdf_page_evidence.py`; `Deep_Reflective_Reader/doc_loaders/pdf_document_loader.py`; `Deep_Reflective_Reader/scripts/test_pdf_page_evidence.py`; `Deep_Reflective_Reader/scripts/test_pdf_document_loader_inspection.py`.
- [x] Defines region-first OCR evidence for mixed and vertical layouts, including bounded OCR regions with source coordinates, local reading order, raw OCR, normalized logical text, rotation metadata, confidence, and quality flags.
  Evidence: `Deep_Reflective_Reader/doc_loaders/pdf_page_evidence.py`; `Deep_Reflective_Reader/scripts/test_pdf_page_evidence.py`.
- [x] Prepares OCR layout and renderer-failure fixtures for artistic vertical TOC, vertical body text, horizontal scans, mixed orientation, circular page numbers, leader lines, and renderer decode fallback.
  Evidence: `Deep_Reflective_Reader/doc_loaders/pdf_ocr_layout_fixtures.py`; `Deep_Reflective_Reader/scripts/test_pdf_ocr_layout_fixtures.py`.

- [x] Detect and expose native PDF Outline/bookmark structure before OCR, including validated destinations and nesting.
  Evidence: `Deep_Reflective_Reader/doc_loaders/pdf_outline.py`; `Deep_Reflective_Reader/doc_loaders/pdf_document_loader.py`; `Deep_Reflective_Reader/scripts/test_pdf_outline.py`; `许三观卖血记.pdf` container verification.

#### Needs Confirmation

No unresolved confirmation items identified in module checklist.

#### Future Direction Preparation

- [x] Page-aware OCR provenance and stable page boundaries for future TOC detection are implemented in `doc_loaders/module-checklist.md`.
  Notes: Loader evidence carries page index, optional label/page number, source hash, raw-text span, and page text; it remains supporting evidence and does not become parser authority.
- [x] Native PDF Outline/bookmark inspection and the `Outline -> OCR TOC -> keyword matching` precedence chain are implemented and recorded in the corresponding module checklists.
- [x] Universal PDF page-layout evidence metadata is implemented in `doc_loaders/module-checklist.md`.
  Notes: Every page-layout evidence entry can carry source file name, source SHA-256, schema version, dimensions, native text/image metrics, orientation, writing mode, reading order, OCR confidence/text, analysis stage, and bounded evidence reasons. Full region metadata and tiered OCR cost policy remain separate future work.
- [x] Deterministic orientation and reading-order evidence is implemented in `doc_loaders/module-checklist.md`.
  Notes: Page-level layout evidence now records selected rotation degrees when deterministic, preserves competing portrait/landscape hypotheses for ambiguous pages, and exposes confidence for horizontal/vertical writing mode and left-to-right/right-to-left reading order. This remains provenance evidence and does not authorize hierarchy splitting by itself.
- [x] Page-level orientation and reading-order evidence contract is implemented in `doc_loaders/module-checklist.md`.
  Notes: Serialized page-layout evidence now includes dimensions, rotation hypotheses, horizontal/vertical mode, left-to-right/right-to-left order, region count, OCR confidence, word boxes, and schema version. Empty OCR pages retain zero regions; coordinate OCR pages expose bounded region counts as provenance only.
- [x] OCR quality and character provenance contract is implemented in `doc_loaders/module-checklist.md`.
  Notes: OCR token evidence now preserves page index, region id, raw and normalized OCR text, confidence, bounding box, normalization version, TSV hierarchy numbers, and low-confidence/symbol-heavy/repeated-garbage quality flags. Token provenance remains supporting evidence and does not become parser authority.
- [x] Tiered PDF inspection and OCR cost policy is implemented in `doc_loaders/module-checklist.md`.
  Notes: Page-layout evidence distinguishes cheap inventory from coordinate OCR with explicit analysis cost tier, stage, OCR engine/language, render DPI, OCR pass count, cache key, cache hit/miss, failure reason, and high-cost permission metadata. Every page receives cheap inventory evidence; coordinate OCR remains bounded to candidate pages; high-cost analysis is not silently triggered.
- [x] Post-recognition region evidence is implemented in `doc_loaders/module-checklist.md`; canonical region-first OCR remains OCR-DIR work.
  Notes: Page-layout evidence now serializes bounded OCR regions with page-local coordinates, region type, writing mode, reading order, raw OCR, normalized logical text, rotation metadata, confidence, token count, and quality flags. Vertical/right-to-left columns and header/footer bands are retained as provenance regions only; they do not become parser authority or task-layout truth.
  Timestamp: 2026-10-21
- [x] OCR layout and renderer-failure fixtures are implemented in `doc_loaders/module-checklist.md`.
  Notes: Synthetic OCR layout fixtures now define expected orientation, reading order, region types, TOC reconstruction behavior, known limitations, and fallback reasons for `暗水幽灵`, `國富論`, horizontal scans, mixed orientation, circular page numbers, leader lines, and renderer decode failure. These fixtures preserve page/region evidence expectations and conservative rejection behavior without promoting OCR evidence to parser authority.
- [x] Renderer-first PDF page normalization for OCR is implemented and recorded in `doc_loaders/module-checklist.md`.
  Evidence: `Dockerfile`; `Deep_Reflective_Reader/doc_loaders/pdf_document_loader.py`; `docker-compose.yml`; `國富論.pdf` page 5 renderer/OCR verification.
- [x] OCR file-cache persistence is removed from PDF loading; OCR pages are reused only in memory during the active prepare pass and durable OCR output goes through structured-store OCR run persistence.
- [x] Multi-PSM OCR candidate selection is implemented for scanned PDF raw loading; default Tesseract output is no longer accepted solely because it is non-empty.
  Evidence: `Deep_Reflective_Reader/doc_loaders/pdf_document_loader.py`; `Deep_Reflective_Reader/scripts/test_pdf_document_loader_inspection.py`.
- [x] Low-quality OCR raw-load failure gate is implemented and maps through preparation as `load_raw_text_ocr_low_quality:<doc_name>:<reason>`.
  Evidence: `Deep_Reflective_Reader/doc_loaders/document_load_errors.py`; `Deep_Reflective_Reader/document_preparation/document_preparation_pipeline.py`; `Deep_Reflective_Reader/scripts/test_document_preparation_raw_load_errors.py`.
- [x] Multi-PSM raw-text scoring is implemented, but does not establish correct vertical Chinese OCR or reading order; OCR-DIR-01..07 track the confirmed gap.
  Evidence: `Deep_Reflective_Reader/doc_loaders/pdf_document_loader.py`; container OCR verification on `國富論lite`.
  Timestamp: 2026-10-21
- [x] `國富論lite` OCR quality regression is covered by tests and container prepare verification; force rebuild now rejects low-quality OCR before structured hierarchy persistence.
  Evidence: `Deep_Reflective_Reader/scripts/test_pdf_document_loader_inspection.py`; container prepare API verification returned `structured_document_ready=false` with `load_raw_text_ocr_low_quality`.

### `document_preparation/`

- Checklist: `document_preparation/module-checklist.md`
- Detailed Design: `document_preparation/module-detailed-design.md`
- Status: `Guarded Enhancement Workflow Accepted`
- Completed item count: `21`
- Needs confirmation count: `0`
- Unchecked item count: `13` (historical items).
- 2026-10-19: source snapshot validation, structure-only execution and end-to-end
  consent acceptance are 3/3. ENH-PREP-02 delivers the no-extraction candidate path:
  seven candidate tests and ten live disposable PostgreSQL tests pass, including
  coordinator/pipeline success/failure without persistence. ENH-PREP-03 now accepts
  guarded public commits with 30 live disposable DB tests and browser/API/DB flow.
  Profile invalidation is version-aware; post-commit layout preparation is explicit.
  Migration/startup is wired but production deployment was not performed.
  Nine consent and thirteen snapshot regressions pass.
  ENH-PREP-01 internal validator originally passed 13 snapshot tests,
  9 identity tests and the structured-reuse regression; DB-shaped fixtures, not live DB.
  Enhancement no longer invokes forced full preparation. Legacy records without
  accepted extraction identity still require explicit Prepare; no fabricated evidence.
- 2026-10-15 mode integration: `2/2`; PDF-MODE-04/05 identity/reuse and request
  propagation complete. Ten integration cases and existing mode/identity/loader/
  preparation regressions pass. Temporary API 501 gate removed; not deployed.
  JSON and repository-substitute tests do not claim live DB acceptance.
  Timestamp: 2026-10-15

#### Completed Work

- [x] Implements ordered prepare pipeline with profile-before-structured sequencing.
- [x] Supports `base` and `free_qa` preparation modes with explicit mode contract.
- [x] Collects non-blocking profile/enrichment errors while preserving structured readiness semantics.
- [x] Discovers native PDF Outline before OCR/layout TOC analysis and preserves conservative fallback behavior.
  Evidence: `Deep_Reflective_Reader/document_preparation/document_preparation_pipeline.py`; `Deep_Reflective_Reader/document_structure/structured_document_builder.py`; `Deep_Reflective_Reader/scripts/test_pdf_outline.py`.
- [x] Reuses existing structured documents before expensive raw loading in base/common/non-force preparation.
  Evidence: `Deep_Reflective_Reader/document_preparation/document_preparation_pipeline.py`; `Deep_Reflective_Reader/scripts/test_prepare_structured_reuse_before_raw_load.py`; `國富論lite.pdf` container API verification.
- [x] Provides source-agnostic manual structure reparse source evidence handoff.
  Evidence: `Deep_Reflective_Reader/document_preparation/document_preparation_pipeline.py`; `Deep_Reflective_Reader/app/section_task_coordinator.py`; `Deep_Reflective_Reader/scripts/test_manual_structure_preparation_handoff.py`.
- [x] Defines manual structure validation and source-evidence failure behavior before commit persistence.
  Evidence: `Deep_Reflective_Reader/document_preparation/document_preparation_pipeline.py`; `Deep_Reflective_Reader/app/section_task_coordinator.py`; `Deep_Reflective_Reader/scripts/test_manual_structure_commit_source_evidence.py`.
- [x] Provides current-structure anchor evidence handoff for task-layout projection without exposing raw/page text in the task-layout DTO or mutating persisted hierarchy/profile/task-layout state.
  Evidence: `Deep_Reflective_Reader/document_preparation/document_preparation_pipeline.py`; `Deep_Reflective_Reader/app/section_task_coordinator.py`; `Deep_Reflective_Reader/document_structure/structure_anchor_evidence.py`; `Deep_Reflective_Reader/section_tasks/document_task_layout.py`; `Deep_Reflective_Reader/scripts/test_task_unit_content_endpoint.py`; `Deep_Reflective_Reader/scripts/test_task_layout_anchor_evidence_dto.py`.
- [x] Defines preparation handoff for layout hypotheses without making preparation/profile/LLM metadata parser authority.
  Evidence: `Deep_Reflective_Reader/document_preparation/document_preparation_pipeline.py`; `Deep_Reflective_Reader/document_structure/structured_document_builder.py`; `Deep_Reflective_Reader/document_structure/toc_detector.py`; `Deep_Reflective_Reader/scripts/test_pdf_page_evidence.py`; `Deep_Reflective_Reader/scripts/test_toc_projection.py`; `Deep_Reflective_Reader/scripts/test_pdf_outline.py`.
- [x] Defines layout/TOC failure and fallback matrix without persisting partial speculative hierarchy.
  Evidence: `Deep_Reflective_Reader/document_preparation/document_preparation_pipeline.py`; `Deep_Reflective_Reader/document_structure/toc_detector.py`; `Deep_Reflective_Reader/document_structure/structured_document_builder.py`; `Deep_Reflective_Reader/scripts/test_document_preparation_raw_load_errors.py`; `Deep_Reflective_Reader/scripts/test_pdf_page_evidence.py`; `Deep_Reflective_Reader/scripts/test_toc_projection.py`.
- [x] Defines layout failure and user-visible fallback contract without allowing task-layout to persist speculative TOC hierarchy.
  Evidence: `Deep_Reflective_Reader/document_preparation/document_preparation_pipeline.py`; `Deep_Reflective_Reader/document_structure/toc_detector.py`; `Deep_Reflective_Reader/document_structure/structured_document_builder.py`; `Deep_Reflective_Reader/scripts/test_document_preparation_raw_load_errors.py`; `Deep_Reflective_Reader/scripts/test_pdf_page_evidence.py`; `Deep_Reflective_Reader/scripts/test_toc_projection.py`.

#### Needs Confirmation

No unresolved confirmation items identified in module checklist.

#### Future Direction Preparation

- [ ] Added unchecked future tasks for DB-backed structured persistence behavior in the preparation pipeline in `document_preparation/module-checklist.md`.
- [ ] Future direction preserves current file-based prepare outputs during migration.
- [ ] Future storage abstraction boundary must separately account for structured/profile/retrieval artifacts; this pass is documentation/checklist preparation only.
- [ ] Future storage independence note clarifies that preparation should prepare artifacts without permanently assuming file-path persistence as the only possible destination.
- [x] Source-agnostic manual structure reparse preparation handoff is implemented in `document_preparation/module-checklist.md`.
- [x] Manual structure preparation supports raw-text character anchors for any loader-backed document type and optional page/OCR evidence placeholders when available, without making evidence parser authority.
- [x] TOC-aware preparation orchestration and conservative fallback policy are implemented in `document_preparation/module-checklist.md`, with bounded page-evidence handoff and conservative parser fallback.
- [x] Universal page inventory, candidate-page analysis, cache reuse, and cost budgets are captured as unchecked future work in `document_preparation/module-checklist.md`.
  Notes: Documentation only; no new preparation runtime behavior is claimed complete.
- [x] Current-structure anchor evidence handoff for task-layout projection is implemented in `document_preparation/module-checklist.md`.
  Notes: Preparation/app layers provide validated page evidence to the read-only anchor projector so UI prefill can use lightweight page/char anchors without raw text exposure or hidden mutation.
- [x] Preparation handoff for layout hypotheses is implemented in `document_preparation/module-checklist.md`.
  Notes: Preparation supplies page layout evidence, page boundaries, and Outline evidence to the structure builder; structure-layer validation remains responsible for detection, usability, projection authority, and atomic fallback.
- [x] Layout/TOC failure and fallback matrix is implemented in `document_preparation/module-checklist.md`.
  Notes: OCR/raw-load failures surface as preparation errors; TOC/layout failures including missing page anchors, incomplete page numbers, ambiguous orientation, page-order failures, low body-title recall, and failed global projection remain advisory and preserve current parser fallback unless the TOC projection is globally authorized.
- [x] Layout failure and user-visible fallback contract is implemented in `document_preparation/module-checklist.md`.
  Notes: Renderer/OCR raw-load failures, empty/artwork OCR, ambiguous orientation, corrupted page numbers, low-confidence evidence, and incomplete multi-page grouping expose explicit error or provenance reasons while preserving current structure parsing. Task-layout remains projection-only and manual reparse remains the explicit mutation path.

### `db/`

- Checklist: `db/module-checklist.md`
- Detailed Design: `db/module-detailed-design.md`
- Status: `ENH-REV-02 Conditional Commit Delivered`
- Completed item count: `54` (direct checklist count, excluding the template example).
- Needs confirmation count: `0`
- Unchecked item count: `4` (historical items; ENH-REV-02 complete).
- 2026-10-21: ENH-REV-02 implements locked three-way classification, no-op without
  writes, title-only updates preserving IDs/version/interactions and snapshot labels,
  and existing full structural replacement. Delayed metadata saves preserve latest
  labels. Evidence: 43 disposable PostgreSQL cases (42 pass, one opt-in skipped),
  20 comparator tests and manual-reparse repository regression. No new migration.
- Count reconciliation: the previous aggregate omitted six already-checked historical
  items under Future Task Policy; this sync counts them without changing their status.
- 2026-10-19: ENH-DB-01 delivered internally: atomic replacement/cleanup, source and
  interaction revision checks, rollback and stale interaction write guards. All 22
  disposable PostgreSQL tests passed. Migration is not deployed to user schemas;
  PREP-03 accepts the guarded public route and version-aware file-profile projection;
  file hierarchy enhancement remains unsupported. Production deployment is separate.

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
- [x] Persists OCR run pages through PostgreSQL cursor batching.
  Evidence: `Deep_Reflective_Reader/db/postgres_structured_document_store.py`; `國富論lite` `/documents/prepare` container verification; PostgreSQL `ocr_runs` and `ocr_pages` row verification.

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
- Status: `ENH-REF-01/02 Real-Book Locator Recovery Accepted`
- Completed item count: `56`
- Needs confirmation count: `0`
- Unchecked item count: `18` (historical items; ENH-REV-06 and ENH-REF-01/02 complete).
- 2026-10-21 ENH-REF: final real-model implementation passes twice on the reported
  private 147401-character book, each after one validator-guided correction. Exact
  source is preserved; real API/disposable PostgreSQL/layout replay yields 106 units.
  Private corpus is Git-ignored; production book/interactions remain unchanged.
  New follow-up 2/2 (100%); this is scoped case acceptance, not universal reliability.
- 2026-10-21 ENH-REV-04: prompt v8 separates autonomous semantic divisions from
  conservative source-heading claims; generated titles, useful subsection hierarchy
  and legitimate single bodies are supported without quotas. Reference-cap fallback,
  unchanged output allowance and safe accepted-plan diagnostics are verified.
  Ten suites: 145 tests, 144 pass, one private-source skip. No paid generation or
  production mutation; actual model usefulness remains unmeasured.
- 2026-10-20 FULL-02: full-source assembly/resource guards delivered; 80 offline
  tests passed. FULL-05 integrated acceptance passed on 2026-10-20; live model
  reliability remains unverified. See owner checklist/design for reproducible evidence.
- 2026-10-19 NATIVE-06 delivery: this module's new track is `6/6` complete.
  `scripts/accept_native_recovery.py` passes 160 regressions, two opt-in real PDF
  cases with temporary storage, and disposable live PostgreSQL version/rollback/
  artifact invalidation checks. Private-source fast runs invoke no OCR/remote LLM.
  Detailed design records provider/embedding/UI integration and deployment limits.
  This backend aggregate does not count the supporting frontend task.
- 2026-10-19 NATIVE-05 delivery: this module's new track is `5/6` complete.
  V2 plans execute through explicit preparation/reparse with mapped anchors,
  conservative heading gates, full coverage and persisted origins/provenance.
  Rejected enhancement does not save a Common fallback or rebuild profile/artifacts.
  92 tests: 91 pass and one INTERACT-01 expected failure; six standalone regressions
  pass. No live DB/LLM/private-document reparse or deployment; NATIVE-06 remains pending.
- 2026-10-19 NATIVE-04 delivery: this module's new track is `4/6` complete.
  Strict v2 plan/prompt, bounded source references and intent contract are tested:
  80 cases, 78 pass and two existing expected failures; legacy LLM smoke passes.
  Planning-only entry is intentionally separate from the old applier. Generated
  runtime requests reject before source loading/LLM/save until NATIVE-05 activation.
  No live LLM, user-document mutation or deployment is claimed.
- 2026-10-19 NATIVE-03 delivery: this module's new track is `3/6` complete.
  Common uses mapped fragment evidence and exact source slices; typed TOC hints
  reach explicit prepare/reparse and participate in cache identity. Single-body
  output stays valid, absent skips dedicated TOC work, present cannot fabricate.
  68 tests: 66 pass, two follow-up expected failures; four standalone regression
  scripts pass. No private-document reparse, deployment or LLM prompt change.
- 2026-10-19 NATIVE-02 delivery: this module's new track is `2/6` complete.
  Pure bounded source mapping preserves original text/hash/line/page evidence;
  only caller-confirmed intervals allow CJK whitespace repair. Eleven new tests
  pass; combined fast/baseline run has 25 passes and 5 expected failures.
  LLM region-plan/TOC projection smoke tests pass. No consumer integration,
  document repair or deployment; prior snapshots below retain their dated scope.
- 2026-10-19 NATIVE-01 delivery: this module's new track is `1/6` complete.
  Synthetic baseline: 5 passing controls and 5 explicit expected failures; two
  private-source fast diagnostics preserve bytes and report metadata only.
  LLM region-plan and TOC projection smoke tests pass. No runtime repair or
  deployment. The planning-only snapshot below predates this baseline delivery.
- 2026-10-19 governance sync: the 2026-10-17 investigation and six new tasks
  remain planning-only, `0/6` implemented. Detailed design records live read-only
  evidence, source-coverage loss, false headings and LLM fallback/anchor risks.
  Count 37 is a direct checklist recount excluding the completed-task template,
  correcting the stale aggregate of 25; no new completion is claimed. Existing
  historical unchecked tasks and completion states are unchanged.

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
- [x] Adds lightweight document discovery repository contract.
- [x] Rejects LLM split plans that resolve main-body sections to TOC-only spans and falls back before hierarchy persistence.
- [x] Records requested/effective parser mode and LLM fallback reason as advisory `StructuredDocument` parse provenance.
- [x] Normalizes native PDF Outline into the current two-layer `chapters[].sections[]` hierarchy for scanned-image books, including level-0 outline chapters and single-root wrapper outlines.
- [x] Defines deterministic manual structure validation and lightweight preview boundary before commit/persistence.
  Evidence: `Deep_Reflective_Reader/document_structure/manual_structure_projection.py`; `Deep_Reflective_Reader/scripts/test_manual_structure_projection.py`.
- [x] Defines manual structure projection into the current two-layer `chapters[].sections[]` hierarchy.
  Evidence: `Deep_Reflective_Reader/document_structure/manual_structure_document_builder.py`; `Deep_Reflective_Reader/scripts/test_manual_structure_document_builder.py`.
- [x] Defines source-agnostic manual structure override contract.
  Evidence: `Deep_Reflective_Reader/api_schemas.py`; `Deep_Reflective_Reader/main.py`; `Deep_Reflective_Reader/app/section_task_coordinator.py`; `Deep_Reflective_Reader/document_preparation/document_preparation_pipeline.py`; `Deep_Reflective_Reader/document_structure/manual_structure_projection.py`; `Deep_Reflective_Reader/document_structure/manual_structure_document_builder.py`; `Deep_Reflective_Reader/scripts/test_manual_structure_api_schemas.py`; `Deep_Reflective_Reader/scripts/test_manual_structure_validate_route.py`; `Deep_Reflective_Reader/scripts/test_manual_structure_commit_source_evidence.py`.
- [x] Defines manual reparse provenance semantics.
  Evidence: `Deep_Reflective_Reader/document_structure/manual_structure_document_builder.py`; `Deep_Reflective_Reader/scripts/test_manual_structure_document_builder.py`; `Deep_Reflective_Reader/scripts/test_manual_structure_commit_source_evidence.py`.
- [x] Defines PDF structure-source precedence and fallback chain.
- [x] Defines layout-hypothesis normalization before TOC matching while preserving raw OCR/page coordinates and keeping normalized evidence detection-only.
  Evidence: `Deep_Reflective_Reader/document_structure/toc_detector.py`; `Deep_Reflective_Reader/scripts/test_pdf_page_evidence.py`.
- [x] Defines TOC global validation and parser authority boundary with deterministic split-usability gates and atomic fallback.
  Evidence: `Deep_Reflective_Reader/document_structure/toc_detector.py`; `Deep_Reflective_Reader/document_structure/structured_document_builder.py`; `Deep_Reflective_Reader/scripts/test_pdf_page_evidence.py`; `Deep_Reflective_Reader/scripts/test_toc_projection.py`.
- [x] Defines OCR layout regression and negative fixtures for TOC/layout detection without promoting unreliable evidence to parser authority.
  Evidence: `Deep_Reflective_Reader/scripts/test_pdf_page_evidence.py`; `Deep_Reflective_Reader/scripts/test_toc_projection.py`; `Deep_Reflective_Reader/scripts/test_pdf_document_loader_inspection.py`; `Deep_Reflective_Reader/document_structure/toc_detector.py`; `Deep_Reflective_Reader/document_structure/structured_document_builder.py`.
- [x] Defines multi-page TOC grouping and termination rules with explicit page-group provenance.
  Evidence: `Deep_Reflective_Reader/document_structure/toc_detector.py`; `Deep_Reflective_Reader/scripts/test_toc_projection.py`; `Deep_Reflective_Reader/scripts/test_pdf_page_evidence.py`.

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
- [x] Source-agnostic manual structure override is implemented as explicit parser input for any loader-backed document type, not OCR-only and not task-layout mutation.
- [x] `document_structure/module-detailed-design.md` captures the source-agnostic manual structure override boundary: user-supplied structure is explicit parser input for any document type, not OCR-only and not a second hierarchy source.
- [x] Manual structure draft building now produces standard `chapters[].sections[]` hierarchy from char-range anchors, rejects unsupported page-range draft building until page-boundary mapping exists, and keeps provenance advisory.
- [x] Manual structure validation/preview boundary is implemented as a deterministic pure document_structure helper with explicit issue codes and no persistence mutation.
- [x] Deterministic TOC detection contract and atomic projection rejection are implemented in `document_structure/module-checklist.md`; shape projection, page boundary validation, multi-page grouping, and global validation are implemented.
- [ ] TOC-derived hierarchy must remain `chapters[].sections[]`; metadata and LLM classification remain advisory and cannot become parser authority.
- [x] Native PDF Outline normalization for scanned-image books is implemented: use normalized outline chapter level, flatten descendants into sections, map content by PDF page index plus OCR page boundaries, and reject incomplete projections atomically.
- [x] Layout-hypothesis normalization before TOC matching is implemented in `document_structure/module-checklist.md`; normalized logical token/pair evidence preserves page identity, raw OCR, coordinates, rotation/orientation, writing mode, reading order, and order hypothesis without becoming parser authority.
- [x] OCR layout regression and negative fixtures are implemented in `document_structure/module-checklist.md`.
  Notes: Coverage includes vertical RTL ordering, horizontal/circled page-number candidates, OCR-fragmented title candidates without page anchors, artwork-only pages, low-quality OCR scans, ambiguous orientation rejection, non-monotonic page-number rejection, and fallback provenance for layout-looking TOCs that are not projection-safe.
- [x] Multi-page grouping is implemented in `document_structure/module-checklist.md`.
  Notes: Adjacent layout-stable TOC candidates form explicit page groups; artwork-only,正文-like, non-candidate, layout-change, or gap pages terminate grouping, and projection records accepted group spans as provenance.
- [ ] TOC projection implementation details still pending as unchecked work: reliable PDF fixture requirements.
- [x] TOC global validation and parser authority boundary is implemented in `document_structure/module-checklist.md`; detection and OCR/layout evidence remain advisory until deterministic split-usability gates pass.

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
- Status: `FULL-01 Capacity Migration Delivered`
- Completed item count: `4`
- Needs confirmation count: `0`

#### Completed Work

- [x] Defines LLM provider abstraction contract for completion and capability reporting.
- [x] Implements OpenAI-backed provider with model-capability mapping and endpoint routing.
- [x] Implements capability-aware prompt text normalization fallback helper.
- [x] FULL-01 capacity semantics and consumer migration; 66 tests and 10 standalone scripts passed.
  Timestamp: 2026-10-20

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
- Status: `Selective OCR Acceptance Complete; Accuracy Limits Recorded`
- Completed item count: `56`
- Needs confirmation count: `0`
- Unchecked item count: `0` (OCR-08 complete).

#### Completed Work

- [x] Maintains regression script suite for hierarchy, task-layout, artifact persistence, and profile metadata.
- [x] Includes real-document and REST smoke script coverage for end-to-end verification paths.
- [x] Covers profile/metadata and language registry hardening through dedicated regression scripts.
- [x] Covers structured-document reuse before raw-load preparation.
  Evidence: `Deep_Reflective_Reader/scripts/test_prepare_structured_reuse_before_raw_load.py`.
- [x] Covers manual structure API request, validation/preview response, and commit reparse request schema validation.
  Evidence: `Deep_Reflective_Reader/scripts/test_manual_structure_api_schemas.py`.
- [x] Covers manual structure validation route behavior.
  Evidence: `Deep_Reflective_Reader/scripts/test_manual_structure_validate_route.py`.
- [x] Covers deterministic manual structure projection behavior.
  Evidence: `Deep_Reflective_Reader/scripts/test_manual_structure_projection.py`.
- [x] Covers manual structure validation route delegation to projector.
  Evidence: `Deep_Reflective_Reader/scripts/test_manual_structure_validate_route.py`.
- [x] Covers manual structure commit validation gate before persistence.
  Evidence: `Deep_Reflective_Reader/scripts/test_manual_structure_validate_route.py`.
- [x] Covers manual structure commit source evidence gates.
  Evidence: `Deep_Reflective_Reader/scripts/test_manual_structure_commit_source_evidence.py`.
- [x] Covers manual structure StructuredDocument draft building.
  Evidence: `Deep_Reflective_Reader/scripts/test_manual_structure_document_builder.py`.
- [x] Covers manual structure commit draft-build gate.
  Evidence: `Deep_Reflective_Reader/scripts/test_manual_structure_commit_source_evidence.py`.
- [x] Covers manual structure commit persistence gate.
  Evidence: `Deep_Reflective_Reader/scripts/test_manual_structure_commit_source_evidence.py`.
- [x] Covers manual structure commit route success mapping.
  Evidence: `Deep_Reflective_Reader/scripts/test_manual_structure_validate_route.py`.
- [x] Covers page-backed manual structure route status mapping.
  Evidence: `Deep_Reflective_Reader/scripts/test_manual_structure_validate_route.py`.
- [x] Covers manual structure preparation source-evidence handoff.
  Evidence: `Deep_Reflective_Reader/scripts/test_manual_structure_preparation_handoff.py`.
- [x] Covers manual structure commit provenance from preparation evidence.
  Evidence: `Deep_Reflective_Reader/scripts/test_manual_structure_commit_source_evidence.py`.

#### Needs Confirmation

No unresolved confirmation items identified in module checklist.

### `section_tasks/`

- Checklist: `section_tasks/module-checklist.md`
- Detailed Design: `section_tasks/module-detailed-design.md`
- Status: `Interaction Policy Recovery Delivered`
- Completed item count: `82`
- Needs confirmation count: `0`
- Unchecked item count: `7` (historical items; INTERACT-01..03 complete).
- 2026-10-19 INTERACT-03 delivery: `3/3` of this module's new track complete.
  Server-owned prompt/language/guard/source identity gates analysis/quiz reuse and
  exposes stale thinking sessions without deleting history or generating on read.
  Four suites pass 38 cases; 12 standalone regressions pass. Additional legacy
  task-layout availability script fails before cache logic because its fixture has
  only root sections. No live PostgreSQL, paid LLM or deployment acceptance claimed.
- 2026-10-19 INTERACT-02 delivery: `2/3` of this module's new track complete.
  Shared profile-first/document-fallback language instructions cover analysis,
  quiz, thinking questions and evaluation; conservative zh/en mismatch rejection
  retains user answers and uses existing failure states. Twelve new tests, 30
  total unittest cases and 13 standalone regressions pass. No live LLM/deployment
  or cache recovery is claimed; INTERACT-03 remains pending.
- 2026-10-19 INTERACT-01 delivery: `1/3` of this module's new track complete.
  Evidence-backed analysis-view preflight preserves canonical source/IDs, retains
  negative guards and emits bounded metadata-only diagnostics. All 61 unittest
  cases and five standalone validity/service/resolver scripts pass. Both private
  PDFs pass read-only fast-extraction semantic diagnostics; no live generation,
  deployment or stored-artifact recovery is claimed. Language/cache tasks remain open.
- 2026-10-19 governance sync: the 2026-10-17 investigation and three new tasks
  were planning-only at that snapshot, `0/3` implemented. Detailed design separates confirmed
  analysis guard/language defects from unverified quiz/thinking failures and
  specifies explicit stale-artifact recovery. Count 79 is a direct checklist
  recount excluding templates, correcting the stale aggregate of 17; it is not
  new work completed this turn. No UI task count is aggregated in this file.

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
- [x] Preserves section-scoped task-layout ownership for TOC-derived hierarchy: task-layout consumes effective `chapters[].sections[].task_units[]` ownership and does not merge content across sections or create a second ownership model.
- [x] Defines task-layout consumption contract for page/layout-derived hierarchy: task-layout may expose lightweight advisory `anchor_evidence`, but does not reconstruct TOC structure from orientation, reading order, TOC scores, OCR/page text, geometry, root `sections[]`, or `structure_nodes`.

#### Needs Confirmation

No unresolved confirmation items identified in module checklist.

#### Future Direction Preparation

- [ ] Added unchecked future preparation tasks for rich content interaction planning in `section_tasks/module-checklist.md`.
- [ ] Task-layout remains lightweight metadata/projection contract; rich content is future-direction-only and remains on-demand API oriented.
- [ ] Advanced segmented content-block projection semantics (beyond current explicit opt-in endpoint integration) remain future implementation work.
- [ ] Duplicate/missing content-block validation behavior remains future implementation work.
- [ ] Segmented block artifact-target alignment behavior remains future implementation work.
- [ ] Future-direction items above remain planning-only; current additive content-block endpoint integration is captured in completed checklist items.
- [x] Section-scoped task-layout ownership for TOC-derived hierarchy is captured as completed contract work in `section_tasks/module-checklist.md`.
- [x] Task-layout consumption of page/layout-derived hierarchy is captured as completed contract work in `section_tasks/module-checklist.md`; task-layout remains projection-only and does not reconstruct TOC structure.

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
- Status: `Reading Interaction Schema Exposure Audit Captured`
- Completed item count: `17`
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
- [x] Defines source-agnostic manual structure request schemas with typed `char_range` / `page_range` anchors and current `chapter -> section` maximum depth validation.
- [x] Defines source-agnostic manual structure validation/preview response schemas with normalized entries, warnings/errors, lightweight preview chapter/section shape, and parser provenance preview.
- [x] Defines source-agnostic manual structure commit reparse request schema with `parser_mode=manual_structure` gating and required manual plan payload.
- [x] Defines page-backed manual-structure validation/commit schema semantics, with route regressions covering schema-valid `page_range`, missing/unsupported page evidence, out-of-range page anchors, stale source evidence, and successful page-backed commit response mapping.
- [x] Audits reading interaction schema exposure and confirms generic reading target plus analysis/target-agnostic quiz/critical-thinking public schemas are not yet implemented.

#### Needs Confirmation

No unresolved confirmation items identified in module checklist.

#### Future Direction Preparation

- [ ] Added unchecked future tasks for rich task-unit content schema evolution in `api_schemas.module-checklist.md`.
- [ ] API schema future direction keeps hierarchy-first contract and id-based targeting while preserving task-layout lightweight response boundary.
- [ ] Future-direction items above remain planning-only; current additive `content_blocks` schema integration is captured in completed checklist items.
- [x] Source-agnostic manual structure request schema is implemented in `api_schemas.py` and covered by `scripts/test_manual_structure_api_schemas.py`.
- [x] Manual structure validation/preview response schema is implemented in `api_schemas.py` and covered by `scripts/test_manual_structure_api_schemas.py`.
- [x] Manual structure commit reparse request schema is implemented in `api_schemas.py` and covered by `scripts/test_manual_structure_api_schemas.py`.
- [x] Manual structure route orchestration and persistence are implemented outside `api_schemas.py` through the explicit coordinator commit path.
- [x] Page-backed manual-structure schema semantics are covered by `scripts/test_manual_structure_api_schemas.py` and `scripts/test_manual_structure_validate_route.py`; `char_range` remains the universal fallback and task-layout remains lightweight/read-only.

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
- Status: `PDF Mode Contract And Execution Integrated`
- Completed item count: `38` (direct checklist recount corrects the stale 18-item aggregate).
- Needs confirmation count: `0`
- Unchecked item count: `3` (historical items unchanged).
- 2026-10-15 PDF-MODE-01: `1/1`; enum/default, route pass-through, typed failures
  and provenance contract verified by 11 tests. PDF-MODE-05 removes the staged
  501 gate through execution/reuse integration; TXT fast and internal legacy paths
  retain their behavior. No engine readiness or deployment claimed.
  Timestamp: 2026-10-15

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
- [x] Adds non-mutating source-agnostic manual structure validation/preview route.
- [x] Maps schema-valid manual structure reparse commit requests to the manual commit coordinator boundary.
- [x] Delegates manual structure validation route projection checks to `document_structure.manual_structure_projection`.
- [x] Supports manual structure commit through the explicit `/documents/reparse-structure` mutation route.
- [x] Maps page-backed manual-structure validation and commit failures through stable route statuses for missing/unsupported page evidence, stale source evidence, out-of-range/unprojectable page anchors, and successful page-backed commit response mapping.
- [x] Audits reading interaction REST route exposure and confirms generic analysis, target-agnostic quiz, and critical-thinking session route families are not yet exposed.

#### Needs Confirmation

No unresolved confirmation items identified in module checklist.

#### Future Direction Preparation

- [ ] Explicit manual reparse route boundary remains future work in `main.module-checklist.md`.
- [x] Manual structure route keeps validation/preview separate from commit reparse and keeps `/documents/task-layout` projection-only.
- [x] Manual structure validation/preview route is implemented as `POST /documents/manual-structure/validate` without hierarchy persistence, task-layout mutation, profile diagnostics write-back, artifact writes, raw text payload, or reparse execution.
- [x] Manual structure commit route recognizes `parser_mode=manual_structure`, routes through the manual commit coordinator boundary, and maps coordinator statuses without legacy common/LLM reparse dispatch.
- [x] Full manual structure commit orchestration is now available through the explicit reparse mutation path; `/documents/task-layout` remains projection-only.
- [x] Page-backed manual structure route mapping is implemented for the explicit reparse mutation path; `/documents/task-layout` still does not accept TOC edits or trigger hidden reparse.

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

## 13. Selective Per-Page OCR Track (2026-10-14)

Design: [loader design section 16](doc_loaders/module-detailed-design.md#16-selective-per-page-ocr-2026-10-14).
Canonical status is counted once per ID across the three module checklists.

| Task | Status | Owning Checklist |
|---|---|---|
| OCR-01: Typed decision contract | Complete | [doc_loaders](doc_loaders/module-checklist.md#selective-per-page-ocr-track) |
| OCR-02: Cheap page inspection | Complete | doc_loaders |
| OCR-03: Three-way policy | Complete | doc_loaders |
| OCR-04: Native-first layout routing | Complete | doc_loaders |
| OCR-05: Mixed-PDF raw recovery | Complete | doc_loaders |
| OCR-06: Lazy preparation scheduling | Complete | [document_preparation](document_preparation/module-checklist.md#selective-per-page-ocr-track) |
| OCR-07: Reuse, budgets and logs | Complete | doc_loaders |
| OCR-08: End-to-end acceptance | Complete (scoped; limitations recorded) | [scripts](scripts/module-checklist.md#selective-per-page-ocr-track) |

Progress: **8/8 = 100%**, none pending in this track. Governance and supporting tests are not
extra completed tasks. OCR-01 evidence: `doc_loaders/pdf_ocr_decision.py` and
seven passing tests in `scripts/test_pdf_ocr_decision.py`; existing loader
inspection and page-evidence regressions also pass. OCR-02 evidence:
`doc_loaders/pdf_page_inspection.py`, `PdfDocumentLoader.inspect_pages` and
18 passing tests in `scripts/test_pdf_page_inspection.py` (local pypdf 6.9.2 and
isolated Docker pypdf 6.17.0); Outline regression also passes.
OCR-03 evidence: `pdf_ocr_policy.py`, additional stable
decision reasons, and 22 passing policy tests locally and in isolated Docker,
plus the earlier contract/inspection and existing loader inspection regressions.
Rules and probe plans are pure; no visual detector is added.
OCR-04 now wires native-first optional layout routing, with 14 passing tests in
`scripts/test_pdf_layout_routing.py` locally and in isolated Compose. Native/blank
skip, uncertain native fallback, zero/disabled/prefix gates, scan/vertical candidates
and page-local failure recovery are covered; earlier PDF regressions pass.
Read-only local four-page `《歸來》` smoke retained 8,187 native characters and made
zero render/OCR calls in 346.1 ms. This is one layout-method measurement, not
end-to-end acceptance or a deployed speedup; the live API and user artifacts are
unchanged.
OCR-05 shares canonical pages across load/load_pages/boundaries, preserving native
pages and selectively recovering suspect pages with per-page quality, coverage,
prefix-budget and failure-atomicity guards. `test_pdf_raw_recovery.py` passes 25
tests locally and in isolated Compose; earlier PDF, manual-structure and logging
regressions pass. The Outline regression now explicitly models recovery of the
raster cover/trailing page and preserves front-matter anchors before the main chapters.
Read-only `《歸來》` smoke retains 8,200 canonical characters and exact four-page
span parity without OCR. General visual detection/region merging and real-scan
acceptance are not claimed; see loader design section 16.9. No live deployment.
OCR-06 now shares one canonical PDF raw/boundary pass and hands a lazy layout
provider to the structure builder. Validated Outline skips optional coordinate
OCR, while required raw recovery still precedes structure; rejected Outline and
optional failures preserve parser fallback. Source mismatch is rejected as
evidence, not silently reused. Fifteen new scheduling tests pass locally and in
isolated Compose, along with all 86 PDF unittest cases and existing standalone
PDF/TOC, raw-error, structured reuse, profile-ordering, manual and logging scripts.
No live deployment, new persistent cache or end-to-end timing claim.
OCR-07 adds bounded context-local evidence reuse and independent required-text /
optional-layout execution budgets. Actual subprocess attempts and clipped timeouts
are enforced; required exhaustion rejects partial text/hierarchy, optional
exhaustion preserves inventory/fallback. Cache identity includes source/page,
purpose, policy/schema, executable build fingerprints, language/DPI/stage/rotation;
request-correlated logs distinguish decisions, actual calls and cache outcomes
without source text. Factory scopes cover preparation and manual-source reads.
All 111 PDF unittest cases (including 25 new execution tests) pass locally and in
isolated Compose; 15 lazy-layout tests and 11 standalone regression scripts pass
locally. Defaults, identity assumptions and subprocess-only timing boundaries are
documented in loader design section 16.11. No live deployment or latency claim.
OCR-08 completes scoped end-to-end acceptance with `scripts/accept_selective_ocr.py`
and three harness tests: real loaders/OCR/builder/isolated JSON, deterministic
language/profile substitutes, no live HTTP/DB/LLM or user-artifact writes.
All 129 PDF/lazy-preparation/harness tests pass locally and in isolated Compose;
11 standalone regressions pass locally. Real-engine horizontal scan and mixed
native/scan/blank fixtures assert known transcript, exact spans and Outline skip.
Reported four-page `《歸來》` retains exactly 8,200 canonical characters and zero
render/OCR calls; final controlled cold/warm run is 789.9/1.9 ms, layout 5.5 ms.
`暗水幽灵` page 8 completes/reuses structure; `國富論lite` page 1 remains
quality-rejected with no partial save. Detailed settings, counts, timings and
reproduction are in scripts design section 15. Real scan transcription accuracy,
whole scanned books, vertical/rotated accuracy and production latency remain
unverified. No fixed speedup or live deployment claim. Broader historical
preparation roadmap tasks are not marked complete by this eight-task acceptance.

The three affected module summary counts were recounted directly from their
checklists (excluding the literal `<completed task>` template). This is a scoped
sync, not a re-audit of historical checklist claims or unrelated module progress.
No frontend progress is included.
