# document_preparation Checklist

## Loader Boundary Governance

- [x] LOADER-03: Consume sessions in preparation/cache/save and remove concrete loader dispatch from persisted-evidence reads.
  Acceptance: no PDF/TXT isinstance or private source-path calls in pipeline; generic cache/source checks, lifecycle/profile authority preserved.
  Evidence: source_session-based preparation/reuse/save; generic source_page_spans, legacy API projection, source_capture/source_extract logs. 95 focused Docker tests and PostgreSQL 88 passing cases/two opt-in skips. Source replacement rejects writes. Live default-Common acceptance incident and layout-only recovery are recorded in detailed design.
  Timestamp: 2026-10-21

## Format-Neutral Extraction Backlog (2026-10-21)

Implemented; evidence and scope are recorded per item. Ordered design: [shared plan](module-detailed-design.md#format-neutral-extraction-plan-2026-10-21).

- [x] EXTRACT-04: Persist and reuse generic extraction results during explicit preparation.
  Depends on: EXTRACT-02, EXTRACT-03.
  Acceptance: Store validated generic provenance only after successful complete extraction; exact text/source/settings reuse checks and DB readback; identity changes never trigger profile regeneration.
  Evidence: Generic preparation persistence/reuse and DB canonical-slice validation; single-pass TXT prepare/reuse and no-profile-work tests pass.
  Timestamp: 2026-10-21

- [x] EXTRACT-05: Normalize legacy PDF identity and support explicit verified TXT backfill.
  Depends on: EXTRACT-04.
  Acceptance: Read-only normalization, conflict/unknown-schema rejection, exact source-to-accepted-text proof for TXT, atomic idempotent explicit migration with stable hierarchy/version/interactions/profile; unprovable legacy data requires explicit preparation.
  Evidence: Explicit backfill script and locked provenance-only migration; Madame Bovary migration, PostgreSQL idempotence/rollback/stable hierarchy and artifact tests pass.
  Timestamp: 2026-10-21

- [x] EXTRACT-06: Generalize accepted-source snapshots and enhancement candidate provenance.
  Depends on: EXTRACT-05.
  Acceptance: Remove PDF-only identity/key assumptions; trusted format-aware source lookup, completeness/hash/coverage/race validation and canonical conservation; missing mappings allowed, zero extraction/OCR/profile calls during enhancement.
  Evidence: Generic accepted snapshots/candidate provenance and legacy normalization; exact source/text/coverage checks and snapshot regressions pass.
  Timestamp: 2026-10-21

## Content Read Performance (2026-10-21)

- [x] CONTENT-02: Restrict profile creation/replacement to initial preparation or explicit full reparse; never rebuild on extraction identity or corrupt-cache reads.
  Acceptance: See app detailed design Content Read Recovery; evidence required before completion.
  Evidence: Lifecycle authority/generated flags, locked initial creation, atomic profile replacement and PostgreSQL canonical-source cache validation. Concurrency, corrupt/missing/failure cases pass; real repeated prepare preserves layout and profile hashes.
  Timestamp: 2026-10-21

## Purpose

This checklist records completed, code-confirmed or design-confirmed tasks for the `document_preparation` module.

It is used to:
- preserve module-level implementation memory
- reduce hallucination in future Codex tasks
- prevent context-window compression from losing completed work
- track future task completion explicitly

## Source Documents

- `Deep_Reflective_Reader/document_preparation/module-detailed-design.md`
- `Deep_Reflective_Reader/proposal.md`
- `Deep_Reflective_Reader/high-level-design.md`
- `Deep_Reflective_Reader/document_preparation/`

## Rules

- Only completed work is listed as checked.
- Future work must not be added unless explicitly requested.
- If a new task is added later, it must first be added unchecked.
- Once completed, it must be checked in this file.
- Uncertain items must go to `Needs Confirmation`, not the completed checklist.

## Completed Checklist

- [x] Add preparation stage timing and cache/readiness diagnostics.
  Evidence: `document_preparation_pipeline.py`; passing `scripts/test_upload_preparation_logging.py`, `scripts/test_prepare_structured_reuse_before_raw_load.py`, and `scripts/test_document_preparation_raw_load_errors.py`.
  Notes: Logs stage starts/elapsed times, cache outcomes, storage backend, readiness and error codes without changing preparation ordering, cache policy, failure mapping or artifact contents.
  Timestamp: 2026-10-14

- [x] Implements ordered prepare pipeline with profile-before-structured sequencing.
  Evidence: `Deep_Reflective_Reader/document_preparation/document_preparation_pipeline.py; Deep_Reflective_Reader/document_preparation/module-detailed-design.md (Preparation Lifecycle)`
  Notes: Current lifecycle includes Step 4.5 post-structure enrichment.

- [x] Supports `base` and `free_qa` preparation modes with explicit mode contract.
  Evidence: `Deep_Reflective_Reader/document_preparation/preparation_mode.py; Deep_Reflective_Reader/document_preparation/module-detailed-design.md (Key Files)`
  Notes: Mode behavior is represented in preparation DTOs and flow control.

- [x] Collects non-blocking profile/enrichment errors while preserving structured readiness semantics.
  Evidence: `Deep_Reflective_Reader/document_preparation/document_preparation_pipeline.py; Deep_Reflective_Reader/document_preparation/module-detailed-design.md (Non-Blocking Policy Matrix)`
  Notes: Prepare result preserves error detail instead of hard-failing all paths.

## Needs Confirmation

No unresolved confirmation items identified in this pass.

## Future Task Policy

### Source-Preserving Enhancement (2026-10-19)

New track: this module 3/3; combined 6/6 (100%; backend 5/5, UI 1/1). Execution order and
requirements are in the owning detailed design. Engineering acceptance is complete;
production deployment and live-model semantic quality are not claimed.

- [x] ENH-PREP-01: Resolve and validate an accepted canonical source snapshot without extraction.
  Acceptance: Namespace/file/text hashes, completion, language and exact coverage validated; PostgreSQL empty raw_text is not mistaken for missing source without checking verifiable canonical section slices. Reject gaps/overlap/hash mismatch/unknown legacy identity with actionable readiness reasons; no loader/render/OCR/LLM call, no source mutation. Preserve accepted extraction identity independently of newer parser/prompt versions.
  Evidence required: File and PostgreSQL readback fixtures, changed/missing source, partial OCR, empty text and explicit no-I/O-to-parser guards.
  Evidence: accepted_source_snapshot.py; scripts/test_accepted_source_snapshot.py (13 passed), extraction identity (9 passed) and structured-reuse regression passed locally. JSON storage roundtrip and PostgreSQL-shaped readback covered; no live DB or public API claim. Storage-owner adapter/version handoff remains APP-01; no existing enhance entry changed.
  Timestamp: 2026-10-19

- [x] ENH-PREP-02: Separate structure-only enhancement from full source preparation.
  Depends on: ENH-PREP-01 and ENH-APP-01; public activation also requires ENH-DB-01.
  Acceptance: Existing snapshot -> bounded in-memory Common context -> LLM plan -> local validation; no raw extraction, language/profile generation, OCR or embeddings. Preserve source/extraction identity and accepted state on candidate failure. Explicit re-extraction remains a separate mode-aware operation; no fallback from enhancement. Do not enable an unguarded persistence path while DB consent checks are pending.
  Evidence required: Real coordinator/pipeline integration with loaders/providers spied or forbidden as appropriate, exact text conservation and failed-candidate preservation.
  Evidence: structure_enhancement_candidate.py; pipeline candidate entry and coordinator consent/lease context. Seven candidate tests, ten live disposable PostgreSQL tests, thirteen snapshot tests and nine consent tests passed locally. Real builder/planner/application with deterministic LLM responses; no live provider call or user-document writes. Success/failure leave existing artifacts unchanged; public execution stays gated pending DB-01.
  Timestamp: 2026-10-19

- [x] ENH-PREP-03: Accept the complete confirmed enhancement workflow across entry points and storage.
  Depends on: ENH-PREP-01/02, ENH-APP-01, ENH-DB-01 and ENH-UI-01.
  Acceptance: Real route-to-storage flow verifies no source/OCR re-entry; consent/cancel/stale/new-interaction races; success version advance and cleanup; provider/validation/commit failures preserve data; post-commit layout failure cannot rerun enhancement. Exercise the reported long-PDF failure using safe isolated fixtures and an opt-in source case without mutating production data. Verify supported file/DB behavior or explicit unsupported-backend failure, and deployed API/UI contract parity.
  Evidence required: Disposable PostgreSQL transactions, source/hash assertions, explicit stage/call counts and browser confirmation tests. Report live-provider semantic-quality evidence separately; no completion based only on mocked planner success.
  Evidence: 30 live disposable-schema cases in test_enhancement_postgres_preflight.py, including opt-in reported PDF copied read-only and explicitly prepared in isolation; all 8,190 canonical characters preserved, no extraction during enhancement. accept_enhancement_browser.py passes real browser/FastAPI/PostgreSQL flow at 1280px/390px, with deterministic planner/unit resolver. Consent/candidate/layout regressions, container wiring, UI browser confirmation, build and shell syntax pass.
  Delivery: guarded public commit and committed-version response; file-profile derived-field invalidation from atomic DB marker; explicit post-commit layout preparation; migration 003 mounted for fresh volumes and applied by local startup for existing volumes. File hierarchy enhancement remains explicitly unsupported.
  Boundary: runtime API/UI parity verified in isolated served fixtures, not the user's running Docker instance. No production books, profile files or schema changed; no paid model call. The legacy reported book requires explicit Prepare to establish accepted extraction identity. Deployment remains an operator action using the existing scripts.
  Timestamp: 2026-10-19

New future tasks for this module must be added here first as unchecked items:

- [ ] Define preparation pipeline behavior for DB-backed structured persistence
- [ ] Preserve current file-based prepare outputs during migration
- [ ] Define future storage abstraction boundary for structured/profile/retrieval artifacts
- [ ] Define raw source upload intake contract.
  Evidence needed: preparation/upload intake validates supported source type, normalized filename/doc_name, source size, source hash, and storage result before returning a document candidate.
  Notes: Upload intake must not run language detection, profile, structured parsing, OCR, retrieval build, task-layout, or interaction generation.
  Timestamp: 2026-10-13
- [ ] Define upload conflict and source-hash policy.
  Evidence needed: duplicate raw source handling is explicit and source hash is available for later prepare/reparse/stale-source checks.
  Notes: Silent overwrite is not allowed without an app-level conflict policy.
  Timestamp: 2026-10-13
- [ ] Define upload-to-prepare handoff tests.
  Evidence needed: uploaded raw candidates can be prepared only through explicit prepare/reparse flows, and failed upload leaves no partial prepared artifacts.
  Notes: Upload success means source availability, not structured/profile/retrieval readiness.
  Timestamp: 2026-10-13
- [x] Define source-agnostic manual structure reparse preparation handoff
  Evidence needed: preparation can provide raw text, language, source identity, and optional page boundaries/provenance to manual structure validation/reparse without assuming OCR/PDF-only input.
  Evidence: `Deep_Reflective_Reader/document_preparation/document_preparation_pipeline.py`; `Deep_Reflective_Reader/app/section_task_coordinator.py`; `Deep_Reflective_Reader/scripts/test_manual_structure_preparation_handoff.py`; `Deep_Reflective_Reader/scripts/test_manual_structure_commit_source_evidence.py`.
  Notes: `load_manual_structure_source_evidence(...)` provides normalized source identity, raw text, source hash, best-effort language, and optional page boundaries without building profile/structured/FAISS artifacts. Character-span anchors work when page evidence is unavailable; page/OCR evidence remains supporting data, not parser authority.
- [x] Define manual structure validation failure behavior in preparation
  Evidence needed: failed manual validation/reparse reports explicit errors and preserves the current structured artifact.
  Evidence: `Deep_Reflective_Reader/document_preparation/document_preparation_pipeline.py`; `Deep_Reflective_Reader/app/section_task_coordinator.py`; `Deep_Reflective_Reader/scripts/test_manual_structure_preparation_handoff.py`; `Deep_Reflective_Reader/scripts/test_manual_structure_commit_source_evidence.py`; `Deep_Reflective_Reader/scripts/test_manual_structure_validate_route.py`.
  Notes: Manual source-evidence failure is reported before save, projection/draft failures return explicit errors, and preview/validation do not persist hierarchy. Failure does not silently fallback to common parser as a successful manual reparse.
- [x] Define TOC-aware preparation orchestration and conservative fallback policy
  Evidence needed: preparation can pass page-aware source evidence to structure parsing, preserve advisory detection provenance, and use the current parser unchanged when TOC confidence is insufficient.
  Evidence: `Deep_Reflective_Reader/document_preparation/document_preparation_pipeline.py`; `Deep_Reflective_Reader/document_structure/structured_document_builder.py`; `Deep_Reflective_Reader/document_structure/toc_detector.py`; `Deep_Reflective_Reader/scripts/test_pdf_page_evidence.py`; container verification on `暗水幽灵.pdf`.
  Notes: Preparation passes bounded PDF page evidence into advisory TOC detection provenance. Uncertain detection does not authorize a split and leaves the existing parser path unchanged; profile/LLM write-back remains outside this path.

- [x] Define ordered PDF structure discovery in preparation
  Evidence: `Deep_Reflective_Reader/document_preparation/document_preparation_pipeline.py`; `Deep_Reflective_Reader/document_structure/structured_document_builder.py`; `Deep_Reflective_Reader/scripts/test_pdf_outline.py`; container verification on `Deep_Reflective_Reader/data/raw/许三观卖血记.pdf`.
  Notes: The precedence chain is `native_outline -> validated_ocr_toc -> current_keyword_heading_parser`. A lower-priority source must not overwrite an accepted higher-priority hierarchy. All rejected-source reasons remain advisory provenance, and raw text remains unchanged.

- [x] Reuse existing structured documents before expensive raw loading in base preparation
  Evidence: `Deep_Reflective_Reader/document_preparation/document_preparation_pipeline.py`; `Deep_Reflective_Reader/scripts/test_prepare_structured_reuse_before_raw_load.py`; container verification with `Deep_Reflective_Reader/data/raw/國富論lite.pdf`.
  Notes: `base` preparation now validates and reuses an existing structured document before raw text loading when `force_rebuild=false` and parser mode is common. This prevents scanned PDFs from entering OCR on repeated `prepare-task-layout` or content-read preparation paths while preserving force rebuild and LLM enhanced reparse behavior.

- [ ] Define universal PDF layout-analysis preparation stages
  Evidence needed: preparation runs page inventory, cheap layout inspection, candidate-page OCR, optional high-cost verification, and structure handoff in deterministic order for every PDF type.
  Notes: Planned order: inventory every page, normalize layout hypotheses, score candidates, group candidates, run optional high-cost verification, perform global TOC validation, then hand evidence to structure parsing. Native-text and scanned PDFs share one evidence contract; OCR is used only when required. No stage may mutate raw text.

- [ ] Define preparation cost budgets and cache reuse
  Evidence needed: page count, candidate count, OCR passes, image resolution, elapsed time, and cache hits/misses are recorded; valid page evidence is reused across repeated preparation.
  Notes: Record page/candidate/group counts, OCR passes, resolution, elapsed time, cache hit/miss, and rejection reason. Invalidate on source PDF hash, engine/version, language set, analysis stage, rotation hypotheses, or schema version. Until evidence cache and cost metrics exist, this remains planning-only.

- [x] Define layout/TOC failure and fallback matrix
  Evidence needed: missing page evidence, OCR failure, ambiguous orientation, incomplete page numbers, and low global TOC confidence map to explicit diagnostics while preserving current structure parsing.
  Evidence: `Deep_Reflective_Reader/document_preparation/document_preparation_pipeline.py`; `Deep_Reflective_Reader/document_structure/toc_detector.py`; `Deep_Reflective_Reader/document_structure/structured_document_builder.py`; `Deep_Reflective_Reader/scripts/test_document_preparation_raw_load_errors.py`; `Deep_Reflective_Reader/scripts/test_pdf_page_evidence.py`; `Deep_Reflective_Reader/scripts/test_toc_projection.py`.
  Notes: Raw-text/OCR unavailability maps to explicit preparation errors; missing page anchors, missing/incomplete page numbers, ambiguous orientation, page-order failures, low body-title recall, and failed global TOC projection remain advisory detection/provenance reasons. Structure building preserves the current parser result unless a TOC projection is globally authorized, so partial TOC plans are not persisted as mixed hierarchy.

- [ ] Define renderer-first and tiered OCR preparation stages
  Evidence needed: every PDF follows bounded stages: native inspection, renderer/image normalization when needed, cheap page inventory, candidate-page layout analysis, region OCR, optional page-number verification, then structure handoff.
  Notes: Native Outline remains first. OCR/layout work is only for documents without an accepted Outline. Expensive region OCR runs only on candidate pages and uses cache keys including source hash, renderer/engine versions, language, DPI, rotation hypotheses, and schema version. No stage mutates canonical raw text.

- [x] Define preparation handoff for layout hypotheses
  Evidence: `Deep_Reflective_Reader/document_preparation/document_preparation_pipeline.py`; `Deep_Reflective_Reader/document_structure/structured_document_builder.py`; `Deep_Reflective_Reader/document_structure/toc_detector.py`; `Deep_Reflective_Reader/scripts/test_pdf_page_evidence.py`; `Deep_Reflective_Reader/scripts/test_toc_projection.py`; `Deep_Reflective_Reader/scripts/test_pdf_outline.py`.
  Notes: Preparation loads bounded page-layout evidence, page text boundaries, and native Outline evidence before structured build, then hands them to `StructuredDocumentBuilder` without mutating raw text or promoting profile/metadata/LLM classification to parser authority. `document_structure` owns candidate/detected/usable semantics, all-or-nothing projection, title recall, page-number mapping, monotonicity, and span-overlap checks.

- [x] Extend manual-structure source evidence with validated page boundaries
  Evidence: `Deep_Reflective_Reader/document_preparation/document_preparation_pipeline.py`; `Deep_Reflective_Reader/scripts/test_manual_structure_preparation_handoff.py`; `Deep_Reflective_Reader/scripts/test_manual_structure_commit_source_evidence.py`.
  Notes: `load_manual_structure_source_evidence(...)` now loads optional page-boundary evidence from pageable loaders, validates page indices, monotonic raw-text ranges, range bounds, page text matching, and preserves optional page labels. Invalid or unavailable page evidence is reported in `errors` and dropped so `char_range` fallback remains available.

- [x] Provide current-structure anchor evidence handoff for task-layout projection
  Evidence: `Deep_Reflective_Reader/document_preparation/document_preparation_pipeline.py`; `Deep_Reflective_Reader/app/section_task_coordinator.py`; `Deep_Reflective_Reader/document_structure/structure_anchor_evidence.py`; `Deep_Reflective_Reader/section_tasks/document_task_layout.py`; `Deep_Reflective_Reader/scripts/test_task_unit_content_endpoint.py`; `Deep_Reflective_Reader/scripts/test_task_layout_anchor_evidence_dto.py`.
  Notes: Preparation-owned page-boundary evidence is retrieved by the app-layer task-layout path and handed to `document_structure` for existing hierarchy anchor projection. The public DTO carries lightweight `anchor_evidence` metadata only; it does not include raw text/page text, mutate profile/task-layout/structured hierarchy, or make page evidence parser authority.

- [ ] Define OCR/layout cost and observability budget
  Evidence needed: page count, candidate pages, rendered pages, OCR passes, DPI, elapsed time, cache hit/miss, renderer warnings, and rejection reasons are recorded.
  Notes: Set deterministic limits for rendered pages, resolution, OCR attempts, and elapsed work. Optional layout budget exhaustion permits advisory current-parser fallback; required canonical OCR text budget exhaustion fails preparation, never a partial hierarchy. Fast forbids renderer/OCR use; coordinate mode-aware logging with PDF-MODE-05 without duplicating its delivery.
  Timestamp: 2026-10-15

- [x] Define layout failure and user-visible fallback contract
  Evidence needed: renderer failure, empty OCR, ambiguous orientation, corrupted page numbers, low confidence, and incomplete multi-page grouping preserve existing structure parsing with explicit provenance.
  Evidence: `Deep_Reflective_Reader/document_preparation/document_preparation_pipeline.py`; `Deep_Reflective_Reader/document_structure/toc_detector.py`; `Deep_Reflective_Reader/document_structure/structured_document_builder.py`; `Deep_Reflective_Reader/scripts/test_document_preparation_raw_load_errors.py`; `Deep_Reflective_Reader/scripts/test_pdf_page_evidence.py`; `Deep_Reflective_Reader/scripts/test_toc_projection.py`.
  Notes: Renderer/OCR raw-load failures surface as explicit preparation errors; empty/artwork OCR, ambiguous orientation, missing or corrupted page numbers, low confidence/body-title recall, and incomplete multi-page grouping remain advisory rejection provenance. Rejected layout/TOC evidence preserves current parser output without `validated_toc_projection`, `toc_projection`, root `sections[]`, or `structure_nodes`. UI may expose the evidence summary, but task-layout does not silently persist speculative TOC hierarchy; manual reparse remains the explicit mutation path.

- [ ] Define end-to-end layout evaluation matrix
  Evidence needed: Outline PDFs, scanned horizontal PDFs, vertical RTL PDFs, mixed-layout PDFs, artistic TOCs, and renderer-failure PDFs are evaluated through prepare, task-layout, and task-unit content APIs.
  Notes: Measure structural accuracy separately from OCR text accuracy: boundary precision, page-anchor validity, hierarchy shape, fallback correctness, UI payload compatibility, latency, and cache reuse. `暗水幽靈.pdf` and `國富論.pdf` are diagnostic fixtures with expected limitations.

- [x] Add prepare reuse regression for existing structured documents on read-like flows
  Evidence: `Deep_Reflective_Reader/scripts/test_prepare_structured_reuse_before_raw_load.py`; validation: `docker compose exec api python scripts/test_prepare_structured_reuse_before_raw_load.py`.
  Notes: Regression now covers both `prepare(base)` and `prepare_and_load(base)`. When `force_rebuild=false`, parser mode is common, and a valid structured document already exists, the pipeline does not call the raw loader, language detector, profile store/builder, or structured builder before returning the existing structured document.

- [ ] Define language detection cache source priority
  Evidence needed: language detection receives the storage/config context required to reuse profile or retrieval-record language evidence before falling back to LLM detection.
  Notes: The intended priority is existing profile/records/source metadata first, deterministic fallback second, LLM fallback last. Repeated UI document selection should not call LLM solely because language cache context was unavailable.

- [ ] Add observability for prepare cache-boundary misses
  Evidence needed: prepare logs expose reason codes for structured artifact miss, source hash mismatch, parser/schema version mismatch, profile cache miss, language cache miss, forced rebuild, and storage-backend mismatch.
  Notes: If `task_layout_cache_hit` appears before OCR or LLM work for the same document request, logs should make the cache boundary that failed explicit.

After implementation, the task owner must update this checklist and mark the task as completed:

- [x] <completed task>

No coding task should be considered complete unless the corresponding module checklist is updated.

## Selective Per-Page OCR Track

- [x] OCR-06: Schedule Outline-first preparation with lazy layout evidence.
  Acceptance: Reuse valid structure first; share inspected page text/boundaries; defer expensive layout until Outline acceptance/fallback is known through structure-owned validation. Accepted Outline skips optional coordinate OCR, never required body-text recovery. Missing/rejected Outline and optional layout failures preserve parser fallback; no extra raw/LLM calls on reuse.
  Evidence: `document_preparation_pipeline.py`; `doc_loaders/pdf_document_loader.py`; `document_structure/structured_document_builder.py`; `scripts/test_prepare_lazy_pdf_layout.py` (15 tests passed locally and in isolated Compose); 86 existing PDF unittest cases and standalone Outline/TOC/page-evidence, raw-error, structured-reuse, profile-ordering, manual source-evidence/handoff and logging regressions passed on 2026-10-14.
  Scope: Request-local combined raw/boundary handoff; one structure-owned acceptance/build and lazy fallback provider, source mismatch rejection, preserved required recovery/cache/force semantics, content-free scheduling diagnostics. No API schema, new durable cache, live deployment or full performance acceptance. Broader reuse/budgets remain OCR-07.
  Depends on: OCR-05 in `../doc_loaders/module-checklist.md`; next is OCR-07 there. Design: section 21 and loader design section 16. One of eight unique OCR tasks, not a duplicate of the broader historical layout roadmap.
  Timestamp: 2026-10-14

## PDF Mode Integration (2026-10-15)

New track: 0/2 implemented. Existing OCR-06 completion remains historical.

- [x] PDF-MODE-04: Make preparation provenance and reuse extraction-mode-aware.
  Acceptance: Define source fingerprint + mode + extraction policy/settings version for raw/OCR/structured reuse and derived profile/layout/retrieval dependencies. Different modes or unknown legacy provenance cannot silently satisfy a new explicit mode request. Existing artifacts remain readable; only explicit preparation can rebuild, using existing validated hierarchy replacement/invalidation boundaries, no premature deletion on failed extraction. Cover same-mode reuse, fast-to-ocr/auto switches, legacy artifacts, source changes and failure preservation in file/DB-backed paths as applicable.
  Depends on: PDF-MODE-01 contract; coordinate repository owners without making mode a hierarchy identity authority.
  Evidence: pdf_extraction_identity.py; prepared_document_assets.py; preparation structured reuse/save and derived-resource hooks; test_pdf_extraction_identity.py (9 tests), structured reuse, lazy layout (15) and API contract (11) regressions passed. JSON persistence tested; DB-shaped repository substituted and existing PostgreSQL parse_provenance mapping inspected, not live DB-tested.
  Boundary: Hooks/identity contract delivered; PDF-MODE-05 supplies actual request snapshot and extraction result. Public PDF prepare stays gated. Derived resources conservatively rebuild when identity-bearing, no new raw cache or hierarchy identity; failed/missing extraction cannot delete existing resources. No deployment or cross-store transaction guarantee.
  Timestamp: 2026-10-15

- [x] PDF-MODE-05: Propagate the selected mode through every preparation/evidence path.
  Acceptance: Carry immutable request-scoped mode from both explicit endpoints through factory/raw/page-boundary/lazy-layout calls. Fast cannot render/OCR via optional TOC fallback; ocr cannot silently switch to native; auto uses existing selective policy. Log mode, stage timings and render/OCR counts without content; concurrent requests cannot leak modes. Persist successful provenance and return truthful readiness; failed extraction cannot publish partial structure. Regression covers defaults, cache hits/misses, optional providers and request cancellation/late completion semantics without claiming server cancellation.
  Depends on: PDF-MODE-01..04 and PDF-FIX-02; coordinates LAYOUT-02 without duplicating that task's read/write-boundary fix.
  Evidence: preparation snapshot/identity/result flow; loader fast/ocr/auto dispatch; result-bound optional providers; stored page spans/manual evidence reuse; test_pdf_mode_preparation.py (10 cases), mode contract (11), identity (9), fast (9), full OCR (7), lazy layout (15), raw recovery (25), execution pass (25), raw errors and structured reuse regressions passed.
  Boundary: Temporary API 501 gate removed in code, not deployed. Auto retains existing OCR configuration and graphics limitations. Tests use temporary sources and simulated OCR/storage/LLM dependencies; live DB/engine acceptance remains PDF-FIX-03. Client cancellation is not server cancellation; no new job or same-document serialization guarantee.
  Timestamp: 2026-10-15

## Maintenance Notes

OCR-07 supporting evidence (counted only in the loader checklist): preparation
and manual-source reads now open the factory's bounded PDF scope, preserving
structured reuse, read-only manual handoff and existing persistence policy.
`test_pdf_execution_pass.py` covers actual factory integration, required-budget
failure without partial saves and optional-budget fallback. All 25 new tests pass
locally/in isolated Compose on 2026-10-14, alongside existing preparation/manual/
logging regressions. Design section 21 records the scope lifecycle; broader
historical layout/TOC roadmap items remain open.

- This checklist is module memory for completed work.
- It does not replace the module detailed design document.
- It does not replace tests and test evidence.
- It does not replace proposal/HLD decisions and governance context.
