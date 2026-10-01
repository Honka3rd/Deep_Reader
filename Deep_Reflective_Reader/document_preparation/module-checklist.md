# document_preparation Checklist

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

New future tasks for this module must be added here first as unchecked items:

- [ ] Define preparation pipeline behavior for DB-backed structured persistence
- [ ] Preserve current file-based prepare outputs during migration
- [ ] Define future storage abstraction boundary for structured/profile/retrieval artifacts
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
  Notes: Set deterministic limits for rendered pages, resolution, OCR attempts, and elapsed work. Budget exhaustion produces advisory diagnostics and current-parser fallback, never a partial hierarchy.

- [x] Define layout failure and user-visible fallback contract
  Evidence needed: renderer failure, empty OCR, ambiguous orientation, corrupted page numbers, low confidence, and incomplete multi-page grouping preserve existing structure parsing with explicit provenance.
  Evidence: `Deep_Reflective_Reader/document_preparation/document_preparation_pipeline.py`; `Deep_Reflective_Reader/document_structure/toc_detector.py`; `Deep_Reflective_Reader/document_structure/structured_document_builder.py`; `Deep_Reflective_Reader/scripts/test_document_preparation_raw_load_errors.py`; `Deep_Reflective_Reader/scripts/test_pdf_page_evidence.py`; `Deep_Reflective_Reader/scripts/test_toc_projection.py`.
  Notes: Renderer/OCR raw-load failures surface as explicit preparation errors; empty/artwork OCR, ambiguous orientation, missing or corrupted page numbers, low confidence/body-title recall, and incomplete multi-page grouping remain advisory rejection provenance. Rejected layout/TOC evidence preserves current parser output without `validated_toc_projection`, `toc_projection`, root `sections[]`, or `structure_nodes`. UI may expose the evidence summary, but task-layout does not silently persist speculative TOC hierarchy; manual reparse remains the explicit mutation path.

- [ ] Define end-to-end layout evaluation matrix
  Evidence needed: Outline PDFs, scanned horizontal PDFs, vertical RTL PDFs, mixed-layout PDFs, artistic TOCs, and renderer-failure PDFs are evaluated through prepare, task-layout, and task-unit content APIs.
  Notes: Measure structural accuracy separately from OCR text accuracy: boundary precision, page-anchor validity, hierarchy shape, fallback correctness, UI payload compatibility, latency, and cache reuse. `暗水幽靈.pdf` and `國富論.pdf` are diagnostic fixtures with expected limitations.

After implementation, the task owner must update this checklist and mark the task as completed:

- [x] <completed task>

No coding task should be considered complete unless the corresponding module checklist is updated.

## Maintenance Notes

- This checklist is module memory for completed work.
- It does not replace the module detailed design document.
- It does not replace tests and test evidence.
- It does not replace proposal/HLD decisions and governance context.
