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

- [ ] Define layout/TOC failure and fallback matrix
  Evidence needed: missing page evidence, OCR failure, ambiguous orientation, incomplete page numbers, and low global TOC confidence map to explicit diagnostics while preserving current structure parsing.
  Notes: Every layout/TOC failure maps to an advisory diagnostic and leaves ordinary preparation available. Only raw-text unavailability may block preparation. A partial TOC plan is never persisted as a mixed hierarchy.

- [ ] Define renderer-first and tiered OCR preparation stages
  Evidence needed: every PDF follows bounded stages: native inspection, renderer/image normalization when needed, cheap page inventory, candidate-page layout analysis, region OCR, optional page-number verification, then structure handoff.
  Notes: Native Outline remains first. OCR/layout work is only for documents without an accepted Outline. Expensive region OCR runs only on candidate pages and uses cache keys including source hash, renderer/engine versions, language, DPI, rotation hypotheses, and schema version. No stage mutates canonical raw text.

- [ ] Define preparation handoff for layout hypotheses
  Evidence needed: preparation passes page/region evidence and normalized TOC candidates to structure parsing without promoting metadata, profile output, or LLM classification to parser authority.
  Notes: Distinguish `candidate`, `detected`, and `usable_for_splitting`. Accepted structure is all-or-nothing and requires title recall, page-number mapping, monotonicity, and span-overlap checks.

- [ ] Define OCR/layout cost and observability budget
  Evidence needed: page count, candidate pages, rendered pages, OCR passes, DPI, elapsed time, cache hit/miss, renderer warnings, and rejection reasons are recorded.
  Notes: Set deterministic limits for rendered pages, resolution, OCR attempts, and elapsed work. Budget exhaustion produces advisory diagnostics and current-parser fallback, never a partial hierarchy.

- [ ] Define layout failure and user-visible fallback contract
  Evidence needed: renderer failure, empty OCR, ambiguous orientation, corrupted page numbers, low confidence, and incomplete multi-page grouping preserve existing structure parsing with explicit provenance.
  Notes: UI may expose recommendation/evidence summary, but task-layout must not silently persist speculative TOC hierarchy. Manual reparse remains the explicit mutation path.

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
