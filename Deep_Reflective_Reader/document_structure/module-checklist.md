# document_structure Checklist

## Purpose

This checklist records completed, code-confirmed or design-confirmed tasks for the `document_structure` module.

It is used to:
- preserve module-level implementation memory
- reduce hallucination in future Codex tasks
- prevent context-window compression from losing completed work
- track future task completion explicitly

## Source Documents

- `Deep_Reflective_Reader/document_structure/module-detailed-design.md`
- `Deep_Reflective_Reader/proposal.md`
- `Deep_Reflective_Reader/high-level-design.md`
- `Deep_Reflective_Reader/document_structure/`

## Rules

- Only completed work is listed as checked.
- Future work must not be added unless explicitly requested.
- If a new task is added later, it must first be added unchecked.
- Once completed, it must be checked in this file.
- Uncertain items must go to `Needs Confirmation`, not the completed checklist.

## Completed Checklist

- [x] Implements hierarchy-first structured model contracts (`StructuredDocument`, chapter, section) with pure-hierarchy write defaults.
  Evidence: `Deep_Reflective_Reader/document_structure/structured_document.py; Deep_Reflective_Reader/document_structure/module-detailed-design.md (Architecture Constraints)`
  Notes: Root legacy mirrors are not default persistence output.

- [x] Implements hierarchy-first effective indexing helpers and section lookup paths.
  Evidence: `Deep_Reflective_Reader/document_structure/document_hierarchy_index.py; Deep_Reflective_Reader/document_structure/module-detailed-design.md (Key Files)`
  Notes: Effective sections derive from `chapters[].sections[]` runtime contract.

- [x] Implements hierarchy-aware artifact repository with strict hierarchy-required runtime load paths.
  Evidence: `Deep_Reflective_Reader/document_structure/structured_document_artifact_repository.py; Deep_Reflective_Reader/document_structure/module-detailed-design.md (Known Legacy / Compatibility Behavior)`
  Notes: Repository remains write-boundary for section/chapter/task-unit artifacts, and runtime read/write path now requires chapters hierarchy.

- [x] document governance cleanup for hierarchy-first persistence terminology
  Evidence: `Deep_Reflective_Reader/document_structure/module-detailed-design.md (Non-Responsibilities, Architecture Constraints, Terminology Governance Audit)`; `Deep_Reflective_Reader/document_structure/structured_document.py`; `Deep_Reflective_Reader/progress.md`
  Notes: 明確分離 primary hierarchy contract 與 legacy compatibility wording，並補齊 task-layout/diagnostics ownership 邊界描述。

- [x] synchronize unresolved confirmation status after governance cleanup
  Evidence: `Deep_Reflective_Reader/document_structure/module-detailed-design.md (Known Legacy / Compatibility Behavior, Terminology Governance Audit, Terminology Validation Notes)`; `Deep_Reflective_Reader/document_structure/document_hierarchy_index.py`; `Deep_Reflective_Reader/progress.md`
  Notes: 已同步 detailed-design/checklist/progress 的 Needs Confirmation 狀態；本輪後 mirror terminology 與 allow_legacy_fallback 退場項目已完成收斂。

- [x] Clarify artifact governance and hierarchy persistence boundary
  Evidence: `Deep_Reflective_Reader/document_structure/module-detailed-design.md (Architecture Constraints, Artifact Governance Boundary, Known Legacy / Compatibility Behavior)`; `Deep_Reflective_Reader/document_structure/document_artifact_repository.py`; `Deep_Reflective_Reader/document_structure/structured_document_artifact_repository.py`; `Deep_Reflective_Reader/progress.md`
  Notes: 已明確分離 hierarchy truth / artifact output / runtime projection ownership，並固定 artifact 不可反向改寫 hierarchy identity。

- [x] governance consistency closure for compatibility terminology and fallback wording
  Evidence: `Deep_Reflective_Reader/document_structure/module-detailed-design.md (Known Legacy / Compatibility Behavior, Terminology Governance Audit, Terminology Validation Notes)`; maintainer-confirmed governance direction in current task; `Deep_Reflective_Reader/progress.md`
  Notes: `mirror` 已退出正式 architecture terminology，統一改為 `legacy compatibility fields` / `compatibility-only fields`；`allow_legacy_fallback` 明確標記為 compatibility-only，且不得暗示 runtime primary path。

- [x] Audit and retire allow_legacy_fallback legacy helper path
  Evidence: `Deep_Reflective_Reader/document_structure/document_hierarchy_index.py`; `Deep_Reflective_Reader/scripts/test_chapter_hierarchy_primary_model.py`; `Deep_Reflective_Reader/document_structure/module-detailed-design.md`; `Deep_Reflective_Reader/progress.md`
  Notes: 已完成 narrow removal：`find_*_effective` 普通 helper API surface 不再暴露 `allow_legacy_fallback`，lookup 全面 hierarchy-only；sections-only legacy 文檔在 effective helper 路徑不再被解析。

- [x] Remove allow_legacy_fallback API surface and enforce hierarchy-only runtime lookup
  Evidence: `Deep_Reflective_Reader/document_structure/document_hierarchy_index.py`; `Deep_Reflective_Reader/app/section_task_coordinator.py`; `Deep_Reflective_Reader/scripts/test_chapter_hierarchy_primary_model.py`; `Deep_Reflective_Reader/scripts/test_hierarchy_first_task_target_resolution.py`
  Notes: helper signatures 已移除 `allow_legacy_fallback`，app chapter-title runtime lookup 不再回退 root sections，也不再合成 legacy chapter。

- [x] Isolate or remove legacy read compatibility from model/repository boundaries
  Evidence: `Deep_Reflective_Reader/document_structure/structured_document.py`; `Deep_Reflective_Reader/document_structure/structured_document_store.py`; `Deep_Reflective_Reader/document_structure/structured_document_artifact_repository.py`; `Deep_Reflective_Reader/scripts/test_pure_hierarchy_json_cleanup.py`; `Deep_Reflective_Reader/scripts/test_hierarchy_artifact_write_sync.py`; `Deep_Reflective_Reader/scripts/test_task_artifact_persistence.py`
  Notes: normal `from_dict/from_json` 與 repository write path 改為 strict hierarchy-only；legacy sections/structure_nodes 讀取保留於 explicit migration-only loader，不再作 ordinary runtime read source。

- [x] Define hierarchy-aware artifact target validation boundary
  Evidence: `Deep_Reflective_Reader/document_structure/module-detailed-design.md (Future Direction Note: Artifact Target Validation Boundary Preparation)`; `Deep_Reflective_Reader/shared/module-detailed-design.md`; `Deep_Reflective_Reader/section_tasks/module-detailed-design.md`; `Deep_Reflective_Reader/progress.md`
  Notes: 明確收斂 ArtifactTargetRef 為 metadata/target intent（非 persistence truth），並固定 future repository trust boundary 必須先做 hierarchy-aware validation；補齊 stale-ref 語義、allowed target combinations、metadata glossary 與 fail-fast error boundary 的 future-direction 契約。

- [x] Add lightweight document discovery repository contract
  Evidence: `Deep_Reflective_Reader/document_structure/document_artifact_repository.py`; `Deep_Reflective_Reader/document_structure/structured_document_artifact_repository.py`; `Deep_Reflective_Reader/scripts/test_document_list_search_api.py`
  Notes: 新增 `DocumentListItem` 與 `list_documents(query, limit)`；file-backed implementation 掃描 `*.structured.json` 並只讀 title metadata，不把 hierarchy/content 作 list API payload。

- [x] Reject LLM split plans that resolve main-body sections to TOC-only spans
  Evidence: `Deep_Reflective_Reader/document_structure/llm_section_splitter.py`; `Deep_Reflective_Reader/scripts/test_llm_section_splitter_region_plan.py`; `PYTHONPATH=Deep_Reflective_Reader python Deep_Reflective_Reader/scripts/test_llm_section_splitter_region_plan.py`; `python -m py_compile Deep_Reflective_Reader/document_structure/llm_section_splitter.py Deep_Reflective_Reader/scripts/test_llm_section_splitter_region_plan.py`
  Notes: LLM split plan remains advisory; local deterministic validation rejects high-ratio tiny/heading-only `main_body` outputs and falls back to the common splitter instead of persisting a TOC-only hierarchy.

- [x] Record structured parser provenance on accepted StructuredDocument output
  Evidence: `Deep_Reflective_Reader/document_structure/structured_document.py`; `Deep_Reflective_Reader/document_structure/structured_document_builder.py`; `Deep_Reflective_Reader/document_structure/section_splitter_selector.py`; `Deep_Reflective_Reader/document_structure/llm_section_splitter.py`; `Deep_Reflective_Reader/scripts/test_llm_section_splitter_region_plan.py`
  Notes: Structured build records requested/effective parser mode and LLM fallback reason as advisory provenance; fallback validation remains deterministic and metadata does not become parser authority.

- [x] Clear derived interaction artifacts on parser-level hard reparse replacement
  Evidence: `Deep_Reflective_Reader/document_structure/structured_document_artifact_repository.py`; `Deep_Reflective_Reader/db/postgres_structured_document_artifact_repository.py`; `Deep_Reflective_Reader/scripts/test_hard_reparse_reading_interaction_cleanup.py`; `Deep_Reflective_Reader/scripts/test_postgres_manual_reparse_repository_boundary.py`; `.venv` execution of `PYTHONPATH=Deep_Reflective_Reader Deep_Reflective_Reader/.venv/bin/python Deep_Reflective_Reader/scripts/test_hard_reparse_reading_interaction_cleanup.py`; `.venv` execution of `PYTHONPATH=Deep_Reflective_Reader Deep_Reflective_Reader/.venv/bin/python Deep_Reflective_Reader/scripts/test_postgres_manual_reparse_repository_boundary.py`.
  Notes: `save_reparsed_document(...)` now sanitizes accepted replacement documents by clearing document-level, chapter-level, section-level, and task-unit-level task artifacts before persistence. This removes stale analysis/quiz/critical-thinking session payloads and referenced-artifact metadata from replacement structured payloads while leaving ordinary artifact/task-layout saves on their non-reparse paths.
  Timestamp: 2026-10-08

## Needs Confirmation

No unresolved confirmation items identified in this pass.

## Future Task Policy

New future tasks for this module must be added here first as unchecked items:

- [ ] Define hierarchy boundary for rich task-unit content model
- [ ] Prevent content-block model from becoming persisted hierarchy source
- [ ] Define migration strategy for string content compatibility
- [ ] Clarify content-block artifact targeting boundary without mutating chapter/section/task_unit identity
- [ ] Define segmentation boundary against hierarchy persistence
- [ ] Define resegmentation stale-target semantics
- [ ] Define future content-block persistence non-authority rule
- [ ] Define DB-centric structured persistence migration contract
- [ ] Preserve hierarchy-first StructuredDocument semantics across file and DB storage
- [ ] Define file-to-DB migration boundary for structured documents
- [ ] Define DB-backed repository validation rules before switching read path
- [ ] Prevent DB schema from reintroducing root sections[], structure_nodes[], or flat task_units as primary flow
- [ ] Define gradual retirement policy for data/ structured JSON after DB readiness validation
- [ ] Complete Phase 1 StructuredDocument JSONB-first evaluation
- [ ] Validate hierarchy parity criteria defined by the evaluation document
- [ ] Resolve readiness-audit gaps before Phase 1 StructuredDocument JSONB-first evaluation
- [ ] Define DB-era task-unit identity strategy before schema design
- [x] Define source-agnostic manual structure override contract
  Evidence needed: manual structure input is modeled as explicit parser input for any document type, not OCR-only and not task-layout mutation.
  Evidence: `Deep_Reflective_Reader/api_schemas.py`; `Deep_Reflective_Reader/main.py`; `Deep_Reflective_Reader/app/section_task_coordinator.py`; `Deep_Reflective_Reader/document_preparation/document_preparation_pipeline.py`; `Deep_Reflective_Reader/document_structure/manual_structure_projection.py`; `Deep_Reflective_Reader/document_structure/manual_structure_document_builder.py`; `Deep_Reflective_Reader/scripts/test_manual_structure_api_schemas.py`; `Deep_Reflective_Reader/scripts/test_manual_structure_validate_route.py`; `Deep_Reflective_Reader/scripts/test_manual_structure_commit_source_evidence.py`.
  Notes: Manual structure is modeled as explicit parser input for any source handled by the document loader factory, not OCR-only or task-layout mutation. Successful commit produces a normal hierarchy-only `StructuredDocument`; validation/preview remains non-mutating.
- [x] Define manual structure validation and preview boundary
  Evidence needed: validation checks anchors/ranges/levels before commit and returns explicit failure reasons without overwriting current structured artifacts.
  Evidence: `Deep_Reflective_Reader/document_structure/manual_structure_projection.py`; `Deep_Reflective_Reader/scripts/test_manual_structure_projection.py`; `Deep_Reflective_Reader/.venv/bin/python Deep_Reflective_Reader/scripts/test_manual_structure_projection.py`.
  Notes: Validation must reject empty titles, out-of-range anchors, overlapping ranges, empty ranges, incompatible levels, and partial projections. Validation failure must not silently fallback to common parser as a successful manual reparse.
- [x] Define manual structure projection into two-layer hierarchy
  Evidence needed: manual entries map deterministically into `chapters[].sections[]` while preserving original entry levels only as provenance.
  Evidence: `Deep_Reflective_Reader/document_structure/manual_structure_document_builder.py`; `Deep_Reflective_Reader/scripts/test_manual_structure_document_builder.py`; `Deep_Reflective_Reader/.venv/bin/python Deep_Reflective_Reader/scripts/test_manual_structure_document_builder.py`.
  Notes: Manual user-defined structure is limited to `chapter -> section` for now. Chapter-only plans create same-name sections; `level>2` and skipped-level plans fail-fast rather than being folded. No root `sections[]`, `structure_nodes`, flat `task_units`, or `Part -> Chapter -> Section` persistence is introduced. Draft building supports char-range anchors directly and page-range anchors when validated page-boundary evidence is available.
- [x] Define manual reparse provenance semantics
  Evidence needed: accepted manual reparse records requested/effective parser mode, user-supplied source, anchor type, validation summary, and entry count as advisory provenance.
  Evidence: `Deep_Reflective_Reader/document_structure/manual_structure_document_builder.py`; `Deep_Reflective_Reader/scripts/test_manual_structure_document_builder.py`; `Deep_Reflective_Reader/scripts/test_manual_structure_commit_source_evidence.py`.
  Notes: Accepted manual drafts record `requested_parser_mode=manual_structure`, `effective_parser_mode=manual_structure_projection`, `source=user_supplied_structure`, entry/chapter/section counts, source hash, anchor type, and validation summary as advisory `parse_provenance`. Provenance is observability only and does not become a second hierarchy source or parser authority.
- [x] Materialize page_range manual anchors into hierarchy drafts
  Evidence: `Deep_Reflective_Reader/document_structure/manual_structure_document_builder.py`; `Deep_Reflective_Reader/app/section_task_coordinator.py`; `Deep_Reflective_Reader/scripts/test_manual_structure_document_builder.py`; `Deep_Reflective_Reader/scripts/test_manual_structure_commit_source_evidence.py`.
  Notes: Manual draft building now resolves validated `page_range` anchors through preparation-provided page-boundary evidence into raw-text spans and produces hierarchy-only `StructuredDocument` output. Missing page-boundary evidence fails explicitly with `page_boundaries_required`; commit path passes source evidence boundaries into the builder and preserves existing artifacts on invalid drafts.

- [x] Reject ambiguous page-boundary evidence for manual page_range drafts
  Evidence: `Deep_Reflective_Reader/document_structure/manual_structure_document_builder.py`; `Deep_Reflective_Reader/scripts/test_manual_structure_document_builder.py`; `Deep_Reflective_Reader/scripts/test_manual_structure_commit_source_evidence.py`.
  Notes: Manual page-backed draft building now requires unique `page_index` values in preparation-provided page-boundary evidence. Duplicate page indexes fail with `ambiguous_page_boundary` before `StructuredDocument` materialization or repository save, preserving the existing structured artifact and keeping `char_range` as the explicit fallback.

- [x] Project existing structure anchor evidence for TOC edit prefill
  Evidence: `Deep_Reflective_Reader/document_structure/structure_anchor_evidence.py`; `Deep_Reflective_Reader/scripts/test_structure_anchor_evidence.py`.
  Notes: `project_structure_anchor_evidence(...)` projects existing chapter/section hierarchy spans into lightweight read-only anchor evidence. It prefers `page_range` when validated page boundaries cover existing spans, falls back to `char_range` when page evidence is absent, and marks missing/invalid spans as explicit unavailable evidence without changing hierarchy identity or artifact persistence.
- [x] Define deterministic table-of-contents detection contract
  Evidence needed: detection rules, confidence thresholds, evidence fields, and rejection conditions for missing, malformed, or ambiguous TOCs.
  Evidence: `Deep_Reflective_Reader/document_structure/toc_detector.py`; `Deep_Reflective_Reader/document_structure/structured_document_builder.py`; `Deep_Reflective_Reader/scripts/test_pdf_page_evidence.py`; `Deep_Reflective_Reader/scripts/test_toc_projection.py`.
  Notes: Detection and global validation are deterministic parser evidence. Missing page anchors, non-monotonic pages, invalid ranges, missing candidate groups, and insufficient matches reject projection; metadata and LLM classification remain advisory.

- [x] Define PDF structure-source precedence and fallback chain
  Evidence: `Deep_Reflective_Reader/document_preparation/document_preparation_pipeline.py`; `Deep_Reflective_Reader/document_structure/structured_document_builder.py`; `Deep_Reflective_Reader/doc_loaders/pdf_document_loader.py`; `Deep_Reflective_Reader/scripts/test_pdf_outline.py`; `Deep_Reflective_Reader/scripts/test_toc_projection.py`; container verification on `Deep_Reflective_Reader/data/raw/许三观卖血记.pdf`.
  Notes: `/Outlines` with valid destinations has priority over visual TOC inference. A malformed, empty, destination-less, or structurally inconsistent Outline is rejected rather than merged with OCR guesses. OCR TOC may authorize projection only after page/character validation; keyword matching remains the final conservative fallback.
- [x] Define native PDF Outline two-layer normalization for scanned-image books
  Evidence: `Deep_Reflective_Reader/document_structure/structured_document_builder.py`; `Deep_Reflective_Reader/document_structure/structured_hierarchy_builder.py`; `Deep_Reflective_Reader/scripts/test_pdf_outline.py`; container verification with existing native PDF outline fixture and synthetic level-0 outline regression.
  Notes: Native PDF Outline projection now normalizes the chapter level from the minimum outline level, skips a single root wrapper when present, flattens descendants as sections, maps ranges through PDF page indices plus page text boundaries, and rejects incomplete/non-monotonic/empty projections atomically. The resulting hierarchy remains `chapters[].sections[]`; root `sections[]`, `structure_nodes`, and `Part -> Chapter -> Section` persistence were not introduced.
- [x] Define TOC shape classification and hierarchy projection
  Evidence needed: flat chapter, chapter-section, and deeper hierarchy cases map deterministically to `chapters[].sections[]` without reintroducing root `sections[]` or `structure_nodes` as primary flow.
  Evidence: `Deep_Reflective_Reader/document_structure/toc_detector.py`; `Deep_Reflective_Reader/document_structure/structured_document_builder.py`; `Deep_Reflective_Reader/document_structure/structured_hierarchy_builder.py`; `Deep_Reflective_Reader/scripts/test_toc_projection.py`.
  Notes: Flat chapter entries become chapter-level projected sections and are materialized as one chapter with one section; chapter/section entries become one chapter with multiple sections; levels deeper than 2 collapse into the nearest projected section and concatenate content in source order. Original TOC level, projected level, printed page number, source page index, and merge reason are preserved in parse provenance only. Task-unit generation remains downstream of the resulting sections.
- [x] Define TOC-based page/character boundary validation and atomic fallback
  Evidence needed: monotonic page order, fuzzy title matching, non-overlapping spans, empty-range rejection, and all-or-nothing fallback to current parsing when evidence is insufficient.
  Evidence: `Deep_Reflective_Reader/document_structure/toc_detector.py`; `Deep_Reflective_Reader/document_structure/structured_document_builder.py`; `Deep_Reflective_Reader/scripts/test_toc_projection.py`.
  Notes: TOC projection now requires sufficient body-title matches, one compatible printed-page-to-source-page offset hypothesis, resolvable page anchors, and non-empty/monotonic/non-overlapping projected character ranges. If any gate fails, `StructuredDocumentBuilder` leaves the current parser result unchanged and does not emit `toc_projection` or claim `validated_toc_projection`.

- [x] Define universal page-level TOC candidate scoring
  Evidence needed: deterministic scoring covers horizontal/vertical text, left-to-right/right-to-left ordering, missing TOC markers, dotted leaders, separated page-number columns, circled page numbers, and OCR-fragmented titles.
  Evidence: `Deep_Reflective_Reader/document_structure/toc_detector.py`; `Deep_Reflective_Reader/scripts/test_pdf_page_evidence.py`; `Deep_Reflective_Reader/scripts/test_toc_projection.py`.
  Notes: Page-level candidate scoring now covers document-relative early position, horizontal/vertical writing mode, left-to-right/right-to-left ordering, multiple columns, coordinate OCR availability, short-title density, dotted leaders, page-number evidence, separated page-number columns, circled/boxed page numbers, OCR-fragmented title density, and missing-marker cases. Candidate detection remains separate from projection usability; later validation still controls splitting authority.

- [x] Define geometry-first TOC entry reconstruction
  Evidence needed: candidate pages reconstruct entries from title regions, leader lines, page-number regions, and column geometry even when linear OCR text is incomplete.
  Evidence: `Deep_Reflective_Reader/doc_loaders/pdf_page_evidence.py`; `Deep_Reflective_Reader/document_structure/toc_detector.py`; `Deep_Reflective_Reader/scripts/test_pdf_page_evidence.py`.
  Notes: OCR TSV analysis now preserves word-level geometry as page-local evidence. `TableOfContentsDetector.reconstruct_page_entries(...)` reconstructs horizontal TOC rows by aligned title/page-number regions and leader-line endpoint evidence, and reconstructs vertical RTL TOC columns by descending x-coordinate with top-to-bottom title ordering. Circled page numbers are parsed within bounded page-number regions. Reconstruction remains detection-only; later validation still controls projection usability.

- [x] Define logical reading-order normalization with coordinate preservation
  Evidence needed: normalized TOC candidates expose logical title/page pairs while retaining page index, region boxes, raw OCR, rotation, writing mode, and order hypothesis.
  Evidence: `Deep_Reflective_Reader/document_structure/toc_detector.py`; `Deep_Reflective_Reader/scripts/test_pdf_page_evidence.py`.
  Notes: `TableOfContentsDetector.normalize_page_reading_order(...)` now emits logical OCR tokens with page index, logical/group indexes, raw OCR text, normalized text, original boxes, confidence, writing mode, reading order, rotation, and order hypothesis. `reconstruct_normalized_page_pairs(...)` exposes auditable title/page pairs with region boxes, raw OCR sequence, evidence, confidence, and confidence breakdown. The normalized stream remains detection-only and does not replace raw text, page evidence, or hierarchy truth.

- [x] Define TOC-specific OCR quality gates
  Evidence needed: “page looks like TOC” is separated from “entries are reliable enough to split,” using title completeness, page-number coverage, geometric pairing, ordering consistency, and body-title recall.
  Evidence: `Deep_Reflective_Reader/document_structure/toc_detector.py`; `Deep_Reflective_Reader/scripts/test_pdf_page_evidence.py`.
  Notes: `TableOfContentsDetector.evaluate_toc_ocr_quality(...)` now separates `detected` from `usable_for_splitting` using reconstructed entry count, title completeness, page-number coverage, geometric pairing coverage, page-number ordering consistency, and body-title recall. High-scoring TOC-like pages with missing page numbers remain detected but unusable; non-monotonic page numbers are rejected; missing body text prevents usability. The gate preserves explicit rejection reasons and does not infer unreadable page numbers or mutate parser output.

- [x] Define page-number region recognition and global validation
  Evidence needed: Arabic, Chinese, Roman, circled, boxed, multi-digit, and fragmented page numbers map to document anchors under explicit offset hypotheses.
  Evidence: `Deep_Reflective_Reader/document_structure/toc_detector.py`; `Deep_Reflective_Reader/scripts/test_pdf_page_evidence.py`; `Deep_Reflective_Reader/scripts/test_toc_projection.py`.
  Notes: `TableOfContentsDetector.recognize_page_number_regions(...)` now preserves raw bounded page-number tokens, normalized value, numeral system, box, confidence, and evidence for Arabic, Chinese, Roman, circled, boxed, multi-digit, and fragmented OCR digits. `validate_page_number_anchor_offsets(...)` validates explicit printed-page-to-source-page offset hypotheses with monotonicity, plausible page anchors, multiple body-title matches, and atomic ambiguity rejection. This remains validation evidence only and does not mutate hierarchy, task-layout, profile, or artifacts.

- [x] Define OCR layout regression and negative fixtures
  Evidence needed: tests cover `暗水幽靈.pdf` artistic vertical TOC, `國富論.pdf` vertical body page, horizontal/multi-page TOC, circular numbers, artwork-only pages, and low-quality scans.
  Evidence: `Deep_Reflective_Reader/scripts/test_pdf_page_evidence.py`; `Deep_Reflective_Reader/scripts/test_toc_projection.py`; `Deep_Reflective_Reader/scripts/test_pdf_document_loader_inspection.py`; `Deep_Reflective_Reader/document_structure/toc_detector.py`; `Deep_Reflective_Reader/document_structure/structured_document_builder.py`.
  Notes: Regression coverage now fixes vertical RTL ordering, horizontal dotted/circled page-number candidates, OCR-fragmented title candidates without page anchors, artwork-only pages, low-quality OCR scans, ambiguous orientation rejection, non-monotonic page-number rejection, and layout-looking TOC fallback provenance. Projection remains all-or-nothing: visually obvious but unreliable TOC evidence does not emit `validated_toc_projection`, `toc_projection`, root `sections[]`, or `structure_nodes`.

- [x] Define layout-hypothesis normalization before TOC matching
  Evidence: `Deep_Reflective_Reader/document_structure/toc_detector.py`; `Deep_Reflective_Reader/scripts/test_pdf_page_evidence.py`.
  Notes: `normalize_page_reading_order(...)` and `reconstruct_normalized_page_pairs(...)` convert OCR words into logical TOC tokens/pairs while preserving page index, raw OCR text, original coordinates, confidence, writing mode, reading order, rotation/orientation, and order hypothesis. The normalized view remains detection/matching evidence only; raw OCR/page evidence and hierarchy truth are not replaced.

- [x] Define multi-page TOC grouping and termination rules
  Evidence needed: adjacent candidate pages can be grouped into one TOC region, with explicit start/end evidence and rejection when a page is front matter, artwork, or正文.
  Evidence: `Deep_Reflective_Reader/document_structure/toc_detector.py`; `Deep_Reflective_Reader/scripts/test_toc_projection.py`; `Deep_Reflective_Reader/scripts/test_pdf_page_evidence.py`.
  Notes: `group_page_candidates(...)` groups only adjacent layout-stable TOC candidates and records explicit start/end page indices plus evidence. Non-candidate pages, artwork-only pages,正文-like pages, layout changes, or gaps terminate the group; proximity alone is insufficient. Projection validation now requires at least one validated candidate group and records `toc_projection_page_group:<start>-<end>` provenance while preserving atomic fallback.

- [x] Define TOC global validation and parser authority boundary
  Evidence: `Deep_Reflective_Reader/document_structure/toc_detector.py`; `Deep_Reflective_Reader/document_structure/structured_document_builder.py`; `Deep_Reflective_Reader/scripts/test_pdf_page_evidence.py`; `Deep_Reflective_Reader/scripts/test_toc_projection.py`.
  Notes: `validate_for_projection(...)`, `evaluate_toc_ocr_quality(...)`, `validate_page_number_anchor_offsets(...)`, and the builder projection gate distinguish detection from split usability. Acceptance requires sufficient entries, page-number coverage, monotonic pages, plausible page ranges, body-title recall/body-anchor matches, compatible offset validation, non-empty projected ranges, and atomic all-or-nothing projection. Metadata, OCR classification, and LLM suggestions remain advisory; rejected plans leave the current parser result unchanged without `toc_projection`.

- [ ] Prepare reliable TOC PDF fixtures before enabling broad regression coverage
  Evidence needed: born-digital flat TOC, scanned horizontal TOC, vertical right-to-left TOC, multi-page TOC, chapter-section TOC, deeper hierarchy requiring merge, and negative artwork/front-matter cases.
  Notes: Each fixture must define expected TOC pages, printed page numbers, hierarchy levels, page/character ranges, and fallback behavior. `暗水幽靈.pdf` is currently a layout-identification case only because its OCR page-number anchors are unreliable.

After implementation, the task owner must update this checklist and mark the task as completed:

- [x] <completed task>

No coding task should be considered complete unless the corresponding module checklist is updated.

## Maintenance Notes

- This checklist is module memory for completed work.
- It does not replace the module detailed design document.
- It does not replace tests and test evidence.
- It does not replace proposal/HLD decisions and governance context.
