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
- [x] Define deterministic table-of-contents detection contract
  Evidence needed: detection rules, confidence thresholds, evidence fields, and rejection conditions for missing, malformed, or ambiguous TOCs.
  Evidence: `Deep_Reflective_Reader/document_structure/toc_detector.py`; `Deep_Reflective_Reader/document_structure/structured_document_builder.py`; `Deep_Reflective_Reader/scripts/test_pdf_page_evidence.py`; `Deep_Reflective_Reader/scripts/test_toc_projection.py`.
  Notes: Detection and global validation are deterministic parser evidence. Missing page anchors, non-monotonic pages, invalid ranges, missing candidate groups, and insufficient matches reject projection; metadata and LLM classification remain advisory.

- [x] Define PDF structure-source precedence and fallback chain
  Evidence: `Deep_Reflective_Reader/document_preparation/document_preparation_pipeline.py`; `Deep_Reflective_Reader/document_structure/structured_document_builder.py`; `Deep_Reflective_Reader/doc_loaders/pdf_document_loader.py`; `Deep_Reflective_Reader/scripts/test_pdf_outline.py`; `Deep_Reflective_Reader/scripts/test_toc_projection.py`; container verification on `Deep_Reflective_Reader/data/raw/许三观卖血记.pdf`.
  Notes: `/Outlines` with valid destinations has priority over visual TOC inference. A malformed, empty, destination-less, or structurally inconsistent Outline is rejected rather than merged with OCR guesses. OCR TOC may authorize projection only after page/character validation; keyword matching remains the final conservative fallback.
- [ ] Define TOC shape classification and hierarchy projection
  Evidence needed: flat chapter, chapter-section, and deeper hierarchy cases map deterministically to `chapters[].sections[]` without reintroducing root `sections[]` or `structure_nodes` as primary flow.
  Notes: Required mapping: flat chapter entries become one chapter with one section; chapter/section entries become one chapter with multiple sections; levels deeper than 2 collapse into the nearest section and concatenate content in source order. Preserve original TOC level, printed page range, source page indices, and merge reason in parse provenance only. Task-unit generation remains downstream of the resulting sections.
- [ ] Define TOC-based page/character boundary validation and atomic fallback
  Evidence needed: monotonic page order, fuzzy title matching, non-overlapping spans, empty-range rejection, and all-or-nothing fallback to current parsing when evidence is insufficient.
  Notes: Convert printed page numbers to document page anchors using an explicit front-matter offset hypothesis, validate that hypothesis against body-title matches, then map sibling ranges to raw-text offsets. Reject missing or ambiguous anchors, reversed/overlapping ranges, empty ranges, low title recall, and incompatible offsets. Persist either the complete TOC hierarchy or the unchanged current-parser result.

- [ ] Define universal page-level TOC candidate scoring
  Evidence needed: deterministic scoring covers horizontal/vertical text, left-to-right/right-to-left ordering, missing TOC markers, dotted leaders, separated page-number columns, circled page numbers, and OCR-fragmented titles.
  Notes: Evidence includes document-relative position, short-title density, repeated alignment, leader-line/shape evidence, page-number evidence, layout confidence, and body-title matches. `detected` and `usable_for_splitting` are separate decisions.

- [ ] Define geometry-first TOC entry reconstruction
  Evidence needed: candidate pages reconstruct entries from title regions, leader lines, page-number regions, and column geometry even when linear OCR text is incomplete.
  Notes: Vertical RTL pages cluster regions by column and descending x-coordinate. Horizontal pages order rows by y-coordinate and columns by reading direction. Associate title and page number by aligned geometry or leader-line endpoint. Circular/boxed numbers require a bounded number-region OCR pass; linear OCR alone is insufficient.

- [ ] Define logical reading-order normalization with coordinate preservation
  Evidence needed: normalized TOC candidates expose logical title/page pairs while retaining page index, region boxes, raw OCR, rotation, writing mode, and order hypothesis.
  Notes: The normalized stream is detection-only. It must not replace raw text or become a second hierarchy source. Every pair carries evidence and a confidence decomposition for audit.

- [ ] Define TOC-specific OCR quality gates
  Evidence needed: “page looks like TOC” is separated from “entries are reliable enough to split,” using title completeness, page-number coverage, geometric pairing, ordering consistency, and body-title recall.
  Notes: A high-scoring page with missing/corrupt page numbers remains `detected=true, usable=false`. Never infer unreadable page numbers from sequence alone. Preserve rejection reasons and unchanged current-parser output.

- [ ] Define page-number region recognition and global validation
  Evidence needed: Arabic, Chinese, Roman, circled, boxed, multi-digit, and fragmented page numbers map to document anchors under explicit offset hypotheses.
  Notes: Number recognition is region-scoped and may use preprocessing, rotation candidates, and character whitelists. Validate offsets using multiple body-title matches, monotonicity, plausible range, and non-overlapping spans. Reject ambiguous offsets atomically.

- [ ] Define OCR layout regression and negative fixtures
  Evidence needed: tests cover `暗水幽靈.pdf` artistic vertical TOC, `國富論.pdf` vertical body page, horizontal/multi-page TOC, circular numbers, artwork-only pages, and low-quality scans.
  Notes: Expected results include orientation/order, candidate pages, entry coverage, accepted/rejected projection, and fallback provenance. A visually obvious TOC with unreliable extraction is a required negative projection case.

- [ ] Define layout-hypothesis normalization before TOC matching
  Evidence needed: OCR regions can be converted to logical reading order while retaining raw page coordinates and source offsets; horizontal, vertical, rotated, and mixed regions have explicit ordering rules.
  Notes: The normalized view is for detection/matching only. Raw OCR text, quote spans, page identity, and source hashes remain traceable.

- [ ] Define multi-page TOC grouping and termination rules
  Evidence needed: adjacent candidate pages can be grouped into one TOC region, with explicit start/end evidence and rejection when a page is front matter, artwork, or正文.
  Notes: Group contiguous or near-contiguous high-confidence candidates. Start evidence is a TOC marker or repeated title/page-number geometry; continuation requires stable writing mode, column order, title density, and page-number alignment. Terminate on layout change,正文 density, artwork-only page, page-number loss beyond tolerance, or validated body transition. Store group start/end page indices and per-page evidence; proximity alone is insufficient.

- [ ] Define TOC global validation and parser authority boundary
  Evidence needed: acceptance requires title recall, page-order monotonicity, page-range plausibility, body-anchor matches, non-overlapping ranges, and minimum confidence thresholds.
  Notes: Run after grouping and before projection. Require entry count, page-number coverage, monotonic pages, plausible range, normalized body-title recall, unique non-overlapping anchors, and consistent group confidence. `detected` means evidence exists; `usable` means the complete plan passed all checks. Metadata, OCR classification, and LLM suggestions remain advisory.

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
