# app Checklist

## Purpose

This checklist records completed, code-confirmed or design-confirmed tasks for the `app` module.

It is used to:
- preserve module-level implementation memory
- reduce hallucination in future Codex tasks
- prevent context-window compression from losing completed work
- track future task completion explicitly

## Source Documents

- `Deep_Reflective_Reader/app/module-detailed-design.md`
- `Deep_Reflective_Reader/proposal.md`
- `Deep_Reflective_Reader/high-level-design.md`
- `Deep_Reflective_Reader/app/`

## Rules

- Only completed work is listed as checked.
- Future work must not be added unless explicitly requested.
- If a new task is added later, it must first be added unchecked.
- Once completed, it must be checked in this file.
- Uncertain items must go to `Needs Confirmation`, not the completed checklist.

## Completed Checklist

- [x] Expose task-unit content lookup through coordinator boundary
  Evidence: `Deep_Reflective_Reader/app/section_task_coordinator.py`; `Deep_Reflective_Reader/scripts/test_task_unit_content_endpoint.py`
  Notes: 新增 `get_task_unit_content(doc_name, task_unit_id)`，只讀 hierarchy sections/task_units，不走 title/root sections/structure_nodes fallback。

- [x] Pass through rich content blocks in task-unit content coordinator response
  Evidence: `Deep_Reflective_Reader/app/section_task_coordinator.py`; `Deep_Reflective_Reader/section_tasks/document_task_layout.py`; `Deep_Reflective_Reader/scripts/test_task_unit_content_endpoint.py`
  Notes: coordinator 透過 `selected_task_unit.to_content_blocks()` 傳遞 additive `content_blocks`，未改動 hierarchy-only lookup / fail-fast semantics。

- [x] Pass segmented content option through task-unit content coordinator
  Evidence: `Deep_Reflective_Reader/app/section_task_coordinator.py`; `Deep_Reflective_Reader/main.py`; `Deep_Reflective_Reader/api_schemas.py`; `Deep_Reflective_Reader/scripts/test_task_unit_content_endpoint.py`
  Notes: `get_task_unit_content(..., segmented: bool = False)` 新增顯式 passthrough；`segmented=true` 使用 shared segmentation helper，`segmented=false/省略` 保持舊行為；未新增 fallback/hidden mutation/profile write-back。

- [x] Suppress duplicated leading hierarchy title in segmented render blocks
  Evidence: `Deep_Reflective_Reader/app/section_task_coordinator.py`; `Deep_Reflective_Reader/scripts/test_task_unit_content_endpoint.py`; live API validation for `/documents/Madame%20Bovary/task-units/2703/content?segmented=true`
  Notes: coordinator 在 `segmented=true` content projection 中移除首個 block 內與 section/chapter title 相同的 leading line，保留原 `TaskUnit.content` 與 quote-span 回溯語義；不改 task-layout、不寫 persistence、不改 parser hierarchy。

- [x] Project structured parse provenance through task-layout coordinator response
  Evidence: `Deep_Reflective_Reader/app/section_task_coordinator.py`; `Deep_Reflective_Reader/scripts/test_task_unit_content_endpoint.py`; live API validation for `/documents/task-layout` and `/api/documents/task-layout`
  Notes: `get_document_task_layout` exposes advisory `parse_provenance` from the accepted `StructuredDocument`; this is read-only projection and does not control parser behavior or mutate hierarchy/profile state.

- [x] Define manual structure commit coordinator boundary before persistence orchestration
  Evidence: `Deep_Reflective_Reader/app/section_task_coordinator.py`; `Deep_Reflective_Reader/main.py`; `Deep_Reflective_Reader/scripts/test_manual_structure_validate_route.py`
  Notes: Adds app-layer manual structure DTO/result contracts and `commit_manual_structure_reparse(...)` as the explicit mutation entry point separate from task-layout.

- [x] Gate manual structure commit path with document_structure projection validation
  Evidence: `Deep_Reflective_Reader/app/section_task_coordinator.py`; `Deep_Reflective_Reader/document_structure/manual_structure_projection.py`; `Deep_Reflective_Reader/scripts/test_manual_structure_validate_route.py`
  Notes: `commit_manual_structure_reparse(...)` delegates schema-valid manual plans to the deterministic document_structure projector before persistence. Invalid projections return stable 422 without hierarchy persistence, task-layout mutation, profile diagnostics write-back, or artifact writes.

- [x] Gate manual structure commit path with raw source evidence loading
  Evidence: `Deep_Reflective_Reader/app/section_task_coordinator.py`; `Deep_Reflective_Reader/scripts/test_manual_structure_commit_source_evidence.py`
  Notes: After projection validation, `commit_manual_structure_reparse(...)` loads canonical source evidence through the preparation handoff. Missing source returns 404 and stale `source_hash` returns 409 before persistence.

- [x] Build manual structure draft hierarchy after source evidence gates
  Evidence: `Deep_Reflective_Reader/app/section_task_coordinator.py`; `Deep_Reflective_Reader/document_structure/manual_structure_document_builder.py`; `Deep_Reflective_Reader/scripts/test_manual_structure_commit_source_evidence.py`
  Notes: After projection validation, raw source loading, and optional `source_hash` match, `commit_manual_structure_reparse(...)` builds an in-memory hierarchy-only `StructuredDocument` draft through `document_structure/`. Draft validation failures return 422 before persistence; valid drafts continue to the explicit repository save step without task-layout mutation, profile diagnostics write-back, or artifact writes.

- [x] Implements QA orchestration via `QACoordinator` across prepare, retrieval, prompt, and session update paths.
  Evidence: `Deep_Reflective_Reader/app/qa_coordinator.py; Deep_Reflective_Reader/app/module-detailed-design.md (Main Responsibilities)`
  Notes: Coordinator layer exists as application orchestration, not API schema code.

- [x] Implements section/chapter task orchestration via `SectionTaskCoordinator`, including task-layout projection assembly.
  Evidence: `Deep_Reflective_Reader/app/section_task_coordinator.py; Deep_Reflective_Reader/app/module-detailed-design.md (Main Flows)`
  Notes: Task-layout and task execution paths are coordinated in one runtime service.

- [x] Maintains hierarchy-required fail-fast behavior for incompatible runtime structure states.
  Evidence: `Deep_Reflective_Reader/app/section_task_coordinator.py; Deep_Reflective_Reader/app/module-detailed-design.md (Error Semantics)`
  Notes: Legacy masking fallbacks are tightened in runtime coordinator paths.

- [x] Replace legacy chapter-title fallback with fail-fast hierarchy lookup behavior.
  Evidence: `Deep_Reflective_Reader/app/section_task_coordinator.py`; `Deep_Reflective_Reader/scripts/test_hierarchy_first_task_target_resolution.py`
  Notes: chapter_title 查找僅接受 hierarchy chapter；缺失/歧義均 fail-fast，不再回退 root sections 或合成 chapter。

- [x] Orchestrate task-layout anchor evidence for TOC editor prefill
  Evidence: `Deep_Reflective_Reader/app/section_task_coordinator.py`; `Deep_Reflective_Reader/document_structure/structure_anchor_evidence.py`; `Deep_Reflective_Reader/scripts/test_task_unit_content_endpoint.py`.
  Notes: `SectionTaskCoordinator.get_document_task_layout(...)` delegates existing hierarchy-span evidence projection to `document_structure/`, converts it into lightweight `AnchorEvidenceDTO`, and attaches it to chapter/section DTOs without raw text, page text, OCR geometry, hidden mutation, profile write-back, or artifact writes. Public API schema/route mapping is covered by the root-module checkpoints.

- [x] Use preparation page boundaries for task-layout anchor evidence prefill
  Evidence: `Deep_Reflective_Reader/app/section_task_coordinator.py`; `Deep_Reflective_Reader/document_preparation/document_preparation_pipeline.py`; `Deep_Reflective_Reader/document_structure/structure_anchor_evidence.py`; `Deep_Reflective_Reader/scripts/test_task_unit_content_endpoint.py`.
  Notes: `SectionTaskCoordinator.get_document_task_layout(...)` loads optional preparation-owned page boundary evidence and passes it to the document_structure anchor projector. Chapters/sections now prefer lightweight `page_range` anchor evidence when boundaries cover the parsed span, while preserving `char_range` fallback and keeping task-layout read-only without raw/page text exposure or artifact/profile mutation.

- [x] Orchestrate page-backed manual-structure validation and commit
  Evidence: `Deep_Reflective_Reader/app/section_task_coordinator.py`; `Deep_Reflective_Reader/document_structure/manual_structure_document_builder.py`; `Deep_Reflective_Reader/scripts/test_manual_structure_commit_source_evidence.py`.
  Notes: `commit_manual_structure_reparse(...)` accepts schema-valid `page_range` manual anchors, loads preparation-owned page-boundary source evidence, rejects missing/stale/ambiguous evidence before save, and commits only after `document_structure` materializes a valid hierarchy-only draft. Invalid page-backed commits do not save hierarchy, mutate task-layout/profile/artifacts, or fallback to common parser.

## Needs Confirmation

No unresolved confirmation items identified in this pass.

## Future Task Policy

New future tasks for this module must be added here first as unchecked items:

- [ ] Support content-block-level lookup/read API orchestration
- [ ] Support fine-grained interaction target resolution (sentence/paragraph/content-block)
- [ ] Preserve fail-fast hierarchy lookup semantics for content-block interactions
- [ ] Keep task-layout API lightweight and separate from on-demand rich-content read API
- [ ] Route future content-block interactions through explicit id-based targeting and service boundaries
- [ ] Define reading-target resolver orchestration for document/chapter/section/task-unit interactions
  Evidence needed: app-layer resolver accepts `doc_name`, `chapter_id`, `section_id`, and `task_unit_id` targets with optional parent-id consistency checks, and rejects title-primary or legacy fallback targeting.
  Notes: This is the prerequisite for target-agnostic analysis, quiz, and critical-thinking generation.
  Timestamp: 2026-10-05
- [ ] Add analysis artifact read/generate orchestration as the first vertical slice
  Evidence needed: read path returns persisted analysis or missing/not-generated without LLM calls; generate path explicitly builds context, calls LLM service, validates output, and persists the artifact.
  Notes: Must preserve explicit write-path semantics and avoid backend session cache as artifact truth.
  Timestamp: 2026-10-05
- [ ] Add quiz artifact read/generate orchestration through the same target abstraction
  Evidence needed: orchestration passes configured target-level max count and valid quiz type enum to the service, then persists only validated quiz output or insufficient-content status.
  Notes: LLM decides the type mix; backend enforces count/type validation before artifact success.
  Timestamp: 2026-10-05
- [ ] Add critical-thinking session orchestration
  Evidence needed: explicit routes/services can generate a persisted question session, submit one user answer for evaluation, and retry failed evaluation without losing the answer.
  Notes: First version is question -> answer -> evaluation only; no question pool or history browsing requirement.
  Timestamp: 2026-10-05
- [ ] Add future cost and permission gates around generation, refresh, evaluation, and retry write paths
  Evidence needed: app orchestration exposes a single guard point before costly LLM actions and never gates ordinary persisted-artifact reads.
  Notes: Permission policy can evolve later, but the route/service split must leave a clean insertion point.
  Timestamp: 2026-10-05
- [ ] Orchestrate artifact-aware context lookup for higher-level generation
  Evidence needed: app layer can request lower-level artifact summaries scoped to the resolved section/chapter/document target and pass them as secondary context without auto-generating missing child artifacts.
  Notes: Primary source context and secondary artifact context must remain separate through orchestration.
  Timestamp: 2026-10-05
- [ ] Expose artifact-aware generation metadata from orchestration results
  Evidence needed: generated artifact result metadata includes whether lower-level artifacts were referenced, referenced artifact ids, and deduplication/abstraction hint flags.
  Notes: Metadata supports UI observability and future debugging.
  Timestamp: 2026-10-05
- [x] Orchestrate batch task-unit content lookup through a single hierarchy load
  Evidence: `Deep_Reflective_Reader/app/section_task_coordinator.py`; `Deep_Reflective_Reader/main.py`; `Deep_Reflective_Reader/scripts/test_task_unit_content_endpoint.py`; validation with `.venv` task-unit content endpoint regression and `py_compile`.
  Notes: Coordinator batch path resolves the document/hierarchy once, validates ordered unique task-unit ids, returns per-task-unit content in request order, and preserves single endpoint segmented semantics. It replaces frontend selected-section request fan-out with one on-demand content call while remaining read-only and avoiding task-layout heavy payload, profile write-back, parser authority, and legacy fallback paths.
  Timestamp: 2026-10-05
- [x] Prevent ordinary task-layout cache-hit reads from loading manual source evidence
  Evidence: `Deep_Reflective_Reader/app/section_task_coordinator.py`; `Deep_Reflective_Reader/scripts/test_task_layout_persistence_cache.py`; `.venv` execution of `PYTHONPATH=Deep_Reflective_Reader Deep_Reflective_Reader/.venv/bin/python Deep_Reflective_Reader/scripts/test_task_layout_persistence_cache.py`.
  Notes: `SectionTaskCoordinator.get_document_task_layout(...)` now tracks whether task-layout reused an existing cache entry and skips preparation-owned manual source evidence/page-boundary loading on ordinary cache-hit reads unless `include_anchor_page_evidence=True` is explicitly requested for TOC edit-existing prefill. Ordinary cache-hit layout projection remains read-only, falls back to character-range anchor evidence, and avoids hidden scanned-PDF OCR/source loading on reading-page access.
  Timestamp: 2026-10-05
- [x] Orchestrate source-agnostic manual structure validation
  Evidence needed: coordinator-level path validates user-supplied structure for any document type without persisting hierarchy during preview.
  Evidence: `Deep_Reflective_Reader/app/section_task_coordinator.py`; `Deep_Reflective_Reader/document_structure/manual_structure_projection.py`; `Deep_Reflective_Reader/scripts/test_manual_structure_validate_route.py`.
  Notes: The app layer should normalize intent and delegate projection semantics to `document_structure/`; it must not mutate task-layout/profile/artifacts during validation.
- [x] Gate manual structure reparse commit with source evidence checks
  Evidence needed: coordinator-level path rejects missing source evidence and stale source hashes before any hierarchy persistence.
  Evidence: `Deep_Reflective_Reader/app/section_task_coordinator.py`; `Deep_Reflective_Reader/scripts/test_manual_structure_commit_source_evidence.py`.
  Notes: This is a pre-commit gate only. It reads canonical raw text for evidence and returns 404 for missing source and 409 for stale source hash without committing a `StructuredDocument`.
- [x] Orchestrate explicit manual structure reparse commit
  Evidence needed: coordinator-level path commits a validated manual structure as a normal `StructuredDocument` hierarchy and preserves existing structured artifacts on failure.
  Evidence: `Deep_Reflective_Reader/app/section_task_coordinator.py`; `Deep_Reflective_Reader/document_structure/manual_structure_document_builder.py`; `Deep_Reflective_Reader/scripts/test_manual_structure_commit_source_evidence.py`.
  Notes: `commit_manual_structure_reparse(...)` now validates projection, source evidence, source hash, and raw-text-backed draft anchors before saving through `document_artifact_repository.save_document(...)`. Projection/source/draft failures do not call save; repository save failures return 500 without success. The mutation path remains explicit and separate from `/documents/task-layout`.
- [x] Orchestrate page-backed manual-structure validation and commit
  Evidence needed: manual commit path accepts schema-valid `page_range`, loads source page evidence, rejects missing/stale/ambiguous page evidence before save, and commits only after document_structure materializes a valid hierarchy draft.
  Evidence: `Deep_Reflective_Reader/app/section_task_coordinator.py`; `Deep_Reflective_Reader/document_structure/manual_structure_document_builder.py`; `Deep_Reflective_Reader/scripts/test_manual_structure_commit_source_evidence.py`.
  Notes: `char_range` remains fallback. Page-backed failures preserve the current structured artifact and must not silently fallback to common parser.
- [x] Route manual_structure commit through an explicit PostgreSQL hierarchy replacement boundary
  Evidence: `Deep_Reflective_Reader/document_structure/document_artifact_repository.py`; `Deep_Reflective_Reader/db/postgres_structured_document_artifact_repository.py`; `Deep_Reflective_Reader/app/section_task_coordinator.py`; `Deep_Reflective_Reader/scripts/test_manual_structure_commit_source_evidence.py`; `.venv` validation of manual structure commit source evidence tests; `.venv` validation of manual structure route tests; `.venv` `py_compile` for changed Python files.
  Notes: `commit_manual_structure_reparse(...)` now calls `save_reparsed_document(...)`. The default repository implementation delegates to `save_document(...)` for file-backed compatibility, while the PostgreSQL repository override calls `PostgresStructuredDocumentStore.save(..., replace_existing_hierarchy=True)`. Regular PostgreSQL `save_document(...)` remains non-replacement for task-layout/artifact updates. Validation/source/draft failures still occur before persistence.
- [x] Add regression coverage for existing-document manual_structure PostgreSQL replacement
  Evidence: `Deep_Reflective_Reader/scripts/test_postgres_manual_reparse_duplicate_chapter_order.py`; rebuilt Docker API container validation with `docker compose exec -T api python scripts/test_postgres_manual_reparse_duplicate_chapter_order.py`; `Deep_Reflective_Reader/scripts/test_postgres_manual_reparse_repository_boundary.py`.
  Notes: Regression covers an existing PostgreSQL document whose current hierarchy already has `chapter_order=0`, then commits a manual-style replacement hierarchy starting again at order `0`. It verifies hierarchy-first readback, old row replacement, structure-version advancement, hard-reparse event persistence, and rollback/no partial write on forced failure. Separate repository-boundary regression confirms artifact/task-layout non-reparse saves preserve non-replacement semantics.

After implementation, the task owner must update this checklist and mark the task as completed:

No coding task should be considered complete unless the corresponding module checklist is updated.

## Maintenance Notes

- This checklist is module memory for completed work.
- It does not replace the module detailed design document.
- It does not replace tests and test evidence.
- It does not replace proposal/HLD decisions and governance context.
