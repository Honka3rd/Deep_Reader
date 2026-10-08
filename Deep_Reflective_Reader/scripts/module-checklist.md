# scripts Checklist

## Purpose

This checklist records completed, code-confirmed or design-confirmed tasks for the `scripts` module.

It is used to:
- preserve module-level implementation memory
- reduce hallucination in future Codex tasks
- prevent context-window compression from losing completed work
- track future task completion explicitly

## Source Documents

- `Deep_Reflective_Reader/scripts/module-detailed-design.md`
- `Deep_Reflective_Reader/proposal.md`
- `Deep_Reflective_Reader/high-level-design.md`
- `Deep_Reflective_Reader/scripts/`

## Rules

- Only completed work is listed as checked.
- Future work must not be added unless explicitly requested.
- If a new task is added later, it must first be added unchecked.
- Once completed, it must be checked in this file.
- Uncertain items must go to `Needs Confirmation`, not the completed checklist.

## Completed Checklist

- [x] Maintains regression script suite for hierarchy, task-layout, artifact persistence, and profile metadata.
  Evidence: `Deep_Reflective_Reader/scripts/test_task_layout_hierarchy_first_read.py; Deep_Reflective_Reader/scripts/test_post_structure_metadata_enrichment.py; Deep_Reflective_Reader/scripts/module-detailed-design.md (Key Files)`
  Notes: Scripts folder acts as primary test entry surface in current repo state.

- [x] Includes real-document and REST smoke script coverage for end-to-end verification paths.
  Evidence: `Deep_Reflective_Reader/scripts/test_rest_structured_parser_modes.py; Deep_Reflective_Reader/scripts/test_rest_dynamic_context.sh; Deep_Reflective_Reader/scripts/module-detailed-design.md (Main Responsibilities)`
  Notes: Smoke scripts validate runtime behavior beyond unit-style checks.

- [x] Covers profile/metadata and language registry hardening through dedicated regression scripts.
  Evidence: `Deep_Reflective_Reader/scripts/test_document_profile_parser_metadata.py; Deep_Reflective_Reader/scripts/test_language_script_registry.py; Deep_Reflective_Reader/scripts/test_language_discourse_registry.py; Deep_Reflective_Reader/scripts/module-detailed-design.md (Main Responsibilities)`
  Notes: Metadata and registry semantics are now tracked by dedicated tests.

- [x] Covers structured-document reuse before raw-load preparation
  Evidence: `Deep_Reflective_Reader/scripts/test_prepare_structured_reuse_before_raw_load.py`; container execution of `PYTHONPATH=. python scripts/test_prepare_structured_reuse_before_raw_load.py`.
  Notes: The regression proves existing structured documents are validated and reused before raw PDF loading in base/common/non-force preparation, while force rebuild still loads raw text and rewrites structure.

- [x] Covers manual structure API request, validation/preview response, and commit reparse request schema validation
  Evidence: `Deep_Reflective_Reader/scripts/test_manual_structure_api_schemas.py`; `Deep_Reflective_Reader/.venv/bin/python Deep_Reflective_Reader/scripts/test_manual_structure_api_schemas.py`.
  Notes: Regression validates source-agnostic manual structure request schema, typed char/page anchors, title/doc normalization, two-level maximum depth, orphan-section rejection, invalid range rejection, lightweight preview response serialization, issue taxonomy fail-fast behavior, provenance preview normalization, response projected-range validation, and explicit manual reparse request schema gating.

- [x] Covers manual structure validation route behavior
  Evidence: `Deep_Reflective_Reader/scripts/test_manual_structure_validate_route.py`; `Deep_Reflective_Reader/.venv/bin/python Deep_Reflective_Reader/scripts/test_manual_structure_validate_route.py`.
  Notes: Regression validates `POST /documents/manual-structure/validate` returns a lightweight normalized preview, rejects invalid manual plans with 422, does not expose raw text, task content, or task-layout payload, and verifies schema-valid `parser_mode=manual_structure` commit requests route through the manual commit coordinator boundary without reaching the legacy common/LLM reparse coordinator.

- [x] Covers deterministic manual structure projection behavior
  Evidence: `Deep_Reflective_Reader/scripts/test_manual_structure_projection.py`; `Deep_Reflective_Reader/.venv/bin/python Deep_Reflective_Reader/scripts/test_manual_structure_projection.py`.
  Notes: Regression validates source-agnostic manual plan projection into a lightweight chapter/section preview, chapter-only same-name section preview, two-level maximum depth enforcement, orphan-section rejection, unsupported/mixed anchor rejection, char/page range validation, and sibling overlap detection without persistence or `StructuredDocument` creation.

- [x] Covers manual structure validation route delegation to projector
  Evidence: `Deep_Reflective_Reader/scripts/test_manual_structure_validate_route.py`; `Deep_Reflective_Reader/.venv/bin/python Deep_Reflective_Reader/scripts/test_manual_structure_validate_route.py`.
  Notes: Regression validates the validation route now surfaces deterministic projection errors such as overlapping sibling ranges as `valid=false` without raw text, task content, task-layout payload, hierarchy persistence, or reparse execution.

- [x] Covers manual structure commit validation gate before persistence
  Evidence: `Deep_Reflective_Reader/scripts/test_manual_structure_validate_route.py`; `Deep_Reflective_Reader/.venv/bin/python Deep_Reflective_Reader/scripts/test_manual_structure_validate_route.py`.
  Notes: Regression validates schema-valid but projection-invalid `parser_mode=manual_structure` commit requests return stable 422 with projector issue evidence and no structured document path or section count.

- [x] Covers manual structure commit source evidence gates
  Evidence: `Deep_Reflective_Reader/scripts/test_manual_structure_commit_source_evidence.py`; `Deep_Reflective_Reader/.venv/bin/python Deep_Reflective_Reader/scripts/test_manual_structure_commit_source_evidence.py`.
  Notes: Regression validates invalid projection avoids source loading, missing source returns 404, stale `source_hash` returns 409, and matching source evidence proceeds to draft/persistence only after validation succeeds.

- [x] Covers manual structure StructuredDocument draft building
  Evidence: `Deep_Reflective_Reader/scripts/test_manual_structure_document_builder.py`; `Deep_Reflective_Reader/.venv/bin/python Deep_Reflective_Reader/scripts/test_manual_structure_document_builder.py`.
  Notes: Regression validates char-range manual plans build hierarchy-only `StructuredDocument` drafts with `chapters[].sections[]`, no root `sections[]` or `structure_nodes`, chapter-only same-name implicit sections, advisory parse provenance, explicit page-range rejection until page-boundary mapping exists, and invalid range rejection without a partial document.

- [x] Covers manual structure commit draft-build gate
  Evidence: `Deep_Reflective_Reader/scripts/test_manual_structure_commit_source_evidence.py`; `Deep_Reflective_Reader/.venv/bin/python Deep_Reflective_Reader/scripts/test_manual_structure_commit_source_evidence.py`.
  Notes: Regression validates the coordinator builds a manual `StructuredDocument` draft only after source evidence passes and rejects raw-text out-of-range draft anchors with 422 before persistence.

- [x] Covers manual structure commit persistence gate
  Evidence: `Deep_Reflective_Reader/scripts/test_manual_structure_commit_source_evidence.py`; `Deep_Reflective_Reader/.venv/bin/python Deep_Reflective_Reader/scripts/test_manual_structure_commit_source_evidence.py`.
  Notes: Regression validates a source-hash-matched manual commit saves exactly one hierarchy-only `StructuredDocument` through the repository, while invalid projection, missing source, stale source, draft anchor failure, and repository save failure do not report a successful commit.

- [x] Covers manual structure commit route success mapping
  Evidence: `Deep_Reflective_Reader/scripts/test_manual_structure_validate_route.py`; `Deep_Reflective_Reader/.venv/bin/python Deep_Reflective_Reader/scripts/test_manual_structure_validate_route.py`.
  Notes: Regression validates `/documents/reparse-structure` maps a `parser_mode=manual_structure` coordinator success result to HTTP 200 with `success=true`, parser mode, section count, nullable backend path, and no legacy common/LLM reparse dispatch.

- [x] Covers page-backed manual structure route status mapping
  Evidence: `Deep_Reflective_Reader/scripts/test_manual_structure_validate_route.py`; `Deep_Reflective_Reader/.venv/bin/python Deep_Reflective_Reader/scripts/test_manual_structure_validate_route.py`.
  Notes: Regression validates schema-valid `page_range` commit routing for missing/unsupported page evidence (`422`), stale source evidence (`409`), out-of-range/unprojectable page anchors (`422`), and successful page-backed commit response mapping (`200`) without falling through to legacy common/LLM reparse.

- [x] Covers manual structure preparation source-evidence handoff
  Evidence: `Deep_Reflective_Reader/scripts/test_manual_structure_preparation_handoff.py`; `Deep_Reflective_Reader/.venv/bin/python Deep_Reflective_Reader/scripts/test_manual_structure_preparation_handoff.py`.
  Notes: Regression validates preparation provides normalized source identity, raw text, source hash, best-effort language, optional empty page-boundary evidence, and empty-source failure without structured artifact writes.

- [x] Covers manual structure commit provenance from preparation evidence
  Evidence: `Deep_Reflective_Reader/scripts/test_manual_structure_commit_source_evidence.py`; `Deep_Reflective_Reader/.venv/bin/python Deep_Reflective_Reader/scripts/test_manual_structure_commit_source_evidence.py`.
  Notes: Regression validates accepted manual commit preserves evidence language and actual raw-text source hash in saved `StructuredDocument.parse_provenance`.

- [x] Covers task-unit content endpoint cache-valid task-layout fixtures
  Evidence: `Deep_Reflective_Reader/scripts/test_task_unit_content_endpoint.py`; `PYTHONPATH=. .venv/bin/python scripts/test_task_unit_content_endpoint.py`.
  Notes: Regression fixtures now derive the expected task-layout resolver version from `SectionTaskCoordinator._TASK_LAYOUT_RESOLVER_VERSION`, so cache-valid smoke coverage verifies no resolver refresh and no persistence write under the current section-scoped metadata contract.

- [x] Covers task-layout cache-hit manual source-evidence regression
  Evidence: `Deep_Reflective_Reader/scripts/test_task_layout_persistence_cache.py`; `.venv` execution of `PYTHONPATH=Deep_Reflective_Reader Deep_Reflective_Reader/.venv/bin/python Deep_Reflective_Reader/scripts/test_task_layout_persistence_cache.py`.
  Notes: Regression simulates a task-layout cache hit and asserts the coordinator must not call `load_manual_structure_source_evidence(...)` on ordinary reads, because that path can trigger scanned-PDF OCR. The same script verifies an explicit `include_anchor_page_evidence=True` cache-hit read still loads page evidence for TOC edit-existing anchor prefill without recomputing task units.
  Timestamp: 2026-10-05

- [x] Covers native Apple PDF page-boundary loading without OCR
  Evidence: `Deep_Reflective_Reader/scripts/test_pdf_document_loader_inspection.py`; `.venv` execution of `PYTHONPATH=Deep_Reflective_Reader Deep_Reflective_Reader/.venv/bin/python Deep_Reflective_Reader/scripts/test_pdf_document_loader_inspection.py`; direct timing check for `APPLE.pdf`.
  Notes: Regression asserts native PDF text extraction happens once per page per caller, and `APPLE.pdf` page-boundary evidence returns one boundary per page with `ocr_started=False`. This protects simple born-digital PDFs from entering OCR or repeated native extraction during page-default evidence loading.
  Timestamp: 2026-10-05

- [x] Covers PostgreSQL manual reparse repository boundary separation
  Evidence: `Deep_Reflective_Reader/scripts/test_postgres_manual_reparse_repository_boundary.py`; `Deep_Reflective_Reader/.venv/bin/python Deep_Reflective_Reader/scripts/test_postgres_manual_reparse_repository_boundary.py`.
  Notes: Regression validates generic PostgreSQL structured document saves preserve current hierarchy/version through `replace_existing_hierarchy=False`, while parser-level manual reparse saves use the explicit replacement boundary with `replace_existing_hierarchy=True`.

- [x] Covers live PostgreSQL manual reparse duplicate chapter_order replacement
  Evidence: `Deep_Reflective_Reader/scripts/test_postgres_manual_reparse_duplicate_chapter_order.py`; rebuilt Docker API container validation with `docker compose exec -T api python scripts/test_postgres_manual_reparse_duplicate_chapter_order.py`.
  Notes: Live regression validates existing-document manual replacement over an existing `chapter_order=0` row, confirms the replacement hierarchy is current, confirms `initial_parse`/`hard_reparse` provenance, and confirms transaction rollback preserves the old hierarchy when replacement fails.

- [x] Covers artifact-aware interaction metadata schema validation
  Evidence: `Deep_Reflective_Reader/scripts/test_artifact_aware_interaction_metadata_schema.py`; `.venv` execution of `PYTHONPATH=Deep_Reflective_Reader Deep_Reflective_Reader/.venv/bin/python Deep_Reflective_Reader/scripts/test_artifact_aware_interaction_metadata_schema.py`; `python3 -m py_compile Deep_Reflective_Reader/api_schemas.py Deep_Reflective_Reader/scripts/test_artifact_aware_interaction_metadata_schema.py`.
  Notes: Regression validates `ArtifactAwareInteractionMetadataResponse` defaults, artifact context mode normalization, primary source evidence ids, referenced artifact ids/types/target levels, coverage counts, deduplication/abstraction flags, invalid shape rejection, and explicit rejection of nested child artifact payload expansion.
  Timestamp: 2026-10-08

- [x] Covers artifact-aware secondary context assembly
  Evidence: `Deep_Reflective_Reader/scripts/test_artifact_aware_context_builder.py`; `.venv` execution of `PYTHONPATH=Deep_Reflective_Reader Deep_Reflective_Reader/.venv/bin/python Deep_Reflective_Reader/scripts/test_artifact_aware_context_builder.py`; `python3 -m py_compile Deep_Reflective_Reader/context/artifact_aware_context.py Deep_Reflective_Reader/scripts/test_artifact_aware_context_builder.py`.
  Notes: Regression validates compact lower-level artifact summary assembly, empty/insufficient-content skipping, duplicate/max-count pruning, artifact context char-budget pruning, coverage counts, deduplication/abstraction hint flags, provenance metadata export, defensive metadata copies, and absence of raw child artifact payload expansion.
  Timestamp: 2026-10-08

- [x] Covers analysis interaction strict output validation
  Evidence: `Deep_Reflective_Reader/scripts/test_analysis_interaction_service.py`; `.venv` execution of `PYTHONPATH=Deep_Reflective_Reader Deep_Reflective_Reader/.venv/bin/python Deep_Reflective_Reader/scripts/test_analysis_interaction_service.py`; `python3 -m py_compile Deep_Reflective_Reader/section_tasks/analysis_interaction_service.py Deep_Reflective_Reader/scripts/test_analysis_interaction_service.py`.
  Notes: Regression validates completed analysis from strict JSON/dict payloads, required `summary`/`reasoning`/`explanation` fields, schema/prompt/context metadata, insufficient-content fast path without generator calls, invalid JSON and missing fields as `generation_failed`, and fail-fast rejection of non-analysis requests.
  Timestamp: 2026-10-08

## Needs Confirmation

No unresolved confirmation items identified in this pass.

## Future Task Policy

New future tasks for this module must be added here first as unchecked items:

- [x] Cover batch task-unit content route order and read-only semantics
  Evidence: `Deep_Reflective_Reader/scripts/test_task_unit_content_endpoint.py`; `.venv` execution of `PYTHONPATH=Deep_Reflective_Reader Deep_Reflective_Reader/.venv/bin/python Deep_Reflective_Reader/scripts/test_task_unit_content_endpoint.py`.
  Notes: Regression verifies batch content returns per-task-unit responses in request order, suppresses raw content by default, rejects duplicate request ids fail-fast, loads the document once for the batch, and does not mutate persistence.
  Timestamp: 2026-10-05
- [x] Cover reading target resolver hierarchy-only semantics
  Evidence: `Deep_Reflective_Reader/scripts/test_reading_target_resolver.py`; `.venv` execution of `PYTHONPATH=Deep_Reflective_Reader Deep_Reflective_Reader/.venv/bin/python Deep_Reflective_Reader/scripts/test_reading_target_resolver.py`; `python3 -m py_compile Deep_Reflective_Reader/section_tasks/reading_target_resolver.py Deep_Reflective_Reader/app/section_task_coordinator.py Deep_Reflective_Reader/scripts/test_reading_target_resolver.py`.
  Notes: Regression covers document/book, chapter, section, and task-unit target resolution, app-layer one-load orchestration, optional parent-id consistency checks, duplicate task-unit rejection, title-primary rejection, and no fallback to root `sections[]` legacy-only targets.
  Timestamp: 2026-10-08
- [x] Cover target-agnostic reading interaction service contracts
  Evidence: `Deep_Reflective_Reader/scripts/test_reading_interaction_service_contracts.py`; `.venv` execution of `PYTHONPATH=Deep_Reflective_Reader Deep_Reflective_Reader/.venv/bin/python Deep_Reflective_Reader/scripts/test_reading_interaction_service_contracts.py`; `python3 -m py_compile Deep_Reflective_Reader/section_tasks/reading_interaction_service_contracts.py Deep_Reflective_Reader/scripts/test_reading_interaction_service_contracts.py`.
  Notes: Regression validates resolved-target requests, metadata defensive copies, successful artifact shape, no raw target/source content serialization, reasoned insufficient-content status, invalid type/status rejection, empty completed artifact rejection, and critical-thinking-only status gating.
  Timestamp: 2026-10-08
- [x] Cover analysis read/generate split and strict output validation
  Evidence: `Deep_Reflective_Reader/scripts/test_analysis_interaction_service.py`; `Deep_Reflective_Reader/scripts/test_analysis_interaction_read_generate_split.py`; `.venv` execution of `PYTHONPATH=Deep_Reflective_Reader Deep_Reflective_Reader/.venv/bin/python Deep_Reflective_Reader/scripts/test_analysis_interaction_service.py`; `.venv` execution of `PYTHONPATH=Deep_Reflective_Reader Deep_Reflective_Reader/.venv/bin/python Deep_Reflective_Reader/scripts/test_analysis_interaction_read_generate_split.py`; `python3 -m py_compile Deep_Reflective_Reader/section_tasks/analysis_interaction_service.py Deep_Reflective_Reader/section_tasks/analysis_interaction_orchestrator.py Deep_Reflective_Reader/scripts/test_analysis_interaction_service.py Deep_Reflective_Reader/scripts/test_analysis_interaction_read_generate_split.py`.
  Notes: Regression verifies read returns persisted analysis or `not_generated` without generator calls or placeholder writes; explicit generate reuses existing artifacts unless refreshed, persists completed and insufficient-content results only, and leaves invalid JSON generation failures unpersisted while preserving strict output validation coverage.
  Timestamp: 2026-10-08
- [x] Cover quiz read/generate split and current-artifact semantics
  Evidence: `Deep_Reflective_Reader/scripts/test_quiz_interaction_service.py`; `Deep_Reflective_Reader/scripts/test_quiz_interaction_read_generate_split.py`; `.venv` execution of `PYTHONPATH=Deep_Reflective_Reader Deep_Reflective_Reader/.venv/bin/python Deep_Reflective_Reader/scripts/test_quiz_interaction_service.py`; `.venv` execution of `PYTHONPATH=Deep_Reflective_Reader Deep_Reflective_Reader/.venv/bin/python Deep_Reflective_Reader/scripts/test_quiz_interaction_read_generate_split.py`; `python3 -m py_compile Deep_Reflective_Reader/section_tasks/quiz_interaction_service.py Deep_Reflective_Reader/section_tasks/quiz_interaction_orchestrator.py Deep_Reflective_Reader/scripts/test_quiz_interaction_service.py Deep_Reflective_Reader/scripts/test_quiz_interaction_read_generate_split.py`.
  Notes: Regression verifies read returns persisted quiz artifacts or `not_generated` without generator calls or placeholder writes; explicit generate reuses existing artifacts unless refreshed, persists completed and insufficient-content results only, and leaves invalid JSON generation failures unpersisted while preserving configured-limit and answer validation coverage.
  Timestamp: 2026-10-08
- [x] Cover quiz validation and configured count limits
  Evidence: `Deep_Reflective_Reader/scripts/test_quiz_interaction_service.py`; `.venv` execution of `PYTHONPATH=Deep_Reflective_Reader Deep_Reflective_Reader/.venv/bin/python Deep_Reflective_Reader/scripts/test_quiz_interaction_service.py`; `python3 -m py_compile Deep_Reflective_Reader/section_tasks/quiz_interaction_service.py Deep_Reflective_Reader/scripts/test_quiz_interaction_service.py`.
  Notes: Regression verifies valid quiz type enum handoff, task-unit/section/chapter/document target-level max defaults, configured max override enforcement, fewer-than-max acceptance, answer payload validation for short-answer/multiple-choice/true-false items, invalid output as `generation_failed`, and insufficient-content fast path without generator calls.
  Timestamp: 2026-10-08
- [x] Cover critical-thinking session lifecycle
  Evidence: `Deep_Reflective_Reader/scripts/test_critical_thinking_session_service.py`; `.venv` execution of `PYTHONPATH=Deep_Reflective_Reader Deep_Reflective_Reader/.venv/bin/python Deep_Reflective_Reader/scripts/test_critical_thinking_session_service.py`; `python3 -m py_compile Deep_Reflective_Reader/section_tasks/critical_thinking_session_service.py Deep_Reflective_Reader/scripts/test_critical_thinking_session_service.py`.
  Notes: Regression verifies generated question status, fixed question/evaluation instruction handoff, generated-but-unanswered preservation, answer submission, strict evaluation validation, completed status, evaluation failure preservation with retry, insufficient-content fast path, invalid question generation failure, and invalid lifecycle transition rejection.
  Timestamp: 2026-10-08
- [x] Cover reading interaction artifact validity semantics
  Evidence: `Deep_Reflective_Reader/scripts/test_reading_interaction_artifact_validity.py`; `.venv` execution of `PYTHONPATH=Deep_Reflective_Reader Deep_Reflective_Reader/.venv/bin/python Deep_Reflective_Reader/scripts/test_reading_interaction_artifact_validity.py`; `python3 -m py_compile Deep_Reflective_Reader/section_tasks/artifact_validity.py Deep_Reflective_Reader/section_tasks/reading_interaction_service_contracts.py Deep_Reflective_Reader/section_tasks/analysis_interaction_service.py Deep_Reflective_Reader/section_tasks/quiz_interaction_service.py Deep_Reflective_Reader/section_tasks/critical_thinking_session_service.py Deep_Reflective_Reader/scripts/test_reading_interaction_artifact_validity.py`.
  Notes: Regression verifies shared validity preflight, stale target priority before insufficient-content checks, no generator call on stale target for analysis/quiz/critical-thinking services, insufficient-content reason preservation across all services, and required reasons for `stale_target` and `insufficient_content` artifacts.
  Timestamp: 2026-10-08
- [x] Cover adaptive full-context versus semantic compact context selection
  Evidence: `Deep_Reflective_Reader/scripts/test_reading_interaction_context_selection.py`; `.venv` execution of `PYTHONPATH=Deep_Reflective_Reader Deep_Reflective_Reader/.venv/bin/python Deep_Reflective_Reader/scripts/test_reading_interaction_context_selection.py`; `python3 -m py_compile Deep_Reflective_Reader/context/reading_interaction_context.py Deep_Reflective_Reader/scripts/test_reading_interaction_context_selection.py`.
  Notes: Regression verifies full target context when within configured/model capability budget, semantic compact fallback when the target exceeds capability, provenance metadata without source text, and fail-fast rejection of empty content without title fallback. No multi-call map-reduce occurs in the first version.
  Timestamp: 2026-10-08
- [x] Cover hard-reparse cleanup for reading interaction artifacts
  Evidence: `Deep_Reflective_Reader/scripts/test_hard_reparse_reading_interaction_cleanup.py`; `Deep_Reflective_Reader/scripts/test_postgres_manual_reparse_repository_boundary.py`; `.venv` execution of `PYTHONPATH=Deep_Reflective_Reader Deep_Reflective_Reader/.venv/bin/python Deep_Reflective_Reader/scripts/test_hard_reparse_reading_interaction_cleanup.py`; `.venv` execution of `PYTHONPATH=Deep_Reflective_Reader Deep_Reflective_Reader/.venv/bin/python Deep_Reflective_Reader/scripts/test_postgres_manual_reparse_repository_boundary.py`; `python3 -m py_compile Deep_Reflective_Reader/document_structure/structured_document_artifact_repository.py Deep_Reflective_Reader/db/postgres_structured_document_artifact_repository.py Deep_Reflective_Reader/scripts/test_hard_reparse_reading_interaction_cleanup.py Deep_Reflective_Reader/scripts/test_postgres_manual_reparse_repository_boundary.py`.
  Notes: Regression verifies parser-level hard reparse saves clear document/chapter/section/task-unit interaction artifacts, critical-thinking session metadata, and referenced-artifact metadata from replacement structured payloads. PostgreSQL replacement remains on `replace_existing_hierarchy=True` while generic saves remain non-replacement.
  Timestamp: 2026-10-08
- [x] Cover artifact-aware secondary context assembly
  Evidence: `Deep_Reflective_Reader/scripts/test_lower_level_artifact_reference_policy.py`; `.venv` execution of `PYTHONPATH=Deep_Reflective_Reader Deep_Reflective_Reader/.venv/bin/python Deep_Reflective_Reader/scripts/test_lower_level_artifact_reference_policy.py`; `python3 -m py_compile Deep_Reflective_Reader/section_tasks/reading_interaction_service_contracts.py Deep_Reflective_Reader/section_tasks/analysis_interaction_service.py Deep_Reflective_Reader/section_tasks/quiz_interaction_service.py Deep_Reflective_Reader/section_tasks/critical_thinking_session_service.py Deep_Reflective_Reader/scripts/test_lower_level_artifact_reference_policy.py`.
  Notes: Regression verifies analysis, quiz, and critical-thinking generation receive compact lower-level artifact summaries as `secondary_context`, preserve primary `target.content`, record artifact provenance metadata without context text or child payloads, and continue generation when no child artifacts exist.
  Timestamp: 2026-10-08
- [x] Cover quiz deduplication from lower-level artifacts
  Evidence: `Deep_Reflective_Reader/scripts/test_quiz_deduplication_from_lower_artifacts.py`; `.venv` execution of `PYTHONPATH=Deep_Reflective_Reader Deep_Reflective_Reader/.venv/bin/python Deep_Reflective_Reader/scripts/test_quiz_deduplication_from_lower_artifacts.py`; `python3 -m py_compile Deep_Reflective_Reader/section_tasks/quiz_interaction_service.py Deep_Reflective_Reader/scripts/test_quiz_deduplication_from_lower_artifacts.py`.
  Notes: Regression verifies higher-level quiz generation receives explicit deduplication guidance plus lower-level quiz coverage signals, preserves primary target content, does not concatenate lower-level quiz focus text into payload items, records bounded metadata only, and fails generation for exact lower-level focus repeats where backend validation can detect duplication.
  Timestamp: 2026-10-08
- [x] Cover critical-thinking abstraction from lower-level sessions
  Evidence: `Deep_Reflective_Reader/scripts/test_critical_thinking_abstraction_from_lower_sessions.py`; `.venv` execution of `PYTHONPATH=Deep_Reflective_Reader Deep_Reflective_Reader/.venv/bin/python Deep_Reflective_Reader/scripts/test_critical_thinking_abstraction_from_lower_sessions.py`; `python3 -m py_compile Deep_Reflective_Reader/section_tasks/critical_thinking_session_service.py Deep_Reflective_Reader/scripts/test_critical_thinking_abstraction_from_lower_sessions.py`.
  Notes: Regression verifies higher-level critical-thinking generation receives fixed learning-continuity guidance, child session focus/outcome signals, unchanged primary source content, predictable base question instruction, bounded provenance metadata without child payloads, and one generated question for the current target.
  Timestamp: 2026-10-08
- [x] Cover referenced-artifact metadata persistence
  Evidence: `Deep_Reflective_Reader/scripts/test_referenced_artifact_metadata_persistence.py`; `.venv` execution of `PYTHONPATH=Deep_Reflective_Reader Deep_Reflective_Reader/.venv/bin/python Deep_Reflective_Reader/scripts/test_referenced_artifact_metadata_persistence.py`; `python3 -m py_compile Deep_Reflective_Reader/section_tasks/reading_interaction_service_contracts.py Deep_Reflective_Reader/section_tasks/analysis_interaction_service.py Deep_Reflective_Reader/section_tasks/quiz_interaction_service.py Deep_Reflective_Reader/section_tasks/critical_thinking_session_service.py Deep_Reflective_Reader/scripts/test_referenced_artifact_metadata_persistence.py`.
  Notes: Regression verifies generated analysis, quiz, and critical-thinking artifacts store top-level referenced-artifact metadata with ids/types/target levels, coverage counts, and hint flags, without secondary context text or child payloads. Hard-reparse cleanup remains tracked by the separate hard-reparse cleanup checklist item.
  Timestamp: 2026-10-08
- [x] Cover reading interaction common artifact mapping
  Evidence: `Deep_Reflective_Reader/scripts/test_reading_interaction_common_artifact_model.py`; `.venv` execution of `PYTHONPATH=Deep_Reflective_Reader Deep_Reflective_Reader/.venv/bin/python Deep_Reflective_Reader/scripts/test_reading_interaction_common_artifact_model.py`; `python3 -m py_compile Deep_Reflective_Reader/shared/common_artifact_model.py Deep_Reflective_Reader/section_tasks/reading_interaction_common_artifact.py Deep_Reflective_Reader/scripts/test_reading_interaction_common_artifact_model.py`.
  Notes: Regression verifies analysis, quiz, and critical-thinking session artifacts all map to the same `CommonArtifact` entity class through `artifact_type`, preserve hierarchy-aware target ids, round-trip back to the service contract, reject unsupported artifact types/invalid targets, and avoid category-specific top-level serialization roots.
  Timestamp: 2026-10-08

After implementation, the task owner must update this checklist and mark the task as completed:

- [x] <completed task>

No coding task should be considered complete unless the corresponding module checklist is updated.

## Maintenance Notes

- This checklist is module memory for completed work.
- It does not replace the module detailed design document.
- It does not replace tests and test evidence.
- It does not replace proposal/HLD decisions and governance context.
