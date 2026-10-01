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

- [x] Covers PostgreSQL manual reparse repository boundary separation
  Evidence: `Deep_Reflective_Reader/scripts/test_postgres_manual_reparse_repository_boundary.py`; `Deep_Reflective_Reader/.venv/bin/python Deep_Reflective_Reader/scripts/test_postgres_manual_reparse_repository_boundary.py`.
  Notes: Regression validates generic PostgreSQL structured document saves preserve current hierarchy/version through `replace_existing_hierarchy=False`, while parser-level manual reparse saves use the explicit replacement boundary with `replace_existing_hierarchy=True`.

- [x] Covers live PostgreSQL manual reparse duplicate chapter_order replacement
  Evidence: `Deep_Reflective_Reader/scripts/test_postgres_manual_reparse_duplicate_chapter_order.py`; rebuilt Docker API container validation with `docker compose exec -T api python scripts/test_postgres_manual_reparse_duplicate_chapter_order.py`.
  Notes: Live regression validates existing-document manual replacement over an existing `chapter_order=0` row, confirms the replacement hierarchy is current, confirms `initial_parse`/`hard_reparse` provenance, and confirms transaction rollback preserves the old hierarchy when replacement fails.

## Needs Confirmation

No unresolved confirmation items identified in this pass.

## Future Task Policy

New future tasks for this module must be added here first as unchecked items:

- [ ] <future task>

After implementation, the task owner must update this checklist and mark the task as completed:

- [x] <completed task>

No coding task should be considered complete unless the corresponding module checklist is updated.

## Maintenance Notes

- This checklist is module memory for completed work.
- It does not replace the module detailed design document.
- It does not replace tests and test evidence.
- It does not replace proposal/HLD decisions and governance context.
