# scripts Checklist

## Loader Boundary Governance

- [x] LOADER-04: Validate loader abstraction with third-format contracts, PDF regressions and backed-up real books.
  Acceptance: no production adapter claimed for synthetic format; exact source/profile preservation, existing API/identity regressions and real-book reports.
  Evidence: test_loader_boundary.py; accept_loader_boundary_books.py; private loader-boundary-20261021T133550.json and final read-only content-recovery-20261021T133544.json. Docker PDF 180 pass, focused 95 pass, PostgreSQL 88 pass/two skips (overlapping suites). Both books preserve canonical hashes and zero warm extraction/profile work. Initial unsafe-default live acceptance and explicit layout recovery recorded in design; read-only flag added.
  Timestamp: 2026-10-21

## Format-Neutral Extraction Backlog (2026-10-21)

Implemented; evidence and scope are recorded per item. Ordered design: [shared plan](../document_preparation/module-detailed-design.md#format-neutral-extraction-plan-2026-10-21).

- [x] EXTRACT-10: Accept real TXT and PDF enhancement using immutable backup books.
  Depends on: EXTRACT-01 through EXTRACT-09.
  Acceptance: Both named books complete actual LLM enhancement/atomic commit/automatic layout from backup-only isolated inputs; exact text and profile hashes preserved, negative matrix and PDF regressions pass; private reports required, mocks/Common parse alone do not close.
  Evidence: Madame Bovary backup fa794acef055f235/20261021T083239624177Z/api-live.json: direct real API/model, three calls, 5 chapters/31 sections/423 units, 652001 characters preserved and profile hashes unchanged. PDF backup 085747b65094d52f/20261021T083612482751Z: two real model calls, 11 chapters/30 sections; api-replay.json proves atomic commit/cleanup and 108 real-layout units with unchanged profiles. Final 95 focused tests pass; 180 PDF tests pass; 75 PostgreSQL regressions pass with two opt-in skips; desktop/mobile confirmation and build pass. Failed live run led to a located semantic-boundary regression/fix (prompt v16), not a weakened validator. All enhancement writes used disposable schemas; live Library TXT provenance was explicitly migrated without replacing hierarchy/profile.
  Timestamp: 2026-10-21

## Content Read Performance (2026-10-21)

- [x] CONTENT-05: Rebuild both authorized books from verified backups and accept full-section latency, constant read counts and profile immutability.
  Acceptance: See app detailed design Content Read Recovery; evidence required before completion.
  Evidence: accept_content_read_books.py and test-content-read-live.mjs pass after authorized deletion/reupload/Common Fast preparation. Private content-recovery-20261021T073403.json and final-deployment content-recovery-20261021T074014.json: largest sections, three HTTP runs each, unchanged profile hashes, unchanged repeated-prepare layout and five interaction reads. 51 focused tests and eight regression scripts pass.
  Timestamp: 2026-10-21

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

- [x] Move real-book fixtures under data and run acceptance against immutable backups.
  Acceptance: Preserve the raw PDF, accepted source and original hierarchy with hash checks; never overwrite a different backup. Both live generation and API replay use backup case IDs without original Library reads. Verify deletion survival, corruption rejection and real-book replay.
  Evidence: `private_book_corpus.py`, migrated `data/test-books/085747b65094d52f`, four passing corpus tests (original deletion, immutable overwrite guard, corruption/missing backup, path containment). Backup-only live attempt `20261021T052028529811Z` passes with 18 chapters/35 sections and exact 147401-character preservation. Backup-only real API/disposable PostgreSQL/layout replay passes with 107 units; production source-read methods are explicitly blocked. Git/data Docker exclusions and diff check pass.
  Timestamp: 2026-10-21

- [x] Validate upload/preparation timing logs without real document parsing or LLM calls.
  Evidence: passing `test_upload_preparation_logging.py` (4 cases), `test_document_upload_route.py`, `test_prepare_structured_reuse_before_raw_load.py`, `test_document_preparation_raw_load_errors.py`, `test_pdf_document_loader_inspection.py`, and `test_main_request_logging.py`.
  Notes: Covers worker/async request-id isolation, stage boundaries, cache reuse, unchanged exception propagation, and exclusion of content/error-detail sentinels from added logs.
  Validation limitation: Additional `test_task_layout_chapters_response.py` fails its existing task-unit ordering assertion both in the working tree and an isolated `git archive HEAD` baseline. No unrelated task-layout fix is included. Live Docker rejected-upload smoke confirms stage records and request id are persisted without creating a document.
  Timestamp: 2026-10-14

- [x] Regress storage-aware deletion, residual database records, failures, and namespace isolation.
  Evidence: passing `test_document_delete_route.py`, `test_document_management_route.py`, `test_document_list_search_api.py`; Docker execution of `test_postgres_document_delete.py` on 2026-10-14.
  Notes: Covers confirmation, missing raw files, full-width slash names, database failures and retry, catalog readback, all document-owned PostgreSQL cascades, and namespace isolation without LLM calls.
  Timestamp: 2026-10-14

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

- [x] Covers analysis interaction public route vertical slice
  Evidence: `Deep_Reflective_Reader/scripts/test_analysis_interaction_routes.py`; `.venv` execution of `PYTHONPATH=Deep_Reflective_Reader Deep_Reflective_Reader/.venv/bin/python Deep_Reflective_Reader/scripts/test_analysis_interaction_routes.py`; `python3 -m py_compile Deep_Reflective_Reader/main.py Deep_Reflective_Reader/scripts/test_analysis_interaction_routes.py`.
  Notes: Regression validates `POST /documents/reading-interactions/insight/read`, `/insight/generate`, and `/insight/refresh` route mapping, schema-normalized target handoff, shared envelope serialization, compact analysis payload mapping, reasoning-to-interpretation fallback, and read/write dispatch separation.
  Timestamp: 2026-10-10

- [x] Covers reading interaction route HTTP status policy for insight routes
  Evidence: `Deep_Reflective_Reader/scripts/test_analysis_interaction_routes.py`; `.venv` execution of `PYTHONPATH=Deep_Reflective_Reader Deep_Reflective_Reader/.venv/bin/python Deep_Reflective_Reader/scripts/test_analysis_interaction_routes.py`; `python3 -m py_compile Deep_Reflective_Reader/main.py Deep_Reflective_Reader/scripts/test_analysis_interaction_routes.py`.
  Notes: Regression validates malformed insight requests fail with `422` before app dispatch, missing hierarchy targets map to `404`, `insufficient_content` remains a `200` envelope state without payload, `stale_target` maps to `409`, `generation_failed` maps to `502`, `validation_failed` maps to `422`, and successful read/generate/refresh remain `200`.
  Timestamp: 2026-10-10

- [x] Covers no-auto-generation insight read route behavior
  Evidence: `Deep_Reflective_Reader/scripts/test_analysis_interaction_routes.py`; `.venv` execution of `PYTHONPATH=Deep_Reflective_Reader Deep_Reflective_Reader/.venv/bin/python Deep_Reflective_Reader/scripts/test_analysis_interaction_routes.py`; `python3 -m py_compile Deep_Reflective_Reader/main.py Deep_Reflective_Reader/scripts/test_analysis_interaction_routes.py`.
  Notes: Regression uses a read-only poison coordinator for `POST /documents/reading-interactions/insight/read` and proves absent artifact reads return `not_generated` without payload while route dispatch remains limited to `read_analysis_artifact`. Accidental non-read coordinator access, including generation/refresh/session/prepare/reparse/task-layout/profile diagnostic paths, fails the test.
  Timestamp: 2026-10-10

- [x] Covers quiz interaction public route vertical slice
  Evidence: `Deep_Reflective_Reader/scripts/test_quiz_interaction_routes.py`; `.venv` execution of `PYTHONPATH=Deep_Reflective_Reader Deep_Reflective_Reader/.venv/bin/python Deep_Reflective_Reader/scripts/test_quiz_interaction_routes.py`; `python3 -m py_compile Deep_Reflective_Reader/main.py Deep_Reflective_Reader/scripts/test_quiz_interaction_routes.py`.
  Notes: Regression validates `POST /documents/reading-interactions/quiz/read`, `/quiz/generate`, and `/quiz/refresh` route mapping, schema-normalized target handoff, shared envelope serialization, drawer quiz payload item mapping, prompt version handoff, request `max_items` validation, generated item type validation, and separation from legacy section/chapter quiz routes.
  Timestamp: 2026-10-10

- [x] Covers critical-thinking interaction public route vertical slice
  Evidence: `Deep_Reflective_Reader/scripts/test_critical_thinking_interaction_routes.py`; `.venv` execution of `PYTHONPATH=Deep_Reflective_Reader Deep_Reflective_Reader/.venv/bin/python Deep_Reflective_Reader/scripts/test_critical_thinking_interaction_routes.py`; `python3 -m py_compile Deep_Reflective_Reader/main.py Deep_Reflective_Reader/api_schemas.py Deep_Reflective_Reader/scripts/test_critical_thinking_interaction_routes.py`.
  Notes: Regression validates `POST /documents/reading-interactions/critical-thinking/read`, `/generate-question`, `/submit-answer`, and `/retry-evaluation` route mapping, no-generation missing reads, shared target handoff, generated session id mapping, answer preservation on evaluation failure, retry completion without regenerating the question, and malformed submit rejection before dispatch.
  Timestamp: 2026-10-10

- [x] Covers complete public reading interaction route exposure set
  Evidence: `Deep_Reflective_Reader/scripts/test_analysis_interaction_routes.py`; `Deep_Reflective_Reader/scripts/test_quiz_interaction_routes.py`; `Deep_Reflective_Reader/scripts/test_critical_thinking_interaction_routes.py`; `Deep_Reflective_Reader/scripts/test_reading_interaction_api_schemas.py`; `.venv` execution of those route/schema regressions.
  Notes: Regression coverage now spans insight read/generate/refresh, quiz read/generate/refresh, and critical-thinking read/generate-question/submit-answer/retry-evaluation with shared target/envelope schema usage, no-auto-generation read behavior, explicit write-path dispatch, and malformed payload rejection before app dispatch.
  Timestamp: 2026-10-10

- [x] Covers document-backed analysis interaction artifact store
  Evidence: `Deep_Reflective_Reader/scripts/test_reading_interaction_artifact_store.py`; `.venv` execution of `PYTHONPATH=Deep_Reflective_Reader Deep_Reflective_Reader/.venv/bin/python Deep_Reflective_Reader/scripts/test_reading_interaction_artifact_store.py`; `python3 -m py_compile Deep_Reflective_Reader/section_tasks/reading_interaction_artifact_store.py Deep_Reflective_Reader/scripts/test_reading_interaction_artifact_store.py`.
  Notes: Regression validates missing current analysis reads return `None`, saved analysis artifacts receive deterministic artifact ids, persisted metadata is stored under `reading_interaction_artifacts`, and readback round-trips through the shared `CommonArtifact` adapter without creating a separate persistence category.
  Timestamp: 2026-10-10

- [x] Covers explicit document upload route behavior
  Evidence: `Deep_Reflective_Reader/scripts/test_document_upload_route.py`; `Deep_Reflective_Reader/main.py`; `Deep_Reflective_Reader/api_schemas.py`; `.venv` execution of `PYTHONPATH=Deep_Reflective_Reader Deep_Reflective_Reader/.venv/bin/python Deep_Reflective_Reader/scripts/test_document_upload_route.py`; `.venv` execution of `PYTHONPATH=Deep_Reflective_Reader Deep_Reflective_Reader/.venv/bin/python Deep_Reflective_Reader/scripts/test_document_list_search_api.py`; `Deep_Reflective_Reader/.venv/bin/python -m py_compile Deep_Reflective_Reader/main.py Deep_Reflective_Reader/scripts/test_document_upload_route.py`.
  Notes: Regression validates `POST /documents/upload` stores raw `.txt` source candidates, returns lightweight typed upload metadata, exposes uploaded raw candidates through `GET /documents`, rejects duplicate uploads by default, supports explicit replace and create-distinct conflict policies, asserts create-distinct returns a unique backend-owned `source_filename` and `doc_name`, rejects unsupported file types with `unsupported_file_type`, rejects unsafe filenames, and does not prepare/reparse/OCR/profile/retrieval/task-layout/generate interactions.
  Timestamp: 2026-10-13

- [x] Covers document management status/list route behavior
  Evidence: `Deep_Reflective_Reader/scripts/test_document_management_route.py`; `Deep_Reflective_Reader/main.py`; `Deep_Reflective_Reader/api_schemas.py`; `.venv` execution of `PYTHONPATH=Deep_Reflective_Reader Deep_Reflective_Reader/.venv/bin/python Deep_Reflective_Reader/scripts/test_document_management_route.py`; `Deep_Reflective_Reader/.venv/bin/python -m py_compile Deep_Reflective_Reader/main.py Deep_Reflective_Reader/api_schemas.py Deep_Reflective_Reader/scripts/test_document_management_route.py`.
  Notes: Regression validates `GET /documents/management` merges structured readiness and raw source metadata, marks structured-only rows as ready, raw-only rows as not prepared, same-name raw+structured rows as ready while preserving raw source metadata, supports query filtering, and does not expose raw text, hierarchy, task-unit content, content blocks, profile payloads, retrieval records, or artifacts.
  Timestamp: 2026-10-13

- [x] Covers document lifecycle action schema behavior
  Evidence: `Deep_Reflective_Reader/scripts/test_document_lifecycle_api_schemas.py`; `Deep_Reflective_Reader/api_schemas.py`; `.venv` execution of `PYTHONPATH=Deep_Reflective_Reader Deep_Reflective_Reader/.venv/bin/python Deep_Reflective_Reader/scripts/test_document_lifecycle_api_schemas.py`; `Deep_Reflective_Reader/.venv/bin/python -m py_compile Deep_Reflective_Reader/api_schemas.py Deep_Reflective_Reader/scripts/test_document_lifecycle_api_schemas.py`.
  Notes: Regression validates explicit `archive`, `restore`, `delete`, and `rename` lifecycle DTO semantics, rename target rules, typed non-success reasons, accepted response shape, source/structured/derived-resource effect flags, and rejection of raw text, hierarchy, task-layout, content-block, and artifact payload fields.
  Timestamp: 2026-10-13

- [x] Covers explicit document delete route behavior
  Evidence: `Deep_Reflective_Reader/scripts/test_document_delete_route.py`; `Deep_Reflective_Reader/main.py`; `Deep_Reflective_Reader/api_schemas.py`; `.venv` execution of `PYTHONPATH=Deep_Reflective_Reader Deep_Reflective_Reader/.venv/bin/python Deep_Reflective_Reader/scripts/test_document_delete_route.py`; `Deep_Reflective_Reader/.venv/bin/python -m py_compile Deep_Reflective_Reader/main.py Deep_Reflective_Reader/scripts/test_document_delete_route.py`.
  Notes: Regression validates `POST /documents/lifecycle/delete` requires explicit confirmation, removes matching raw source, structured JSON, and derived FAISS/profile artifacts, returns typed missing/unconfirmed/wrong-action lifecycle responses, keeps response payload lightweight, and does not expose raw text, hierarchy, task-unit content, content blocks, profile payloads, retrieval records, or artifacts.
  Timestamp: 2026-10-13

- [x] Covers explicit document rename route behavior
  Evidence: `Deep_Reflective_Reader/scripts/test_document_rename_route.py`; `Deep_Reflective_Reader/main.py`; `Deep_Reflective_Reader/api_schemas.py`; `.venv` execution of `PYTHONPATH=Deep_Reflective_Reader Deep_Reflective_Reader/.venv/bin/python Deep_Reflective_Reader/scripts/test_document_rename_route.py`; `Deep_Reflective_Reader/.venv/bin/python -m py_compile Deep_Reflective_Reader/main.py Deep_Reflective_Reader/scripts/test_document_rename_route.py`.
  Notes: Regression validates `POST /documents/lifecycle/rename` moves matching raw source, structured JSON, and derived FAISS/profile artifacts, updates lightweight structured JSON identity metadata, rejects target conflicts without partial overwrite, returns typed missing/wrong-action lifecycle responses, keeps response payload lightweight, and does not expose raw text, hierarchy, task-unit content, content blocks, profile payloads, retrieval records, or artifacts.
  Timestamp: 2026-10-13

- [x] Supports local development env-file injection for private LLM keys
  Evidence: `scripts/load_local_env.sh`; `scripts/start_local_dev.sh`; `.env.local.example`; `.env.example`; validation with `bash -n scripts/load_local_env.sh scripts/start_local_dev.sh`; dry-run source validation against a temporary env file.
  Notes: Local development startup now sources a reusable env loader that imports `.env.local` into the shell before Docker Compose receives the same env file. The template documents `.env.local` as the private ignored file, uses provider-neutral `DEEP_READER_LLM_API_KEY` for developer-owned secrets, bridges that value to the current backend runtime variable for compatibility, and includes `DEEP_READER_UI_PORT`.
  Timestamp: 2026-10-13

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

## Selective Per-Page OCR Track

OCR-01 supporting evidence (counted only in the loader checklist):
`test_pdf_ocr_decision.py` passed 7 contract tests on 2026-10-14; existing
`test_pdf_document_loader_inspection.py` and `test_pdf_page_evidence.py` passed.
Runtime routing and real-document performance acceptance remain pending.

OCR-02 supporting evidence (counted only in the loader checklist):
`test_pdf_page_inspection.py` passed 18 tests on 2026-10-14; OCR-01's 7 tests and
existing PDF loader inspection, page-evidence and Outline regressions also passed.
Evidence covers real synthetic PDF streams, single native extraction, clipped
image union, unknown geometry/errors, no image decoding/subprocess, metadata
content exclusion and source replacement. No extra delivery task is counted.
The 18 inspection tests also passed with pinned pypdf 6.17.0 in a disposable
Compose container, without rebuilding the API or preparing user documents.

OCR-03 supporting evidence (counted only in the loader checklist):
`test_pdf_ocr_policy.py` passed 22 tests locally and in an isolated Compose
container on 2026-10-14; existing OCR-01 (7), OCR-02 (18) and PDF loader inspection
also passed. Tests cover three-way decisions, numeric/Chinese/Latin fixtures,
conservative mixed/hidden/header handling, conflicts, threshold/config validation
and bounded probe plans. Visual checks are fixtures; no actual detector or routing
is implemented by this task.

OCR-04 supporting evidence (counted only in the loader checklist):
`test_pdf_layout_routing.py` passed 14 tests locally and in isolated Compose on
2026-10-14, alongside earlier decision/policy/inspection, loader, page-evidence,
Outline and OCR layout fixture regressions. Read-only local four-page `《歸來》`
layout smoke retained 8,187 native characters with zero render/OCR calls in
346.1 ms; this is not full preparation acceptance or a deployed timing claim.

OCR-05 supporting evidence (counted only in the loader checklist):
`test_pdf_raw_recovery.py` passed 25 tests locally and in isolated Compose on
2026-10-14. Earlier PDF and manual-structure regressions, including the updated
mixed-PDF Outline test, pass; upload/preparation logging tests pass. Read-only
`《歸來》` smoke retained exact native text and four-page raw-span parity with OCR
disabled. Raster cover/trailing recovery in the Outline test is mocked, not a
real OCR quality claim. Design section 15 and loader section 16.9 record limits.

OCR-06 supporting evidence (counted only in the preparation checklist):
`test_prepare_lazy_pdf_layout.py` passes 15 tests locally and in isolated Compose
on 2026-10-14; all 86 PDF unittest cases and standalone PDF/TOC, raw-error,
structured-reuse, profile-ordering, manual and logging regressions pass. Shared
raw/spans, lazy acceptance, source mismatch, required recovery, force/cache and
failure paths are covered with isolated fixtures. No full latency acceptance.

OCR-07 supporting evidence (counted only in the loader checklist):
`test_pdf_execution_pass.py` adds 25 tests; all 111 PDF unittest cases pass locally
and in isolated Compose on 2026-10-14. Fifteen lazy-layout tests and 11 standalone
PDF/TOC/preparation/manual/logging scripts pass locally. Coverage includes bounded
cache identity/lifecycle, worker isolation, actual attempts/deadlines, required
failure atomicity, optional fallback and content-free correlated diagnostics.
Earlier memory fixtures explicitly declare their shared scope; timeout fixtures
expect the new budget-specific reason. No live deployment or OCR-08 completion.

- [x] OCR-08: Validate selective OCR correctness and measured cost end to end.
  Acceptance: Run fixture matrix in design section 15; assert render/OCR call counts, native/scanned/mixed extraction parity, quality rejection, boundary order, Outline/fallback behavior, disabled OCR, budget and cache invalidation. Compare cold/warm stage timings on the reported four-page PDF and representative scans using disposable preparation storage; record unavailable cases explicitly.
  Evidence: `accept_selective_ocr.py`; `test_selective_ocr_acceptance.py` (3 tests); 111 PDF and 15 lazy-preparation tests, all passing locally and in isolated Compose. Eleven standalone PDF/TOC/preparation/manual/logging scripts pass locally. Real-engine native/scanned/mixed fixtures verify full known transcript, exact native/blank spans and zero layout OCR for accepted Outline. Design section 15 records measured cases, environment, commands and limitations.
  Results: Four-page `《歸來》` retains 8,200 exact canonical characters with zero render/OCR calls; controlled cold/warm preparation 789.9/1.9 ms. Real `暗水幽灵` page 8 completes and reuses structure; `國富論lite` page 1 is quality-rejected without persistence. Scanned-page timings/counts are recorded separately, not hidden behind overall latency.
  Limits: Real language/profile/LLM, HTTP/PostgreSQL costs, full scanned-book accuracy and real vertical/rotated transcription accuracy are unverified. Excerpt quality-gate success is not ground-truth accuracy. No live deployment, user-artifact changes or matched historical speedup claim; broader historical roadmap items remain open.
  Depends on: OCR-07 in `../doc_loaders/module-checklist.md`. Final task of eight; earlier contract/policy tests belong to their implementation task, not extra progress items.
  Timestamp: 2026-10-14

## Maintenance Notes

- This checklist is module memory for completed work.
- It does not replace the module detailed design document.
- It does not replace tests and test evidence.
- It does not replace proposal/HLD decisions and governance context.
