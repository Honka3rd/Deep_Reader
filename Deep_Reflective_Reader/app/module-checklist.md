# app Checklist

## Format-Neutral Extraction Backlog (2026-10-21)

Implemented; evidence and scope are recorded per item. Ordered design: [shared plan](../document_preparation/module-detailed-design.md#format-neutral-extraction-plan-2026-10-21).

- [x] EXTRACT-08: Connect TXT/PDF preflight and execution to generic accepted sources.
  Depends on: EXTRACT-06, EXTRACT-07.
  Acceptance: Common full-prompt capacity gate, source versus capacity reasons, normalized consent, correlated logs and automatic layout flow; zero model calls for failed preflight and zero extraction/profile regeneration.
  Evidence: Live Madame Bovary preflight ready=true, execution_available=true, capacity=fits; generic accepted snapshots reach existing API/consent/capacity flows.
  Timestamp: 2026-10-21

## Enhancement Outcome Follow-up

- [x] ENH-REV-03: Expose three-way enhancement results and conditional consent.
  Depends on: document_structure ENH-REV-01 and db ENH-REV-02.
  Acceptance: Coordinate validated comparison/commit; additive typed change_kind/committed/version/cleanup/layout-required semantics distinguish no-op, label commit and structural commit. Preflight warns conditionally before generation and binds baseline labels as well as hierarchy/source/policy. API schemas/routes must be implemented before UI adoption. Permit retry of transient generation failures without cost gates; reconcile unknown outcomes and prevent duplicate commit/cleanup. No hidden task-layout writes.
  Evidence: `enhancement_operations.py`, preflight/coordinator, reparse result/DTO/routes and safe provider failure classification; `test_enhancement_outcome_api.py` and `test_enhancement_operations.py` cover three outcomes, legacy defaults, stale label consent, same-version response-loss reconciliation/replay, in-flight exclusion, transient retry, invalid plans and unknown state after restart/record expiry. Existing API/DB tests retain capacity-zero-call and explicit layout-only recovery coverage.
  Validation: Seven-suite acceptance runs 124 tests: 123 passed, one private-source opt-in skipped. No paid LLM or production mutation. Process-local result registry: 1024 records, one-hour inactive retention, unknown on another worker/restart; no durable job guarantee. UI integration remains ENH-REV-05.
  Timestamp: 2026-10-21

## Full-Source Enhancement Follow-up

- [x] FULL-03: Expose capacity-aware preflight and consent-bound execution checks.
  Depends on: FULL-01 and FULL-02.
  Acceptance: Shared full-source budgeting exposes safe nullable capacity/count details; no token or execution when capacity fails. Bind model/prompt/output/budget identity to consent and revalidate before dispatch. Preserve existing source readiness, atomic cleanup and concurrency guards; no OCR, LLM or writes in preflight.
  Evidence: `enhancement_preflight.py`, coordinator/pipeline capacity handoff, shared builder observation, request-scoped dispatch identity, typed API capacity response and `test_enhancement_capacity_preflight.py`. No token unless storage/source/capacity checks pass; consent v2 and under-lease rechecks preserve existing atomic commit guards.
  Validation: 139 tests discovered, 138 passed, one explicit private-source opt-in skip; includes 11 new cases and isolated PostgreSQL/API regression. git diff --check passed. No paid LLM, production mutation or frontend implementation.
  Timestamp: 2026-10-20

## Content Read Performance (2026-10-21)

- [x] CONTENT-01: Read content from one persisted snapshot; remove prepare and per-block storage/profile I/O.
  Acceptance: See app detailed design Content Read Recovery; evidence required before completion.
  Evidence: Coordinator direct repository reads, snapshot artifact adapter, request-level profile reuse and version-aware status projection; content endpoint regressions and ten new recovery cases pass. Real largest-section API acceptance passes.
  Timestamp: 2026-10-21

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

- [x] Audit app-layer reading interaction orchestration exposure
  Evidence: `Deep_Reflective_Reader/app/section_task_coordinator.py`; `Deep_Reflective_Reader/app/module-detailed-design.md`; `Deep_Reflective_Reader/section_tasks/module-detailed-design.md`; `Deep_Reflective_Reader/main.py`.
  Notes: Confirmed that app-layer target resolution exists, and `section_tasks/` owns code-confirmed service/orchestrator behavior for analysis, quiz interaction, and critical-thinking sessions. At audit time, `SectionTaskCoordinator` had not yet exposed public REST-facing read/generate/submit/retry orchestration methods; FE-INT-05 records the later method-contract implementation.
  Timestamp: 2026-10-08

- [x] FE-INT-05 Define REST-facing app orchestration method contracts
  Evidence: `Deep_Reflective_Reader/app/section_task_coordinator.py`; `Deep_Reflective_Reader/scripts/test_app_reading_interaction_orchestration.py`; `PYTHONPATH=Deep_Reflective_Reader Deep_Reflective_Reader/.venv/bin/python Deep_Reflective_Reader/scripts/test_app_reading_interaction_orchestration.py`; `python3 -m py_compile Deep_Reflective_Reader/app/section_task_coordinator.py Deep_Reflective_Reader/scripts/test_app_reading_interaction_orchestration.py`.
  Notes: `SectionTaskCoordinator` now exposes app methods for analysis read/generate/refresh, quiz read/generate/refresh, and critical-thinking read/generate-question/submit-answer/retry-evaluation. Each method resolves the target once, delegates to the configured service/orchestrator/session-store dependency, and returns response-safe `ReadingInteractionResponseDTO` metadata without raw target content or frontend UI state.
  Timestamp: 2026-10-09

- [x] FE-INT-06 Define artifact persistence/read contract for public interaction APIs
  Evidence: `Deep_Reflective_Reader/app/section_task_coordinator.py`; `Deep_Reflective_Reader/section_tasks/analysis_interaction_orchestrator.py`; `Deep_Reflective_Reader/section_tasks/quiz_interaction_orchestrator.py`; `Deep_Reflective_Reader/section_tasks/critical_thinking_session_service.py`; `Deep_Reflective_Reader/scripts/test_app_reading_interaction_persistence_contract.py`; `Deep_Reflective_Reader/scripts/test_postgres_manual_reparse_repository_boundary.py`; `PYTHONPATH=Deep_Reflective_Reader Deep_Reflective_Reader/.venv/bin/python Deep_Reflective_Reader/scripts/test_app_reading_interaction_persistence_contract.py`; `PYTHONPATH=Deep_Reflective_Reader Deep_Reflective_Reader/.venv/bin/python Deep_Reflective_Reader/scripts/test_postgres_manual_reparse_repository_boundary.py`; `python3 -m py_compile Deep_Reflective_Reader/scripts/test_app_reading_interaction_persistence_contract.py Deep_Reflective_Reader/app/section_task_coordinator.py`.
  Notes: App-level public interaction behavior is now locked by regression coverage: read paths return persisted current artifacts/sessions or `not_generated` without generation or placeholder writes; explicit generate/refresh writes persist only validated current-artifact statuses; completed current artifacts are reused by generate, while persisted insufficient-content analysis/quiz artifacts are preflight-rechecked on read/generate so stale insufficient-content results can recover; stale targets surface as `stale_target` without overwriting current artifacts; critical-thinking failed evaluation preserves and saves the submitted answer for retry under the same session id. Hard reparse derived-resource cleanup remains owned by the repository reparse boundary, not ordinary interaction read/generate methods.
  Timestamp: 2026-10-11

- [x] Add analysis artifact read/generate orchestration as the first vertical slice
  Evidence: `Deep_Reflective_Reader/app/section_task_coordinator.py`; `Deep_Reflective_Reader/section_tasks/analysis_interaction_llm_generator.py`; `Deep_Reflective_Reader/section_tasks/reading_interaction_artifact_store.py`; `Deep_Reflective_Reader/config/container.py`; `Deep_Reflective_Reader/main.py`; `Deep_Reflective_Reader/scripts/test_analysis_interaction_routes.py`; `Deep_Reflective_Reader/scripts/test_reading_interaction_artifact_store.py`; `PYTHONPATH=Deep_Reflective_Reader Deep_Reflective_Reader/.venv/bin/python Deep_Reflective_Reader/scripts/test_analysis_interaction_routes.py`; `PYTHONPATH=Deep_Reflective_Reader Deep_Reflective_Reader/.venv/bin/python Deep_Reflective_Reader/scripts/test_reading_interaction_artifact_store.py`.
  Notes: Analysis/insight now has a wired app vertical slice: route requests dispatch through `SectionTaskCoordinator` analysis read/generate/refresh methods, target resolution remains hierarchy-only, reads return persisted current analysis or `not_generated`, and explicit generate/refresh calls the LLM-backed analysis service through the configured orchestrator/store path. The response DTO remains compact for inline insight and does not carry raw target content or frontend UI state.
  Timestamp: 2026-10-10

- [x] Add quiz artifact read/generate orchestration through the same target abstraction
  Evidence: `Deep_Reflective_Reader/app/section_task_coordinator.py`; `Deep_Reflective_Reader/scripts/test_app_reading_interaction_orchestration.py`; `Deep_Reflective_Reader/scripts/test_app_reading_interaction_persistence_contract.py`; `Deep_Reflective_Reader/scripts/test_quiz_interaction_routes.py`.
  Notes: App orchestration exposes `read_quiz_artifact(...)`, `generate_quiz_artifact(...)`, and `refresh_quiz_artifact(...)` using the shared hierarchy-aware target resolver. The coordinator delegates to the quiz interaction orchestrator, returns response-ready DTOs, keeps LLM type-mix decisions in the service layer, and route regressions verify the generic drawer API stays separate from legacy section/chapter quiz endpoints.
  Timestamp: 2026-10-10

- [x] Add critical-thinking session orchestration
  Evidence: `Deep_Reflective_Reader/app/section_task_coordinator.py`; `Deep_Reflective_Reader/scripts/test_app_reading_interaction_orchestration.py`; `Deep_Reflective_Reader/scripts/test_app_reading_interaction_persistence_contract.py`; `Deep_Reflective_Reader/scripts/test_critical_thinking_interaction_routes.py`.
  Notes: App orchestration exposes `read_critical_thinking_session(...)`, `generate_critical_thinking_question(...)`, `submit_critical_thinking_answer(...)`, and `retry_critical_thinking_evaluation(...)`. Reads remain separate from question generation, answer submission preserves failed evaluations under the same session id, and route regressions verify retry completes evaluation without regenerating the question.
  Timestamp: 2026-10-10

- [x] Reuse current critical-thinking session on read without session id
  Evidence: `Deep_Reflective_Reader/app/section_task_coordinator.py`; `Deep_Reflective_Reader/scripts/test_app_reading_interaction_orchestration.py`; `PYTHONPATH=Deep_Reflective_Reader Deep_Reflective_Reader/.venv/bin/python Deep_Reflective_Reader/scripts/test_app_reading_interaction_orchestration.py`; `python3 -m py_compile Deep_Reflective_Reader/app/section_task_coordinator.py Deep_Reflective_Reader/scripts/test_app_reading_interaction_orchestration.py`.
  Notes: `read_critical_thinking_session(...)` now loads the latest persisted current session for the resolved target when callers omit `session_id`, and returns `not_generated` only when no current session exists. The response can derive `session_id` from persisted session metadata, so frontend read-first target clicks can reopen existing questions without generating new sessions or needing prior session-id knowledge.
  Timestamp: 2026-10-13

- [x] Return response-ready reading interaction DTOs from app orchestration
  Evidence: `Deep_Reflective_Reader/app/section_task_coordinator.py`; `Deep_Reflective_Reader/scripts/test_app_reading_interaction_orchestration.py`; `Deep_Reflective_Reader/scripts/test_analysis_interaction_routes.py`; `Deep_Reflective_Reader/scripts/test_quiz_interaction_routes.py`; `Deep_Reflective_Reader/scripts/test_critical_thinking_interaction_routes.py`.
  Notes: App orchestration maps analysis artifacts, quiz artifacts, and critical-thinking sessions into `ReadingInteractionResponseDTO` with shared target metadata, status, artifact/session ids, provenance, and metadata handoff. DTOs do not carry frontend-only state such as drawer visibility, menu state, inline expansion, loading state, or pending answer drafts.
  Timestamp: 2026-10-10

- [x] Accept content-block target identity at the app reading-interaction boundary
  Evidence: `Deep_Reflective_Reader/app/section_task_coordinator.py`; `Deep_Reflective_Reader/section_tasks/reading_target_resolver.py`; `Deep_Reflective_Reader/scripts/test_analysis_interaction_routes.py`; `Deep_Reflective_Reader/scripts/test_reading_target_resolver.py`; `.venv` execution of `PYTHONPATH=Deep_Reflective_Reader Deep_Reflective_Reader/.venv/bin/python Deep_Reflective_Reader/scripts/test_analysis_interaction_routes.py`; `.venv` execution of `PYTHONPATH=Deep_Reflective_Reader Deep_Reflective_Reader/.venv/bin/python Deep_Reflective_Reader/scripts/test_reading_target_resolver.py`; `python3 -m py_compile Deep_Reflective_Reader/api_schemas.py Deep_Reflective_Reader/main.py Deep_Reflective_Reader/app/section_task_coordinator.py Deep_Reflective_Reader/section_tasks/reading_target_resolver.py Deep_Reflective_Reader/scripts/test_reading_interaction_api_schemas.py Deep_Reflective_Reader/scripts/test_analysis_interaction_routes.py`.
  Notes: App-level reading interaction methods now accept and forward `content_block_id` to the resolver boundary so route dispatch does not drop block target identity. Resolver support remains explicitly unimplemented for this slice; the app layer does not slice block source text, write block artifacts, mutate task-layout, or add successful block-level generation behavior.
  Timestamp: 2026-10-11

- [x] Project content-block interaction availability during task-unit content reads
  Evidence: `Deep_Reflective_Reader/app/section_task_coordinator.py`; `Deep_Reflective_Reader/section_tasks/reading_interaction_artifact_store.py`; `Deep_Reflective_Reader/scripts/test_task_unit_content_endpoint.py`; `Deep_Reflective_Reader/scripts/test_reading_interaction_artifact_store.py`; `.venv` execution of `PYTHONPATH=Deep_Reflective_Reader Deep_Reflective_Reader/.venv/bin/python Deep_Reflective_Reader/scripts/test_task_unit_content_endpoint.py`; `.venv` execution of `PYTHONPATH=Deep_Reflective_Reader Deep_Reflective_Reader/.venv/bin/python Deep_Reflective_Reader/scripts/test_reading_interaction_artifact_store.py`; `python3 -m py_compile Deep_Reflective_Reader/app/section_task_coordinator.py Deep_Reflective_Reader/section_tasks/reading_interaction_artifact_store.py Deep_Reflective_Reader/scripts/test_task_unit_content_endpoint.py Deep_Reflective_Reader/scripts/test_reading_interaction_artifact_store.py`.
  Notes: App task-unit content assembly now projects menu-ready availability for canonical content-block targets from existing analysis, quiz, and critical-thinking state. The projection remains read-only, emits no artifact payload or UI state, does not persist placeholders, and skips non-canonical segmented render blocks until segmented artifact-target alignment is explicitly implemented.
  Timestamp: 2026-10-11

## Needs Confirmation

No unresolved confirmation items identified in this pass.

## Future Task Policy

### Source-Preserving Enhancement (2026-10-19)

- [x] ENH-APP-01: Add read-only enhancement preflight and explicit discard consent contract.
  Depends on: ENH-PREP-01; public execution remains gated by ENH-DB-01.
  Scope: app orchestration plus main/api_schemas adapters; readiness/impact inventory, server-bound token, explicit consent, stale/missing-consent conflicts and single-flight handoff.
  Acceptance: no preflight writes/OCR/LLM; no paid work without valid consent; token bound to document/source/structure/interaction state; execution availability is explicit. Actual committed-version/timeout reconciliation remains integration acceptance under DB-01/PREP-03, not a claim of successful execution here.
  Evidence: storage-backed coordinator reader; PostgreSQL repeatable-read read-only snapshot and full-row revision fingerprint; embedded/relational inventory with unknown counts; advisory single-flight handoff; explicit unsupported file backend. Eight live disposable-schema tests, nine consent tests and fourteen PDF-mode regressions passed. No production documents modified or live deployment claim.
  Boundary: ready is not execution_available. PREP-02/DB-01 remain mandatory before execution; process-local tokens require renewed consent after restart/worker change.
  Timestamp: 2026-10-19

New future tasks for this module must be added here first as unchecked items:

- [ ] Support content-block-level lookup/read API orchestration
- [ ] Support fine-grained interaction target resolution (sentence/paragraph/content-block)
- [ ] Preserve fail-fast hierarchy lookup semantics for content-block interactions
- [ ] Keep task-layout API lightweight and separate from on-demand rich-content read API
- [ ] Route future content-block interactions through explicit id-based targeting and service boundaries
- [ ] Define document management orchestration service boundary.
  Evidence needed: app-layer service/coordinator contract reads document source/preparation status without entering prepare, OCR, profile, retrieval, task-layout mutation, or interaction generation.
  Notes: File-management UI state remains frontend-local; app responses should be response-ready DTOs only.
  Timestamp: 2026-10-13
- [ ] Define upload orchestration boundary.
  Evidence needed: app-layer upload flow validates/registers raw source candidates, returns lightweight source metadata/status, and stops before prepare unless explicitly requested by a separate flow.
  Notes: Conflict handling should be centralized here rather than in route code or UI.
  Timestamp: 2026-10-13
- [ ] Define document lifecycle cleanup policy.
  Evidence needed: archive/delete/replace behavior coordinates raw source, structured/profile/retrieval artifacts, and reading-interaction artifacts through explicit policy.
  Notes: Destructive behavior must preserve user-scope and raw-file ownership boundaries.
  Timestamp: 2026-10-13
- [x] Define reading-target resolver orchestration for document/chapter/section/task-unit interactions
  Evidence: `Deep_Reflective_Reader/app/section_task_coordinator.py`; `Deep_Reflective_Reader/section_tasks/reading_target_resolver.py`; `Deep_Reflective_Reader/scripts/test_reading_target_resolver.py`; `.venv` execution of `PYTHONPATH=Deep_Reflective_Reader Deep_Reflective_Reader/.venv/bin/python Deep_Reflective_Reader/scripts/test_reading_target_resolver.py`.
  Notes: `SectionTaskCoordinator.resolve_reading_target(...)` loads one base hierarchy snapshot and delegates to `ReadingTargetResolver` for document/book, chapter, section, and task-unit targets. The resolver validates optional parent ids and rejects title-primary lookup, root `sections[]`, `structure_nodes`, and duplicate hierarchy matches.
  Timestamp: 2026-10-08
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
- [x] Add app orchestration methods for summary read/generate/refresh
  Evidence needed: coordinator resolves a hierarchy-aware target once, dispatches summary read without generation, and performs higher-level summary preparation only for explicit generate/refresh paths.
  Notes: Response DTOs should carry summary envelope/payload metadata without raw ancestor text or frontend UI state.
  Evidence: coordinator methods and configured service provider; direct repository read (no prepare), existing profile only, stored version validation. Real route/coordinator tests and ten live targets pass.
  Timestamp: 2026-10-21
- [x] Add app orchestration methods for user-authored notes
  Evidence needed: coordinator supports note read/list/save/delete or an explicitly chosen note route pattern with target validation and length-limit errors before persistence.
  Notes: Notes must not call LLMs, prepare/reparse documents, generate summaries, mutate task-layout, or write profile diagnostics.
  Evidence: read_note/save_note/delete_note dispatch explicit NoteInteractionService, no provider dependency. File/PostgreSQL and actual API tests pass including conflict/length checks.
  Timestamp: 2026-10-21
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

## Task-Layout Readiness / Latency Follow-up (2026-10-15)

Design's dated investigation section records logs, residual timing uncertainty and
the read/write boundary gap. New track: 3/3 implemented. Coordinate with UI LIB-01..05
and loader PDF-MODE-02/03, PDF-FIX-02/03, deferred PDF-AUTO-01; main PDF-MODE-01
and preparation PDF-MODE-04/05 own mode integration. UI now includes LIB-06.
Existing preparation language-cache tasks are not duplicated.

- [x] LAYOUT-01: Instrument the post-preparation task-layout stages and establish matched baselines.
  Acceptance: Request-correlated task-unit/embedding orchestration, cache hit/miss reason, repository and optional anchor-evidence spans; actual external call counts without source/prompt/credentials. Cold, structured-ready and layout-ready measurements in disposable storage distinguish nested spans and residual work. Cover content-free logs with tests; do not attribute all timestamp gaps to remote API latency.
  Evidence: `section_task_coordinator.py`, `logging_config.py`, preparation repository-read spans, embedding SDK spans; `scripts/test_task_layout_logging.py` (5 passing tests), existing persistence-cache and upload/preparation logging regressions. Same-fixture temporary JSON cold/structured-ready/layout-ready baselines are recorded in detailed design; fake external dependencies, not production latency claims. HTTP counters measure completed responses including retry statuses, not transport attempts without responses. Historical requests remain observational context, not matched benchmarks.
  Timestamp: 2026-10-15

- [x] LAYOUT-02: Move preparation/cache repair writes behind explicit reader-readiness orchestration.
  Acceptance: Ordinary task-layout reads do not prepare, resolve/persist new task units, repair IDs, call OCR/LLM/embeddings or write on cache miss. Missing/stale/invalid layout yields actionable readiness failure; explicit prepare-task-layout still creates valid hierarchy/task-unit cache and readback before success. Preserve existing management structured-ready semantics; define any finer readiness response before changing it. Poison-write/LLM tests cover both hits and misses, plus explicit prepare success and Library service handling.
  Depends on: LAYOUT-01 baseline; coordinates main route/preparation/section-task services and UI LIB-03/04 without moving parser authority or heavy content into DTOs.
  Evidence: `ensure_document_task_layout`, `TaskLayoutNotReadyError`, explicit prepare route with persisted readback, Library `prepareReader`, and layout-only refresh after explicit repair/TOC commit. `test_task_layout_readiness.py` (10 tests), persistence-cache, logging, PDF API contract and intercepted Playwright Library tests pass; UI build passes. Management structured-ready semantics unchanged; LAYOUT-03 evidence refactor remains pending.
  Timestamp: 2026-10-16

- [x] LAYOUT-03: Remove implicit page-evidence reload and language detection from ordinary reading.
  Acceptance: No manual source-evidence calls for ordinary task-layout reads, including misses; use character anchors when page evidence was not requested. Explicit TOC/page-evidence reads use source-validated boundary-only evidence without unrelated language/profile generation. Retain manual commit freshness checks and regression coverage for native/scanned/cache/changed-source cases.
  Depends on: LAYOUT-01 and LAYOUT-02. Broader language detection cache priority remains in the existing preparation checklist.
  Evidence: Dedicated persisted boundary-only pipeline read and same-snapshot coordinator handoff. `test_task_layout_page_evidence.py` (8), `test_task_layout_readiness.py` (11), `test_pdf_mode_preparation.py` (10), cache smoke and manual preparation/commit source-evidence regressions pass. Source/hash/schema/span guards fail closed to character anchors; no read-time extraction/LLM/write or relaxed commit validation.
  Timestamp: 2026-10-16

## Maintenance Notes

- This checklist is module memory for completed work.
- It does not replace the module detailed design document.
- It does not replace tests and test evidence.
- It does not replace proposal/HLD decisions and governance context.
