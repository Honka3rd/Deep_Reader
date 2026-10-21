# main.py Checklist

## Purpose

This checklist records completed, code-confirmed or design-confirmed tasks for the `main.py` module.

It is used to:
- preserve module-level implementation memory
- reduce hallucination in future Codex tasks
- prevent context-window compression from losing completed work
- track future task completion explicitly

## Source Documents

- `Deep_Reflective_Reader/main.module-detailed-design.md`
- `Deep_Reflective_Reader/main.py`
- `Deep_Reflective_Reader/proposal.md`
- `Deep_Reflective_Reader/high-level-design.md`

## Rules

- Only completed work is listed as checked.
- Future work must not be added unless explicitly requested.
- If a new task is added later, it must first be added unchecked.
- Once completed, it must be checked in this file.
- Uncertain items must go to `Needs Confirmation`, not the completed checklist.

## Completed Checklist

- [x] Add request-correlated upload and task-layout timing logs.
  Evidence: `main.py`; `logging_config.py`; passing `scripts/test_upload_preparation_logging.py`, `scripts/test_document_upload_route.py`, and `scripts/test_main_request_logging.py`.
  Notes: Records body receive, multipart parse, write, hash, rejection reasons and request elapsed time; propagates generated request ids to worker logs and response headers. Upload still does not prepare documents. No latency cause is inferred.
  Timestamp: 2026-10-14

- [x] Repair storage-aware document deletion and retry after partial file cleanup.
  Evidence: `main.py`; `scripts/test_document_delete_route.py`; passing delete, management, and list/search regressions; local Docker API deletion and both catalog readbacks on 2026-10-14.
  Notes: Confirmed deletion uses the active repository after raw/FAISS cleanup. Database-only residual records can be removed; storage errors do not report accepted. The reported document id 18 and its FAISS directory were removed, with other catalog names unchanged.
  Timestamp: 2026-10-14

- [x] Add task-unit content read endpoint
  Evidence: `Deep_Reflective_Reader/main.py`; `Deep_Reflective_Reader/api_schemas.py`; `Deep_Reflective_Reader/scripts/test_task_unit_content_endpoint.py`
  Notes: 新增 `GET /documents/{doc_name}/task-units/{task_unit_id}/content`，read-only、id-based lookup，404/400 fail-fast error mapping。

- [x] Expose rich content blocks in task-unit content endpoint response
  Evidence: `Deep_Reflective_Reader/main.py`; `Deep_Reflective_Reader/api_schemas.py`; `Deep_Reflective_Reader/app/section_task_coordinator.py`; `Deep_Reflective_Reader/scripts/test_task_unit_content_endpoint.py`
  Notes: endpoint 保留 `content` 同時新增 additive `content_blocks` 回傳，且仍維持既有錯誤語義與 read-only 行為。

- [x] Normalize rich-content endpoint response mapping
  Evidence: `Deep_Reflective_Reader/main.py`; `Deep_Reflective_Reader/api_schemas.py`; `Deep_Reflective_Reader/scripts/test_task_unit_content_endpoint.py`
  Notes: endpoint mapping 改用 official top-level `TaskUnitContentBlockResponse`，序列化形狀穩定且維持 backward-compatible `content + content_blocks` 契約。

- [x] Map content-block artifact target metadata in task-unit content endpoint response
  Evidence: `Deep_Reflective_Reader/main.py`; `Deep_Reflective_Reader/api_schemas.py`; `Deep_Reflective_Reader/scripts/test_task_unit_content_endpoint.py`
  Notes: endpoint 將 shared `artifact_target_refs` 安全 pass-through 到 response schema；target metadata 經 glossary key 篩選後輸出，未引入 artifact repository read/write 或 API breaking change。

- [x] Add explicit segmented content query option to task-unit content endpoint
  Evidence: `Deep_Reflective_Reader/main.py`; `Deep_Reflective_Reader/api_schemas.py`; `Deep_Reflective_Reader/app/section_task_coordinator.py`; `Deep_Reflective_Reader/scripts/test_task_unit_content_endpoint.py`
  Notes: `GET /documents/{doc_name}/task-units/{task_unit_id}/content` 新增 `segmented: bool` query；`segmented=true` 走 shared segmentation helper，`segmented=false/省略` 保持既有 backward-compatible 回傳。

- [x] Stop returning raw task-unit content by default in content endpoint
  Evidence: `Deep_Reflective_Reader/main.py`; `Deep_Reflective_Reader/api_schemas.py`; `Deep_Reflective_Reader/scripts/test_task_unit_content_endpoint.py`
  Notes: endpoint 新增 `include_raw_content` query flag；預設回應不輸出 raw `content`（`content=null`），以 `content_blocks` 作主要 render payload；`include_raw_content=true` 保留 legacy compatibility path。

- [x] Defines FastAPI entrypoint and route registration for prepare/ask/task-layout/summary/quiz/reparse endpoints.
  Evidence: `Deep_Reflective_Reader/main.py; Deep_Reflective_Reader/main.module-detailed-design.md (Main Responsibilities)`
  Notes: Main module is route dispatch boundary for external clients.

- [x] Maps API schemas to coordinator execution paths and response payload construction.
  Evidence: `Deep_Reflective_Reader/main.py; Deep_Reflective_Reader/api_schemas.py; Deep_Reflective_Reader/main.module-detailed-design.md (Route-to-Coordinator Mapping)`
  Notes: Main keeps request/response mapping separate from core business logic.

- [x] Implements explicit projection/mutation route boundary including manual reparse endpoint.
  Evidence: `Deep_Reflective_Reader/main.py; Deep_Reflective_Reader/main.module-detailed-design.md (Projection-Only and Mutation Boundary)`
  Notes: Task-layout remains projection path while reparse/summary/quiz are explicit mutation paths.

- [x] Add lightweight document list/search endpoint
  Evidence: `Deep_Reflective_Reader/main.py`; `Deep_Reflective_Reader/api_schemas.py`; `Deep_Reflective_Reader/scripts/test_document_list_search_api.py`
  Notes: 新增 `GET /documents?q=<query>&limit=<n>`，透過 repository backend selection 讀取 document candidates；read-only、no-heavy-payload、不觸發 prepare/reparse。

- [x] Map task-layout parse provenance to public REST response
  Evidence: `Deep_Reflective_Reader/main.py`; `Deep_Reflective_Reader/api_schemas.py`; `Deep_Reflective_Reader/scripts/test_task_unit_content_endpoint.py`; live `/documents/task-layout` and `/api/documents/task-layout` validation for `Madame Bovary`
  Notes: Route mapping returns optional `parse_provenance` without changing task-layout input, hierarchy response shape, or content payload boundaries.

- [x] Map task-layout anchor evidence for TOC editor prefill
  Evidence: `Deep_Reflective_Reader/main.py`; `Deep_Reflective_Reader/api_schemas.py`; `Deep_Reflective_Reader/scripts/test_task_unit_content_endpoint.py`
  Notes: `/documents/task-layout` pass-through maps optional chapter/section `anchor_evidence` from coordinator DTOs to public response fields without raw text, page text, OCR geometry, task-unit content, hidden mutation, or route-level anchor inference.

- [x] Add source-agnostic manual structure validation route
  Evidence: `Deep_Reflective_Reader/main.py`; `Deep_Reflective_Reader/api_schemas.py`; `Deep_Reflective_Reader/scripts/test_manual_structure_validate_route.py`
  Notes: Adds `POST /documents/manual-structure/validate` as a non-mutating validation/preview route. It maps a validated manual structure plan into normalized entries, lightweight chapter/section preview, and provenance preview without raw text, task content, task-layout mutation, hierarchy persistence, profile diagnostics write-back, artifact writes, or reparse execution.

- [x] Map manual structure reparse commit path to coordinator boundary
  Evidence: `Deep_Reflective_Reader/main.py`; `Deep_Reflective_Reader/api_schemas.py`; `Deep_Reflective_Reader/scripts/test_manual_structure_validate_route.py`
  Notes: `/documents/reparse-structure` recognizes schema-valid `parser_mode=manual_structure` requests, routes them through the manual commit coordinator boundary, and maps the coordinator result/status back to the public reparse response without calling the legacy common/LLM reparse coordinator or treating manual structure as an unknown parser mode.

- [x] Delegate manual structure validation route to document_structure projector
  Evidence: `Deep_Reflective_Reader/main.py`; `Deep_Reflective_Reader/document_structure/manual_structure_projection.py`; `Deep_Reflective_Reader/scripts/test_manual_structure_validate_route.py`
  Notes: `POST /documents/manual-structure/validate` now maps API request DTOs into the deterministic `document_structure` projector and maps the projector result back to the existing public response schema. Route logic no longer owns overlap/projection validation, while remaining non-mutating and lightweight.

- [x] Audit reading interaction REST route exposure
  Evidence: `Deep_Reflective_Reader/main.py`; `Deep_Reflective_Reader/main.module-detailed-design.md`; route decorator scan for `GET/POST` handlers.
  Notes: Confirmed that generic reading-interaction REST routes for `analysis`, target-agnostic quiz artifacts, and `critical_thinking_session` are not exposed yet. Existing `/documents/section-quiz` and `/documents/chapter-quiz` remain legacy section/chapter quiz generation routes, not the new read/generate/submit/retry route family.
  Timestamp: 2026-10-08

- [x] FE-INT-09 Add insight vertical slice route tests
  Evidence: `Deep_Reflective_Reader/main.py`; `Deep_Reflective_Reader/api_schemas.py`; `Deep_Reflective_Reader/app/section_task_coordinator.py`; `Deep_Reflective_Reader/scripts/test_analysis_interaction_routes.py`; `PYTHONPATH=Deep_Reflective_Reader Deep_Reflective_Reader/.venv/bin/python Deep_Reflective_Reader/scripts/test_analysis_interaction_routes.py`; `python3 -m py_compile Deep_Reflective_Reader/main.py Deep_Reflective_Reader/scripts/test_analysis_interaction_routes.py`.
  Notes: Adds `POST /documents/reading-interactions/insight/read`, `/insight/generate`, and `/insight/refresh`. Route tests prove read dispatches only to app read and returns `not_generated` without payload, generate/refresh dispatch explicitly to their write paths, shared envelope fields map to public schema, and compact analysis payloads preserve reasoning/interpretation behavior for inline insight rendering.
  Timestamp: 2026-10-10

- [x] FE-INT-07 Define route family and HTTP status mapping
  Evidence: `Deep_Reflective_Reader/main.py`; `Deep_Reflective_Reader/scripts/test_analysis_interaction_routes.py`; `Deep_Reflective_Reader/main.module-detailed-design.md`; `PYTHONPATH=Deep_Reflective_Reader Deep_Reflective_Reader/.venv/bin/python Deep_Reflective_Reader/scripts/test_analysis_interaction_routes.py`; `python3 -m py_compile Deep_Reflective_Reader/main.py Deep_Reflective_Reader/scripts/test_analysis_interaction_routes.py`.
  Notes: Adds route-level reading interaction HTTP status helpers and applies them to insight read/generate/refresh. Malformed schema requests remain FastAPI `422`; missing document/target maps to `404`; recoverable envelope states `not_generated`, `completed`, `insufficient_content`, `question_generated`, `answer_submitted`, and `evaluation_failed` map to `200`; `stale_target` maps to `409`; `validation_failed` maps to `422`; `generation_failed` maps to `502`; unexpected route exceptions map to `500`. Route handlers still only validate schema, dispatch to app orchestration, and map responses.
  Timestamp: 2026-10-10

- [x] FE-INT-08 Add no-auto-generation route regressions for interaction reads
  Evidence: `Deep_Reflective_Reader/scripts/test_analysis_interaction_routes.py`; `Deep_Reflective_Reader/main.py`; `PYTHONPATH=Deep_Reflective_Reader Deep_Reflective_Reader/.venv/bin/python Deep_Reflective_Reader/scripts/test_analysis_interaction_routes.py`; `python3 -m py_compile Deep_Reflective_Reader/main.py Deep_Reflective_Reader/scripts/test_analysis_interaction_routes.py`.
  Notes: Adds a read-only poison coordinator regression for the currently exposed insight read route. The test proves absent artifact reads return HTTP `200` with envelope status `not_generated` and no payload while the route is limited to `read_analysis_artifact`; any accidental coordinator access for generation, refresh, session creation, prepare/reparse, task-layout mutation, or profile diagnostics write-back would fail the regression. Planned quiz and critical-thinking read routes remain future work and must reuse the same no-auto-generation policy when exposed.
  Timestamp: 2026-10-10

- [x] FE-INT-10 Add quiz vertical slice route tests
  Evidence: `Deep_Reflective_Reader/main.py`; `Deep_Reflective_Reader/scripts/test_quiz_interaction_routes.py`; `Deep_Reflective_Reader/api_schemas.py`; `Deep_Reflective_Reader/app/section_task_coordinator.py`; `PYTHONPATH=Deep_Reflective_Reader Deep_Reflective_Reader/.venv/bin/python Deep_Reflective_Reader/scripts/test_quiz_interaction_routes.py`; `python3 -m py_compile Deep_Reflective_Reader/main.py Deep_Reflective_Reader/scripts/test_quiz_interaction_routes.py`.
  Notes: Adds generic `POST /documents/reading-interactions/quiz/read`, `/quiz/generate`, and `/quiz/refresh` route mapping. Route tests prove quiz read returns `not_generated` without payload or write dispatch, generate/refresh call the new generic quiz app orchestration methods, legacy `/section-quiz` and `/chapter-quiz` generation methods are not used, drawer quiz payload items are mapped with stable item ids/options/answers, and invalid request counts or invalid generated item types fail with `422`.
  Timestamp: 2026-10-10

- [x] Add explicit quiz generate/refresh route
  Evidence: `Deep_Reflective_Reader/main.py`; `Deep_Reflective_Reader/scripts/test_quiz_interaction_routes.py`; `PYTHONPATH=Deep_Reflective_Reader Deep_Reflective_Reader/.venv/bin/python Deep_Reflective_Reader/scripts/test_quiz_interaction_routes.py`; `python3 -m py_compile Deep_Reflective_Reader/main.py Deep_Reflective_Reader/scripts/test_quiz_interaction_routes.py`.
  Notes: `POST /documents/reading-interactions/quiz/generate` and `/quiz/refresh` validate the shared reading target request, dispatch through `SectionTaskCoordinator.generate_quiz_artifact(...)` and `refresh_quiz_artifact(...)`, map completed quiz artifacts into `QuizInteractionResponse`, preserve prompt instruction version handoff, and keep the legacy section/chapter quiz routes separate from the frontend drawer API.
  Timestamp: 2026-10-10

- [x] Add read-only reading interaction artifact routes
  Evidence: `Deep_Reflective_Reader/main.py`; `Deep_Reflective_Reader/scripts/test_analysis_interaction_routes.py`; `Deep_Reflective_Reader/scripts/test_quiz_interaction_routes.py`; `Deep_Reflective_Reader/scripts/test_critical_thinking_interaction_routes.py`; `.venv` execution of all three route regression scripts; `python3 -m py_compile Deep_Reflective_Reader/main.py Deep_Reflective_Reader/scripts/test_critical_thinking_interaction_routes.py`.
  Notes: Completes the read route family for insight, quiz, and critical-thinking sessions. The critical-thinking read route returns `not_generated` without session id or payload when no session is requested/found, and the route test proves it does not dispatch generation, answer submission, or evaluation retry.
  Timestamp: 2026-10-10

- [x] FE-INT-11 Add critical-thinking session route tests
  Evidence: `Deep_Reflective_Reader/main.py`; `Deep_Reflective_Reader/api_schemas.py`; `Deep_Reflective_Reader/scripts/test_critical_thinking_interaction_routes.py`; `Deep_Reflective_Reader/scripts/test_reading_interaction_api_schemas.py`; `PYTHONPATH=Deep_Reflective_Reader Deep_Reflective_Reader/.venv/bin/python Deep_Reflective_Reader/scripts/test_critical_thinking_interaction_routes.py`; `PYTHONPATH=Deep_Reflective_Reader Deep_Reflective_Reader/.venv/bin/python Deep_Reflective_Reader/scripts/test_reading_interaction_api_schemas.py`; `python3 -m py_compile Deep_Reflective_Reader/main.py Deep_Reflective_Reader/api_schemas.py Deep_Reflective_Reader/scripts/test_critical_thinking_interaction_routes.py`.
  Notes: Adds generic critical-thinking `read`, `generate-question`, `submit-answer`, and `retry-evaluation` route mapping. Route tests cover no-generation missing reads, generated question session persistence, existing-session read, evaluation failure preserving submitted answer with retry eligibility, retry completion without regenerating the question, and malformed submit rejection before dispatch.
  Timestamp: 2026-10-10

- [x] Add critical-thinking session routes
  Evidence: `Deep_Reflective_Reader/main.py`; `Deep_Reflective_Reader/api_schemas.py`; `Deep_Reflective_Reader/scripts/test_critical_thinking_interaction_routes.py`; `PYTHONPATH=Deep_Reflective_Reader Deep_Reflective_Reader/.venv/bin/python Deep_Reflective_Reader/scripts/test_critical_thinking_interaction_routes.py`; `python3 -m py_compile Deep_Reflective_Reader/main.py Deep_Reflective_Reader/api_schemas.py Deep_Reflective_Reader/scripts/test_critical_thinking_interaction_routes.py`.
  Notes: Exposes `POST /documents/reading-interactions/critical-thinking/read`, `/generate-question`, `/submit-answer`, and `/retry-evaluation`. Submit/retry requests carry the shared reading target plus `session_id`, preserving the hierarchy-aware app/session-store boundary rather than requiring route-level global session lookup.
  Timestamp: 2026-10-10

- [x] FE-INT-12 Synchronize implementation documentation and checklists after API exposure
  Evidence: `Deep_Reflective_Reader/main.module-detailed-design.md`; `Deep_Reflective_Reader/main.module-checklist.md`; `Deep_Reflective_Reader/api_schemas.module-detailed-design.md`; `Deep_Reflective_Reader/api_schemas.module-checklist.md`; `Deep_Reflective_Reader/app/module-detailed-design.md`; `Deep_Reflective_Reader/app/module-checklist.md`; `Deep_Reflective_Reader/scripts/module-detailed-design.md`; `Deep_Reflective_Reader/scripts/module-checklist.md`; `Deep_Reflective_Reader/main.py`; `Deep_Reflective_Reader/api_schemas.py`; route regressions for analysis, quiz, and critical-thinking interactions.
  Notes: Module memory now reflects the code-confirmed public route names, schema names, app orchestration methods, route regression coverage, and route/app/schema boundaries for insight, quiz, and critical-thinking API exposure. `progress.md` was intentionally not updated because this pass was not a progress-sync task.
  Timestamp: 2026-10-10

- [x] Preserve content-block target identity through reading-interaction routes
  Evidence: `Deep_Reflective_Reader/main.py`; `Deep_Reflective_Reader/api_schemas.py`; `Deep_Reflective_Reader/scripts/test_analysis_interaction_routes.py`; `Deep_Reflective_Reader/scripts/test_quiz_interaction_routes.py`; `Deep_Reflective_Reader/scripts/test_critical_thinking_interaction_routes.py`; `.venv` execution of analysis, quiz, and critical-thinking route regression scripts; `python3 -m py_compile Deep_Reflective_Reader/scripts/test_analysis_interaction_routes.py Deep_Reflective_Reader/scripts/test_quiz_interaction_routes.py Deep_Reflective_Reader/scripts/test_critical_thinking_interaction_routes.py Deep_Reflective_Reader/main.py Deep_Reflective_Reader/api_schemas.py`.
  Notes: Shared reading-interaction route mapping forwards schema-normalized `content_block_id` with `target_level=content_block` to the app boundary and maps returned shared envelopes with the same block identity. Coverage now spans insight read/generate/refresh, quiz read/generate/refresh, and critical-thinking read/generate-question/submit-answer/retry-evaluation while keeping route code free of hierarchy search, prompt/context selection, artifact persistence internals, legacy quiz dispatch, and UI state.
  Timestamp: 2026-10-11

## Needs Confirmation

No unresolved confirmation items identified in this pass.

## Future Task Policy

New future tasks for this module must be added here first as unchecked items:

- [x] Suppress successful health-check request lifecycle logs
  Evidence: `Deep_Reflective_Reader/main.py`; `Deep_Reflective_Reader/scripts/test_main_request_logging.py`; live API verification that repeated `/health` calls do not append `request_completed path=/health` lines.
  Notes: Health probes are high-frequency operational noise and can hide useful OCR/prepare logs.
- [x] Add explicit manual structure reparse commit route or extend reparse route
  Evidence needed: route supports `manual_structure` parser mode or equivalent source-agnostic commit path and maps failures to stable HTTP statuses.
  Evidence: `Deep_Reflective_Reader/main.py`; `Deep_Reflective_Reader/api_schemas.py`; `Deep_Reflective_Reader/app/section_task_coordinator.py`; `Deep_Reflective_Reader/scripts/test_manual_structure_commit_source_evidence.py`.
  Notes: `/documents/reparse-structure` now accepts `parser_mode=manual_structure` through the existing explicit mutation endpoint and maps coordinator statuses including 200 success, 400 malformed request, 404 missing source, 409 stale source hash, 422 unprojectable plan/draft anchor failure, and 500 persistence failure. `/documents/task-layout` remains projection-only and does not accept edits or trigger hidden reparse.
- [x] Map page-backed manual-structure validation and commit failures
  Evidence: `Deep_Reflective_Reader/main.py`; `Deep_Reflective_Reader/api_schemas.py`; `Deep_Reflective_Reader/scripts/test_manual_structure_validate_route.py`; `Deep_Reflective_Reader/scripts/test_manual_structure_commit_source_evidence.py`.
  Notes: `/documents/reparse-structure` with `parser_mode=manual_structure` returns stable page-backed manual-structure statuses: `422` for missing/unsupported page evidence, `422` for out-of-range/unprojectable page anchors, `409` for stale source evidence, and `200` for successful page-backed commit mapping. `/documents/task-layout` remains projection-only and does not accept edits or trigger hidden reparse.

- [ ] Separate existing-layout read flow from prepare-then-read flow
  Evidence needed: route behavior or client-facing contract makes it clear that `/documents/task-layout` is the preferred existing-layout read path, while `/documents/prepare-task-layout` is reserved for first-time prepare, explicit repair, or fallback.
  Notes: Repeated UI selection of an already prepared document should not implicitly enter OCR/language/profile or LLM-backed work merely to display the current layout. Route observability should report whether prepare-then-read reused existing structured artifacts or performed expensive preparation.

- [x] Expose summary reading-interaction route family
  Evidence needed: routes map summary read/generate/refresh requests through app orchestration, reuse shared envelope/status mapping, and keep read paths no-auto-generation.
  Notes: Route handlers should not build prompts, prepare ancestor summaries, persist artifacts directly, or expose raw source/ancestor text.
  Evidence: POST /documents/reading-interactions/summary/{read,generate,refresh}, shared envelope/error mapping; real coordinator and live two-book acceptance pass. Failed refresh returns 502 without replacing saved output.
  Timestamp: 2026-10-21

- [x] Expose note route family for user-authored annotations
  Evidence needed: routes map note read/list/save/delete or equivalent explicit note operations through app orchestration with stable validation and HTTP status mapping.
  Notes: Note routes must not call LLMs, prepare/reparse, generate summaries, mutate task-layout, or persist frontend-only UI state.
  Evidence: POST /documents/reading-interactions/note/{read,save,delete}, strict schemas and 409 conflict mapping. Actual API tests and ten real-book targets verify persistence and idempotent retry.
  Timestamp: 2026-10-21

- [x] Add explicit document upload route.
  Evidence: `Deep_Reflective_Reader/main.py`; `Deep_Reflective_Reader/api_schemas.py`; `Deep_Reflective_Reader/scripts/test_document_upload_route.py`; validation: `PYTHONPATH=Deep_Reflective_Reader Deep_Reflective_Reader/.venv/bin/python Deep_Reflective_Reader/scripts/test_document_upload_route.py`; `PYTHONPATH=Deep_Reflective_Reader Deep_Reflective_Reader/.venv/bin/python Deep_Reflective_Reader/scripts/test_document_list_search_api.py`; `Deep_Reflective_Reader/.venv/bin/python -m py_compile Deep_Reflective_Reader/main.py Deep_Reflective_Reader/scripts/test_document_upload_route.py`.
  Notes: Adds `POST /documents/upload` as an explicit multipart raw-source write route for `.txt` and `.pdf` candidates. The route stores raw source bytes under the raw document directory, returns lightweight `DocumentUploadResponse` metadata, supports `reject`, `replace`, and `create_distinct` conflict policies, and returns typed failure reasons for duplicate, unsupported type, unsafe filename, missing source, validation, too-large, and storage failures. `create_distinct` preserves the existing source and returns the backend-owned unique filename/doc_name, such as `Book (1).txt` / `Book (1)`. Upload does not silently prepare, reparse, run OCR/profile, build retrieval, call task-layout, or generate interactions.
  Timestamp: 2026-10-13

- [x] Add document management status/list route contract.
  Evidence: `Deep_Reflective_Reader/main.py`; `Deep_Reflective_Reader/api_schemas.py`; `Deep_Reflective_Reader/scripts/test_document_management_route.py`; validation: `PYTHONPATH=Deep_Reflective_Reader Deep_Reflective_Reader/.venv/bin/python Deep_Reflective_Reader/scripts/test_document_management_route.py`; `PYTHONPATH=Deep_Reflective_Reader Deep_Reflective_Reader/.venv/bin/python Deep_Reflective_Reader/scripts/test_document_list_search_api.py`; `Deep_Reflective_Reader/.venv/bin/python -m py_compile Deep_Reflective_Reader/main.py Deep_Reflective_Reader/api_schemas.py Deep_Reflective_Reader/scripts/test_document_management_route.py`.
  Notes: Adds `GET /documents/management` as an additive lightweight management/status read route. It keeps `GET /documents` compact, merges structured document readiness with raw source metadata, exposes source/preparation status for the management UI, and does not return raw text, hierarchy, task-unit content, content blocks, profile payloads, retrieval records, or artifacts.
  Timestamp: 2026-10-13

- [x] Add explicit document lifecycle mutation route policy.
  Evidence: `Deep_Reflective_Reader/main.module-detailed-design.md`; `Deep_Reflective_Reader/api_schemas.py`; `Deep_Reflective_Reader/api_schemas.module-detailed-design.md`; `Deep_Reflective_Reader/scripts/test_document_lifecycle_api_schemas.py`; validation: `PYTHONPATH=Deep_Reflective_Reader Deep_Reflective_Reader/.venv/bin/python Deep_Reflective_Reader/scripts/test_document_lifecycle_api_schemas.py`; `Deep_Reflective_Reader/.venv/bin/python -m py_compile Deep_Reflective_Reader/api_schemas.py Deep_Reflective_Reader/scripts/test_document_lifecycle_api_schemas.py`.
  Notes: Defines lifecycle mutation route policy and schema boundary for future `archive`, `restore`, `delete`, and `rename` actions. Destructive routes are intentionally not implemented in this task; future route handlers must stay explicit, use stable lifecycle response statuses, and delegate raw-source / structured-artifact / derived-resource cleanup outside `main.py`.
  Timestamp: 2026-10-13

- [x] Implement explicit document delete route.
  Evidence: `Deep_Reflective_Reader/main.py`; `Deep_Reflective_Reader/api_schemas.py`; `Deep_Reflective_Reader/scripts/test_document_delete_route.py`; validation: `PYTHONPATH=Deep_Reflective_Reader Deep_Reflective_Reader/.venv/bin/python Deep_Reflective_Reader/scripts/test_document_delete_route.py`; `PYTHONPATH=Deep_Reflective_Reader Deep_Reflective_Reader/.venv/bin/python Deep_Reflective_Reader/scripts/test_document_management_route.py`; `Deep_Reflective_Reader/.venv/bin/python -m py_compile Deep_Reflective_Reader/main.py Deep_Reflective_Reader/scripts/test_document_delete_route.py`.
  Notes: Adds `POST /documents/lifecycle/delete` as an explicit destructive lifecycle route requiring `action=delete` and `confirm=true`. The route removes matching file-backed raw source, structured JSON artifact, and derived FAISS/profile directory when present, maps missing/unconfirmed/wrong-action cases to typed lifecycle responses, and does not hide deletion behind list/read/open/upload/task-layout routes or trigger prepare/reparse/OCR/profile/retrieval/interaction generation.
  Timestamp: 2026-10-13

- [x] Implement explicit document rename route.
  Evidence: `Deep_Reflective_Reader/main.py`; `Deep_Reflective_Reader/api_schemas.py`; `Deep_Reflective_Reader/scripts/test_document_rename_route.py`; validation: `PYTHONPATH=Deep_Reflective_Reader Deep_Reflective_Reader/.venv/bin/python Deep_Reflective_Reader/scripts/test_document_rename_route.py`; `PYTHONPATH=Deep_Reflective_Reader Deep_Reflective_Reader/.venv/bin/python Deep_Reflective_Reader/scripts/test_document_delete_route.py`; `Deep_Reflective_Reader/.venv/bin/python -m py_compile Deep_Reflective_Reader/main.py Deep_Reflective_Reader/scripts/test_document_rename_route.py`.
  Notes: Adds `POST /documents/lifecycle/rename` as an explicit lifecycle mutation route requiring `action=rename` and `new_doc_name`. The route renames matching file-backed raw source, structured JSON artifact, and derived FAISS/profile directory when present, updates lightweight top-level structured JSON identity metadata, maps missing/wrong-action/conflict cases to typed lifecycle responses, and does not simulate rename through upload replace/create-distinct behavior.
  Timestamp: 2026-10-13

- [x] Add batch task-unit content read route
  Evidence: `Deep_Reflective_Reader/main.py`; `Deep_Reflective_Reader/api_schemas.py`; `Deep_Reflective_Reader/app/section_task_coordinator.py`; `Deep_Reflective_Reader/scripts/test_task_unit_content_endpoint.py`; validation with `.venv` task-unit content endpoint regression and `py_compile`.
  Notes: `POST /documents/{doc_name}/task-units/content` maps an ordered batch request to coordinator content lookup once, returns ordered per-task-unit content responses, preserves the single task-unit content endpoint as fallback, and covers segmented/raw-content options. It remains read-only and does not expand task-layout payloads, trigger prepare/reparse, mutate profile diagnostics, or write artifacts.
  Timestamp: 2026-10-05

After implementation, the task owner must update this checklist and mark the task as completed:

No coding task should be considered complete unless the corresponding module checklist is updated.

## PDF Extraction Mode Contract (2026-10-15)

- [x] PDF-MODE-01: Expose a validated fast/ocr/auto extraction contract on explicit preparation routes.
  Acceptance: Add pdf_extraction_mode with omitted default fast on prepare and prepare-task-layout, validate enum before expensive work, pass through app/preparation, and expose effective mode/provenance plus typed no-text/engine/quality/budget failures. Keep base/free_qa and structured_parser_mode independent. Non-PDF omission/fast is a no-op; explicit ocr/auto for non-PDF is rejected clearly. Upload remains raw-only; no mode-triggered mutation on list/Open/read routes. Contract tests cover omitted/invalid/non-PDF modes and both explicit endpoints; specify legacy unknown-provenance artifacts without relabeling them fast.
  Depends on: Interface agreement with PDF-MODE-04/05; runtime delivery requires their propagation. This task owns schemas/routes and supporting app pass-through, not loader policy.
  Evidence: api_schemas.py; main.py; doc_loaders/pdf_extraction_contract.py; document_preparation/document_preparation_pipeline.py and prepared_document_assets.py; scripts/test_pdf_extraction_mode_contract.py (11 tests passed). Existing raw-load error, structured-reuse, lazy-layout and upload/preparation logging regressions also passed.
  Delivery boundary: Contract only. Both public PDF prepare routes return typed 501 mode_unavailable before any cache/mutation, including omitted fast; TXT fast remains operational and TXT ocr/auto returns 422. Internal legacy callers remain unknown-provenance. Engine failure mappings are tested with injected errors; executors/cache integration and live deployment are not claimed.
  Timestamp: 2026-10-15

## Maintenance Notes

- This checklist is module memory for completed work.
- It does not replace the module detailed design document.
- It does not replace tests and test evidence.
- It does not replace proposal/HLD decisions and governance context.
