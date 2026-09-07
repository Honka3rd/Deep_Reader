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

## Needs Confirmation

No unresolved confirmation items identified in this pass.

## Future Task Policy

New future tasks for this module must be added here first as unchecked items:

- [x] Suppress successful health-check request lifecycle logs
  Evidence: `Deep_Reflective_Reader/main.py`; `Deep_Reflective_Reader/scripts/test_main_request_logging.py`; live API verification that repeated `/health` calls do not append `request_completed path=/health` lines.
  Notes: Health probes are high-frequency operational noise and can hide useful OCR/prepare logs.

After implementation, the task owner must update this checklist and mark the task as completed:

No coding task should be considered complete unless the corresponding module checklist is updated.

## Maintenance Notes

- This checklist is module memory for completed work.
- It does not replace the module detailed design document.
- It does not replace tests and test evidence.
- It does not replace proposal/HLD decisions and governance context.
