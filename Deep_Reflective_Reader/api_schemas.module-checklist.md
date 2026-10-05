# api_schemas.py Checklist

## Purpose

This checklist records completed, code-confirmed or design-confirmed tasks for the `api_schemas.py` module.

It is used to:
- preserve module-level implementation memory
- reduce hallucination in future Codex tasks
- prevent context-window compression from losing completed work
- track future task completion explicitly

## Source Documents

- `Deep_Reflective_Reader/api_schemas.module-detailed-design.md`
- `Deep_Reflective_Reader/api_schemas.py`
- `Deep_Reflective_Reader/proposal.md`
- `Deep_Reflective_Reader/high-level-design.md`

## Rules

- Only completed work is listed as checked.
- Future work must not be added unless explicitly requested.
- If a new task is added later, it must first be added unchecked.
- Once completed, it must be checked in this file.
- Uncertain items must go to `Needs Confirmation`, not the completed checklist.

## Completed Checklist

- [x] Define task-unit content API request/response schema
  Evidence: `Deep_Reflective_Reader/api_schemas.py`; `Deep_Reflective_Reader/main.py`; `Deep_Reflective_Reader/scripts/test_task_unit_content_endpoint.py`
  Notes: 新增 `TaskUnitContentResponse`，支援 frontend 按需讀取 render content，且不回填 task-layout heavy payload。

- [x] Add backward-compatible content block payload to task-unit content response schema
  Evidence: `Deep_Reflective_Reader/api_schemas.py`; `Deep_Reflective_Reader/main.py`; `Deep_Reflective_Reader/scripts/test_task_unit_content_endpoint.py`; `Deep_Reflective_Reader/shared/task_unit_model.py`
  Notes: 在保留 `content: str` 下新增 `content_blocks` schema，對既有客戶端保持 backward-compatible additive evolution。

- [x] Normalize official rich-content API response schema
  Evidence: `Deep_Reflective_Reader/api_schemas.py`; `Deep_Reflective_Reader/main.py`; `Deep_Reflective_Reader/scripts/test_task_unit_content_endpoint.py`; `Deep_Reflective_Reader/api_schemas.module-detailed-design.md`
  Notes: 正式建立 top-level `TaskUnitContentBlockResponse` 與標準化 `TaskUnitContentResponse`（`content + content_blocks`）；保持 additive evolution，無 API breaking change/無 task-layout heavy payload 擴張。

- [x] Validate content-block artifact target metadata response schema
  Evidence: `Deep_Reflective_Reader/api_schemas.py`; `Deep_Reflective_Reader/main.py`; `Deep_Reflective_Reader/scripts/test_task_unit_content_endpoint.py`; `Deep_Reflective_Reader/scripts/test_shared_task_unit_content_blocks.py`
  Notes: 新增 `ArtifactTargetRefResponse` 並重用 shared `ArtifactTargetLevel`；非法 `target_level` fail-fast，`content_block`/`task_unit` level 目標欄位需符合最小約束；metadata key vocabulary 收斂為 approved glossary。

- [x] Reuse shared artifact target level contract to remove duplicated enum definitions
  Evidence: `Deep_Reflective_Reader/api_schemas.py`; `Deep_Reflective_Reader/shared/artifact_target_model.py`; `Deep_Reflective_Reader/shared/task_unit_model.py`; `Deep_Reflective_Reader/scripts/test_task_unit_content_endpoint.py`
  Notes: 移除 API schema 層重複 enum 定義，改為共用 shared single-source `ArtifactTargetLevel`，降低跨模組 vocabulary drift 風險。

- [x] Preserve backward-compatible schema for segmented task-unit content response
  Evidence: `Deep_Reflective_Reader/api_schemas.py`; `Deep_Reflective_Reader/main.py`; `Deep_Reflective_Reader/app/section_task_coordinator.py`; `Deep_Reflective_Reader/scripts/test_task_unit_content_endpoint.py`
  Notes: `GetTaskUnitContentRequest` 新增 `segmented: bool = False` 顯式 opt-in；`TaskUnitContentResponse` 保持 `content + content_blocks` additive contract，不移除 `content`、不引入 breaking change。

- [x] Reduce raw content exposure in task-unit content API response
  Evidence: `Deep_Reflective_Reader/api_schemas.py`; `Deep_Reflective_Reader/main.py`; `Deep_Reflective_Reader/scripts/test_task_unit_content_endpoint.py`
  Notes: `TaskUnitContentResponse.content` 改為 nullable compatibility/debug field；新增 `include_raw_content: bool = False` request flag，預設 content-block-first（`content` 不填），`include_raw_content=true` 才回傳 legacy raw content；`content_blocks` 仍為必備 render payload。

- [x] Defines external API schemas used by request and response boundaries.
  Evidence: `Deep_Reflective_Reader/api_schemas.py; Deep_Reflective_Reader/api_schemas.module-detailed-design.md (Main Responsibilities)`
  Notes: Root Python module documented as an API contract boundary.

- [x] Defines task-layout response contract with chapters-first projection fields and diagnostics response model.
  Evidence: `Deep_Reflective_Reader/api_schemas.py; Deep_Reflective_Reader/api_schemas.module-detailed-design.md (Important Data Structures / Contracts)`
  Notes: Task-layout public contract is represented in schema layer.

- [x] Defines chapter summary/quiz request validation boundary for id/title target fields.
  Evidence: `Deep_Reflective_Reader/api_schemas.py; Deep_Reflective_Reader/api_schemas.module-detailed-design.md (Main Responsibilities)`
  Notes: Schema-level validation supports chapter targeting constraints.

- [x] Define lightweight document list/search API schemas
  Evidence: `Deep_Reflective_Reader/api_schemas.py`; `Deep_Reflective_Reader/main.py`; `Deep_Reflective_Reader/scripts/test_document_list_search_api.py`
  Notes: 新增 `DocumentListItemResponse` / `DocumentListResponse`；response 只返回 document candidates，不暴露 hierarchy/content/diagnostics heavy payload。

- [x] Define task-layout parse provenance response schema
  Evidence: `Deep_Reflective_Reader/api_schemas.py`; `Deep_Reflective_Reader/main.py`; `Deep_Reflective_Reader/scripts/test_task_unit_content_endpoint.py`; live `/api/documents/task-layout` validation
  Notes: `DocumentTaskLayoutResponse` includes optional `parse_provenance` with requested/effective parser mode and fallback metadata. The field is lightweight observability metadata and does not expose raw text or task-unit content.

- [x] Define task-layout anchor evidence response schema for TOC editing
  Evidence: `Deep_Reflective_Reader/api_schemas.py`; `Deep_Reflective_Reader/main.py`; `Deep_Reflective_Reader/scripts/test_task_unit_content_endpoint.py`
  Notes: `AnchorEvidenceResponse` is exposed as optional metadata on `DocumentTaskLayoutChapterResponse` and `SectionTaskLayoutResponse`. It carries anchor type/status/reason, char offsets, and optional page indices/labels without raw text, page text, OCR boxes, task-unit content, or a second hierarchy source.

## Needs Confirmation

No unresolved confirmation items identified in this pass.

## Future Task Policy

New future tasks for this module must be added here first as unchecked items:

- [ ] Design rich task-unit content response schema (future direction, not implemented)
- [ ] Define content-block artifact metadata schema (future direction, not implemented)
- [ ] Define backward-compatible content response evolution strategy (future direction, not implemented)
- [ ] Define generic reading target request schema
  Evidence needed: schema validates document, chapter, section, and task-unit targets using ids as primary identity, with optional parent ids only for consistency checks.
  Notes: No title-primary or legacy fallback semantics should be expressed in public schema.
  Timestamp: 2026-10-05
- [ ] Define analysis artifact read/generate schemas
  Evidence needed: schemas distinguish persisted read response, generate request, successful artifact payload, missing/not-generated status, insufficient-content status, and validation/generation failure.
  Notes: Analysis payload should be structured, versioned, and target-aware.
  Timestamp: 2026-10-05
- [ ] Define quiz artifact schemas with strict validation envelope
  Evidence needed: schemas support `short_answer`, `multiple_choice`, and `true_false` only, enforce max item count, include answers, and support insufficient-content status.
  Notes: The model chooses type mix; backend schema validation enforces the allowed surface.
  Timestamp: 2026-10-05
- [ ] Define critical-thinking session schemas
  Evidence needed: schemas cover question generation, answer submission, evaluation retry, and statuses `question_generated`, `insufficient_content`, `answer_submitted`, `evaluation_failed`, and `completed`.
  Notes: Generated question must be serializable before the user answers.
  Timestamp: 2026-10-05
- [ ] Define artifact-aware interaction metadata schema
  Evidence needed: schemas can represent artifact context mode, referenced artifact ids, referenced types/target levels, coverage counts, and deduplication/abstraction hint flags.
  Notes: Metadata must distinguish primary source evidence from secondary artifact references.
  Timestamp: 2026-10-05
- [ ] Prevent nested child artifact payload expansion in higher-level responses
  Evidence needed: response schemas expose referenced artifact metadata without embedding full lower-level artifact payloads by default.
  Notes: This keeps higher-level artifact responses bounded and avoids turning artifacts into source truth.
  Timestamp: 2026-10-05
- [x] Design batch task-unit content request/response schema
  Evidence: `Deep_Reflective_Reader/api_schemas.py`; `Deep_Reflective_Reader/main.py`; `Deep_Reflective_Reader/scripts/test_task_unit_content_endpoint.py`; validation with `.venv` task-unit content endpoint regression and `py_compile`.
  Notes: `BatchTaskUnitContentRequest` accepts ordered non-empty unique `task_unit_ids` plus `segmented` / `include_raw_content`; `BatchTaskUnitContentResponse` returns ordered per-task-unit `TaskUnitContentResponse` items while preserving the single endpoint schema contract. Batch content remains an on-demand content API path, not a task-layout payload expansion.
  Timestamp: 2026-10-05
- [x] Design source-agnostic manual structure override request schema
  Evidence: `Deep_Reflective_Reader/api_schemas.py`; `Deep_Reflective_Reader/scripts/test_manual_structure_api_schemas.py`
  Notes: Adds `ManualStructureAnchorRequest`, `ManualStructureEntryRequest`, `ManualStructurePlanRequest`, and `ManualStructureValidationRequest`. Request schema supports typed `char_range` / `page_range` anchors, trims doc/title metadata, rejects empty titles, rejects mixed anchor fields, rejects orphan sections, and enforces current `chapter -> section` maximum depth.
- [x] Design manual structure validation/preview response schema
  Evidence: `Deep_Reflective_Reader/api_schemas.py`; `Deep_Reflective_Reader/scripts/test_manual_structure_api_schemas.py`
  Notes: Adds `ManualStructureValidationResponse` and nested issue/normalized-entry/preview/provenance DTOs. Response schema exposes validation status, normalized entries, warnings/errors, lightweight preview chapter/section shape, and `manual_structure` provenance preview without raw text or task content; issue code validation covers malformed payload, unsupported anchor, out-of-range anchor, overlapping range, empty projected range, invalid level sequence, unsupported depth, and stale source evidence.
- [x] Design manual structure commit reparse schema evolution
  Evidence: `Deep_Reflective_Reader/api_schemas.py`; `Deep_Reflective_Reader/scripts/test_manual_structure_api_schemas.py`
  Notes: `ReparseDocumentStructureRequest` now accepts normalized `parser_mode=manual_structure` with a required `manual_structure` plan, preserves existing `common` / `llm_enhanced` modes, rejects manual plans for non-manual modes, and remains schema-only without task-layout payload expansion or runtime/persistence behavior.
- [x] Define page-backed manual-structure validation/commit schema semantics
  Evidence: `Deep_Reflective_Reader/api_schemas.py`; `Deep_Reflective_Reader/main.py`; `Deep_Reflective_Reader/scripts/test_manual_structure_api_schemas.py`; `Deep_Reflective_Reader/scripts/test_manual_structure_validate_route.py`.
  Notes: Schema-valid `page_range` remains explicit vocabulary while route regressions distinguish missing/unsupported page evidence (`422`), out-of-range page anchors (`422`), stale source evidence (`409`), and successful page-backed commit response mapping (`200`). `char_range` remains the universal fallback and task-layout remains lightweight/read-only.

After implementation, the task owner must update this checklist and mark the task as completed:

No coding task should be considered complete unless the corresponding module checklist is updated.

## Maintenance Notes

- This checklist is module memory for completed work.
- It does not replace the module detailed design document.
- It does not replace tests and test evidence.
- It does not replace proposal/HLD decisions and governance context.
