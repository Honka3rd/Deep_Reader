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

## Needs Confirmation

No unresolved confirmation items identified in this pass.

## Future Task Policy

New future tasks for this module must be added here first as unchecked items:

- [ ] Design rich task-unit content response schema (future direction, not implemented)
- [ ] Define content-block artifact metadata schema (future direction, not implemented)
- [ ] Define backward-compatible content response evolution strategy (future direction, not implemented)

After implementation, the task owner must update this checklist and mark the task as completed:

No coding task should be considered complete unless the corresponding module checklist is updated.

## Maintenance Notes

- This checklist is module memory for completed work.
- It does not replace the module detailed design document.
- It does not replace tests and test evidence.
- It does not replace proposal/HLD decisions and governance context.
