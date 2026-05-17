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

- [x] <completed task>

No coding task should be considered complete unless the corresponding module checklist is updated.

## Maintenance Notes

- This checklist is module memory for completed work.
- It does not replace the module detailed design document.
- It does not replace tests and test evidence.
- It does not replace proposal/HLD decisions and governance context.
