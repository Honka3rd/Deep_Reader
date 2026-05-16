# section_tasks Checklist

## Purpose

This checklist records completed, code-confirmed or design-confirmed tasks for the `section_tasks` module.

It is used to:
- preserve module-level implementation memory
- reduce hallucination in future Codex tasks
- prevent context-window compression from losing completed work
- track future task completion explicitly

## Source Documents

- `Deep_Reflective_Reader/section_tasks/module-detailed-design.md`
- `Deep_Reflective_Reader/proposal.md`
- `Deep_Reflective_Reader/high-level-design.md`
- `Deep_Reflective_Reader/section_tasks/`

## Rules

- Only completed work is listed as checked.
- Future work must not be added unless explicitly requested.
- If a new task is added later, it must first be added unchecked.
- Once completed, it must be checked in this file.
- Uncertain items must go to `Needs Confirmation`, not the completed checklist.

## Completed Checklist

- [x] Implements chapters-first task-layout DTO contracts and diagnostics DTO types.
  Evidence: `Deep_Reflective_Reader/section_tasks/document_task_layout.py; Deep_Reflective_Reader/section_tasks/module-detailed-design.md (Important Data Structures / Contracts)`
  Notes: Task-layout contract is explicit at module level.

- [x] Implements hierarchy-first task unit resolution entry using effective hierarchy sections.
  Evidence: `Deep_Reflective_Reader/section_tasks/task_unit_resolver.py; Deep_Reflective_Reader/section_tasks/module-detailed-design.md (Key Files)`
  Notes: Resolver path was aligned with hierarchy-first model.

- [x] Implements section task context lookup with hierarchy-only section resolution behavior.
  Evidence: `Deep_Reflective_Reader/section_tasks/section_task_context_builder.py; Deep_Reflective_Reader/section_tasks/module-detailed-design.md (Known Legacy / Compatibility Behavior)`
  Notes: Legacy section fallback is disabled by default at context-builder path.

- [x] Documentation governance cleanup for hierarchy-first task-layout semantics
  Evidence: `Deep_Reflective_Reader/section_tasks/module-detailed-design.md (Read/Write Boundary Matrix, Public API Boundary Clarification, Terminology Governance Audit)`; `Deep_Reflective_Reader/main.py`; `Deep_Reflective_Reader/api_schemas.py`; `Deep_Reflective_Reader/progress.md`
  Notes: 已明確分離 task-layout projection/read path 與 artifact persistence write path，並固定 public chapters-first contract 與 diagnostics no-write-back 邊界。

## Needs Confirmation

- [ ] 是否要在 section_tasks 文檔中全面淘汰 `artifact mirror` 用語，統一改為 `transitional internal field`。  
  Reason: `module-detailed-design.md` 已將其標註為歷史語境，但全專案術語尚未完全統一。  
  Needed confirmation: maintainer 是否要在下一輪做全局術語替換策略。

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
