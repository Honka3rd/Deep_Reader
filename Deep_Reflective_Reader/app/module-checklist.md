# app Checklist

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

- [x] Implements QA orchestration via `QACoordinator` across prepare, retrieval, prompt, and session update paths.
  Evidence: `Deep_Reflective_Reader/app/qa_coordinator.py; Deep_Reflective_Reader/app/module-detailed-design.md (Main Responsibilities)`
  Notes: Coordinator layer exists as application orchestration, not API schema code.

- [x] Implements section/chapter task orchestration via `SectionTaskCoordinator`, including task-layout projection assembly.
  Evidence: `Deep_Reflective_Reader/app/section_task_coordinator.py; Deep_Reflective_Reader/app/module-detailed-design.md (Main Flows)`
  Notes: Task-layout and task execution paths are coordinated in one runtime service.

- [x] Maintains hierarchy-required fail-fast behavior for incompatible runtime structure states.
  Evidence: `Deep_Reflective_Reader/app/section_task_coordinator.py; Deep_Reflective_Reader/app/module-detailed-design.md (Error Semantics)`
  Notes: Legacy masking fallbacks are tightened in runtime coordinator paths.

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
