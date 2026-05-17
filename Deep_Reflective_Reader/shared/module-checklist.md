# shared Checklist

## Purpose

This checklist records completed, code-confirmed or design-confirmed tasks for the `shared` module.

It is used to:
- preserve module-level implementation memory
- reduce hallucination in future Codex tasks
- prevent context-window compression from losing completed work
- track future task completion explicitly

## Source Documents

- `Deep_Reflective_Reader/shared/module-detailed-design.md`
- `Deep_Reflective_Reader/proposal.md`
- `Deep_Reflective_Reader/high-level-design.md`
- `Deep_Reflective_Reader/shared/`

## Rules

- Only completed work is listed as checked.
- Future work must not be added unless explicitly requested.
- If a new task is added later, it must first be added unchecked.
- Once completed, it must be checked in this file.
- Uncertain items must go to `Needs Confirmation`, not the completed checklist.

## Completed Checklist

- [x] Defines cross-module `TaskUnit` contract including parent identity and artifact payload fields.
  Evidence: `Deep_Reflective_Reader/shared/task_unit_model.py; Deep_Reflective_Reader/shared/module-detailed-design.md (Important Data Structures / Contracts)`
  Notes: Task unit serialization/deserialization contract is centralized.

- [x] Defines summary/quiz artifact schemas and document-level artifact container models.
  Evidence: `Deep_Reflective_Reader/shared/task_artifacts.py; Deep_Reflective_Reader/shared/module-detailed-design.md (Main Responsibilities)`
  Notes: Artifact metadata/version fields are part of shared contract.

- [x] Defines generic abstract result contract for service execution outputs.
  Evidence: `Deep_Reflective_Reader/shared/abstract_result.py; Deep_Reflective_Reader/shared/module-detailed-design.md (Key Files)`
  Notes: Success/failure payload structure is standardized.

## Needs Confirmation

No unresolved confirmation items identified in this pass.

## Future Task Policy

New future tasks for this module must be added here first as unchecked items:

- [ ] Design shared rich task-unit content block model
- [ ] Define content-block identity and artifact attachment semantics
- [ ] Define backward compatibility adapter from string content to single content block

After implementation, the task owner must update this checklist and mark the task as completed:

- [x] <completed task>

No coding task should be considered complete unless the corresponding module checklist is updated.

## Maintenance Notes

- This checklist is module memory for completed work.
- It does not replace the module detailed design document.
- It does not replace tests and test evidence.
- It does not replace proposal/HLD decisions and governance context.
