# context Checklist

## Purpose

This checklist records completed, code-confirmed or design-confirmed tasks for the `context` module.

It is used to:
- preserve module-level implementation memory
- reduce hallucination in future Codex tasks
- prevent context-window compression from losing completed work
- track future task completion explicitly

## Source Documents

- `Deep_Reflective_Reader/context/module-detailed-design.md`
- `Deep_Reflective_Reader/proposal.md`
- `Deep_Reflective_Reader/high-level-design.md`
- `Deep_Reflective_Reader/context/`

## Rules

- Only completed work is listed as checked.
- Future work must not be added unless explicitly requested.
- If a new task is added later, it must first be added unchecked.
- Once completed, it must be checked in this file.
- Uncertain items must go to `Needs Confirmation`, not the completed checklist.

## Completed Checklist

- [x] Implements context-mode orchestration for local window, retrieval, and full-text paths.
  Evidence: `Deep_Reflective_Reader/context/context_orchestrator.py; Deep_Reflective_Reader/context/module-detailed-design.md (Main Responsibilities)`
  Notes: Context routing is explicit and mode-aware.

- [x] Builds ordered context chunks with budget controls via `DocumentContextBuilder`.
  Evidence: `Deep_Reflective_Reader/context/document_context_builder.py; Deep_Reflective_Reader/context/module-detailed-design.md (Main Flows)`
  Notes: Includes neighbor expansion and truncation handling.

- [x] Provides prompt-aware token budgeting and truncation utilities through `TokenBudgetManager`.
  Evidence: `Deep_Reflective_Reader/context/token_budget_manager.py; Deep_Reflective_Reader/context/module-detailed-design.md (Key Files)`
  Notes: Budget computation uses non-context prompt estimation plus reserves.

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
