# prompts Checklist

## Purpose

This checklist records completed, code-confirmed or design-confirmed tasks for the `prompts` module.

It is used to:
- preserve module-level implementation memory
- reduce hallucination in future Codex tasks
- prevent context-window compression from losing completed work
- track future task completion explicitly

## Source Documents

- `Deep_Reflective_Reader/prompts/module-detailed-design.md`
- `Deep_Reflective_Reader/proposal.md`
- `Deep_Reflective_Reader/high-level-design.md`
- `Deep_Reflective_Reader/prompts/`

## Rules

- Only completed work is listed as checked.
- Future work must not be added unless explicitly requested.
- If a new task is added later, it must first be added unchecked.
- Once completed, it must be checked in this file.
- Uncertain items must go to `Needs Confirmation`, not the completed checklist.

## Completed Checklist

- [x] Implements profile rendering block for answer prompts.
  Evidence: `Deep_Reflective_Reader/prompts/prompt_assembler.py; Deep_Reflective_Reader/prompts/module-detailed-design.md (Main Responsibilities)`
  Notes: Prompt includes topic, language code, and summary.

- [x] Implements answer-rule rendering by `AnswerMode` strictness levels.
  Evidence: `Deep_Reflective_Reader/prompts/prompt_assembler.py; Deep_Reflective_Reader/prompts/module-detailed-design.md (Main Responsibilities)`
  Notes: Strict/cautious/reject instruction sets are explicit.

- [x] Implements mode-specific guidance for local reading, retrieval, and full-text prompts.
  Evidence: `Deep_Reflective_Reader/prompts/prompt_assembler.py; Deep_Reflective_Reader/prompts/module-detailed-design.md (Main Flows)`
  Notes: Guidance text is selected through `PromptMode`.

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
