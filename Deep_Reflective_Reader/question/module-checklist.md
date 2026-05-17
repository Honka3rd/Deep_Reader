# question Checklist

## Purpose

This checklist records completed, code-confirmed or design-confirmed tasks for the `question` module.

It is used to:
- preserve module-level implementation memory
- reduce hallucination in future Codex tasks
- prevent context-window compression from losing completed work
- track future task completion explicitly

## Source Documents

- `Deep_Reflective_Reader/question/module-detailed-design.md`
- `Deep_Reflective_Reader/proposal.md`
- `Deep_Reflective_Reader/high-level-design.md`
- `Deep_Reflective_Reader/question/`

## Rules

- Only completed work is listed as checked.
- Future work must not be added unless explicitly requested.
- If a new task is added later, it must first be added unchecked.
- Once completed, it must be checked in this file.
- Uncertain items must go to `Needs Confirmation`, not the completed checklist.

## Completed Checklist

- [x] Defines query-related enums and standardized question contract.
  Evidence: `Deep_Reflective_Reader/question/qa_enums.py; Deep_Reflective_Reader/question/standardized/standardized_question.py; Deep_Reflective_Reader/question/module-detailed-design.md (Important Data Structures / Contracts)`
  Notes: Scope and prompt modes are formalized as enums.

- [x] Implements LLM-based question standardization with strict JSON parsing and language normalization.
  Evidence: `Deep_Reflective_Reader/question/standardized/question_standardizer.py; Deep_Reflective_Reader/question/module-detailed-design.md (Main Responsibilities)`
  Notes: Standardizer returns normalized query and user/document language codes.

- [x] Implements scope resolution using lexical, semantic, and optional LLM fallback paths with diagnostics.
  Evidence: `Deep_Reflective_Reader/question/question_scope_resolver.py; Deep_Reflective_Reader/question/module-detailed-design.md (Main Flows)`
  Notes: Resolver includes local-reference signal analysis and method attribution.

## Needs Confirmation

No unresolved confirmation items identified in this pass.

## Future Task Policy

New future tasks for this module must be added here first as unchecked items:

- [ ] Support content-block-level question targeting
- [ ] Define evidence/content-block reference semantics
- [ ] Define sentence/paragraph-level interaction flow
- [ ] Preserve id-based hierarchy context when resolving fine-grained question targets

After implementation, the task owner must update this checklist and mark the task as completed:

- [x] <completed task>

No coding task should be considered complete unless the corresponding module checklist is updated.

## Maintenance Notes

- This checklist is module memory for completed work.
- It does not replace the module detailed design document.
- It does not replace tests and test evidence.
- It does not replace proposal/HLD decisions and governance context.
