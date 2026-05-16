# language Checklist

## Purpose

This checklist records completed, code-confirmed or design-confirmed tasks for the `language` module.

It is used to:
- preserve module-level implementation memory
- reduce hallucination in future Codex tasks
- prevent context-window compression from losing completed work
- track future task completion explicitly

## Source Documents

- `Deep_Reflective_Reader/language/module-detailed-design.md`
- `Deep_Reflective_Reader/proposal.md`
- `Deep_Reflective_Reader/high-level-design.md`
- `Deep_Reflective_Reader/language/`

## Rules

- Only completed work is listed as checked.
- Future work must not be added unless explicitly requested.
- If a new task is added later, it must first be added unchecked.
- Once completed, it must be checked in this file.
- Uncertain items must go to `Needs Confirmation`, not the completed checklist.

## Completed Checklist

- [x] Defines canonical language code enum and resolver/inference utilities.
  Evidence: `Deep_Reflective_Reader/language/language_code.py; Deep_Reflective_Reader/language/module-detailed-design.md (Key Files)`
  Notes: Language normalization supports aliases and script-based inference fallback.

- [x] Implements document language detection with profile/records reuse and LLM fallback.
  Evidence: `Deep_Reflective_Reader/language/document_language_detector.py; Deep_Reflective_Reader/language/module-detailed-design.md (Main Flows)`
  Notes: Detector prefers persisted metadata before model call.

- [x] Implements script/discourse/profile registries for language-scoped heuristics and cues.
  Evidence: `Deep_Reflective_Reader/language/language_script_registry.py; Deep_Reflective_Reader/language/language_discourse_registry.py; Deep_Reflective_Reader/language/language_profile_registry.py; Deep_Reflective_Reader/language/module-detailed-design.md (Main Responsibilities)`
  Notes: Registry boundaries are explicit and consumed by profile/question modules.

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
