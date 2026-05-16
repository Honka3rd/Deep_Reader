# scripts Checklist

## Purpose

This checklist records completed, code-confirmed or design-confirmed tasks for the `scripts` module.

It is used to:
- preserve module-level implementation memory
- reduce hallucination in future Codex tasks
- prevent context-window compression from losing completed work
- track future task completion explicitly

## Source Documents

- `Deep_Reflective_Reader/scripts/module-detailed-design.md`
- `Deep_Reflective_Reader/proposal.md`
- `Deep_Reflective_Reader/high-level-design.md`
- `Deep_Reflective_Reader/scripts/`

## Rules

- Only completed work is listed as checked.
- Future work must not be added unless explicitly requested.
- If a new task is added later, it must first be added unchecked.
- Once completed, it must be checked in this file.
- Uncertain items must go to `Needs Confirmation`, not the completed checklist.

## Completed Checklist

- [x] Maintains regression script suite for hierarchy, task-layout, artifact persistence, and profile metadata.
  Evidence: `Deep_Reflective_Reader/scripts/test_task_layout_hierarchy_first_read.py; Deep_Reflective_Reader/scripts/test_post_structure_metadata_enrichment.py; Deep_Reflective_Reader/scripts/module-detailed-design.md (Key Files)`
  Notes: Scripts folder acts as primary test entry surface in current repo state.

- [x] Includes real-document and REST smoke script coverage for end-to-end verification paths.
  Evidence: `Deep_Reflective_Reader/scripts/test_rest_structured_parser_modes.py; Deep_Reflective_Reader/scripts/test_rest_dynamic_context.sh; Deep_Reflective_Reader/scripts/module-detailed-design.md (Main Responsibilities)`
  Notes: Smoke scripts validate runtime behavior beyond unit-style checks.

- [x] Covers profile/metadata and language registry hardening through dedicated regression scripts.
  Evidence: `Deep_Reflective_Reader/scripts/test_document_profile_parser_metadata.py; Deep_Reflective_Reader/scripts/test_language_script_registry.py; Deep_Reflective_Reader/scripts/test_language_discourse_registry.py; Deep_Reflective_Reader/scripts/module-detailed-design.md (Main Responsibilities)`
  Notes: Metadata and registry semantics are now tracked by dedicated tests.

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
