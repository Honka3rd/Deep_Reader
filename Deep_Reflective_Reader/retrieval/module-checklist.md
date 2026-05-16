# retrieval Checklist

## Purpose

This checklist records completed, code-confirmed or design-confirmed tasks for the `retrieval` module.

It is used to:
- preserve module-level implementation memory
- reduce hallucination in future Codex tasks
- prevent context-window compression from losing completed work
- track future task completion explicitly

## Source Documents

- `Deep_Reflective_Reader/retrieval/module-detailed-design.md`
- `Deep_Reflective_Reader/proposal.md`
- `Deep_Reflective_Reader/high-level-design.md`
- `Deep_Reflective_Reader/retrieval/`

## Rules

- Only completed work is listed as checked.
- Future work must not be added unless explicitly requested.
- If a new task is added later, it must first be added unchecked.
- Once completed, it must be checked in this file.
- Uncertain items must go to `Needs Confirmation`, not the completed checklist.

## Completed Checklist

- [x] Parses raw text into node sequence with positional metadata through `NodeProvider`.
  Evidence: `Deep_Reflective_Reader/retrieval/node_provider.py; Deep_Reflective_Reader/retrieval/module-detailed-design.md (Main Responsibilities)`
  Notes: Node metadata includes chunk index, char offsets, and neighbor links.

- [x] Builds FAISS bundles from parsed nodes with capability-aware token budgets.
  Evidence: `Deep_Reflective_Reader/retrieval/faiss_index_builder.py; Deep_Reflective_Reader/retrieval/module-detailed-design.md (Main Flows)`
  Notes: Builder returns `FaissIndexBundle` with runtime budget fields.

- [x] Persists and reloads FAISS artifacts with record-schema checks and rebuild guards.
  Evidence: `Deep_Reflective_Reader/retrieval/faiss_index_store.py; Deep_Reflective_Reader/retrieval/module-detailed-design.md (Known Legacy / Compatibility Behavior)`
  Notes: Store can detect incomplete legacy record schema via position metadata checks.

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
