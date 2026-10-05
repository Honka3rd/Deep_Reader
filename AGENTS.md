# Deep Reader Agent Instructions

## Repository role

This repository is the persistent engineering memory for Deep Reader.

Agents must treat markdown files as controlled project memory, not casual notes.

## Required global reading

Before any task that changes code or documentation, read:

- `Deep_Reflective_Reader/proposal.md`
- `Deep_Reflective_Reader/high-level-design.md`
- `Deep_Reflective_Reader/docs/modules/index.md` if present
- the target module `module-detailed-design.md`
- the target module `module-checklist.md`

For `Deep_Reader_UI` tasks, the target module documents are:

- `Deep_Reader_UI/module-detailed-design.md`
- `Deep_Reader_UI/module-checklist.md`

## Global architecture rules

Do not reintroduce:

- root `sections[]` as the primary document source
- `structure_nodes` as the main flow
- hidden task-layout persistence mutation
- diagnostics profile write-back
- metadata or LLM classification as parser authority
- non-hierarchy-aware artifact writes

## Markdown governance policy

Markdown files are persistent project memory.

Codex must not edit markdown files unless the active skill explicitly allows that file type or the user has explicitly requested a maintainer documentation/governance update.

Audit tasks are read-only.

Maintainer tasks may write only their owned markdown files.

`Deep_Reader_UI` is a governed module. Its owned markdown files are:

- `Deep_Reader_UI/module-detailed-design.md`
- `Deep_Reader_UI/module-checklist.md`

UI module documentation edits must preserve the frontend boundary:

- backend hierarchy remains the source of truth
- `/documents/task-layout` remains a lightweight projection
- UI state must not become backend truth
- UI changes must not introduce hidden backend mutation
- UI documentation must not claim backend API or schema behavior unless implemented in the backend

Every markdown edit must report:

- inspected markdown files
- modified markdown files
- reason for each modification
- evidence
- validation result

Checklist governance:

- Any newly added or status-updated checklist item must include a final metadata line formatted exactly as `Timestamp: YYYY-MM-DD`.
- The timestamp must be the date of the checklist change, not an inferred feature completion date.
- If a checklist item is moved from unchecked to checked, update or add its final `Timestamp: YYYY-MM-DD` line during the same edit.
- Do not add timestamps to unrelated existing checklist items unless they are being changed in the same task.

## Progress update policy

`progress.md` may only be updated when:

1. the active skill explicitly owns progress synchronization, or
2. the user explicitly requested a progress update.

`Deep_Reflective_Reader/progress.md` aggregates `Deep_Reflective_Reader` module progress only. Do not update it for `Deep_Reader_UI` work unless a UI progress aggregation document is explicitly introduced.

Never invent completion status.
If evidence is insufficient, keep status as `Needs Confirmation`.
