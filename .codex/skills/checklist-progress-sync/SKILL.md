---
name: checklist-progress-sync
description: Use when synchronizing Deep_Reflective_Reader module checklist status with progress.md. This skill is for controlled markdown governance only.
---

# Checklist Progress Sync

## Purpose

Synchronize one Deep_Reflective_Reader module's checklist state with `Deep_Reflective_Reader/progress.md`.

This skill turns markdown documentation into controlled repository memory.

## Allowed files

This skill may inspect:

- `AGENTS.md`
- `Deep_Reflective_Reader/progress.md`
- `Deep_Reflective_Reader/proposal.md`
- `Deep_Reflective_Reader/high-level-design.md`
- `Deep_Reflective_Reader/docs/modules/index.md`
- target module `module-detailed-design.md`
- target module `module-checklist.md`

This skill may modify only:

- `Deep_Reflective_Reader/progress.md`
- target module `module-checklist.md`

## Forbidden edits

Do not modify:

- Python source files
- tests
- API schemas
- `proposal.md`
- `high-level-design.md`
- unrelated module documentation
- frontend or mobile files

## Required reading order

1. `AGENTS.md`
2. `Deep_Reflective_Reader/progress.md`
3. `Deep_Reflective_Reader/proposal.md`
4. `Deep_Reflective_Reader/high-level-design.md`
5. target module `module-detailed-design.md`
6. target module `module-checklist.md`

## Sync rules

- Count unresolved checklist items directly from the target `module-checklist.md`.
- Do not invent completed work.
- Only mark checklist items complete when the repository evidence proves completion.
- If evidence is incomplete, keep the item unresolved.
- If module checklist and `progress.md` disagree, update only the minimal progress fields needed.
- If they are already synchronized, make no changes.

## Validation`

Before final response:

- Run `git diff --stat`.
- Run `git diff -- Deep_Reflective_Reader/progress.md`.
- Run `git diff -- <target-module>/module-checklist.md`.
- Confirm no source code files were modified.

## Required final report

Report:

1. target module
2. markdown files inspected
3. markdown files modified
4. source files modified: yes/no
5. before status
6. after status
7. unresolved checklist count
8. validation result
9. remaining concerns, if any