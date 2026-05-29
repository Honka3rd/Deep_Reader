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

Codex must not edit markdown files unless the active skill explicitly allows that file type.

Audit tasks are read-only.

Maintainer tasks may write only their owned markdown files.

Every markdown edit must report:

- inspected markdown files
- modified markdown files
- reason for each modification
- evidence
- validation result

## Progress update policy

`progress.md` may only be updated when:

1. the active skill explicitly owns progress synchronization, or
2. the user explicitly requested a progress update.

Never invent completion status.
If evidence is insufficient, keep status as `Needs Confirmation`.