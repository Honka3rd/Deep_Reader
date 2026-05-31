---
name: documentation-sync
description: Synchronize Deep Reader markdown project memory after implementation, audit, or governance tasks. Use only for controlled documentation updates.
---

# Documentation Sync

## Purpose

This skill synchronizes Deep Reader markdown project memory after completed work.

The goal is to keep repository memory consistent across:

- `Deep_Reflective_Reader/progress.md`
- target module `module-checklist.md`
- target module `module-detailed-design.md`
- `Deep_Reflective_Reader/docs/modules/index.md`

This skill has controlled markdown write authority.

It must not modify source code or tests.

---

# When To Use

Use this skill after:

- implementation tasks
- module governance cleanup
- module boundary audit
- checklist completion
- architecture documentation cleanup
- Codex-generated code changes
- manual code changes that affect module state

Use it when repository memory may need synchronization.

---

# When Not To Use

Do not use this skill:

- before implementation
- as a replacement for code review
- as a replacement for implementation-validation
- for casual documentation rewriting
- for README/user-facing documentation
- for unrelated markdown cleanup

Use `implementation-validation` before this skill if repository changes have not yet been validated.

---

# Permissions

## Allowed To Inspect

- `AGENTS.md`
- `.codex/skill-design.md`
- `Deep_Reflective_Reader/proposal.md`
- `Deep_Reflective_Reader/high-level-design.md`
- `Deep_Reflective_Reader/progress.md`
- `Deep_Reflective_Reader/docs/modules/index.md`
- target module `module-detailed-design.md`
- target module `module-checklist.md`
- changed source files
- changed test files
- git diff
- git status

## Allowed To Modify

Only when evidence supports the change:

- `Deep_Reflective_Reader/progress.md`
- target module `module-checklist.md`
- target module `module-detailed-design.md`
- `Deep_Reflective_Reader/docs/modules/index.md`

## Forbidden To Modify

- Python source files
- test files
- API schema files
- configuration files
- unrelated module documentation
- `proposal.md`
- `high-level-design.md`
- README files
- frontend or mobile files

---

# Required Inputs

The caller should provide:

```text
Target Module:
<module_name>

Task Type:
<implementation | audit | governance | documentation>

Completed Work Summary:
<brief summary of the completed work>
```

Example:

```text
Target Module:
section_tasks

Task Type:
implementation

Completed Work Summary:
Segmented task-unit endpoint behavior was stabilized and regression coverage was added.
```

---

# Required Reading Order

Load:

1. `AGENTS.md`
2. `.codex/skill-design.md`
3. `Deep_Reflective_Reader/proposal.md`
4. `Deep_Reflective_Reader/high-level-design.md`
5. `Deep_Reflective_Reader/docs/modules/index.md`
6. `Deep_Reflective_Reader/progress.md`

Then load target module memory:

7. target module `module-detailed-design.md`
8. target module `module-checklist.md`

Then inspect implementation evidence:

9. `git status`
10. `git diff --stat`
11. relevant changed files

---

# Synchronization Rules

## Rule 1: Evidence First

Do not update documentation based only on intent.

A documentation change requires evidence from at least one of:

- changed source code
- changed tests
- completed checklist item
- explicit user instruction
- validated implementation report
- repository diff

If evidence is insufficient, report:

```text
Documentation Sync Blocked:
insufficient evidence
```

---

## Rule 2: Minimal Updates

Make the smallest documentation update needed.

Do not rewrite large sections unless explicitly requested.

Prefer:

- checklist status updates
- short progress notes
- concise module design corrections
- module registry alignment

Avoid:

- broad prose rewriting
- speculative future architecture
- unrelated cleanup
- style-only edits

---

## Rule 3: Checklist Authority

Use target module `module-checklist.md` as the primary source for task-level completion state.

Only mark an item complete when evidence proves completion.

If work is partial, keep the item unresolved and add a concise note if appropriate.

---

## Rule 4: Progress Authority

Use `Deep_Reflective_Reader/progress.md` as the cross-module runtime state summary.

Update it only when:

- checklist status changed
- module status changed
- unresolved count changed
- completed task materially affects project progress

Do not invent module progress.

---

## Rule 5: Module Design Authority

Use target module `module-detailed-design.md` for module boundary, responsibilities, and design notes.

Update it only when:

- implementation changed the module contract
- governance cleanup clarified module responsibility
- previous module documentation is outdated or misleading

Do not place checklist state inside detailed design.

---

## Rule 6: Module Registry Authority

Use `Deep_Reflective_Reader/docs/modules/index.md` as the module ownership and registry source.

Update it only when:

- a module is added
- a module is removed
- a module responsibility changes
- module relationship or ownership wording is outdated

Do not duplicate module-level checklist details in the registry.

---

## Rule 7: Architecture Protection

Do not modify:

- `proposal.md`
- `high-level-design.md`

If these appear outdated, report:

```text
Architecture Memory Update Recommended
```

but do not edit them.

---

# Workflow

## Step 1: Load Memory

Load required global and module memory.

Confirm target module exists.

---

## Step 2: Inspect Evidence

Inspect:

```text
git status
git diff --stat
```

Review relevant changed files.

Identify whether implementation, tests, or documentation prove a state change.

---

## Step 3: Determine Required Sync

Classify documentation need:

```text
No sync needed
Checklist sync needed
Progress sync needed
Module design sync needed
Module registry sync needed
Blocked due to insufficient evidence
```

---

## Step 4: Apply Controlled Markdown Updates

Modify only allowed markdown files.

Each change must map to evidence.

---

## Step 5: Validate Diff

Run:

```text
git diff --stat
git diff -- Deep_Reflective_Reader/progress.md
git diff -- <target-module>/module-checklist.md
git diff -- <target-module>/module-detailed-design.md
git diff -- Deep_Reflective_Reader/docs/modules/index.md
```

Confirm no source code or tests were modified by this skill.

---

# Required Final Report

```text
Skill:
documentation-sync

Target Module:
...

Task Type:
...

Completed Work Summary:
...

Documentation Need:
No sync needed / Checklist sync needed / Progress sync needed / Module design sync needed / Module registry sync needed / Blocked

Files Inspected:
- ...

Files Modified:
- ...

Evidence Used:
- ...

Changes Made:
- ...

Source Files Modified By This Skill:
yes/no

Tests Modified By This Skill:
yes/no

Validation Performed:
- ...

Recommended Follow-up Skill:
implementation-validation / checklist-progress-sync / none

Overall Result:
success / partial / blocked / no changes needed

Remaining Concerns:
- ...
```

---

# Write Protection

If this skill needs to modify files outside its allowed scope, stop and report:

```text
ERROR:
documentation-sync does not own this file type.
No modification performed.
```

If evidence is insufficient, stop and report:

```text
ERROR:
documentation-sync requires repository evidence before modifying project memory.
No modification performed.
```