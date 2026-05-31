---
name: implementation-validation
description: Validate repository changes after implementation, governance, or documentation tasks. Ensure compliance with AGENTS.md, architecture boundaries, checklist synchronization, and repository memory rules.
---

# Implementation Validation

## Purpose

This skill validates repository changes after a task has been completed.

The goal is to prevent:

- architecture drift
- module boundary violations
- unauthorized file modifications
- forgotten documentation updates
- checklist/progress inconsistencies
- forbidden regressions

This skill is primarily read-only.

By default it must not modify files.

If documentation synchronization is required, it must recommend a documentation skill instead of performing the modification itself.

---

# When To Use

Use this skill:

- after implementation tasks
- after documentation updates
- after governance tasks
- before creating a commit
- before creating a pull request
- before merging a branch

Examples:

- Python module implementation
- React feature implementation
- React Native feature implementation
- architecture cleanup
- checklist maintenance

---

# When Not To Use

Do not use this skill:

- as a replacement for implementation
- as a replacement for documentation synchronization
- as a memory loading skill

Use:

- module-memory-loader
- documentation-sync
- checklist-progress-sync

for those purposes.

---

# Permissions

## Allowed

Read:

- source files
- tests
- markdown files
- configuration files
- git status
- git diff

## Forbidden

Do not:

- modify source code
- modify markdown files
- create commits
- create branches
- merge branches

Validation only.

---

# Required Inputs

The caller should provide:

```text
Task Type:
<implementation | audit | documentation>

Target Module:
<module_name>
```

Example:

```text
Task Type:
implementation

Target Module:
document_structure
```

---

# Required Reading Order

Load:

1. AGENTS.md
2. Deep_Reflective_Reader/proposal.md
3. Deep_Reflective_Reader/high-level-design.md
4. Deep_Reflective_Reader/docs/modules/index.md
5. Deep_Reflective_Reader/progress.md

Then:

6. target module module-detailed-design.md
7. target module module-checklist.md

Finally:

8. git diff
9. git status

---

# Validation Workflow

Perform validation in the following order.

## Step 1

Repository Change Discovery

Collect:

```text
git status
git diff --stat
```

Report:

- changed files
- added files
- removed files

---

## Step 2

Scope Validation

Verify:

- only expected files changed
- no unrelated module modifications
- no accidental repository-wide edits

Flag:

```text
Scope Violation
```

if unrelated files were modified.

---

## Step 3

Architecture Validation

Verify:

Changes do not violate:

- AGENTS.md
- proposal.md
- high-level-design.md

Check for forbidden regressions:

- structure_nodes as primary flow
- root sections mirror
- diagnostics profile write-back
- parser authority leakage
- hidden persistence mutation

Flag:

```text
Architecture Violation
```

if detected.

---

## Step 4

Module Registry Validation

Verify:

The implementation remains consistent with:

```text
Deep_Reflective_Reader/docs/modules/index.md
```

Check:

- module ownership
- declared responsibilities
- module relationships
- allowed dependencies

Flag:

```text
Module Registry Violation
```

if implementation contradicts module registry definitions.

## Step 5

Documentation Validation

Verify:

Whether:

```text
module-checklist.md
progress.md
```

may require synchronization.

Do not modify them.

Instead report:

```text
Documentation Sync Required
```

if necessary.

---

## Step 6

Testing Validation

If tests exist:

Verify:

```text
relevant tests executed
```

or

```text
tests not executed
```

Do not claim tests passed unless evidence exists.

---

## Step 7

Memory Validation

Verify:

Changes remain consistent with:

```text
proposal.md
high-level-design.md
module-detailed-design.md
```

Flag:

```text
Memory Conflict
```

if implementation contradicts repository memory.

---

# Validation Severity

## PASS

No issues found.

---

## WARNING

Task is acceptable but:

- documentation sync recommended
- tests not executed
- minor concerns exist

---

## FAIL

One or more:

- architecture violations
- boundary violations
- memory conflicts
- scope violations

detected.

---

# Recommended Follow-up Skills

Documentation issue:

```text
documentation-sync
```

Checklist issue:

```text
checklist-progress-sync
```

Architecture issue:

```text
module-boundary-audit
```

Memory issue:

```text
module-memory-loader
```

---

# Required Final Report

```text
Skill:
implementation-validation

Task Type:
...

Target Module:
...

Files Changed:
- ...

Files Added:
- ...

Files Removed:
- ...

Scope Validation:
PASS / WARNING / FAIL

Architecture Validation:
PASS / WARNING / FAIL

Boundary Validation:
PASS / WARNING / FAIL

Documentation Validation:
PASS / WARNING / FAIL

Testing Validation:
PASS / WARNING / FAIL

Memory Validation:
PASS / WARNING / FAIL

Recommended Follow-up Skill:
...

Overall Result:
PASS / WARNING / FAIL

Files Modified By Validation:
none
```

---

# Validation Rule

This skill must never claim success without evidence.

Unknown information must be reported as:

```text
Unknown
```

not:

```text
Assumed
```

---

# Write Protection

If this skill attempts to modify files:

Return:

```text
ERROR:
implementation-validation is validation-only.
No repository modifications are permitted.
```