---
name: module-memory-loader
description: Load Deep Reader repository memory for a module-specific task before implementation, audit, validation, or documentation work.
---

# Module Memory Loader

## Purpose

This skill loads the minimum required repository memory for a module-specific task.

The goal is to make repository memory deterministic and reduce dependence on chat context.

This skill is read-only.

It must not modify source code or markdown files.

---

# When To Use

Use this skill when:

- working on a specific module
- auditing a module
- implementing module changes
- validating module changes
- synchronizing module documentation
- generating implementation plans

Examples:

- document_structure task
- section_tasks task
- profile task
- retrieval task
- evaluated_answer task

---

# When Not To Use

Do not use this skill when:

- performing repository-wide architecture review
- modifying global architecture documents
- working on cross-module integration tasks
- performing frontend/mobile integration planning

Use future architecture-level memory loaders instead.

---

# Modification Permissions

## Allowed

Read repository files.

## Forbidden

Do not modify:

- source code
- tests
- markdown files
- configuration files

This skill is strictly read-only.

---

# Required Inputs

The caller must provide:

```text
Target Module:
<module_name>

Task Type:
<implementation | audit | validation | documentation>
```

Example:

```text
Target Module:
document_structure

Task Type:
audit
```

---

# Global Memory Loading Order

Always load:

1. AGENTS.md
2. Deep_Reflective_Reader/proposal.md
3. Deep_Reflective_Reader/high-level-design.md
4. Deep_Reflective_Reader/progress.md
5. Deep_Reflective_Reader/docs/modules/index.md

---

# Module Memory Loading Order

For the target module:

Load:

```text
<module>/module-detailed-design.md
```

Then:

```text
<module>/module-checklist.md
```

If additional module documentation exists:

```text
README.md
design-notes.md
future-work.md
```

load them after the required files.

---

# Memory Priority

Repository memory must be interpreted in the following order:

Priority 1

AGENTS.md

Priority 2

proposal.md

Priority 3

high-level-design.md

Priority 4

docs/modules/index.md

Priority 5

module-detailed-design.md

Priority 6

module-checklist.md

Priority 7

progress.md

Priority 8

implementation files

If conflicts are discovered:

Higher priority memory wins.

---

# Output Requirements

The skill must report:

```text
Target Module:

Task Type:

Files Loaded:

Global Memory:
- ...

Module Memory:
- ...

Additional Memory:
- ...

Potential Conflicts:
- ...

Recommended Next Skill:
- ...
```

---

# Recommended Next Skill

Implementation task:

```text
implementation-validation
```

Audit task:

```text
module-boundary-audit
```

Documentation task:

```text
documentation-sync
```

Checklist synchronization:

```text
checklist-progress-sync
```

---

# Validation

Before completing:

Confirm:

- target module exists
- required memory files were loaded
- no files were modified

Report:

```text
Files Modified:
none
```

If any file was modified:

Return:

```text
ERROR:
module-memory-loader is read-only.
```

---

# Final Report Format

```text
Skill:
module-memory-loader

Target Module:
...

Task Type:
...

Global Memory Loaded:
- ...

Module Memory Loaded:
- ...

Additional Memory Loaded:
- ...

Potential Conflicts:
- ...

Recommended Next Skill:
- ...

Files Modified:
none

Validation Result:
success
```