---
name: module-memory-loader
description: Load Deep Reader repository memory before implementation, audit, validation, documentation, or repository-routing work.
---

# Module Memory Loader

## Purpose

This skill loads the required repository memory before execution.

The goal is to:

- make repository memory deterministic
- reduce dependence on chat context
- validate module ownership
- identify architecture concerns
- identify repository memory conflicts

This skill is strictly read-only.

It must never modify:

- source code
- tests
- markdown files
- configuration files

---

# Supported Modes

## Module-Specific Mode

Used when the target module is already known.

Examples:

text document_structure section_tasks retrieval profile evaluated_answer 

---

## Repository-Routing Mode

Used when the target module is not yet known.

Purpose:

text Determine ownership.  Determine affected modules.  Recommend target module.  Recommend required module memory. 

---

# When To Use

Use this skill before:

- implementation
- audit
- validation
- documentation synchronization
- architecture review
- repository routing

---

# Modification Permissions

## Allowed

Read repository files.

---

## Forbidden

Do not modify:

- source code
- tests
- markdown files
- configuration files

Files Modified:

text none 

---

# Required Inputs

The caller must provide:

text Target Module: <module_name | Unknown / repository-routing>  Task Type: <implementation | audit | validation | documentation | repository-routing> 

---

# Global Memory Loading Order

Always load:

1. AGENTS.md
2. Deep_Reflective_Reader/proposal.md
3. Deep_Reflective_Reader/high-level-design.md
4. Deep_Reflective_Reader/docs/modules/index.md
5. Deep_Reflective_Reader/progress.md

---

# Repository-Routing Mode

Activate repository-routing mode if:

text Target Module: Unknown / repository-routing 

or:

text Task Type: repository-routing 

In repository-routing mode:

Load only:

text AGENTS.md  Deep_Reflective_Reader/proposal.md  Deep_Reflective_Reader/high-level-design.md  Deep_Reflective_Reader/docs/modules/index.md  Deep_Reflective_Reader/progress.md 

Do not load:

text module-detailed-design.md  module-checklist.md 

Do not attempt to locate:

text repository-routing/ 

because repository-routing is a task type, not a module.

Repository-routing mode must report:

text Candidate Owning Modules  Supporting Modules  Affected Modules  Ownership Concerns  Architecture Concerns  Recommended Target Module  Required Module Docs For Next Stage 

Stop after reporting.

---

# Module-Specific Mode

Activate module-specific mode if:

text Target Module: <existing module> 

For the target module load:

text <module>/module-detailed-design.md 

Then:

text <module>/module-checklist.md 

If available:

text README.md  design-notes.md  future-work.md 

load them afterwards.

---

# Memory Priority

Repository memory must be interpreted in the following order.

Priority 1

text AGENTS.md 

Priority 2

text proposal.md 

Priority 3

text high-level-design.md 

Priority 4

text docs/modules/index.md 

Priority 5

text module-detailed-design.md 

Priority 6

text module-checklist.md 

Priority 7

text progress.md 

Priority 8

text implementation files 

If conflicts are discovered:

Higher priority memory wins.

---

# Output Requirements

The skill must report:

text Mode  Target Module  Task Type  Files Loaded  Potential Conflicts  Architecture Concerns  Recommended Next Step 

---

# Validation

Before completing:

Confirm:

text required memory files loaded  no files modified 

For module-specific mode:

text target module exists 

For repository-routing mode:

text no module-specific docs loaded 

If any file was modified:

text ERROR:  module-memory-loader is read-only. 

---

# Final Report Format

## Module-Specific Mode

text Skill: module-memory-loader  Mode: module-specific  Target Module: ...  Task Type: ...  Global Memory Loaded: - ...  Module Memory Loaded: - ...  Additional Memory Loaded: - ...  Potential Conflicts: - ...  Architecture Concerns: - ...  Recommended Next Step: - ...  Files Modified: none  Validation Result: success 

---

## Repository-Routing Mode

text Skill: module-memory-loader  Mode: repository-routing  Target Module: Unknown / repository-routing  Task Type: repository-routing  Global Memory Loaded: - ...  Module Memory Loaded: none  Candidate Owning Modules: - ...  Supporting Modules: - ...  Affected Modules: - ...  Ownership Concerns: - ...  Architecture Concerns: - ...  Recommended Target Module: ...  Required Module Docs For Next Stage: - ...  Files Modified: none  Validation Result: success 