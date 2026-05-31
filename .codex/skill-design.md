# Deep Reader Codex Skill Design

## Purpose

This document is the registry and design authority for repo-local Codex skills in Deep Reader.

Codex skills are not casual prompts. They are controlled workflow units used to load repository memory, execute repeatable tasks, validate changes, and synchronize documentation state.

This file records:

- which skills exist
- why each skill exists
- when each skill should be used
- which files each skill may inspect
- which files each skill may modify
- what output each skill must report

## Design Principles

### 1. Repository memory comes before chat context

The repository is the persistent engineering memory.

Important project knowledge must live in:

- `AGENTS.md`
- `Deep_Reflective_Reader/proposal.md`
- `Deep_Reflective_Reader/high-level-design.md`
- `Deep_Reflective_Reader/progress.md`
- `Deep_Reflective_Reader/docs/modules/index.md`
- module-level `module-detailed-design.md`
- module-level `module-checklist.md`
- `.codex/skill-design.md`
- `.codex/skills/*/SKILL.md`

Chat context may help plan work, but it must not be the only source of workflow rules.

### 2. Skills must be narrow

Each skill should do one type of work.

A skill should not combine unrelated responsibilities such as:

- memory loading
- implementation
- validation
- documentation synchronization
- architecture review

If a workflow needs multiple phases, compose multiple skills instead of creating one large skill.

### 3. Skills must have explicit file permissions

Every skill must clearly define:

- files it may inspect
- files it may modify
- files it must not modify

If a file is not explicitly allowed, the skill should treat it as read-only.

### 4. Audit skills are read-only

Audit skills may inspect and report findings.

They must not modify files unless the user explicitly asks for a maintainer/write task.

### 5. Maintainer skills have controlled write authority

Maintainer skills may modify documentation or source files only within their declared scope.

Every modification must be justified by repository evidence.

### 6. Validation is mandatory after execution

Any skill that causes file changes must report:

- inspected files
- modified files
- reason for each modification
- validation commands or checks performed
- remaining risks or unresolved items

## Skill Categories

### Memory Loader Skills

Memory loader skills load the correct repository memory for a task.

They should usually be read-only.

Examples:

- `module-memory-loader`
- future `architecture-memory-loader`
- future `runtime-state-loader`

### Governance Skills

Governance skills check architecture boundaries, module ownership, and forbidden regressions.

They should usually be read-only.

Examples:

- future `module-boundary-audit`
- future `hierarchy-contract-audit`

### Implementation Skills

Implementation skills perform controlled code changes.

They must follow `AGENTS.md`, module docs, and checklist constraints.

Examples:

- future `python-module-task`
- future `react-web-task`
- future `react-native-mobile-task`

### Validation Skills

Validation skills inspect diffs and verify whether a task respected repository rules.

Examples:

- `implementation-validation`

### Documentation Synchronization Skills

Documentation synchronization skills update markdown project memory after completed work.

Examples:

- `checklist-progress-sync`
- future `documentation-sync`

## Current Skill Registry

| Skill | Status | Category | Write Authority | Purpose |
|---|---|---|---|---|
| `checklist-progress-sync` | Active | Documentation Synchronization | Controlled markdown write | Synchronize module checklist state with `progress.md` |
| `module-memory-loader` | Planned | Memory Loader | Read-only | Load required global and module-level memory before a module task |
| `implementation-validation` | Planned | Validation | Read-only by default | Validate changed files, module boundaries, checklist/progress sync, and forbidden regressions |
| `documentation-sync` | Planned | Documentation Synchronization | Controlled markdown write | Synchronize markdown documentation after implementation or governance tasks |

## Standard Skill Requirements

Every `SKILL.md` should include:

1. `name`
2. `description`
3. purpose
4. when to use
5. when not to use
6. allowed files to inspect
7. allowed files to modify
8. forbidden edits
9. required reading order
10. workflow steps
11. validation rules
12. required final report format

## Standard Final Report Format

Skills should report:

```text
Skill:
Task type:
Target module or area:

Files inspected:
- ...

Files modified:
- ...

Source files modified:
- yes/no

Markdown files modified:
- yes/no

Validation performed:
- ...

Result:
- success / partial / blocked / no changes needed

Remaining concerns:
- ...
Markdown Governance Rule

Markdown files are persistent project memory.

They must not be edited casually.

A markdown file may be modified only when:

1. the active skill explicitly owns that markdown file type, or
2. the user explicitly requested that exact markdown update, or
3. the update is necessary to keep checklist/progress/design state synchronized.

Skill Evolution Rule

Do not create a new skill until one of these is true:

1. the same workflow has been repeated at least twice
2. the workflow has high risk if performed manually
3. the workflow protects important repository memory
4. the workflow will be reused across backend, web, or mobile development

Current Milestone

Milestone 1 focuses on stabilizing workflow memory before React Web and React Native development.

Required skills for Milestone 1:

1. module-memory-loader
2. implementation-validation
3. documentation-sync

Do not expand into frontend/mobile skills until the Python backend workflow is stable.