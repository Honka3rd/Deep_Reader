# Main Agent Response Template

## Task Type

<implementation | audit | documentation | repository-routing | architecture | planning | validation | governance>

---

## Ownership Analysis

### Owning Module

<module>

### Supporting Modules

- <module>
- <module>

### Affected Modules

- <module>
- <module>

### Ownership Reasoning

<why this module owns the task>

---

## Repository Memory Required

### Global Memory

- AGENTS.md
- Deep_Reflective_Reader/proposal.md
- Deep_Reflective_Reader/high-level-design.md
- Deep_Reflective_Reader/docs/modules/index.md

### Module Memory

- <module>/module-detailed-design.md
- <module>/module-checklist.md

### Additional Memory

- <file>
- <file>

---

## Current Stage

<module-resolution | memory-loading | implementation | documentation-sync | validation | review>

---

## Codex CLI Prompt

Use module-memory-loader.

Target Module:
<module>

Task Type:
<task type>

Load repository memory.

Required Reading:

- AGENTS.md
- Deep_Reflective_Reader/proposal.md
- Deep_Reflective_Reader/high-level-design.md
- Deep_Reflective_Reader/docs/modules/index.md
- <module>/module-detailed-design.md
- <module>/module-checklist.md

Report:

- files loaded
- ownership concerns
- architecture concerns
- module boundary concerns
- repository memory conflicts

Stop after reporting.

Do not modify any files.

---

## Risks

### Ownership Risks

- ...

### Architecture Risks

- ...

### Boundary Risks

- ...

### Repository Memory Risks

- ...

---

## Completion Criteria For Current Stage

- Required repository memory loaded
- Ownership validated
- Architecture concerns identified
- Memory conflicts reported

---

## Next Step

After Codex CLI completes the current stage:

Return the result to Main Agent V2.

Main Agent V2 will determine the next stage and generate the next Codex CLI prompt.