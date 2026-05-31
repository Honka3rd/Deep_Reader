# Deep Reader Workflow V2

## Purpose

This document describes the maintainer workflow for Deep Reader.

It is a human-facing operational guide.

It is not repository memory.

It is not architecture authority.

It must not be referenced by Codex skills when making implementation decisions.

Repository authority remains:

1. AGENTS.md
2. proposal.md
3. high-level-design.md
4. docs/modules/index.md
5. module-detailed-design.md
6. module-checklist.md
7. progress.md

---

# Workflow Overview

The workflow is:

User
→ Main Agent V2
→ Codex CLI
→ Skills
→ Main Agent V2 Review
→ Next Step

Main Agent V2 performs routing and orchestration.

Codex CLI performs execution.

Skills perform governance.

Repository markdown files remain the single source of truth.

---

# Step 1

## Submit Natural Language Requirement

The maintainer describes the requirement using natural language.

Examples:

text Introduce JSON DB persistence.  Review the latest commit.  Design content block persistence.  Add a retrieval capability.  Audit the document_structure module. 

Send the requirement to:

text Deep Reflective Reader Main Agent V2 

---

# Step 2

## Main Agent V2 Analysis

Main Agent V2 determines:

text Task Type  Owning Module  Supporting Modules  Affected Modules  Required Repository Memory 

Main Agent V2 generates:

text Codex CLI Prompt 

The prompt always represents exactly one stage.

Main Agent V2 does not implement code.

---

# Step 3

## Execute Prompt In Codex CLI

Copy:

text Codex CLI Prompt 

from Main Agent V2.

Paste into:

bash codex 

Execute the prompt.

---

# Step 4

## Module Memory Loading

The first stage is usually:

text module-memory-loader 

Purpose:

text Load repository memory.  Validate ownership.  Identify architecture concerns.  Identify memory conflicts. 

Expected output:

text Loaded files  Ownership concerns  Architecture concerns  Memory conflicts 

No repository modifications are allowed.

---

# Step 5

## Return Result To Main Agent V2

Copy the Codex CLI output.

Send it back to:

text Deep Reflective Reader Main Agent V2 

Main Agent V2 determines:

text Whether ownership is correct.  Whether architecture concerns exist.  Whether implementation should continue. 

Main Agent V2 generates the next prompt.

---

# Step 6

## Implementation Stage

Codex CLI performs implementation.

Expected modifications:

text Source code  Tests  Module-specific files 

Forbidden modifications:

text proposal.md  high-level-design.md  Unrelated modules 

Implementation completes.

---

# Step 7

## Documentation Synchronization

Run:

text documentation-sync 

Purpose:

text Update repository memory. 

Possible updates:

text progress.md  module-checklist.md  module-detailed-design.md  docs/modules/index.md 

Changes must be supported by repository evidence.

---

# Step 8

## Validation

Run:

text implementation-validation 

Validation checks:

text Module boundaries  Architecture compliance  Repository memory compliance  Documentation synchronization  Git diff 

Expected result:

text PASS  WARNING  FAIL 

---

# Step 9

## Final Review

Return validation results to:

text Deep Reflective Reader Main Agent V2 

Main Agent V2 determines:

text Accept  Revise  Split Task  Rollback  Additional Audit 

Main Agent V2 recommends the next action.

---

# Operational Rule

Never skip:

text module-memory-loader 

before implementation.

Never skip:

text implementation-validation 

before accepting work.

Repository memory always overrides historical chat context.

The repository remains the single source of truth.