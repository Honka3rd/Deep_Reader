# Deep Reader Agent Instructions

## Repository role

This repository is the persistent engineering memory for Deep Reader.

Agents must treat markdown files as controlled project memory, not casual notes.

## Delivery workflow

For implementation delivery, follow [.codex/workflow.md](.codex/workflow.md).
It defines the prompt-driven prototype: scope and clarification, documentation
before implementation, tested units with developer confirmation before the next
unit of a large task, local review commit, external quality-review/fix loop, then
Docker rebuild, deployment acceptance and verified remote task-branch push.

External review must explicitly approve the latest candidate SHA before Docker
work. Self-validation is not external approval. Subsequent changes return to the
review gate, with affected deployment checks repeated before delivery. Do not
create a PR or merge as part of this prototype.

Use [.codex/templates/main-agent-instructions.md](.codex/templates/main-agent-instructions.md)
as the reusable entry prompt. Explicitly limited planning, audit, documentation or
instruction-maintenance requests end at their requested, validated deliverable;
they do not automatically authorize commit, deployment or push. The workflow does
not override architecture contracts, Markdown ownership or read-only skill limits.

Use `scripts/commit_review.sh --message-file <file>` for authorized local review
commits after explicit staging. After exact-SHA approval, use
`scripts/start_local_dev.sh --backend-only` for Docker rebuilds; do not recreate
its startup sequence in temporary scripts. See the workflow for isolation, data
preservation and acceptance requirements. Neither script bypasses review gates.

## Scope-first repository reading

Read applicable `AGENTS.md` instructions first. Then determine which modules the
request is likely to change before loading their design documents and checklists.
Do not read every module's documentation to discover the task's scope.

### 1. Determine affected modules

Use the request, supplied paths, `rg --files`, targeted code searches, interfaces
and nearby callers/tests to identify likely owners. This read-only reconnaissance
is allowed before module-memory loading. Consult relevant entries in
`Deep_Reflective_Reader/docs/modules/index.md` when ownership or document locations
are unclear; the UI module is documented separately below.

Briefly identify modules likely to change, supporting modules whose contracts must
be understood, and any uncertainty. A supporting dependency does not automatically
become an implementation target. Routing is provisional: revise it when evidence
changes. Do not require the user to name modules that repository inspection can
identify, or pause an authorized implementation merely to approve the routing.

### 2. Load the relevant contracts before editing

For each affected module, inspect its document headings, then read its current
ownership/boundary rules, the complete design sections governing the feature and
the relevant checklist items with their dependencies and evidence requirements.
Use the module index's paths for root-file modules; do not assume every owner is a
directory. `Deep_Reader_UI` owns:

- `Deep_Reader_UI/module-detailed-design.md`
- `Deep_Reader_UI/module-checklist.md`

Supporting modules need only the contracts consumed or potentially affected by the
change; read their checklists when delivery status or task dependencies matter.
Read enough surrounding context and current corrections to avoid applying a
historical design as today's contract. Search hits alone are not a contract read.
Whole-file reads are appropriate when the scope spans the file or targeted reading
cannot establish the rules, not the default for a small change.

Read relevant sections of `Deep_Reflective_Reader/proposal.md` and
`Deep_Reflective_Reader/high-level-design.md` when the work affects architecture,
cross-module contracts, persistence/source ownership or lifecycle, or when module
memory is ambiguous or conflicting. Their authority is unchanged; they need not
be reread wholesale for a local implementation detail. Read `progress.md` only for
progress/status work or a specific unresolved delivery dependency. Historical
acceptance, README and future-work files are loaded only when relevant.

For repository instructions or skill maintenance, identify the governance area
instead of inventing an application module. Read the affected instructions/skills
and relevant referenced policies; unrelated application checklists are unnecessary.

### 3. Expand or reuse context deliberately

If a new module or contract becomes affected, update the scope and load its relevant
memory before editing it. Within the same task, reuse sections already read if they
remain current; reread after relevant file changes, changed scope, missing context
or conflicting evidence. Do not treat a summary or filename listing as having read
a previously unread contract. Report the files/sections actually inspected and any
material gaps; do not claim a whole-file or repository-wide audit from partial reads.

This policy applies to implementation, documentation, audit and validation workflows.
It supersedes blanket read-all sequences in repository skills, including mandatory
`progress.md` loading. Retain additional task-specific reads that are necessary for
correctness, and all existing write-authorization and validation requirements.

## Required development standards

Before implementation, read the standards applicable to the identified scope:

- Frontend/UI changes: [Frontend development standards](docs/standards/frontend-development.md).
- Backend changes: [Backend development standards](docs/standards/backend-development.md).
- Changes spanning both: read both. Governance-only work reads standards only when
  their rules are being changed or interpreted.

Reuse unchanged standards already read within the same task; load newly applicable
standards before expanding the implementation scope. Check the implementation
against the applicable standards before reporting completion.
If a standards file is empty, do not invent missing rules.
Existing architecture and governance rules still apply.

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
