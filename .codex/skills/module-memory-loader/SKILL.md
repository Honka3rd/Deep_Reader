---
name: module-memory-loader
description: Identify the Deep Reader modules affected by a request, then load only the relevant repository contracts before implementation, audit, validation or documentation work.
---

# Module Memory Loader

Use the scope-first reading policy in [AGENTS.md](../../../AGENTS.md). This skill
performs read-only routing and memory loading. It does not modify source, tests,
configuration or markdown, authorize writes, or replace the requested work.

## Inputs and routing

Start from the user's request and task type. Module names and paths are optional;
infer them through repository inspection instead of asking the user to supply them.
For a known module, verify that it owns the requested behavior. For an unknown or
cross-module request, identify candidate owners first. `repository-routing` is a
workflow, not a module directory.

1. Read applicable `AGENTS.md` instructions.
2. Use supplied paths, `rg --files`, focused code/symbol searches and nearby callers,
   interfaces or tests to identify likely changes. Read-only code inspection is
   allowed before design/checklist loading.
3. Consult relevant entries in `Deep_Reflective_Reader/docs/modules/index.md` if
   ownership or document paths are unclear. Do not load all module designs as a
   routing shortcut. Root-file owners may use named design/checklist files instead
   of a module directory; use the registered paths.
4. State the likely change owners, supporting contracts and remaining uncertainty.
   A called dependency may require understanding without requiring modification.

For instructions/skill maintenance, the affected governance files are the owners;
no application module or checklist needs to be invented. For explicitly requested
routing-only work, report the scope and required next reads, then finish. Otherwise
continue into targeted loading and the already-authorized task without an approval
checkpoint. Routing remains provisional until the relevant contracts are inspected.

## Targeted memory loading

For each module likely to change, inspect headings and read:

- Its current ownership and architecture-boundary rules.
- Complete design sections governing the requested behavior, including applicable
  corrections and directly referenced constraints.
- Relevant checklist items, dependencies and required completion evidence.

For `Deep_Reader_UI`, these live in `Deep_Reader_UI/module-detailed-design.md` and
`Deep_Reader_UI/module-checklist.md`. Other owners use their registered documents.
For supporting modules, load the consumed/affected interface contracts; read their
checklists only when dependency or delivery status matters. A search hit or file
listing is not sufficient evidence that a contract was read.

Choose additional memory by its relevance:

| Source | Read when |
| --- | --- |
| `Deep_Reflective_Reader/proposal.md` and `high-level-design.md` | Architecture, cross-module contracts, persistence/source ownership or lifecycle is affected, or local memory is ambiguous/conflicting. Read the governing sections and expand if needed. |
| Development standards | Frontend standard for UI changes, backend standard for backend changes, both for work spanning them. Governance-only work needs them when changing/interpreting their rules. |
| `Deep_Reflective_Reader/progress.md` | The task concerns progress/status or a specific unresolved delivery dependency. It is not a default prerequisite. |
| README, design notes, historical acceptance, future work | They resolve a concrete question or supply evidence required by the affected contract. Do not automatically load adjacent files. |

Read full files when the requested scope covers them or targeted reading cannot
establish the constraints. Do not confuse historical implementation evidence with
current policy. This loading policy does not reduce the authority of global memory
or the requirement to validate an implementation.

## Scope changes, conflicts and reuse

If inspection or implementation reveals another affected module/contract, revise
the scope and load its relevant memory before editing it. The new module is not
covered merely because the initial module was inspected. Reuse sections already
read in the same task only while current; reread after relevant file changes, scope
changes, missing context or conflicting evidence. A conversation summary can guide
search but is not a substitute for a previously unread contract.

Memory precedence remains: AGENTS instructions, proposal, high-level design, module
index, module design, module checklist, then progress/history. Inspect code to
verify implementation claims and report contradictions rather than inventing
completion status. Missing or ambiguous memory requires focused investigation;
ask only if the unresolved information is necessary and cannot be found locally.
Do not make dependent edits while the required boundary remains unresolved; continue
independent work within the confirmed scope.

## Output and validation

Provide a concise scope/loading note, not a second copy of the documents:

- Task type; owners likely to change; supporting modules and why they matter.
- Paths and sections actually read, including applicable standards.
- Material gaps/conflicts, any scope expansion, and the next action.
- Files modified by this loader: none.

Before handing back to implementation or review, confirm that the relevant current
contracts and checklist dependencies were inspected, the paths/owners exist, and
no files were changed by the loader. Do not claim whole-file or repository-wide
coverage from selected sections. Documentation edits or progress synchronization
remain separate, authorized work under AGENTS governance; this read-only skill
does not grant those permissions.
