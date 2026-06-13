# db Checklist

## Purpose

This checklist records completed documentation decisions and future implementation tasks for the documentation-only `db/` module.

It is used to:

- preserve DB-era identity and lifecycle decisions from maintainer grill-me review
- prevent future schema work from centering current Python-generated `unit_id`
- keep DB implementation work gated behind explicit future tasks
- synchronize DB planning status with `progress.md`

## Source Documents

- `Deep_Reflective_Reader/db/module-detailed-design.md`
- `Deep_Reflective_Reader/proposal.md`
- `Deep_Reflective_Reader/high-level-design.md`
- `Deep_Reflective_Reader/db/storage-contract-inventory.md`
- `Deep_Reflective_Reader/db/storage-contract-design.md`
- `Deep_Reflective_Reader/db/structured-document-jsonb-evaluation.md`
- `Deep_Reflective_Reader/db/structured-document-jsonb-evaluation-readiness.md`

## Rules

- Only completed documentation decisions are listed as checked.
- Future implementation work must remain unchecked until code, tests, and documentation evidence exist.
- This checklist does not authorize schema design, ORM work, migrations, repository interfaces, fixtures, or runtime behavior changes by itself.
- Uncertain items must go to `Needs Confirmation`, not the completed checklist.

## Completed Checklist

- [x] Capture DB-era identity strategy as documentation-only golden source.
  Evidence: `Deep_Reflective_Reader/db/module-detailed-design.md (DB-Era Identity Strategy)`; maintainer-confirmed grill-me decisions in current architecture task.
  Notes: DB-generated primary keys are the default internal identity; public/domain IDs are introduced only for concrete external stability needs; current Python-generated `unit_id` is reference/import evidence only.

- [x] Capture document-level `structure_version` policy.
  Evidence: `Deep_Reflective_Reader/db/module-detailed-design.md (Structure Version Policy)`; maintainer-confirmed grill-me decisions in current architecture task.
  Notes: `documents.current_structure_version` is authoritative; version starts at `1` on initial successful parse and advances monotonically on successful hard reparse.

- [x] Capture current-state-only hierarchy policy for early DB design.
  Evidence: `Deep_Reflective_Reader/db/module-detailed-design.md (Current-State-Only Hierarchy Model)`; maintainer-confirmed grill-me decisions in current architecture task.
  Notes: No historical hierarchy snapshots, row aliases, staging hierarchy tables, candidate hierarchy states, promotion workflows, or per-row hierarchy versions in the first DB design.

- [x] Capture hard reparse transaction and cleanup policy.
  Evidence: `Deep_Reflective_Reader/db/module-detailed-design.md (Hard Reparse Policy)`; maintainer-confirmed grill-me decisions in current architecture task.
  Notes: Validate candidate first; in one transaction replace current hierarchy, advance version, explicitly delete all content blocks/artifacts for the document, write parse event, and commit.

- [x] Capture derived-resource provenance and validation policy.
  Evidence: `Deep_Reflective_Reader/db/module-detailed-design.md (Content-Block Persistence Policy, Artifact Target Policy, Validation Semantics)`; maintainer-confirmed grill-me decisions in current architecture task.
  Notes: Derived rows store `source_structure_version` for provenance and application-level defensive validation, not DB-enforced lifecycle constraints.

- [x] Capture minimal parse event provenance policy.
  Evidence: `Deep_Reflective_Reader/db/module-detailed-design.md (Parse Event Provenance)`; maintainer-confirmed grill-me decisions in current architecture task.
  Notes: Required event types are `initial_parse` and `hard_reparse`; events are provenance only, not full audit/history or version authority.

## Needs Confirmation

No unresolved confirmation items identified in this pass.

## Future Task Policy

New future tasks for this module must be added here first as unchecked items:

- [ ] Derive a first DB schema proposal from the DB golden source.
- [ ] Define DB repository/storage interfaces without making schema authority.
- [ ] Define migration/evaluation fixtures that do not production-migrate existing JSON identity.
- [ ] Define transaction-level hard reparse implementation plan.
- [ ] Define parse event persistence implementation plan.
- [ ] Define content-block relational persistence implementation plan.
- [ ] Define artifact relational persistence implementation plan.
- [ ] Define application-level stale/invalid derived-row validation behavior.
- [ ] Define backend configuration integration with `config/`.
- [ ] Define validation tests for current-state-only hierarchy replacement.

After implementation, the task owner must update this checklist and mark the task as completed:

- [x] <completed task>

No DB coding task should be considered complete unless the corresponding module checklist item is updated.

## Maintenance Notes

- This checklist is module memory for DB planning.
- It does not replace `proposal.md`, `high-level-design.md`, or storage contract documents.
- It does not replace future tests or schema validation evidence.
- It intentionally keeps implementation tasks unchecked until implemented.
