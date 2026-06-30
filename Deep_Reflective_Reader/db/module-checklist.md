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
- `Deep_Reflective_Reader/db/phase-1-schema-design.md`

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

- [x] Derive Phase 1 logical DB schema proposal from repository memory.
  Evidence: `Deep_Reflective_Reader/db/module-detailed-design.md (Logical DB Schema Proposal (Phase 1))`; `Deep_Reflective_Reader/db/storage-contract-design.md`; `Deep_Reflective_Reader/document_structure/module-detailed-design.md`; `Deep_Reflective_Reader/shared/module-detailed-design.md`; `Deep_Reflective_Reader/document_preparation/module-detailed-design.md`; `Deep_Reflective_Reader/config/module-detailed-design.md`; maintainer request for logical schema proposal in current architecture task.
  Notes: Documentation-only logical persistence model covering domains, entities, responsibilities, relationships, ownership boundaries, JSONB-vs-relational placement rationale, and open design questions; no SQL, DDL, ORM, repository interface, migration, fixtures, runtime behavior, or API changes.

- [x] Confirm optional `StructuredDocument` JSONB snapshot boundary for Phase 1.
  Evidence: `Deep_Reflective_Reader/db/module-detailed-design.md (JSONB vs Relational Placement Rationale)`; maintainer clarification in current architecture task.
  Notes: Optional full `StructuredDocument` JSONB snapshot may exist only as a validation/parity/debug artifact; relational current hierarchy and `Document.current_structure_version` remain authoritative, and JSONB conflict is a validation failure rather than fallback or dual-authority behavior.

- [x] Confirm Phase 1 advisory profile snapshot placement and minimum metadata.
  Evidence: `Deep_Reflective_Reader/db/module-detailed-design.md (Logical DB Schema Proposal (Phase 1), ProfileSnapshot)`; maintainer clarification in current architecture task.
  Notes: Phase 1 includes `ProfileSnapshot` / `document_profiles` as a separate document-scoped logical entity linked to `Document`, storing profile payload, version metadata, generation/source metadata, optional language/script/title/author/source metadata, optional `source_structure_version`, and advisory diagnostics when produced. It remains advisory only and must not become hierarchy truth, parser authority, artifact availability authority, retrieval authority, or a mutator of chapters/sections/task units.

- [x] Confirm Phase 1 artifact payload grouping strategy.
  Evidence: `Deep_Reflective_Reader/db/module-detailed-design.md (Logical DB Schema Proposal (Phase 1), Artifact)`; maintainer clarification in current architecture task.
  Notes: Phase 1 uses one common logical `Artifact` entity with `artifact_type`, target metadata, type-specific payload, and lifecycle/provenance metadata. It must not split into category-specific artifact tables in Phase 1; splitting can be reconsidered later only if schemas stabilize, type-specific constraints become important, query patterns require dedicated tables, or generic payload validation becomes insufficient.

- [x] Confirm Phase 1 public document identity requirement.
  Evidence: `Deep_Reflective_Reader/db/module-detailed-design.md (DB-Era Identity Strategy, Logical DB Schema Proposal (Phase 1))`; maintainer clarification in current architecture task.
  Notes: There is no current explicit business requirement for a separate public document identity. `documents.id` is sufficient for the first DB-backed internal API and application use. Public document UUID/key/slug remains a future extensibility point only for concrete external stability requirements such as public sharing, external SDK/API, cross-system integration, permanent external references, or multi-tenant public URLs.

- [x] Confirm Phase 1 parse event retention policy.
  Evidence: `Deep_Reflective_Reader/db/module-detailed-design.md (Parse Event Provenance, Retention Policy)`; maintainer clarification in current architecture task.
  Notes: Parse events are retained for the lifetime of the document and deleted when the document is deleted. They remain document-scoped provenance only; Phase 1 does not keep parse events after document deletion as independent audit records and does not introduce archival, TTL, or compliance retention.

- [x] Confirm Phase 1 raw document storage boundary.
  Evidence: `Deep_Reflective_Reader/db/module-detailed-design.md (Logical DB Schema Proposal (Phase 1), RawSourceMetadata)`; maintainer clarification in current architecture task.
  Notes: Raw uploaded documents remain canonical user-owned file-backed source files for the first DB rollout. The DB stores only raw-source metadata such as `document_id`, source location/file path/object key, original filename, MIME type, file size, checksum/fingerprint when available, `uploaded_at`, and ownership scope when applicable. Raw bytes, future object storage, and DB-backed raw storage remain separate tracks.

## Needs Confirmation

No unresolved confirmation items identified in this pass.

## Future Task Policy

New future tasks for this module must be added here first as unchecked items:

- [ ] Convert Phase 1 logical schema into implementation-ready schema design
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
