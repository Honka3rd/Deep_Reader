# Phase 1 SQL/DDL Migration Plan

## 1. Purpose

This document converts `db/phase-1-schema-design.md` into a SQL/DDL migration planning reference.

This document is the planning reference. Later implementation materializes the first core PostgreSQL migration shape in `db/migrations/001_phase_1_core_hierarchy.sql`; this document still does not define ORM models, repository interfaces, runtime read/write behavior, fixtures, API changes, or backend selection.

The goal is to define the minimum migration sequence, DDL work units, dependency order, rollback expectations, and validation gates that a future implementation task must follow.

## 2. Inputs

Source documents:

- `Deep_Reflective_Reader/db/phase-1-schema-design.md`
- `Deep_Reflective_Reader/db/module-detailed-design.md`
- `Deep_Reflective_Reader/db/module-checklist.md`
- `Deep_Reflective_Reader/db/storage-contract-design.md`
- `Deep_Reflective_Reader/db/structured-document-jsonb-evaluation.md`
- `Deep_Reflective_Reader/db/structured-document-jsonb-evaluation-readiness.md`
- `Deep_Reflective_Reader/proposal.md`
- `Deep_Reflective_Reader/high-level-design.md`

The migration plan inherits these fixed decisions:

- DB-generated primary keys by default.
- No production dependency on current Python-generated `unit_id`.
- No automatic public/domain IDs for hierarchy nodes.
- `documents.current_structure_version` is authoritative.
- Initial accepted parse creates `structure_version = 1`.
- Current-state-only hierarchy for Phase 1.
- No row-level hierarchy `structure_version`.
- No immutable hierarchy history.
- No durable staging hierarchy tables.
- Content blocks are lazy materialized and separately persisted.
- Artifacts use one logical `artifacts` table surface with `artifact_type` and type-specific payload.
- `document_profile` is advisory document-scoped metadata.
- Raw bytes remain file-backed/object-backed; DB stores only raw-source metadata.
- Parse events are minimal provenance, not event sourcing.
- Hard reparse physically deletes document-level content blocks and artifacts.
- Derived `source_structure_version` validation remains application-level.

## 3. Migration Scope

Phase 1 migration planning covers future DDL work units for:

1. `documents`
2. `raw_source_metadata`
3. `document_profile`
4. `chapters`
5. `sections`
6. `task_units`
7. `content_blocks`
8. `artifacts`
9. `parse_events`
10. optional `structured_document_snapshots`

Out of scope:

- Complete executable SQL DDL for every Phase 1 entity beyond the first core hierarchy subset.
- ORM class definitions.
- Repository/storage interfaces.
- Runtime read/write switching.
- Production migration of existing JSON identities.
- DB storage of raw document bytes or extracted raw text.
- Retrieval index physical storage.
- Category-specific artifact tables.
- Public document UUID/key/slug.
- Historical hierarchy tables.
- Staging hierarchy tables.
- DB triggers for `source_structure_version` freshness.

## 4. DDL Work Units

Future implementation should split Phase 1 DDL into small, reviewable work units:

| Work Unit | Purpose | Dependency |
|---|---|---|
| `documents` | Create document lifecycle anchor and authoritative `current_structure_version`. | none |
| `raw_source_metadata` | Store metadata-only reference to file/object-backed raw source. | `documents` |
| `document_profile` | Store advisory document-scoped profile snapshot. | `documents` |
| `chapters` | Store current accepted chapter rows. | `documents` |
| `sections` | Store current accepted section rows. | `documents`, `chapters` |
| `task_units` | Store current task-unit rows under sections. | `documents`, `sections` |
| `content_blocks` | Store lazy materialized derived blocks under task units. | `documents`, `task_units` |
| `artifacts` | Store common typed artifact records with validated target metadata. | `documents`; target validation is application-level |
| `parse_events` | Store minimal document-scoped parse provenance. | `documents` |
| `structured_document_snapshots` | Optional validation/parity/debug snapshot only. | `documents` |

This ordering keeps the hierarchy anchor available before child rows and keeps derived resources after authoritative hierarchy tables.

## 5. Creation Order

Future migration files should create objects in this order:

1. Document lifecycle table.
2. Metadata tables that directly depend only on document identity:
   - raw-source metadata
   - advisory document profile
   - parse events
   - optional structured document snapshots
3. Current hierarchy tables:
   - chapters
   - sections
   - task units
4. Derived resource tables:
   - content blocks
   - artifacts
5. Basic foreign keys and ownership constraints.
6. Candidate uniqueness and lookup indexes after physical query patterns are confirmed.

Index details remain a separate checklist task because query patterns, physical column types, and backend-specific choices still need implementation-stage confirmation.

## 6. Foreign-Key Direction

Future DDL should preserve these ownership directions:

```text
documents
  -> raw_source_metadata
  -> document_profile
  -> parse_events
  -> structured_document_snapshots
  -> chapters
    -> sections
      -> task_units
        -> content_blocks
  -> artifacts
```

Foreign keys should enforce ownership and prevent orphan rows where the target table is static and unambiguous.

Artifact target references need special care. Phase 1 artifacts use one logical artifact entity with `target_type` and `target_id`; a single DB foreign key cannot represent every possible target table without polymorphic complexity. Future implementation should keep artifact target resolution hierarchy-aware at the application/repository layer unless a later design introduces dedicated target join tables.

## 7. Delete and Retention Strategy

Document deletion:

- Deleting a document should remove document-scoped DB rows.
- Parse events are retained only for the lifetime of the document.
- Raw-source metadata should be removed with the document, while raw bytes and extracted raw text remain governed by the raw file/object storage deletion policy.

Hard reparse:

- Hard reparse is not document deletion.
- Candidate hierarchy is built and validated before destructive DB work starts.
- In one transaction, future runtime should replace current hierarchy rows, advance `documents.current_structure_version`, explicitly delete all `content_blocks` for the document, explicitly delete all `artifacts` for the document, write one parse event, and commit.
- Cascades may be defensive, but the service layer must still own the explicit hard-reparse lifecycle.

Derived rows:

- `content_blocks.source_structure_version` and `artifacts.source_structure_version` are provenance and application-level stale-validation metadata.
- Phase 1 should not use DB triggers or cross-table constraints to enforce freshness against `documents.current_structure_version`.

## 8. Rollback Plan

Future DDL rollback should reverse creation order:

1. Drop derived resource tables.
2. Drop hierarchy child tables.
3. Drop hierarchy parent tables.
4. Drop document-scoped metadata/provenance tables.
5. Drop document lifecycle table last.

Rollback planning must preserve file-backed source data. Since this plan does not move raw bytes or extracted raw text into the DB and does not production-migrate existing JSON identity, a failed DB rollout must not invalidate existing file-backed structured outputs.

Rollback must not depend on rehydrating hierarchy from artifacts, profile metadata, parse events, or JSONB parity snapshots. Those surfaces are not hierarchy authority.

## 9. Validation Gates

Future migration implementation should not be considered ready until these documentation-to-implementation gates are satisfied:

1. Physical table names, column types, nullability, FK rules, uniqueness, and index candidates are confirmed.
2. Empty-database migration can create all Phase 1 tables in dependency order.
3. New-document ingestion can write document, hierarchy, raw-source metadata, optional profile, and initial parse event with `current_structure_version = 1`.
4. Current hierarchy reads resolve through `documents -> chapters -> sections -> task_units`.
5. No runtime path depends on Python-generated `unit_id` as production identity.
6. Content blocks can be created lazily after task-unit persistence and treated as stale by application logic when their source version mismatches.
7. Artifacts write through one common artifact surface and remain non-authoritative.
8. Hard reparse validation proves candidate failure leaves existing hierarchy and derived rows untouched.
9. Hard reparse success proves current hierarchy replacement, version advancement, content-block deletion, artifact deletion, and parse-event insertion happen transactionally.
10. Optional `structured_document_snapshots` conflicts are treated as validation failures, not fallback reads.

## 10. Future Migration File Shape

Future migration implementation may choose one or more migration files, but the first reviewable shape should be:

1. Phase 1 core document and metadata DDL.
2. Phase 1 hierarchy DDL.
3. Phase 1 derived resource DDL.
4. Phase 1 optional parity snapshot DDL.
5. Phase 1 validation fixtures and migration smoke checks.

The first core hierarchy implementation uses `db/migrations/001_phase_1_core_hierarchy.sql` as the PostgreSQL-targeted DDL shape and keeps SQLite-only validation SQL under `db/sqlite_validation/`. This plan still does not prescribe a migration framework or ORM migration tooling.

## 11. Governance Validation

This migration plan does not introduce:

- root `sections[]`
- `structure_nodes`
- flat `task_units` as primary runtime truth
- profile authority
- artifact hierarchy authority
- raw-byte DB storage
- category-specific artifact tables in Phase 1
- DB schema as parser authority
- runtime read/write switch
- ORM models
- repository interfaces
- migration scripts

## 12. Completion Boundary

The checklist item "Convert Phase 1 schema design into SQL/DDL migration plan" is complete when this document is present and linked from `db/module-checklist.md`. The later core hierarchy implementation is tracked separately by the implementation-slice checklist evidence.

The following remain separate unchecked implementation-planning tasks:

- physical table, column, foreign-key, uniqueness, and index candidates
- ORM/model mapping
- repository/storage interfaces
- migration/evaluation fixtures
- hard reparse transaction implementation details
- content-block and artifact persistence implementation
- backend configuration integration
- runtime read/write switch planning
