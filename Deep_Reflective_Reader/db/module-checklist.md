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
- `Deep_Reflective_Reader/db/phase-1-sql-ddl-migration-plan.md`
- `Deep_Reflective_Reader/db/phase-1-physical-schema-candidates.md`
- `Deep_Reflective_Reader/db/phase-1-orm-model-mapping-plan.md`
- `Deep_Reflective_Reader/db/phase-1-repository-storage-interface-plan.md`
- `Deep_Reflective_Reader/db/phase-1-document-structure-storage-integration-plan.md`
- `Deep_Reflective_Reader/db/phase-1-raw-source-metadata-persistence-plan.md`
- `Deep_Reflective_Reader/db/phase-1-document-profile-persistence-plan.md`
- `Deep_Reflective_Reader/db/phase-1-migration-evaluation-fixtures-plan.md`
- `Deep_Reflective_Reader/db/phase-1-empty-database-ingestion-path-plan.md`
- `Deep_Reflective_Reader/db/phase-1-hard-reparse-transaction-plan.md`
- `Deep_Reflective_Reader/db/phase-1-parse-event-persistence-plan.md`
- `Deep_Reflective_Reader/db/phase-1-content-block-persistence-plan.md`
- `Deep_Reflective_Reader/db/phase-1-artifact-persistence-plan.md`
- `Deep_Reflective_Reader/db/phase-1-derived-row-validation-plan.md`
- `Deep_Reflective_Reader/db/phase-1-config-backend-integration-plan.md`
- `Deep_Reflective_Reader/db/phase-1-structured-document-jsonb-parity-snapshot-plan.md`
- `Deep_Reflective_Reader/db/phase-1-current-state-hierarchy-validation-test-plan.md`
- `Deep_Reflective_Reader/db/phase-1-db-parity-validation-fixtures-plan.md`
- `Deep_Reflective_Reader/db/phase-1-runtime-read-write-switch-plan.md`
- `Deep_Reflective_Reader/db/phase-1-db-failure-mode-rollback-plan.md`

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

- [x] Convert Phase 1 schema design into SQL/DDL migration plan.
  Evidence: `Deep_Reflective_Reader/db/phase-1-sql-ddl-migration-plan.md`; `Deep_Reflective_Reader/db/phase-1-schema-design.md`; `Deep_Reflective_Reader/db/module-detailed-design.md`.
  Notes: Documentation-only migration planning reference covering DDL work units, dependency order, foreign-key direction, delete/retention strategy, rollback expectations, validation gates, future migration file shape, and governance guardrails. It does not create executable SQL DDL, ORM models, repository interfaces, migration scripts, runtime behavior, fixtures, tests, API changes, or backend selection.

- [x] Define physical table, column, foreign-key, uniqueness, and index candidates for Phase 1.
  Evidence: `Deep_Reflective_Reader/db/phase-1-physical-schema-candidates.md`; `Deep_Reflective_Reader/db/phase-1-schema-design.md`; `Deep_Reflective_Reader/db/phase-1-sql-ddl-migration-plan.md`.
  Notes: Documentation-only candidate physical schema reference covering table names, column/type candidates, required/nullability candidates, foreign-key directions, uniqueness candidates, index candidates, delete behavior candidates, cross-table consistency rules, and governance guardrails. It does not create executable SQL DDL, ORM models, repository interfaces, migration scripts, runtime behavior, fixtures, tests, API changes, or backend selection.

- [x] Define ORM/model mapping plan for Phase 1 entities without making ORM classes parser authority.
  Evidence: `Deep_Reflective_Reader/db/phase-1-orm-model-mapping-plan.md`; `Deep_Reflective_Reader/db/phase-1-physical-schema-candidates.md`; `Deep_Reflective_Reader/db/phase-1-schema-design.md`.
  Notes: Documentation-only ORM/model mapping plan covering candidate ORM record classes, table mapping, relationship mapping, model responsibility boundaries, DTO/domain conversion boundaries, loading/session policy candidates, cascade/lifecycle boundaries, and validation responsibility split. It does not create executable ORM model code, SQL DDL, repository interfaces, migration scripts, runtime behavior, fixtures, tests, API changes, or backend selection.

- [x] Define DB repository/storage interfaces without making schema authority.
  Evidence: `Deep_Reflective_Reader/db/phase-1-repository-storage-interface-plan.md`; `Deep_Reflective_Reader/db/phase-1-orm-model-mapping-plan.md`; `Deep_Reflective_Reader/db/phase-1-physical-schema-candidates.md`.
  Notes: Documentation-only repository/storage interface plan covering candidate storage ports, ownership boundaries, pseudo-interface operations, transaction/unit-of-work boundary, DTO boundary candidates, schema authority guardrails, and runtime integration boundaries. It does not create Python repository interfaces, executable ORM model code, SQL DDL, migration scripts, runtime behavior, fixtures, tests, API changes, or backend selection.

- [x] Define document_structure storage integration plan for DB-backed hierarchy writes and reads.
  Evidence: `Deep_Reflective_Reader/db/phase-1-document-structure-storage-integration-plan.md`; `Deep_Reflective_Reader/db/phase-1-repository-storage-interface-plan.md`; `Deep_Reflective_Reader/db/phase-1-schema-design.md`.
  Notes: Documentation-only integration plan covering `document_structure` ownership, DB-backed initial hierarchy writes, current hierarchy reads, hard-reparse replacement, mapper/DTO boundaries, fail-fast behavior, file/DB coexistence, and governance guardrails. It does not create Python code, repository interfaces, executable ORM model code, SQL DDL, migration scripts, runtime behavior, fixtures, tests, API changes, or backend selection.

- [x] Define raw-source metadata persistence plan while keeping raw bytes file-backed/object-backed.
  Evidence: `Deep_Reflective_Reader/db/phase-1-raw-source-metadata-persistence-plan.md`; `Deep_Reflective_Reader/db/phase-1-schema-design.md`; `Deep_Reflective_Reader/db/module-detailed-design.md`.
  Notes: Documentation-only plan for DB metadata rows that reference file/object-backed raw sources without storing raw bytes or making source metadata parser authority. It does not create Python code, SQL DDL, ORM models, repository interfaces, migrations, fixtures, tests, runtime behavior, API changes, object storage integration, or backend selection.

- [x] Define advisory document_profile persistence plan without profile hierarchy authority.
  Evidence: `Deep_Reflective_Reader/db/phase-1-document-profile-persistence-plan.md`; `Deep_Reflective_Reader/profile/module-detailed-design.md`; `Deep_Reflective_Reader/db/phase-1-schema-design.md`.
  Notes: Documentation-only plan for document-scoped advisory profile snapshots, including metadata boundaries and profile non-authority rules. It does not create Python code, SQL DDL, ORM models, repository interfaces, migrations, fixtures, tests, runtime behavior, API changes, profile parser changes, or backend selection.

- [x] Define migration/evaluation fixtures that do not production-migrate existing JSON identity.
  Evidence: `Deep_Reflective_Reader/db/phase-1-migration-evaluation-fixtures-plan.md`; `Deep_Reflective_Reader/db/structured-document-jsonb-evaluation-readiness.md`; `Deep_Reflective_Reader/db/phase-1-schema-design.md`.
  Notes: Documentation-only fixture planning that treats existing JSON IDs as reference evidence only and focuses future validation on semantic hierarchy shape and ordering. It does not create fixtures, Python code, SQL DDL, ORM models, repository interfaces, migrations, tests, runtime behavior, API changes, or backend selection.

- [x] Define empty-database / new-document ingestion migration path before any existing JSON production migration.
  Evidence: `Deep_Reflective_Reader/db/phase-1-empty-database-ingestion-path-plan.md`; `Deep_Reflective_Reader/db/phase-1-document-structure-storage-integration-plan.md`; `Deep_Reflective_Reader/db/phase-1-raw-source-metadata-persistence-plan.md`.
  Notes: Documentation-only rollout plan that prioritizes new-document ingestion into an empty DB before production migration of existing JSON outputs. It does not create Python code, SQL DDL, ORM models, repository interfaces, migrations, fixtures, tests, runtime behavior, API changes, or backend selection.

- [x] Define transaction-level hard reparse implementation plan.
  Evidence: `Deep_Reflective_Reader/db/phase-1-hard-reparse-transaction-plan.md`; `Deep_Reflective_Reader/db/phase-1-document-structure-storage-integration-plan.md`; `Deep_Reflective_Reader/db/module-detailed-design.md`.
  Notes: Documentation-only transaction plan for validated hard reparse replacement, document-level structure-version advance, derived-row deletion, and parse-event append. It does not create Python code, SQL DDL, ORM models, repository interfaces, migrations, fixtures, tests, runtime behavior, API changes, or backend selection.

- [x] Define parse event persistence implementation plan.
  Evidence: `Deep_Reflective_Reader/db/phase-1-parse-event-persistence-plan.md`; `Deep_Reflective_Reader/db/module-detailed-design.md`; `Deep_Reflective_Reader/db/phase-1-schema-design.md`.
  Notes: Documentation-only plan for minimal `initial_parse` and `hard_reparse` provenance retained for document lifetime. It does not create Python code, SQL DDL, ORM models, repository interfaces, migrations, fixtures, tests, runtime behavior, API changes, or backend selection.

- [x] Define content-block relational persistence implementation plan.
  Evidence: `Deep_Reflective_Reader/db/phase-1-content-block-persistence-plan.md`; `Deep_Reflective_Reader/shared/module-detailed-design.md`; `Deep_Reflective_Reader/db/phase-1-schema-design.md`.
  Notes: Documentation-only plan for lazy materialized content blocks linked to current task-unit hierarchy with application-level freshness validation. It does not create Python code, SQL DDL, ORM models, repository interfaces, migrations, fixtures, tests, runtime behavior, API changes, or backend selection.

- [x] Define artifact relational persistence implementation plan.
  Evidence: `Deep_Reflective_Reader/db/phase-1-artifact-persistence-plan.md`; `Deep_Reflective_Reader/document_structure/module-detailed-design.md`; `Deep_Reflective_Reader/db/phase-1-schema-design.md`.
  Notes: Documentation-only plan for one common artifact entity with `artifact_type`, validated hierarchy-aware target metadata, type-specific payload, and provenance metadata. It does not create Python code, SQL DDL, ORM models, repository interfaces, migrations, fixtures, tests, runtime behavior, API changes, artifact generators, or backend selection.

- [x] Define application-level stale/invalid derived-row validation behavior.
  Evidence: `Deep_Reflective_Reader/db/phase-1-derived-row-validation-plan.md`; `Deep_Reflective_Reader/db/phase-1-content-block-persistence-plan.md`; `Deep_Reflective_Reader/db/phase-1-artifact-persistence-plan.md`.
  Notes: Documentation-only plan for validating derived-row freshness against `documents.current_structure_version` at the application layer. It does not create Python code, SQL DDL, ORM models, repository interfaces, migrations, fixtures, tests, runtime behavior, API changes, or backend selection.

- [x] Define backend configuration integration with `config/`.
  Evidence: `Deep_Reflective_Reader/db/phase-1-config-backend-integration-plan.md`; `Deep_Reflective_Reader/config/module-detailed-design.md`; `Deep_Reflective_Reader/db/phase-1-repository-storage-interface-plan.md`.
  Notes: Documentation-only plan for future backend policy ownership in `config/` without enabling DB reads, DB writes, hidden dual-write behavior, or runtime fallback. It does not create Python code, config implementation, SQL DDL, ORM models, repository interfaces, migrations, fixtures, tests, runtime behavior, API changes, or backend selection.

- [x] Define optional `StructuredDocument` JSONB parity snapshot validation plan without runtime fallback authority.
  Evidence: `Deep_Reflective_Reader/db/phase-1-structured-document-jsonb-parity-snapshot-plan.md`; `Deep_Reflective_Reader/db/structured-document-jsonb-evaluation.md`; `Deep_Reflective_Reader/db/phase-1-schema-design.md`.
  Notes: Documentation-only plan for optional JSONB parity/debug snapshots as validation evidence only, not runtime hierarchy authority or fallback. It does not create Python code, SQL DDL, ORM models, repository interfaces, migrations, fixtures, tests, runtime behavior, API changes, JSONB persistence, or backend selection.

- [x] Define validation tests for current-state-only hierarchy replacement.
  Evidence: `Deep_Reflective_Reader/db/phase-1-current-state-hierarchy-validation-test-plan.md`; `Deep_Reflective_Reader/db/phase-1-hard-reparse-transaction-plan.md`; `Deep_Reflective_Reader/db/phase-1-document-structure-storage-integration-plan.md`.
  Notes: Documentation-only future test plan for initial parse, current hierarchy reads, hard reparse replacement, derived-row deletion, and no JSONB fallback. It does not create tests, fixtures, Python code, SQL DDL, ORM models, repository interfaces, migrations, runtime behavior, API changes, or backend selection.

- [x] Define DB parity validation fixtures against file-backed `StructuredDocument` golden outputs.
  Evidence: `Deep_Reflective_Reader/db/phase-1-db-parity-validation-fixtures-plan.md`; `Deep_Reflective_Reader/db/structured-document-jsonb-evaluation-readiness.md`; `Deep_Reflective_Reader/db/phase-1-migration-evaluation-fixtures-plan.md`.
  Notes: Documentation-only fixture plan for future semantic parity checks between DB-backed relational hierarchy and file-backed golden outputs without making JSON IDs production identity. It does not create fixtures, tests, Python code, SQL DDL, ORM models, repository interfaces, migrations, runtime behavior, API changes, or backend selection.

- [x] Define runtime read/write switch plan behind explicit backend policy, without enabling it by default.
  Evidence: `Deep_Reflective_Reader/db/phase-1-runtime-read-write-switch-plan.md`; `Deep_Reflective_Reader/db/phase-1-config-backend-integration-plan.md`; `Deep_Reflective_Reader/config/module-detailed-design.md`.
  Notes: Documentation-only plan for future explicit, observable, reversible backend read/write policy. It does not enable DB runtime reads, DB runtime writes, hidden dual writes, silent fallback, Python code, SQL DDL, ORM models, repository interfaces, migrations, fixtures, tests, API changes, or backend selection.

- [x] Define rollback and failure-mode behavior for failed initial parse, failed hard reparse, and failed derived-resource cleanup.
  Evidence: `Deep_Reflective_Reader/db/phase-1-db-failure-mode-rollback-plan.md`; `Deep_Reflective_Reader/db/phase-1-hard-reparse-transaction-plan.md`; `Deep_Reflective_Reader/db/module-detailed-design.md`.
  Notes: Documentation-only failure-mode plan covering initial parse failure, hard reparse validation failure, hard reparse transaction failure, and derived cleanup failure. It does not create Python code, SQL DDL, ORM models, repository interfaces, migrations, fixtures, tests, runtime behavior, API changes, or backend selection.

- [x] Implement Phase 1 core DB hierarchy persistence slice for new-document isolated validation.
  Evidence: `Deep_Reflective_Reader/db/sqlite_validation/phase_1_core_hierarchy_schema.sql`; `Deep_Reflective_Reader/db/phase_1_core_schema.py`; `Deep_Reflective_Reader/db/sqlite_core_document_store.py`; `Deep_Reflective_Reader/scripts/test_db_phase_1_core_hierarchy_persistence.py`.
  Notes: Keeps the executable SQLite schema under `db/sqlite_validation/` for isolated local validation. The validation slice covers schema application, accepted hierarchy write, required namespace/document-name identity, DB-generated identity readback, raw-source metadata readback, no `documents.raw_text`, and initial parse provenance validation. It does not enable production runtime DB reads/writes, profile persistence, content-block persistence, artifact persistence, JSONB runtime fallback, public/domain IDs, existing JSON production migration, or backend selection.

- [x] Complete PostgreSQL Phase 1 DDL surface for remaining schema entities.
  Evidence: `Deep_Reflective_Reader/db/migrations/001_phase_1_core_hierarchy.sql`; `Deep_Reflective_Reader/scripts/test_db_phase_1_postgresql_migration_shape.py`; `Deep_Reflective_Reader/db/phase-1-schema-design.md`; `Deep_Reflective_Reader/db/phase-1-physical-schema-candidates.md`.
  Notes: Extends the PostgreSQL-targeted migration shape beyond the core hierarchy subset to include `document_profile`, `content_blocks`, `artifacts`, and optional `structured_document_snapshots`, while preserving raw-source metadata-only storage, one common artifact table, application-level polymorphic artifact target validation, and no runtime backend switch.

- [x] Enforce same-document parent consistency for PostgreSQL hierarchy and content-block rows.
  Evidence: `Deep_Reflective_Reader/db/migrations/001_phase_1_core_hierarchy.sql`; `Deep_Reflective_Reader/scripts/test_db_phase_1_postgresql_migration_shape.py`; `Deep_Reflective_Reader/scripts/test_db_phase_1_postgresql_relational_consistency_smoke.py`; `Deep_Reflective_Reader/db/phase-1-physical-schema-candidates.md`; `Deep_Reflective_Reader/db/phase-1-sql-ddl-migration-plan.md`.
  Notes: Adds parent-side `unique(id, document_id)` constraints and composite foreign keys for `sections(chapter_id, document_id)`, `task_units(section_id, document_id)`, and `content_blocks(task_unit_id, document_id)` so duplicated child `document_id` values cannot disagree with static parent ownership. Artifact polymorphic target validation remains application-level and no composite artifact target foreign keys were added.

- [x] Enforce PostgreSQL parse_event event-specific row-shape constraints.
  Evidence: `Deep_Reflective_Reader/db/migrations/001_phase_1_core_hierarchy.sql`; `Deep_Reflective_Reader/scripts/test_db_phase_1_postgresql_migration_shape.py`; `Deep_Reflective_Reader/scripts/test_db_phase_1_postgresql_relational_consistency_smoke.py`; `Deep_Reflective_Reader/db/phase-1-parse-event-persistence-plan.md`; `Deep_Reflective_Reader/db/phase-1-hard-reparse-transaction-plan.md`.
  Notes: Adds named CHECK constraints for `initial_parse` and `hard_reparse` version semantics plus non-negative invalidation counts. The constraints validate parse-event provenance row shape only; `documents.current_structure_version` remains authoritative and no triggers, event sourcing, hierarchy history, or parse-event-driven version advancement were introduced.

- [x] Enforce conservative PostgreSQL numeric and span representation constraints.
  Evidence: `Deep_Reflective_Reader/db/migrations/001_phase_1_core_hierarchy.sql`; `Deep_Reflective_Reader/scripts/test_db_phase_1_postgresql_migration_shape.py`; `Deep_Reflective_Reader/scripts/test_db_phase_1_postgresql_relational_consistency_smoke.py`; `Deep_Reflective_Reader/db/phase-1-schema-design.md`; `Deep_Reflective_Reader/db/phase-1-physical-schema-candidates.md`.
  Notes: Adds named CHECK constraints for non-negative hierarchy/content ordering, raw-source file size, section char offsets, and quote-span endpoints, plus ordered section and quote-span ranges. One-sided quote spans remain allowed because quote-span metadata is optional; the DB validates representation integrity only and does not infer, repair, or classify parser/content spans.

- [x] Define PostgreSQL `updated_at` ownership as application-managed.
  Evidence: `Deep_Reflective_Reader/db/migrations/001_phase_1_core_hierarchy.sql`; `Deep_Reflective_Reader/scripts/test_db_phase_1_postgresql_migration_shape.py`; `Deep_Reflective_Reader/scripts/test_db_phase_1_postgresql_relational_consistency_smoke.py`; `Deep_Reflective_Reader/db/phase-1-schema-design.md`; `Deep_Reflective_Reader/db/phase-1-physical-schema-candidates.md`; `Deep_Reflective_Reader/db/phase-1-orm-model-mapping-plan.md`; `Deep_Reflective_Reader/db/phase-1-repository-storage-interface-plan.md`.
  Notes: Chooses application-managed timestamps for Phase 1. `documents.updated_at` keeps its insert default but must be explicitly set by document mutation statements. `artifacts.updated_at` remains nullable until the first explicit artifact mutation. No PostgreSQL timestamp trigger, lifecycle trigger, hidden ORM hook, or domain mutation trigger was introduced.

- [x] Implement Docker PostgreSQL structured document runtime read/write switch for new documents.
  Evidence: `Deep_Reflective_Reader/db/postgres_structured_document_store.py`; `Deep_Reflective_Reader/db/postgres_structured_document_artifact_repository.py`; `Deep_Reflective_Reader/config/app_DI_config.py`; `Deep_Reflective_Reader/config/container.py`; `Deep_Reflective_Reader/document_preparation/document_preparation_pipeline.py`; `docker-compose.yml`; Docker smoke validation against `documents`, `chapters`, `sections`, `task_units`, and `structured_document_snapshots`; `python -m py_compile`; `Deep_Reflective_Reader/scripts/test_db_phase_1_core_hierarchy_persistence.py`; `Deep_Reflective_Reader/scripts/test_db_phase_1_postgresql_migration_shape.py`.
  Notes: Adds an explicit `file` / `postgres` structured storage backend switch, with Docker defaulting to PostgreSQL and local runtime defaulting to file storage. PostgreSQL writes current hierarchy rows and reads runtime hierarchy from relational `documents -> chapters -> sections -> task_units` rows; optional `structured_document_snapshots` remain parity/debug evidence only, not runtime authority. Existing `data/structured` files are not production-migrated and remain outside this new-document DB path.

- [x] Fix PostgreSQL structured runtime read path for section-summary verification.
  Evidence: `Deep_Reflective_Reader/document_preparation/document_preparation_pipeline.py`; `Deep_Reflective_Reader/db/postgres_structured_document_store.py`; `Deep_Reflective_Reader/scripts/test_postgres_structured_uri_prepare_and_load.py`; `Deep_Reflective_Reader/docs/postgres-structured-runtime-verification.txt`; Docker verification of `/documents/task-layout` returning HTTP 200 after the URI fix; Docker verification of `/documents/section-summary` for `Madame Bovary` runtime section id `2` returning HTTP 200.
  Notes: Preserves `postgres://structured/...` targets through the structured store abstraction instead of converting them through local filesystem `Path` handling, and makes PostgreSQL task-unit writes idempotent on `(section_id, task_unit_order)` to avoid duplicate-key failures during repeated or overlapping task-layout writes. The fix keeps relational hierarchy rows as runtime authority and does not introduce JSONB snapshot fallback.

- [x] Add PostgreSQL lightweight document list/search support
  Evidence: `Deep_Reflective_Reader/db/postgres_structured_document_store.py`; `Deep_Reflective_Reader/db/postgres_structured_document_artifact_repository.py`; `Deep_Reflective_Reader/main.py`
  Notes: Adds `list_documents(query, limit)` against active `documents` rows with namespace scoping and title/name filtering. The API remains read-only and no-heavy-payload; it does not alter hierarchy persistence, parser authority, or task-layout semantics.

- [x] Fix PostgreSQL hard reparse replacement and task-layout id alignment.
  Evidence: `Deep_Reflective_Reader/db/postgres_structured_document_store.py`; `Deep_Reflective_Reader/db/postgres_structured_document_artifact_repository.py`; Docker verification of `/documents/reparse-structure` for `Madame Bovary` with `parser_mode=llm_enhanced` returning HTTP 200 and `section_count=29`; Docker verification of `/documents/task-layout` with `refresh_task_units=true` returning DB task-unit id `1364`; Docker verification of `/documents/Madame%20Bovary/task-units/1364/content?segmented=true` returning HTTP 200 with 8 content blocks.
  Notes: Existing-document parser replacement now clears current hierarchy and derived rows inside the PostgreSQL save transaction before inserting the accepted candidate hierarchy and writing a `hard_reparse` parse event. Repository task-layout/artifact saves preserve `current_structure_version`, and PostgreSQL task-layout refresh returns reloaded DB-generated task-unit ids that the content endpoint can resolve.

- [x] Persist PostgreSQL structured parse provenance for task-layout observability
  Evidence: `Deep_Reflective_Reader/db/postgres_structured_document_store.py`; live reparse validation for `Madame Bovary`; live `/documents/task-layout` validation showing `requested_parser_mode=llm_enhanced`, `effective_parser_mode=common`, `fallback_used=true`, `fallback_reason=abnormal_section_output`
  Notes: PostgreSQL document metadata stores `parse_provenance`, and parse events record the effective parser mode. This is provenance only; `documents.current_structure_version` remains hierarchy version authority.

- [x] Persist OCR run pages through PostgreSQL cursor batching
  Evidence: `Deep_Reflective_Reader/db/postgres_structured_document_store.py`; container `/documents/prepare` verification for `國富論lite`; PostgreSQL verification showing one completed `ocr_runs` row and three `ocr_pages` rows for `國富論lite`.
  Notes: OCR run persistence now uses cursor-level `executemany`, avoiding the previous psycopg connection-method failure. OCR text and pages are persisted as OCR provenance/output only and do not become parser or hierarchy authority.

## Needs Confirmation

No unresolved confirmation items identified in this pass.

## Future Task Policy

New future tasks for this module must be added here first as unchecked items:

- [ ] Production DB rollout still needs broader hard-reparse transaction implementation, relational artifact/content-block persistence beyond structured-document payload metadata, profile persistence wiring, existing JSON migration tooling if required, and broader API/path regression coverage.

After implementation, the task owner must update this checklist and mark the task as completed:

- [x] <completed task>

No DB coding task should be considered complete unless the corresponding module checklist item is updated.

## Maintenance Notes

- This checklist is module memory for DB planning.
- It does not replace `proposal.md`, `high-level-design.md`, or storage contract documents.
- It does not replace future tests or schema validation evidence.
- It intentionally keeps implementation tasks unchecked until implemented.
