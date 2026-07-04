# db Detailed Design

## 1. Module Purpose

`db/` is the golden source for DB-era persistence planning and isolated Phase 1 DB validation implementation. It records maintainer-approved DB identity, versioning, reparse, and derived-resource lifecycle rules, and now includes the first isolated core hierarchy persistence slice. **[Maintainer-Confirmed] + [Code-Confirmed]**

The implemented slice is intentionally isolated from production runtime read/write selection. It exists to validate the core DB hierarchy path before broader schema, ORM, repository, migration, backend configuration, API, or runtime rollout work. **[Code-Confirmed]**

## 2. Position in Overall Architecture

- Future DB persistence planning / governance layer
- Isolated Phase 1 core hierarchy persistence validation layer
- Cross-module design source for future schema and storage implementation work

The existing file-backed runtime remains valid until DB readiness, rollout, and validation gates are explicitly satisfied. **[From Proposal]**

## 3. Key Files

| File | Responsibility | Notes |
|---|---|---|
| `db/module-detailed-design.md` | Golden source for DB-era identity, versioning, reparse, derived-resource lifecycle, and logical schema proposal policy | Documentation only; no SQL/schema implementation approval |
| `db/module-checklist.md` | Tracks DB-era planning status and future implementation tasks | Future tasks remain unchecked until implemented with evidence |
| `db/storage-contract-inventory.md` | Inventory of current storage contracts before DB migration design | Documentation-only preparation |
| `db/storage-contract-design.md` | Conceptual storage contract boundaries before schema/backend implementation | Documentation-only preparation |
| `db/structured-document-jsonb-evaluation.md` | Phase 1 `StructuredDocument` JSONB-first evaluation plan | Evaluation planning only; no schema approval |
| `db/structured-document-jsonb-evaluation-readiness.md` | Readiness audit for future Phase 1 JSONB evaluation | Audit/planning only; no implementation impact |
| `db/phase-1-schema-design.md` | Implementation-ready schema design reference for Phase 1 logical tables/entities, candidate fields, relationships, ownership boundaries, and constraint candidates | Documentation only; no SQL, ORM, repository interface, migration, runtime behavior, or backend selection |
| `db/migrations/001_phase_1_core_hierarchy.sql` | Executable SQLite schema for the isolated Phase 1 core hierarchy validation slice | Covers `documents`, `raw_source_metadata`, `parse_events`, `chapters`, `sections`, and `task_units`; not production backend rollout |
| `db/phase_1_core_schema.py` | Applies the isolated Phase 1 core hierarchy schema to a SQLite connection | Validation helper only; no production runtime switch |
| `db/sqlite_core_document_store.py` | Minimal SQLite-backed accepted hierarchy write/read adapter | Uses DB-generated IDs on readback; not a production repository abstraction or backend selection mechanism |

## 4. Main Responsibilities

1. Define DB-era identity policy before schema design. **[Maintainer-Confirmed]**
2. Define document-level `structure_version` semantics. **[Maintainer-Confirmed]**
3. Define current-state-only hierarchy persistence policy for early DB design. **[Maintainer-Confirmed]**
4. Define hard reparse transaction and derived-resource invalidation policy. **[Maintainer-Confirmed]**
5. Define minimal parse event provenance requirements. **[Maintainer-Confirmed]**
6. Preserve boundaries from existing repository memory: hierarchy-first, no hidden mutation, metadata advisory-only, and no backend/schema authority. **[From Proposal] + [From HLD]**
7. Derive the first logical relational persistence model from documented domain semantics. **[Maintainer-Confirmed] + [Doc-Confirmed]**
8. Provide an isolated executable validation slice for new-document core hierarchy persistence, without enabling production runtime DB reads or writes. **[Code-Confirmed]**

## 5. Non-Responsibilities

1. Does not implement production database rollout, ORM models, production repositories, API integration, backend selection, or runtime read/write behavior. **[Maintainer-Confirmed] + [Code-Confirmed]**
2. Does not select a final DB backend beyond previously documented PostgreSQL JSONB evaluation assumptions. **[From Proposal]**
3. Does not redefine `StructuredDocument` hierarchy truth. **[From HLD]**
4. Does not make database schema a parser, hierarchy, artifact, profile, retrieval, or raw-document authority. **[From Proposal]**
5. Does not preserve current Python-generated `unit_id` as production identity. **[Maintainer-Confirmed]**
6. Does not introduce immutable hierarchy history, staging hierarchy tables, or candidate promotion workflows in the first DB design. **[Maintainer-Confirmed]**

## 5.1 Implemented Phase 1 Core Validation Slice

The implemented slice validates the smallest DB-backed hierarchy path for new documents. **[Code-Confirmed]**

Implemented scope:

- applies an executable SQLite schema for `documents`, `raw_source_metadata`, `parse_events`, `chapters`, `sections`, and `task_units`
- persists an already accepted `StructuredDocument` hierarchy with `current_structure_version = 1`
- stores raw-source metadata only, not raw bytes
- appends an `initial_parse` parse event
- reads current hierarchy back through `Document -> Chapter -> Section -> TaskUnit`
- exposes DB-generated IDs in the read model instead of current Python-generated `unit_id`
- validates the slice with `scripts/test_db_phase_1_core_hierarchy_persistence.py`

Out of scope:

- production runtime DB read/write switch
- profile persistence
- content-block persistence
- artifact persistence
- hard reparse transaction implementation
- JSONB parity snapshot implementation
- migration of existing JSON outputs
- public/domain identity introduction

## 6. DB-Era Identity Strategy

### 6.1 Default Identity Rule

DB-generated primary keys are the default internal identity and relational link foundation. **[Maintainer-Confirmed]**

They should be used for:

- document rows
- chapter rows
- section rows
- task-unit rows
- content-block rows
- artifact rows
- internal joins and foreign keys
- persistence consistency

### 6.2 Public / Domain Identity Rule

Do not create a separate public/domain identity for every hierarchy surface by default. **[Maintainer-Confirmed]**

At the current stage, there is no explicit business requirement for a separate public document identity. `documents.id` is sufficient for the first DB-backed API and application use. **[Maintainer-Confirmed]**

A separate public/domain identity may be introduced only when there is a concrete long-lived external reference requirement beyond the DB row lifecycle. Candidate areas include:

- public document identity
- artifact target snapshots
- content-block validation metadata

Concrete external stability requirements for public document identity may include public sharing, external SDK/API usage, cross-system integration, permanent external references, or multi-tenant public URLs. Until such a requirement exists, public document identity remains a future extensibility point rather than a Phase 1 requirement. **[Maintainer-Confirmed]**

Separate public/domain IDs carry generation, uniqueness, persistence, mapping, and lifecycle costs. They must be deliberate, not automatic. **[Maintainer-Confirmed]**

### 6.3 Reference / Import Identity Rule

Existing file-based JSON data is reference/test data only and will not be production-migrated as stable API-visible identity. Current Python-generated `unit_id` values are reference/import identity evidence only. **[Maintainer-Confirmed]**

Future DB work must not depend on current Python-generated `unit_id` as the production task-unit identity strategy. **[Maintainer-Confirmed]**

### 6.4 API Identity Rule

Early API references may use DB-generated IDs unless a stable public API requirement is explicitly introduced later. **[Maintainer-Confirmed]**

For the first DB-backed API, the document DB-generated identifier is acceptable as the API-facing document identifier for internal APIs and application use. A separate public document key, UUID, or slug should not be introduced in Phase 1 without a concrete external stability requirement. **[Maintainer-Confirmed]**

API-visible DB IDs are valid only within the current document structure version. If a client uses an ID from an older structure version, the system may return a stale/version error when enough metadata exists, otherwise it may fail as not found. **[Maintainer-Confirmed]**

## 7. Structure Version Policy

### 7.1 Authority

`documents.current_structure_version` is the authoritative current hierarchy version. Parse events are provenance records only and must not become an event-sourced version authority. **[Maintainer-Confirmed]**

### 7.2 Scope

`structure_version` is scoped to the whole document hierarchy snapshot, not to individual chapter, section, or task-unit rows. **[Maintainer-Confirmed]**

The first successful parse creates `structure_version = 1`. Each successful hard reparse advances the version to `2`, `3`, and so on. **[Maintainer-Confirmed]**

### 7.3 Version Type

`structure_version` is a simple monotonic integer per document. It is advanced only after a successful hierarchy replacement transaction. **[Maintainer-Confirmed]**

It is not derived from content hash, parser hash, or structure hash. Those hashes may exist as separate metadata for validation, cache behavior, or observability, but they are not the hierarchy version value. **[Maintainer-Confirmed]**

### 7.4 Hierarchy Row Versioning

Chapter, section, and task-unit rows do not store separate `structure_version` values in the first DB design. **[Maintainer-Confirmed]**

The early model is current-state-only:

- hierarchy rows represent only the current accepted structure
- no historical hierarchy snapshots are retained by default
- no child-level reparse exists
- row-level structure versions can be reconsidered only if future historical or partial-reparse requirements appear

## 8. Current-State-Only Hierarchy Model

Early DB design should model only current hierarchy persistence and transaction cutover. **[Maintainer-Confirmed]**

It should not introduce:

- immutable historical hierarchy snapshots
- old row aliases
- staging hierarchy tables
- candidate hierarchy states
- promotion workflows
- non-current hierarchy persistence areas

Candidate hierarchies are built outside durable DB storage, then validated before the transaction starts. **[Maintainer-Confirmed]**

## 9. Content-Block Persistence Policy

Content blocks are lazy materialized resources linked to task units. They are not embedded inside `StructuredDocument` JSONB as the default persistence strategy. **[Maintainer-Confirmed]**

Once computed, content blocks should be persisted separately and relationally linked. **[Maintainer-Confirmed]**

Content blocks should store:

- `document_id`
- task-unit DB foreign key
- `source_structure_version`
- source hash
- block index or deterministic block metadata
- segmentation version
- span metadata when applicable
- schema/version metadata

Derived `source_structure_version` is for provenance and application-level defensive validation. It is not a DB-enforced cross-table lifecycle constraint in the first design. **[Maintainer-Confirmed]**

## 10. Artifact Target Policy

Artifact targets should not rely on old Python-generated `unit_id`. **[Maintainer-Confirmed]**

Early artifact targets should use DB foreign keys plus validation metadata where applicable:

- target table / target type
- target DB id
- `source_structure_version`
- source hash
- quote span if applicable
- schema/version metadata

Artifacts are derived interaction outputs and must not become hierarchy truth, parser authority, or a way to recreate hierarchy identity. **[From HLD] + [Maintainer-Confirmed]**

## 11. Hard Reparse Policy

### 11.1 Meaning

Hard reparse is a document-level hard refresh / debug recovery operation. It is used when parse quality is too low or when the user rejects the current parsed structure. **[Maintainer-Confirmed]**

It is not a normal production versioned update, and early DB design should not preserve previous derived rows or hierarchy rows as historical records by default. **[Maintainer-Confirmed]**

### 11.2 Candidate Validation

The candidate hierarchy must be built and validated before any destructive DB changes occur. If validation fails, nothing is deleted and the existing current hierarchy remains usable. **[Maintainer-Confirmed]**

### 11.3 Transaction Order

The hard reparse transaction order is:

1. Replace current hierarchy rows with the validated candidate hierarchy.
2. Advance `documents.current_structure_version`.
3. Explicitly delete all derived rows for the document.
4. Write the minimal parse event record.
5. Commit.

All steps occur in one transaction, so external readers must not observe intermediate state. **[Maintainer-Confirmed]**

### 11.4 Derived Resource Cleanup

Successful hard reparse physically deletes all content blocks and all artifacts for the document. **[Maintainer-Confirmed]**

Cleanup is by document ownership, not by `source_structure_version` matching:

- delete content blocks where `document_id = ?`
- delete artifacts where `document_id = ?`

This prioritizes exhaustive document-level hard refresh and avoids leaving stale derived rows behind due to inconsistent metadata. **[Maintainer-Confirmed]**

Cleanup should be an explicit service-layer lifecycle step. Database `ON DELETE CASCADE` may be used defensively to prevent orphans, but it must not be the primary mechanism defining hard reparse semantics. **[Maintainer-Confirmed]**

## 12. Parse Event Provenance

Minimal parse event records are required for both `initial_parse` and `hard_reparse`. **[Maintainer-Confirmed]**

Parse events are append-only provenance metadata. They are not hierarchy history, full audit logs, compliance records, event sourcing, or a retained version-history system. **[Maintainer-Confirmed]**

### 12.1 Event Types

Required event types:

- `initial_parse`
- `hard_reparse`

### 12.2 Initial Parse Event

Initial parse event fields should include:

- `document_id`
- `previous_structure_version = null`
- `new_structure_version = 1`
- `trigger_source = upload / prepare`
- parser mode or parser strategy if available
- `parsed_at`

### 12.3 Hard Reparse Event

Hard reparse event fields should include:

- `document_id`
- `previous_structure_version`
- `new_structure_version`
- `reparse_reason`
- `trigger_source`, such as `internal_debug`, `user_requested`, or `low_parse_score`
- parser mode or parser strategy if available
- `reparsed_at`
- invalidated artifact count
- invalidated content-block count

The successful parse/reparse transaction updates `documents.current_structure_version` and writes the matching parse event together. **[Maintainer-Confirmed]**

### 12.4 Retention Policy

Parse event records should be retained for the lifetime of the document and deleted when the document is deleted. **[Maintainer-Confirmed]**

Retention rules:

- retain parse events while the document exists
- delete parse events when the document is deleted
- do not keep parse events after document deletion as independent audit records
- do not introduce separate archival, TTL, or compliance retention in Phase 1
- keep parse events document-scoped provenance only

This policy reflects that parse events are minimal provenance explaining how accepted structure versions were created, why `structure_version` advanced, and why derived resources may have been cleared. They are not compliance audit logs or a long-term operational history system. **[Maintainer-Confirmed]**

## 13. Validation Semantics

Application reads may validate:

```text
derived.source_structure_version == documents.current_structure_version
```

If a mismatch appears, the row should be treated as stale or invalid defensive data. **[Maintainer-Confirmed]**

This check remains application-level in the first DB design. DB-level triggers or cross-table constraints may be reconsidered later if stale derived rows become a real operational problem. **[Maintainer-Confirmed]**

## 14. Relationship to Existing Planning Documents

This document supersedes any interpretation that current Python-generated `unit_id` should be centered as production DB identity. Existing JSON files remain useful as reference, fixtures, and evaluation material only. **[Maintainer-Confirmed]**

This document does not invalidate Phase 1 StructuredDocument JSONB evaluation planning. It narrows identity and lifecycle assumptions that future DB planning must use before schema design. **[Doc-Confirmed]**

## 15. Non-Goals for First DB Design

1. No production migration of existing JSON identity.
2. No automatic dual-ID system for every hierarchy node.
3. No immutable hierarchy snapshots.
4. No retained retired artifact rows.
5. No retained retired content-block rows.
6. No content-block reuse across hard reparse.
7. No staging hierarchy persistence.
8. No row-level hierarchy versioning.
9. No DB-enforced lifecycle constraint for derived `source_structure_version`.
10. No full audit/history system.
11. No schema/table/column design in this document.
12. No ORM, repository interface, migration, fixture, or runtime behavior implementation in this document.

## 16. Logical DB Schema Proposal (Phase 1)

> This section is a logical persistence model proposal only. It does not define SQL, DDL, physical indexes, database-specific types, ORM models, repository interfaces, migration scripts, fixtures, runtime behavior, or backend selection. **[Maintainer-Confirmed] + [Doc-Confirmed]**

### 16.1 Scope

Phase 1 logical DB planning covers the persistence domains needed to represent the current-state document hierarchy and its directly related derived resources. It follows repository memory rather than current Python file layout details. **[From Proposal] + [From HLD] + [Maintainer-Confirmed]**

Included logical persistence domains:

1. Document and structure lifecycle.
2. Current structured hierarchy.
3. Minimal parse event provenance.
4. Lazy content-block materialization.
5. Artifact persistence and target references.
6. Advisory profile snapshot placement.
7. Raw-source metadata placement.
8. Backend policy placement.

Excluded from Phase 1 logical schema proposal:

1. SQL / DDL / PostgreSQL-specific implementation.
2. ORM models.
3. Repository interfaces.
4. Migration scripts.
5. Runtime read-path switch.
6. Immutable hierarchy history.
7. Staged candidate hierarchy persistence.
8. Production migration of existing JSON identities.
9. Retrieval index physical storage.
10. Raw document binary/blob storage.

### 16.2 Logical Persistence Domains

| Domain | Owner | Authority | Phase 1 Logical Role |
|---|---|---:|---|
| Document identity and lifecycle | `document_structure` for structured semantics; broader upload ownership remains separate | Authoritative for current structure version | Holds document row identity, current `structure_version`, document name/namespace identity, and optional public identity if later required. |
| Current hierarchy | `document_structure` | Authoritative hierarchy truth | Represents current `Document -> Chapter -> Section -> TaskUnit` rows only. |
| Parse events | `db` planning + `document_structure` semantics | Provenance only | Records accepted initial parse / hard reparse events matching document state. |
| Content blocks | `shared` contract + `document_structure` hierarchy link | Derived, not hierarchy authority | Stores lazy materialized content blocks linked to current task-unit rows. |
| Artifacts | `document_structure` persistence semantics + `shared` target contract | Interaction output, not hierarchy authority | Stores summaries/quizzes/interaction outputs and their validated target references. |
| Profile snapshot | `profile` | Advisory only | Candidate DB surface for profile metadata snapshots; not parser authority. |
| Raw-source metadata | raw document ownership remains separate; `db` records metadata boundary only | Source reference metadata only | Tracks file-backed raw source location and descriptive metadata; does not store raw bytes. |
| Backend policy | `config` | Not domain authority | Selects file/DB/coexistence policy later; does not define data meaning. |

Retrieval artifacts and raw document bytes remain separate migration tracks. Raw-source metadata may be tracked in DB, but canonical uploaded raw bytes remain file-backed for the first DB rollout. **[From Proposal] + [From HLD] + [Maintainer-Confirmed]**

### 16.3 Proposed Logical Entities

| Entity | Owner | Responsibility | Authority |
|---|---|---|---|
| `Document` | `document_structure` semantics; raw ownership remains separate | Current document identity, namespace/name isolation, current `structure_version`, and DB-generated document identifier for first DB-backed APIs. Public document identity remains a future extensibility point only. | Authoritative for current structure version. |
| `Chapter` | `document_structure` | Current chapter node under a document. | Hierarchy truth as part of current structure. |
| `Section` | `document_structure` | Current section node under a chapter. | Hierarchy truth as part of current structure. |
| `TaskUnit` | `document_structure` + `shared` DTO contract | Current task-unit interaction container under a section, with compatibility content payload as needed. | Hierarchy interaction container; not a content-block hierarchy parent in the persistence-authority sense. |
| `ParseEvent` | `document_structure` lifecycle semantics | Minimal provenance for accepted initial parse and hard reparse. | Provenance only; not source of current version authority. |
| `ContentBlock` | `shared` content-block contract | Lazy materialized render/interaction segment linked to a task unit. | Derived resource, not hierarchy truth. |
| `Artifact` | `document_structure` artifact persistence semantics + `shared` artifact DTOs | Common persisted interaction output entity with `artifact_type`, target metadata, type-specific payload, and lifecycle/provenance metadata. | Interaction output only. |
| `ArtifactTarget` | `shared.ArtifactTargetRef` semantics | Validated target reference metadata for artifact attachment. | Target metadata; not hierarchy authority. |
| `ProfileSnapshot` | `profile` | Advisory document-scoped profile metadata snapshot, including profile payload, version metadata, source parser/preparation mode, language/script metadata, title/author/source metadata, optional source structure version, and advisory diagnostics when produced. | Advisory only. |
| `RawSourceMetadata` | raw document ownership remains separate | Metadata-only reference to file-backed raw source, including source location/path/object key, original filename, MIME type, file size, checksum/fingerprint when available, upload time, and ownership scope when applicable. | Source reference metadata only; not raw-byte storage. |
| `StorageBackendPolicy` | `config` | Future backend selection / rollout / coexistence policy. | Configuration only, not persistence meaning. |

Entity names are logical labels, not table names. They do not prescribe physical naming, columns, constraints, or implementation types. **[Doc-Confirmed]**

### 16.4 Entity Responsibilities

#### `Document`

Responsibilities:

- Own the DB-generated internal document primary key.
- Preserve document name / namespace isolation currently represented by file paths and storage configs.
- Store authoritative `current_structure_version`.
- Use the DB-generated document identifier for first DB-backed internal APIs and application use.
- Keep separate public document identity as a future extensibility point only if a concrete external stability requirement appears.
- Provide the lifecycle anchor for hard reparse cleanup and parse events.

Non-responsibilities:

- Does not contain raw document bytes by default in this Phase 1 model.
- Does not make profile, retrieval, or artifacts authoritative.
- Does not derive current version from parse events.
- Does not introduce a public document UUID, key, slug, or permanent external identifier in Phase 1.

#### `Chapter`

Responsibilities:

- Represent current accepted chapter hierarchy under one document.
- Preserve ordering and chapter-level display metadata needed for hierarchy parity.
- Link to sections through current-state-only relationships.

Non-responsibilities:

- Does not store row-level `structure_version` in the first design.
- Does not preserve old aliases after hard reparse.
- Does not own artifact payloads directly.

#### `Section`

Responsibilities:

- Represent current accepted section hierarchy under one chapter.
- Preserve ordering and section-level display metadata needed for hierarchy parity.
- Link to task units through current-state-only relationships.

Non-responsibilities:

- Does not reintroduce root `sections[]`.
- Does not serve as a flat primary source independent of chapters.
- Does not store row-level `structure_version` in the first design.

#### `TaskUnit`

Responsibilities:

- Represent the current task-unit interaction container under a section.
- Preserve task-unit order and current task-unit content payload needed for Phase 1 parity.
- Serve as the parent link for lazily materialized content blocks.

Non-responsibilities:

- Does not use current Python-generated `unit_id` as production DB identity.
- Does not create a separate domain ID by default.
- Does not make content blocks a persisted hierarchy level.

#### `ParseEvent`

Responsibilities:

- Record accepted `initial_parse` and `hard_reparse` events.
- Store previous/new structure version values.
- Store trigger source, parser mode/strategy if available, timestamp, and hard-reparse invalidation counts.
- Retain events for the lifetime of the document and delete them when the document is deleted.

Non-responsibilities:

- Does not define current document version.
- Does not retain historical hierarchy snapshots.
- Does not become compliance audit or event sourcing.
- Does not survive document deletion as an independent audit record.
- Does not require separate archival, TTL, or compliance retention in Phase 1.

#### `ContentBlock`

Responsibilities:

- Persist lazy materialized content-block resources separately from `StructuredDocument` JSONB.
- Link internally to the owning document and task-unit DB row.
- Store `source_structure_version`, source hash, block index / deterministic block metadata, segmentation version, span metadata, and schema/version metadata.
- Support application-level stale validation against `Document.current_structure_version`.

Non-responsibilities:

- Does not become hierarchy truth.
- Does not survive hard reparse by default.
- Does not require cross-version reuse in Phase 1.

#### `Artifact`

Responsibilities:

- Persist interaction output payloads in one common logical artifact entity for Phase 1.
- Store document ownership, `artifact_type`, target metadata, type-specific payload, and `source_structure_version`.
- Store source hash / span metadata where applicable, creation time, and other lifecycle/provenance metadata needed by artifact validity checks.
- Support explicit deletion for the whole document on successful hard reparse.
- Preserve payload metadata needed for cache validity and future validation.

Non-responsibilities:

- Does not create, rename, or re-own hierarchy identity.
- Does not become parser authority.
- Does not survive hard reparse by default.
- Does not split into category-specific logical tables such as `summary_artifacts`, `quiz_artifacts`, or `answer_artifacts` in Phase 1.
- Does not prevent later category-specific tables if artifact schemas stabilize, type-specific constraints become important, query patterns require dedicated tables, or generic payload validation becomes insufficient.

#### `ArtifactTarget`

Responsibilities:

- Represent target type and resolved DB target id for artifact attachment.
- Store validation metadata such as `source_structure_version`, source hash, quote span, and schema/version metadata where applicable.
- Allow application logic to classify malformed / unresolved / stale / source-mismatched targets later.

Non-responsibilities:

- Does not allow repository writes without hierarchy-aware validation.
- Does not depend on current Python-generated `unit_id`.
- Does not turn content blocks into hierarchy nodes.

#### `ProfileSnapshot`

Responsibilities:

- Represent the first DB-backed advisory profile persistence track as a document-scoped logical entity linked to `Document`.
- Preserve profile metadata as a snapshot payload, not as normalized hierarchy truth.
- Store minimum logical metadata: `document_id`, profile payload / profile snapshot, `profile_version` or `schema_version`, `generated_at`, source parser / preparation mode when available, language / script metadata when available, title / author / source metadata when available, `source_structure_version` when generated after structure creation, and advisory diagnostics when currently produced.
- Support freshness/provenance metadata conceptually without blocking hierarchy persistence.

Non-responsibilities:

- Does not control parser structure or act as parser authority.
- Does not define hierarchy truth, artifact availability, retrieval authority, or raw document authority.
- Does not create, mutate, rename, or delete chapters, sections, task units, content blocks, or artifacts.
- Does not block `StructuredDocument` hierarchy persistence.
- Does not receive diagnostics profile write-back through hidden read-path mutation.

#### `RawSourceMetadata`

Responsibilities:

- Track metadata for the canonical file-backed raw uploaded document.
- Link raw-source metadata to the owning document.
- Store raw source location / file path / object key, original filename, MIME type, file size, checksum / fingerprint when available, `uploaded_at`, and ownership scope when applicable.
- Keep raw document storage as a separate track from first-rollout DB-backed structured outputs.

Non-responsibilities:

- Does not store raw document bytes in the DB for the first rollout.
- Does not make raw source storage part of the first DB-backed structured-output lifecycle.
- Does not weaken raw file ownership, deletion, retention, or copyright boundaries.
- Does not prevent future object storage or DB-backed raw storage from being evaluated separately.

#### `StorageBackendPolicy`

Responsibilities:

- Represent future file/DB/coexistence rollout policy in configuration.
- Select backend implementations when runtime work is later approved.

Non-responsibilities:

- Does not define schema meaning.
- Does not define hierarchy, profile, retrieval, raw document, or artifact semantics.

### 16.5 Logical Relationships

```text
Document
  -> Chapter
    -> Section
      -> TaskUnit
        -> ContentBlock

Document
  -> ParseEvent

Document
  -> Artifact
    -> ArtifactTarget

Document
  -> ProfileSnapshot

Document
  -> RawSourceMetadata
```

Relationship rules:

1. `Document -> Chapter -> Section -> TaskUnit` is the current hierarchy truth path.
2. `ContentBlock` links to `TaskUnit` but is not a hierarchy level and is not embedded in `StructuredDocument` JSONB by default.
3. `Artifact` belongs to `Document` and may have one or more target references depending on future artifact semantics.
4. `ArtifactTarget` resolves to document/chapter/section/task-unit/content-block targets only after hierarchy-aware validation.
5. `ParseEvent` belongs to `Document` and records provenance for accepted versions, but the document row remains version authority.
6. `ProfileSnapshot` belongs to `Document` and remains advisory.
7. `RawSourceMetadata` belongs to `Document` as metadata-only reference to file-backed raw bytes.
8. Hard reparse is document-scoped: replace current hierarchy, advance document version, delete document content blocks and artifacts, write parse event, commit.

### 16.6 JSONB vs Relational Placement Rationale

| Data | Proposed Placement | Rationale |
|---|---|---|
| Document lifecycle fields | Relational logical entity | `current_structure_version`, document identity, and namespace isolation are frequently referenced lifecycle anchors. |
| Chapter / section / task-unit hierarchy | Relational logical entities | Hierarchy lookup, target validation, ordering, and fail-fast behavior need explicit relationships and current-state replacement semantics. |
| Optional full `StructuredDocument` snapshot | JSONB, validation-only / parity aid | JSONB can preserve model round-trip parity and help debug missing or incorrectly mapped fields, but it must not become hierarchy authority over relational current hierarchy semantics. |
| Task-unit content string / opaque payload | Relational field or payload field under `TaskUnit` | Phase 1 parity requires opaque content preservation; it is task-unit payload, not identity. |
| Content blocks | Relational logical entity with metadata payload as needed | Lazy materialized resources need independent lifecycle, task-unit linkage, version/source metadata, and hard-reparse deletion. |
| Artifact payload | One common `Artifact` logical entity with relational lifecycle/target fields plus type-specific payload | Phase 1 stabilizes the common artifact lifecycle first: document ownership, hierarchy/content-block target reference, `artifact_type`, `source_structure_version`, source hash/span metadata where applicable, creation time, hard-reparse invalidation, and non-authority boundary. Generated payload shape may vary by artifact type, but it should initially remain inside the common artifact entity rather than split into category-specific tables. |
| Artifact target metadata | Relational target fields plus structured metadata payload | Target type and resolved DB id need checkable references; source hash/span/version metadata may remain flexible. |
| Profile snapshot | JSON payload under profile-owned document-scoped entity | Profile metadata is advisory and snapshot-like; relationalizing it early would add little authority and may imply parser control. The first profile track should store `document_id`, profile payload, version metadata, generation/source metadata, optional language/script/title/author/source metadata, optional `source_structure_version`, and advisory diagnostics when produced. |
| Backend policy | Configuration, not domain persistence | `config` owns selection/rollout, not semantic data storage. |
| Retrieval index | Outside Phase 1 logical schema | Retrieval remains derived, partially DB-candidate, and track-separated. |
| Raw-source metadata | Metadata-only DB reference to file-backed raw source | First DB rollout should track `document_id`, raw source location / file path / object key, original filename, MIME type, file size, checksum / fingerprint when available, `uploaded_at`, and ownership scope when applicable. |
| Raw document file/blob | File-backed storage outside first DB rollout | Raw uploaded documents are canonical user-owned source files with larger storage, deletion, retention, and copyright boundaries. Moving raw bytes into DB would expand the migration scope too early; future object storage or DB-backed raw storage can be considered separately. |

The placement rule is: relationalize identity, ownership, ordering, lifecycle, and target validation; keep variable advisory/generated payloads flexible where they are not authority. **[Inferred]**

Phase 1 may keep an optional full `StructuredDocument` JSONB snapshot alongside the relational current hierarchy only as a validation/parity artifact. The relational current hierarchy remains the operational read/write model, `Document.current_structure_version` remains authoritative, and the snapshot is not parser authority, runtime hierarchy truth, or a fallback read path. If relational rows and the JSONB snapshot conflict, the conflict is a validation failure, not dual-authority behavior. **[Maintainer-Confirmed]**

### 16.7 Ownership Boundaries

1. `db/` owns this logical schema proposal as documentation memory. It does not own runtime implementation.
2. `document_structure` owns hierarchy semantics, current-state hierarchy persistence meaning, parse lifecycle semantics, and artifact persistence semantics against validated hierarchy targets.
3. `shared` owns DTO vocabulary for `TaskUnit`, `TaskUnitContentBlock`, `ArtifactTargetLevel`, `ArtifactTargetRef`, and artifact payload shapes.
4. `config` owns future backend selection, rollout flags, and DI policy, not domain semantics.
5. `document_preparation` owns preparation lifecycle and should eventually write through storage policy without assuming file paths as the only destination.
6. `profile` owns advisory profile snapshot semantics as a Phase 1 logical entity; it remains document-scoped and non-authoritative.
7. Raw document bytes remain file-backed for the first DB rollout; only raw-source metadata belongs in the Phase 1 logical DB proposal.
8. `retrieval` and raw document byte storage remain separate tracks and should not be silently pulled into Phase 1 hierarchy schema.

### 16.8 Open Design Questions

No unresolved Phase 1 logical schema confirmation items remain. **[Maintainer-Confirmed]**

### 16.9 Phase 1 Logical Schema Non-Goals

1. No SQL or DDL.
2. No physical table names beyond logical entity labels.
3. No column list.
4. No indexes or constraints.
5. No ORM model design.
6. No repository interface design.
7. No migration script.
8. No fixtures.
9. No runtime behavior change.
10. No backend selection.
11. No public API contract change.
12. No category-specific artifact tables in Phase 1.
13. No DB storage of raw document bytes in the first rollout.

## 17. Current Risks

1. risk: future DB implementation may accidentally introduce a second identity system for every hierarchy node.
- why: domain IDs add generation, mapping, uniqueness, and lifecycle costs.
- guardrail: use DB-generated IDs by default; add public/domain IDs only for concrete external stability requirements.

2. risk: schema design may preserve current `unit_id` as production identity.
- why: existing JSON fixtures expose `unit_id`, but they are not production data.
- guardrail: treat current `unit_id` as reference/import evidence only.

3. risk: hard reparse cleanup may be hidden behind cascade behavior.
- why: invalidating derived rows is a domain lifecycle decision.
- guardrail: keep explicit service-layer cleanup and parse-event counts.

4. risk: parse events may drift into event sourcing or history.
- why: version provenance can be mistaken for authoritative state.
- guardrail: keep `documents.current_structure_version` authoritative.

5. risk: logical schema proposal may be mistaken for implementation approval.
- why: entity labels can look like table design.
- guardrail: keep this section explicitly logical and require a separate implementation task before SQL, ORM, repository, migration, or runtime work.

## 18. Open Questions for Maintainer

Open questions for Phase 1 logical schema are listed in section 16.8. No additional unresolved confirmation items are identified for the previously captured DB-era identity and hard-reparse policy.

## 19. Suggested Next Documentation Improvements

1. When schema work is explicitly requested, derive an implementation checklist from this document before proposing tables.
2. Before any DB runtime work, reconcile this document with `document_structure`, `config`, `document_preparation`, `shared`, and `section_tasks` module checklists.
3. If a public API stability requirement appears, add a targeted public/domain identity decision record before implementation.
