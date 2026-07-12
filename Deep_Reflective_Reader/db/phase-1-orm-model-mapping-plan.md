# Phase 1 ORM Model Mapping Plan

## 1. Purpose

This document defines the Phase 1 ORM/model mapping plan for DB-backed persistence entities.

It is documentation-only. It does not create ORM classes, SQL DDL, migration scripts, repository interfaces, runtime read/write behavior, fixtures, tests, API changes, or backend selection.

The goal is to plan how future ORM models should map to physical table candidates without making ORM classes parser authority, hierarchy authority, artifact authority, profile authority, or API/domain DTO authority.

## 2. Mapping Principles

1. ORM models are persistence records, not domain models.
2. `document_structure` remains the owner of hierarchy semantics.
3. Parser output remains validated before persistence; ORM classes must not parse, classify, repair, or infer hierarchy.
4. DB-generated primary keys remain the default internal persistence identity.
5. Existing Python-generated `unit_id` remains optional import/reference evidence only.
6. ORM relationships may support loading, but they must not define parser truth.
7. ORM cascades may prevent orphans, but hard reparse cleanup remains an explicit service-layer lifecycle operation.
8. `source_structure_version` freshness remains application-level validation.
9. JSON/JSONB payload fields remain flexible payload carriers, not authority boundaries.
10. ORM models must not introduce runtime read/write switching by themselves.
11. ORM models may map timestamp fields, but update timestamp ownership remains application/repository-managed; ORM events or database triggers must not hide lifecycle mutation.

## 3. Candidate Model Set

| ORM Model Candidate | Table Candidate | Role | Authority Boundary |
|---|---|---|---|
| `DocumentRecord` | `documents` | Document lifecycle and current structure version persistence. | Authoritative only for DB document row identity and `current_structure_version`; not parser authority. |
| `RawSourceMetadataRecord` | `raw_source_metadata` | Metadata-only raw source reference. | Does not store raw bytes or extracted raw text or define raw storage policy. |
| `DocumentProfileRecord` | `document_profile` | Advisory document-scoped profile snapshot. | Profile metadata only; not hierarchy, parser, retrieval, or artifact authority. |
| `ChapterRecord` | `chapters` | Current chapter row. | Persistence representation of current hierarchy after validation. |
| `SectionRecord` | `sections` | Current section row. | Persistence representation under a chapter; no root section authority. |
| `TaskUnitRecord` | `task_units` | Current task-unit row under a section. | Interaction container record; not flat task-unit truth. |
| `ContentBlockRecord` | `content_blocks` | Lazy materialized content block record. | Derived content resource; not persisted hierarchy level. |
| `ArtifactRecord` | `artifacts` | Common artifact record with type-specific payload. | Interaction output only; not hierarchy or parser authority. |
| `ParseEventRecord` | `parse_events` | Minimal parse provenance record. | Provenance only; not event sourcing. |
| `StructuredDocumentSnapshotRecord` | `structured_document_snapshots` | Optional validation/parity/debug snapshot. | Not runtime hierarchy source or fallback. |

Model names are candidates only. Future implementation may choose different names if the boundaries remain intact.

## 4. Relationship Mapping Candidates

Future ORM relationships may mirror ownership for loading and cleanup:

```text
DocumentRecord
  -> RawSourceMetadataRecord
  -> DocumentProfileRecord
  -> ParseEventRecord
  -> StructuredDocumentSnapshotRecord
  -> ChapterRecord
    -> SectionRecord
      -> TaskUnitRecord
        -> ContentBlockRecord
  -> ArtifactRecord
```

Relationship rules:

- Use ORM relationships for persistence navigation, not parser construction.
- Keep the authoritative hierarchy path `DocumentRecord -> ChapterRecord -> SectionRecord -> TaskUnitRecord`.
- Do not add root `sections` or root `task_units` model relationships as primary runtime truth.
- `ArtifactRecord.target_type` and `ArtifactRecord.target_id` remain polymorphic target metadata; ORM should not fake a single hard FK relationship to every target table.
- `DocumentProfileRecord` should not have relationships that mutate hierarchy rows.
- `StructuredDocumentSnapshotRecord` should remain optional and validation-only.

## 5. Model Responsibility Boundaries

### 5.1 `DocumentRecord`

Responsibilities:

- Map DB document row identity.
- Store required `namespace`, required `document_name`, and `current_structure_version`.
- Map `updated_at` as an application-managed field: insert defaults may initialize it, but update operations must explicitly set it.
- Own ORM relationships to document-scoped records.

Non-responsibilities:

- Does not parse raw input.
- Does not store raw text or raw bytes in the document row.
- Does not derive hierarchy from profile, artifacts, or snapshots.
- Does not create public document identity in Phase 1.

### 5.2 `ChapterRecord`, `SectionRecord`, `TaskUnitRecord`

Responsibilities:

- Represent already-validated current hierarchy rows.
- Preserve parent-child persistence links and ordering.
- Carry non-authoritative metadata payloads when needed.

Non-responsibilities:

- Do not run parser logic.
- Do not store row-level hierarchy `structure_version`.
- Do not preserve immutable hierarchy history.
- Do not introduce `structure_nodes`.
- Do not use `TaskUnitRecord.reference_unit_id` as production identity.

### 5.3 `ContentBlockRecord`

Responsibilities:

- Represent lazy materialized derived content blocks.
- Link to owning document and task-unit row.
- Store `source_structure_version` and materialization metadata.

Non-responsibilities:

- Does not become a hierarchy node.
- Does not survive hard reparse by default.
- Does not enforce freshness through ORM magic.

### 5.4 `ArtifactRecord`

Responsibilities:

- Represent one common artifact persistence surface.
- Store `artifact_type`, target metadata, type-specific payload, and provenance metadata.
- Keep `updated_at` nullable until a mutable artifact update path explicitly sets it.
- Support application-level stale validation through `source_structure_version`.

Non-responsibilities:

- Does not create category-specific ORM model families for Phase 1.
- Does not infer hierarchy from targets.
- Does not bypass hierarchy-aware target validation before write.

### 5.5 `DocumentProfileRecord`

Responsibilities:

- Represent advisory profile snapshot payload and metadata.
- Link to the owning document.
- Optionally store `source_structure_version` when generated after structure creation.

Non-responsibilities:

- Does not mutate chapters, sections, task units, content blocks, or artifacts.
- Does not act as parser authority.
- Does not block hierarchy persistence.
- Does not receive diagnostics write-back through read paths.

### 5.6 `RawSourceMetadataRecord`

Responsibilities:

- Represent raw source metadata linked to a document.
- Store source location, filename, MIME type, size, checksum, upload time, and scope metadata when available.

Non-responsibilities:

- Does not store raw bytes or extracted raw text.
- Does not define object storage behavior.
- Does not imply cross-user source sharing.

### 5.7 `ParseEventRecord`

Responsibilities:

- Represent accepted initial parse and hard reparse provenance.
- Store previous/new structure version, trigger/source metadata, timestamps, and invalidation counts.

Non-responsibilities:

- Does not define current document version.
- Does not event-source hierarchy.
- Does not survive document deletion as independent audit data.

### 5.8 `StructuredDocumentSnapshotRecord`

Responsibilities:

- Represent optional validation/parity/debug payload.
- Help compare relational mapping against `StructuredDocument` round-trip behavior.

Non-responsibilities:

- Does not become runtime read path.
- Does not become hierarchy fallback.
- Does not override relational current hierarchy.

## 6. DTO and Domain Conversion Boundary

Future implementation should keep explicit conversion functions or repository-layer mappers between ORM records and domain/DTO objects.

Allowed:

- ORM record -> persistence DTO for repository reads.
- Validated domain structure -> ORM records for persistence writes.
- ORM hierarchy rows -> `document_structure` read model through an explicit mapper.

Not allowed:

- Parser returns ORM records directly.
- API returns ORM records directly.
- ORM constructors perform parser classification.
- ORM relationships define artifact availability or parser quality.
- ORM model methods mutate profile diagnostics or task layout projections as hidden side effects.

## 7. Session and Loading Policy Candidates

Future ORM usage should prefer explicit loading boundaries:

- Load document lifecycle and current hierarchy through repository methods.
- Use relationship loading only where it matches a known read use case.
- Avoid implicit lazy loading in parser, API schema serialization, and task-layout projection paths.
- Avoid leaking ORM session-bound objects across service boundaries.
- Convert to detached DTO/read models before returning from repository boundaries.

These are candidates only; final session policy belongs to repository/storage interface design.

## 8. Cascade and Lifecycle Policy Candidates

ORM cascade configuration may be used defensively for document deletion and orphan prevention.

It must not replace explicit lifecycle behavior:

- Hard reparse service must explicitly replace current hierarchy.
- Hard reparse service must explicitly delete all document content blocks.
- Hard reparse service must explicitly delete all document artifacts.
- Hard reparse service must write parse event provenance.

ORM cascades should not hide hard reparse invalidation counts or make cleanup invisible to application logic.

## 9. Validation Responsibilities

Validation that should happen before ORM write:

- Candidate hierarchy shape is valid.
- Chapter, section, and task-unit ordering is valid.
- Artifact target type/id is hierarchy-aware and resolved.
- Profile payload is advisory and document-scoped.
- Raw-source metadata does not contain raw bytes or extracted raw text.

Validation that remains application-level after read:

- `source_structure_version` freshness for content blocks, artifacts, profile snapshot, and optional snapshots.
- Stale artifact/content-block handling.
- JSONB parity snapshot conflict detection.

Validation not assigned to ORM:

- Parser quality classification.
- Enhanced parse recommendation.
- Task-layout diagnostics projection.
- Retrieval authority.

## 10. Governance Validation

This mapping plan does not introduce:

- ORM classes as parser authority
- ORM classes as hierarchy authority
- executable ORM model code
- executable SQL DDL
- migration scripts
- repository interfaces
- runtime read/write switch
- root `sections[]`
- `structure_nodes`
- flat `task_units` as primary runtime truth
- profile authority
- artifact hierarchy authority
- raw-byte DB storage
- category-specific artifact ORM tables in Phase 1
- DB schema as parser authority
- row-level hierarchy `structure_version`
- immutable hierarchy history
- staging hierarchy persistence

## 11. Completion Boundary

The checklist item "Define ORM/model mapping plan for Phase 1 entities without making ORM classes parser authority" is complete when this document is present and linked from `db/module-checklist.md`.

Remaining separate tasks include repository/storage interfaces, document-structure integration, raw-source metadata persistence behavior, profile persistence behavior, fixtures, transaction-level hard reparse implementation planning, backend configuration integration, runtime switch planning, and executable migration work.
