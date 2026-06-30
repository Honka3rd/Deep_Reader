# Phase 1 DB Schema Design

## 1. Purpose

This document converts the confirmed Phase 1 logical DB schema direction into an implementation-ready schema design reference.

It is documentation-only. It defines logical tables/entities, candidate fields, relationships, ownership boundaries, and constraint candidates. It does not define SQL DDL, database-specific types, physical indexes, ORM models, repository interfaces, migrations, runtime read/write behavior, fixtures, tests, API changes, or backend selection.

## 2. Review of Current Direction

`db/module-detailed-design.md` already captures the confirmed Phase 1 logical schema decisions:

- DB-generated primary keys are the default internal identity.
- Current Python-generated `unit_id` is reference/import evidence only.
- `documents.current_structure_version` is authoritative.
- First successful parse creates `structure_version = 1`.
- Early DB hierarchy persistence is current-state-only.
- Hierarchy rows do not carry row-level `structure_version`.
- No immutable hierarchy history or staging hierarchy persistence is introduced.
- Content blocks are lazy materialized derived resources persisted separately.
- Artifacts use one common logical entity with `artifact_type` and type-specific payload.
- Profile is included as an advisory document-scoped snapshot.
- Raw source bytes remain file-backed/object-backed; DB tracks metadata only.
- Parse events are minimal document-scoped provenance, not event sourcing.
- Hard reparse deletes document-level content blocks and artifacts.
- Derived rows use `source_structure_version` for application-level defensive validation.
- Optional `StructuredDocument` JSONB snapshot is validation/parity/debug only.

No confirmed Phase 1 logical schema decision is missing. The main clarification this document adds is candidate field grouping and relationship/constraint candidates for later implementation planning.

## 3. Schema Principles

1. Use DB-generated primary keys by default.
2. Do not production-depend on current Python-generated `unit_id`.
3. Do not automatically create public/domain IDs for every hierarchy node.
4. Treat `documents.current_structure_version` as the authoritative current hierarchy version.
5. Create `structure_version = 1` on initial successful parse.
6. Persist only current accepted hierarchy rows in Phase 1.
7. Do not store row-level hierarchy `structure_version` on chapters, sections, or task units.
8. Do not preserve immutable hierarchy history or staging hierarchy persistence.
9. Persist content blocks separately and lazily after materialization.
10. Persist artifacts through one logical `artifacts` entity with `artifact_type` and type-specific payload.
11. Persist `document_profile` as advisory document-scoped snapshot metadata.
12. Keep raw document bytes file-backed/object-backed; store only raw-source metadata in DB.
13. Keep parse events minimal provenance only.
14. Treat derived `source_structure_version` checks as application-level validation.
15. Keep DB schema as representation, not parser authority or hierarchy authority.

## 4. Logical Tables

Table names in this document are candidate logical names. They are not physical table names or SQL DDL.

### 4.1 `documents`

Purpose:

- Anchor document identity and current hierarchy lifecycle.
- Own the authoritative current `structure_version`.
- Link all document-scoped derived resources.

Candidate fields:

| Field | Purpose |
|---|---|
| `id` | DB-generated internal primary key. |
| `document_name` | Current application document name or namespace-facing identifier. |
| `namespace` | Optional storage namespace / isolation key when needed. |
| `current_structure_version` | Authoritative current hierarchy version. |
| `created_at` | Document record creation time. |
| `updated_at` | Document record update time. |
| `status` | Optional lifecycle status such as active/deleted if soft deletion is later chosen. |
| `metadata_payload` | Optional flexible metadata for non-authoritative operational details. |

Constraint candidates:

- `id` is the internal relational link foundation.
- `current_structure_version` starts at `1` after initial successful parse.
- `current_structure_version` advances only after successful document-level hard reparse transaction.
- Namespace/name uniqueness may be needed if runtime document lookup still depends on name and namespace.

Non-goals:

- No public UUID/key/slug in Phase 1.
- No raw document bytes.
- No hierarchy JSON as the authority.

### 4.2 `raw_source_metadata`

Purpose:

- Track metadata for the canonical file-backed/object-backed raw source.
- Keep raw byte storage outside the first DB rollout.

Candidate fields:

| Field | Purpose |
|---|---|
| `id` | DB-generated internal primary key. |
| `document_id` | Owning document reference. |
| `source_location` | File path, object key, or equivalent raw source location. |
| `original_filename` | User-provided source filename when available. |
| `mime_type` | Source MIME type when available. |
| `file_size` | Source size metadata. |
| `checksum` | Source checksum / fingerprint when available. |
| `uploaded_at` | Upload/import time. |
| `ownership_scope` | User/tenant/scope metadata if available. |
| `metadata_payload` | Optional non-authoritative source metadata. |

Constraint candidates:

- `document_id` links raw-source metadata to one document.
- Raw-source metadata is metadata only; raw bytes remain outside DB.
- Deletion/retention must preserve user-owned source boundaries.

Non-goals:

- No DB blob/raw-byte storage.
- No cross-user sharing implication.
- No object-storage migration decision.

### 4.3 `document_profile`

Purpose:

- Store an advisory document-scoped profile snapshot.
- Preserve preparation/profile output without making it hierarchy truth.

Candidate fields:

| Field | Purpose |
|---|---|
| `id` | DB-generated internal primary key. |
| `document_id` | Owning document reference. |
| `profile_payload` | Full advisory profile snapshot payload. |
| `profile_version` / `schema_version` | Profile payload version metadata. |
| `generated_at` | Profile generation timestamp. |
| `source_parser` | Source parser when available. |
| `preparation_mode` | Preparation mode when available. |
| `language_metadata` | Language/script metadata when available. |
| `title_metadata` | Title/source/title-like metadata when available. |
| `author_metadata` | Author metadata when available. |
| `source_metadata` | Source metadata when available. |
| `source_structure_version` | Structure version if generated after structure creation. |
| `advisory_diagnostics` | Advisory diagnostics currently produced. |

Constraint candidates:

- Profile belongs to a document.
- Profile is advisory-only.
- `source_structure_version` may be compared to `documents.current_structure_version` by application logic when relevant.

Non-goals:

- No parser authority.
- No hierarchy authority.
- No artifact availability authority.
- No diagnostics write-back through read paths.

### 4.4 `chapters`

Purpose:

- Represent current accepted chapter hierarchy under a document.

Candidate fields:

| Field | Purpose |
|---|---|
| `id` | DB-generated internal primary key. |
| `document_id` | Owning document reference. |
| `chapter_order` | Current chapter order within document. |
| `title` | Display title. |
| `source_anchor` | Optional source/span/page anchor metadata. |
| `metadata_payload` | Optional non-authoritative chapter metadata. |

Constraint candidates:

- Chapter rows are current-state only.
- Ordering is scoped to document.
- Hard reparse replaces current hierarchy rows rather than preserving old rows.

Non-goals:

- No row-level `structure_version`.
- No old aliases.
- No `Part -> Chapter` persisted layer in Phase 1.

### 4.5 `sections`

Purpose:

- Represent current accepted section hierarchy under a chapter.

Candidate fields:

| Field | Purpose |
|---|---|
| `id` | DB-generated internal primary key. |
| `document_id` | Owning document reference for cleanup/query scope. |
| `chapter_id` | Parent chapter reference. |
| `section_order` | Current section order within chapter. |
| `title` | Display title. |
| `source_anchor` | Optional source/span/page anchor metadata. |
| `metadata_payload` | Optional non-authoritative section metadata. |

Constraint candidates:

- Section rows are current-state only.
- Section parent must be a current chapter in the same document.
- Ordering is scoped to chapter.

Non-goals:

- No root `sections[]` as primary source.
- No row-level `structure_version`.
- No flat section authority independent of chapters.

### 4.6 `task_units`

Purpose:

- Represent current task-unit interaction containers under sections.
- Preserve opaque current task-unit content payload needed for Phase 1 parity.

Candidate fields:

| Field | Purpose |
|---|---|
| `id` | DB-generated internal primary key. |
| `document_id` | Owning document reference for cleanup/query scope. |
| `section_id` | Parent section reference. |
| `task_unit_order` | Current order within section. |
| `content_payload` | Opaque task-unit content payload/string for parity. |
| `reference_unit_id` | Optional import/reference identity from current JSON `unit_id`; not production identity. |
| `metadata_payload` | Optional non-authoritative task-unit metadata. |

Constraint candidates:

- Task-unit rows are current-state only.
- Parent section must belong to the same document.
- Ordering is scoped to section.
- `reference_unit_id` must not become the production identity foundation.

Non-goals:

- No flat `task_units` primary runtime truth.
- No row-level `structure_version`.
- No automatic public/domain task-unit id.
- No content-block hierarchy level.

### 4.7 `content_blocks`

Purpose:

- Store lazy materialized content blocks linked to task units.
- Support finer-grained render/interaction targets without becoming hierarchy.

Candidate fields:

| Field | Purpose |
|---|---|
| `id` | DB-generated internal primary key. |
| `document_id` | Owning document reference. |
| `task_unit_id` | Parent task-unit DB reference. |
| `block_order` | Deterministic block order within task unit. |
| `content` | Block text/content payload. |
| `source_structure_version` | Structure version used when block was generated. |
| `source_hash` | Source text hash / validation metadata when available. |
| `segmentation_version` | Segmentation/schema version metadata. |
| `quote_span_start` | Optional span start metadata. |
| `quote_span_end` | Optional span end metadata. |
| `metadata_payload` | Optional flexible block metadata. |
| `created_at` | Materialization timestamp. |

Constraint candidates:

- `source_structure_version` is application-level defensive validation metadata.
- Reads may treat rows as stale if `source_structure_version != documents.current_structure_version`.
- Successful hard reparse physically deletes all content blocks for the document.

Non-goals:

- No content block as hierarchy truth.
- No embedding inside `StructuredDocument` JSONB as default persistence.
- No cross-reparse reuse requirement in Phase 1.

### 4.8 `artifacts`

Purpose:

- Store interaction output artifacts through one common logical entity.
- Attach artifacts to validated document/chapter/section/task-unit/content-block targets.

Candidate fields:

| Field | Purpose |
|---|---|
| `id` | DB-generated internal primary key. |
| `document_id` | Owning document reference. |
| `artifact_type` | Artifact category such as summary, quiz, answer, note, evidence. |
| `target_type` | Target kind: document/chapter/section/task_unit/content_block. |
| `target_id` | Resolved DB target id for the target type. |
| `source_structure_version` | Structure version used when artifact was produced. |
| `source_hash` | Optional source validation hash. |
| `quote_span_start` | Optional quote/evidence span start. |
| `quote_span_end` | Optional quote/evidence span end. |
| `payload` | Type-specific artifact payload. |
| `schema_version` | Artifact payload schema/version metadata. |
| `created_at` | Artifact creation time. |
| `updated_at` | Artifact update time if mutable artifacts are later allowed. |
| `metadata_payload` | Optional lifecycle/cache/validity metadata. |

Constraint candidates:

- Artifact target must be resolved through hierarchy-aware validation before write.
- `source_structure_version` supports application-level stale validation.
- Successful hard reparse physically deletes all artifacts for the document.
- `artifact_type` plus `payload` handles type-specific shape in Phase 1.

Non-goals:

- No category-specific artifact tables in Phase 1.
- No artifact hierarchy authority.
- No parser authority.
- No artifact survival across hard reparse by default.

### 4.9 `parse_events`

Purpose:

- Store minimal document-scoped provenance for accepted initial parse and hard reparse.

Candidate fields:

| Field | Purpose |
|---|---|
| `id` | DB-generated internal primary key. |
| `document_id` | Owning document reference. |
| `event_type` | `initial_parse` or `hard_reparse`. |
| `previous_structure_version` | Previous structure version, null for initial parse. |
| `new_structure_version` | Accepted structure version after event. |
| `trigger_source` | Upload/prepare/user/internal/low-score trigger metadata. |
| `parser_mode` | Parser mode/strategy when available. |
| `reparse_reason` | Hard reparse reason when applicable. |
| `invalidated_artifact_count` | Hard reparse invalidated artifact count. |
| `invalidated_content_block_count` | Hard reparse invalidated content-block count. |
| `occurred_at` | Event timestamp. |
| `metadata_payload` | Optional provenance metadata. |

Constraint candidates:

- Events are document-scoped provenance only.
- Events are retained for the lifetime of the document and deleted when the document is deleted.
- Parse events do not define current version authority.
- Initial parse creates `new_structure_version = 1`.

Non-goals:

- No event sourcing.
- No compliance audit log.
- No immutable hierarchy history.
- No retention after document deletion in Phase 1.

### 4.10 `structured_document_snapshots`

Purpose:

- Optionally store full `StructuredDocument` JSONB-like snapshot as validation/parity/debug artifact only.
- Help compare relational current hierarchy mapping against model round-trip behavior.

Candidate fields:

| Field | Purpose |
|---|---|
| `id` | DB-generated internal primary key. |
| `document_id` | Owning document reference. |
| `source_structure_version` | Structure version represented by the snapshot. |
| `structured_document_payload` | Full structured document payload for validation/parity/debug. |
| `schema_version` | Snapshot schema/version metadata. |
| `created_at` | Snapshot creation time. |
| `metadata_payload` | Optional validation metadata. |

Constraint candidates:

- Snapshot is optional.
- Snapshot must not become hierarchy authority.
- Snapshot conflicts with relational current hierarchy are validation failures, not fallback behavior.

Non-goals:

- No runtime hierarchy source.
- No parser authority.
- No fallback read path.
- No replacement for relational current hierarchy.

## 5. Relationship Summary

```text
documents
  -> raw_source_metadata
  -> document_profile
  -> chapters
    -> sections
      -> task_units
        -> content_blocks
  -> artifacts
  -> parse_events
  -> structured_document_snapshots
```

Relationship rules:

- `documents -> chapters -> sections -> task_units` is the authoritative current hierarchy path.
- `content_blocks` links to `task_units` and remains derived.
- `artifacts` links to validated targets and remains interaction output.
- `document_profile` belongs to `documents` and remains advisory.
- `raw_source_metadata` belongs to `documents`; raw bytes remain file-backed/object-backed.
- `parse_events` belongs to `documents` and remains provenance only.
- `structured_document_snapshots` belongs to `documents` and remains validation/parity/debug only.

## 6. Hard Reparse Lifecycle

Candidate hard reparse transaction semantics:

1. Build and validate candidate hierarchy outside durable current hierarchy tables.
2. Start one transaction after validation succeeds.
3. Replace current chapter/section/task-unit rows for the document.
4. Advance `documents.current_structure_version`.
5. Delete all `content_blocks` for the document.
6. Delete all `artifacts` for the document.
7. Write one `parse_events` record.
8. Optionally write a validation/parity `structured_document_snapshots` row.
9. Commit.

The lifecycle must be explicit at the service layer. Defensive database cascades may prevent orphans later, but cascades must not be the primary domain definition of hard reparse.

## 7. Application-Level Validation

Derived resources may be treated as stale when:

```text
derived.source_structure_version != documents.current_structure_version
```

This applies to:

- `content_blocks`
- `artifacts`
- `document_profile` when `source_structure_version` is present
- `structured_document_snapshots` when used for validation/parity checks

This is application-level defensive validation in Phase 1. It is not a DB-enforced cross-table lifecycle constraint in this design.

## 8. Ownership Boundaries

| Area | Owner | Boundary |
|---|---|---|
| Hierarchy semantics | `document_structure` | Owns `documents`, `chapters`, `sections`, `task_units` semantics. |
| Artifact persistence semantics | `document_structure` + `shared` target vocabulary | Owns artifact lifecycle against validated targets. |
| Content-block vocabulary | `shared` | Defines content block DTO vocabulary; DB persists derived blocks only. |
| Profile snapshot semantics | `profile` | Owns advisory profile payload semantics. |
| Backend selection | `config` | Future backend policy only, not domain meaning. |
| Preparation lifecycle | `document_preparation` | Produces structured/profile/retrieval readiness; later writes through storage policy. |
| Raw source bytes | raw upload/loading track | Remains file-backed/object-backed outside first DB rollout. |
| Retrieval | `retrieval` | Separate derived migration track outside this Phase 1 schema design. |

## 9. Governance Validation

This design does not introduce:

- root `sections[]`
- `structure_nodes`
- flat `task_units` as primary runtime truth
- profile authority
- artifact hierarchy authority
- raw-byte DB storage
- category-specific artifact tables in Phase 1
- DB schema as parser authority
- runtime read/write switch
- SQL DDL
- ORM
- repository interface
- migration script

## 10. Open Questions

No unresolved Phase 1 logical-schema confirmation items remain.

Implementation-stage questions remain outside this document:

- physical table naming
- physical type selection
- index strategy
- transaction implementation details
- repository interface design
- migration tooling
- rollout configuration
- validation fixtures

Those require separate future tasks and must not be inferred from this documentation-only design.
