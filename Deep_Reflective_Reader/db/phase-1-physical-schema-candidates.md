# Phase 1 Physical Schema Candidates

## 1. Purpose

This document defines physical table, column, foreign-key, uniqueness, and index candidates for the Phase 1 DB design.

It is documentation-only. It does not create executable SQL DDL, migration scripts, ORM models, repository interfaces, runtime read/write behavior, fixtures, tests, API changes, or backend selection.

The goal is to make the next migration-design step reviewable before implementation.

## 2. Type Candidate Policy

Type names are candidates, not committed database-specific DDL.

Suggested type vocabulary:

| Candidate Type | Meaning |
|---|---|
| `bigint identity` | DB-generated internal primary key candidate. |
| `text` | Unbounded string candidate. |
| `integer` | Numeric ordering/version/count candidate. |
| `timestamp` | Timestamp candidate; timezone policy remains implementation-stage. |
| `json/jsonb` | Flexible payload candidate; JSONB remains backend-specific and must be confirmed before executable DDL. |
| `boolean` | Boolean flag candidate if later needed. |

Phase 1 should use DB-generated primary keys by default and should not introduce public/domain IDs unless a concrete external stability requirement appears.

## 3. Table Candidates

### 3.1 `documents`

Purpose:

- Anchor document identity.
- Own authoritative `current_structure_version`.
- Scope hierarchy, metadata, parse events, content blocks, and artifacts.

| Column | Type Candidate | Required | Notes |
|---|---|---:|---|
| `id` | `bigint identity` | yes | Internal primary key. |
| `document_name` | `text` | yes | Current application document name or namespace-facing identifier. |
| `namespace` | `text` | yes | Required storage namespace / isolation key for user/tenant-ready document isolation. |
| `current_structure_version` | `integer` | yes | Authoritative current hierarchy version; starts at `1` after initial parse. |
| `status` | `text` | no | Optional lifecycle status candidate. |
| `metadata_payload` | `json/jsonb` | no | Non-authoritative operational metadata. |
| `created_at` | `timestamp` | yes | Record creation time. |
| `updated_at` | `timestamp` | yes | Record update time. |

Foreign-key candidates:

- none.

Uniqueness candidates:

- `unique(namespace, document_name)` is required for Phase 1 namespace/document isolation.
- Do not use global `unique(document_name)`; the same document name must be allowed in different namespaces.

Index candidates:

- lookup index on `(namespace, document_name)`.
- optional index on `status` if soft-deletion or lifecycle filtering is implemented.

Delete behavior candidates:

- document delete removes document-scoped DB rows.
- raw bytes and raw text remain governed by raw file/object storage policy and must not be stored in `documents`.

### 3.2 `raw_source_metadata`

Purpose:

- Track metadata-only references to file-backed/object-backed raw sources.
- Keep raw bytes and extracted raw text outside the DB.

| Column | Type Candidate | Required | Notes |
|---|---|---:|---|
| `id` | `bigint identity` | yes | Internal primary key. |
| `document_id` | `bigint` | yes | Owning document. |
| `source_location` | `text` | yes | File path, object key, or equivalent source location. |
| `original_filename` | `text` | no | User-provided filename when available. |
| `mime_type` | `text` | no | MIME type when available. |
| `file_size` | `integer` | no | File size candidate; widen later if needed. |
| `checksum` | `text` | no | Source checksum/fingerprint. |
| `uploaded_at` | `timestamp` | yes | Upload/import timestamp. |
| `ownership_scope` | `text` | no | User/tenant/scope metadata candidate; SQLite validation slice may omit until user table exists. |
| `metadata_payload` | `json/jsonb` | no | Non-authoritative source metadata. |

Foreign-key candidates:

- `raw_source_metadata.document_id -> documents.id`.

Uniqueness candidates:

- `unique(document_id)` for the Phase 1 core validation slice: one canonical raw-source metadata row per document.
- `unique(document_id, source_location)` may replace this only if multiple raw-source records are deliberately retained later.

Index candidates:

- index on `document_id`.
- optional index on `checksum` if duplicate-source detection becomes a query path.

Delete behavior candidates:

- delete with owning document.
- do not imply raw-byte or extracted-text deletion unless raw storage policy explicitly requires it.

### 3.3 `document_profile`

Purpose:

- Store advisory document-scoped profile snapshot metadata.
- Preserve profile output without hierarchy authority.

| Column | Type Candidate | Required | Notes |
|---|---|---:|---|
| `id` | `bigint identity` | yes | Internal primary key. |
| `document_id` | `bigint` | yes | Owning document. |
| `profile_payload` | `json/jsonb` | yes | Advisory profile snapshot payload. |
| `profile_version` | `text` | no | Profile version candidate. |
| `schema_version` | `text` | no | Payload schema version candidate. |
| `generated_at` | `timestamp` | yes | Profile generation timestamp. |
| `source_parser` | `text` | no | Parser source when available. |
| `preparation_mode` | `text` | no | Preparation mode when available. |
| `language_metadata` | `json/jsonb` | no | Language/script metadata. |
| `title_metadata` | `json/jsonb` | no | Title/source/title-like metadata. |
| `author_metadata` | `json/jsonb` | no | Author metadata. |
| `source_metadata` | `json/jsonb` | no | Source metadata. |
| `source_structure_version` | `integer` | no | Structure version if profile is generated after structure creation. |
| `advisory_diagnostics` | `json/jsonb` | no | Advisory diagnostics currently produced. |

Foreign-key candidates:

- `document_profile.document_id -> documents.id`.

Uniqueness candidates:

- `unique(document_id)` if Phase 1 keeps one current profile snapshot.

Index candidates:

- index on `document_id`.
- optional index on `(document_id, source_structure_version)` for freshness checks.

Delete behavior candidates:

- delete with owning document.
- profile rows must not create, mutate, or block hierarchy persistence.

### 3.4 `chapters`

Purpose:

- Store current accepted chapter rows under a document.

| Column | Type Candidate | Required | Notes |
|---|---|---:|---|
| `id` | `bigint identity` | yes | Internal primary key. |
| `document_id` | `bigint` | yes | Owning document. |
| `chapter_order` | `integer` | yes | Order within document. |
| `title` | `text` | no | Display title. |
| `level` | `integer` | yes | Readback/parity projection field from current hierarchy model; not parser authority. |
| `chapter_role` | `text` | no | Readback/parity projection field. |
| `reference_chapter_id` | `text` | no | Import/reference evidence only; not production identity. |
| `source_anchor` | `json/jsonb` | no | Source/span/page anchor candidate. |
| `metadata_payload` | `json/jsonb` | no | Non-authoritative chapter metadata. |

Foreign-key candidates:

- `chapters.document_id -> documents.id`.

Uniqueness candidates:

- `unique(document_id, chapter_order)`.

Index candidates:

- index on `(document_id, chapter_order)`.

Delete behavior candidates:

- delete with owning document.
- hard reparse replaces current rows; no history rows in Phase 1.

### 3.5 `sections`

Purpose:

- Store current accepted section rows under chapters.

| Column | Type Candidate | Required | Notes |
|---|---|---:|---|
| `id` | `bigint identity` | yes | Internal primary key. |
| `document_id` | `bigint` | yes | Owning document for cleanup/query scope. |
| `chapter_id` | `bigint` | yes | Parent chapter. |
| `section_order` | `integer` | yes | Order within chapter. |
| `title` | `text` | no | Display title. |
| `level` | `integer` | yes | Readback/parity projection field from current hierarchy model; not parser authority. |
| `content` | `text` | yes | Accepted structured section content for current hierarchy readback; not raw document storage and not a content-block hierarchy level. |
| `char_start` | `integer` | yes | Source span start for accepted section. |
| `char_end` | `integer` | yes | Source span end for accepted section. |
| `container_title` | `text` | no | Readback/parity projection field. |
| `section_role` | `text` | no | Readback/parity projection field. |
| `section_kind` | `text` | no | Readback/parity projection field. |
| `is_implicit_section` | `boolean` | yes | Readback/parity projection field. |
| `reference_section_id` | `text` | no | Import/reference evidence only; not production identity. |
| `source_anchor` | `json/jsonb` | no | Future normalized source/span/page anchor; SQLite validation slice currently uses explicit span columns. |
| `metadata_payload` | `json/jsonb` | no | Non-authoritative section metadata. |

Foreign-key candidates:

- `sections.document_id -> documents.id`.
- `sections.chapter_id -> chapters.id`.

Uniqueness candidates:

- `unique(chapter_id, section_order)`.

Index candidates:

- index on `(document_id, chapter_id, section_order)`.
- index on `chapter_id`.

Delete behavior candidates:

- delete with owning document or parent chapter.
- no root `sections[]` authority.

### 3.6 `task_units`

Purpose:

- Store current task-unit interaction containers under sections.
- Preserve current content payload for Phase 1 parity.

| Column | Type Candidate | Required | Notes |
|---|---|---:|---|
| `id` | `bigint identity` | yes | Internal primary key. |
| `document_id` | `bigint` | yes | Owning document for cleanup/query scope. |
| `section_id` | `bigint` | yes | Parent section. |
| `task_unit_order` | `integer` | yes | Order within section. |
| `content_payload` | `json/jsonb` or `text` | yes | Accepted task-unit content payload/string for Phase 1 readback; SQLite validation slice uses text `content_payload`. |
| `title` | `text` | no | Readback/parity projection field. |
| `container_title` | `text` | no | Readback/parity projection field. |
| `source_section_ids_payload` | `json/jsonb` | no | Import/reference source-section evidence from current model; SQLite validation slice uses `source_section_ids_payload`. |
| `is_fallback_generated` | `boolean` | yes | Readback/parity projection field. |
| `reference_unit_id` | `text` | no | Import/reference evidence from existing JSON; not production identity. |
| `metadata_payload` | `json/jsonb` | no | Non-authoritative task-unit metadata. |

Foreign-key candidates:

- `task_units.document_id -> documents.id`.
- `task_units.section_id -> sections.id`.

Uniqueness candidates:

- `unique(section_id, task_unit_order)`.
- no uniqueness guarantee on `reference_unit_id` unless fixtures prove it is needed for import diagnostics only.

Index candidates:

- index on `(document_id, section_id, task_unit_order)`.
- index on `section_id`.
- optional non-unique index on `reference_unit_id` for migration/debug lookup only.

Delete behavior candidates:

- delete with owning document or parent section.
- no row-level `structure_version`.
- no flat `task_units` primary runtime truth.

### 3.7 `content_blocks`

Purpose:

- Store lazy materialized derived content blocks linked to task units.
- Support finer-grained interaction targets without becoming hierarchy truth.

| Column | Type Candidate | Required | Notes |
|---|---|---:|---|
| `id` | `bigint identity` | yes | Internal primary key. |
| `document_id` | `bigint` | yes | Owning document. |
| `task_unit_id` | `bigint` | yes | Parent task-unit DB row. |
| `block_order` | `integer` | yes | Deterministic order within task unit. |
| `content` | `text` | yes | Block content. |
| `source_structure_version` | `integer` | yes | Provenance/freshness validation metadata. |
| `source_hash` | `text` | no | Source text hash candidate. |
| `segmentation_version` | `text` | no | Segmentation/schema version metadata. |
| `quote_span_start` | `integer` | no | Optional span start. |
| `quote_span_end` | `integer` | no | Optional span end. |
| `metadata_payload` | `json/jsonb` | no | Flexible block metadata. |
| `created_at` | `timestamp` | yes | Materialization timestamp. |

Foreign-key candidates:

- `content_blocks.document_id -> documents.id`.
- `content_blocks.task_unit_id -> task_units.id`.

Uniqueness candidates:

- `unique(task_unit_id, block_order, segmentation_version)` if repeated materialization should replace equivalent blocks.
- otherwise use no uniqueness beyond primary key until materialization behavior is confirmed.

Index candidates:

- index on `(document_id, task_unit_id, block_order)`.
- index on `(document_id, source_structure_version)` for application-level stale scans.

Delete behavior candidates:

- delete with owning document or parent task unit.
- successful hard reparse explicitly deletes all rows where `document_id` matches.
- no DB trigger enforces freshness against `documents.current_structure_version`.

### 3.8 `artifacts`

Purpose:

- Store common typed interaction output artifacts.
- Attach to validated document/chapter/section/task-unit/content-block targets.

| Column | Type Candidate | Required | Notes |
|---|---|---:|---|
| `id` | `bigint identity` | yes | Internal primary key. |
| `document_id` | `bigint` | yes | Owning document. |
| `artifact_type` | `text` | yes | Summary, quiz, answer, note, evidence, or later type. |
| `target_type` | `text` | yes | `document`, `chapter`, `section`, `task_unit`, or `content_block`. |
| `target_id` | `bigint` | yes | Resolved target DB id; polymorphic by `target_type`. |
| `source_structure_version` | `integer` | yes | Provenance/freshness validation metadata. |
| `source_hash` | `text` | no | Optional source validation hash. |
| `quote_span_start` | `integer` | no | Optional evidence span start. |
| `quote_span_end` | `integer` | no | Optional evidence span end. |
| `payload` | `json/jsonb` | yes | Type-specific artifact payload. |
| `schema_version` | `text` | no | Artifact payload schema/version metadata. |
| `metadata_payload` | `json/jsonb` | no | Lifecycle/cache/validity metadata. |
| `created_at` | `timestamp` | yes | Artifact creation time. |
| `updated_at` | `timestamp` | no | Update time if mutable artifacts are allowed. |

Foreign-key candidates:

- `artifacts.document_id -> documents.id`.
- no single FK for `target_id` in Phase 1 because target table is polymorphic.
- application/repository validation must resolve `target_type` and `target_id` before write.

Uniqueness candidates:

- none by default.
- later candidates may include `(document_id, artifact_type, target_type, target_id, source_structure_version)` if artifact overwrite/cache semantics require one current artifact per target/type/version.

Index candidates:

- index on `(document_id, artifact_type)`.
- index on `(document_id, target_type, target_id)`.
- index on `(document_id, source_structure_version)` for stale scans.
- optional index on `created_at` for recency views.

Delete behavior candidates:

- delete with owning document.
- successful hard reparse explicitly deletes all rows where `document_id` matches.
- no category-specific artifact tables in Phase 1.

### 3.9 `parse_events`

Purpose:

- Store minimal document-scoped parse provenance for accepted initial parse and hard reparse.

| Column | Type Candidate | Required | Notes |
|---|---|---:|---|
| `id` | `bigint identity` | yes | Internal primary key. |
| `document_id` | `bigint` | yes | Owning document. |
| `event_type` | `text` | yes | `initial_parse` or `hard_reparse`. |
| `previous_structure_version` | `integer` | no | Null for initial parse. |
| `new_structure_version` | `integer` | yes | Accepted structure version after event. |
| `trigger_source` | `text` | no | Upload/prepare/user/internal/low-score source metadata. |
| `parser_mode` | `text` | no | Parser mode/strategy when available. |
| `reparse_reason` | `text` | no | Hard reparse reason when applicable. |
| `invalidated_artifact_count` | `integer` | no | Hard reparse invalidation count. |
| `invalidated_content_block_count` | `integer` | no | Hard reparse invalidation count. |
| `occurred_at` | `timestamp` | yes | Event timestamp. |
| `metadata_payload` | `json/jsonb` | no | Optional provenance metadata. |

Foreign-key candidates:

- `parse_events.document_id -> documents.id`.

Uniqueness candidates:

- optional `unique(document_id, new_structure_version, event_type)` if each accepted version should have one parse event.

Index candidates:

- index on `(document_id, occurred_at)`.
- index on `(document_id, new_structure_version)`.

Delete behavior candidates:

- delete with owning document.
- do not retain after document deletion.
- do not use as event-sourced hierarchy authority.

### 3.10 `structured_document_snapshots`

Purpose:

- Optional validation/parity/debug snapshot of full `StructuredDocument` payload.
- Not hierarchy authority and not runtime fallback.

| Column | Type Candidate | Required | Notes |
|---|---|---:|---|
| `id` | `bigint identity` | yes | Internal primary key. |
| `document_id` | `bigint` | yes | Owning document. |
| `source_structure_version` | `integer` | yes | Structure version represented by snapshot. |
| `structured_document_payload` | `json/jsonb` | yes | Full snapshot payload for parity/debug. |
| `schema_version` | `text` | no | Snapshot schema/version metadata. |
| `metadata_payload` | `json/jsonb` | no | Optional validation metadata. |
| `created_at` | `timestamp` | yes | Snapshot creation time. |

Foreign-key candidates:

- `structured_document_snapshots.document_id -> documents.id`.

Uniqueness candidates:

- `unique(document_id, source_structure_version)` if Phase 1 keeps one parity snapshot per accepted structure version.

Index candidates:

- index on `(document_id, source_structure_version)`.

Delete behavior candidates:

- delete with owning document.
- optional deletion/replacement on hard reparse depends on whether parity snapshots are retained for debug within document lifetime; they must not become runtime history authority.

## 4. Cross-Table Consistency Candidates

Candidate consistency rules:

1. Child rows should duplicate `document_id` for cleanup/query scope, while parent FKs preserve hierarchy ownership.
2. Application logic should validate that child `document_id` matches the parent row's `document_id` when inserting sections, task units, and content blocks.
3. Artifact target validation remains application/repository-level because `target_id` is polymorphic.
4. Derived `source_structure_version` freshness remains application-level.
5. No table should store row-level hierarchy `structure_version` for chapters, sections, or task units.

## 5. Candidate Index Summary

Minimum likely lookup candidates:

- `documents(namespace, document_name)`
- `chapters(document_id, chapter_order)`
- `sections(document_id, chapter_id, section_order)`
- `task_units(document_id, section_id, task_unit_order)`
- `content_blocks(document_id, task_unit_id, block_order)`
- `artifacts(document_id, artifact_type)`
- `artifacts(document_id, target_type, target_id)`
- `parse_events(document_id, occurred_at)`

Stale/validation scan candidates:

- `content_blocks(document_id, source_structure_version)`
- `artifacts(document_id, source_structure_version)`
- `document_profile(document_id, source_structure_version)`
- `structured_document_snapshots(document_id, source_structure_version)`

These are candidates only. Final index selection should follow observed repository query paths and migration smoke-check requirements.

## 6. Governance Validation

This candidate design does not introduce:

- executable SQL DDL
- migration scripts
- ORM models
- repository interfaces
- runtime read/write switch
- root `sections[]`
- `structure_nodes`
- flat `task_units` as primary runtime truth
- profile authority
- artifact hierarchy authority
- raw-byte DB storage
- category-specific artifact tables in Phase 1
- DB schema as parser authority
- row-level hierarchy `structure_version`
- immutable hierarchy history
- staging hierarchy persistence

## 7. Completion Boundary

The checklist item "Define physical table, column, foreign-key, uniqueness, and index candidates for Phase 1" is complete when this document is present and linked from `db/module-checklist.md`.

Remaining separate tasks include ORM/model mapping, repository/storage interfaces, fixtures, transaction-level hard reparse implementation planning, runtime switch planning, and executable migration work.
