# Phase 1 Repository / Storage Interface Plan

## 1. Purpose

This document defines DB repository/storage interface candidates for Phase 1 without making database schema, ORM models, or storage adapters domain authority.

It is documentation-only. It does not create Python interfaces, repository classes, ORM models, SQL DDL, migration scripts, runtime read/write behavior, fixtures, tests, API changes, or backend selection.

The goal is to define future storage boundaries before code exists.

## 2. Interface Principles

1. Domain semantics own interface meaning; schema and ORM only implement persistence.
2. `document_structure` owns current hierarchy semantics.
3. `shared` owns shared DTO vocabulary for task units, content blocks, and artifact targets.
4. `profile` owns advisory profile payload semantics.
5. `config` owns future backend selection, not domain behavior.
6. Repository interfaces must not parse, classify, repair, or infer hierarchy.
7. Repository interfaces must not introduce root `sections[]`, `structure_nodes`, or flat `task_units` as primary runtime truth.
8. Repository interfaces must not expose ORM session-bound objects across domain/service boundaries.
9. Repository interfaces should accept validated domain inputs or explicit persistence DTOs, not raw parser internals.
10. Runtime read/write switch remains a separate future task.

## 3. Candidate Interface Ownership

| Interface Candidate | Owner | Purpose | Implementation Boundary |
|---|---|---|---|
| `StructuredDocumentStoragePort` | `document_structure` semantics | Persist and read current document hierarchy. | DB/file adapters implement it; schema does not define hierarchy truth. |
| `HardReparseUnitOfWorkPort` | `document_structure` lifecycle semantics | Replace hierarchy, advance version, delete derived rows, write parse event transactionally. | DB adapter owns transaction mechanics only. |
| `ContentBlockStoragePort` | `shared` vocabulary + `document_structure` linkage | Persist and read lazy materialized content blocks. | Derived resource storage only. |
| `ArtifactStoragePort` | `document_structure` artifact lifecycle + `shared` target refs | Persist and read common artifacts. | Must require hierarchy-aware target validation before write. |
| `ParseEventStoragePort` | `document_structure` lifecycle provenance | Store minimal parse provenance. | Provenance only, not event sourcing. |
| `DocumentProfileStoragePort` | `profile` semantics | Store advisory document-scoped profile snapshot. | Advisory metadata only. |
| `RawSourceMetadataStoragePort` | raw source metadata boundary | Store metadata-only raw source reference. | Does not store raw bytes or extracted raw text. |
| `StructuredDocumentSnapshotStoragePort` | DB validation/parity track | Store optional parity/debug snapshot. | Not runtime hierarchy fallback. |

These names are candidates. Future code may choose different names if ownership and authority boundaries remain intact.

## 4. Pseudo-Interface Candidates

Pseudo-interfaces below are documentation shapes, not Python code.

### 4.1 `StructuredDocumentStoragePort`

Purpose:

- Persist initial accepted hierarchy.
- Load current hierarchy through `Document -> Chapter -> Section -> TaskUnit`.
- Avoid dependency on Python-generated `unit_id` as production identity.

Candidate operations:

```text
create_document_with_initial_structure(input) -> document_identity
load_current_structure(document_id) -> current_structure_read_model
load_document_lifecycle(document_id) -> document_lifecycle_read_model
find_document_by_name(namespace, document_name) -> document_identity | not_found
```

Required boundaries:

- Input hierarchy must already be parser-validated.
- Output should be detached DTO/read model, not ORM records.
- No root `sections[]` or `structure_nodes` read path.
- No hidden profile diagnostics write-back.

### 4.2 `HardReparseUnitOfWorkPort`

Purpose:

- Define one explicit transaction boundary for accepted hard reparse.

Candidate operation:

```text
replace_structure_after_validated_hard_reparse(input) -> hard_reparse_result
```

Required transaction semantics:

1. Candidate hierarchy is validated before the unit of work starts.
2. Replace current hierarchy rows for the document.
3. Advance `documents.current_structure_version`.
4. Delete all document content blocks.
5. Delete all document artifacts.
6. Write one parse event.
7. Commit atomically.

Required boundaries:

- Candidate validation failure must not enter destructive transaction.
- Cascades may be defensive but must not hide explicit lifecycle cleanup.
- The result should include invalidated content-block and artifact counts.

### 4.3 `ContentBlockStoragePort`

Purpose:

- Store lazy materialized content blocks linked to task-unit DB identity.
- Support read paths for task-unit content detail without making blocks hierarchy truth.

Candidate operations:

```text
list_content_blocks(document_id, task_unit_id) -> content_block_read_models
replace_task_unit_content_blocks(document_id, task_unit_id, blocks, source_structure_version) -> write_result
delete_document_content_blocks(document_id) -> deletion_count
```

Required boundaries:

- Blocks are derived resources.
- `source_structure_version` is provenance and application-level freshness metadata.
- Hard reparse deletes all document blocks.
- Content blocks do not become `Document -> Chapter -> Section -> TaskUnit -> Block` hierarchy authority.

### 4.4 `ArtifactStoragePort`

Purpose:

- Store and read one common artifact entity with `artifact_type` and type-specific payload.
- Preserve hierarchy-aware target validation boundary.

Candidate operations:

```text
save_artifact(document_id, validated_target, artifact_type, payload, provenance) -> artifact_identity
list_artifacts_for_target(document_id, validated_target) -> artifact_read_models
list_artifacts_for_document(document_id, filters) -> artifact_read_models
delete_document_artifacts(document_id) -> deletion_count
```

Required boundaries:

- `validated_target` must be resolved before write.
- Repository should not accept unresolved raw `target_id` without target validation metadata.
- Artifacts are interaction outputs only.
- No category-specific artifact repositories in Phase 1.
- Hard reparse deletes all document artifacts.

### 4.5 `ParseEventStoragePort`

Purpose:

- Persist accepted initial parse and hard reparse provenance.

Candidate operations:

```text
append_initial_parse_event(document_id, new_structure_version, metadata) -> parse_event_identity
append_hard_reparse_event(document_id, previous_structure_version, new_structure_version, invalidation_counts, metadata) -> parse_event_identity
list_parse_events(document_id) -> parse_event_read_models
```

Required boundaries:

- Parse events do not define current structure version.
- Parse events are not event sourcing.
- Parse events are retained only for the lifetime of the document.

### 4.6 `DocumentProfileStoragePort`

Purpose:

- Persist advisory document-scoped profile snapshot.

Candidate operations:

```text
upsert_document_profile(document_id, profile_snapshot, metadata) -> profile_identity
load_document_profile(document_id) -> profile_read_model | not_found
```

Required boundaries:

- Profile is advisory metadata.
- Profile cannot create, mutate, delete, or block hierarchy rows.
- Diagnostics must not be written back through read paths.

### 4.7 `RawSourceMetadataStoragePort`

Purpose:

- Persist raw-source metadata while raw bytes and extracted raw text remain file-backed/object-backed.

Candidate operations:

```text
upsert_raw_source_metadata(document_id, raw_source_metadata) -> raw_source_metadata_identity
load_raw_source_metadata(document_id) -> raw_source_metadata_read_model | not_found
```

Required boundaries:

- No raw bytes or extracted raw text in DB.
- No cross-user sharing implication.
- Raw file/object deletion policy remains separate.

### 4.8 `StructuredDocumentSnapshotStoragePort`

Purpose:

- Persist optional full `StructuredDocument` parity/debug snapshot.

Candidate operations:

```text
save_validation_snapshot(document_id, source_structure_version, structured_document_payload, metadata) -> snapshot_identity
load_validation_snapshot(document_id, source_structure_version) -> snapshot_read_model | not_found
```

Required boundaries:

- Snapshot is validation/parity/debug only.
- Snapshot conflict is validation failure, not fallback behavior.
- Snapshot is not runtime hierarchy source.

## 5. DTO Boundary Candidates

Future code should define explicit DTO/read-model shapes before implementation.

Candidate DTO groups:

- document lifecycle DTO
- current hierarchy read model
- validated hierarchy write input
- hard reparse input/result
- content block read/write DTO
- validated artifact target DTO
- artifact read/write DTO
- parse event read/write DTO
- advisory profile snapshot DTO
- raw-source metadata DTO
- optional snapshot DTO

DTOs should be detached from ORM sessions and must not expose database implementation internals to parser, API, or task-layout code.

## 6. Schema Authority Guardrails

Repository/storage interfaces must not:

- derive parser structure from table shape
- accept database rows as parser output
- make ORM model relationships parser truth
- use `reference_unit_id` as production identity
- expose `sections` or `task_units` as flat primary runtime truth
- decide artifact availability without hierarchy-aware target validation
- write profile diagnostics during read paths
- decide runtime backend switching

## 7. Runtime Integration Boundary

This plan does not enable DB runtime reads or writes.

Future integration must wait for separate tasks:

- `document_structure` storage integration plan
- raw-source metadata persistence plan
- advisory profile persistence plan
- fixtures and parity validation plan
- transaction-level hard reparse implementation plan
- backend configuration integration with `config`
- explicit runtime read/write switch plan

## 8. Governance Validation

This interface plan does not introduce:

- Python repository interfaces
- executable ORM model code
- executable SQL DDL
- migration scripts
- runtime read/write switch
- root `sections[]`
- `structure_nodes`
- flat `task_units` as primary runtime truth
- profile authority
- artifact hierarchy authority
- raw-byte DB storage
- category-specific artifact repositories in Phase 1
- DB schema as parser authority
- hidden task-layout persistence mutation
- diagnostics profile write-back

## 9. Completion Boundary

The checklist item "Define DB repository/storage interfaces without making schema authority" is complete when this document is present and linked from `db/module-checklist.md`.

Remaining separate tasks include document-structure integration, raw-source metadata persistence behavior, profile persistence behavior, fixtures, hard reparse transaction implementation planning, content-block and artifact persistence implementation planning, backend configuration integration, runtime switch planning, and executable code work.
