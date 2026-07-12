# Phase 1 document_structure Storage Integration Plan

## 1. Purpose

This document defines the Phase 1 integration plan between `document_structure` and future DB-backed hierarchy persistence.

It is documentation-only. It does not create Python code, repository interfaces, ORM models, SQL DDL, migration scripts, runtime read/write behavior, fixtures, tests, API changes, or backend selection.

The goal is to define how DB-backed hierarchy writes and reads should integrate with `document_structure` without making database schema, ORM classes, or storage adapters hierarchy authority.

## 2. Source References

- `Deep_Reflective_Reader/db/phase-1-schema-design.md`
- `Deep_Reflective_Reader/db/phase-1-physical-schema-candidates.md`
- `Deep_Reflective_Reader/db/phase-1-orm-model-mapping-plan.md`
- `Deep_Reflective_Reader/db/phase-1-repository-storage-interface-plan.md`
- `Deep_Reflective_Reader/db/module-detailed-design.md`
- `Deep_Reflective_Reader/document_structure/module-detailed-design.md`
- `Deep_Reflective_Reader/shared/module-detailed-design.md`
- `Deep_Reflective_Reader/document_preparation/module-detailed-design.md`

## 3. Fixed Integration Boundaries

`document_structure` owns hierarchy semantics.

Future DB adapters may persist accepted hierarchy rows, but they must not:

- decide parser output validity
- infer chapters, sections, or task units from table shape
- repair invalid parser output
- expose ORM records as domain objects
- use current Python-generated `unit_id` as production identity
- introduce automatic public/domain IDs for every hierarchy node
- introduce root `sections[]`
- introduce `structure_nodes`
- use flat `task_units` as primary runtime truth

The accepted hierarchy path remains:

```text
Document -> Chapter -> Section -> TaskUnit
```

The DB-backed current hierarchy is current-state-only. `documents.current_structure_version` is authoritative. Initial successful parse creates `structure_version = 1`. Hard reparse advances the document-level structure version only after a validated replacement is accepted.

Hierarchy rows should not carry row-level `structure_version`, immutable hierarchy history, staging-state markers, or candidate hierarchy states in Phase 1.

## 4. Candidate Integration Surfaces

### 4.1 Initial Hierarchy Write

Candidate owner flow:

1. `document_preparation` obtains raw text and parser output.
2. `document_structure` validates or constructs the accepted hierarchy model.
3. A storage-facing mapper converts the accepted hierarchy model into detached persistence DTOs.
4. `StructuredDocumentStoragePort.create_document_with_initial_structure(...)` persists:
   - document lifecycle row
   - current hierarchy rows
   - initial parse provenance
   - optional validation/parity snapshot, if enabled by a separate validation track
5. The write returns document identity and current structure version.

Required behavior:

- The DB adapter stores an already accepted hierarchy.
- The DB adapter does not parse, classify, split, merge, or repair hierarchy.
- The initial hierarchy write must create document-level `current_structure_version = 1`.
- Parser diagnostics or profile metadata must not mutate hierarchy during persistence.
- The optional `StructuredDocument` JSONB snapshot, if present, remains validation/parity/debug evidence only.

### 4.2 Current Hierarchy Read

Candidate owner flow:

1. Application code requests current hierarchy by `document_id`.
2. `StructuredDocumentStoragePort.load_current_structure(document_id)` reads:
   - document lifecycle metadata
   - ordered chapters
   - ordered sections under each chapter
   - ordered task units under each section
3. The adapter returns a detached current hierarchy read model.
4. `document_structure` maps or validates the read model into the runtime hierarchy representation needed by callers.

Required behavior:

- Reads use `Document -> Chapter -> Section -> TaskUnit`.
- Reads must not fall back to root `sections[]`, `structure_nodes`, flat task-unit truth, or optional JSONB snapshot authority.
- Missing hierarchy, inconsistent ownership, duplicate ordering, or cross-document parent linkage should fail fast.
- ORM session-bound rows must not escape into application, parser, profile, artifact, retrieval, or API boundaries.

### 4.3 Document Lifecycle Read

Future code may need a lightweight lifecycle read separate from full hierarchy loading.

Candidate fields:

- `document_id`
- `namespace` and `document_name`
- `current_structure_version`
- created/update timestamps
- raw-source metadata presence
- optional profile snapshot presence

Required behavior:

- Lifecycle reads report document state only.
- Lifecycle reads do not imply hierarchy validity if full hierarchy has not been loaded.
- Profile, raw-source metadata, artifacts, and content blocks remain non-authoritative for hierarchy.

### 4.4 Hard Reparse Replacement

Candidate owner flow:

1. A user-triggered hard reparse creates a candidate hierarchy outside durable hierarchy storage.
2. `document_structure` validates the candidate hierarchy before destructive persistence begins.
3. `HardReparseUnitOfWorkPort.replace_structure_after_validated_hard_reparse(...)` performs one transaction:
   - replace current chapter/section/task-unit rows
   - advance `documents.current_structure_version`
   - delete document-level content blocks
   - delete document-level artifacts
   - append one hard-reparse parse event
4. The transaction returns new structure version and invalidation counts.

Required behavior:

- Candidate validation failure must not delete existing hierarchy, content blocks, or artifacts.
- Hard reparse must not persist staging hierarchy.
- Hard reparse must not keep immutable hierarchy history.
- Content-block and artifact invalidation is physical deletion at document scope.
- `source_structure_version` freshness remains application-level validation for derived rows, not DB authority.

## 5. Mapper and DTO Boundary

Future implementation should keep three layers distinct:

```text
document_structure domain model
-> detached storage DTO / read model
-> ORM records / SQL rows hidden inside adapter
```

Candidate DTO groups:

- accepted hierarchy write input
- current hierarchy read model
- document lifecycle read model
- hard reparse input/result
- parse provenance write/read DTO
- optional validation snapshot DTO

Mapping rules:

- Domain objects should not depend on ORM classes.
- Parser output should not depend on table names or database constraints.
- API callers should not receive ORM records.
- Storage DTOs may carry DB-generated IDs after persistence.
- `reference_unit_id` or legacy Python `unit_id` may remain import/reference evidence only, not production identity.

## 6. Fail-Fast Conditions

Future DB-backed hierarchy reads and writes should fail explicitly when they encounter:

- missing document for requested `document_id`
- document with no current hierarchy after accepted parse
- chapter, section, or task unit row linked to a different document
- section without a chapter
- task unit without a section
- duplicate sibling ordering where ordering is expected to be unique
- stale derived-resource write attempted against a non-current structure version
- artifact write attempted without validated hierarchy-aware target metadata

Failure handling should be explicit application behavior. The DB schema may provide defensive constraints, but it must not become parser authority or hidden repair logic.

## 7. File-Backed Coexistence Boundary

This plan does not enable a runtime read/write switch.

Existing file-backed structured outputs remain valid until a separate backend policy task defines how DB-backed persistence is selected, tested, rolled out, and rolled back.

During coexistence planning:

- file-backed runtime behavior remains the baseline
- DB-backed hierarchy persistence remains opt-in future implementation work
- optional DB parity snapshots may support validation only
- no runtime fallback from relational hierarchy to JSONB snapshot should be introduced
- no production migration should treat current Python-generated `unit_id` as stable identity

## 8. Relationship to Other Phase 1 Tracks

This integration plan only covers `document_structure` hierarchy persistence integration.

Separate future tasks must define:

- raw-source metadata persistence while raw bytes and extracted raw text remain file-backed/object-backed
- advisory `document_profile` persistence
- parse event persistence implementation details
- content-block relational persistence
- artifact relational persistence
- application-level stale derived-row validation
- backend configuration integration with `config/`
- validation fixtures and parity checks
- runtime read/write switch policy, if later approved

## 9. Governance Validation

This plan explicitly does not introduce:

- root `sections[]`
- `structure_nodes`
- flat `task_units` as primary runtime truth
- profile authority
- artifact hierarchy authority
- raw-byte DB storage
- category-specific artifact tables
- DB schema as parser authority
- ORM classes as parser authority
- runtime read/write switch
- SQL DDL
- ORM implementation
- repository interface code
- migration script
- tests or fixtures

## 10. Completion Boundary

This task is complete when the repository contains this documentation plan and the DB checklist/progress files record it as planning evidence.

Implementation remains incomplete until future tasks create and validate actual storage ports, adapters, migrations, ORM mappings, transaction behavior, backend configuration, fixtures, and tests.
