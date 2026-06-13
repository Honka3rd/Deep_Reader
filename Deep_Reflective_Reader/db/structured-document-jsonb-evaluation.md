# StructuredDocument JSONB-First Evaluation

## Purpose

This document defines Phase 1 evaluation criteria for a future `StructuredDocument` DB persistence evaluation before any DB implementation work.

It captures a planning boundary between the long-lived storage contract design and any future schema design, migration design, backend implementation, or runtime rollout. This is documentation-only planning. It does not approve a schema, create a repository interface, introduce an ORM, execute migration work, or switch runtime read paths.

## Evaluation Backend Assumption

PostgreSQL is the assumed evaluation backend for Phase 1 because JSONB is the evaluated representation.

This assumption is not a final backend commitment. PostgreSQL selection can still be revisited before implementation, PostgreSQL schema is not authoritative, and PostgreSQL storage must not become the hierarchy authority.

`StructuredDocument` remains authoritative for hierarchy validation and hierarchy semantics.

## Evaluated Representation

Phase 1 evaluates a JSONB-first representation for `StructuredDocument`.

This is not a final representation decision. Later planning may evaluate relational-first or hybrid relational plus JSONB approaches after JSONB-first hierarchy parity has been assessed.

This document does not define:

- schema design
- table design
- JSONB layout
- ORM design
- repository design
- migration implementation

## Validation Authority

`StructuredDocument` is the validation authority.

The storage backend is not authority. Database schema is not authority. Storage integrity checks may exist in a later implementation, but hierarchy acceptance must depend on whether a JSONB-loaded payload can be restored into the existing `StructuredDocument` model and pass the existing hierarchy validation flow.

The database may reject broken storage. It must not decide chapter, section, task-unit, parser, artifact-target, or fail-fast hierarchy semantics.

## Phase 1 Scope

### Included

Phase 1 covers bare `StructuredDocument` hierarchy parity:

- document identity
- namespace identity when currently used by the storage path
- chapter hierarchy
- section hierarchy
- task unit hierarchy
- task unit ids
- task unit ordering
- parent chapter and section placement
- hierarchy-first lookup parity
- fail-fast hierarchy behavior
- `task_unit.content` preservation as opaque payload

`task_unit.content` must be preserved if present and exported back without semantic loss, but it is not hierarchy identity and must not be used to validate content-block semantics or artifact target semantics.

### Task Unit Identity Clarification

Phase 1 does not resolve the final DB-era task-unit identity strategy.

For Phase 1 parity evidence, current persisted `unit_id` values may be used only as reference/import identity baselines from existing structured JSON. They must not be treated as proof that the current Python-generated identity algorithm is sufficient for long-term DB-backed identity.

Future pre-schema planning must distinguish:

- DB-generated internal primary key
- public/domain identity only where a concrete external stability requirement exists
- reference/import identity from existing structured JSON

DB-generated internal primary keys are the default internal identity and relational link foundation for early DB design. Current Python-generated `unit_id` values must not be treated as production DB identity, public API identity, or proof that a separate task-unit domain ID is required. Artifact targets, content-block links, API references, and evaluation records may need validation metadata or deliberate public/domain identity later, but this document does not choose an ID format, ID generator, table design, column design, or migration algorithm.

### Excluded

Phase 1 explicitly excludes:

- content blocks
- content segmentation semantics
- lazy-loading derivation rules
- artifact targets
- artifact persistence
- artifact target reference parity
- retrieval persistence
- profile persistence
- raw document governance
- user account ownership and authorization
- deletion and retention policy
- cross-user sharing policy
- production runtime switching
- backend cutover

Artifact target references and content-block-level target validation belong to later evaluation stages after bare hierarchy parity is proven.

### Lazy Content Block Persistence Clarification

Content blocks remain excluded from Phase 1 acceptance. They should not be embedded into the normal Phase 1 `StructuredDocument` JSONB payload as the default persistence strategy.

Future direction:

- Content blocks should remain lazy-loaded or lazily materialized resources linked to task units.
- If a user does not request content blocks, they should not be eagerly computed.
- Once computed, content blocks should be persisted separately from `StructuredDocument` JSONB.
- Related artifacts should also be persisted separately and linked to hierarchy or content-block targets.

This preserves the Phase 1 focus on bare hierarchy parity while acknowledging that content-block-aware persistence needs a later, separate planning track. This document does not design tables, columns, artifact schema, lazy-loading behavior, or content-block persistence implementation.

## Positive Validation Requirements

Phase 1 must validate semantic hierarchy parity, not byte-level JSON equality.

Required positive validation areas:

- Semantic hierarchy parity: chapters, sections, task units, task unit ids, task unit order, and parent placement remain equivalent between the file-backed baseline and the JSONB-backed reload.
- Runtime lookup parity: both representations are loaded into `StructuredDocument`, and existing hierarchy lookup helpers return equivalent results.
- Import parity: file-backed structured documents can be imported as non-destructive JSONB copies without changing the source files.
- DB-to-model reload parity: the JSONB representation can be reloaded into `StructuredDocument` and pass the existing hierarchy validation flow.
- Namespace and document isolation: `doc_name` is preserved, the namespace is preserved when currently used, two documents do not overwrite each other, the same `doc_name` under different namespaces remains isolated, and reload returns the requested `StructuredDocument`.
- Opaque task-unit payload preservation: `task_unit.content` is not lost or corrupted, while remaining outside hierarchy identity.

The evaluation should use both a curated fixture set and representative real structured documents from the current `data/structured` repository. The real-document set does not need to include every existing structured document, but it must provide repository realism beyond idealized fixtures.

The DB-to-model reload path is validation-only. It must not become a production read path, backend switch, or runtime replacement for file-backed loading.

## Negative Validation Requirements

Phase 1 must also prove that JSONB-backed validation does not weaken the existing fail-fast hierarchy contract.

Required negative cases include:

- legacy-only payload rejection
- missing chapter failures
- duplicate hierarchy ID failures
- malformed hierarchy placement failures
- invalid primary hierarchy structures

Failure comparison is by category, not by exact implementation details. The JSONB-backed reload path should reject the same invalid hierarchy states for the same conceptual hierarchy-validation reason, but it does not need identical exception classes, messages, stack traces, or internal failure paths.

Legacy compatibility fields are migration concerns, not Phase 1 hierarchy parity criteria. Strict hierarchy documents should be part of parity validation. Legacy-only payloads should be explicitly rejected or handled by separate migration-only tooling before validation.

## Rollback Position

Phase 1 is strictly non-destructive.

`data/structured/` remains the read-only baseline and compatibility source during evaluation. Existing structured files may be read as import source and parity baseline, but Phase 1 must not rewrite, normalize, delete, or replace them.

Allowed:

- read existing structured files
- import copies into PostgreSQL JSONB for evaluation
- reload from JSONB into `StructuredDocument` for validation
- compare against the file-backed baseline

Forbidden:

- modify `data/structured/`
- delete `data/structured/`
- rewrite structured JSON files
- export DB records back into `data/structured/` as a Phase 1 requirement
- switch file-backed runtime to DB-backed runtime

The original structured file remains the rollback source. A DB-to-file export path may be evaluated later before any runtime read-path switch, but it is not a Phase 1 acceptance requirement.

## Evidence Artifact Format

The Phase 1 decision gate should use a lightweight evidence record. It should not prescribe audit tooling, test frameworks, migration tooling, or implementation report format.

Status values:

- Pass
- Fail
- Partial

Minimum evidence table:

| Evaluation Area | Expected Evidence | Status | Notes |
|---|---|---|---|
| Hierarchy parity | Chapters, sections, task units, ids, order, and placement match semantically after JSONB reload into `StructuredDocument`. | Pass / Fail / Partial | |
| Runtime lookup parity | Existing hierarchy-first lookup helpers return equivalent results for file-backed and JSONB-reloaded `StructuredDocument` instances. | Pass / Fail / Partial | |
| Import safety | File-backed structured documents are imported as non-destructive DB copies; original files remain unchanged. | Pass / Fail / Partial | |
| DB-to-model reload parity | JSONB records reload into `StructuredDocument` and pass existing hierarchy validation. | Pass / Fail / Partial | |
| Negative validation parity | Invalid hierarchy states fail for equivalent validation categories. | Pass / Fail / Partial | |
| Namespace/document isolation | `doc_name` and namespace identity are preserved; records do not overwrite or cross-load between identities. | Pass / Fail / Partial | |
| Rollback assumptions | `data/structured/` remains read-only and remains the rollback source. | Pass / Fail / Partial | |

## Decision Gate

The Phase 1 gate should be written and maintainer-approved before moving to Phase 2.

`document_structure` owns the technical hierarchy parity evidence because Phase 1 evaluates `StructuredDocument`. `config` supports future backend policy only. The maintainer owns final phase-transition approval.

Recommended Outcome:

- Proceed: Phase 1 evidence satisfies the required JSONB-first semantic hierarchy parity criteria. The project may move to Phase 2 planning while keeping file-backed storage as the runtime source.
- Revise: Phase 1 evidence is partially successful, but specific gaps must be fixed or re-evaluated before Phase 2. Examples include missing negative validation, incomplete runtime lookup parity, unclear namespace isolation, or insufficient real-document coverage.
- Stop: Phase 1 evidence shows JSONB-first storage is unsafe or incompatible with the `StructuredDocument` contract. The project should not proceed with JSONB-first planning until the strategy is reconsidered.

Final Decision:

- Maintainer Approval

## Explicit Non-Goals

- No schema design.
- No table design.
- No JSONB layout.
- No repository design.
- No repository interface implementation.
- No ORM design.
- No migration implementation.
- No migration scripts.
- No dual-write design.
- No runtime read-path switch.
- No backend cutover.
- No production backend selection.
- No API change.
- No runtime behavior change.
- No dependency change.
- No source code change.
