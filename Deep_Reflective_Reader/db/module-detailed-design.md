# db Detailed Design

## 1. Module Purpose

`db/` is the documentation-only golden source for future DB-era persistence planning. It records maintainer-approved DB identity, versioning, reparse, and derived-resource lifecycle rules before any schema, ORM, repository interface, migration, or runtime behavior is implemented. **[Maintainer-Confirmed]**

This module does not represent an implemented Python package in the current codebase. It exists to prevent future DB work from rediscovering or contradicting the agreed architecture decisions. **[Doc-Confirmed]**

## 2. Position in Overall Architecture

- Future DB persistence planning / governance layer
- Cross-module design source for future schema and storage implementation work

The existing file-backed runtime remains valid until DB readiness, rollout, and validation gates are explicitly satisfied. **[From Proposal]**

## 3. Key Files

| File | Responsibility | Notes |
|---|---|---|
| `db/module-detailed-design.md` | Golden source for DB-era identity, versioning, reparse, and derived-resource lifecycle policy | Documentation only; no schema or implementation approval |
| `db/module-checklist.md` | Tracks DB-era planning status and future implementation tasks | Future tasks remain unchecked until implemented with evidence |
| `db/storage-contract-inventory.md` | Inventory of current storage contracts before DB migration design | Documentation-only preparation |
| `db/storage-contract-design.md` | Conceptual storage contract boundaries before schema/backend implementation | Documentation-only preparation |
| `db/structured-document-jsonb-evaluation.md` | Phase 1 `StructuredDocument` JSONB-first evaluation plan | Evaluation planning only; no schema approval |
| `db/structured-document-jsonb-evaluation-readiness.md` | Readiness audit for future Phase 1 JSONB evaluation | Audit/planning only; no implementation impact |

## 4. Main Responsibilities

1. Define DB-era identity policy before schema design. **[Maintainer-Confirmed]**
2. Define document-level `structure_version` semantics. **[Maintainer-Confirmed]**
3. Define current-state-only hierarchy persistence policy for early DB design. **[Maintainer-Confirmed]**
4. Define hard reparse transaction and derived-resource invalidation policy. **[Maintainer-Confirmed]**
5. Define minimal parse event provenance requirements. **[Maintainer-Confirmed]**
6. Preserve boundaries from existing repository memory: hierarchy-first, no hidden mutation, metadata advisory-only, and no backend/schema authority. **[From Proposal] + [From HLD]**

## 5. Non-Responsibilities

1. Does not implement database schema, tables, migrations, ORM models, repositories, fixtures, or runtime read/write behavior. **[Maintainer-Confirmed]**
2. Does not select a final DB backend beyond previously documented PostgreSQL JSONB evaluation assumptions. **[From Proposal]**
3. Does not redefine `StructuredDocument` hierarchy truth. **[From HLD]**
4. Does not make database schema a parser, hierarchy, artifact, profile, retrieval, or raw-document authority. **[From Proposal]**
5. Does not preserve current Python-generated `unit_id` as production identity. **[Maintainer-Confirmed]**
6. Does not introduce immutable hierarchy history, staging hierarchy tables, or candidate promotion workflows in the first DB design. **[Maintainer-Confirmed]**

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

A separate public/domain identity may be introduced only when there is a concrete long-lived external reference requirement beyond the DB row lifecycle. Candidate areas include:

- public document identity
- artifact target snapshots
- content-block validation metadata

Separate public/domain IDs carry generation, uniqueness, persistence, mapping, and lifecycle costs. They must be deliberate, not automatic. **[Maintainer-Confirmed]**

### 6.3 Reference / Import Identity Rule

Existing file-based JSON data is reference/test data only and will not be production-migrated as stable API-visible identity. Current Python-generated `unit_id` values are reference/import identity evidence only. **[Maintainer-Confirmed]**

Future DB work must not depend on current Python-generated `unit_id` as the production task-unit identity strategy. **[Maintainer-Confirmed]**

### 6.4 API Identity Rule

Early API references may use DB-generated IDs unless a stable public API requirement is explicitly introduced later. **[Maintainer-Confirmed]**

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

## 16. Current Risks

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

## 17. Open Questions for Maintainer

No unresolved confirmation items identified for the first DB-era identity and hard-reparse policy captured in this pass.

## 18. Suggested Next Documentation Improvements

1. When schema work is explicitly requested, derive an implementation checklist from this document before proposing tables.
2. Before any DB runtime work, reconcile this document with `document_structure`, `config`, `document_preparation`, `shared`, and `section_tasks` module checklists.
3. If a public API stability requirement appears, add a targeted public/domain identity decision record before implementation.
