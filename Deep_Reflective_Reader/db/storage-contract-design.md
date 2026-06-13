# Storage Contract Design

## Purpose

This document defines conceptual storage contract boundaries for `Deep_Reflective_Reader` before any DB schema design, migration design, or backend implementation.

It sits between:

- `db/storage-contract-inventory.md`
- future schema design
- future migration design
- future backend implementation

This document is documentation-only. It does not define database tables, JSONB layouts, ORM mappings, repository interfaces, migration scripts, dual-write behavior, read-path switching, API changes, or runtime behavior changes.

## Core Principle

Future persistence should follow this conceptual boundary:

```text
Domain Module
  -> Domain Storage Contract
  -> Backend Implementation
```

The domain module owns persistence meaning. The domain storage contract preserves that meaning across possible backend implementations. The backend implementation is replaceable and must not become domain authority.

`config` owns backend selection and rollout policy only. It may eventually select file-backed storage, DB-backed storage, coexistence mode, or rollout switches, but it must not define hierarchy semantics, profile semantics, retrieval semantics, parser semantics, raw document ownership, or artifact meaning.

Database schema is not domain authority. A future schema can represent domain data, but it must not redefine parser behavior, hierarchy identity, artifact target meaning, profile advisory semantics, retrieval authority, or raw document ownership.

## PostgreSQL Reference Note

PostgreSQL can support both relational modeling and JSONB document-style storage. This matters because future DB planning may choose:

- relational representation
- JSONB representation
- hybrid relational plus JSONB representation

This document does not choose among those options. It does not define tables, columns, JSONB layouts, indexes, migrations, ORM mappings, query strategy, or backend implementation details.

## Shared Storage Contract Invariants

All future storage contracts must preserve these invariants:

- No backend becomes source of truth.
- No schema becomes parser authority.
- No storage implementation becomes hierarchy authority.
- No hidden read-path mutation.
- No diagnostics profile write-back.
- No reintroduction of root `sections[]` as primary persistence or runtime flow.
- No reintroduction of `structure_nodes` as main flow.
- No flat `task_units` as primary persistence or runtime flow.
- No non-hierarchy-aware artifact write.
- Existing `data/` file storage remains valid during migration planning and rollout.
- Migration must be gradual and track-separated.
- Compatibility handling must remain explicit and migration-only where legacy structured payloads are involved.
- Runtime hierarchy lookup must remain hierarchy-first and fail-fast.

## Domain Contract Taxonomy

### Structured Document Storage Contract

Owner: `document_structure`

Minimum capabilities:

- Save hierarchy-first `StructuredDocument`.
- Load hierarchy-first `StructuredDocument`.
- Validate primary hierarchy.
- Fail fast on invalid primary structure.
- Support explicit migration-only legacy handling.

Authoritative: yes.

Must exclude:

- parser decisions
- DB schema
- root `sections[]` as primary source
- `structure_nodes` as main flow
- flat `task_units` as primary persistence

The structured document contract is the highest-priority storage contract because it protects `chapters[].sections[].task_units[]`, hierarchy identity, target safety, and fail-fast runtime behavior.

### Profile Storage Contract

Owner: `profile`

Minimum capabilities:

- Save profile snapshot.
- Load profile snapshot.
- Preserve advisory metadata semantics.
- Identify rebuild triggers conceptually.

Authoritative: no.

Must exclude:

- parser authority
- diagnostics write-back
- hierarchy truth
- artifact availability truth

Profile storage records metadata snapshots. `parser_metadata`, `post_structure_metadata`, LLM classification, and diagnostics material must remain advisory and must not control parser structure or runtime hierarchy.

### Retrieval Storage Contract

Owner: `retrieval`

Minimum capabilities:

- Persist derived retrieval artifacts.
- Load retrieval artifacts.
- Validate cache/index freshness conceptually.
- Allow rebuild from raw text and embedding config.

Authoritative: no.

Must exclude:

- hierarchy truth
- raw document truth
- structured document authority

Retrieval storage is a derived persistence surface. FAISS artifacts, records, and fingerprint metadata support search and cache reuse; they must not override structured hierarchy or raw document ownership.

### Artifact Storage Contract

Owner: current owner `document_structure`; future split may need confirmation.

Minimum capabilities:

- Persist interaction outputs.
- Load interaction outputs.
- Bind artifacts to validated hierarchy targets.
- Detect stale targets conceptually.

Authoritative: not hierarchy authority.

Must exclude:

- hierarchy identity creation
- hierarchy mutation
- parser authority
- task-layout projection ownership

Artifacts are interaction outputs. They may be persisted against document, chapter, section, task-unit, or future content-block targets only after hierarchy-aware target validation. Artifact payloads must never create, rename, or re-own hierarchy identity.

### Raw Document Storage Contract

Owner: `doc_loaders`; broader upload/retention ownership needs confirmation.

Minimum capabilities:

- Preserve user-scoped canonical raw input.
- Load raw content.
- Preserve upload identity.
- Support deletion/retention policy conceptually.

Authoritative: yes, as canonical user document.

Must exclude:

- cross-user sharing
- parser output
- retrieval index semantics
- implicit DB candidacy decision

Raw documents are a separate storage track because uploaded files carry user ownership, copyright, deletion, retention, and upload identity concerns. Future DB migration planning must not imply cross-user sharing or automatic raw-file DB adoption.

### Runtime Bundle / Cache Contract

Owner: `bundle_provider` / `bundle_factory`

Minimum capabilities:

- Cache loaded runtime bundle.
- Invalidate cache conceptually.
- Rebuild from persisted retrieval/profile artifacts.

Authoritative: no.

Must exclude:

- DB migration target
- hierarchy truth
- permanent persistence semantics

Runtime bundles are cache-only. They may coordinate loaded retrieval/profile artifacts for runtime use, but they should remain outside DB migration except as rebuildable consumers of persisted artifacts.

### Config Backend Policy Contract

Owner: `config`

Minimum capabilities:

- Select backend policy conceptually.
- Hold rollout flags conceptually.
- Support coexistence policy conceptually.

Authoritative: no.

Must exclude:

- domain semantics
- hierarchy contract
- parser semantics
- profile/retrieval/raw document meaning

`config` may eventually govern file/DB backend selection and rollout. It must not define what structured documents, profiles, retrieval artifacts, raw documents, or runtime caches mean.

## Contract Boundary Matrix

| Domain | Owner | Authoritative? | Rebuildable? | Backend-Selectable? | DB Candidate? | Current Storage | Future Contract Boundary |
|---|---|---:|---:|---:|---:|---|---|
| Structured document | `document_structure` | Yes | Yes, from canonical raw document and parser mode | Yes | Yes | `data/structured/*.structured.json` | Hierarchy-first `StructuredDocument` persistence contract |
| Profile | `profile` | No | Yes | Yes | Yes | `data/faiss/<namespace>/profile.json` | Advisory profile snapshot persistence contract |
| Retrieval index/records | `retrieval` | No | Yes | Yes | Partial | `data/faiss/<namespace>/index.faiss`, `records.json`, `meta.json` | Derived retrieval artifact and freshness contract |
| Artifacts | `document_structure`; future split needs confirmation | Not hierarchy authority | Partially | Yes | Yes | Embedded through structured artifact repository | Interaction output persistence against validated hierarchy targets |
| Raw documents | `doc_loaders`; broader ownership needs confirmation | Yes, canonical user document | No, unless re-uploaded | Needs confirmation | Needs confirmation | `data/raw` uploaded files | User-scoped raw input storage contract |
| Runtime bundle/cache | `bundle_provider` / `bundle_factory` | No | Yes | No | No | In-memory bundle cache | Runtime cache lifecycle contract, outside DB migration |
| Backend policy | `config` | No | N/A | N/A | N/A | path/namespace config and DI wiring | Backend selection, rollout, and coexistence policy only |

## Migration Track Separation

Future migration must remain track-separated:

- Structured persistence
- Profile persistence
- Artifact persistence
- Retrieval persistence
- Raw document storage
- Runtime cache remains outside DB migration

There should be no big-bang migration. Each track needs its own readiness checks, validation rules, rollback posture, and retirement gate. `data/` retirement must be gradual and gated by validation, not a breaking deletion.

## Explicit Non-Goals

- No schema.
- No table design.
- No JSONB layout.
- No ORM.
- No repository interface implementation.
- No migration script.
- No dual-write.
- No read-path switch.
- No API change.
- No runtime behavior change.
- No dependency change.
- No backend implementation.
- No PostgreSQL selection.
- No relational vs JSONB decision.
