# Storage Contract Inventory

## Purpose

This document inventories the current storage contracts in `Deep_Reflective_Reader` before any DB-centric migration design. It identifies persistence surfaces, ownership boundaries, source-of-truth status, rebuildability, and migration risks.

This is documentation-only preparation. It does not define a database schema, introduce an ORM, create dual-write behavior, switch runtime reads, or change any existing file-backed storage path.

## Storage Classification Matrix

| Persisted Artifact | Owner Module | Persistence Format | Authoritative? | Rebuildable? | Future DB Candidate? | Migration Difficulty |
|---|---|---|---:|---:|---:|---|
| Structured document | `document_structure` | `data/structured/*.structured.json` | Yes | Yes, from canonical raw document and parser mode | Yes | High |
| Profile | `profile` | `data/faiss/<namespace>/profile.json` | No, advisory metadata snapshot only | Yes, from raw text and language/profile builders | Yes | Medium |
| Retrieval index | `retrieval` | `data/faiss/<namespace>/index.faiss` | No | Yes, from raw text + embedding config | Partial | Medium |
| Retrieval records | `retrieval` | `data/faiss/<namespace>/records.json` | No | Yes, from raw text chunking/index build | Partial | Medium |
| Fingerprint metadata | `fingerprint_handler` with `retrieval` / `bundle_factory` consumers | `data/faiss/<namespace>/meta.json` | No, cache validation metadata | Yes, from raw text and index config | Partial | Low |
| Task artifacts | `document_structure` | Embedded in structured JSON via artifact repository | No | Partially; generated outputs may require task rerun | Yes | Medium |
| Raw documents | `doc_loaders` | Uploaded files under `data/raw` | Yes, canonical user document | No, unless the user re-uploads | Needs Confirmation | Separate Track |
| Runtime bundles | `bundle_provider` / `bundle_factory` | In-memory `OrderedDict` cache over loaded `FaissIndexBundle` | No | Yes, from persisted retrieval/profile artifacts | No | None |

## Source-of-Truth Matrix

| Domain | Current Source of Truth | Authority Classification | Notes |
|---|---|---|---|
| Hierarchy truth | `StructuredDocument.chapters[].sections[].task_units[]` | Authoritative | Only hierarchy document structure is authoritative. |
| Profile truth | `DocumentProfile` persisted snapshot | Advisory snapshot only | `parser_metadata` and `post_structure_metadata` must not become parser authority. |
| Retrieval truth | FAISS index + node records + fingerprint metadata | Not authoritative | Retrieval artifacts are rebuildable cache/index support derived from raw text. |
| Artifact truth | Summary/quiz/task artifact payloads in repository-managed structured document | Interaction output, not hierarchy truth | Artifacts must not create, rename, or re-own hierarchy identity. |
| Raw document truth | User-uploaded source file | Canonical user document | Raw file ownership and copyright boundaries remain user-scoped. |
| Runtime bundle truth | In-memory runtime bundle cache | Not authoritative | Runtime bundles are cache-only and rebuildable from persisted retrieval/profile artifacts. |

Only hierarchy document structure is authoritative for runtime navigation and hierarchy identity. Profile, retrieval, artifacts, and runtime bundles must not override the hierarchy contract.

## Future Migration Tracks

### Track A: Structured Persistence

Structured document migration is the highest-priority DB candidate because it owns the hierarchy contract. A future DB representation must preserve `chapters[].sections[].task_units[]`, hierarchy-first lookup, fail-fast runtime behavior, and explicit migration-only legacy handling.

### Track B: Profile Persistence

Profile migration should treat profile records as advisory snapshots. DB-backed profile storage must not make `parser_metadata`, `post_structure_metadata`, LLM classification, or diagnostics into parser authority.

### Track C: Artifact Persistence

Artifact migration should remain separate from hierarchy migration. Summary, quiz, task-unit, chapter, and document-level artifacts are interaction outputs and must continue to target resolved hierarchy identities without becoming hierarchy truth.

### Track D: Retrieval Persistence

Retrieval migration is partial because FAISS index files, node records, and fingerprint metadata have different persistence properties. Retrieval data is rebuildable and not authoritative; future DB planning should avoid treating retrieval records as hierarchy or raw-document truth.

### Track E: Raw Document Storage

Raw document storage is a separate track because uploaded files are canonical user documents and carry ownership/copyright boundaries. Future DB candidacy needs maintainer confirmation and must not imply cross-user document sharing.

These tracks are intentionally independent. There should be no big-bang migration.

## Non Goals

- No schema design.
- No ORM.
- No DB implementation.
- No dual write.
- No runtime switch.
- No API change.
- No storage abstraction implementation.
- No file migration.
- No retirement of `data/` storage.
