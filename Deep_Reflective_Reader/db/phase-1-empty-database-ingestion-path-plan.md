# Phase 1 Empty-Database and New-Document Ingestion Path Plan

## Purpose

This document defines the future empty-database and new-document ingestion path before any existing JSON production migration.

It is documentation-only. It does not create Python code, SQL DDL, ORM models, repository interfaces, migrations, fixtures, tests, runtime behavior, API changes, or backend selection.

## Rollout Principle

The first DB-backed path should support new documents in an empty database before migrating existing structured JSON outputs.

This avoids treating current file identities as production DB identity and keeps DB rollout scoped to accepted new ingestion behavior.

## Candidate Ingestion Flow

1. Store raw source bytes through the existing file-backed/object-backed raw source path.
2. Create `documents` lifecycle row with required `namespace` and `document_name`.
3. Persist raw-source metadata linked to the document.
4. Parse and validate accepted hierarchy through `document_structure`.
5. Persist current hierarchy with `current_structure_version = 1`.
6. Append initial parse event.
7. Persist optional advisory profile snapshot.
8. Optionally persist validation/parity snapshot if enabled by a separate validation track.

If hierarchy creation fails, the document should not become a partially accepted structured document.

## Migration Boundary

Existing JSON outputs remain file-backed compatibility and evaluation material until separate migration readiness work defines a safe import path.

Phase 1 should not require:

- importing existing JSON into production DB
- preserving current Python-generated `unit_id` as production identity
- dual-writing old and new structured outputs by default
- runtime read fallback from DB to JSON snapshot

## Guardrails

This plan does not introduce:

- production JSON identity migration
- SQL DDL
- ORM implementation
- repository code
- migration script
- runtime backend switch
- tests or fixtures
