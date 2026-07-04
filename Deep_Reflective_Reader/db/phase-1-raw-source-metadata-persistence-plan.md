# Phase 1 Raw-Source Metadata Persistence Plan

## Purpose

This document defines the Phase 1 persistence plan for raw-source metadata while raw document bytes remain file-backed or object-backed.

It is documentation-only. It does not create Python code, SQL DDL, ORM models, repository interfaces, migrations, fixtures, tests, runtime behavior, API changes, object storage integration, or backend selection.

## Ownership Boundary

Raw uploaded documents remain canonical user-owned source files outside the DB for Phase 1.

The DB stores metadata only:

- `document_id`
- source location, file path, or object key
- original filename
- MIME type
- file size
- checksum or fingerprint when available
- uploaded timestamp
- ownership scope when applicable
- source parser or ingestion mode when available

The metadata row explains where the raw source lives. It does not contain raw bytes and does not become parser authority.

## Candidate Write Flow

1. Raw source is received and stored by the existing file-backed/object-backed source path.
2. The source storage layer returns stable metadata for the stored source.
3. A future raw-source metadata storage adapter persists metadata linked to `documents.id`.
4. Structured parsing uses the stored source through existing preparation boundaries.

The DB write should happen only after raw-source storage succeeds. A failed metadata write must not imply that raw bytes were deleted unless a separate cleanup policy explicitly owns that behavior.

## Candidate Read Flow

Raw-source metadata reads may support:

- document detail display
- ingestion provenance
- reparse source lookup
- deletion coordination
- validation diagnostics

Reads must return detached metadata DTOs, not ORM rows.

## Delete and Retention

Raw-source metadata should be deleted when the document is deleted.

Raw byte deletion remains a separate storage concern. Phase 1 should not define DB retention as the authority for file/object deletion until a later source-storage lifecycle policy exists.

## Guardrails

This plan does not introduce:

- raw-byte DB storage
- source file parsing inside DB adapters
- DB schema as parser authority
- cross-user source sharing semantics
- public document identity requirements
- runtime backend switch
- SQL DDL, ORM, repository code, migrations, fixtures, or tests
