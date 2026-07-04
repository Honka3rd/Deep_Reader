# Phase 1 Artifact Relational Persistence Plan

## Purpose

This document defines future relational persistence planning for common artifacts.

It is documentation-only. It does not create Python code, SQL DDL, ORM models, repository interfaces, migrations, fixtures, tests, runtime behavior, API changes, artifact generators, or backend selection.

## Ownership Boundary

Artifacts are derived interaction outputs.

Phase 1 uses one logical `Artifact` entity with:

- `artifact_type`
- hierarchy/content target metadata
- type-specific payload
- lifecycle and provenance metadata

Artifacts must not become hierarchy authority or artifact availability authority without hierarchy-aware target validation.

## Candidate Persistence Semantics

Artifact rows should include:

- `document_id`
- validated target type
- target DB identity when applicable
- artifact type
- type-specific payload
- `source_structure_version`
- source hash or span metadata when applicable
- created timestamp
- invalidation behavior tied to hard reparse

## Candidate Write Flow

1. Caller resolves artifact target through hierarchy-aware validation.
2. Caller prepares artifact payload and provenance metadata.
3. Application-level validation confirms source structure freshness.
4. Storage adapter saves one common artifact row.

The adapter must not accept unresolved raw target IDs as sufficient target authority.

## Candidate Read Flow

Artifacts may be listed by document, target, type, or recency.

Reads should return detached DTOs and should not reconstruct hierarchy from artifact rows.

## Guardrails

This plan does not introduce:

- category-specific artifact tables in Phase 1
- non-hierarchy-aware artifact writes
- artifact hierarchy authority
- retrieval authority
- SQL DDL, ORM, repository code, migrations, fixtures, tests, or runtime switch
