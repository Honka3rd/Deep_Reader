# Phase 1 Advisory Document Profile Persistence Plan

## Purpose

This document defines the Phase 1 persistence plan for advisory document profile snapshots.

It is documentation-only. It does not create Python code, SQL DDL, ORM models, repository interfaces, migrations, fixtures, tests, runtime behavior, API changes, profile parser changes, or backend selection.

## Ownership Boundary

The `profile` module owns profile semantics. The DB stores a document-scoped snapshot for persistence and retrieval.

Profile data is advisory metadata only. It must not:

- create, mutate, delete, or block chapters
- create, mutate, delete, or block sections
- create, mutate, delete, or block task units
- decide parser authority
- decide artifact availability
- decide retrieval authority
- write diagnostics back into hierarchy during read paths

## Candidate Fields

Candidate persisted profile metadata:

- `document_id`
- profile payload or profile snapshot
- profile version or schema version
- generated timestamp
- source parser or preparation mode when available
- language or script metadata when available
- title, author, or source metadata when available
- `source_structure_version` when generated after structure creation
- advisory diagnostics when currently produced

## Candidate Write Flow

1. Profile generation runs inside document preparation or an explicit profile refresh path.
2. The profile result is normalized into a detached profile snapshot DTO.
3. The profile storage adapter upserts the snapshot for `document_id`.
4. If `source_structure_version` is present, application code treats it as freshness metadata only.

Profile write failure should not mutate accepted hierarchy. Whether it blocks document preparation remains a preparation policy decision, not a DB schema decision.

## Candidate Read Flow

Profile reads return advisory snapshot DTOs.

Consumers may use the snapshot for display, recommendations, or diagnostics, but must not use it as hierarchy truth or parser authority.

## Guardrails

This plan does not introduce:

- profile authority
- diagnostics profile write-back
- metadata or LLM classification as parser authority
- profile-driven hierarchy mutation
- profile-driven artifact availability
- SQL DDL, ORM, repository code, migrations, fixtures, tests, or runtime switch
