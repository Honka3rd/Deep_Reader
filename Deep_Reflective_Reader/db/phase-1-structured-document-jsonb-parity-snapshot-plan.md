# Phase 1 StructuredDocument JSONB Parity Snapshot Validation Plan

## Purpose

This document defines future planning for an optional `StructuredDocument` JSONB parity snapshot.

It is documentation-only. It does not create Python code, SQL DDL, ORM models, repository interfaces, migrations, fixtures, tests, runtime behavior, API changes, JSONB persistence, or backend selection.

## Snapshot Boundary

The optional snapshot is validation, parity, and debug evidence only.

It must not become:

- runtime hierarchy source
- fallback source
- parser authority
- dual authority with relational hierarchy
- migration shortcut around relational hierarchy validation

Relational current hierarchy plus `documents.current_structure_version` remain authoritative.

## Candidate Snapshot Use

Future validation may persist a full `StructuredDocument` payload after accepted hierarchy creation to compare:

- document metadata
- chapter ordering
- section ordering
- task-unit ordering
- title/text fields
- reference/import IDs as evidence only

Snapshot conflict should be treated as validation failure, not fallback behavior.

## Candidate Lifecycle

Snapshot rows, if implemented, should be linked to:

- `document_id`
- `source_structure_version`
- generated timestamp
- snapshot schema version
- validation metadata

Snapshots should be deleted with the document. Hard reparse behavior should be defined by future validation policy, but snapshots must not preserve immutable hierarchy history for runtime use.

## Guardrails

This plan does not introduce:

- JSONB runtime fallback
- JSONB hierarchy authority
- root `sections[]`
- `structure_nodes`
- production migration from current JSON identity
- SQL DDL, ORM, repository code, migrations, fixtures, tests, or runtime switch
