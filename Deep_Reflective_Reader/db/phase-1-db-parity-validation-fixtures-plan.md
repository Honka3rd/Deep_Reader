# Phase 1 DB Parity Validation Fixtures Plan

## Purpose

This document defines future DB parity validation fixture planning against file-backed `StructuredDocument` golden outputs.

It is documentation-only. It does not create fixtures, tests, Python code, SQL DDL, ORM models, repository interfaces, migrations, runtime behavior, API changes, or backend selection.

## Parity Goal

Future parity fixtures should compare DB-backed relational hierarchy output with accepted file-backed `StructuredDocument` golden outputs.

Parity should focus on semantic structure:

- document metadata relevant to structure
- chapter order and titles
- section order and titles
- task-unit order and content references
- valid no-task-unit hierarchy cases
- optional advisory profile metadata as metadata only

## Identity Boundary

Current JSON IDs may be retained as fixture reference evidence only.

They must not become:

- DB primary keys
- production identity
- public IDs
- required API identity
- durable cross-system references

## Candidate Fixture Workflow

1. Select representative file-backed golden outputs.
2. Load them through a migration/evaluation-only fixture harness.
3. Persist equivalent relational hierarchy into an isolated DB test environment.
4. Read back current hierarchy through DB storage.
5. Compare semantic shape and ordering.
6. Report parity failures without falling back to JSONB snapshot authority.

## Guardrails

This plan does not introduce:

- actual fixtures
- production migration from JSON identity
- runtime DB/file fallback
- JSONB authority
- SQL DDL, ORM, repository code, migrations, tests, or API changes
