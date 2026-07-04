# Phase 1 Migration and Evaluation Fixtures Plan

## Purpose

This document defines a future fixture plan for DB migration and evaluation work without production-migrating existing JSON identity.

It is documentation-only. It does not create fixtures, Python code, SQL DDL, ORM models, repository interfaces, migrations, tests, runtime behavior, API changes, or backend selection.

## Fixture Purpose

Future fixtures should prove that DB-backed persistence can preserve accepted hierarchy semantics from existing file-backed `StructuredDocument` outputs.

Fixtures should evaluate:

- document lifecycle metadata
- current hierarchy shape
- chapter ordering
- section ordering
- task-unit ordering
- optional content-block linkage
- optional artifact target linkage
- parse event provenance
- optional JSONB parity snapshot behavior

## Identity Boundary

Existing JSON `unit_id` values may be fixture reference evidence only.

They must not become:

- production primary keys
- public IDs
- durable domain IDs
- cross-system references
- required DB import identities

Future fixture assertions should compare semantic hierarchy shape and ordering, not require production dependency on current Python-generated IDs.

## Candidate Fixture Classes

Candidate fixture groups:

- simple document with chapters, sections, and task units
- document with no task units where valid
- document with dense hierarchy
- document with advisory profile metadata
- document with lazy content-block materialization
- document with artifact targets
- document before and after hard reparse
- malformed fixture used only for validation-failure tests

## Guardrails

This plan does not introduce:

- production migration from current JSON identity
- runtime fallback to JSON files
- root `sections[]`
- `structure_nodes`
- flat `task_units` as primary runtime truth
- SQL DDL, ORM, repository code, migration scripts, fixtures, tests, or runtime switch
