# Phase 1 Derived-Row Freshness Validation Plan

## Purpose

This document defines future application-level validation behavior for stale or invalid derived rows.

It is documentation-only. It does not create Python code, SQL DDL, ORM models, repository interfaces, migrations, fixtures, tests, runtime behavior, API changes, or backend selection.

## Validation Principle

Derived rows store `source_structure_version` as provenance and freshness metadata.

The DB may store and index the value, but application code owns validation against `documents.current_structure_version`.

Derived rows include:

- content blocks
- artifacts
- optional profile snapshots generated after structure creation
- optional validation/parity snapshots

## Candidate Validation States

Future application behavior may classify derived rows as:

- `current`
- `stale_structure_version`
- `missing_target`
- `source_hash_mismatch`
- `malformed_payload`
- `unvalidated`

The exact enum is future implementation detail. The boundary is that stale rows must not silently drive current user-facing behavior.

## Write Validation

Before writing content blocks or artifacts, application code should confirm:

- document exists
- target belongs to document
- target exists in current hierarchy
- `source_structure_version` equals `documents.current_structure_version`
- payload shape is valid for the requested type

## Read Validation

Read paths should either:

- filter to current derived rows, or
- return explicit freshness metadata so callers can fail fast or suppress stale output

Hard reparse physically deletes document-level content blocks and artifacts, so stale derived rows should be exceptional rather than expected.

## Guardrails

This plan does not introduce:

- DB-enforced lifecycle authority for derived rows
- content blocks as hierarchy truth
- artifacts as hierarchy truth
- profile authority
- runtime fallback behavior
- SQL DDL, ORM, repository code, migrations, fixtures, tests, or API changes
