# Phase 1 Hard Reparse Transaction Plan

## Purpose

This document defines the future transaction-level implementation plan for hard reparse.

It is documentation-only. It does not create Python code, SQL DDL, ORM models, repository interfaces, migrations, fixtures, tests, runtime behavior, API changes, or backend selection.

## Transaction Boundary

Hard reparse is user-triggered and rare. It must replace accepted current hierarchy only after a candidate hierarchy has been validated outside the destructive transaction.

The transaction should:

1. lock or otherwise protect the target document lifecycle row
2. confirm current structure version
3. delete current hierarchy rows or replace them according to the physical schema plan
4. insert accepted replacement hierarchy rows
5. advance `documents.current_structure_version`
6. physically delete document-level content blocks
7. physically delete document-level artifacts
8. append one hard-reparse parse event
9. commit atomically

## Validation Boundary

Candidate validation happens before destructive persistence begins.

Validation failure must not:

- delete current hierarchy
- advance `current_structure_version`
- delete content blocks
- delete artifacts
- append a hard-reparse parse event

## Result Shape

Future implementation should return:

- document identity
- previous structure version
- new structure version
- deleted content-block count
- deleted artifact count
- parse event identity

## Guardrails

This plan does not introduce:

- staging hierarchy persistence
- immutable hierarchy history
- row-level hierarchy structure versions
- event sourcing
- JSONB fallback authority
- SQL DDL, ORM, repository code, migration scripts, fixtures, tests, or runtime switch
