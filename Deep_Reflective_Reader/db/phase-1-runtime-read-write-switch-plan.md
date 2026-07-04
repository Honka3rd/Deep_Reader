# Phase 1 Runtime Read/Write Switch Plan

## Purpose

This document defines future planning for a runtime read/write switch behind explicit backend policy.

It is documentation-only. It does not create Python code, config implementation, SQL DDL, ORM models, repository interfaces, migrations, fixtures, tests, runtime behavior, API changes, or backend selection.

## Switch Principle

DB-backed runtime reads and writes must not be enabled implicitly.

Future switch behavior should be explicit, configured, observable, and reversible.

## Candidate Modes

Candidate modes for future discussion:

- file read / file write baseline
- DB evaluation write only
- DB read in isolated evaluation
- DB read/write for new documents only
- rollback to file-backed baseline

Mode names are placeholders. Future implementation may choose different names if authority boundaries remain intact.

## Required Preconditions

Before enabling any DB runtime path, the repository should have:

- executable migrations
- ORM mappings or equivalent storage adapter implementation
- storage interfaces/adapters
- backend configuration implementation
- parity validation fixtures
- current-state hierarchy replacement tests
- hard reparse transaction tests
- failure-mode behavior
- rollback procedure

## Fallback Boundary

Fallback behavior must be explicit. Silent fallback from relational hierarchy to JSONB snapshot or file-backed structured output should not be introduced as normal runtime behavior.

## Guardrails

This plan does not introduce:

- runtime switch implementation
- hidden dual write
- silent fallback
- JSONB runtime authority
- DB schema as parser authority
- SQL DDL, ORM, repository code, migrations, fixtures, tests, or API changes
