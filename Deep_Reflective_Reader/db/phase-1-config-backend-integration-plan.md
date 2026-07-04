# Phase 1 Backend Configuration Integration Plan

## Purpose

This document defines future backend configuration integration planning with `config/`.

It is documentation-only. It does not create Python code, config classes, SQL DDL, ORM models, repository interfaces, migrations, fixtures, tests, runtime behavior, API changes, or backend selection.

## Ownership Boundary

`config/` owns backend selection and rollout policy.

Domain modules own semantics:

- `document_structure` owns hierarchy semantics
- `profile` owns advisory profile semantics
- `shared` owns shared DTO vocabulary
- `db` owns DB persistence design memory

Configuration must not redefine hierarchy or persistence semantics.

## Candidate Configuration Concerns

Future configuration may need:

- storage backend policy
- DB connection settings
- file-backed coexistence settings
- read path selection
- write path selection
- migration/evaluation mode
- parity snapshot enablement
- rollout guardrails
- failure-mode policy

## Rollout Boundary

Runtime read/write switch is a separate future task.

This plan only says that `config/` should own the future switch surface. It does not enable DB reads, DB writes, dual writes, or fallback behavior.

## Guardrails

Configuration must not:

- make DB schema parser authority
- make optional JSONB snapshot runtime authority
- enable hidden dual-write behavior
- silently fallback between file and DB backends
- treat current Python-generated `unit_id` as production identity

This plan does not introduce SQL DDL, ORM, repository code, migrations, fixtures, tests, runtime switch, or API changes.
