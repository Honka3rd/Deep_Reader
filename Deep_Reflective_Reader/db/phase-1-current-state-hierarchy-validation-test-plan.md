# Phase 1 Current-State-Only Hierarchy Validation Test Plan

## Purpose

This document defines future validation test planning for current-state-only hierarchy replacement.

It is documentation-only. It does not create tests, fixtures, Python code, SQL DDL, ORM models, repository interfaces, migrations, runtime behavior, API changes, or backend selection.

## Test Intent

Future tests should prove that DB-backed hierarchy persistence preserves the current-state-only model.

The tests should verify:

- initial parse creates `current_structure_version = 1`
- current hierarchy reads through `Document -> Chapter -> Section -> TaskUnit`
- hierarchy rows do not carry row-level structure versions
- hard reparse replaces current hierarchy
- hard reparse advances document-level version
- hard reparse deletes content blocks and artifacts
- parse event is appended as provenance only
- stale derived rows do not silently remain active

## Candidate Test Groups

Candidate groups:

- initial parse success
- initial parse failure
- current hierarchy read
- hard reparse success
- hard reparse candidate validation failure
- hard reparse transaction failure
- derived row deletion verification
- no JSONB fallback verification
- no flat task-unit primary truth verification

## Non-Goals

These future tests should not require production migration from current JSON identity.

They should not treat existing Python-generated `unit_id` as DB identity.

## Guardrails

This plan does not introduce:

- actual tests or fixtures
- SQL DDL
- ORM implementation
- repository code
- migration scripts
- runtime switch
- root `sections[]`
- `structure_nodes`
