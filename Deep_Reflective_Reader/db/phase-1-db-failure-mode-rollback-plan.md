# Phase 1 DB Failure Mode and Rollback Plan

## Purpose

This document defines future rollback and failure-mode planning for failed initial parse, failed hard reparse, and failed derived-resource cleanup.

It is documentation-only. It does not create Python code, SQL DDL, ORM models, repository interfaces, migrations, fixtures, tests, runtime behavior, API changes, or backend selection.

## Initial Parse Failure

If initial parse fails before accepted hierarchy exists:

- no accepted hierarchy should be exposed
- `current_structure_version` should not be advanced to `1`
- parse event should not claim successful initial parse
- raw-source metadata may remain only if source storage succeeded and lifecycle policy allows retry

The document must not appear as a complete structured document.

## Hard Reparse Failure

If candidate validation fails:

- keep existing hierarchy
- keep existing `current_structure_version`
- keep existing content blocks and artifacts
- do not append hard-reparse parse event

If transaction execution fails:

- rollback hierarchy replacement
- rollback structure version advance
- rollback derived-resource deletion
- rollback parse event append
- surface explicit failure to the caller

## Derived Cleanup Failure

Hard reparse derived cleanup must be part of the same transaction as hierarchy replacement where possible.

If cleanup cannot be guaranteed, future implementation must fail the hard reparse rather than commit a new hierarchy with stale document-level content blocks or artifacts.

## Rollback Evidence

Future implementation should return or log:

- failure stage
- document identity
- previous structure version
- candidate version when applicable
- cleanup counts when available
- parse event write status

## Guardrails

This plan does not introduce:

- partial hard reparse commit
- stale derived-resource survival after accepted hard reparse
- immutable hierarchy history
- event sourcing
- SQL DDL, ORM, repository code, migrations, fixtures, tests, or runtime switch
