# Phase 1 Parse Event Persistence Plan

## Purpose

This document defines future parse event persistence implementation planning.

It is documentation-only. It does not create Python code, SQL DDL, ORM models, repository interfaces, migrations, fixtures, tests, runtime behavior, API changes, or backend selection.

## Event Scope

Parse events are minimal document-scoped provenance.

Required event types:

- `initial_parse`
- `hard_reparse`

They explain how accepted structure versions were created. They are not event sourcing, not hierarchy history, and not compliance audit logs.

## Candidate Metadata

Candidate fields:

- `document_id`
- event type
- previous structure version when applicable
- new structure version
- parser or preparation mode when available
- source checksum or source metadata when available
- invalidated content-block count for hard reparse
- invalidated artifact count for hard reparse
- created timestamp
- optional diagnostic metadata

## Write Rules

Initial parse appends one event after accepted structure version `1` is created.

Hard reparse appends one event inside the same transaction that replaces hierarchy, advances structure version, and deletes derived rows.

Parse event writes should not decide current structure version. `documents.current_structure_version` remains authoritative.

## Retention

Parse events are retained for the lifetime of the document and deleted when the document is deleted.

Phase 1 does not introduce archival, TTL, or independent audit retention.

## Guardrails

This plan does not introduce:

- event-sourced hierarchy reconstruction
- immutable hierarchy history
- compliance audit retention
- parse events as version authority
- SQL DDL, ORM, repository code, migrations, fixtures, tests, or runtime switch
