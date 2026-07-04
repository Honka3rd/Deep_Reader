# Phase 1 Content-Block Relational Persistence Plan

## Purpose

This document defines future relational persistence planning for lazy materialized content blocks.

It is documentation-only. It does not create Python code, SQL DDL, ORM models, repository interfaces, migrations, fixtures, tests, runtime behavior, API changes, or backend selection.

## Ownership Boundary

Content blocks are derived resources linked to accepted task units.

They must not become:

- hierarchy truth
- parser authority
- retrieval authority by themselves
- a reason to mutate chapters, sections, or task units

`document_structure` owns hierarchy semantics. `shared` owns common content-block vocabulary.

## Candidate Persistence Semantics

Content blocks should be:

- lazy materialized
- linked to `document_id`
- linked to task-unit DB identity
- ordered within the task unit when order matters
- tagged with `source_structure_version`
- optionally tagged with source span or source hash metadata
- physically deleted on hard reparse at document scope

## Candidate Write Flow

1. Caller resolves current task-unit target through hierarchy-aware lookup.
2. Caller prepares content-block DTOs.
3. Application-level validation confirms `source_structure_version` matches current document structure version.
4. Storage adapter replaces or writes blocks for the task unit.

The DB may enforce foreign keys, but freshness validation remains application-level behavior.

## Candidate Read Flow

Content-block reads support task-unit detail views, downstream evidence, and artifact generation.

Reads should return detached DTOs and should not imply that task-unit hierarchy is stored under content blocks.

## Guardrails

This plan does not introduce:

- content blocks as hierarchy authority
- hidden task-layout persistence mutation
- row-level hierarchy versioning
- automatic resegmentation on read
- SQL DDL, ORM, repository code, migrations, fixtures, tests, or runtime switch
