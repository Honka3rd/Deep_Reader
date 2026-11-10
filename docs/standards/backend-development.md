# Backend Development Standards

Applies to `Deep_Reflective_Reader`. Read with [AGENTS.md](../../AGENTS.md)
and the owning module's design/checklist. These are implementation requirements,
not a claim that all existing code already complies. Report conflicts explicitly.

## Cohesion and Boundaries

- Keep each module focused on one responsibility and reason to change. Keep domain policy with its owner; avoid growing coordinators into collections of unrelated features.
- Routes and schemas handle transport; `app` and dedicated application modules such as `learning` orchestrate their owned use cases; domain modules plan and validate; repositories own persistence. Wire dependencies through existing configuration/injection patterns.
- Depend on small public contracts, not another module's private methods, SQL, or provider internals. Expose only the operations and response fields callers need.
- Reuse established helpers and typed models. Prefer explicit, simple code; add abstractions only for demonstrated duplication or a real boundary. Keep unrelated refactors out of feature changes.

## Project Invariants

- Preserve `chapters[].sections[].task_units[]` as hierarchy truth and hierarchy-aware artifact ownership. Do not restore root `sections[]` or `structure_nodes` as authority.
- Reads, diagnostics and task-layout projections must not silently mutate persistence or regenerate profiles. Reuse the existing profile outside authorized initial-load/reparse workflows.
- Downstream pipelines consume loader contracts and extraction identity/provenance, not PDF/TXT-specific branching. Keep language-specific parsing/OCR rules inside their owning handlers.
- Treat LLM output as an untrusted proposal. The backend validates plans, resolves boundaries, assigns IDs and controls execution. Preserve canonical source text; apply only explicitly permitted transformations.

## Database Atomicity and Concurrency

- Define the transaction boundary around one invariant-preserving business write. Related content/hierarchy changes, versions, invalidations and required audit/operation receipts commit together or roll back together. Report persistence success only after commit.
- Pass the same connection/transaction through participating repository helpers. Helpers must not independently commit or open a second write transaction inside that operation.
- Keep transactions short: perform LLM, OCR and remote calls outside them. Before committing, revalidate source identity and relevant revisions against current persisted state.
- Atomicity alone does not prevent concurrent conflicts. Use conditional version updates or appropriately scoped locks, database constraints, and consistent lock ordering. Reject stale writes; do not silently overwrite newer work.
- Use parameterized SQL and explicit namespace/document scoping. Apply schema changes through reviewed migrations, not incidental read-time repair.
- Make retryable mutations idempotent. Reconcile unknown commit outcomes using durable operation state before replay; retry known transient database failures at the whole-transaction boundary, without blindly repeating external side effects.
- A database transaction does not include filesystem or remote-service writes. Where both are involved, define staging, recovery or compensation explicitly; do not claim cross-resource atomicity.

## Reliability and Verification

- Validate inputs at boundaries and use structured parsers for structured data. Distinguish missing data, invalid input, conflict, provider failure and unknown outcome; never turn arbitrary exceptions into empty data or success.
- Keep queries and response payloads bounded; avoid N+1 access and unnecessary reloads. Log operation/stage, duration, counts and outcome so slow paths can be located. Never log keys, authorization tokens or full private books.
- Test domain rules and failure paths, not just happy paths. Persistence changes need real-database rollback, concurrency and duplicate-request coverage where applicable; mocks alone do not establish atomicity.
- Parsing/OCR/LLM changes require the designated backed-up real-book cases as well as focused regressions. Preserve source backups, report skipped/unavailable checks, and never equate source changes with deployed behavior.
- Keep edits scoped, synchronize governed design/checklist changes when required, and report verification evidence. Store development secrets in ignored `.env.local`, inject them at startup, and keep templates secret-free.

## Basis

Project boundaries: [High-Level Design](../../Deep_Reflective_Reader/high-level-design.md),
[Application Design](../../Deep_Reflective_Reader/app/module-detailed-design.md),
[Database Design](../../Deep_Reflective_Reader/db/module-detailed-design.md).
Transaction semantics: PostgreSQL [Transactions](https://www.postgresql.org/docs/current/tutorial-transactions.html)
and [Transaction Isolation](https://www.postgresql.org/docs/current/transaction-iso.html).
