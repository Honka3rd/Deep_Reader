# StructuredDocument JSONB Evaluation Readiness Audit

## Purpose

This document audits whether repository assets are ready to support a future Phase 1 `StructuredDocument` JSONB-first evaluation.

This is not the evaluation itself. It does not create schema, connect PostgreSQL, implement JSONB persistence, introduce repositories, create migration scripts, define dual-write behavior, switch runtime read paths, or change runtime behavior.

The audit treats existing structured files as read-only evidence. The root-level `data/structured/` path was not present in this checkout; the current structured repository is `Deep_Reflective_Reader/data/structured/`.

## Repository Memory Basis

Phase 1 evaluation remains bounded by:

- `StructuredDocument` is the validation authority.
- Hierarchy truth remains `chapters[].sections[].task_units[]`.
- PostgreSQL JSONB is an evaluation assumption, not a final backend commitment.
- Existing file-backed structured storage remains the rollback source and compatibility baseline.
- Content blocks, artifacts, profile, retrieval, raw document governance, production read-path switching, and backend cutover are out of Phase 1 scope.

## Structured Document Inventory

Inventory source: `Deep_Reflective_Reader/data/structured/*.structured.json`.

Non-document file observed: `.DS_Store`; ignored for this inventory.

| Document | File | Approx File Size | Language | Chapters | Sections | Task Units | Approx Hierarchy Size | Unusual Characteristics |
|---|---|---:|---|---:|---:|---:|---:|---|
| `APPLE` | `APPLE.structured.json` | 752 KB | `en` | 11 | 11 | 0 | 22 hierarchy nodes excluding document | Valid `StructuredDocument` reference file with chapter/section hierarchy and no task units; useful hierarchy evaluation material. Zero task units must not be interpreted as legacy schema, production migration constraint, or DB identity requirement. |
| `Madame Bovary` | `Madame Bovary.structured.json` | 2.2 MB | `en` | 35 | 35 | 500 | 570 hierarchy nodes excluding document | Largest task-unit set; one section per chapter; repeated local chapter titles across book parts; strong content-preservation candidate. |
| `中式思维` | `中式思维.structured.json` | 123 KB | `zh` | 5 | 5 | 14 | 24 hierarchy nodes excluding document | Chinese document with front matter and back matter; back matter has 10 task units; compact but content-heavy per unit. |
| `许三观卖血记` | `许三观卖血记.structured.json` | 1.2 MB | `zh` | 30 | 34 | 98 | 162 hierarchy nodes excluding document | Chinese long-form narrative; front matter has 5 sections and 6 task units; most chapters have one section; good multi-section chapter coverage. |

Additional observed shape signals:

| Document | Max Sections Per Chapter | Max Task Units Per Section | Task Unit ID Key | Unique Task Unit IDs | Content Blocks Present? |
|---|---:|---:|---|---:|---|
| `APPLE` | 1 | 0 | N/A | 0 / 0 | No |
| `Madame Bovary` | 1 | 34 | `unit_id` | 500 / 500 | No |
| `中式思维` | 1 | 10 | `unit_id` | 14 / 14 | No |
| `许三观卖血记` | 5 | 12 | `unit_id` | 98 / 98 | No |

Readiness note: persisted task units currently expose `unit_id`, not `task_unit_id`, in the structured JSON files. This is not only a wording mismatch. It is a pre-schema task-unit identity strategy risk.

Current `unit_id` values should be treated as reference/import identity evidence for Phase 1 parity checks. Future DB planning must distinguish:

- DB-generated internal primary key
- public/domain identity only where a concrete external stability requirement exists
- reference/import identity from existing structured JSON

DB-generated internal primary keys are the default internal identity and relational link foundation for early DB design. Current `unit_id` values must not become production DB identity. Artifact targets, content-block links, API references, and evaluation records may require validation metadata or deliberate public/domain identity later. This audit does not choose UUID, bigint, or any other identity implementation.

## Representative Evaluation Candidates

Recommended small evaluation set:

| Category | Candidate | Reason |
|---|---|---|
| Small / compact hierarchy | `中式思维` | Small chapter count, Chinese text, front/back matter, and dense task-unit content make it a good smoke candidate. |
| Medium hierarchy / multi-section chapter | `许三观卖血记` | Covers Chinese long-form structure and the only observed multi-section chapter pattern through front matter. |
| Large hierarchy / content-heavy | `Madame Bovary` | Largest task-unit count and largest structured file; best stress candidate for import parity, reload parity, lookup parity, and opaque content preservation. |
| Hierarchy reference / no-task-unit valid document | `APPLE` | Valid `StructuredDocument` reference file with chapter/section hierarchy and zero task units. Useful for hierarchy parity, reload parity, and no-task-unit acceptance coverage; not task-unit identity/order/content evidence. |

Coverage gaps:

- No deeply nested hierarchy exists because the current architecture intentionally normalizes to chapter -> section -> task unit.
- No single document appears multilingual; repository-level coverage includes English and Chinese documents.
- No content-block persistence examples are present, which is acceptable because content blocks are out of Phase 1 scope.
- `APPLE` is a valid no-task-unit `StructuredDocument` reference file. Its absence of task units must not be interpreted as legacy schema, production migration constraint, or DB identity requirement.

## Phase 1 Positive Validation Readiness

| Validation Area | Readiness | Rationale |
|---|---|---|
| Semantic hierarchy parity | Ready with caveat | `APPLE` provides valid chapter/section hierarchy coverage with zero task units; three other documents include full chapter/section/task-unit hierarchy with unique `unit_id` values. |
| Hierarchy lookup parity | Partially Ready | The repository has real hierarchy files and existing hierarchy-first helper/test coverage, but this audit does not create a validation harness that compares file-backed and JSONB-reloaded models. |
| DB-to-model reload parity | Partially Ready | File-backed JSON payloads can serve as import/reload baselines, but no DB-to-model validation path exists yet and this task does not implement one. |
| Namespace/document isolation validation | Partially Ready | Real documents have distinct names and file identities, but there is no observed same-`doc_name` cross-namespace fixture or DB identity fixture. Future validation will need synthetic namespace collision/isolation cases. |
| Opaque `task_unit.content` preservation validation | Ready with caveat | `Madame Bovary`, `中式思维`, and `许三观卖血记` provide substantial task-unit content payloads. `APPLE` has no task units, so it is not content-preservation evidence, but that absence is a valid document shape rather than a legacy classification. |

## Phase 1 Negative Validation Readiness

The real structured document repository does not currently contain invalid structured JSON examples. Existing script-level tests include some in-code negative or defensive cases, but there are no reusable negative fixture files under `data/structured/`.

| Negative Case | Present In Real Structured Repository? | Current Evidence | Future Fixture Need |
|---|---:|---|---|
| Legacy-only payload examples | No | Scripts reference sections-only legacy payload behavior, but real structured files do not include root `sections[]` or `structure_nodes[]`. | Add synthetic legacy-only fixture for validation-only use. |
| Duplicate hierarchy ID examples | No | Real document chapter IDs, section IDs, and `unit_id` values are unique within observed files. Scripts contain duplicate defensive checks. | Add synthetic duplicate chapter, section, and task-unit id fixtures. |
| Malformed hierarchy examples | No | Real files are structured enough for inventory. No malformed placement fixture was observed in `data/structured/`. | Add synthetic malformed placement fixtures, such as task unit under wrong parent section or section parent mismatch. |
| Missing chapter examples | No | All real structured files contain `chapters`. | Add synthetic missing/empty `chapters` fixtures for strict rejection and failure-category parity. |
| Legacy compatibility field parity | Not applicable to Phase 1 | Phase 1 should validate strict hierarchy documents and reject or separately migrate legacy-only payloads. | Keep legacy compatibility as migration-only fixture coverage, not acceptance parity coverage. |

Negative validation should compare failure categories rather than exact exception classes or messages.

## Evaluation Fixture Strategy

Use real repository documents for:

- semantic hierarchy parity on representative valid documents
- task unit identity, order, and placement parity
- hierarchy lookup parity on valid documents
- opaque `task_unit.content` preservation on content-heavy documents
- file-backed baseline and non-destructive import safety
- English and Chinese document coverage
- large-payload stress coverage through `Madame Bovary`

Use synthetic fixtures for:

- legacy-only payload rejection
- missing `chapters`
- duplicate chapter IDs
- duplicate section IDs
- duplicate task-unit IDs
- malformed hierarchy placement
- namespace/document collision or same `doc_name` under different namespaces
- valid no-task-unit hierarchy acceptance for `APPLE`-like files

Do not use synthetic fixtures to define hierarchy truth. Synthetic fixtures should only exercise rejection, isolation, and edge-case categories that are not naturally present in the current structured repository.

## Readiness Risks

1. Insufficient invalid fixture coverage.
   - Real structured files are positive examples only. Phase 1 fail-fast parity cannot be fully evidenced without synthetic negative fixtures.

2. Namespace isolation coverage is incomplete.
   - Current files have distinct names. They do not prove same-document-name isolation across namespaces.

3. `APPLE` is not task-unit evidence.
   - `APPLE` is a valid `StructuredDocument` reference file with chapter/section hierarchy and zero task units. It remains useful hierarchy evaluation material, but it cannot validate task-unit identity/order/content parity.

4. No deeply nested hierarchy coverage.
   - This is expected under the current two-layer normalized hierarchy scope, but future reviewers should not mistake the absence of deeper nesting for an untested DB limitation.

5. Task-unit identity strategy risk.
   - The evaluation document says `task_unit_id`, while current persisted task units use `unit_id`. Current `unit_id` should be reference/import evidence only. Future DB design defaults to DB-generated internal primary keys and should add public/domain identity only for concrete external stability requirements.

6. Lazy content-block persistence boundary remains future work.
   - Content blocks are out of Phase 1 acceptance and should not be embedded into the normal `StructuredDocument` JSONB payload. Future planning should treat them as lazily materialized resources linked to task units and persisted separately after computation.

## Lazy Content Block Persistence Direction

Content blocks are logically linked to `StructuredDocument` task units, but they are not hierarchy truth and are not Phase 1 acceptance criteria.

Future planning direction:

- If the user does not request content blocks, they should not be eagerly computed.
- Once computed, content blocks should be persisted separately from the `StructuredDocument` JSONB payload.
- Future persistence should likely use a separate persistence surface linked to task units.
- Related artifacts should also be persisted separately and linked to hierarchy or content-block targets.
- This keeps Phase 1 JSONB focused on bare hierarchy parity.

This audit does not design tables, columns, artifact schema, lazy-loading mechanics, or content-block persistence implementation.

## Recommendation

Recommendation: Ready with Minor Preparation.

Justification:

- The repository has enough real structured documents to cover valid hierarchy parity across small, medium, large, English, Chinese, and content-heavy examples.
- The repository has `APPLE` as a valid no-task-unit `StructuredDocument` reference file, which should be included as hierarchy evaluation material while excluded from task-unit parity acceptance.
- The repository does not yet have reusable negative validation fixtures for legacy-only payloads, missing chapters, duplicate hierarchy IDs, malformed hierarchy placement, or namespace collision.
- Phase 1 can proceed after minor preparation defines the representative candidate set, creates validation-only negative fixtures in a future task, records `APPLE` as valid no-task-unit hierarchy evidence, and records `unit_id` as reference/import identity evidence while deferring any deliberate public/domain identity decisions to pre-schema planning.

This recommendation does not mark Phase 1 evaluation complete. It only states readiness for future evaluation planning after minor fixture preparation.

## Explicit Non-Goals

- No schema design.
- No table design.
- No JSONB layout.
- No PostgreSQL implementation.
- No PostgreSQL connection.
- No JSONB persistence implementation.
- No migration execution.
- No migration scripts.
- No repository abstraction implementation.
- No repository interface implementation.
- No dual-write strategy.
- No runtime switch.
- No backend cutover.
- No source code changes.
- No API changes.
- No dependency changes.
