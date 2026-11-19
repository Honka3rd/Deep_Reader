# retrieval Checklist

## RETRIEVAL-04 default QA correction (2026-11-19)

- [x] RETRIEVAL-04: Verify compatible default QA and preserve explicit inadequate-cap rejection.
  Acceptance: Real default DI, cold/warm HTTP 200, one answer, successful session; explicit 3200 safe rejection, unchanged trio/warm reuse, bounded source-copy evidence.
  Evidence: `/tmp/retrieval-unit4-d698ab5/report.txt`; external 3/3 red-to-green, formal 5/5, scoped 193/193 zero skips (real PostgreSQL), four immutable-book default-DI sample flows.
  Status: Locally verified; RETRIEVAL-04/05 cumulative external review pending.
  Timestamp: 2026-11-19

- [x] RETRIEVAL-05: Validate final dispatch against the selected context mode's effective budget.
  Acceptance: Legal GLOBAL full-text, local/retrieval caps, actual output reserved once, genuine overflow refusal; actual QA/context controls.
  Evidence: `/tmp/retrieval-unit5-d698ab5/report.txt`; external 4/4 red-to-green, dedicated 9/9, cumulative 202/202 zero skips, 8 immutable-book GLOBAL flows.
  Status: Locally verified; cumulative exact-SHA external review pending.
  Timestamp: 2026-11-19

## Retrieval audit correction (2026-11-19)

- [x] RETRIEVAL-01: Validate unique FAISS/map/record identities at all load/publication gates.
  Acceptance: Red/green real store/pipeline, non-contiguous positive controls, previous snapshot and candidate rejection, old trio preservation; no read-time rebuild.
  Evidence: `/tmp/retrieval-unit1-51813cd/report.txt`; external 4 failures -> 5/5; new file/SQL 24/24; cumulative 78/78, zero skips; immutable-book recovery controls.
  Status: Externally verified on d698ab5 for RETRIEVAL-01; RETRIEVAL-04/05 corrections locally verified, cumulative candidate review pending.
  Timestamp: 2026-11-19

- [x] RETRIEVAL-02: Remove source-bearing search diagnostics while preserving full hits.
  Acceptance: Direct/QA INFO/ERROR sink privacy and unchanged ranking/content/call counts.
  Evidence: `/tmp/retrieval-unit2-51813cd/report.txt`; external 4 failures -> 2/2 plus 9 controls; new 13/13, scoped 74/74 zero skips, 8 immutable-book sample flows. INFO/ERROR/DEBUG sinks preserve privacy and complete returned data.
  Status: Externally verified on d698ab5 for RETRIEVAL-02; RETRIEVAL-04/05 corrections locally verified, cumulative candidate review pending.
  Timestamp: 2026-11-19

- [x] RETRIEVAL-03: Use provider dispatch allowance for cold/warm bundle output reserve.
  Acceptance: Near-limit red/green, explicit-small-cap/floor controls, both endpoints, capacity and completion regressions.
  Evidence: `/tmp/retrieval-unit3-51813cd/report.txt`; original 3/3 red-to-green, new 10/10, cumulative 188/188 zero skips; both endpoints, cold/warm, precise capacity refusal, 8 immutable-book sample flows.
  Status: Core reserve alignment externally verified on d698ab5; RETRIEVAL-04/05 corrections locally verified and await cumulative external review.
  Timestamp: 2026-11-19

## Language audit unit 1 (2026-11-17)

- [x] LANG-01/02: Reject unknown detector authority and avoid content-bearing diagnostics.
  Status: Locally verified; language external review pending.
  Acceptance: Normalization/Prepare/retrieval and captured-log red/green; preserve accepted data and warm reuse, no automatic retry.
  Evidence: New boundary suite 13/13 (real file/PostgreSQL); external detector 4/4 and normalization 3/3; adjacent/focused total 207/207, zero skips after documented runner cwd correction. Two full hash-verified book copies across file/SQL retain accepted assets under three invalid rebuild replies. `/tmp/lang-unit1-847f0d6/report.txt`. Unit 1 evidence retained; LANG-03 is separately verified in language/section_tasks. External language approval pending.
  Timestamp: 2026-11-17


## PREP-05 audit correction (2026-11-13)

- [x] PREP-05 support: Validate every record against the canonical source, including journal previous snapshots.
  Evidence: `scripts/test_preparation_index_identity.py`: 10 methods / 66 matrix subcases, file and real PostgreSQL, real splitter/FAISS. External independent 8/8, safety 24/24, existing 142/142, SQL 16/16, original six 6/6 and 7 scripts pass; two real-book copies preserve warm zero-generation and exact slices. `/tmp/prep05-fix-a8c0ca1/`; external review pending, no deployment.
  Timestamp: 2026-11-13

## Preparation audit correction (2026-11-13)

- [x] PREP-04: Stage and validate index/records/fingerprint, with durable rollback and coherent read/recovery.
  Evidence: `test_preparation_runtime_safety.py` file/real PostgreSQL matrix, original regressions and isolated real-book checks; `/tmp/prep-fix-e3eb6a8/`. Local implementation verified; external review/deployment pending, no production data mutation.
  Timestamp: 2026-11-13

## Purpose

This checklist records completed, code-confirmed or design-confirmed tasks for the `retrieval` module.

It is used to:
- preserve module-level implementation memory
- reduce hallucination in future Codex tasks
- prevent context-window compression from losing completed work
- track future task completion explicitly

## Source Documents

- `Deep_Reflective_Reader/retrieval/module-detailed-design.md`
- `Deep_Reflective_Reader/proposal.md`
- `Deep_Reflective_Reader/high-level-design.md`
- `Deep_Reflective_Reader/retrieval/`

## Rules

- Only completed work is listed as checked.
- Future work must not be added unless explicitly requested.
- If a new task is added later, it must first be added unchecked.
- Once completed, it must be checked in this file.
- Uncertain items must go to `Needs Confirmation`, not the completed checklist.

## Completed Checklist

- [x] Parses raw text into node sequence with positional metadata through `NodeProvider`.
  Evidence: `Deep_Reflective_Reader/retrieval/node_provider.py; Deep_Reflective_Reader/retrieval/module-detailed-design.md (Main Responsibilities)`
  Notes: Node metadata includes chunk index, char offsets, and neighbor links.

- [x] Builds FAISS bundles from parsed nodes with capability-aware token budgets.
  Evidence: `Deep_Reflective_Reader/retrieval/faiss_index_builder.py; Deep_Reflective_Reader/retrieval/module-detailed-design.md (Main Flows)`
  Notes: Builder returns `FaissIndexBundle` with runtime budget fields.

- [x] Persists and reloads FAISS artifacts with record-schema checks and rebuild guards.
  Evidence: `Deep_Reflective_Reader/retrieval/faiss_index_store.py; Deep_Reflective_Reader/retrieval/module-detailed-design.md (Known Legacy / Compatibility Behavior)`
  Notes: Store can detect incomplete legacy record schema via position metadata checks.

## Needs Confirmation

No unimplemented retrieval audit items; cumulative external review pending.

## Future Task Policy

New future tasks for this module must be added here first as unchecked items:

- [ ] <future task>

After implementation, the task owner must update this checklist and mark the task as completed:

- [x] <completed task>

No coding task should be considered complete unless the corresponding module checklist is updated.

## Maintenance Notes

- This checklist is module memory for completed work.
- It does not replace the module detailed design document.
- It does not replace tests and test evidence.
- It does not replace proposal/HLD decisions and governance context.
