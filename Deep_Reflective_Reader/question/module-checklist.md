# question Checklist

## Quoted term-role correction (2026-11-18)

- [x] QUESTION-08: Preserve target-specific term and meaning roles.
  Status: Locally verified; external exact-SHA question review remains open.
  Evidence: Exact 4763650 baseline: core external 3 methods /2 failed sub-assertions; supporting 4 methods /4 failed sub-assertions; fixed 3/3 and 4/4, zero errors/skips. Expanded requested-scope suite 15 methods /33 baseline failures -> 15/15; cumulative 12 suites 99/99, zero skips. Prior quoted-positive/extent/envelope 4/4 each; QUESTION-01–04 probes 6/4/5/4 and current-contract smoke 11/11 pass. Twenty-one bounded immutable-book flows pass with canonical/raw/manifest hashes preserved. Groups overlap. `/tmp/question-terms-4763650/report.txt`.
  Timestamp: 2026-11-18

## Quoted range correction (2026-11-18)

- [x] QUESTION-07: QUESTION-07 quoted requested ranges.
  Status: Locally verified; external exact-SHA question review remains open.
  Evidence: At exact 56bbadd baseline, external quoted-positive suite: 4 methods /2 failed sub-assertions, zero errors/skips; fixed 4/4. Expanded formal scope suite: 12 methods /23 baseline failures -> 12/12; cumulative 12 suites: 96/96, zero skips. QUESTION-05 extent and QUESTION-06 envelope 4/4 each; original QUESTION-01–04 probes 6/4/5/4 pass and current-contract smoke 11/11. Sixteen bounded immutable-book flows pass with full backup hashes unchanged. Overlapping groups are not unique-case totals. `/tmp/question-quoted-56bbadd/report.txt`.
  Timestamp: 2026-11-18

## Question follow-up (2026-11-18)

- [x] QUESTION-05/06: Correct requested extent and complete wrapped JSON admission.
  Status: Locally verified; exact-SHA external question review remains open.
  Evidence: Exact 57a7d76 baseline: new formal suite 9 methods / 53 failed sub-assertions / zero errors or skips; fixed 9/9. External extent 4 methods / 6 failures and envelope 4 methods / 2 failures become 4/4 each. Cumulative 12 formal suites: 93/93 executions, zero skips; original external lexical/standardizer/context/diagnostic 6/4/5/4 pass; fixed-method smoke 11/11. Actual envelope context probe and twelve bounded immutable-book flows pass; full backup hashes unchanged. Groups overlap and are not unique-case totals. `/tmp/question-followup-57a7d76/report.txt`.
  Timestamp: 2026-11-18

## Question audit unit 1 (2026-11-17)

- [x] QUESTION-01/02: Fix lexical boundaries and preserve explicit requested scope.
  Status: Locally verified; external question review pending.
  Evidence: New routing suite: baseline 9 methods / 50 failed sub-assertions / zero errors or skips; fixed 10/10 (one new extent-registry interface test). External lexical 6/6; external context routing 4/4 methods, with the separate invalid-standardization method retaining five QUESTION-03 failures. Ten formal suites: 69/69 executions, zero skips; offline smoke 11/11 and standalone context-builder CLI pass. Two hash-verified complete book copies supply four bounded routing flows; backup hashes unchanged. Evidence groups overlap and are not summed as unique cases. `/tmp/question-unit1-252db05/report.txt`.
  Timestamp: 2026-11-18

- [x] QUESTION-03/04: Validate standardized outputs and safe diagnostic/error propagation.
  Status: Locally verified; cumulative exact-SHA external question review pending.
  Acceptance: Root/type/duplicate/non-finite matrices, no retrieval on invalid response, safe logs/errors/HTTP detail, one-call/fallback controls.
  Evidence: New output-boundary suite: exact baseline 15 methods / 85 failed sub-assertions / zero errors or skips; corrected 15/15. Cumulative 11 formal suites: 84/84, zero skips. Original external lexical 6/6, standardizer 4/4, context 5/5 and diagnostic/ASGI route 4/4 pass. Original offline smoke retains two failures in one method expecting free-form model reason; a separately recorded copy changes only the method expectation plus output directory and passes 11/11, retaining routing/call/budget assertions. Full immutable book hashes unchanged; ten bounded deterministic routing/admission/diagnostic flows pass. Evidence groups overlap; not unique-case totals. `/tmp/question-unit2-252db05/report.txt`.
  Timestamp: 2026-11-18

## Purpose

This checklist records completed, code-confirmed or design-confirmed tasks for the `question` module.

It is used to:
- preserve module-level implementation memory
- reduce hallucination in future Codex tasks
- prevent context-window compression from losing completed work
- track future task completion explicitly

## Source Documents

- `Deep_Reflective_Reader/question/module-detailed-design.md`
- `Deep_Reflective_Reader/proposal.md`
- `Deep_Reflective_Reader/high-level-design.md`
- `Deep_Reflective_Reader/question/`

## Rules

- Only completed work is listed as checked.
- Future work must not be added unless explicitly requested.
- If a new task is added later, it must first be added unchecked.
- Once completed, it must be checked in this file.
- Uncertain items must go to `Needs Confirmation`, not the completed checklist.

## Completed Checklist

- [x] Defines query-related enums and standardized question contract.
  Evidence: `Deep_Reflective_Reader/question/qa_enums.py; Deep_Reflective_Reader/question/standardized/standardized_question.py; Deep_Reflective_Reader/question/module-detailed-design.md (Important Data Structures / Contracts)`
  Notes: Scope and prompt modes are formalized as enums.

- [x] Implements LLM-based question standardization with strict JSON/type validation and language normalization.
  Evidence: `Deep_Reflective_Reader/question/standardized/question_standardizer.py; Deep_Reflective_Reader/question/module-detailed-design.md (Main Responsibilities)`
  Notes: QUESTION-03/04 closes the audited baseline admission gap; invalid completed outputs fail before retrieval with safe diagnostics. Current evidence is recorded above.
  Timestamp: 2026-11-18

- [x] Implements scope resolution using lexical, semantic, and optional LLM fallback paths with diagnostics.
  Evidence: `Deep_Reflective_Reader/question/question_scope_resolver.py; Deep_Reflective_Reader/question/module-detailed-design.md (Main Flows)`
  Notes: Resolver includes local-reference signal analysis and method attribution.

## Needs Confirmation

QUESTION-01–08 are locally verified and await cumulative exact-SHA external review. Four pre-existing rich-content future tasks below remain unchecked.

## Future Task Policy

New future tasks for this module must be added here first as unchecked items:

- [ ] Support content-block-level question targeting
- [ ] Define evidence/content-block reference semantics
- [ ] Define sentence/paragraph-level interaction flow
- [ ] Preserve id-based hierarchy context when resolving fine-grained question targets

After implementation, the task owner must update this checklist and mark the task as completed:

- [x] <completed task>

No coding task should be considered complete unless the corresponding module checklist is updated.

## Maintenance Notes

- This checklist is module memory for completed work.
- It does not replace the module detailed design document.
- It does not replace tests and test evidence.
- It does not replace proposal/HLD decisions and governance context.
