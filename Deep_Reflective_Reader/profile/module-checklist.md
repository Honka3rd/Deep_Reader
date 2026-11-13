# profile Checklist

## Purpose

This checklist records completed, code-confirmed or design-confirmed tasks for the `profile` module.

It is used to:
- preserve module-level implementation memory
- reduce hallucination in future Codex tasks
- prevent context-window compression from losing completed work
- track future task completion explicitly

## Source Documents

- `Deep_Reflective_Reader/profile/module-detailed-design.md`
- `Deep_Reflective_Reader/proposal.md`
- `Deep_Reflective_Reader/high-level-design.md`
- `Deep_Reflective_Reader/profile/`

## Rules

- Only completed work is listed as checked.
- Future work must not be added unless explicitly requested.
- If a new task is added later, it must first be added unchecked.
- Once completed, it must be checked in this file.
- Uncertain items must go to `Needs Confirmation`, not the completed checklist.

## Completed Checklist

- [x] PROFILE-03: Bound complete classification and fallback prompts before dispatch.
  Acceptance: Actual reserve, full JSON/evidence/metadata, provider counter and conservative fallback; CJK/numeric/English, dense candidates, small/large/unknown capacities; immutable full-book evidence and original excerpt assertions.
  Evidence: `profile/profile_prompt_budget.py`, builder/evidence reducer; `scripts/test_profile_classification_contract.py` (11 methods shared with PROFILE-04, all pass); original excerpt suite 2/2; `/tmp/profile-fix-fc1dd285/unit1-green.log`, `real-books-unit1.json` (immutable full copies, 1242/1475 tokens < 6000). Local validation only, external review pending.
  Timestamp: 2026-11-13

- [x] PROFILE-04: Accept absent/null confidence without losing valid classification.
  Acceptance: One call for absent/null/0/1/ordinary values; reject wrong type, non-finite and out-of-range values; preserve malformed-response fallback and deterministic metadata.
  Evidence: `scripts/test_profile_classification_contract.py`, parser metadata suite 6/6; external `unit1-external-reproduce.log` records null confidence one-call success; full boundary red/green logs under `/tmp/profile-fix-fc1dd285/`. Local validation only, external review pending.
  Timestamp: 2026-11-13

- [x] PROFILE-01: Reject source/lifecycle-invalid profile generation and enrichment publication.
  Acceptance: File/real SQL late/new results, source replacement and rename/delete/reimport; previous valid bytes survive and rejected results are not ready.
  Evidence: `scripts/test_profile_persistence_contract.py` 32/32 file/real SQL methods; external six 6/6; core 42/42, learning 29/29 and enhancement 32 pass/one private opt-in skip. `/tmp/profile-fix-fc1dd285/final-report.txt`; external review pending.
  Timestamp: 2026-11-13

- [x] PROFILE-02: Validate persisted core profile fields on read without repair.
  Acceptance: Empty/missing/wrong-type/malformed payloads fail readiness; valid legacy optional omissions and source-only learning reads remain compatible in both compositions; reconcile current diagnostics/enhancement fixtures.
  Evidence: `scripts/test_profile_persistence_contract.py` 32/32 file/real SQL methods; external six 6/6; core 42/42, learning 29/29 and enhancement 32 pass/one private opt-in skip. `/tmp/profile-fix-fc1dd285/final-report.txt`; external review pending.
  Timestamp: 2026-11-13

- [x] Defines `DocumentProfile` contract with `parser_metadata` and `post_structure_metadata` plus legacy compatibility fields.
  Evidence: `Deep_Reflective_Reader/profile/document_profile.py; Deep_Reflective_Reader/profile/module-detailed-design.md (Important Data Structures / Contracts)`
  Notes: Profile schema supports both current and compatibility fields.

- [x] Builds pre-structure profile via deterministic extraction plus lightweight LLM classification/fallback.
  Evidence: `Deep_Reflective_Reader/profile/document_profile_builder.py; Deep_Reflective_Reader/profile/parser_metadata_extractor.py; Deep_Reflective_Reader/profile/module-detailed-design.md (Main Flows)`
  Notes: Builder does not make metadata parser authority.

- [x] Implements post-structure metadata enrichment and profile persistence store operations.
  Evidence: `Deep_Reflective_Reader/profile/post_structure_metadata_enricher.py; Deep_Reflective_Reader/profile/document_profile_store.py; Deep_Reflective_Reader/profile/module-detailed-design.md (Main Responsibilities)`
  Notes: Post snapshot is hierarchy-grounded and used as advisory diagnostics source.

- [x] Audit and clean profile governance documentation boundaries
  Evidence: `Deep_Reflective_Reader/profile/module-detailed-design.md (Non-Responsibilities, Metadata Semantics Boundary, Terminology Governance Audit)`; `Deep_Reflective_Reader/document_preparation/document_preparation_pipeline.py`; `Deep_Reflective_Reader/app/section_task_coordinator.py`; `Deep_Reflective_Reader/progress.md`
  Notes: 已明確 advisory-only metadata、snapshot semantics、runtime diagnostics projection no-write-back，以及 profile 非 hierarchy/artifact persistence authority。

- [x] Clarify artifact governance and profile metadata boundary
  Evidence: `Deep_Reflective_Reader/profile/module-detailed-design.md (Artifact Governance Boundary, Persistence / Side Effects, Terminology Governance Audit)`; `Deep_Reflective_Reader/profile/document_profile.py`; `Deep_Reflective_Reader/profile/post_structure_metadata_enricher.py`; `Deep_Reflective_Reader/app/section_task_coordinator.py`; `Deep_Reflective_Reader/progress.md`
  Notes: 已固定 profile metadata advisory-only、post-structure snapshot 非 runtime availability truth、diagnostics/runtime projection 不回寫 profile，並明確 profile 不承擔 artifact persistence/availability authority。

## Needs Confirmation

The optional cache-schema/reason-code questions in design sections 11 and 16 remain
Needs Confirmation; this audit remediation does not resolve those future contracts.

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
