# context Checklist

## Purpose

This checklist records completed, code-confirmed or design-confirmed tasks for the `context` module.

It is used to:
- preserve module-level implementation memory
- reduce hallucination in future Codex tasks
- prevent context-window compression from losing completed work
- track future task completion explicitly

## Source Documents

- `Deep_Reflective_Reader/context/module-detailed-design.md`
- `Deep_Reflective_Reader/proposal.md`
- `Deep_Reflective_Reader/high-level-design.md`
- `Deep_Reflective_Reader/context/`

## Rules

- Only completed work is listed as checked.
- Future work must not be added unless explicitly requested.
- If a new task is added later, it must first be added unchecked.
- Once completed, it must be checked in this file.
- Uncertain items must go to `Needs Confirmation`, not the completed checklist.

## Completed Checklist

- [x] Implements context-mode orchestration for local window, retrieval, and full-text paths.
  Evidence: `Deep_Reflective_Reader/context/context_orchestrator.py; Deep_Reflective_Reader/context/module-detailed-design.md (Main Responsibilities)`
  Notes: Context routing is explicit and mode-aware.

- [x] Builds ordered context chunks with budget controls via `DocumentContextBuilder`.
  Evidence: `Deep_Reflective_Reader/context/document_context_builder.py; Deep_Reflective_Reader/context/module-detailed-design.md (Main Flows)`
  Notes: Includes neighbor expansion and truncation handling.

- [x] Provides prompt-aware token budgeting and truncation utilities through `TokenBudgetManager`.
  Evidence: `Deep_Reflective_Reader/context/token_budget_manager.py; Deep_Reflective_Reader/context/module-detailed-design.md (Key Files)`
  Notes: Budget computation uses non-context prompt estimation plus reserves.

## Needs Confirmation

No unresolved confirmation items identified in this pass.

## Future Task Policy

New future tasks for this module must be added here first as unchecked items:

- [ ] Define reading target context adapter
  Evidence needed: context module accepts resolved document/chapter/section/task-unit targets and builds target-scoped text/evidence without title fallback.
  Notes: Adapter should return metadata needed for artifact provenance.
  Timestamp: 2026-10-05
- [ ] Support full-target context gate for reading interactions
  Evidence needed: context selection uses model capability and configured budget to decide when the whole target can fit.
  Notes: This should reuse or align with existing token budget resolver behavior.
  Timestamp: 2026-10-05
- [ ] Support semantic compact fallback for large reading targets
  Evidence needed: context selection produces a compacted single-call context with evidence ids when target content exceeds budget.
  Notes: First version should not require multi-call map-reduce.
  Timestamp: 2026-10-05
- [ ] Expose interaction context provenance metadata
  Evidence needed: context result carries context mode, token estimate, effective budget, evidence ids, and truncation/compaction reason for artifact metadata.
  Notes: Provenance supports cost visibility and regeneration decisions.
  Timestamp: 2026-10-05
- [ ] Add artifact-aware secondary context assembly
  Evidence needed: context builder can include compact lower-level artifact summaries for section/chapter/document generation without replacing current target source text.
  Notes: Use lower-level artifacts for deduplication, coverage awareness, abstraction hints, and difficulty escalation only.
  Timestamp: 2026-10-05
- [ ] Add artifact context budget and pruning policy
  Evidence needed: context selection prioritizes fixed instruction and current target source context before lower-level artifact summaries, and records pruning/compaction reasons.
  Notes: Artifact context should be summarized/projection-based, not raw artifact payload dumping.
  Timestamp: 2026-10-05
- [ ] Record artifact context provenance metadata
  Evidence needed: context result includes `artifact_context_mode`, `referenced_artifact_ids`, target-level coverage counts, deduplication hints, and artifact pruning reasons.
  Notes: Absence of lower-level artifacts should not block generation.
  Timestamp: 2026-10-05

After implementation, the task owner must update this checklist and mark the task as completed:

- [x] <completed task>

No coding task should be considered complete unless the corresponding module checklist is updated.

## Maintenance Notes

- This checklist is module memory for completed work.
- It does not replace the module detailed design document.
- It does not replace tests and test evidence.
- It does not replace proposal/HLD decisions and governance context.
