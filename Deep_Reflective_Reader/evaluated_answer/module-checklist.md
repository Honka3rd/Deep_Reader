# evaluated_answer Checklist

## Purpose

This checklist records completed, code-confirmed or design-confirmed tasks for the `evaluated_answer` module.

It is used to:
- preserve module-level implementation memory
- reduce hallucination in future Codex tasks
- prevent context-window compression from losing completed work
- track future task completion explicitly

## Source Documents

- `Deep_Reflective_Reader/evaluated_answer/module-detailed-design.md`
- `Deep_Reflective_Reader/proposal.md`
- `Deep_Reflective_Reader/high-level-design.md`
- `Deep_Reflective_Reader/evaluated_answer/`

## Rules

- Only completed work is listed as checked.
- Future work must not be added unless explicitly requested.
- If a new task is added later, it must first be added unchecked.
- Once completed, it must be checked in this file.
- Uncertain items must go to `Needs Confirmation`, not the completed checklist.

## Completed Checklist

- [x] Defines answer strictness result DTO via `AnswerMode`.
  Evidence: `Deep_Reflective_Reader/evaluated_answer/answer_mode.py; Deep_Reflective_Reader/evaluated_answer/module-detailed-design.md (Key Files)`
  Notes: Result contains level and reason for downstream prompt rules.

- [x] Implements retrieval-score to answer-mode mapping in `QuestionRelevanceEvaluator`.
  Evidence: `Deep_Reflective_Reader/evaluated_answer/question_relevance.py; Deep_Reflective_Reader/evaluated_answer/module-detailed-design.md (Main Responsibilities)`
  Notes: Score thresholds map to strict/cautious/reject levels.

- [x] Handles empty retrieval result as explicit reject mode.
  Evidence: `Deep_Reflective_Reader/evaluated_answer/question_relevance.py; Deep_Reflective_Reader/evaluated_answer/module-detailed-design.md (Main Responsibilities)`
  Notes: Reject path is deterministic for no-results scenario.

## Needs Confirmation

- [ ] Define minimum deterministic contract for quote/evidence validation.
  Reason: Added as unresolved in `evaluated_answer/module-detailed-design.md` section `14.1 Needs Confirmation`.
  Needed confirmation: Whether strict span match or semantic match should be the baseline deterministic rule.

- [ ] Decide whether content-block-level evidence needs persisted trace metadata or runtime projection only.
  Reason: Added as unresolved in `evaluated_answer/module-detailed-design.md` section `14.1 Needs Confirmation`.
  Needed confirmation: Whether to require persisted trace metadata at evaluation boundary.

## Future Task Policy

New future tasks for this module must be added here first as unchecked items:

- [ ] Support content-block-linked answer evidence
- [ ] Define content-block reference validation semantics
- [ ] Define artifact-linked evaluation flow
- [ ] Define quote/evidence validation boundaries
- [ ] Clarify evaluation evidence persistence boundaries
- [ ] Define hierarchy-aware evidence targeting semantics

After implementation, the task owner must update this checklist and mark the task as completed:

- [x] <completed task>

No coding task should be considered complete unless the corresponding module checklist is updated.

## Maintenance Notes

- This checklist is module memory for completed work.
- It does not replace the module detailed design document.
- It does not replace tests and test evidence.
- It does not replace proposal/HLD decisions and governance context.
