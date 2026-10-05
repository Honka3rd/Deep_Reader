# prompts Checklist

## Purpose

This checklist records completed, code-confirmed or design-confirmed tasks for the `prompts` module.

It is used to:
- preserve module-level implementation memory
- reduce hallucination in future Codex tasks
- prevent context-window compression from losing completed work
- track future task completion explicitly

## Source Documents

- `Deep_Reflective_Reader/prompts/module-detailed-design.md`
- `Deep_Reflective_Reader/proposal.md`
- `Deep_Reflective_Reader/high-level-design.md`
- `Deep_Reflective_Reader/prompts/`

## Rules

- Only completed work is listed as checked.
- Future work must not be added unless explicitly requested.
- If a new task is added later, it must first be added unchecked.
- Once completed, it must be checked in this file.
- Uncertain items must go to `Needs Confirmation`, not the completed checklist.

## Completed Checklist

- [x] Implements profile rendering block for answer prompts.
  Evidence: `Deep_Reflective_Reader/prompts/prompt_assembler.py; Deep_Reflective_Reader/prompts/module-detailed-design.md (Main Responsibilities)`
  Notes: Prompt includes topic, language code, and summary.

- [x] Implements answer-rule rendering by `AnswerMode` strictness levels.
  Evidence: `Deep_Reflective_Reader/prompts/prompt_assembler.py; Deep_Reflective_Reader/prompts/module-detailed-design.md (Main Responsibilities)`
  Notes: Strict/cautious/reject instruction sets are explicit.

- [x] Implements mode-specific guidance for local reading, retrieval, and full-text prompts.
  Evidence: `Deep_Reflective_Reader/prompts/prompt_assembler.py; Deep_Reflective_Reader/prompts/module-detailed-design.md (Main Flows)`
  Notes: Guidance text is selected through `PromptMode`.

## Needs Confirmation

No unresolved confirmation items identified in this pass.

## Future Task Policy

New future tasks for this module must be added here first as unchecked items:

- [ ] Define fixed analysis prompt instruction
  Evidence needed: versioned prompt template requests summary, reasoning/interpretation, parsing/explanation, strict JSON, and insufficient-content handling.
  Notes: Only context/target/options should be dynamic.
  Timestamp: 2026-10-05
- [ ] Define fixed quiz prompt instruction with valid type and max-count constraints
  Evidence needed: prompt template lists `short_answer`, `multiple_choice`, and `true_false`, passes target-level max count, and allows fewer-than-max generation.
  Notes: LLM chooses the type mix; backend validation remains authoritative.
  Timestamp: 2026-10-05
- [ ] Define fixed critical-thinking question prompt instruction
  Evidence needed: prompt template explicitly frames the task as critical-thinking training and asks for one focused question tied to the target context.
  Notes: This prompt generates a persistable session even before the user answers.
  Timestamp: 2026-10-05
- [ ] Define fixed critical-thinking evaluation prompt instruction
  Evidence needed: prompt template evaluates one submitted answer against the generated question and context with strict JSON output.
  Notes: It must preserve the session model and not generate a replacement question.
  Timestamp: 2026-10-05
- [ ] Define JSON validation and retry prompt policy for reading interactions
  Evidence needed: prompt/service contract identifies invalid JSON as generation failure and supports explicit retry policy without persisting invalid successful artifacts.
  Notes: Prompt compliance is advisory; server validation is the hard boundary.
  Timestamp: 2026-10-05
- [ ] Define artifact-aware prompt input sections
  Evidence needed: prompt templates clearly separate primary source context from secondary lower-level artifact context.
  Notes: Lower-level artifacts support deduplication and abstraction, not source replacement.
  Timestamp: 2026-10-05
- [ ] Add fixed quiz deduplication instruction
  Evidence needed: quiz prompt tells the model to avoid repeating lower-level quiz questions and to favor higher-level synthesis when child artifacts exist.
  Notes: Backend validation still controls valid type/count/answer shape.
  Timestamp: 2026-10-05
- [ ] Add fixed critical-thinking abstraction instruction
  Evidence needed: critical-thinking prompt tells the model to build on lower-level focus areas and move toward synthesis, transfer, and critique.
  Notes: The instruction must still produce only one question in the first version.
  Timestamp: 2026-10-05

After implementation, the task owner must update this checklist and mark the task as completed:

- [x] <completed task>

No coding task should be considered complete unless the corresponding module checklist is updated.

## Maintenance Notes

- This checklist is module memory for completed work.
- It does not replace the module detailed design document.
- It does not replace tests and test evidence.
- It does not replace proposal/HLD decisions and governance context.
