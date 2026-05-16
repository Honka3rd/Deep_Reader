# llm Checklist

## Purpose

This checklist records completed, code-confirmed or design-confirmed tasks for the `llm` module.

It is used to:
- preserve module-level implementation memory
- reduce hallucination in future Codex tasks
- prevent context-window compression from losing completed work
- track future task completion explicitly

## Source Documents

- `Deep_Reflective_Reader/llm/module-detailed-design.md`
- `Deep_Reflective_Reader/proposal.md`
- `Deep_Reflective_Reader/high-level-design.md`
- `Deep_Reflective_Reader/llm/`

## Rules

- Only completed work is listed as checked.
- Future work must not be added unless explicitly requested.
- If a new task is added later, it must first be added unchecked.
- Once completed, it must be checked in this file.
- Uncertain items must go to `Needs Confirmation`, not the completed checklist.

## Completed Checklist

- [x] Defines LLM provider abstraction contract for completion and capability reporting.
  Evidence: `Deep_Reflective_Reader/llm/llm_provider.py; Deep_Reflective_Reader/llm/module-detailed-design.md (Key Files)`
  Notes: Shared provider contract is used across multiple modules.

- [x] Implements OpenAI-backed provider with model-capability mapping and endpoint routing.
  Evidence: `Deep_Reflective_Reader/llm/openai_llm_provider.py; Deep_Reflective_Reader/llm/module-detailed-design.md (Main Responsibilities)`
  Notes: Provider selects responses vs chat-completions based on model capabilities.

- [x] Implements capability-aware prompt text normalization fallback helper.
  Evidence: `Deep_Reflective_Reader/llm/llm_provider.py; Deep_Reflective_Reader/llm/module-detailed-design.md (Main Flows)`
  Notes: Normalization computes safe excerpts with fallback when capabilities unavailable.

## Needs Confirmation

No unresolved confirmation items identified in this pass.

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
