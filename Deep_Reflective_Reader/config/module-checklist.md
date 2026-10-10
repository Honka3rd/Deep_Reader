# config Checklist

## Purpose

This checklist records completed, code-confirmed or design-confirmed tasks for the `config` module.

It is used to:
- preserve module-level implementation memory
- reduce hallucination in future Codex tasks
- prevent context-window compression from losing completed work
- track future task completion explicitly

## Source Documents

- `Deep_Reflective_Reader/config/module-detailed-design.md`
- `Deep_Reflective_Reader/proposal.md`
- `Deep_Reflective_Reader/high-level-design.md`
- `Deep_Reflective_Reader/config/`

## Rules

- Only completed work is listed as checked.
- Future work must not be added unless explicitly requested.
- If a new task is added later, it must first be added unchecked.
- Once completed, it must be checked in this file.
- Uncertain items must go to `Needs Confirmation`, not the completed checklist.

## Completed Checklist

- [x] Defines grouped runtime policy dataclasses in `AppDIConfig` and related config types.
  Evidence: `Deep_Reflective_Reader/config/app_DI_config.py; Deep_Reflective_Reader/config/module-detailed-design.md (Key Files)`
  Notes: Policy values are centralized for DI and runtime behavior control.

- [x] Assembles core dependencies through `ApplicationLookupContainer`.
  Evidence: `Deep_Reflective_Reader/config/container.py; Deep_Reflective_Reader/config/module-detailed-design.md (Main Responsibilities)`
  Notes: Container wires providers, repositories, coordinators, and service selectors.

- [x] Implements namespace normalization and legacy namespace/file migration for artifact storage configs.
  Evidence: `Deep_Reflective_Reader/config/faiss_storage_config.py; Deep_Reflective_Reader/config/structured_document_storage_config.py; Deep_Reflective_Reader/config/module-detailed-design.md (Known Legacy / Compatibility Behavior)`
  Notes: Storage naming compatibility is handled at config boundary.

- [x] Wire analysis reading interaction dependencies into the container
  Evidence: `Deep_Reflective_Reader/config/container.py`; `Deep_Reflective_Reader/section_tasks/analysis_interaction_llm_generator.py`; `Deep_Reflective_Reader/section_tasks/analysis_interaction_service.py`; `Deep_Reflective_Reader/section_tasks/analysis_interaction_orchestrator.py`; `Deep_Reflective_Reader/section_tasks/reading_interaction_artifact_store.py`; `Deep_Reflective_Reader/scripts/test_analysis_interaction_routes.py`; `Deep_Reflective_Reader/scripts/test_reading_interaction_artifact_store.py`.
  Notes: `ApplicationLookupContainer` now assembles the first analysis/insight vertical slice by wiring the LLM-backed generator, strict analysis service, document-backed current-artifact store, analysis orchestrator, and `SectionTaskCoordinator` injection. Config owns assembly only; hierarchy semantics, route behavior, prompt validation, and artifact persistence semantics stay in their owning modules.
  Timestamp: 2026-10-10

## Needs Confirmation

No unresolved confirmation items identified in this pass.

## Future Task Policy

New future tasks for this module must be added here first as unchecked items:

- [ ] Define storage backend selection policy for file / DB coexistence
- [ ] Define configuration boundary for DB-backed structured storage
- [ ] Preserve existing file path behavior during DB migration rollout
- [ ] Define storage backend configuration contract
- [ ] Define backend rollout policy contract
- [ ] Define file-to-db coexistence configuration model
- [ ] Review backend-selection implications after Phase 1 evaluation
- [ ] Define reading interaction configuration contract
  Evidence needed: config exposes target-level quiz max counts, allowed quiz type policy, and generation/evaluation cost-gate policy hooks.
  Notes: Defaults should be task_unit=3, section=5, chapter=10, document/book=25.
  Timestamp: 2026-10-05
- [ ] Keep persisted artifact reads outside cost/permission generation gates
  Evidence needed: config/service integration distinguishes free read of existing artifacts from costly generate/refresh/evaluate/retry actions.
  Notes: This preserves the read/generate split from the Grill-me decision.
  Timestamp: 2026-10-05

After implementation, the task owner must update this checklist and mark the task as completed:

- [x] <completed task>

No coding task should be considered complete unless the corresponding module checklist is updated.

## Maintenance Notes

- This checklist is module memory for completed work.
- It does not replace the module detailed design document.
- It does not replace tests and test evidence.
- It does not replace proposal/HLD decisions and governance context.
