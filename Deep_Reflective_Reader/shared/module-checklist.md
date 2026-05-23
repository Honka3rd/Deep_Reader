# shared Checklist

## Purpose

This checklist records completed, code-confirmed or design-confirmed tasks for the `shared` module.

It is used to:
- preserve module-level implementation memory
- reduce hallucination in future Codex tasks
- prevent context-window compression from losing completed work
- track future task completion explicitly

## Source Documents

- `Deep_Reflective_Reader/shared/module-detailed-design.md`
- `Deep_Reflective_Reader/proposal.md`
- `Deep_Reflective_Reader/high-level-design.md`
- `Deep_Reflective_Reader/shared/`

## Rules

- Only completed work is listed as checked.
- Future work must not be added unless explicitly requested.
- If a new task is added later, it must first be added unchecked.
- Once completed, it must be checked in this file.
- Uncertain items must go to `Needs Confirmation`, not the completed checklist.

## Completed Checklist

- [x] Defines cross-module `TaskUnit` contract including parent identity and artifact payload fields.
  Evidence: `Deep_Reflective_Reader/shared/task_unit_model.py; Deep_Reflective_Reader/shared/module-detailed-design.md (Important Data Structures / Contracts)`
  Notes: Task unit serialization/deserialization contract is centralized.

- [x] Defines summary/quiz artifact schemas and document-level artifact container models.
  Evidence: `Deep_Reflective_Reader/shared/task_artifacts.py; Deep_Reflective_Reader/shared/module-detailed-design.md (Main Responsibilities)`
  Notes: Artifact metadata/version fields are part of shared contract.

- [x] Defines generic abstract result contract for service execution outputs.
  Evidence: `Deep_Reflective_Reader/shared/abstract_result.py; Deep_Reflective_Reader/shared/module-detailed-design.md (Key Files)`
  Notes: Success/failure payload structure is standardized.

- [x] Implements shared rich task-unit content block foundation with deterministic adapter behavior.
  Evidence: `Deep_Reflective_Reader/shared/task_unit_model.py; Deep_Reflective_Reader/scripts/test_shared_task_unit_content_blocks.py; Deep_Reflective_Reader/shared/module-detailed-design.md (Rich Task-Unit Content Foundation and Future Direction)`
  Notes: Added `TaskUnitContentBlock`, deterministic block id builder (`<task_unit_id>:content:<index>`), and `TaskUnit.to_content_blocks()` adapter while preserving `TaskUnit.content: str`; task-layout/API/persistence behavior unchanged.

- [x] Stabilize TaskUnit rich content internal representation
  Evidence: `Deep_Reflective_Reader/shared/task_unit_model.py`; `Deep_Reflective_Reader/scripts/test_shared_task_unit_content_blocks.py`; `Deep_Reflective_Reader/shared/module-detailed-design.md (Rich Task-Unit Content Foundation and Future Direction)`
  Notes: Added additive `TaskUnit.content_blocks` field with post-init auto-stabilization from `content`; `TaskUnit.content` remains supported; `to_dict(include_content_blocks=True)` enables additive round-trip while default serialization remains compatibility-safe; no endpoint/task-layout/API/persistence migration changes in this slice.

- [x] Define shared content-block artifact target foundation
  Evidence: `Deep_Reflective_Reader/shared/task_unit_model.py`; `Deep_Reflective_Reader/scripts/test_shared_task_unit_content_blocks.py`; `Deep_Reflective_Reader/shared/module-detailed-design.md (Rich Task-Unit Content Foundation and Future Direction)`
  Notes: Added shared `ArtifactTargetLevel` + `ArtifactTargetRef` metadata model with supported levels (`document/chapter/section/task_unit/content_block`), plus additive `TaskUnitContentBlock.artifact_target_refs`; serialization/deserialization remains backward-compatible and keeps `artifact_ids` behavior unchanged; no artifact persistence/API/task-layout/repository changes.

- [x] Extract shared artifact target level contract into dedicated shared module
  Evidence: `Deep_Reflective_Reader/shared/artifact_target_model.py`; `Deep_Reflective_Reader/shared/task_unit_model.py`; `Deep_Reflective_Reader/api_schemas.py`; `Deep_Reflective_Reader/scripts/test_shared_task_unit_content_blocks.py`; `Deep_Reflective_Reader/scripts/test_task_unit_content_endpoint.py`
  Notes: Moved `ArtifactTargetLevel`/`ArtifactTargetRef` to `shared/artifact_target_model.py` as single source of truth; kept `shared.task_unit_model` import compatibility and aligned API schema to reuse the same enum contract.

- [x] Prepare shared segmentation design direction for future deterministic content-block generation
  Evidence: `Deep_Reflective_Reader/shared/module-detailed-design.md (Future Direction Note: Content Block Segmentation Design Preparation)`; `Deep_Reflective_Reader/shared/module-checklist.md`; `Deep_Reflective_Reader/progress.md`
  Notes: Documentation/design-preparation only; defined segmentation contract, deterministic id/source-hash/span policy direction, reparse-stability risk model, and compatibility staging from `content` to multi-block generation; no source code/tests/API/persistence/task-layout behavior changes.

- [x] Implement deterministic content block segmentation foundation
  Evidence: `Deep_Reflective_Reader/shared/task_unit_model.py`; `Deep_Reflective_Reader/scripts/test_shared_task_unit_content_blocks.py`; `Deep_Reflective_Reader/shared/module-detailed-design.md`; `Deep_Reflective_Reader/progress.md`
  Notes: Added explicit opt-in shared segmentation (`TaskUnit.segment_content_blocks()` / `segment_task_unit_content`) with deterministic paragraph-first + list-item-safe split rules, deterministic block ids (`<task_unit_id>:content:<index>`), advisory metadata (`source_hash`, `content_block_id`, `quote_span_start`, `quote_span_end`, `schema_version`), and idempotent behavior; preserved default `to_content_blocks()` compatibility, old payload support, and no endpoint/API/task-layout/persistence/retrieval/LLM changes.

## Needs Confirmation

No unresolved confirmation items identified in this pass.

## Future Task Policy

New future tasks for this module must be added here first as unchecked items:

- [ ] Finalize content-block identity and artifact attachment semantics beyond shared foundation metadata
- [ ] Harden deterministic segmentation rules for heading/sentence/table-like structures beyond paragraph/list baseline
- [ ] Define reparse-resilient block-id/source-hash/span evolution strategy across segmentation-version changes
- [ ] Define promotion strategy from explicit opt-in segmentation to default multi-block behavior

After implementation, the task owner must update this checklist and mark the task as completed:

- [x] <completed task>

No coding task should be considered complete unless the corresponding module checklist is updated.

## Maintenance Notes

- This checklist is module memory for completed work.
- It does not replace the module detailed design document.
- It does not replace tests and test evidence.
- It does not replace proposal/HLD decisions and governance context.
