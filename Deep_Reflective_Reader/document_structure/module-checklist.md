# document_structure Checklist

## Purpose

This checklist records completed, code-confirmed or design-confirmed tasks for the `document_structure` module.

It is used to:
- preserve module-level implementation memory
- reduce hallucination in future Codex tasks
- prevent context-window compression from losing completed work
- track future task completion explicitly

## Source Documents

- `Deep_Reflective_Reader/document_structure/module-detailed-design.md`
- `Deep_Reflective_Reader/proposal.md`
- `Deep_Reflective_Reader/high-level-design.md`
- `Deep_Reflective_Reader/document_structure/`

## Rules

- Only completed work is listed as checked.
- Future work must not be added unless explicitly requested.
- If a new task is added later, it must first be added unchecked.
- Once completed, it must be checked in this file.
- Uncertain items must go to `Needs Confirmation`, not the completed checklist.

## Completed Checklist

- [x] Implements hierarchy-first structured model contracts (`StructuredDocument`, chapter, section) with pure-hierarchy write defaults.
  Evidence: `Deep_Reflective_Reader/document_structure/structured_document.py; Deep_Reflective_Reader/document_structure/module-detailed-design.md (Architecture Constraints)`
  Notes: Root legacy mirrors are not default persistence output.

- [x] Implements hierarchy-first effective indexing helpers and section lookup paths.
  Evidence: `Deep_Reflective_Reader/document_structure/document_hierarchy_index.py; Deep_Reflective_Reader/document_structure/module-detailed-design.md (Key Files)`
  Notes: Effective sections derive from `chapters[].sections[]` runtime contract.

- [x] Implements hierarchy-aware artifact repository with strict hierarchy-required runtime load paths.
  Evidence: `Deep_Reflective_Reader/document_structure/structured_document_artifact_repository.py; Deep_Reflective_Reader/document_structure/module-detailed-design.md (Known Legacy / Compatibility Behavior)`
  Notes: Repository remains write-boundary for section/chapter/task-unit artifacts, and runtime read/write path now requires chapters hierarchy.

- [x] document governance cleanup for hierarchy-first persistence terminology
  Evidence: `Deep_Reflective_Reader/document_structure/module-detailed-design.md (Non-Responsibilities, Architecture Constraints, Terminology Governance Audit)`; `Deep_Reflective_Reader/document_structure/structured_document.py`; `Deep_Reflective_Reader/progress.md`
  Notes: 明確分離 primary hierarchy contract 與 legacy compatibility wording，並補齊 task-layout/diagnostics ownership 邊界描述。

- [x] synchronize unresolved confirmation status after governance cleanup
  Evidence: `Deep_Reflective_Reader/document_structure/module-detailed-design.md (Known Legacy / Compatibility Behavior, Terminology Governance Audit, Terminology Validation Notes)`; `Deep_Reflective_Reader/document_structure/document_hierarchy_index.py`; `Deep_Reflective_Reader/progress.md`
  Notes: 已同步 detailed-design/checklist/progress 的 Needs Confirmation 狀態；本輪後 mirror terminology 與 allow_legacy_fallback 退場項目已完成收斂。

- [x] Clarify artifact governance and hierarchy persistence boundary
  Evidence: `Deep_Reflective_Reader/document_structure/module-detailed-design.md (Architecture Constraints, Artifact Governance Boundary, Known Legacy / Compatibility Behavior)`; `Deep_Reflective_Reader/document_structure/document_artifact_repository.py`; `Deep_Reflective_Reader/document_structure/structured_document_artifact_repository.py`; `Deep_Reflective_Reader/progress.md`
  Notes: 已明確分離 hierarchy truth / artifact output / runtime projection ownership，並固定 artifact 不可反向改寫 hierarchy identity。

- [x] governance consistency closure for compatibility terminology and fallback wording
  Evidence: `Deep_Reflective_Reader/document_structure/module-detailed-design.md (Known Legacy / Compatibility Behavior, Terminology Governance Audit, Terminology Validation Notes)`; maintainer-confirmed governance direction in current task; `Deep_Reflective_Reader/progress.md`
  Notes: `mirror` 已退出正式 architecture terminology，統一改為 `legacy compatibility fields` / `compatibility-only fields`；`allow_legacy_fallback` 明確標記為 compatibility-only，且不得暗示 runtime primary path。

- [x] Audit and retire allow_legacy_fallback legacy helper path
  Evidence: `Deep_Reflective_Reader/document_structure/document_hierarchy_index.py`; `Deep_Reflective_Reader/scripts/test_chapter_hierarchy_primary_model.py`; `Deep_Reflective_Reader/document_structure/module-detailed-design.md`; `Deep_Reflective_Reader/progress.md`
  Notes: 已完成 narrow removal：`find_*_effective` 普通 helper API surface 不再暴露 `allow_legacy_fallback`，lookup 全面 hierarchy-only；sections-only legacy 文檔在 effective helper 路徑不再被解析。

- [x] Remove allow_legacy_fallback API surface and enforce hierarchy-only runtime lookup
  Evidence: `Deep_Reflective_Reader/document_structure/document_hierarchy_index.py`; `Deep_Reflective_Reader/app/section_task_coordinator.py`; `Deep_Reflective_Reader/scripts/test_chapter_hierarchy_primary_model.py`; `Deep_Reflective_Reader/scripts/test_hierarchy_first_task_target_resolution.py`
  Notes: helper signatures 已移除 `allow_legacy_fallback`，app chapter-title runtime lookup 不再回退 root sections，也不再合成 legacy chapter。

- [x] Isolate or remove legacy read compatibility from model/repository boundaries
  Evidence: `Deep_Reflective_Reader/document_structure/structured_document.py`; `Deep_Reflective_Reader/document_structure/structured_document_store.py`; `Deep_Reflective_Reader/document_structure/structured_document_artifact_repository.py`; `Deep_Reflective_Reader/scripts/test_pure_hierarchy_json_cleanup.py`; `Deep_Reflective_Reader/scripts/test_hierarchy_artifact_write_sync.py`; `Deep_Reflective_Reader/scripts/test_task_artifact_persistence.py`
  Notes: normal `from_dict/from_json` 與 repository write path 改為 strict hierarchy-only；legacy sections/structure_nodes 讀取保留於 explicit migration-only loader，不再作 ordinary runtime read source。

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
