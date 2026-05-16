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

- [x] Implements hierarchy-aware artifact repository with compatibility migration support on load paths.
  Evidence: `Deep_Reflective_Reader/document_structure/structured_document_artifact_repository.py; Deep_Reflective_Reader/document_structure/module-detailed-design.md (Known Legacy / Compatibility Behavior)`
  Notes: Repository remains write-boundary for section/chapter/task-unit artifacts.

- [x] document governance cleanup for hierarchy-first persistence terminology
  Evidence: `Deep_Reflective_Reader/document_structure/module-detailed-design.md (Non-Responsibilities, Architecture Constraints, Terminology Governance Audit)`; `Deep_Reflective_Reader/document_structure/structured_document.py`; `Deep_Reflective_Reader/progress.md`
  Notes: 明確分離 primary hierarchy contract 與 legacy compatibility wording，並補齊 task-layout/diagnostics ownership 邊界描述。

- [x] synchronize unresolved confirmation status after governance cleanup
  Evidence: `Deep_Reflective_Reader/document_structure/module-detailed-design.md (Known Legacy / Compatibility Behavior, Terminology Governance Audit, Terminology Validation Notes)`; `Deep_Reflective_Reader/document_structure/document_hierarchy_index.py`; `Deep_Reflective_Reader/progress.md`
  Notes: 已同步 detailed-design/checklist/progress 的 Needs Confirmation 狀態，保留仍未拍板的 fallback 退場時程與 mirror 用詞治理項目。

- [x] Clarify artifact governance and hierarchy persistence boundary
  Evidence: `Deep_Reflective_Reader/document_structure/module-detailed-design.md (Architecture Constraints, Artifact Governance Boundary, Known Legacy / Compatibility Behavior)`; `Deep_Reflective_Reader/document_structure/document_artifact_repository.py`; `Deep_Reflective_Reader/document_structure/structured_document_artifact_repository.py`; `Deep_Reflective_Reader/progress.md`
  Notes: 已明確分離 hierarchy truth / artifact output / runtime projection ownership，並固定 artifact 不可反向改寫 hierarchy identity。

- [x] governance consistency closure for compatibility terminology and fallback wording
  Evidence: `Deep_Reflective_Reader/document_structure/module-detailed-design.md (Known Legacy / Compatibility Behavior, Terminology Governance Audit, Terminology Validation Notes)`; maintainer-confirmed governance direction in current task; `Deep_Reflective_Reader/progress.md`
  Notes: `mirror` 已退出正式 architecture terminology，統一改為 `legacy compatibility fields` / `compatibility-only fields`；`allow_legacy_fallback` 明確標記為 compatibility-only，且不得暗示 runtime primary path。

## Needs Confirmation

- [ ] `find_*_effective(...allow_legacy_fallback=...)` compatibility 分支是否有正式退場時程。  
  Reason: detailed design 仍標註 `Needs Confirmation`；程式碼仍保留可選 fallback 參數。  
  Needed confirmation: maintainer 是否提供明確 deprecation phase/date。

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
