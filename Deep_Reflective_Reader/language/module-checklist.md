# language Checklist

## Language Learning Reading Mode Queue (2026-10-28)

**LLRN-01 implemented and verified; language foundation only.** [Shared dependency queue](../section_tasks/module-checklist.md#language-learning-reading-mode-queue-2026-10-28)；[canonical design](../section_tasks/module-detailed-design.md#language-learning-reading-mode-plan-2026-10-28)。本檔只持有 language 的唯一 checklist；完整模式依賴仍待後續實作。

### LLRN-01

- [x] LLRN-01: Registry-bound proficiency definitions.
  Owner: language. Status: Implemented and verified (foundation only).
  Depends on: None.
  Deliverable: 新增獨立 proficiency definitions，以現有 registry code 引用語言；定義 scheme ID/version、排序 level ID、讀者可懂標籤及供 LLM 使用的閱讀能力描述。
  Acceptance: 覆蓋 registry 全部支援語言（排除 unknown）；不可另外維護語言白名單、強套 A1–C2 或宣稱不同考試等值。逐語言方案與標籤在此任務查證並記錄依據。
  Evidence required: Registry 完整性、非法 scheme/level 組合、版本變更測試，以及逐語言分級選擇的可審閱表。
  Evidence: language_proficiency_registry.py 覆蓋現有 19 registry codes；5 種有版本方案、不可變有序資料與 strict selection/coverage guards。18 專項 unittest 與既有 script/discourse registry 腳本通過；來源與逐語言選擇表在 paired design。未接入 UI/API/偏好持久化或 LLM。
  Timestamp: 2026-10-28

## Sentence Segmentation Support (2026-10-28)

- [x] Supply versioned sentence prefix rules for section_tasks-owned LLRN-04.
  Owner: language (supporting data only; no duplicate LLRN completion count).
  Evidence: `learning_sentence_rules.py` provides immutable exact-code rules with system-registry coverage checks. `scripts/test_learning_sentence_segmenter.py` passes all 19 tests, including per-language boundaries and invalid/missing/duplicate registry coverage. Full algorithm, source invariants and private-book acceptance remain recorded by section_tasks.
  Timestamp: 2026-10-28

## Purpose

This checklist records completed, code-confirmed or design-confirmed tasks for the `language` module.

It is used to:
- preserve module-level implementation memory
- reduce hallucination in future Codex tasks
- prevent context-window compression from losing completed work
- track future task completion explicitly

## Source Documents

- `Deep_Reflective_Reader/language/module-detailed-design.md`
- `Deep_Reflective_Reader/proposal.md`
- `Deep_Reflective_Reader/high-level-design.md`
- `Deep_Reflective_Reader/language/`

## Rules

- Only completed work is listed as checked.
- Future work must not be added unless explicitly requested.
- If a new task is added later, it must first be added unchecked.
- Once completed, it must be checked in this file.
- Uncertain items must go to `Needs Confirmation`, not the completed checklist.

## Completed Checklist

- [x] Defines canonical language code enum and resolver/inference utilities.
  Evidence: `Deep_Reflective_Reader/language/language_code.py; Deep_Reflective_Reader/language/module-detailed-design.md (Key Files)`
  Notes: Language normalization supports aliases and script-based inference fallback.

- [x] Implements document language detection with profile/records reuse and LLM fallback.
  Evidence: `Deep_Reflective_Reader/language/document_language_detector.py; Deep_Reflective_Reader/language/module-detailed-design.md (Main Flows)`
  Notes: Detector prefers persisted metadata before model call.

- [x] Implements script/discourse/profile registries for language-scoped heuristics and cues.
  Evidence: `Deep_Reflective_Reader/language/language_script_registry.py; Deep_Reflective_Reader/language/language_discourse_registry.py; Deep_Reflective_Reader/language/language_profile_registry.py; Deep_Reflective_Reader/language/module-detailed-design.md (Main Responsibilities)`
  Notes: Registry boundaries are explicit and consumed by profile/question modules.

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
