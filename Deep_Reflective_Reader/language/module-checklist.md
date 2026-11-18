# language Checklist

## Question audit unit 1 (2026-11-17)

- [x] QUESTION-01/02: Supply distinct multilingual document-extent query cues.
  Status: Locally verified; external question review pending.
  Evidence: New routing suite: baseline 9 methods / 50 failed sub-assertions / zero errors or skips; fixed 10/10 (one new extent-registry interface test). External lexical 6/6; external context routing 4/4 methods, with the separate invalid-standardization method retaining five QUESTION-03 failures. Ten formal suites: 69/69 executions, zero skips; offline smoke 11/11 and standalone context-builder CLI pass. Two hash-verified complete book copies supply four bounded routing flows; backup hashes unchanged. Evidence groups overlap and are not summed as unique cases. `/tmp/question-unit1-252db05/report.txt`.
  Timestamp: 2026-11-18

## Language audit unit 1 (2026-11-17)

- [x] LANG-01/02: Reject unknown detector authority and avoid content-bearing diagnostics.
  Status: Locally verified; language external review pending.
  Acceptance: Normalization/Prepare/retrieval and captured-log red/green; preserve accepted data and warm reuse, no automatic retry.
  Evidence: New boundary suite 13/13 (real file/PostgreSQL); external detector 4/4 and normalization 3/3; adjacent/focused total 207/207, zero skips after documented runner cwd correction. Two full hash-verified book copies across file/SQL retain accepted assets under three invalid rebuild replies. `/tmp/lang-unit1-847f0d6/report.txt`. Unit 1 evidence retained; LANG-03 now locally verified, external language approval pending.
  Timestamp: 2026-11-17


- [x] LANG-03: Reject deterministic cross-script hints contradictions with versioned policy compatibility.
  Status: Locally verified; external language review pending.
  Acceptance: Real saved/reload/reuse evidence, conservative Chinese/Japanese controls, mixed/null exclusion, unchanged source/fence protection.
  Evidence: New script boundary suite: baseline 10 methods / 7 failed assertions / zero errors or skips; fixed 10/10 including real file/PostgreSQL persistence. Cumulative 23 suites: 470/470 executions, zero skips (overlapping inherited scenarios are not unique business cases). External persistence probe and 11 controls pass with documented observation-only harness changes. Both immutable books retain hashes; bounded English negative / Chinese positive samples pass four file/SQL flows. `/tmp/lang-unit2-847f0d6/report.txt`. External exact-SHA language review remains pending.
  Timestamp: 2026-11-17

## Language Learning Reading Mode Queue (2026-10-28)

**LLRN-01 implemented and verified; language foundation only.** [Shared dependency queue](../section_tasks/module-checklist.md#language-learning-reading-mode-queue-2026-10-28)；[canonical design](../section_tasks/module-detailed-design.md#language-learning-reading-mode-plan-2026-10-28)。本檔只持有 language 的唯一 checklist；2026-10-28 foundation evidence 為歷史範圍，當前模式整合與交付狀態以各依賴 owner 的 checklist 為準。

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
