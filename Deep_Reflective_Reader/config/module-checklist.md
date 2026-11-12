# config Checklist

- [x] CONFIG-03: share upload/rename destination conflicts and reject legacy ambiguity before effects.
  Acceptance: file/real PostgreSQL HTTP `Legacy` + `Legacy.pdf`, `Source.txt -> Legacy.txt` rejects without changing body, notes, raw/index, journals or sidecars; independent qualified sources, replay and generation fences remain valid.
  Evidence: `scripts/test_config_storage_audit.py` passes 35 cases (12 new); 199 related regressions, two real-backup file/PostgreSQL acceptance cases and seven standalone scripts pass, zero skips. Original two HTTP repros fail on 2a213b9. `/tmp/audit-config-03/validation-report.txt`; external review pending.
  Timestamp: 2026-11-12

- [x] CONFIG-01: preserve distinct source identities across upload, body, index and lifecycle.
  Acceptance: file/real PostgreSQL source and note isolation, upload conflict policies, safe legacy aliases and fail-closed ambiguity, rename/delete and stale command protection.
  Evidence: `scripts/test_config_storage_audit.py`; 23 source/read-path cases; 479 backend tests without skips, real SQL/TCP and regression evidence in `/tmp/audit-config-dcb3095/`. External review pending.
  Timestamp: 2026-11-12

- [x] CONFIG-02: make storage resolution and ordinary reads free of filesystem mutations.
  Acceptance: snapshot paths/contents for layout, content and profile reads with legacy-only, coexisting and missing directories; explicit writes still create required parents.
  Evidence: `scripts/test_config_storage_audit.py`; 23 source/read-path cases; 479 backend tests without skips, real SQL/TCP and regression evidence in `/tmp/audit-config-dcb3095/`. External review pending.
  Timestamp: 2026-11-12

## Shared sentence translation (2026-11-09)

- [x] Wire ordinary translation to the existing learning runtime and single-attempt provider.
  Owner: config. Status: Implemented and verified.
  Acceptance: Follow the shared sentence translation contract in section_tasks; preserve reads, compatibility and explicit generation boundaries.
  Evidence: 470 backend tests pass, including 35 shared file/PostgreSQL cases; actual API/React mode roundtrips at 1280/390, two backed-up real-source offline cases and type/build checks pass. No duplicate aggregate is persisted.
  Timestamp: 2026-11-09


## Language Learning Reading Mode Queue (2026-10-28)

**Implemented and verified: LLRN-15.** [Shared dependency queue](../section_tasks/module-checklist.md#language-learning-reading-mode-queue-2026-10-28)；[canonical design](../section_tasks/module-detailed-design.md#language-learning-reading-mode-plan-2026-10-28)。本檔只持有 config 的唯一 checklist；production API/lifespan 綁定已由 LLRN-16 完成，完整模式及 UI 尚未交付。

### LLRN-15

- [x] LLRN-15: Learning dependency composition.
  Owner: config. Status: Implemented and verified (file/SQL runtime composition; production routes/lifespan delivered by LLRN-16).
  Depends on: LLRN-08, LLRN-09, LLRN-10, LLRN-11, LLRN-13.
  Deliverable: 組裝偏好/學習 repository、services、provider、admission policy 與 learning application module；沿用現有 file/SQL backend 選擇。
  Acceptance: 不把母語/能力寫入 env，不創新語言清單或散落 provider config；啟動不生成，缺依賴明確 unavailable；既有模型 capability 為容量來源。
  Evidence: `container.py`、`learning_storage.py`、`learning_runtime.py`、`AppDIConfig.learning_runtime` 及 supporting profile/LLM/SQL public ports；`scripts/test_learning_container_wiring.py` 29 tests + `scripts/test_learning_provider_once.py` 4 tests passed。涵蓋 file/真 PostgreSQL instance/scope/DSN 一致、missing schema/key/capability 明確拒絕、零 import/startup network/files、純讀、offline reuse、profile source-only、admission claim/dispatch、shutdown drain/reinit、rename/delete 與 deterministic late writer 拒絕。真 SDK HTTP mock 驗證 responses/chat 成功及 429/500/503/timeout 無隱藏 retry。453 項既有 regression 中發現 file rename 先完成時漏出 FileNotFoundError，已用鎖內 tombstone guard 修正；重驗 45 file repository + 43 lifecycle + 81 orchestration 均通過，其餘 284 項亦通過。另 9 capacity tests 及既有 config grouping/reading interaction wiring scripts passed。SQL 僅 disposable PostgreSQL；沒有 live LLM 品質宣稱、使用者 migration、書籍修改或 production API/UI。
  Timestamp: 2026-11-10

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

- [x] Provides lossless source identities and pure legacy alias resolution for artifact storage configs.
  Evidence: `Deep_Reflective_Reader/config/faiss_storage_config.py; Deep_Reflective_Reader/config/structured_document_storage_config.py; Deep_Reflective_Reader/config/module-detailed-design.md (Known Legacy / Compatibility Behavior)`
  Notes: CONFIG-01/02 replace lossy suffix stripping and implicit migration with checked, non-mutating resolution.
  Timestamp: 2026-11-12

- [x] Wire analysis reading interaction dependencies into the container
  Evidence: `Deep_Reflective_Reader/config/container.py`; `Deep_Reflective_Reader/section_tasks/analysis_interaction_llm_generator.py`; `Deep_Reflective_Reader/section_tasks/analysis_interaction_service.py`; `Deep_Reflective_Reader/section_tasks/analysis_interaction_orchestrator.py`; `Deep_Reflective_Reader/section_tasks/reading_interaction_artifact_store.py`; `Deep_Reflective_Reader/scripts/test_analysis_interaction_routes.py`; `Deep_Reflective_Reader/scripts/test_reading_interaction_artifact_store.py`.
  Notes: `ApplicationLookupContainer` now assembles the first analysis/insight vertical slice by wiring the LLM-backed generator, strict analysis service, document-backed current-artifact store, analysis orchestrator, and `SectionTaskCoordinator` injection. Config owns assembly only; hierarchy semantics, route behavior, prompt validation, and artifact persistence semantics stay in their owning modules.
  Timestamp: 2026-10-10

- [x] Wire quiz and critical-thinking reading interaction dependencies into the container
  Evidence: `Deep_Reflective_Reader/config/container.py`; `Deep_Reflective_Reader/section_tasks/quiz_interaction_llm_generator.py`; `Deep_Reflective_Reader/section_tasks/critical_thinking_llm_generator.py`; `Deep_Reflective_Reader/section_tasks/reading_interaction_artifact_store.py`; `Deep_Reflective_Reader/scripts/test_reading_interaction_container_wiring.py`; `Deep_Reflective_Reader/scripts/test_quiz_interaction_routes.py`; `Deep_Reflective_Reader/scripts/test_critical_thinking_interaction_routes.py`.
  Notes: `ApplicationLookupContainer` now injects `quiz_interaction_orchestrator`, `critical_thinking_session_service`, and `critical_thinking_session_store` into `SectionTaskCoordinator`. This closes the route-exists-but-service-unconfigured gap for `/documents/reading-interactions/quiz/generate` and the critical-thinking route family while keeping route mapping, hierarchy resolution, prompt validation, and persistence semantics in their owning modules.
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
