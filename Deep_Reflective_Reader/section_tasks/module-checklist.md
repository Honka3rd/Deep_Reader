# section_tasks Checklist

## Purpose

This checklist records completed, code-confirmed or design-confirmed tasks for the `section_tasks` module.

It is used to:
- preserve module-level implementation memory
- reduce hallucination in future Codex tasks
- prevent context-window compression from losing completed work
- track future task completion explicitly

## Source Documents

- `Deep_Reflective_Reader/section_tasks/module-detailed-design.md`
- `Deep_Reflective_Reader/proposal.md`
- `Deep_Reflective_Reader/high-level-design.md`
- `Deep_Reflective_Reader/section_tasks/`

## Rules

- Only completed work is listed as checked.
- Future work must not be added unless explicitly requested.
- If a new task is added later, it must first be added unchecked.
- Once completed, it must be checked in this file.
- Uncertain items must go to `Needs Confirmation`, not the completed checklist.

## Completed Checklist

- [x] Add task-unit content lookup support for frontend rendering
  Evidence: `Deep_Reflective_Reader/section_tasks/document_task_layout.py`; `Deep_Reflective_Reader/app/section_task_coordinator.py`; `Deep_Reflective_Reader/scripts/test_task_unit_content_endpoint.py`
  Notes: 新增 `TaskUnitContentDTO` 與 hierarchy-only `task_unit_id` 查詢能力；missing/duplicate 明確 fail-fast；未修改 task-layout DTO 的 content contract。

- [x] Add rich content blocks to task-unit content lookup response
  Evidence: `Deep_Reflective_Reader/section_tasks/document_task_layout.py`; `Deep_Reflective_Reader/app/section_task_coordinator.py`; `Deep_Reflective_Reader/main.py`; `Deep_Reflective_Reader/scripts/test_task_unit_content_endpoint.py`; `Deep_Reflective_Reader/shared/task_unit_model.py`
  Notes: 既有 `content: str` 保留，新增 additive `content_blocks`，且由 `TaskUnit.to_content_blocks()` 生成；task-layout response contract 未變更且不返回 `content_blocks`。

- [x] Add explicit segmented content-block option to task-unit content endpoint
  Evidence: `Deep_Reflective_Reader/section_tasks/document_task_layout.py`; `Deep_Reflective_Reader/app/section_task_coordinator.py`; `Deep_Reflective_Reader/main.py`; `Deep_Reflective_Reader/api_schemas.py`; `Deep_Reflective_Reader/scripts/test_task_unit_content_endpoint.py`; `Deep_Reflective_Reader/scripts/test_shared_task_unit_content_blocks.py`
  Notes: 新增顯式 opt-in 參數 `segmented`；`segmented=true` 使用 shared `TaskUnit.segment_content_blocks()`，`segmented=false/省略` 保持 compatibility-safe 預設 block 行為；`content: str` 保留，不改 task-layout/API route/persistence contract。

- [x] Implements chapters-first task-layout DTO contracts and diagnostics DTO types.
  Evidence: `Deep_Reflective_Reader/section_tasks/document_task_layout.py; Deep_Reflective_Reader/section_tasks/module-detailed-design.md (Important Data Structures / Contracts)`
  Notes: Task-layout contract is explicit at module level.

- [x] Implements hierarchy-first task unit resolution entry using effective hierarchy sections.
  Evidence: `Deep_Reflective_Reader/section_tasks/task_unit_resolver.py; Deep_Reflective_Reader/section_tasks/module-detailed-design.md (Key Files)`
  Notes: Resolver path was aligned with hierarchy-first model.

- [x] Implements section task context lookup with hierarchy-only section resolution behavior.
  Evidence: `Deep_Reflective_Reader/section_tasks/section_task_context_builder.py; Deep_Reflective_Reader/section_tasks/module-detailed-design.md (Known Legacy / Compatibility Behavior)`
  Notes: Legacy section fallback is disabled by default at context-builder path.

- [x] Documentation governance cleanup for hierarchy-first task-layout semantics
  Evidence: `Deep_Reflective_Reader/section_tasks/module-detailed-design.md (Read/Write Boundary Matrix, Public API Boundary Clarification, Terminology Governance Audit)`; `Deep_Reflective_Reader/main.py`; `Deep_Reflective_Reader/api_schemas.py`; `Deep_Reflective_Reader/progress.md`
  Notes: 已明確分離 task-layout projection/read path 與 artifact persistence write path，並固定 public chapters-first contract 與 diagnostics no-write-back 邊界。

- [x] Clarify artifact availability projection and task-layout boundary
  Evidence: `Deep_Reflective_Reader/section_tasks/module-detailed-design.md (Artifact Governance and Projection Boundary, Terminology Governance Audit, Persistence / Side Effects)`; `Deep_Reflective_Reader/section_tasks/document_task_layout.py`; `Deep_Reflective_Reader/app/section_task_coordinator.py`; `Deep_Reflective_Reader/progress.md`
  Notes: 明確分離 persisted hierarchy truth、artifact persistence write path、runtime availability projection、diagnostics projection 與 API DTO shape，並保持 no hidden mutation/no profile write-back。

- [x] Close terminology governance item: deprecate `artifact mirror` as formal contract wording
  Evidence: `Deep_Reflective_Reader/section_tasks/module-detailed-design.md (Terminology Governance Audit, Terminology Validation Notes)`; maintainer decision in current documentation-governance task; `Deep_Reflective_Reader/progress.md`
  Notes: `artifact mirror` 僅保留為 historical/migration/compatibility reference；正式術語統一為 `transitional internal field`，並明確非 persistence authority/non-public-contract terminology。

- [x] Pass through content-block artifact target metadata safely in task-unit content response
  Evidence: `Deep_Reflective_Reader/main.py`; `Deep_Reflective_Reader/api_schemas.py`; `Deep_Reflective_Reader/scripts/test_task_unit_content_endpoint.py`; `Deep_Reflective_Reader/scripts/test_shared_task_unit_content_blocks.py`
  Notes: task-unit content endpoint 在保留 `content + content_blocks` 下新增 pass-through `artifact_target_refs`；metadata 僅允許最小 glossary keys（`source_hash`, `content_block_id`, `quote_span_start`, `quote_span_end`, `schema_version`）；不觸發 artifact repository read/write，不改 task-layout payload。

- [x] Document minimum artifact target metadata glossary for content endpoint pass-through
  Evidence: `Deep_Reflective_Reader/section_tasks/module-detailed-design.md (Artifact Target Metadata Glossary and Target Constraints)`; `Deep_Reflective_Reader/section_tasks/module-checklist.md`; `Deep_Reflective_Reader/progress.md`
  Notes: 明確固定 metadata keys（`source_hash`, `content_block_id`, `quote_span_start`, `quote_span_end`, `schema_version`）、allowed target combinations（content_block 需 `task_unit_id + content_block_id`；task_unit 需 `task_unit_id`），並標示 metadata-only boundary；no source code changes in this documentation patch.

- [x] Prepare section_tasks segmentation design direction for future content-block projection behavior
  Evidence: `Deep_Reflective_Reader/section_tasks/module-detailed-design.md (Future Direction Note: Content Block Segmentation Design Preparation)`; `Deep_Reflective_Reader/section_tasks/module-checklist.md`; `Deep_Reflective_Reader/progress.md`
  Notes: Documentation/design-preparation only; defined segmented content endpoint projection semantics, failure/validation boundaries, task_unit/content_block identity constraints, and artifact-target alignment guardrails; no runtime/API/task-layout/persistence changes.

- [x] Harden segmented task-unit content endpoint behavior and multilingual fixtures
  Evidence: `Deep_Reflective_Reader/scripts/test_task_unit_content_endpoint.py`; `Deep_Reflective_Reader/scripts/test_shared_task_unit_content_blocks.py`; `Deep_Reflective_Reader/section_tasks/module-detailed-design.md`; `Deep_Reflective_Reader/progress.md`
  Notes: 強化 omitted/false/true segmented flag 行為矩陣、compatibility regression、deterministic block id/metadata key 驗證，並新增 CJK/mixed-format fixture coverage（Chinese/Japanese paragraph、mixed paragraph+list、heading-like、table-like）；未修改 task-layout payload/persistence/artifact repository/retrieval/LLM/evaluated_answer。

- [x] Suppress duplicated section/chapter heading line in segmented content endpoint projection
  Evidence: `Deep_Reflective_Reader/app/section_task_coordinator.py`; `Deep_Reflective_Reader/scripts/test_task_unit_content_endpoint.py`; live API validation for `/documents/Madame%20Bovary/task-units/2703/content?segmented=true`
  Notes: `segmented=true` response no longer exposes a duplicated leading hierarchy title inside `content_blocks[0].content`; metadata spans remain mapped to original task-unit content. This is render projection only and does not change task-layout payload, parser authority, hierarchy truth, or artifact persistence.

- [x] Add lightweight parse provenance DTO to task-layout projection
  Evidence: `Deep_Reflective_Reader/section_tasks/document_task_layout.py`; `Deep_Reflective_Reader/app/section_task_coordinator.py`; `Deep_Reflective_Reader/scripts/test_task_unit_content_endpoint.py`
  Notes: `DocumentTaskLayout` carries optional `ParseProvenanceDTO` with requested/effective parser mode and fallback metadata. It does not add heavy content and does not alter Chapter -> Section -> Task Unit hierarchy rendering.

- [x] Add lightweight chapter/section anchor evidence to task-layout DTOs
  Evidence: `Deep_Reflective_Reader/section_tasks/document_task_layout.py`; `Deep_Reflective_Reader/scripts/test_task_layout_anchor_evidence_dto.py`
  Notes: `DocumentTaskLayoutChapterDTO` and `DocumentTaskLayoutSectionDTO` can carry optional `AnchorEvidenceDTO` for UI edit-existing prefill. The DTO supports page-range and char-range metadata, omits unset evidence to preserve current payload shape before API mapping, and excludes raw text, OCR text, geometry, and content blocks.

## Needs Confirmation

No unresolved confirmation items identified in this pass.

## Future Task Policy

New future tasks for this module must be added here first as unchecked items:

- [ ] Support content-block-level interaction targeting
- [ ] Define content-block artifact availability projection
- [ ] Preserve lightweight task-layout metadata contract
- [ ] Define segmented content-block endpoint projection semantics
- [ ] Define duplicate/missing content-block validation behavior
- [ ] Define segmented block artifact-target alignment behavior
- [ ] Define target-agnostic reading interaction service contracts
  Evidence needed: service contracts accept a resolved document/chapter/section/task-unit target and return validated interaction DTOs without owning route mapping or persistence internals.
  Notes: This should become the shared foundation for analysis, quiz, and critical-thinking services.
  Timestamp: 2026-10-05
- [ ] Implement analysis interaction service with strict JSON validation
  Evidence needed: service produces validated analysis payload containing summary, reasoning/interpretation, and parsing/explanation fields, or an insufficient-content result.
  Notes: Invalid model output must fail generation rather than becoming a successful artifact.
  Timestamp: 2026-10-05
- [ ] Implement quiz interaction service with configured target-level limits
  Evidence needed: service enforces valid quiz types and configured max counts for task-unit, section, chapter, and document targets.
  Notes: The LLM may choose the quiz type mix and may generate fewer than max.
  Timestamp: 2026-10-05
- [ ] Implement critical-thinking session service
  Evidence needed: service supports persisted question generation, answer submission, answer evaluation, and evaluation retry with recoverable failure status.
  Notes: First version remains question -> answer -> evaluation and does not introduce question pools.
  Timestamp: 2026-10-05
- [ ] Add interaction artifact validity and insufficient-content semantics
  Evidence needed: all interaction services can return or persist insufficient-content status with reason metadata and can report stale/invalid target context.
  Notes: This guards noisy OCR-derived units and avoids repeated LLM spending.
  Timestamp: 2026-10-05
- [ ] Add lower-level artifact reference policy for higher-level generation
  Evidence needed: section/chapter/document generation services can consume compact child artifact summaries as secondary context while preserving source text as primary context.
  Notes: Applies recursively from unit -> section -> chapter -> document/book.
  Timestamp: 2026-10-05
- [ ] Add quiz deduplication behavior from lower-level artifacts
  Evidence needed: quiz service can use child quiz artifacts to avoid repeated concepts/questions without concatenating lower-level quiz items into the higher-level quiz.
  Notes: Higher-level quiz should favor synthesis, transfer, comparison, and cross-unit understanding.
  Timestamp: 2026-10-05
- [ ] Add critical-thinking learning-continuity behavior from lower-level sessions
  Evidence needed: critical-thinking service can use child session summaries to generate broader questions that build on prior local training.
  Notes: Child sessions remain secondary learning memory, not source truth.
  Timestamp: 2026-10-05
- [ ] Persist artifact reference metadata for generated interaction artifacts
  Evidence needed: generated artifacts record referenced artifact ids/types/target levels and whether deduplication or abstraction-escalation hints were used.
  Notes: Metadata supports later observability and cost/debug analysis.
  Timestamp: 2026-10-05
- [x] Preserve section-scoped task-layout ownership for TOC-derived hierarchy
  Evidence: `Deep_Reflective_Reader/section_tasks/document_task_layout.py`; `Deep_Reflective_Reader/app/section_task_coordinator.py`; `Deep_Reflective_Reader/scripts/test_task_unit_content_endpoint.py`; `PYTHONPATH=. .venv/bin/python scripts/test_task_unit_content_endpoint.py`.
  Notes: Task-layout projection consumes effective `chapters[].sections[].task_units[]` ownership and exposes section-scoped task-unit metadata only. TOC-aware parsing may change section boundaries upstream, but task-layout does not merge content across sections or create a second ownership model.

- [x] Define task-layout consumption contract for page/layout-derived hierarchy
  Evidence: `Deep_Reflective_Reader/section_tasks/document_task_layout.py`; `Deep_Reflective_Reader/main.py`; `Deep_Reflective_Reader/scripts/test_task_layout_anchor_evidence_dto.py`; `Deep_Reflective_Reader/scripts/test_task_unit_content_endpoint.py`; `Deep_Reflective_Reader/.venv/bin/python Deep_Reflective_Reader/scripts/test_task_layout_anchor_evidence_dto.py`; `PYTHONPATH=. .venv/bin/python scripts/test_task_unit_content_endpoint.py`.
  Notes: Task-layout consumes only the validated active hierarchy and may expose lightweight `anchor_evidence` as advisory prefill metadata. It does not reconstruct TOC structure from orientation, reading order, TOC scores, OCR/page text, geometry, root `sections[]`, or `structure_nodes`.

After implementation, the task owner must update this checklist and mark the task as completed:

No coding task should be considered complete unless the corresponding module checklist is updated.

## Maintenance Notes

- This checklist is module memory for completed work.
- It does not replace the module detailed design document.
- It does not replace tests and test evidence.
- It does not replace proposal/HLD decisions and governance context.
