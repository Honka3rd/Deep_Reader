# document_structure Checklist

## Common Multilingual Correctness Backlog

- [x] COMMON-LANG-01: Resolve the mismatch between language-registry candidate detection and predominantly Chinese/English final heading validation.
  Priority: High; implemented and accepted after maintainer authorization.
  Acceptance: Reproduce candidate/final disagreement with reasons; unify language-aware evidence without unconditional candidate acceptance; cover supported registry languages, newline variations, numbering/region markers and prose/list negatives. Preserve canonical text, untitled single-body behavior and shared projection/LLM caller regressions. Unsupported fixture coverage remains explicitly unverified.
  Evidence: Common now shares _heading_kind registry evidence between candidate detection and final filtering; unnumbered aliases remain region/layout evidence, not numbered-heading authority. Source-line, punctuation and TOC-leader guards remain. Common cache policy is common-source-preserving-v3; shared projection/LLM callers keep their existing validation contract. This delivery supersedes the deferred implementation status in the historical detailed-design diagnosis.
  Validation: 63 unittest cases pass, including six new matrix/regression methods covering 19 registered languages, LF/double-LF/CRLF, region markers, prose/list negatives, foreign-title layout fallback and source conservation. Four standalone regression scripts pass (region awareness, heading normalization, LLM region plans, manual projection). Updated the obsolete preface-discard assertion to require source preservation and identifiable body chapters.
  Real acceptance: verified nine immutable backup files, deleted only Madame Bovary and 红色赌盘_261014_145614 through lifecycle API, verified raw/profile/DB removal, reuploaded backup bytes and rebuilt Common Fast/progressive layout. content-recovery-20261021T093002.json records reset; common-language-20261021T094245.json records 652001/147401 preserved characters, 4 chapters/36 sections/425 units and 1 chapter/1 section/93 units respectively. Final-deployment content-recovery-20261021T094246.json verifies five interaction reads, unchanged profile hashes and largest-section HTTP reads of 0.240-0.497 seconds. Reports remain private in data/test-books/summary-note-20261021.
  Boundaries: registry selection uses the existing language input, not a new classifier or regenerated profile. UNKNOWN retains its explicit unsupported-language error; foreign headings can still qualify through independent layout evidence, not an automatic all-language scan. Controlled multilingual fixtures are not native-language corpus acceptance; real books exercise English and Chinese only. No new real LLM-enhance acceptance is claimed. Unrelated historical backlog remains unchanged.
  Timestamp: 2026-10-21

Supporting EXTRACT-10 (owned by scripts): locate source-to-generated title
downgrade failures before plan application, retain the semantic-boundary guard,
and cover the reported Madame Bovary failure with regression and live acceptance.
This is supporting scope of the existing real-book acceptance item, not a second
completion checkbox.

## Semantic Plan Ownership Refactor

Current track supersedes model-facing boundary/segment ID selection in ENH-REF.
Execute SEM-01 -> SEM-02 -> SEM-03 -> SEM-04; real-book acceptance is mandatory.

- [x] SEM-01: Define an ID-free semantic plan and full-source prompt.
  Acceptance: Model receives readable source and returns titles, hierarchy and exact short opening quotes. Extend the exact opening to disambiguate; no preceding-title field, model-facing IDs, offsets or occurrence counting. Reject unknown fields.
  Evidence: `semantic_structure_plan.py`, plain full-source framing, strict provider schema, `test_semantic_structure_plan.py`. Old V2 prompt explicitly renamed `legacy_plan_prompt` and removed from production selection.
  Timestamp: 2026-10-21
- [x] SEM-02: Resolve quotes uniquely and create executable positions on the backend.
  Acceptance: Exact canonical/readable-view mapping and unique quote matching; fail closed on missing/ambiguous/duplicate locations. Preserve all canonical characters. IDs remain backend-owned.
  Evidence: Exact quote resolution and backend-only executable-map adapter; tests cover repeated text, no fuzzy/whitespace repair, fragmentation, old-window crossings, duplicate failures, internal IDs and original-body reconstruction.
  Timestamp: 2026-10-21
- [x] SEM-03: Integrate semantic planning, bounded corrections and safe diagnostics.
  Acceptance: Production enhancement uses the new contract only; corrections refer to quote failures, never IDs. Retain full-request capacity/consent guards, atomic commit and automatic reading preparation. No silent legacy-output fallback.
  Evidence: v15 semantic provider adapter, native response schema and request-local output/timeout; exact quote/Unicode-indentation checks, batched role/boundary diagnostics, deduplicated source-evidence corrections and per-dispatch capacity/identity gates. Backend acceptance runs 180 tests: 179 pass, one explicitly opt-in private-source test skipped. Live acceptance is tracked separately in SEM-04.
  Timestamp: 2026-10-21
- [x] SEM-04: Accept real-book generation and isolated API/layout execution.
  Acceptance: Run the reported book repeatedly with the configured live model; require real structure change, exact body preservation and no original writes. Exercise real API/PostgreSQL/layout and regression/fault cases. Missing real-data prerequisites cannot count as success.
  Evidence: Final v15 private case `085747b65094d52f` succeeds in independent attempts `20261021T044839792942Z` (3 chapters/33 sections, 2 calls) and `20261021T044933593384Z` (4/27, 3 calls), with all 147401 canonical characters preserved and original snapshot unchanged. First accepted response passes real API/disposable PostgreSQL structural commit, artifact deletion, paid real embedding/layout resolver and prepare/read with 107 task units. Backend suite: 180 cases, 179 pass and one private-source opt-in skip; separate explicit live harnesses are not skipped. SEM track 4/4 (100%).
  Limits: Two final-code successes are incident acceptance, not a statistical reliability guarantee. Generated grouping/title quality remains variable; earlier development failures are retained privately, not counted as passes. Historical future tasks remain open.
  Validation: Final lifecycle entry passes UI build and browser suites at 1280px/390px; screenshots inspected and diff check clean. Local Compose API rebuilt, health endpoint passes and container reports `structure_plan_prompt_v15_semantic`. No production book reparse was performed.
  Timestamp: 2026-10-21

## Live Reference Rejection Follow-up

Historical mitigation only. ID-copying and ID-repair prompts below are superseded
by the Semantic Plan Ownership Refactor; passing this older track is not acceptance
of the new contract.

- [x] ENH-REF-01: Distinguish missing boundary references from invalid reference occurrence without logging source/model values.
  Acceptance: Retain strict rejection and existing public error contract; log bounded boundary index, reference membership, locator form and occurrence validity. Add privacy and valid/invalid reference regression tests.
  Evidence: `BoundaryReferenceRejected`, `rejection_diagnostic`, safe attempt/exhaustion logs; reference membership/occurrence/privacy regressions in `test_plan_rejection.py` pass.
  Timestamp: 2026-10-21

- [x] ENH-REF-02: Reproduce the live locator failure and validate a targeted correction.
  Acceptance: Use safe reference diagnostics to establish the actual failure subtype; preserve exact source coverage and prohibit guessed offsets/fuzzy matching. Report real-model outcome separately from deterministic tests. No reliability-complete claim until the reported book succeeds under an explicitly authorized live validation.
  Evidence: v11 located passages/end instructions; bounded validation-feedback correction with per-call capacity and consent-identity guards. Private real-book case `085747b65094d52f` succeeds twice independently (12 chapters/25 sections and 28/28, two provider calls each, exact 147401-character preservation). First result passes real API/disposable PostgreSQL commit, cleanup and real layout resolver (106 units). `accept_enhancement_real_book.py`, `verify_real_book_enhancement_api.py`, `test_enhancement_plan_repair.py`; private corpus ignored by Git. Production book/interactions unchanged.
  Limits: Two successes establish this incident's acceptance, not guaranteed model reliability or universal semantic quality. Future feature acceptance requires representative real data; mocks alone cannot close live-model incidents.
  Timestamp: 2026-10-21

## Enhancement Autonomy Follow-up

Current requirements and investigation: design section "Enhancement Autonomy And
Change Semantics (2026-10-20)". Combined sequence:
UI ENH-UX-01 -> UI ENH-UX-02 -> ENH-REV-01 -> db ENH-REV-02 -> app ENH-REV-03 ->
ENH-REV-04 -> UI ENH-REV-05 -> ENH-REV-06. New track 8/8 accepted (100%) as of 2026-10-21;
prior FULL 5/5 remains historical capacity-engineering acceptance.

- [x] ENH-REV-01: Classify unchanged, title-only and structural plans deterministically.
  Acceptance: Compare applied hierarchy on identical canonical source, ignoring technical IDs/timestamps; signatures include intervals, levels and parent/grouping. Labels differ separately. Invalid body changes fail before mutation. Report safe proposed/accepted counts and change kind without source text. Comparator does not call LLM or write storage.
  Evidence: `enhancement_change.py`; 20 passing `scripts/test_enhancement_change.py` tests cover ID-independent comparison, label changes, boundaries/levels/parent grouping/roles, Unicode/coverage, rejected source modification, safe diagnostics and stable old-node correspondence. Existing plan contract/application suite passes 42 tests.
  Notes: Pure helper only; no database/API integration or production behavior change yet. Diagnostics expose accepted applied counts; raw proposal counts remain owned by the planner.
  Timestamp: 2026-10-21

- [x] ENH-REV-04: Give LLM semantic planning autonomy without Common anti-splitting policy.
  Depends on: ENH-REV-01/02/03 so unchanged or title-only output cannot trigger destructive replacement.
  Acceptance: Remove the blanket uncertain-do-not-split and smallest-output bias from semantic instructions. Actively organize reasonable chapters/sections and generated titles without requiring printed headings. Keep source-origin honesty, body preservation and valid ordered locators; no chapter quota, model switch or automatic regeneration of a valid no-op. Transient LLM-call recovery is permitted under ENH-REV-03 without cost gates. Audit locator navigation/output allowance with evidence rather than assuming more tokens fixes quality.
  Evidence: `structure_plan_contract.py` v8; `llm_section_splitter.py` safe accepted-plan diagnostics; `test_enhancement_autonomy.py` plus prompt/application/metrics/full-source regressions verify autonomy, generated chapter/subsection hierarchy, Common parity, legitimate single body, exact body preservation and reference-cap anchor fallback. Design records locator/output audit and unavailable finish/usage fields at the text-only provider boundary.
  Validation: Ten-suite acceptance: 145 tests, 144 passed, one private-source opt-in skipped. No Common/validator/model/token-limit changes, paid LLM, production mutation or deployment. Live semantic-quality improvement remains unverified; deterministic fixtures are contract evidence only.
  Timestamp: 2026-10-21

- [x] ENH-REV-06: Accept three-way lifecycle and one-step reader recovery end to end.
  Depends on: ENH-UX-01/02 and ENH-REV-01..05.
  Acceptance: Isolated API/PostgreSQL/browser coverage proves unchanged=no version/write/cleanup; titles_only=stable structure version/IDs/interactions; structure_changed=one new version/full cleanup/automatic layout prepare before read. Transient LLM failure can retry successfully without cost gates; unknown/concurrent requests cannot duplicate commit/cleanup. Downstream recovery reuses completed generation. Capacity rejection spends zero generation calls. No source mutation, normal-path manual Prepare or hidden task-layout writes.
  Evidence: `scripts/accept_enhancement_lifecycle.py`, expanded `accept_enhancement_browser.py`, UI `test-enhancement-lifecycle-live.mjs`. Backend 155 tests: 154 pass, one private-source opt-in skipped. Confirmation/outcomes/reader-identity suites and real API/PostgreSQL browser tests pass at 1280px/390px; all three outcomes, nonempty saved payload retention, no-op row/xmin equality, title snapshot/IDs, transient retry, lost-response lookup, automatic prepare/read and preparation-only retry. Inherited tests cover updated consent, delayed writers, concurrency and transaction rollback. Build, screenshots and diff checks pass.
  Limits: Synthetic provider/unit resolver only; live semantic usefulness explicitly not evaluated. No paid calls, private-source processing, production-document mutation, Docker restart or runtime-code changes. Separate opt-in semantic review is specified in detailed design.
  Timestamp: 2026-10-21

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

- [x] Add a storage-independent explicit document deletion repository contract.
  Evidence: `document_artifact_repository.py`; `structured_document_artifact_repository.py`; `db/postgres_structured_document_artifact_repository.py`; passing `scripts/test_document_delete_route.py` and `scripts/test_postgres_document_delete.py`.
  Notes: File and PostgreSQL implementations delete owned structured storage without loading hierarchy or generating artifacts; absence returns false and storage errors propagate.
  Timestamp: 2026-10-14

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

- [x] Define hierarchy-aware artifact target validation boundary
  Evidence: `Deep_Reflective_Reader/document_structure/module-detailed-design.md (Future Direction Note: Artifact Target Validation Boundary Preparation)`; `Deep_Reflective_Reader/shared/module-detailed-design.md`; `Deep_Reflective_Reader/section_tasks/module-detailed-design.md`; `Deep_Reflective_Reader/progress.md`
  Notes: 明確收斂 ArtifactTargetRef 為 metadata/target intent（非 persistence truth），並固定 future repository trust boundary 必須先做 hierarchy-aware validation；補齊 stale-ref 語義、allowed target combinations、metadata glossary 與 fail-fast error boundary 的 future-direction 契約。

- [x] Add lightweight document discovery repository contract
  Evidence: `Deep_Reflective_Reader/document_structure/document_artifact_repository.py`; `Deep_Reflective_Reader/document_structure/structured_document_artifact_repository.py`; `Deep_Reflective_Reader/scripts/test_document_list_search_api.py`
  Notes: 新增 `DocumentListItem` 與 `list_documents(query, limit)`；file-backed implementation 掃描 `*.structured.json` 並只讀 title metadata，不把 hierarchy/content 作 list API payload。

- [x] Reject LLM split plans that resolve main-body sections to TOC-only spans
  Evidence: `Deep_Reflective_Reader/document_structure/llm_section_splitter.py`; `Deep_Reflective_Reader/scripts/test_llm_section_splitter_region_plan.py`; `PYTHONPATH=Deep_Reflective_Reader python Deep_Reflective_Reader/scripts/test_llm_section_splitter_region_plan.py`; `python -m py_compile Deep_Reflective_Reader/document_structure/llm_section_splitter.py Deep_Reflective_Reader/scripts/test_llm_section_splitter_region_plan.py`
  Notes: LLM split plan remains advisory; local deterministic validation rejects high-ratio tiny/heading-only `main_body` outputs and falls back to the common splitter instead of persisting a TOC-only hierarchy.

- [x] Record structured parser provenance on accepted StructuredDocument output
  Evidence: `Deep_Reflective_Reader/document_structure/structured_document.py`; `Deep_Reflective_Reader/document_structure/structured_document_builder.py`; `Deep_Reflective_Reader/document_structure/section_splitter_selector.py`; `Deep_Reflective_Reader/document_structure/llm_section_splitter.py`; `Deep_Reflective_Reader/scripts/test_llm_section_splitter_region_plan.py`
  Notes: Structured build records requested/effective parser mode and LLM fallback reason as advisory provenance; fallback validation remains deterministic and metadata does not become parser authority.

- [x] Clear derived interaction artifacts on parser-level hard reparse replacement
  Evidence: `Deep_Reflective_Reader/document_structure/structured_document_artifact_repository.py`; `Deep_Reflective_Reader/db/postgres_structured_document_artifact_repository.py`; `Deep_Reflective_Reader/scripts/test_hard_reparse_reading_interaction_cleanup.py`; `Deep_Reflective_Reader/scripts/test_postgres_manual_reparse_repository_boundary.py`; `.venv` execution of `PYTHONPATH=Deep_Reflective_Reader Deep_Reflective_Reader/.venv/bin/python Deep_Reflective_Reader/scripts/test_hard_reparse_reading_interaction_cleanup.py`; `.venv` execution of `PYTHONPATH=Deep_Reflective_Reader Deep_Reflective_Reader/.venv/bin/python Deep_Reflective_Reader/scripts/test_postgres_manual_reparse_repository_boundary.py`.
  Notes: `save_reparsed_document(...)` now sanitizes accepted replacement documents by clearing document-level, chapter-level, section-level, and task-unit-level task artifacts before persistence. This removes stale analysis/quiz/critical-thinking session payloads and referenced-artifact metadata from replacement structured payloads while leaving ordinary artifact/task-layout saves on their non-reparse paths.
  Timestamp: 2026-10-08

## Needs Confirmation

No unresolved confirmation items identified in this pass.

## Future Task Policy

New future tasks for this module must be added here first as unchecked items:

- [ ] Define hierarchy boundary for rich task-unit content model
- [ ] Prevent content-block model from becoming persisted hierarchy source
- [ ] Define migration strategy for string content compatibility
- [ ] Clarify content-block artifact targeting boundary without mutating chapter/section/task_unit identity
- [ ] Define segmentation boundary against hierarchy persistence
- [ ] Define resegmentation stale-target semantics
- [ ] Define future content-block persistence non-authority rule
- [ ] Define DB-centric structured persistence migration contract
- [ ] Preserve hierarchy-first StructuredDocument semantics across file and DB storage
- [ ] Define file-to-DB migration boundary for structured documents
- [ ] Define DB-backed repository validation rules before switching read path
- [ ] Prevent DB schema from reintroducing root sections[], structure_nodes[], or flat task_units as primary flow
- [ ] Define gradual retirement policy for data/ structured JSON after DB readiness validation
- [ ] Complete Phase 1 StructuredDocument JSONB-first evaluation
- [ ] Validate hierarchy parity criteria defined by the evaluation document
- [ ] Resolve readiness-audit gaps before Phase 1 StructuredDocument JSONB-first evaluation
- [ ] Define DB-era task-unit identity strategy before schema design
- [x] Define source-agnostic manual structure override contract
  Evidence needed: manual structure input is modeled as explicit parser input for any document type, not OCR-only and not task-layout mutation.
  Evidence: `Deep_Reflective_Reader/api_schemas.py`; `Deep_Reflective_Reader/main.py`; `Deep_Reflective_Reader/app/section_task_coordinator.py`; `Deep_Reflective_Reader/document_preparation/document_preparation_pipeline.py`; `Deep_Reflective_Reader/document_structure/manual_structure_projection.py`; `Deep_Reflective_Reader/document_structure/manual_structure_document_builder.py`; `Deep_Reflective_Reader/scripts/test_manual_structure_api_schemas.py`; `Deep_Reflective_Reader/scripts/test_manual_structure_validate_route.py`; `Deep_Reflective_Reader/scripts/test_manual_structure_commit_source_evidence.py`.
  Notes: Manual structure is modeled as explicit parser input for any source handled by the document loader factory, not OCR-only or task-layout mutation. Successful commit produces a normal hierarchy-only `StructuredDocument`; validation/preview remains non-mutating.
- [x] Define manual structure validation and preview boundary
  Evidence needed: validation checks anchors/ranges/levels before commit and returns explicit failure reasons without overwriting current structured artifacts.
  Evidence: `Deep_Reflective_Reader/document_structure/manual_structure_projection.py`; `Deep_Reflective_Reader/scripts/test_manual_structure_projection.py`; `Deep_Reflective_Reader/.venv/bin/python Deep_Reflective_Reader/scripts/test_manual_structure_projection.py`.
  Notes: Validation must reject empty titles, out-of-range anchors, overlapping ranges, empty ranges, incompatible levels, and partial projections. Validation failure must not silently fallback to common parser as a successful manual reparse.
- [x] Define manual structure projection into two-layer hierarchy
  Evidence needed: manual entries map deterministically into `chapters[].sections[]` while preserving original entry levels only as provenance.
  Evidence: `Deep_Reflective_Reader/document_structure/manual_structure_document_builder.py`; `Deep_Reflective_Reader/scripts/test_manual_structure_document_builder.py`; `Deep_Reflective_Reader/.venv/bin/python Deep_Reflective_Reader/scripts/test_manual_structure_document_builder.py`.
  Notes: Manual user-defined structure is limited to `chapter -> section` for now. Chapter-only plans create same-name sections; `level>2` and skipped-level plans fail-fast rather than being folded. No root `sections[]`, `structure_nodes`, flat `task_units`, or `Part -> Chapter -> Section` persistence is introduced. Draft building supports char-range anchors directly and page-range anchors when validated page-boundary evidence is available.
- [x] Define manual reparse provenance semantics
  Evidence needed: accepted manual reparse records requested/effective parser mode, user-supplied source, anchor type, validation summary, and entry count as advisory provenance.
  Evidence: `Deep_Reflective_Reader/document_structure/manual_structure_document_builder.py`; `Deep_Reflective_Reader/scripts/test_manual_structure_document_builder.py`; `Deep_Reflective_Reader/scripts/test_manual_structure_commit_source_evidence.py`.
  Notes: Accepted manual drafts record `requested_parser_mode=manual_structure`, `effective_parser_mode=manual_structure_projection`, `source=user_supplied_structure`, entry/chapter/section counts, source hash, anchor type, and validation summary as advisory `parse_provenance`. Provenance is observability only and does not become a second hierarchy source or parser authority.
- [x] Materialize page_range manual anchors into hierarchy drafts
  Evidence: `Deep_Reflective_Reader/document_structure/manual_structure_document_builder.py`; `Deep_Reflective_Reader/app/section_task_coordinator.py`; `Deep_Reflective_Reader/scripts/test_manual_structure_document_builder.py`; `Deep_Reflective_Reader/scripts/test_manual_structure_commit_source_evidence.py`.
  Notes: Manual draft building now resolves validated `page_range` anchors through preparation-provided page-boundary evidence into raw-text spans and produces hierarchy-only `StructuredDocument` output. Missing page-boundary evidence fails explicitly with `page_boundaries_required`; commit path passes source evidence boundaries into the builder and preserves existing artifacts on invalid drafts.

- [x] Reject ambiguous page-boundary evidence for manual page_range drafts
  Evidence: `Deep_Reflective_Reader/document_structure/manual_structure_document_builder.py`; `Deep_Reflective_Reader/scripts/test_manual_structure_document_builder.py`; `Deep_Reflective_Reader/scripts/test_manual_structure_commit_source_evidence.py`.
  Notes: Manual page-backed draft building now requires unique `page_index` values in preparation-provided page-boundary evidence. Duplicate page indexes fail with `ambiguous_page_boundary` before `StructuredDocument` materialization or repository save, preserving the existing structured artifact and keeping `char_range` as the explicit fallback.

- [x] Project existing structure anchor evidence for TOC edit prefill
  Evidence: `Deep_Reflective_Reader/document_structure/structure_anchor_evidence.py`; `Deep_Reflective_Reader/scripts/test_structure_anchor_evidence.py`.
  Notes: `project_structure_anchor_evidence(...)` projects existing chapter/section hierarchy spans into lightweight read-only anchor evidence. It prefers `page_range` when validated page boundaries cover existing spans, falls back to `char_range` when page evidence is absent, and marks missing/invalid spans as explicit unavailable evidence without changing hierarchy identity or artifact persistence.
- [x] Define deterministic table-of-contents detection contract
  Evidence needed: detection rules, confidence thresholds, evidence fields, and rejection conditions for missing, malformed, or ambiguous TOCs.
  Evidence: `Deep_Reflective_Reader/document_structure/toc_detector.py`; `Deep_Reflective_Reader/document_structure/structured_document_builder.py`; `Deep_Reflective_Reader/scripts/test_pdf_page_evidence.py`; `Deep_Reflective_Reader/scripts/test_toc_projection.py`.
  Notes: Detection and global validation are deterministic parser evidence. Missing page anchors, non-monotonic pages, invalid ranges, missing candidate groups, and insufficient matches reject projection; metadata and LLM classification remain advisory.

- [x] Define PDF structure-source precedence and fallback chain
  Evidence: `Deep_Reflective_Reader/document_preparation/document_preparation_pipeline.py`; `Deep_Reflective_Reader/document_structure/structured_document_builder.py`; `Deep_Reflective_Reader/doc_loaders/pdf_document_loader.py`; `Deep_Reflective_Reader/scripts/test_pdf_outline.py`; `Deep_Reflective_Reader/scripts/test_toc_projection.py`; container verification on `Deep_Reflective_Reader/data/raw/许三观卖血记.pdf`.
  Notes: `/Outlines` with valid destinations has priority over visual TOC inference. A malformed, empty, destination-less, or structurally inconsistent Outline is rejected rather than merged with OCR guesses. OCR TOC may authorize projection only after page/character validation; keyword matching remains the final conservative fallback.
- [x] Define native PDF Outline two-layer normalization for scanned-image books
  Evidence: `Deep_Reflective_Reader/document_structure/structured_document_builder.py`; `Deep_Reflective_Reader/document_structure/structured_hierarchy_builder.py`; `Deep_Reflective_Reader/scripts/test_pdf_outline.py`; container verification with existing native PDF outline fixture and synthetic level-0 outline regression.
  Notes: Native PDF Outline projection now normalizes the chapter level from the minimum outline level, skips a single root wrapper when present, flattens descendants as sections, maps ranges through PDF page indices plus page text boundaries, and rejects incomplete/non-monotonic/empty projections atomically. The resulting hierarchy remains `chapters[].sections[]`; root `sections[]`, `structure_nodes`, and `Part -> Chapter -> Section` persistence were not introduced.
- [x] Define TOC shape classification and hierarchy projection
  Evidence needed: flat chapter, chapter-section, and deeper hierarchy cases map deterministically to `chapters[].sections[]` without reintroducing root `sections[]` or `structure_nodes` as primary flow.
  Evidence: `Deep_Reflective_Reader/document_structure/toc_detector.py`; `Deep_Reflective_Reader/document_structure/structured_document_builder.py`; `Deep_Reflective_Reader/document_structure/structured_hierarchy_builder.py`; `Deep_Reflective_Reader/scripts/test_toc_projection.py`.
  Notes: Flat chapter entries become chapter-level projected sections and are materialized as one chapter with one section; chapter/section entries become one chapter with multiple sections; levels deeper than 2 collapse into the nearest projected section and concatenate content in source order. Original TOC level, projected level, printed page number, source page index, and merge reason are preserved in parse provenance only. Task-unit generation remains downstream of the resulting sections.
- [x] Define TOC-based page/character boundary validation and atomic fallback
  Evidence needed: monotonic page order, fuzzy title matching, non-overlapping spans, empty-range rejection, and all-or-nothing fallback to current parsing when evidence is insufficient.
  Evidence: `Deep_Reflective_Reader/document_structure/toc_detector.py`; `Deep_Reflective_Reader/document_structure/structured_document_builder.py`; `Deep_Reflective_Reader/scripts/test_toc_projection.py`.
  Notes: TOC projection now requires sufficient body-title matches, one compatible printed-page-to-source-page offset hypothesis, resolvable page anchors, and non-empty/monotonic/non-overlapping projected character ranges. If any gate fails, `StructuredDocumentBuilder` leaves the current parser result unchanged and does not emit `toc_projection` or claim `validated_toc_projection`.

- [x] Define universal page-level TOC candidate scoring
  Evidence needed: deterministic scoring covers horizontal/vertical text, left-to-right/right-to-left ordering, missing TOC markers, dotted leaders, separated page-number columns, circled page numbers, and OCR-fragmented titles.
  Evidence: `Deep_Reflective_Reader/document_structure/toc_detector.py`; `Deep_Reflective_Reader/scripts/test_pdf_page_evidence.py`; `Deep_Reflective_Reader/scripts/test_toc_projection.py`.
  Notes: Page-level candidate scoring now covers document-relative early position, horizontal/vertical writing mode, left-to-right/right-to-left ordering, multiple columns, coordinate OCR availability, short-title density, dotted leaders, page-number evidence, separated page-number columns, circled/boxed page numbers, OCR-fragmented title density, and missing-marker cases. Candidate detection remains separate from projection usability; later validation still controls splitting authority.

- [x] Define geometry-first TOC entry reconstruction
  Evidence needed: candidate pages reconstruct entries from title regions, leader lines, page-number regions, and column geometry even when linear OCR text is incomplete.
  Evidence: `Deep_Reflective_Reader/doc_loaders/pdf_page_evidence.py`; `Deep_Reflective_Reader/document_structure/toc_detector.py`; `Deep_Reflective_Reader/scripts/test_pdf_page_evidence.py`.
  Notes: OCR TSV analysis now preserves word-level geometry as page-local evidence. `TableOfContentsDetector.reconstruct_page_entries(...)` reconstructs horizontal TOC rows by aligned title/page-number regions and leader-line endpoint evidence, and reconstructs vertical RTL TOC columns by descending x-coordinate with top-to-bottom title ordering. Circled page numbers are parsed within bounded page-number regions. Reconstruction remains detection-only; later validation still controls projection usability.

- [x] Define logical reading-order normalization with coordinate preservation
  Evidence needed: normalized TOC candidates expose logical title/page pairs while retaining page index, region boxes, raw OCR, rotation, writing mode, and order hypothesis.
  Evidence: `Deep_Reflective_Reader/document_structure/toc_detector.py`; `Deep_Reflective_Reader/scripts/test_pdf_page_evidence.py`.
  Notes: `TableOfContentsDetector.normalize_page_reading_order(...)` now emits logical OCR tokens with page index, logical/group indexes, raw OCR text, normalized text, original boxes, confidence, writing mode, reading order, rotation, and order hypothesis. `reconstruct_normalized_page_pairs(...)` exposes auditable title/page pairs with region boxes, raw OCR sequence, evidence, confidence, and confidence breakdown. The normalized stream remains detection-only and does not replace raw text, page evidence, or hierarchy truth.

- [x] Define TOC-specific OCR quality gates
  Evidence needed: “page looks like TOC” is separated from “entries are reliable enough to split,” using title completeness, page-number coverage, geometric pairing, ordering consistency, and body-title recall.
  Evidence: `Deep_Reflective_Reader/document_structure/toc_detector.py`; `Deep_Reflective_Reader/scripts/test_pdf_page_evidence.py`.
  Notes: `TableOfContentsDetector.evaluate_toc_ocr_quality(...)` now separates `detected` from `usable_for_splitting` using reconstructed entry count, title completeness, page-number coverage, geometric pairing coverage, page-number ordering consistency, and body-title recall. High-scoring TOC-like pages with missing page numbers remain detected but unusable; non-monotonic page numbers are rejected; missing body text prevents usability. The gate preserves explicit rejection reasons and does not infer unreadable page numbers or mutate parser output.

- [x] Define page-number region recognition and global validation
  Evidence needed: Arabic, Chinese, Roman, circled, boxed, multi-digit, and fragmented page numbers map to document anchors under explicit offset hypotheses.
  Evidence: `Deep_Reflective_Reader/document_structure/toc_detector.py`; `Deep_Reflective_Reader/scripts/test_pdf_page_evidence.py`; `Deep_Reflective_Reader/scripts/test_toc_projection.py`.
  Notes: `TableOfContentsDetector.recognize_page_number_regions(...)` now preserves raw bounded page-number tokens, normalized value, numeral system, box, confidence, and evidence for Arabic, Chinese, Roman, circled, boxed, multi-digit, and fragmented OCR digits. `validate_page_number_anchor_offsets(...)` validates explicit printed-page-to-source-page offset hypotheses with monotonicity, plausible page anchors, multiple body-title matches, and atomic ambiguity rejection. This remains validation evidence only and does not mutate hierarchy, task-layout, profile, or artifacts.

- [x] Define OCR layout regression and negative fixtures
  Evidence needed: tests cover `暗水幽靈.pdf` artistic vertical TOC, `國富論.pdf` vertical body page, horizontal/multi-page TOC, circular numbers, artwork-only pages, and low-quality scans.
  Evidence: `Deep_Reflective_Reader/scripts/test_pdf_page_evidence.py`; `Deep_Reflective_Reader/scripts/test_toc_projection.py`; `Deep_Reflective_Reader/scripts/test_pdf_document_loader_inspection.py`; `Deep_Reflective_Reader/document_structure/toc_detector.py`; `Deep_Reflective_Reader/document_structure/structured_document_builder.py`.
  Notes: Regression coverage now fixes vertical RTL ordering, horizontal dotted/circled page-number candidates, OCR-fragmented title candidates without page anchors, artwork-only pages, low-quality OCR scans, ambiguous orientation rejection, non-monotonic page-number rejection, and layout-looking TOC fallback provenance. Projection remains all-or-nothing: visually obvious but unreliable TOC evidence does not emit `validated_toc_projection`, `toc_projection`, root `sections[]`, or `structure_nodes`.

- [x] Define layout-hypothesis normalization before TOC matching
  Evidence: `Deep_Reflective_Reader/document_structure/toc_detector.py`; `Deep_Reflective_Reader/scripts/test_pdf_page_evidence.py`.
  Notes: `normalize_page_reading_order(...)` and `reconstruct_normalized_page_pairs(...)` convert OCR words into logical TOC tokens/pairs while preserving page index, raw OCR text, original coordinates, confidence, writing mode, reading order, rotation/orientation, and order hypothesis. The normalized view remains detection/matching evidence only; raw OCR/page evidence and hierarchy truth are not replaced.

- [x] Define multi-page TOC grouping and termination rules
  Evidence needed: adjacent candidate pages can be grouped into one TOC region, with explicit start/end evidence and rejection when a page is front matter, artwork, or正文.
  Evidence: `Deep_Reflective_Reader/document_structure/toc_detector.py`; `Deep_Reflective_Reader/scripts/test_toc_projection.py`; `Deep_Reflective_Reader/scripts/test_pdf_page_evidence.py`.
  Notes: `group_page_candidates(...)` groups only adjacent layout-stable TOC candidates and records explicit start/end page indices plus evidence. Non-candidate pages, artwork-only pages,正文-like pages, layout changes, or gaps terminate the group; proximity alone is insufficient. Projection validation now requires at least one validated candidate group and records `toc_projection_page_group:<start>-<end>` provenance while preserving atomic fallback.

- [x] Define TOC global validation and parser authority boundary
  Evidence: `Deep_Reflective_Reader/document_structure/toc_detector.py`; `Deep_Reflective_Reader/document_structure/structured_document_builder.py`; `Deep_Reflective_Reader/scripts/test_pdf_page_evidence.py`; `Deep_Reflective_Reader/scripts/test_toc_projection.py`.
  Notes: `validate_for_projection(...)`, `evaluate_toc_ocr_quality(...)`, `validate_page_number_anchor_offsets(...)`, and the builder projection gate distinguish detection from split usability. Acceptance requires sufficient entries, page-number coverage, monotonic pages, plausible page ranges, body-title recall/body-anchor matches, compatible offset validation, non-empty projected ranges, and atomic all-or-nothing projection. Metadata, OCR classification, and LLM suggestions remain advisory; rejected plans leave the current parser result unchanged without `toc_projection`.

- [ ] Prepare reliable TOC PDF fixtures before enabling broad regression coverage
  Evidence needed: born-digital flat TOC, scanned horizontal TOC, vertical right-to-left TOC, multi-page TOC, chapter-section TOC, deeper hierarchy requiring merge, and negative artwork/front-matter cases.
  Notes: Each fixture must define expected TOC pages, printed page numbers, hierarchy levels, page/character ranges, and fallback behavior. `暗水幽靈.pdf` is currently a layout-identification case only because its OCR page-number anchors are unreliable.

After implementation, the task owner must update this checklist and mark the task as completed:

- [x] <completed task>

No coding task should be considered complete unless the corresponding module checklist is updated.

## Fragmented Native PDF / LLM Reparse Follow-up (2026-10-17)

New track: 10/10 combined (100%); this module owns 6 tasks (6 complete), section_tasks owns 3 (3 complete), UI owns 1 (complete).
Execution order and evidence: detailed design's dated investigation. Historical
completed PDF mode/TOC items are unchanged. NATIVE-01..06, INTERACT-01..03 and READER-01 are delivered. This completion covers isolated engineering acceptance, not deployment or live LLM quality. UI evidence is in its owned checklist. Updated: 2026-10-19.

- [x] NATIVE-01: Establish fragmented-native-text and source-coverage regression fixtures.
  Acceptance: Reproduce false `序` headings, dropped prefix, false TOC candidate, fragmented-prose guard rejection and cross-line anchor failure with synthetic fixtures; add opt-in read-only diagnostics for the two private PDFs. Record runtime/extraction identity and expected original coverage without committing private source text or calling LLMs.
  Evidence: `scripts/test_native_fragmentation_baseline.py` runs 10 tests: 5 safety controls pass and 5 desired-behavior contracts are explicit expected failures. `scripts/diagnose_native_fragmentation.py` inspects both private PDFs through fast snapshot extraction, reports hashes/runtime/page spans/coverage without source text, rejects subprocess calls and verifies unchanged source bytes. Existing LLM region-plan and TOC projection smoke tests pass. No parser/guard repair is claimed; follow-up owners remove expected-failure markers as contracts are implemented.
  Timestamp: 2026-10-19

- [x] NATIVE-02: Build a bounded source-mapped analysis view for native CJK fragmentation.
  Acceptance: Map reconstructed analysis intervals back to canonical source/page ranges; preserve paragraph/page evidence and exact raw hashes/spans. Negative controls retain genuine short headings, poetry/lists, Latin spacing and uncertainty. No renderer/OCR/LLM in fast mode; no canonical-text rewrite.
  Depends on: NATIVE-01. Consumers: common/LLM splitter and semantic guard.
  Evidence: `source_mapped_text.py` and `scripts/test_source_mapped_text.py` (11 tests passed). Immutable analysis view retains canonical source, text hash, original line/page intervals, confirmed ranges and exact character mapping. Default identity preserves uncertain text; only explicit caller-confirmed intervals permit CJK whitespace repair. Reject invalid/cross-page evidence and budgets rather than truncate. Fast/baseline combined run: 30 tests, 25 passed and 5 existing expected failures; LLM region-plan and TOC projection smoke checks pass.
  Scope: Pure mapping primitive only. NATIVE-03 owns fragmentation evidence selection and common-parser integration; NATIVE-05/INTERACT-01 own other consumers. No runtime parser, API, DB or user-document change.
  Timestamp: 2026-10-19

- [x] NATIVE-03: Correct heading/TOC evidence and conserve all source text in flat fallback.
  Acceptance: Characters inside words never become `序` headings; real prefaces/headings still work. Distinguish physical TOC from internal containers and rejected candidates. No-heading articles stay implicit single-body; source prefix and every nonempty source interval remain covered without duplicate content.
  Depends on: NATIVE-02. No profile classification as parser authority.
  Additional scope: Validate and propagate proposed toc_hint=auto/present/absent through explicit preparation paths; declaration is advisory. Absent skips dedicated TOC search, not body-heading recognition; present cannot fabricate a TOC. Common single-body output remains readable when otherwise ready. Distinguish native Outline bookmarks from printed TOC; retain original source on mismatch. Supporting API/preparation contract tests belong to this slice.
  Evidence: `common_parse_policy.py`, Common splitter/selector, TOC detector/builder, request schemas/routes and preparation/reparse adapters. `test_common_parse_policy.py`, baseline/mapping, PDF-mode preparation/lazy-layout/API suites: 68 tests, 66 passing and two follow-up expected failures. Heading normalization, LLM region-plan, TOC projection and structured reuse standalone regressions pass. Source slices stay canonical; explicit policy/hint changes invalidate reuse. No user-document rebuild or deployment; LLM prompt consumption remains NATIVE-04.
  Timestamp: 2026-10-19

- [x] NATIVE-04: Revise LLM plan schema and prompt for optional structure and executable anchors.
  Acceptance (corrected 2026-10-19): One explicit enhancement optimizes bounded Common results with source-mapped chapter/section navigation. Both legacy wire intents allow labelled semantic headings; they are not separate user choices. Generated origins never imply a physical TOC. Valid single-body output differs from malformed/empty output; regions are optional. Preserve exact source anchors/coverage and advisory TOC declarations. Outcome is derived from typed boundaries, not a redundant model label.
  Depends on: NATIVE-02. No live paid LLM call required for contract tests.
  Conservative-heading addition: Common and source-origin headings require locally verifiable evidence; newline/short length/keyword/confidence alone cannot justify a source heading. Explicit LLM enhancement may propose labelled semantic boundaries, subject to local source mapping and split validation. No implicit LLM call on read.
  Evidence: `structure_plan_contract.py`, `LLMSectionSplitter.build_candidate_plan`, intent request/adapters and `test_structure_plan_contract.py`. Seven unittest suites: 80 cases, 78 passing and two follow-up expected failures; legacy LLM region-plan smoke passes. V2 schema/prompt enforce typed outcomes/origins and bounded exact-source segments/anchors; API intent is validated/forwarded. Delivery is planning-only: old restore execution remains compatible, generated runtime application explicitly returns 400 before loading/generating/saving until NATIVE-05 activates the new applier. No live LLM or private-document change.
  Correction evidence: Prompt v3 supplies bounded Common context and unified semantics. Planner/application regressions cover default-intent semantic splitting, fenced JSON, derived outcomes/request identity/source order, safe unique boundary IDs, generated-origin correction, retained prefixes, invalid references and full source preservation. Current acceptance runner passes 174 tests, zero skips/expected failures. Historical staged-delivery evidence above is not current runtime policy.
  Prompt refinement: v4 adds meaningful-transition/minimal-segmentation guidance, argument/quotation/list cohesion, neutral titles, coherent chapter/section levels and output self-check. `test_prompt_v4_quality_and_self_check_contract` and the 175-case regression run pass. Validator/API/persistence behavior is unchanged; v4 live semantic quality is unmeasured. Earlier v3 live evidence remains historical, not a v4 quality claim.
  Timestamp: 2026-10-19

- [x] NATIVE-05: Apply validated LLM plans without restoring faulty common boundaries.
  Acceptance: Source-mapped cross-line matching, exact occurrence bounds, ambiguity rejection and bounded regex policy; valid single-body plan succeeds. Candidate selection preserves full coverage and does not force common boundaries back in. Requested/effective mode, fallback/count/coverage reasons are observable without raw prompt logging; public exposure is explicitly tested if extended.
  Depends on: NATIVE-03/04. Supporting app/API wiring stays read-only outside explicit reparse.
  Activation: Staged runtime guard removed; enhanced execution now uses v2 prompt/candidate validation and application, never the legacy merger. Request-local preview is shared between planning and application. Observed TOC evidence, source mapping, generated-origin persistence and explicit replacement tests are connected.
  Additional scope: Persist effective hint/intent, source-versus-generated origin and policy versions; include them in preparation reuse identity. Failed enhancement preserves usable common hierarchy/readiness/version/artifacts; only validated explicit replacement commits and invalidates derived data. Legacy missing identity and changed settings must not silently reuse incompatible preparation or trigger writes on read.
  Conservative-heading addition: Shared local acceptance policy covers Common, LLM source-origin headings and bookmark/TOC destination candidates. Source isolation, context and corroboration gate exact short-marker matches. Generated navigation uses its own safe-boundary gate within the same explicit enhancement. Uncertain boundaries retain source; rejected Common boundaries are not merged into v2 output. Policy identity is versioned; doubtful headings do not trigger additional OCR/LLM work.
  Evidence: `heading_acceptance.py`, `structure_plan_application.py`, planner/splitter/selector/builder and preparation updates; `test_structure_plan_application.py` (12 cases). Eight suites: 92 tests, 91 passing and one INTERACT-01 expected failure. Six standalone regressions pass, including artifact cleanup and profile ordering. API Prepare/Reparse tests use fake providers and isolated stores; failed enhancement saves neither hierarchy nor profile. Provenance/origins serialize through existing models. No live DB/LLM/private-document acceptance or deployment is claimed; NATIVE-06 owns that broader evidence.
  Correction evidence: Read-only live reproduction on the ritual-material PDF exposed `single_body_requires_one_boundary`, previously hidden by generic rejection. Safe allowlisted diagnostics retain causes without private text. The final live gpt-4.1-mini planner/application sample accepts 8,200 characters into five generated-origin sections with exact source coverage. Earlier live attempts exposed copied-title/reference and boundary errors addressed in the current design. API rebuilt; diagnostics never saved production hierarchy/profile/artifacts. This single sample is not a reliability benchmark or full live UI/API reparse acceptance. Failed enhancement still saves nothing.
  Timestamp: 2026-10-19

- [x] NATIVE-06: Validate hierarchy, interactions and explicit existing-document recovery end to end.
  Acceptance: Isolated storage checks both private cases plus synthetic positive/negative controls, unchanged fast/OCR/auto isolation, complete source coverage, all target levels, language and cache behavior. Exercise atomic explicit reparse/version/artifact invalidation and failed-reparse preservation; never repair user data during reads. Report unresolved LLM/live UI evidence honestly.
  Depends on: NATIVE-01..05, INTERACT-01..03, READER-01. This is one cross-module acceptance task, not extra tasks per supporting module.
  Additional scope: Cover three TOC declarations, contradictory/missing TOCs, bookmarks, both enhancement intents, common single-body readiness, generated heading labels, unchanged OCR selection, settings-sensitive reuse and failed-enhancement preservation across explicit preparation entry points.
  Conservative-heading addition: Exercise inline/quoted chapter references, ambiguous short markers, fragmented lines, poetry/lists and contradictory bookmarks as negative controls; real prefaces, numbered/unnumbered corroborated headings and single-chapter excerpts as positive controls. Assert complete source coverage, no implicit OCR/LLM calls and source/generated heading provenance separation within unified enhancement. Common uncertainty does not fail preparation; invalid enhanced candidates preserve the existing document.
  Evidence: `scripts/accept_native_recovery.py` passes 160 regression cases (no skips/expected failures), two real local PDFs with three Common hints each, 24 explicit valid/invalid enhancement attempts and 30 interaction/target combinations using fake providers and temporary file storage. Both sources retain exact bytes and complete canonical coverage; Common yields one chapter/section for each hint and fast extraction invokes no OCR/renderer. A disposable live PostgreSQL schema verifies version advancement, failure rollback, preservation of three artifact types on failure and cleanup after accepted replacement. Schema is removed in finally. Seven desktop/mobile Library/reader browser suites pass with mocked APIs.
  Limits: Live LLM semantic quality, full live browser-to-current-API integration, production latency and deployment remain unverified. Private-source task units are deterministic fixtures rather than embedding-generated units; layout/target contracts have separate regression coverage. No production document, profile or parser setting was changed. This acceptance does not close unrelated historical tasks.
  Timestamp: 2026-10-19

## Enhancement Prompt Reliability Follow-up

Scope: latest four v4 live candidate rejections and 647232-character assembled
prompt for 147401-character source. Design: Enhancement Prompt Reliability Plan.
New track progress: 6/6 (100% engineering and isolated integration acceptance).
Live model reliability and production deployment remain Needs Confirmation.
Historical completed items remain historical evidence, not proof of live reliability.

- [x] ENH-PLAN-01: Measure assembled prompt costs and freeze regression budgets.
  Acceptance: Reproduce serialization without LLM/storage writes; component character totals reconcile with the final prompt. Report real tokens or explicitly labelled estimates, source coverage and reference counts. Establish numeric overhead/total-budget acceptance caps on short, long and fragmented fixtures before compaction. Read private source only when necessary; no raw prompt/source logging.
  Evidence needed: Offline baseline fixture/report and accounting tests, including the reported long-source shape; explain observed 4.39x character expansion without claiming measured token inflation.
  Delivery: `plan_prompt_metrics.py`, both candidate entry points in `llm_section_splitter.py`, `test_plan_prompt_metrics.py`; exact component reconciliation, coverage/reference counts, privacy-safe pre-call metrics and timed response/provider-failure logs. Runtime token values are explicitly labelled UTF-8 byte upper estimates, not tokenizer/provider usage. Design records reproducible short/long/fragmented baselines and numeric future compaction caps.
  Validation: Metrics/contract suites 24/24 pass; broader application run 41/42, with one existing unconfirmed-reparse expectation failing at 409 confirmation_required, also reproduced with metrics bypassed. No paid calls, source mutation or deployment; compaction/schema/budget enforcement remain pending.
  Timestamp: 2026-10-19

- [x] ENH-PLAN-02: Align model-visible boundary schema with executable validation.
  Depends on: ENH-PLAN-01.
  Acceptance: Explicit exclusive boundary_id versus segment_id+anchor branches, consistent required/null rules, valid single/multi-boundary examples and both/neither negative cases. Update prompt/contract identity as needed; preserve exact-reference/source/title validation. No invented IDs or silent fallback success.
  Evidence needed: Schema-versus-runtime parity and prompt tests; provider-native schema enforcement remains unclaimed unless actually supported and tested.
  Delivery: `structure_plan_contract.py` exports exclusive reference branches with required/non-null locator fields and no misleading null defaults; prompt v5 adds executable fictional single/multi-boundary examples and explicit invalid locator forms. Runtime compatibility for redundant segment_id and exact-source/title/coverage validation remain unchanged; schema stays v2.
  Validation: 27 contract/metrics plus 17 application tests pass. Existing no-consent API test excluded, not repaired. Branch-shape/runtime matrix parity tested; no external JSON Schema engine or live model evaluation. Metrics show bounded +1605 characters, not compaction; task 03 targets unchanged. No deployment or data mutation.
  Timestamp: 2026-10-19

- [x] ENH-PLAN-03: Compact source context without losing executable references.
  Depends on: ENH-PLAN-01/02.
  Acceptance: One readable representation per visible span; canonical text/mapping retained locally. Remove routine duplicated analysis/raw text and repeated candidate preview text; bound Common context and metadata. Meet frozen overhead caps with equal source coverage, valid reference resolution and unchanged source bytes.
  Evidence needed: Before/after component/token accounting, long/short/fragmented regression cases and exact local coverage checks. Do not meet caps merely by hiding more source.
  Delivery: `compact_prompt_source` projects one readable text plus ID-to-display-position map, retaining canonical previews locally and rejecting unmappable references. Prompt v6, compact JSON, bounded Common titles/fields; metrics v2 supports both wire representations. Long fixture 642537 -> 192629 chars, references 501711 -> 52491, unchanged 120000 visible chars/3840 references; short 8215 and fragmented 9962 meet frozen caps. Tokens remain explicitly estimated, not provider usage.
  Validation: 30 metrics/contract plus 17 application tests pass; canonical mapping/no mutation/source coverage/caps checked. Prior no-consent API regression remains excluded, not repaired. No live model/production writes/deployment; full budget enforcement remains task 04.
  Timestamp: 2026-10-20

- [x] ENH-PLAN-04: Enforce the complete serialized request budget.
  Depends on: ENH-PLAN-03.
  Acceptance: Include instructions/schema/reference overhead, output reserve and safety margin. Bounded deterministic rebudgeting or explicit recoverable pre-call rejection; no hidden OCR, LLM fan-out or guessed omitted-range structure. Preserve source visibility and single-body restrictions.
  Evidence needed: Boundary/budget exhaustion tests, provider-call counts, missing-tokenizer fallback and oversized-schema cases; supporting provider/config changes require their module contracts to be inspected first.
  Delivery: `plan_prompt_budget.py`, both planner entries and builder reason mapping; optional provider-local tokenizer with OpenAI tiktoken implementation. Full prompt gate subtracts output reserve and max(256, ceil(5% input limit)); unavailable limits fail closed. No source truncation/retry; sanitized budget logs identify local count versus byte fallback. Public HTTP classification remains task 05.
  Validation: 45 budget/metrics/contract/candidate plus 17 application tests pass. Existing no-consent API regression excluded. Zero-call rejection, exact/over boundary, tokenizer failure and source-independent schema overhead tested; no live model/private data writes/deployment.
  Timestamp: 2026-10-20

- [x] ENH-PLAN-05: Surface safe actionable enhancement rejection diagnostics.
  Depends on: ENH-PLAN-02/04.
  Acceptance: Log safe stage/field path/index and missing-versus-conflicting-reference flags, no field values/raw input. Distinguish model validation, budget, provider and commit failures through compatible recoverable app/API responses instead of an unexplained 500. No automatic paid repair/retry.
  Evidence needed: Sanitization, error mapping and failure-preservation tests; load app/API module memory before supporting changes. Success-only atomic cleanup remains unchanged.
  Delivery: `plan_rejection.py`, planner/builder reason propagation and main reparse status mapping. Bounded stage/path/index and locator-presence diagnostics redact model values/unknown field names; API retains envelope with 422 invalid plan, 413 budget exhaustion, 502 provider failure, 503 unavailable/configuration, 500 internal failure. Unexpected enhanced exceptions are redacted; consent/commit conflicts unchanged. No automatic retry.
  Validation: 50 diagnostic/budget/metrics/contract/candidate plus 17 application tests pass, including direct mocked route results and sensitive-field rejection. Existing no-consent API regression excluded; full HTTP/storage acceptance remains task 06. No paid calls/production writes/deployment.
  Timestamp: 2026-10-20

- [x] ENH-PLAN-06: Validate compact planning and report reliability separately.
  Depends on: ENH-PLAN-01..05.
  Acceptance: Offline planner/application plus isolated API/storage regressions cover both valid reference branches, malformed/missing/conflicting references, multilingual/fragmented long input, omitted ranges, titles, duplicate positions and budgets. Failure preserves hierarchy/version/interactions; success commits once; zero extraction/OCR during enhancement.
  Evidence needed: Reproducible acceptance report with prompt size/coverage deltas. Live model reliability remains Needs Confirmation until separately authorized bounded evaluation records sample count, versions, rejection reasons, token use, latency and cost; synthetic pass rates are not live success rates.
  Delivery: `scripts/accept_enhancement_prompt.py` requires isolated DB configuration and refuses private-source opt-in. Extended PostgreSQL HTTP tests verify safe 422/413/503/502 responses, zero-call budget rejection, unchanged failure snapshots and source bytes; existing atomic cleanup/replay/rollback tests run. Corrected obsolete no-consent API expectation to 409 with no extra generation/save. Added long English/Chinese/French canonical coverage checks.
  Validation: 101 discovered, 100 passed, zero failures/errors, one intentional private-document opt-in skip. Unique PostgreSQL schema created/migrated/removed; live DB integration executed. Long fixture prompt-size/coverage caps remain passing. No paid model, production data mutation or deployment; live reliability remains explicitly Needs Confirmation.
  Timestamp: 2026-10-20

## Full-Source Enhancement Follow-up (3/5 Combined)

Execution order: FULL-01 in llm -> FULL-02 here -> FULL-03 in app -> FULL-04 in UI -> FULL-05 here.
This module owns two completed tasks; prior ENH-PLAN 6/6 is historical engineering acceptance.

- [x] FULL-02: Assemble complete source and reject requests exceeding dynamic capacity.
  Depends on: FULL-01.
  Acceptance: Remove enhancement-only fixed/head-tail excerpting; retain compact references and canonical mapping. Shared complete-prompt budget includes all overhead/reserves. Exceeding model capacity/unknown capability stops before LLM. Establish a measured configurable local resource ceiling with a distinct error; no truncation or automatic retries. Full coverage and omitted_ranges=[] are dispatch invariants.
  Evidence needed: Full 147401-character regression, multilingual coverage, numeric local-ceiling rationale, exact/over budget and oversized-schema tests, zero generation/storage calls on rejection. Audit preview consumers; do not alter unrelated workflows blindly.
  Delivery: Shared read-only full assembly in both production entries, prompt v7, contiguous-source dispatch invariant, configurable pre-mapping resources, v3 full-prompt budget plus transport allowance. Legacy numeric preview behavior retained. Resource errors preserve storage and map to HTTP 413; public capacity preflight remains FULL-03.
  Validation: 80 offline tests passed including 9 new full-source cases; measured 147401/2000000-character assembly at 8.95/121.46MiB peak Python allocation with zero LLM calls (details/limits in design). git diff --check passed; no live deployment or paid model evaluation.
  Timestamp: 2026-10-20

- [x] FULL-05: Validate full-source capacity and frontend failure handling end to end.
  Depends on: FULL-01..04.
  Acceptance: Isolated HTTP/storage/browser tests prove full-source dispatch, no truncation, preflight/execution parity, model/source/policy change invalidation, boundary limits, tokenizer fallback, local ceiling, zero-call rejection and unchanged hierarchy/interactions. Frontend shows accurate capacity and preservation reasons without implying billed usage or success.
  Evidence: `scripts/accept_enhancement_prompt.py` ran 139 tests (138 passed, one explicit private-source opt-in skipped); `scripts/accept_enhancement_browser.py` and UI `scripts/test-enhancement-live.mjs` passed at 1280px/390px through real FastAPI and disposable PostgreSQL. Exact 147401-character multilingual dispatch, rejected preflight/execution storage equality, zero generation on rejection, source/model/policy invalidation, successful atomic interaction cleanup and explicit layout recovery verified. `test-enhancement-confirmation.mjs`, UI build and `git diff --check` passed. No test schemas remained after cleanup.
  Scope: Deterministic synthetic provider only; live model semantic quality remains Needs Confirmation. No private-source reads, paid model calls or production deployment. Combined FULL track 5/5 (100%); backend 4/4.
  Timestamp: 2026-10-20

## Maintenance Notes

OCR-06 supporting evidence (counted only in the preparation checklist):
`StructuredDocumentBuilder.build` now requests lazy layout only after Outline
projection fails/is absent, while retaining eager evidence callers. Canonical
span/source validation, optional failure and provenance are described in design
section 32. `scripts/test_prepare_lazy_pdf_layout.py` passes 15 tests locally and
in isolated Compose on 2026-10-14; existing Outline/TOC/page-evidence regressions
pass. No extra checklist task or broader layout-roadmap completion is counted.

- This checklist is module memory for completed work.
- It does not replace the module detailed design document.
- It does not replace tests and test evidence.
- It does not replace proposal/HLD decisions and governance context.
