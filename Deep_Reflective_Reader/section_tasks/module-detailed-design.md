# section_tasks Detailed Design

## Reviewed Reading Sources (2026-10-21)

OCR-REVIEW-06: ReadingTargetResolver applies the pure reading overlay to block,
unit and ancestor content; revision and needs-review disposition travel with the
resolved target. Any unresolved region conservatively blocks all four generated
actions document-wide before secondary context or model calls. Notes remain usable.
Policy identity includes effective text hash and reading revision. Generated writes
check the revision under the repository mutation lock; summary writes also check
current content/version. Ancestor-summary writes use the same lock and reject
unreviewed/stale corrected content. Correction invalidates generated artifacts
document-wide rather than claiming selective descendant dependency tracking.
Notes resolve against original stable identity and expose source_stale after a
reading mutation; no silent note deletion. Source text/profile are not regenerated.

## Content Availability Snapshot (2026-10-21)

Supporting app CONTENT-01: DocumentReadingInteractionArtifactStore.from_snapshot
creates a request-local read-only adapter for an already loaded authoritative
StructuredDocument. It rejects another doc_name and provides no mutation adapter.
Analysis/quiz read accept an explicit snapshot store and retain normal server policy,
preflight and stale projection. Thinking current-session reads use the same snapshot.
No global cache, source mutation, model generation or profile construction is allowed.

## Enhancement Commit Result (2026-10-19)

PREP-03 adds nullable structure_version to ReparseDocumentStructureResult, populated
only from successful repository readback. It distinguishes committed replacement
from later task-layout readiness/reload failure; it is not client authority.
The preparation checklist records API/DB/browser acceptance.

## Enhancement Version Safety (2026-10-19)

Supporting ENH-DB-01: ResolvedReadingTarget carries optional server-owned
source_structure_version. Generation policy metadata includes it before LLM work;
the CommonArtifact adapter preserves it when no explicit version is supplied.
PostgreSQL rejects changed stale/unversioned artifacts rather than stamping delayed
results with the new structure version. File-backed targets without a storage version
retain their existing policy identity. Completion belongs to db, not this checklist.
Evidence: real PostgreSQL artifact-store delayed-write test, policy identity tests,
CommonArtifact model, artifact-store and app orchestration regressions.

## READER-01 Layout DTO (2026-10-19)

ParseProvenanceDTO adds nullable toc_hint, toc_observation, enhancement_intent,
implicit_single_body and declaration_unconfirmed. Section DTO adds nullable
title_origin. These are bounded read projections, not authority to change parsing,
hierarchy or interaction artifacts. No full source/prompt is added to layout.
Evidence: four `test_reader_provenance.py` cases; completion is owned by UI READER-01.

## INTERACT-03 Delivery (2026-10-19)

`interaction_policy_identity.py` owns `interaction_policy_v1`, persisted under
`metadata.interaction_policy`. It includes interaction type, effective server prompt
version, output-language policy/version and selected language, semantic/quantity
guard version, source-mapping policy version, SHA-256 of exact resolved target text
and the complete target hierarchy identity. Optional client prompt versions cannot
select runtime policy. Coordinator/orchestrator requests use server defaults;
services always stamp actual server metadata, including insufficient outcomes.
Source text and hierarchy are not rewritten. Exact source hashes capture changes
without trusting client source hashes; this is not a replacement for existing
structure-version checks or explicit reparse cleanup.

Analysis/quiz reads recheck current preflight and compare identities. Missing or
different identity yields `stale_target` with `interaction_policy_stale`, without
generation or interaction persistence. Stale analysis/quiz payloads are withheld.
Old insufficient outcomes become `not_generated` when source now passes, otherwise
the current preflight reason is returned. Explicit generate reuses completed output
only when policy and preflight match; explicit refresh always regenerates. A failed
generation does not overwrite an existing artifact. Successful generation uses
the existing hierarchy-aware save path and records the new identity.

The document artifact store remains a raw history/persistence boundary, not a cache
authority: its getters intentionally return old artifacts so orchestrators and
session history can inspect them. Real file-store round trips prove policy metadata
survives serialization. No schema migration, blanket deletion, profile write-back,
background regeneration or new read-path preparation behavior was introduced.

Thinking reads by latest target or explicit session ID project a stale status on
policy mismatch without mutating the stored status, question, answer, evaluation
or session ID. Current sessions are reused on ordinary reads. Explicit generate
creates a new session using the existing append/history behavior; old sessions
remain retrievable internally. Submit/retry still operate on the original stored
session and do not relabel its old question as current. Public responses retain
the existing 409/stale_target contract and session ID with null payload; stored
history is preserved, but showing that stale history in the UI is not delivered
here. Content-block availability uses the same policy projection and profile-first
language selection. No UI progress is claimed by this backend change.

Validation: eight new `test_interaction_policy_identity.py` cases exercise temporary
real JSON storage, each policy dimension, missing identity, changed source/language,
untrusted client versions, same-policy reuse, refresh failure, old insufficient
recovery, thinking history and real FastAPI-to-coordinator defaults/409 recovery.
Four unittest suites pass 38 tests. Twelve standalone scripts pass: three services,
analysis/quiz read-generate split, app orchestration, three route suites, artifact
store, artifact validity and hard-reparse cleanup. Existing reuse fixtures now
explicitly carry current server identity; legacy cases are tested separately.

Residual validation warning: `test_task_layout_artifact_availability.py` constructs
a root-sections-only document; it fails in hierarchy loading before interaction
cache logic. That legacy fixture is unchanged and is not counted as passing.
No live PostgreSQL, real provider, private-document write or deployment was run.
Current track: 8/10 (80%), section_tasks 3/3. Next is READER-01, then NATIVE-06.

## INTERACT-02 Delivery (2026-10-19)

`interaction_output_language.py` is the shared generation-only policy for analysis,
quiz, thinking questions and thinking evaluation. The existing coordinator chooses
profile language first; the policy consumes its `context.document_language`, then
falls back to `target.document_language` when metadata is unknown/unrecognized.
Only canonical codes/labels reach instructions. Explicit zh-Hans/zh-Hant are
preserved; plain zh never invents a simplified/traditional preference. Ancestor
summaries and secondary artifacts do not choose output language. No parser,
hierarchy, source content, schema, profile or read-path mutation was introduced.

Every generator appends an explicit instruction covering all user-facing fields,
while preserving keys, type codes, booleans, numbers, names and brief quotations.
Insight's thesis/detail/contrast and non-summary rules remain unchanged. Unknown
language instructs the model to follow Primary Source, not default to English.
Thinking sessions snapshot the effective document fallback in existing context
metadata so evaluation preserves the question-generation language even if the
user answers in another language. Legacy sessions without metadata follow the
question's language, never the user's answer. No legacy session is rewritten.

Validation runs on parsed, normalized generated payloads before success. The typed
`OutputLanguageMismatch` has stable reason `output_language_mismatch`, routed to
existing `generation_failed` or `evaluation_failed` states. There is no automatic
retry. Failed evaluation retains the original question, user answer, identity and
metadata; rejected generated text is not returned as a successful payload.
Machine fields and numeric scores are excluded. Brief quoted spans are excluded
from script checks. A zh field is rejected only with no Han characters and at
least 40 Latin letters/eight words; an en field only with at least 20 Han
characters and fewer than four Latin letters. Short, mixed, uncertain and other
language pairs abstain. This is a conservative script guard, not a universal
language detector or a simplified/traditional script verifier. Full-field quotes
can also escape this heuristic; prompt constraints remain the first line of defense.

Prompt versions are analysis v6, quiz v2 and thinking v2. This does not enforce
server-owned cache identity or repair stored English results: INTERACT-03 owns
that work. Existing cached artifacts may still be served until explicit recovery.

Evidence: `test_interaction_output_language.py` has 12 cases covering precedence,
unknown/variants, all prompt entry points, individual generated fields, positive
and negative Chinese/English cases, quotes/names, no retries, session fallback,
legacy sessions and foreign-language user answers. With fragmentation baselines,
30 unittest cases pass. Thirteen standalone regressions pass: three services,
thinking generator, prompt context, app orchestration (including real coordinator
profile precedence), artifact validity, analysis/quiz read-generate split, three
interaction route suites and thinking abstraction. All providers are fake; no paid
LLM, Docker deployment, live DB migration or private-document mutation occurred.
Current combined progress: 7/10 (70%); this module 2/3. Next: INTERACT-03.

## INTERACT-01 Delivery (2026-10-19)

`artifact_validity.py` now evaluates semantic checks against the bounded
`common_analysis_view`, using NATIVE-02 mapping and NATIVE-03 sentence-bearing
CJK fragmentation evidence. This is a validation-only view: request content,
source hashes, hierarchy identifiers, persisted text and prompt evidence are not
rewritten. Existing minimum character/alphanumeric thresholds and stale-target
precedence remain unchanged. No OCR, provider call or profile fallback is added.

Repetition is evaluated on repaired tokens rather than individual PDF glyphs;
dominant repeated CJK characters remain rejected. Original and analysis-view TOC
checks both run. The fragmented-heading line-count test is bypassed only when
actual whitespace repairs occurred and over half of CJK character mass lies in
lines longer than the existing 12-character heading threshold. This handles
partially repaired prose without letting a small prose appendix rescue a TOC.
Repetition and OCR-noise checks still apply. These are deterministic heuristics,
not proof that any arbitrary text has meaning.

The source-map budgets (2,000,000 characters and 4096 evidence ranges) bound work;
exhaustion returns an explicit semantic-analysis/evidence-budget reason through
the existing insufficient-content contract, never a truncated acceptance or 500.
Logs contain policy `fragment-evidence-v2`, decision/reason and counts of source
characters, readable characters, lines, evidence ranges and removed gaps. They
exclude document names, IDs, source text and provider output. Policy identity is
diagnostic only in this task; persistence/cache enforcement belongs to INTERACT-03.

Only analysis opts into semantic checks today. Quiz and critical thinking retain
quantity-only checks; their behavior was separately verified, not inferred from
analysis. Eight new tests include all 15 combinations of three services and five
resolved target levels, exact source/identity preservation, tiny-input rejection,
repeated text, heading lists, TOCs, noise, normal prose, partial repairs, budgets
and log privacy. Six unittest suites pass 61 cases with no expected failures;
five standalone validity/service/resolver regressions also pass.

Read-only local fast extraction (Python 3.12.2, pypdf 6.9.2) now accepts both
reported PDFs: `家醜不可外揚` (3557 characters, 1241 nonempty lines) and
`《歸來》白事／喪葬／鄉村儀式素材整理稿` (8190 characters, 2688 lines).
Both retain original file hashes and complete source coverage. This is local
guard evidence, not deployed/private-document LLM acceptance. Existing cached
failures are not rewritten; INTERACT-03 owns explicit recovery. Combined progress
is 6/10 (60%), this module 1/3; next is INTERACT-02 output language.

## Investigation: Fragmented Prose And Output Language (2026-10-17)

Historical investigation; INTERACT-01..03 are delivered above.
The implementation index is owned by the dated investigation in
`document_structure/module-detailed-design.md`. This module owns INTERACT-01..03.

### Confirmed Evidence

Read-only PostgreSQL inspection of `家醜不可外揚` (document 32, version 2)
shows 3590 section characters and three task units totaling 3576 characters.
Document/chapter analysis failures explicitly say `target content looks like
fragmented headings rather than readable source text for analysis`, not `too short`.
Hierarchy aggregation exists in `ReadingTargetResolver`; this case does not prove
a return of the earlier empty-document-source bug. PDF extraction creates many
single-character lines, which meet `_looks_like_fragmented_headings` thresholds.
The running pypdf 6.17.0 diagnostic reproduces rejection; using the existing
normalized view removes it. That experiment is causal evidence, not a safe fix:
blind normalization could disguise actual heading lists or corrupt source offsets.

The first stored block analysis is English, the following one Chinese, while both
carry `context.document_language=zh`. `analysis_interaction_llm_generator.py`
v5 provides English instructions and contextual language metadata, but no explicit
output-language requirement. Quiz has the same missing instruction risk; critical
thinking already adds one. This explains an unconstrained output opportunity,
not a deterministic claim about every model response. No new LLM call was made.
No DB `document_profile` rows exist for these two documents; this alone is not a
missing-profile diagnosis because the active profile path can be file-backed.

Only analysis failures were confirmed in stored artifacts. Quiz and thinking
at document/chapter level still need separate reproduction; analysis alone enables
the semantic guard. Do not generalize its failure to all three services as fact.

### Planned Behavior

- INTERACT-01 consumes NATIVE-02's evidence-backed analysis view for semantic
  checks, while keeping exact selected source as prompt evidence and unchanged
  hierarchy identifiers. Distinguish low text quantity, extraction fragmentation,
  actual heading/index content, repeated boilerplate and OCR noise. Bound work and
  emit metadata-only diagnostic counts (raw/readable size, line fragmentation,
  decision/reason/version). Do not globally relax thresholds, trigger OCR/LLM
  from guard evaluation, or treat profile summaries as replacement source text.
  Confirm all target levels; preserve existing meaningless-content negatives.
- INTERACT-02 shares profile-first, document-fallback output-language selection
  across analysis, quiz and critical-thinking question/evaluation. Language is
  advisory to generation, never parser authority. Explicitly require every
  user-facing JSON field in that language, allowing brief original quotations,
  names and technical terms; keys/type codes stay stable. Unknown language gets
  a documented source-language fallback, not forced English. Test conflicting
  ancestor/secondary context, Chinese/English, mixed quotations and missing profile.
  Add conservative mismatch detection with typed failure and no unbounded retry;
  do not persist a clearly mismatched result as completed. Language variants
  must use supported profile evidence, not invent a simplified/traditional choice.
- INTERACT-03 makes effective prompt/language/guard/source policy identity
  server-owned and persisted; prompt constants alone are insufficient. Existing
  reported artifacts omit effective prompt versions. The analysis orchestrator
  reuses completed artifacts unless refresh is requested, so a prompt-only fix
  cannot guarantee old English output disappears. Audit store reuse as well as
  orchestrator checks; missing/old identity is unverified, not automatically current.
  Reads remain side-effect-free and surface recoverable stale/not-generated state
  under the existing supported API contract; regeneration happens only on explicit
  generate/refresh. Preserve valid same-version reuse and critical-thinking session
  history; never create a new session just by opening a drawer.
- Existing insufficient-content reads already re-run preflight and can return
  `not_generated` when source becomes eligible. Preserve that recovery, and test
  cached bad status versus wrong-language completed artifacts separately. Parser
  fixes need explicit versioned reparse; language-only fixes need no hierarchy
  rewrite or destructive deletion of unrelated artifacts/user answers.

Validation uses stub generators and isolated stores first. Real LLM adherence,
exact historical failed anchors and unobserved quiz/thinking failures remain
Needs Confirmation until explicit evidence exists. No new runtime/schema guarantee
is claimed by this planning document.

## 1. Module Purpose

`section_tasks/` 定義 section/chapter/task-unit 任務層：
- task unit resolve/split
- task layout DTO/projection model
- summary/quiz 任務 prompt/service
- cache validity 相關輔助
- artifact availability runtime projection（供 task-layout/read path 可觀測）

**[Code-Confirmed]**

## 2. Position in Overall Architecture

- Task Layout / Task Unit Layer

## 3. Key Files

| File | Responsibility | Notes |
|---|---|---|
| `section_tasks/document_task_layout.py` | task-layout DTO contract（chapters-first + diagnostics DTO） | public response 映射來源之一 **[Code-Confirmed]** |
| `section_tasks/task_unit_resolver.py` | task unit resolution 主入口 | 使用 `get_effective_sections`（hierarchy-first） **[Code-Confirmed]** |
| `section_tasks/reading_target_resolver.py` | document/chapter/section/task-unit reading interaction target resolution | hierarchy-only, id-based, parent consistency checks **[Code-Confirmed]** |
| `section_tasks/reading_interaction_service_contracts.py` | target-agnostic reading interaction request/result/service contracts | validated DTOs, no route or persistence ownership **[Code-Confirmed]** |
| `section_tasks/analysis_interaction_service.py` | target-agnostic analysis interaction service | strict JSON validation, insufficient-content fast path **[Code-Confirmed]** |
| `section_tasks/analysis_interaction_llm_generator.py` | LLM-backed analysis generator adapter | interpretive strict-JSON insight prompt over one resolved target; treats `summary` as insight thesis, grounds insight in concrete primary-source details, asks for complexity/tension moves, and guards against recap-dominated output **[Code-Confirmed]** |
| `section_tasks/analysis_interaction_orchestrator.py` | analysis read/generate split orchestration | read does not generate; explicit generate persists only completed/insufficient-content artifacts **[Code-Confirmed]** |
| `section_tasks/quiz_interaction_llm_generator.py` | LLM-backed quiz generator adapter | strict JSON quiz prompt over one resolved target **[Code-Confirmed]** |
| `section_tasks/quiz_interaction_service.py` | target-agnostic quiz interaction service | valid quiz types, configured target-level limits, answer payload validation, insufficient-content fast path **[Code-Confirmed]** |
| `section_tasks/quiz_interaction_orchestrator.py` | quiz read/generate split orchestration | read does not generate; explicit generate persists only completed/insufficient-content artifacts **[Code-Confirmed]** |
| `section_tasks/critical_thinking_llm_generator.py` | LLM-backed critical-thinking question/evaluation adapter | strict JSON question/evaluation prompt over one resolved target/session **[Code-Confirmed]** |
| `section_tasks/critical_thinking_session_service.py` | target-agnostic critical-thinking session service | fixed instructions, question -> answer -> evaluation lifecycle, retryable evaluation failure **[Code-Confirmed]** |
| `section_tasks/reading_interaction_prompt_context.py` | shared evidence-priority prompt context assembly for reading interactions | labels primary source, ancestor summaries, document profile/language, and secondary artifact signals without concatenating raw ancestor text **[Code-Confirmed]** |
| `section_tasks/reading_interaction_common_artifact.py` | maps reading interaction service artifacts to/from the shared common artifact entity | preserves hierarchy-aware target metadata without owning persistence **[Code-Confirmed]** |
| `section_tasks/reading_interaction_artifact_store.py` | document-backed current analysis/quiz artifact, critical-thinking session, and ancestor summary stores | persists `CommonArtifact` payloads and ancestor summary artifacts under document task artifact metadata without hierarchy mutation **[Code-Confirmed]** |
| `section_tasks/ancestor_summary_context.py` | ancestor summary prompt-context contracts and explicit preparation service | defines hierarchy-aware ancestor summary target/artifact/context metadata, pure readiness/staleness policy, and generate/refresh-only preparation without persistence orchestration ownership **[Code-Confirmed]** |
| `section_tasks/task_unit_split_resolver_selector.py` | split mode resolver selector | semantic/progressive/llm resolvers **[Code-Confirmed]** |
| `section_tasks/heuristic_task_unit_split_resolver.py` | deterministic split path | semantic boundary scoring integration **[Code-Confirmed]** |
| `section_tasks/llm_task_unit_split_resolver.py` | llm split path | fallback to heuristic path **[Code-Confirmed]** |
| `section_tasks/task_unit_id_normalizer.py` | task unit id normalization and parent identity fill | hierarchy scope uniqueness support **[Code-Confirmed]** |
| `section_tasks/chapter_summary_service.py` | summary generation service | consume context/prompt factory + LLM **[Code-Confirmed]** |
| `section_tasks/chapter_quiz_service.py` | quiz generation service | consume context/prompt factory + LLM **[Code-Confirmed]** |
| `section_tasks/section_task_context_builder.py` | section/task unit context build | hierarchy-only lookup path **[Code-Confirmed]** |
| `section_tasks/artifact_validity.py` | artifact validity and interaction preflight contract | cache validity, stale target context, and insufficient-content reason surface **[Code-Confirmed]** |

## 4. Main Responsibilities

1. 定義 task-layer DTO 與 task mode 表達（direct/split/merged）。 **[Code-Confirmed]**
2. 根據 document structure 解析 task units。 **[Code-Confirmed]**
3. 建立 summary/quiz prompt + context 並調用 LLM 服務。 **[Code-Confirmed]**
4. 提供 task artifact validity 判斷契約。 **[Code-Confirmed]**
5. 支援 hierarchy-first target resolution 所需 context。 **[Code-Confirmed]**

## 5. Non-Responsibilities

1. 不負責 API route mapping。 **[From HLD]**
2. 不應在 read-only task-layout endpoint 直接做 persistence write（由 coordinator/repository 管控）。 **[From HLD]**
3. 不應把 profile metadata 直接當 parser rule。 **[From HLD]**
4. 不應污染 `StructuredDocument` persistence contract（不得重新引入 root sections mirror / flat task_units 作 persisted truth）。 **[Code-Confirmed] + [From HLD]**
5. 不負責 artifact persistence truth 定義；artifact source-of-truth 與 hierarchy persistence 邊界由 `document_structure` + repository 層承載。 **[Code-Confirmed] + [From HLD]**

## 6. Important Data Structures / Contracts

- `DocumentTaskLayout`
- `DocumentTaskLayoutChapterDTO`
- `DocumentTaskLayoutSectionDTO`
- `TaskUnitDTO`
- `ProfileStructureDiagnosticsDTO`
- `TaskUnit`
- `ArtifactValidityResult`
- `ResolvedReadingTarget`
- `ReadingInteractionRequest`
- `ReadingInteractionArtifact`
- `ReadingInteractionService`
- `CommonArtifact`
- `CommonArtifactTarget`
- `ReadingInteractionTargetValidity`
- `ReadingInteractionValidityPolicy`
- `ReadingInteractionPreflightResult`
- `AnalysisInteractionArtifactStore`
- `AnalysisInteractionOrchestrator`
- `QuizInteractionArtifactStore`
- `QuizInteractionOrchestrator`
- `ArtifactAwareContextResult`（secondary artifact context metadata）

## 7. Read/Write Boundary Matrix

| Path | Responsibility | Mutation Allowed |
|---|---|---:|
| task-layout projection | assemble chapters/sections/task_units DTO | No |
| artifact availability projection | expose has_summary/has_quiz/cache_valid/invalid_reason as runtime view | No |
| task-unit resolve/split | compute runtime/refresh units | Depends on caller path |
| section/chapter summary/quiz service | generate content payload | No direct store write |
| artifact persistence | handled by coordinator + repository (`document_structure` boundary) | Yes |

補充：`/documents/task-layout` 是 projection/read path，不應 hidden mutation。 **[From HLD] + [Code-Confirmed]**

## 8. Public API Boundary Clarification

1. `DocumentTaskLayout` 內含 `sections` / `task_units` / `chapter_artifacts` 屬 internal transitional DTO fields。 **[Code-Confirmed]**
2. `/documents/task-layout` public response 由 `main.py` 映射為 chapters-first，未返回 top-level `sections` / top-level `task_units` / top-level `chapter_artifacts`。 **[Code-Confirmed]**
3. `profile_diagnostics` 是 runtime projection；task-unit coverage 取自 current layout sections，不是 persisted profile snapshot overwrite。 **[Code-Confirmed]**
4. diagnostics projection 不應回寫 profile artifact。 **[Code-Confirmed] + [From HLD]**

## 9. Artifact Governance and Projection Boundary

1. persisted hierarchy truth 來自 `StructuredDocument` 的 hierarchy path（`chapters[].sections[].task_units[]`）；section_tasks 不定義 hierarchy truth。 **[Code-Confirmed] + [From HLD]**
2. section/chapter/task-unit artifacts 是 interaction output（summary/quiz 等），不是 hierarchy identity source。 **[Code-Confirmed] + [From HLD]**
3. section_tasks 負責的是 artifact availability 的 runtime projection，不是 artifact persistence authority。 **[Code-Confirmed]**
4. task-layout diagnostics/availability 不得寫回 profile，不得覆寫 persisted profile snapshot。 **[Code-Confirmed]**
5. API response DTO shape 是 transport contract，不等於 persistence schema。 **[Code-Confirmed]**

## 10. Target Resolution Priority

1. chapter-level target：`chapter_id` 優先。 **[Code-Confirmed]**
2. `chapter_title` 只作相容路徑；ambiguous 時應 fail-fast。 **[Code-Confirmed]**
3. section-level target：以 hierarchy section id 為核心。 **[Code-Confirmed]**

## 11. Module Relationships

- depends on:
  - `document_structure/`
  - `profile/`（作為 prompt/diagnostics 輸入）
  - `llm/`
  - `shared/`
- used by:
  - `app/section_task_coordinator.py`
- projection relationship:
  - `document_task_layout.py` 是 projection DTO contract（含 artifact availability/runtime diagnostics 投影）
- advisory relationship:
  - consume profile/document metadata，但非 parser authority

## 12. Main Flows Involving This Module

1. task-unit resolution flow（split mode dependent）。 **[Code-Confirmed]**
2. section/chapter summary flow。 **[Code-Confirmed]**
3. section/chapter quiz flow。 **[Code-Confirmed]**
4. task-layout projection data contract flow（供 coordinator/API mapping）。 **[Code-Confirmed]**
5. artifact validity flow（cache_valid + invalid_reason）。 **[Code-Confirmed]**
6. analysis interaction read/generate split flow: read returns current persisted artifact or `not_generated` without generator calls; persisted insufficient-content artifacts are preflight-rechecked on read and generate so stale insufficient results do not mask now-readable targets; explicit generate reuses completed artifacts unless refresh is requested, and only completed or insufficient-content results are persisted. **[Code-Confirmed]**
7. quiz interaction read/generate split flow: read returns current persisted artifact or `not_generated` without generator calls; persisted insufficient-content artifacts are preflight-rechecked on read and generate so stale insufficient results do not mask now-readable targets; explicit generate reuses completed artifacts unless refresh is requested, and only completed or insufficient-content results are persisted. **[Code-Confirmed]**
8. analysis/quiz output parsing flow: service validation accepts a strict JSON object directly, a fenced JSON object, or the first JSON object after model-prefixed text; extracted payloads still pass the same required-field, item-type, answer, and limit validation before completion. **[Code-Confirmed]**

## 13. Persistence / Side Effects

- read persistence：間接（由 coordinator/repository 提供 document/artifacts）
- write persistence：部分（service 層通常不直接落盤；document-backed interaction store adapter delegates explicit artifact updates to repository）
- mutate structured document：否（主要由 repository/coordinator）
- generate runtime projection：是（DTO contract）
- call LLM：是（summary/quiz/some split resolvers）
- diagnostics only：部分（artifact validity / DTO）
- artifact availability state：runtime projection（非 persisted truth）

## 14. Known Legacy / Compatibility Behavior

1. `DocumentTaskLayout` 仍保留內部 transitional top-level fields（sections/task_units/chapter_artifacts）供 backward-compatible 測試/mapper。 **[Code-Confirmed]**
2. public API 已 chapters-first，不應依賴 transitional fields。 **[Code-Confirmed]**
3. `section_task_context_builder` 已 hierarchy-only，不再默認 legacy root sections fallback。 **[Code-Confirmed]**

## 15. Terminology Governance Audit

| Term | Current Meaning in This Module | Classification | Governance Decision |
|---|---|---|---|
| top-level sections | 僅 internal transitional DTO field (`DocumentTaskLayout.sections`)，非 public API contract | **[Code-Confirmed]** | 不得描述為 `/documents/task-layout` response source of truth |
| top-level task_units | 僅 internal transitional DTO field (`DocumentTaskLayout.task_units`) | **[Code-Confirmed]** | 不得描述為 persistence contract 或 public API main path |
| chapter_artifacts (top-level) | internal transitional map；chapter artifact availability public path 走 `chapters[].artifacts` | **[Code-Confirmed]** | 不得在文檔描述為 public response required field |
| artifact source of truth | 本模組不定義 artifact truth；僅消費 repository/hierarchy-aware 結果做可觀測投影 | **[Code-Confirmed] + [From HLD]** | 不得描述為由 task-layout DTO 決定 persisted truth |
| artifact persistence | write path 在 coordinator + repository，不在 task-layout projection DTO | **[Code-Confirmed]** | 不得把 projection path 描述為 write authority |
| artifact availability | runtime/projection state（has_summary/has_quiz/cache_valid/invalid_reason） | **[Code-Confirmed]** | 不得描述為 persistence snapshot overwrite |
| availability cache / artifact validity | 僅可觀測 cache-validity 訊號，不改 hierarchy identity | **[Code-Confirmed]** | 避免被誤讀為 hierarchy source |
| top-level legacy mirrors（deprecated wording） | 歷史/文檔語境用詞，正式術語已收斂為 `transitional internal field` / `compatibility-only field` | **[Maintainer-Confirmed] + [Doc-Confirmed]** | 僅允許 historical/deprecated/compatibility 說明，不得作現行 contract terminology |
| flat task_units | historical/compatibility語境，非 hierarchy persistence truth | **[Code-Confirmed] + [From HLD]** | 禁止重新引入 flat persisted semantics |
| artifact mirror（deprecated wording） | 僅 historical/migration 搜尋關鍵字；正式術語改為 `transitional internal field` | **[Maintainer-Confirmed] + [Doc-Confirmed]** | 不得作為現行 contract terminology、persistence authority 或 hierarchy truth 描述 |
| write-back / mutation | task-layout request path 應視為 projection/read；artifact write path 需顯式呼叫 repository update | **[Code-Confirmed] + [From HLD]** | 禁止 hidden mutation |
| profile diagnostics | runtime mixed-source projection，非 persisted profile body | **[Code-Confirmed]** | 禁止描述為 profile snapshot overwrite |
| root sections mirror | 不屬 section_tasks contract；不得在本模組文件暗示可回退為主路徑 | **[Code-Confirmed] + [From HLD]** | 維持 hierarchy-first contract 邊界 |
| API response shape | transport DTO（chapters-first） | **[Code-Confirmed]** | 不等於 persistence schema |

### Terminology Validation Notes

1. 本模組文檔已明確區分 internal transitional DTO fields 與 public chapters-first API contract。  
2. task-layout 與 diagnostics 皆定義為 runtime projection；不承擔 profile write-back。  
3. section/task-unit/chapter artifact availability 的讀取語義固定為 hierarchy-aware path（read-side observability）。  
4. `artifact mirror` 一詞已降級為 deprecated terminology；正式 contract 用語統一為 `transitional internal field`。 **[Maintainer-Confirmed]**  
5. API response shape 已與 persistence schema 語義分離，避免 DTO 漂移成 artifact truth source。  

## 16. Current Risks

1. risk：task unit split mode 行為差異造成結果波動
- why：同文檔在不同 mode 下結構變化可能大
- guardrail：mode-specific regression tests

2. risk：transitional DTO fields 被誤當 public API 契約
- why：可能重引 flat mirrors
- guardrail：在 docs/schema 明確 internal-only

3. risk：LLM split 與 heuristic split 邊界不清
- why：debug 成本高
- guardrail：維持 selector 與 fallback reason 可觀測

4. risk：task unit id normalization 若失效會破壞 artifact 對齊
- why：cache validity/target resolution 錯位
- guardrail：document-scope unique id tests + parent_section_id checks

## 17. Open Questions for Maintainer

1. transitional DTO fields 何時可正式標記為 deprecated-for-removal？
2. summary/quiz prompt builder contract 是否需要獨立文檔（非實作細節）？
3. task-unit split mode 對外可觀測性是否需統一 reason code？
4. `DocumentTaskLayout` internal transitional fields 是否需要在類註解中加 deprecation horizon（版本或階段）？

## 18. Suggested Next Documentation Improvements

1. 增加 task-unit split mode behavior matrix。
2. 增加 task-layout DTO public/internal field boundary表。
3. 補 artifact validity reason taxonomy 文檔。

## 19. Future Direction Note: Rich Content Interaction Preparation

> 本節為 future-direction preparation，非當前 implementation。  
> 目的僅為收斂邊界，不代表已實作 runtime/API/schema 變更。 **[Doc-Confirmed]**

1. `task-layout` 維持 lightweight metadata/projection contract；large content 不回流至 task-layout DTO。 **[Code-Confirmed] + [Future Direction]**
2. rich content 應維持 on-demand read model，透過 task-unit content lookup path 演進，不改 task-layout read-path定位。 **[Code-Confirmed] + [Future Direction]**
3. future content blocks 可成為 sentence/paragraph/block-level interaction target，但不應形成新的 hierarchy level。 **[Maintainer-Confirmed] + [Future Direction]**
4. future content-block artifact targeting/availability projection 屬 interaction 層語義，不得污染 hierarchy persistence truth source。 **[Future Direction]**
5. interaction targeting 應維持 id-based deterministic path，不得回退為 title-only lookup。 **[Code-Confirmed] + [Future Direction]**
6. richer interaction state 不得 hidden write-back profile；diagnostics/projection 與 persisted profile 需保持邊界分離。 **[Code-Confirmed] + [Future Direction]**

### 19.1 Future Content-Block Interaction Backend Decomposition

> Maintainer decision: visible content blocks need to become interactive business
> targets. This requires backend support before the UI should claim true block-level
> interaction persistence. **[Maintainer-Provided] + [Future Direction]**

Recommended backend task order:

1. Public target schema: extend the shared reading-interaction target contract to
   represent `content_block` targets with `doc_name`, `target_type=content_block`,
   `task_unit_id`, `content_block_id`, and optional parent `chapter_id` /
   `section_id` consistency fields. The public request schema and route dispatch now
   preserve `content_block_id`; resolver identity validation and primary source-context
   slicing are now implemented, while service thresholds, artifact persistence,
   route-level content-block validation, and frontend handoff remain future work.
   **[Code-Confirmed] + [Future Direction]**
2. Content-block read-model activation: resolver validation reads the hierarchy-
   resolved task unit plus the same deterministic segmented render projection exposed
   by the task-unit content endpoint. Explicit custom `TaskUnit.content_blocks` remain
   accepted when their ids do not collide with segmented render ids. Resolver
   validation does not read the relational `content_blocks` table directly or use UI
   row position/title text as authority. **[Code-Confirmed]**
3. Source hash and stale block policy: define whether a client-observed
   `source_hash` / structure version mismatch produces `stale_target`, fail-fast
   validation, or regeneration with updated block evidence. Current policy is
   recoverable `stale_target`: missing client source evidence is treated as current,
   while supplied `source_hash` or `source_structure_version` must match the selected
   content block metadata. Missing block metadata for supplied evidence is also stale.
   This policy runs before LLM generation through the shared validity/preflight
   semantics and does not hard-fail schema validation or silently regenerate against
   stale block evidence. **[Code-Confirmed]**
4. Block artifact availability projection: expose lightweight availability for block
   menus without placing heavy block text or block-level artifact payloads in
   `/documents/task-layout`. The public content-block response schema now allows an
   optional `interaction_availability` object with `insight`, `quiz`, and
   `critical_thinking` slots. Each slot carries only interaction type, status,
   artifact/session id, timestamps, and reason. It forbids artifact payload, block
   text, and UI-only state such as drawer/menu state. Current implementation defines
   and maps the lightweight contract when provided by the read model; automatic
   artifact-store population remains a later projection task. **[Code-Confirmed] +
   [Future Direction]**
5. Content-block identity validation: validate that a requested block id belongs to the
   resolved task unit's current segmented content projection; missing, duplicate, or
   stale block ids must fail fast with explicit reason codes. Current resolver support
   resolves `content_block` targets only through hierarchy-first task-unit ownership:
   `task_unit_id` is required, optional `chapter_id` / `section_id` must match the
   hierarchy parent path, and `content_block_id` is looked up inside the segmented
   task-unit content read model for that resolved task unit. Missing blocks,
   duplicate block ids within the accepted read model, parent mismatches, and supplied
   stale `source_hash` / `source_structure_version` evidence fail fast with explicit
   reasons. **[Code-Confirmed]**
6. Resolver support: extend hierarchy-first reading target resolution from
   document/chapter/section/task-unit to content-block without adding content blocks as
   a persisted hierarchy level and without root `sections[]` / `structure_nodes`
   fallback. Current support returns `target_level=content_block`, uses the
   `content_block_id` as `target_id`, and preserves parent chapter/section/task-unit
   metadata plus the selected `TaskUnitContentBlock` object. Source context is now
   populated by the source-context slicing step below; resolver identity support alone
   should still not be described as completed block-level generation support because
   artifact persistence, artifact-aware context, route exposure validation, and final
   UI handoff remain separate tasks. **[Code-Confirmed] + [Future Direction]**
7. Source-context slicing: build `ResolvedReadingTarget.content` from the selected
   content block text while preserving parent task-unit/chapter/section metadata for
   breadcrumb, artifact scope, and stale-target checks. Current implementation assigns
   the selected `TaskUnitContentBlock.content` directly as the primary source context;
   it does not concatenate parent task-unit or sibling block content. Secondary context
   policy remains a later explicit task. **[Code-Confirmed] + [Future Direction]**
8. Service preflight and limits: define readable-content thresholds, token budgeting,
   quiz item limits, and critical-thinking suitability for block-level targets; tiny
   blocks should return reasoned `insufficient_content`, not empty successful
   artifacts. Current implementation routes `content_block` targets through the shared
   `ReadingInteractionValidityPolicy` used by analysis, quiz, and critical-thinking
   services: default readable thresholds remain `min_content_chars=20` and
   `min_alnum_chars=8`, so tiny or symbol-only blocks return reasoned
   `insufficient_content` without generator calls. Quiz now has an explicit
   `content_block` default max of `1`, lower than the task-unit default of `3`, and
   generated item lists are capped to that limit. Source budgeting remains bounded by
   the selected block as primary source context; no parent/sibling text is added by
   this task. **[Code-Confirmed] + [Future Direction]**
9. Artifact identity and persistence: include `content_block_id` in interaction artifact
   target metadata/current-artifact keys so block-level insight, quiz, and
   critical-thinking sessions do not collide with parent task-unit artifacts.
   Current implementation preserves `content_block_id` through service artifacts,
   common-artifact target metadata, document-backed current-artifact persistence, and
   critical-thinking sessions. Analysis and quiz current-artifact ids append an
   explicit `content_block::<content_block_id>` segment when present; generated
   critical-thinking session ids do the same before the UUID suffix. Store matching
   also compares `content_block_id`, so parent task-unit requests cannot read a
   content-block artifact/session even when they share the same parent unit.
   **[Code-Confirmed]**
10. Artifact-aware context: decide whether sibling/parent task-unit artifacts can be
   secondary context for block-level generation, while preserving the selected block as
   the primary source context. Current policy allows parent task-unit artifacts for the
   same `task_unit_id` as compact secondary context for `content_block` targets, but
   excludes sibling `content_block` artifacts until a dedicated sibling-context policy
   exists. The selected block remains `request.target.content`; artifact context is
   carried through `secondary_context` and bounded provenance metadata only. Persisted
   artifact reference metadata records referenced ids/types/target levels, coverage,
   and deduplication/abstraction flags, and does not store secondary context text or
   child artifact payloads. **[Code-Confirmed]**
11. Route exposure and response mapping: expose read/generate/refresh/submit/retry
   routes for `content_block` targets through the same generic reading-interaction
   route families after schema/resolver/service support exists. Current generic route
   families accept `content_block` targets through the shared target schema and
   preserve `content_block_id` through response envelopes and app dispatch for
   insight read/generate/refresh, quiz read/generate/refresh, and critical-thinking
   read/generate-question/submit-answer/retry-evaluation. Route mapping does not
   introduce route-local target variants, legacy section/chapter quiz dispatch, UI
   state, hierarchy search, prompt construction, context selection, or artifact
   persistence internals. **[Code-Confirmed]**
12. Frontend handoff: the backend target contract is now ready for UI per-block menus.
   The canonical UI mapping is: selected document `doc_name` + task-unit content
   response parent ids (`chapter_id`, `section_id`, `task_unit_id`) +
   `content_blocks[].block_id` submitted as reading-interaction
   `target.content_block_id`. `content_blocks[].metadata.source_hash`, when present,
   may be submitted as optional stale-target evidence; `metadata.content_block_id` is
   only a duplicate/advisory metadata value and must not replace the canonical
   `block_id`. `ContentBlockView` may therefore render a per-block `MoreVert` menu for
   insight, quiz, and critical-thinking actions without silently mapping block actions
   to the parent task unit. This is a backend-to-frontend contract checkpoint, not a UI
   implementation; UI menu open state, drawer state, draft answers, pagination state,
   and selected rows remain frontend-local and must not become backend artifact truth.
   **[Code-Confirmed]**

Guardrails:

- Content blocks remain interaction identities inside a task unit, not a new persisted
  hierarchy level. **[From Proposal] + [Future Direction]**
- `/documents/task-layout` must remain lightweight and must not embed block text or
  block-level artifact payloads. **[Code-Confirmed] + [Future Direction]**
- Content-block targeting must be id-based and deterministic; title/visible-index
  targeting is not acceptable as backend authority. **[Future Direction]**
- UI state such as open menus, selected block rows, or draft answers must not become
  backend artifact truth. **[Code-Confirmed]**

### 19.2 Frontend Content-Block Target Handoff Contract

This contract is the backend-owned handoff for UI per-block interaction menus.
It describes how a `ContentBlockView` can build a shared reading-interaction target
without changing `/documents/task-layout` or inventing frontend target identity.
**[Code-Confirmed]**

Source response:

- Use the on-demand task-unit content response, single or batch. **[Code-Confirmed]**
- Use `TaskUnitContentResponse.task_unit_id`, `chapter_id`, and `section_id` as the
  parent identity fields when available. **[Code-Confirmed]**
- Use each `content_blocks[].block_id` as the canonical `content_block_id` for
  reading-interaction requests. **[Code-Confirmed]**
- Use `content_blocks[].metadata.source_hash` only as optional source evidence for
  stale-target checks. **[Code-Confirmed]**
- Do not use block visible row position, rendered text, title text, pagination index,
  or `interaction_availability` slot state as backend target identity. **[From HLD] +
  [Code-Confirmed]**

Request shape for UI per-block interaction actions:

```json
{
  "target": {
    "doc_name": "<loaded document doc_name>",
    "target_type": "content_block",
    "chapter_id": "<TaskUnitContentResponse.chapter_id>",
    "section_id": "<TaskUnitContentResponse.section_id>",
    "task_unit_id": "<TaskUnitContentResponse.task_unit_id>",
    "content_block_id": "<content_blocks[].block_id>",
    "source_hash": "<content_blocks[].metadata.source_hash or null>",
    "source_structure_version": null
  }
}
```

Route family for the same target object:

- insight: `/documents/reading-interactions/insight/read`,
  `/generate`, `/refresh`
- quiz: `/documents/reading-interactions/quiz/read`,
  `/generate`, `/refresh`
- critical-thinking: `/documents/reading-interactions/critical-thinking/read`,
  `/generate-question`, `/submit-answer`, `/retry-evaluation`

Boundary rules:

1. This handoff does not add content blocks to `/documents/task-layout`.
   **[Code-Confirmed]**
2. This handoff does not make content blocks persisted hierarchy nodes.
   **[From Proposal] + [Code-Confirmed]**
3. This handoff does not require UI actions to use legacy `/section-quiz` or
   `/chapter-quiz` routes. **[Code-Confirmed]**
4. This handoff does not make frontend menu/drawer/collapse/draft-answer state backend
   truth. **[From HLD] + [Code-Confirmed]**
5. Backend responses echo resolved `target.content_block_id` in the shared envelope;
   clients should use that echo for display/cache reconciliation, not as a new
   hierarchy source. **[Code-Confirmed]**

## 20. Artifact Target Metadata Glossary and Target Constraints (Content Endpoint Context)

> 本節描述 `/documents/{doc_name}/task-units/{task_unit_id}/content` 回應中的
> content-block artifact target metadata 邊界；屬 pass-through/documentation contract，
> 不是 artifact persistence contract。 **[Code-Confirmed] + [Doc-Confirmed]**

### 20.1 Minimum Metadata Glossary

在 content endpoint context 中，artifact target metadata 的最小允許 key 僅限：

- `source_hash`
- `content_block_id`
- `quote_span_start`
- `quote_span_end`
- `schema_version`

治理規則：

1. 不應任意擴張 metadata vocabulary。 **[Maintainer-Confirmed]**
2. 以上欄位僅支援 pass-through / interaction targeting support。 **[Code-Confirmed]**
3. metadata 不是 artifact persistence truth，也不是 hierarchy truth。 **[Code-Confirmed] + [From HLD]**

### 20.2 Allowed Target Combinations (Task-Unit Content Endpoint)

在此 endpoint context 下的最小 target constraints：

1. `target_level = content_block` 必須包含：`task_unit_id`、`content_block_id`。 **[Code-Confirmed]**
2. `target_level = task_unit` 必須包含：`task_unit_id`。 **[Code-Confirmed]**
3. 其他 target levels 可保留 enum vocabulary，但本 endpoint 不宣稱完整 artifact write support。 **[Code-Confirmed] + [Doc-Confirmed]**

### 20.3 Future Validation Boundary

1. future artifact repository write/read flow 必須做 hierarchy-aware validation。 **[Future Direction]**
2. 本 endpoint 的 `artifact_target_refs` 仍僅 pass through metadata：不驗證 referenced artifact existence、不創建 artifact target、不做 artifact persistence。 **[Code-Confirmed]**
3. `interaction_availability` 是獨立的 lightweight read projection，可讀取既有 interaction artifact/session state，但不得回傳 artifact payload、block text、secondary context text、或 UI state。 **[Code-Confirmed]**

### 20.4 Content-Block Interaction Availability Contract

`/documents/{doc_name}/task-units/{task_unit_id}/content` 的
`content_blocks[]` 可攜帶 optional `interaction_availability`。此欄位是
block menu projection contract，不是 artifact payload transport。 **[Code-Confirmed]**

允許 slot：

- `insight`：對應 reading interaction `analysis`
- `quiz`：對應 reading interaction `quiz`
- `critical_thinking`：對應 reading interaction `critical_thinking_session`

每個 slot 僅允許：

- `interaction_type`
- `status`
- `artifact_id`
- `session_id`
- `reason`
- `generated_at`
- `updated_at`

治理規則：

1. `interaction_availability` 不得包含 artifact `payload`、block `content`、secondary context text、或 UI state（例如 drawer/menu open state）。 **[Code-Confirmed]**
2. `not_generated` slot 不得攜帶 `artifact_id` 或 `session_id`。 **[Code-Confirmed]**
3. 需要 reason 的狀態（如 `insufficient_content`、`stale_target`、`generation_failed`、`evaluation_failed`）必須攜帶 reason。 **[Code-Confirmed]**
4. 此 contract 不把 `/documents/task-layout` 變成 block payload 或 artifact payload transport；task-layout 仍保持 lightweight metadata/projection。 **[Code-Confirmed] + [From HLD]**
5. 目前 content endpoint 會對 segmented task-unit content read model 可解析的 block 自動投影既有 interaction state：analysis/quiz 使用 current-artifact read path，critical-thinking 使用 latest matching current-session lookup；缺失時投影 `not_generated`。此投影仍不保存 placeholder、不生成 artifact、不改 task-layout。 **[Code-Confirmed]**
6. `segmented=true` 產生的 deterministic render blocks 已與 content-block resolver read model 對齊；缺少必要 target identity 的 block 仍會被跳過而不是讓 content API 失敗。 **[Code-Confirmed]**
7. `/documents/task-layout` 對 content-block interaction 仍只暴露 task-unit metadata 與既有 summary/quiz availability，不回傳 `content_blocks`、`interaction_availability`、`artifact_target_refs`、quote span metadata、block text、或 interaction artifact payload。此邊界由 task-layout/content endpoint regression coverage 固定。 **[Code-Confirmed]**

## 21. Content Block Segmentation Design

> 本節是 section_tasks 層的 segmentation design。Current endpoint projection semantics are implemented for read-only rendering, and artifact-target alignment is bounded by the canonical resolver read model. **[Code-Confirmed]**

### 21.1 Segmented Content Endpoint Projection Semantics

1. `/documents/{doc_name}/task-units/{task_unit_id}/content?segmented=true` returns deterministic multi-block `content_blocks` for render/interaction preparation while preserving the existing response shape. **[Code-Confirmed]**
2. `content` remains compatibility/debug text and is suppressed by default unless `include_raw_content=true`; segmented `content_blocks[]` is the primary render payload for this mode. **[Code-Confirmed]**
3. Segmented block ids are deterministic within the task unit (`<task_unit_id>:content:<index>`) after heading-trim adjustment, and do not depend on random ids, LLM output, retrieval, or UI row position. **[Code-Confirmed]**
4. Segmented block metadata is limited to segmentation evidence: `source_hash`, `content_block_id`, `quote_span_start`, `quote_span_end`, `schema_version`, plus `display_normalization` when OCR whitespace normalization changes display text. **[Code-Confirmed]**
5. Segmented render blocks do not auto-invent `artifact_target_refs`; resolver-backed `interaction_availability` may be projected when reading current artifact/session state succeeds for the same segmented id. **[Code-Confirmed]**
6. Segmentation is a runtime projection concern, not hierarchy/persistence truth; it does not add content blocks to `/documents/task-layout`, mutate task-layout/profile, or create artifact persistence entries. **[Code-Confirmed] + [From HLD]**

### 21.2 Content-Block Target Validation Semantics

Current resolver validation is fail-fast and hierarchy-owned:

1. `target_level=content_block` requires both `task_unit_id` and `content_block_id`; non-content-block targets must not include `content_block_id`. **[Code-Confirmed]**
2. Optional `chapter_id` and `section_id` are consistency fields only; mismatches against the hierarchy-resolved parent chapter/section fail fast with explicit parent-mismatch errors. **[Code-Confirmed]**
3. Missing block ids fail with an explicit not-found error scoped to the resolved parent `task_unit_id`; resolver does not use title text, row position, visible index, or segmented render order as fallback identity. **[Code-Confirmed]**
4. Duplicate block ids inside the resolved task unit fail with `duplicate_content_block_id:<id>` instead of selecting the first match. **[Code-Confirmed]**
5. Supplied `source_hash` / `source_structure_version` must match selected block metadata when present; mismatches or unavailable metadata for supplied evidence fail as `stale_content_block_target`. **[Code-Confirmed]**
6. Segmented render block ids exposed by the task-unit content endpoint are valid artifact targets when they can be reconstructed from the hierarchy-resolved parent task unit using the same deterministic heading-trim, reindexing, and OCR-display normalization rules. **[Code-Confirmed]**

Validation boundary remains fail-fast + explicit reason classification; it must not silently fallback to title lookup, UI row order, legacy root `sections[]`, `structure_nodes`, or hidden mutation. **[Code-Confirmed] + [From HLD]**

### 21.3 Identity Relationship and Deterministic Constraints

1. hierarchy lookup 仍以 `task_unit_id` 為入口；`content_block_id` 是 task-unit 內 interaction identity。 **[Code-Confirmed] + [Future Direction]**
2. block-id generation 應保持 deterministic；不得依賴 random uuid/LLM/retrieval。 **[Maintainer-Confirmed] + [Future Direction]**
3. `content_block_id` 不得升格為 hierarchy identity；segmentation 不得改變 `task_unit_id` identity。 **[Maintainer-Confirmed] + [From HLD]**

### 21.4 Segmented Block Artifact-Target Alignment

Current alignment behavior is intentionally conservative:

1. Segmented `content_blocks[]` remain render projections, but their deterministic ids are accepted as interaction targets when reconstructed from the hierarchy-owned parent task unit. **[Code-Confirmed]**
2. Reading-interaction artifact targets resolve through hierarchy-first task-unit ownership and the segmented task-unit content read model. Explicit custom `TaskUnit.content_blocks` remain accepted when their ids do not collide with segmented render ids. **[Code-Confirmed]**
3. A segmented id such as `<task_unit_id>:content:1` resolves to the matching segmented block text when the parent task unit deterministically produces that block; resolver still does not fallback to visible row index, title text, arbitrary block text, or UI state. **[Code-Confirmed]**
4. Segmented endpoint responses may include resolver-backed `interaction_availability` slots for accepted block ids, but must not invent `artifact_target_refs`, placeholder artifacts, UI state, or artifact payloads. **[Code-Confirmed]**
5. Future promotion to a persisted rich-content model may replace the deterministic runtime projection, but it must remain hierarchy-aware and must not mutate task-layout/profile state from a read projection. **[Future Direction]**

### 21.5 Guardrails for This Preparation Slice

1. 不把 segmented `content_blocks` 加入 task-layout response。 **[Maintainer-Confirmed] + [Code-Confirmed]**
2. segmentation 不改變 task-unit identity，不修改 hierarchy truth。 **[Maintainer-Confirmed] + [From HLD]**
3. segmentation 不做 hidden mutation，不寫回 profile。 **[Maintainer-Confirmed] + [Code-Confirmed]**
4. 不接 retrieval/LLM/question/evaluated_answer integration。 **[Maintainer-Confirmed]**
5. 不做 artifact persistence/repository read-write integration。 **[Maintainer-Confirmed]**
6. content block identity 是 interaction identity，不是 persisted hierarchy identity。 **[Maintainer-Confirmed] + [Future Direction]**

## 22. Segmented Endpoint Behavior Stabilization Note

1. task-unit content endpoint 的 segmented 行為矩陣固定為：
   - omitted / `segmented=false`：compatibility-safe default projection（single/default block behavior）
   - `segmented=true`：explicit opt-in segmented projection。 **[Code-Confirmed]**
2. segmentation 在本模組定位為 runtime projection concern，不是 persistence truth，不改 hierarchy identity。 **[Code-Confirmed] + [From HLD]**
3. task-layout payload 維持 lightweight metadata/projection，不承載 segmented blocks。 **[Code-Confirmed]**
4. segmented projection 不觸發 artifact persistence、不觸發 retrieval/LLM/evaluated_answer integration。 **[Code-Confirmed]**
5. 本輪 stabilization 僅強化 deterministic/compatibility regression 與 multilingual fixture coverage，不擴張 endpoint 功能範圍。 **[Doc-Confirmed]**

## 23. Task-Layout Anchor Evidence Projection

> 本節支援 UI TOC editor 的 edit-existing 預填與 page-first anchor UX。
> 目前已落地 DTO 承載能力；coordinator/API route mapping 仍屬後續 checkpoint，不代表 public response 已開始輸出 anchor evidence。 **[Code-Confirmed] + [Future Direction]**

1. `DocumentTaskLayoutChapterDTO` and `DocumentTaskLayoutSectionDTO` can carry optional `AnchorEvidenceDTO` for UI TOC edit-existing prefill. **[Code-Confirmed]**
2. `AnchorEvidenceDTO` is lightweight runtime projection metadata. It is not hierarchy truth, not parser authority, and not a persistence schema. **[Code-Confirmed] + [From HLD]**
3. The projection should prefer validated page evidence for pageable documents and fall back to `char_range` when page evidence is unavailable or unreliable. **[Maintainer-Provided] + [Future Direction]**
4. Missing anchor evidence remains explicit by omission/null DTO ownership; unset evidence does not alter the current serialized payload shape before API mapping. **[Code-Confirmed]**
5. Task-layout must not invent anchors from titles, page labels, OCR guesses, or profile metadata. **[Future Direction]**
6. The DTO must stay lightweight: no raw text, full page text, OCR region geometry, or content blocks. Detailed evidence belongs to preparation/loader diagnostics, not task-layout. **[Code-Confirmed] + [Future Direction]**
7. DTO fields currently support `anchor_type`, `status`, `reason`, `char_start`, `char_end`, `page_start_index`, `page_end_index`, `page_start_label`, and `page_end_label`. **[Code-Confirmed]**
8. Public API mapping must preserve chapters-first response shape and keep anchor evidence attached to existing chapter/section nodes only. **[From HLD] + [Future Direction]**

## 24. Future Direction Note: Reading Interaction Services

> 本節記錄 target-agnostic interaction service planning and implementation status；未標示 **[Code-Confirmed]** 的項目仍屬 future direction。 **[Maintainer-Provided] + [Future Direction]**

1. `section_tasks/` owns service-level interaction contracts for reading target analysis, quiz generation, and critical-thinking sessions, while app/main own orchestration and route mapping. **[Code-Confirmed]**
2. Services consume a resolved reading target object, not raw title strings or unresolved ids. The resolved target carries canonical target id, hierarchy path evidence, and source text input. **[Code-Confirmed]**
3. `analysis` output supports summary, reasoning, and explanation sections as structured validated JSON, not free-form opaque text. Invalid JSON or missing required fields return `generation_failed` rather than a successful artifact. **[Code-Confirmed]**
4. `quiz` output accepts only `short_answer`, `multiple_choice`, and `true_false` item types. The backend provides valid types plus max count; the model may return fewer than max, and model-overrun outputs are capped to the target-level max before per-item validation. **[Code-Confirmed]**
4.1. `multiple_choice` service payload is canonicalized to `choices`, while service validation accepts `options` as an LLM-output alias and normalizes it to `choices` before route mapping exposes frontend-facing `options`. **[Code-Confirmed]**
5. Default quiz maximums are configurable at service construction with defaults: `task_unit=3`, `section=5`, `chapter=10`, `document/book=25`. **[Code-Confirmed]**
6. Critical-thinking service state is session-shaped: generated question returns `question_generated`, answer submission returns `answer_submitted`, evaluation success returns `completed`, and evaluation failure returns `evaluation_failed` while preserving the answer for retry. **[Code-Confirmed]**
6.1. Critical-thinking question/evaluation parsing accepts a strict JSON object directly, a fenced JSON object, or the first JSON object after model-added leading text; success still requires the validated `question` or evaluation fields. Free-form text without a valid JSON object remains `generation_failed` / `evaluation_failed`. **[Code-Confirmed]**
7. All interaction service contracts allow `insufficient_content` as a valid result/session status with required reason metadata. This is the correct fast path for OCR noise or symbol-only units. **[Code-Confirmed]**
8. Strict DTO validation is mandatory before marking an interaction artifact/session successful. Invalid or empty completed output is rejected at the contract boundary. **[Code-Confirmed]**
9. Services should record context metadata such as `context_mode`, token estimate/budget, evidence ids, prompt instruction version, and output schema version in artifact metadata. **[Maintainer-Provided] + [Future Direction]**
10. Services must not mutate task-layout, profile diagnostics, parser metadata, or hierarchy truth. Artifact persistence remains interaction output only. **[From HLD] + [Future Direction]**
11. Critical-thinking prompt behavior keeps fixed service-level question/evaluation instructions and leaves target context plus user answer as the dynamic parts. **[Code-Confirmed]**
12. Interaction services share `ReadingInteractionValidityPolicy` preflight checks so stale target context returns `stale_target` with a required reason before any generator call, and insufficient readable content returns `insufficient_content` with a required reason. **[Code-Confirmed]**
13. Interaction services may consume lower-level artifact summaries through an `artifact_context_provider`; this injects compact secondary context into `ReadingInteractionRequest.secondary_context` while preserving `target.content` as the primary source context. Artifact metadata records provenance only, not secondary context text or child payloads. **[Code-Confirmed]**
14. Reading interaction service artifacts can be mapped onto the shared `CommonArtifact` entity through `reading_interaction_common_artifact.py`; `analysis`, `quiz`, and `critical_thinking_session` remain artifact types on one entity shape rather than separate persistence categories. **[Code-Confirmed]**
15. Analysis and quiz current-artifact semantics are explicit: read paths return only current persisted artifacts or `not_generated`, while generation/refresh is the only path that invokes generators and writes completed or insufficient-content artifacts. Read and explicit generate recheck persisted insufficient-content artifacts through service preflight only; readable targets are surfaced as `not_generated` on read or regenerated on generate, while still-insufficient targets keep the fast-path insufficient status without LLM calls. **[Code-Confirmed]**
16. `AnalysisInteractionLLMGenerator` is the first concrete LLM-backed generator adapter for analysis/insight; it receives an already-resolved `ReadingInteractionRequest`, keeps target content as the primary source context, optionally includes secondary artifact context, and returns strict JSON text for downstream validation. **[Code-Confirmed]**
17. `DocumentReadingInteractionArtifactStore` is the document-backed adapter for current analysis artifacts, current quiz artifacts, and critical-thinking sessions. It persists `CommonArtifact` dictionaries under `DocumentTaskArtifacts.metadata["reading_interaction_artifacts"]`; analysis and quiz use current-artifact keys by type/document/target, while critical-thinking sessions use session ids and preserve target matching on read. Actual document artifact updates remain delegated to the structured document repository boundary. **[Code-Confirmed]**
18. `QuizInteractionLLMGenerator` and `CriticalThinkingLLMGenerator` are concrete LLM adapter slices used by container wiring. They only assemble prompt text from an already resolved target/session and delegate strict JSON validation to their owning services. **[Code-Confirmed]**
19. Analysis/insight generation needs a semantic-content guard beyond minimum character
    and alphanumeric counts. A maintainer test with `國富論 lite` showed that a target
    with little or no meaningful readable content can still receive a fluent-looking
    insight, which is worse than an `insufficient_content` state. The guard should
    detect boilerplate, OCR junk, fragmented headings, index-like text, repeated tokens,
    or otherwise non-propositional source content before the LLM call when possible,
    and reject vacuous completed outputs after generation when preflight cannot decide.
    The expected response is a reasoned `insufficient_content` or `validation_failed`
    state, not a successful pseudo-insight. This must not use profile metadata as parser
    authority, must not mutate task-layout/profile diagnostics, and must preserve the
    hierarchy-resolved target as the only primary source context. **[Maintainer-Provided]
    + [Future Direction]**

### 24.1 Lower-Level Artifact Reference Policy

1. When generating a higher-level artifact, services can receive lower-level artifact summaries from the context/repository boundary as secondary input. **[Code-Confirmed]**
2. Lower-level artifacts help the LLM avoid repeated questions, identify already-covered concepts, and raise abstraction from unit-level recall to section/chapter/book-level synthesis, transfer, and critique. **[Code-Confirmed]**
3. Quiz generation uses lower-level quiz artifacts as deduplication and coverage signals, not as items to concatenate into the higher-level quiz; the quiz service appends fixed deduplication guidance when signals exist and rejects exact repeats of lower-level focus phrases where validation can determine the duplicate. **[Code-Confirmed]**
4. Critical-thinking generation uses lower-level critical-thinking sessions as learning-continuity signals; the service appends fixed continuity guidance when abstraction signals exist so the model can ask one broader current-target question that builds on prior local training without repeating child session questions. **[Code-Confirmed]**
5. Analysis generation may use lower-level analysis artifacts as abstraction hints, but the higher-level analysis must still be grounded in the current target source context. **[Maintainer-Provided] + [Future Direction]**
6. Services preserve a hard distinction between source context and artifact context in prompt inputs, validation metadata, and artifact metadata. **[Code-Confirmed]**
7. Artifact reference metadata is persisted on generated interaction artifacts as top-level `metadata.artifact_reference`, mirroring bounded provenance fields from `context.artifact_context`: referenced artifact ids, artifact types, target levels, coverage count, and whether deduplication or abstraction-escalation hints were applied. It does not include secondary context text or child payloads. **[Code-Confirmed]**

### 24.1.1 Future Direction: Critical-Thinking Evaluation Information Gain

> 本節治理 critical-thinking answer evaluation 的 information-gain rubric
> 方向。Prompt-level rubric wording is implemented, while explicit payload fields,
> route contracts, and schema-version migration remain out of scope for the current
> version. **[Maintainer-Provided] + [Code-Confirmed] + [Future Direction]**

Problem statement:

1. Critical-thinking evaluation should not only ask whether a user answer is correct,
   textually grounded, or logically valid. It should also evaluate whether the answer
   creates useful information relative to the prompt, source text, prior session
   context, and obvious restatement. **[Maintainer-Provided] + [Future Direction]**
2. The information-theory analogy is advisory: low information gain means the response
   is predictable, copied, or mostly restates available text; high information gain
   means the response introduces a supported distinction, reframing, objection,
   uncertainty reduction, or surprising but defensible interpretive move.
   **[Maintainer-Provided] + [Future Direction]**
3. This must not reward novelty for novelty's sake. High-scoring answers still need
   critical-thinking discipline: evidence, clear inference, uncertainty awareness,
   avoidance of overclaiming, and separation between textual fact, interpretation, and
   speculation. **[Maintainer-Provided] + [Future Direction]**

Evaluation principle:

1. Copying source text, repeating the generated question, paraphrasing an existing
   insight, or relying only on an uncited borrowed view should be treated as low
   information gain even when the answer is fluent. **[Maintainer-Provided] +
   [Future Direction]**
2. An answer should score higher when it uses textual evidence to create a new valid
   distinction, such as identifying a hidden assumption, narrowing an ambiguity,
   comparing alternatives, qualifying a claim, or explaining why an apparently simple
   reading is incomplete. **[Maintainer-Provided] + [Future Direction]**
3. The evaluation should distinguish:
   - raw repetition of the text
   - supported paraphrase
   - grounded interpretation
   - critical transformation that changes what the reader can now notice
   **[Future Direction]**
4. Surprise should mean evidence-bound surprise: the answer should reduce uncertainty
   or reveal a non-obvious relation without inventing unsupported facts. **[Future
   Direction]**

Planned rubric dimensions:

1. `textual_grounding`: Does the answer use relevant source evidence rather than
   unsupported assertion? Current payload-shape decision: use as evaluator guidance
   inside existing feedback text, not as a standalone JSON field. **[Code-Confirmed]**
2. `information_gain`: Does the answer add a supported distinction, reframing, or
   uncertainty reduction beyond obvious restatement? Current payload-shape decision:
   use as evaluator guidance inside existing feedback text and optional `score`, not
   as a standalone JSON field. **[Code-Confirmed]**
3. `originality`: Does the answer avoid merely copying the source, prompt, generated
   insight, or a borrowed viewpoint? Current payload-shape decision: use as evaluator
   guidance inside existing feedback text, not as a standalone JSON field.
   **[Code-Confirmed]**
4. `critical_reasoning_quality`: Does the answer reason carefully, qualify claims, and
   avoid overreach? Current payload-shape decision: use as evaluator guidance inside
   existing feedback text and optional `score`, not as a standalone JSON field.
   **[Code-Confirmed]**
5. `productive_surprise`: Does the answer create a useful "surprise" that remains
   constrained by the text and critical-thinking norms? **[Future Direction]**

Payload-shape decision:

1. Current version keeps the evaluation payload shape unchanged: required non-empty
   string fields `feedback`, `strengths`, and `improvements`, plus optional numeric
   `score` from 0 to 5. **[Code-Confirmed]**
2. `information_gain`, `originality`, `critical_reasoning_quality`, and
   `productive_surprise` are rubric concepts for prompt guidance and feedback wording
   only in the first implementation. They are not public API fields, persisted metadata
   keys, route request fields, or frontend contract fields. **[Code-Confirmed]**
3. This decision preserves backward compatibility for existing critical-thinking
   sessions, strict JSON validation, route DTO mapping, frontend rendering, and artifact
   persistence. **[From HLD] + [Code-Confirmed]**
4. If future product work needs machine-readable rubric analytics, it should be handled
   as a separate schema-versioned change, for example by introducing nested rubric
   metadata under a new output schema version and covering API schema, route mapping,
   frontend display, persistence compatibility, and migration behavior explicitly.
   **[Future Direction]**

Future implementation decomposition:

1. Extend critical-thinking evaluation prompt wording so the evaluator explicitly
   scores low-information answers lower when they only restate the source or copy
   existing opinions. Implementation status: the fixed evaluation instruction now
   tells the evaluator to consider information gain, penalize repetition of the
   question/source/existing insight/uncited borrowed views, reward evidence-bound
   distinctions, reframings, uncertainty reduction, and productive surprise, and keep
   the current strict JSON fields unchanged. **[Code-Confirmed]**
2. Decide whether `information_gain`, `originality`, and `critical_reasoning_quality`
   should be explicit payload fields, nested rubric metadata, or only feedback text in
   the first implementation. Decision: first implementation is feedback-only plus the
   existing optional `score`; explicit fields and nested rubric metadata are deferred
   to a future schema-versioned task if needed. **[Code-Confirmed]**
3. Add regression coverage with paired low/high information-gain answers:
   one answer that mostly copies source/prompt language, and one answer that introduces
   a supported distinction while acknowledging uncertainty. Implementation status:
   generator prompt regression now includes a low-information answer that repeats the
   primary source and a high-information answer that adds an evidence-bound distinction
   while preserving uncertainty; the regression verifies both prompts carry the rubric
   and keep the existing JSON contract, but does not claim deterministic LLM scoring.
   **[Code-Confirmed]**
4. Preserve existing strict JSON validation, session lifecycle, document-language
   guidance, no-external-research default, hierarchy-aware target identity, and
   artifact persistence boundaries. **[Code-Confirmed] + [Future Direction]**

### 24.2 Reading Interaction Follow-Up Issues

> 本節記錄 maintainer-observed interaction issues and their implementation status；
> 它是 documentation governance / planning memory。 **[Maintainer-Provided]**

1. Book/document-level insight, quiz, and critical-thinking generation must validate that
   document/book targets receive real readable source context before applying the shared
   `insufficient_content` fast path. A maintainer test with `Madame Bovary` observed
   book-level insight reporting insufficient text even though the document plainly has
   content, and quiz / critical-thinking are expected to share the same target-context
   risk. The fix should inspect document/book target content aggregation and preflight
   readable-content thresholds without adding heavy content to `/documents/task-layout`
   or using root `sections[]` / `structure_nodes` as a fallback source. Resolution:
   `ReadingTargetResolver` now assembles document/book, chapter, and section interaction
   source context from hierarchy task-unit content, falling back only to the hierarchy
   section's own content when no task-unit content exists; resolver regression tests cover
   empty `raw_text` with readable hierarchy content and reject legacy root-section
   backfill. **[Maintainer-Provided] + [Code-Confirmed]**
2. Critical-thinking question and evaluation prompt output language should follow the
   document profile / detected document language. A maintainer test with
   `许三观卖血记` produced English critical-thinking text, which indicates the current
   prompt/generator path does not sufficiently constrain output language from profile
   language context. The fix should use profile language as advisory prompt context,
   not parser authority, and should keep strict JSON validation and hierarchy target
   resolution unchanged. Resolution: app-layer reading interaction requests now carry
   profile-first `document_language` metadata, falling back to the structured document
   language; `CriticalThinkingLLMGenerator` uses that metadata to add an output-language
   instruction for both question and evaluation prompts, while preserving strict JSON
   validation and hierarchy target resolution. **[Maintainer-Provided] + [Code-Confirmed]**

### 24.3 Future Direction: Ancestor Summary Context for Block-Level Interactions

> 本節治理 block-level reading interaction 的 prompt context strategy。它是
> future-direction planning，不代表目前 runtime 已實作 ancestor summary 生成、
> 新 artifact schema、或 external research mode。 **[Maintainer-Provided] + [Future Direction]**

Problem statement:

1. `content_block` target 的 primary source context 應保持為被選中的 block text；
   不能為了讓 LLM 理解上下文而把 parent task-unit / section / chapter 的 raw text
   全部拼進 prompt。這會放大 token 成本、模糊 evidence priority，並提高幻覺風險。
   **[Maintainer-Provided] + [Future Direction]**
2. Block-level insight / quiz / critical-thinking 需要可理解的上級語境；
   但語境應以 bounded summary 形式注入，而不是 raw ancestor content concatenation。
   **[Maintainer-Provided] + [Future Direction]**
3. Document/book level orientation already belongs to document profile / profile summary
   semantics and should remain advisory prompt context, not parser authority or hierarchy
   truth. **[From HLD] + [Future Direction]**

Proposed context hierarchy:

1. Primary evidence: selected target source text. For `content_block`, this is only the
   resolved block content in `request.target.content`. **[Code-Confirmed] + [Future Direction]**
2. Ancestor summaries: compact generated summaries for parent `task_unit`, `section`,
   and `chapter`, when the target level is below those ancestors. These summaries orient
   the model but do not override the selected source text. **[Future Direction]**
3. Document profile summary: document/book-level profile summary and language metadata
   provide high-level orientation and output-language guidance. They remain advisory and
   must not become parser or hierarchy authority. **[From HLD] + [Future Direction]**
4. Lower-level artifact context: existing artifact-aware context may still provide
   deduplication / abstraction hints, but it remains secondary context with bounded
   provenance metadata and no raw child payload persistence. **[Code-Confirmed]**

On-demand preparation flow:

1. Read paths must not generate ancestor summaries, LLM outputs, placeholder artifacts,
   or profile/task-layout mutations. `read` continues to return current persisted state
   or `not_generated`. **[Code-Confirmed] + [Future Direction]**
2. Explicit `generate` / `refresh` may run an `ensure_ancestor_summaries` preparation
   phase before generating the requested insight, quiz, or critical-thinking question.
   This is an explicit user-triggered derived-resource preparation step, not a hidden
   read-side mutation. **[Future Direction]**
3. For a `content_block` target, the desired ancestor summary chain is:
   parent `task_unit` -> parent `section` -> parent `chapter` -> document profile
   summary. For `task_unit`, use `section` -> `chapter` -> document profile. For
   `section`, use `chapter` -> document profile. For `chapter`, use document profile
   only. For `document` / `book`, no ancestor generation is needed. **[Future Direction]**
4. Ancestor summaries should be generated only when missing, stale, or explicitly
   refreshed. Staleness should be keyed by target identity, source hash or structure
   version when available, summary schema version, and prompt instruction version.
   **[Future Direction]**
5. Persisted ancestor summaries should be stored as hierarchy-aware derived artifacts
   or summary artifacts keyed by document/target level/target id, without mutating
   `/documents/task-layout`, profile diagnostics, parser metadata, hierarchy truth, or
   content-block segmentation state. **[From HLD] + [Future Direction]**
6. If ancestor summary preparation fails, generation must not silently pretend full
   context exists. The result should either fail with a clear
   `context_preparation_failed` reason or proceed in an explicit degraded mode with
   `ancestor_context_status=partial` metadata. The first implementation should choose
   one policy and cover it with regressions. **[Future Direction]**

Prompt assembly rules:

1. Prompt inputs should be divided into labelled sections such as `Primary Source`,
   `Ancestor Summaries`, `Document Profile`, and `Secondary Artifact Signals`.
   **[Future Direction]**
2. The prompt must state evidence priority: answer from the primary source first;
   use ancestor summaries only to interpret local context; if the selected source does
   not support a claim, mark it as ambiguous or insufficient instead of inventing.
   **[Future Direction]**
3. Ancestor summaries should include target level, target id, title when available,
   summary schema version, source hash / structure version when available, and
   generated timestamp. The prompt should not include ancestor raw text by default.
   **[Future Direction]**
4. Output language should follow document profile / detected document language as
   advisory prompt context across analysis, quiz, and critical-thinking interactions,
   while strict JSON validation remains service-owned. **[Maintainer-Provided] +
   [Future Direction]**

External research policy:

1. Reading interactions default to text-only plus internal derived context. The backend
   should not encourage or invoke web/external search by default for insight, quiz, or
   critical-thinking generation. **[Maintainer-Provided] + [Future Direction]**
2. An explicit future `external_research` mode may be designed later, but it must be
   opt-in, provenance-labelled, and clearly separated from source-text evidence. External
   material can provide background only; it must not override the document text or become
   quiz answer authority unless the user explicitly selected a background-research task.
   **[Maintainer-Provided] + [Future Direction]**
3. Any external-research output must record source provenance separately from hierarchy
   and artifact identity metadata. It must not mutate document profile, task-layout,
   parser metadata, hierarchy truth, or content-block ids. **[From HLD] + [Future Direction]**

Implemented contract checkpoint:

1. `AncestorSummaryTarget` defines the hierarchy-aware identity for task-unit, section,
   and chapter summaries, including document id, target id, parent ids, and title.
   It intentionally excludes `content_block` and document/book targets because block
   interactions consume ancestor summaries, while document/book orientation remains
   document-profile context. **[Code-Confirmed]**
2. `AncestorSummaryArtifact` defines the compact summary artifact shape used for future
   prompt context: summary text, schema version, prompt instruction version, source
   structure version, source hash, document language, generated timestamp, artifact id,
   and metadata. It does not include raw ancestor source text. **[Code-Confirmed]**
3. `AncestorSummaryPromptContext` defines ordered prompt references plus bounded
   provenance metadata (`context_status`, summary ids/levels/ids, missing/stale levels,
   and pruned reason). Persisted metadata intentionally excludes raw summary text and
   raw ancestor content. **[Code-Confirmed]**
4. `AncestorSummaryRequirement`, `AncestorSummaryReadinessResult`, and
   `evaluate_ancestor_summary_readiness(...)` define a pure readiness/staleness policy
   over target identity, schema version, prompt instruction version, source structure
   version, source hash, document language, and explicit refresh intent. Results are
   limited to `reusable`, `missing`, `stale`, and `refresh_required`; the policy does
   not call LLMs, read UI state, use title lookup, consult root `sections[]` /
   `structure_nodes`, or mutate task-layout/profile/hierarchy state. **[Code-Confirmed]**
5. `AncestorSummaryPreparationService` prepares ancestor summaries only for explicit
   `generate` or `refresh` operations. It rejects read-like operations before querying
   the reader or generator, reuses summaries classified as `reusable`, generates
   summaries for `missing` / `stale` / `refresh_required` requirements, and validates
   generated summaries against the same requirement before admitting them into
   `AncestorSummaryPromptContext`. **[Code-Confirmed]**
6. `AncestorSummaryPreparationResult` records prompt context, readiness decisions, and
   newly generated summaries as in-memory preparation output. **[Code-Confirmed]**
7. `DocumentAncestorSummaryArtifactStore` persists and reads ancestor summaries through
   document-level `DocumentTaskArtifacts.metadata["ancestor_summary_artifacts"]`, keyed
   by document id, target level, and target id. This store preserves task-layout,
   profile diagnostics, parser metadata, hierarchy truth, and content-block segmentation
   boundaries; it does not create `reading_interaction_artifacts` entries. **[Code-Confirmed]**
8. `ReadingInteractionRequest` can carry optional `AncestorSummaryPromptContext`, and
   `build_evidence_priority_context_text(...)` assembles labelled prompt sections:
   `Evidence priority rules`, `Target Identity`, `Primary Source`, `Ancestor Summaries`,
   `Document Profile`, and secondary artifact context. The helper keeps the selected
   target text as primary evidence, treats ancestor summaries/profile as orientation,
   and uses `AncestorSummaryArtifact.to_prompt_reference()` so raw ancestor source
   metadata is not concatenated into prompts. **[Code-Confirmed]**
9. Analysis, quiz, and critical-thinking question generators now use the shared
   evidence-priority context assembly while preserving their strict JSON output
   instructions and existing secondary artifact labels. Critical-thinking answer
   evaluation remains session-based and does not consume source prompt context.
   **[Code-Confirmed]**
10. `build_ancestor_context_metadata(...)` records bounded ancestor-context provenance
   on generated analysis, quiz, and critical-thinking artifacts. The metadata includes
   context mode/status, summary artifact ids, summary target levels/ids, missing/stale
   target levels, pruned reason, and primary source target identity including
   content-block id when present. It intentionally excludes ancestor summary text, raw
   ancestor source text, and child payloads. Critical-thinking answer submission and
   evaluation preserve the same metadata through the session lifecycle.
   **[Code-Confirmed]**
11. Ancestor summary preparation failure semantics are explicit. `prepare(...)`
   defaults to `failure_policy="fail_fast"` and raises
   `AncestorSummaryPreparationError` when summary generation fails or produces a stale
   summary, so downstream generation cannot silently claim full context. Callers may
   opt into `failure_policy="degraded"`; in that mode failed requirements are omitted
   from prompt summaries, `AncestorSummaryPromptContext.context_status` becomes
   `partial`, missing/stale target levels and `pruned_reason` are recorded, and bounded
   `AncestorSummaryPreparationFailure` metadata records target level/id, readiness
   status, reason, and error type without raw ancestor text or summary text.
   **[Code-Confirmed]**
12. Reading interaction prompts enforce a no-external-research default. The shared
   evidence-priority prompt context tells analysis, quiz, and critical-thinking
   question prompts not to use web search, external research, outside facts, or
   unstated background knowledge unless an explicit `external_research` mode is
   provided. Critical-thinking answer evaluation uses the same default instruction.
   This is a prompt/contract guardrail only; no external research mode is implemented.
   **[Code-Confirmed]**
13. Explicit external research mode now has a bounded contract shape. A
   `ReadingInteractionRequest` may carry `ExternalResearchContext` only when the caller
   explicitly provides labelled `ExternalResearchSourceReference` entries. Prompt
   assembly then renders an `External Research Mode` section with source labels and
   provenance, keeps external material subordinate to `Primary Source`, and requires
   external claims to remain source-labelled. Generated analysis, quiz, and
   critical-thinking artifacts persist top-level `external_research` provenance
   containing mode, usage scope, source count, labels, source types, titles, URLs, and
   retrieved timestamps. The contract does not store external body text or payloads and
   does not mutate profile, task-layout, parser metadata, hierarchy truth, or
   content-block ids. **[Code-Confirmed]**
14. Regression coverage now locks the ancestor-context and read/no-generate boundaries:
   read paths with ancestor context still return current state / `not_generated`
   without interaction generation or placeholder persistence; read-like ancestor
   preparation is rejected before reader/generator side effects; explicit generate
   preparation records bounded metadata without raw ancestor text; prompt assembly
   excludes raw ancestor metadata, keeps external research default-off, and renders
   labelled external research only for explicit mode. **[Code-Confirmed]**
15. This checkpoint does not implement route wiring that prepares ancestor summaries
   before generation, external search providers, source fetching, or external research
   generation orchestration. Those remain separate checklist tasks.
   **[Code-Confirmed] + [Future Direction]**

### 24.4 Future Direction: Insight Prompt Depth and Summary Separation

> 本節治理 `analysis` / `insight` 的語義邊界與 prompt-depth 方向。它是
> documentation-governance planning，不代表目前 prompt、schema、route、或
> persistence behavior 已修改。 **[Maintainer-Provided] + [Future Direction]**

Problem statement:

1. Current insight behavior is stable but can be too shallow when it behaves like a
   compact summary. The desired output should help the reader notice how concrete
   textual details change interpretation, not merely restate what happened.
   **[Maintainer-Provided] + [Future Direction]**
2. If an insight payload contains a summary-like portion, that portion should be
   removed or demoted because summary is a separate future interactive action. Insight
   should not answer "what happened here?" as its main job. **[Maintainer-Provided] +
   [Future Direction]**
3. A future `summary` interactive action may be added later with its own action type,
   routes, artifact semantics, and prompt contract. That future action must not be
   simulated by weakening the current insight contract into a mixed
   summary-plus-analysis response. **[Maintainer-Provided] + [Future Direction]**

Desired abstraction level:

1. Insight starts from one to three concrete textual details such as action,
   image, object, bodily condition, metaphor, diction, narrative angle, or ambiguity.
   The prompt requires details from the `Primary Source`, while broader semantic
   validation for ungrounded claims remains future work. **[Maintainer-Provided] +
   [Code-Confirmed] + [Future Direction]**
2. Insight should produce an interpretive thesis: what the detail reveals, complicates,
   or reframes about a character, theme, relation, scene, or narrative method.
   **[Maintainer-Provided] + [Future Direction]**
3. Insight includes a complexity move. The model should identify how the selected
   source resists a simple reading, for example "not laziness, but a gap between
   effort and understanding" or "comic surface and sympathetic hardship coexist."
   **[Maintainer-Provided] + [Code-Confirmed]**
4. Insight identifies a tension, contrast, or pressure inside the passage when
   supported by the text, such as effort versus incapacity, bodily hardship versus
   intellectual failure, sympathy versus irony, or local detail versus larger theme.
   **[Maintainer-Provided] + [Code-Confirmed]**
5. Insight should end with a durable reading payoff: a claim that helps the reader read
   subsequent passages more attentively, without turning into a plot recap.
   **[Maintainer-Provided] + [Future Direction]**

Prompt contract direction:

1. The existing strict JSON schema remains compatible. The field currently named
   `summary` is treated as a short insight thesis, not a plot summary or
   summary-action payload, unless a future schema migration renames it.
   **[Code-Confirmed]**
2. `reasoning` should carry the concrete textual evidence and inference chain.
   `explanation` should explain why the insight matters and what simple reading it
   complicates. Prompt wording for the complexity/tension requirement is implemented;
   stricter semantic validation remains future work. **[Code-Confirmed] +
   [Future Direction]**
3. The prompt explicitly forbids primary output patterns such as plot recap,
   chapter overview, "this section mainly says", or generic moral/thematic summary.
   Necessary orientation may be limited to a phrase, but it must not dominate the
   artifact. **[Code-Confirmed]**
4. The prompt should preserve the shared evidence-priority context: selected target
   source text remains primary; ancestor summaries, document profile, lower-level
   artifacts, and explicit external research context remain subordinate advisory
   material. **[Code-Confirmed] + [Future Direction]**
5. The prompt should continue to follow document language guidance and the
   no-external-research default unless explicit `external_research` context is
   provided. **[Code-Confirmed]**

Future summary-action separation:

1. `summary` as a future interactive action should have its own purpose: concise
   content compression and recall support at document/chapter/section/task-unit/block
   levels. **[Future Direction]**
2. Summary artifacts should not be conflated with insight artifacts, even if both may
   reuse reading-target resolution, ancestor summary context, artifact persistence
   infrastructure, or language guidance. **[Future Direction]**
3. If the current `analysis` schema keeps a field named `summary` for compatibility,
   docs and prompt tests should make clear that this field contains an insight thesis,
   not a summary-action payload. **[Future Direction]**
4. Future standalone summary action should be introduced as a separate interaction type,
   such as `summary`, rather than by changing `analysis` semantics. It should have
   dedicated route/API schema decisions, artifact identity, output schema version,
   prompt instruction version, and frontend presentation rules. **[Future Direction]**
5. Summary action output should prioritize compression and recall: what happened, what
   claims or concepts were introduced, and what local reading state should be retained.
   It should not be asked to produce interpretive complexity, tension, or close-reading
   payoff unless a later summary variant explicitly requests that behavior.
   **[Future Direction]**
6. Summary action may reuse hierarchy-aware reading-target resolution, content-block
   targeting, readable-content preflight, ancestor summary preparation, document
   language guidance, no-external-research default, and common artifact persistence
   infrastructure. Reuse must remain service-level reuse; summary artifacts must not
   overwrite analysis artifacts or make task-layout/profile state backend truth.
   **[From HLD] + [Future Direction]**
7. Frontend integration for summary should be a separate menu/action decision. Existing
   `insight` UI should continue to display interpretive insight content and should not
   become the fallback renderer for summary-action payloads. **[Future Direction]**
8. A future implementation should first add an unchecked backend checklist for summary
   contracts, then route exposure, then frontend handoff. It should not rename the
   existing `analysis.payload.summary` field without an explicit compatibility plan.
   **[Future Direction]**

Validation direction:

1. Prompt regression coverage asserts that analysis generation instructions ask
   for concrete textual details, interpretive thesis, complexity move, tension/contrast,
   no plot-summary primary output, document-language guidance, and evidence-priority
   context ordering. **[Code-Confirmed]**
2. Service-level validation may later add light semantic guards for empty or purely
   recap-like analysis payloads, but the first slice should prefer prompt contract and
   regression coverage to avoid overfitting validation to wording. **[Future Direction]**
3. Any future schema rename from `summary` to `thesis` must be handled as an explicit
   API/backward-compatibility task across backend and frontend contracts. It should not
   be smuggled into a prompt-only change. **[Future Direction]**

### 24.5 Summary and Note Interactive Actions

#### Resumed Delivery Requirements (2026-10-21)

Contract implementation: `summary_note_contracts.py` defines strict Summary
`headline` (1..300), `body` (1..12000), `key_points` (0..12, each 1..1000),
with independent v1 schema/prompt identity. All counts are Unicode code points.
Notes use one plain-text note per resolved target, max 10000 code points;
blank-only and isolated surrogate text are rejected, other whitespace preserved.
No title/tags in v1. Server-generated note_id, positive revision and timestamps
are returned. Save requires expected_revision (0=create), mutation_id for retry
deduplication and a server-issued target_token from read; delete requires the
current revision/token. Contracts, services, persistence, coordinator and routes
are implemented. File/PostgreSQL and real-book tests provide evidence below.

`SummaryInteractionService` reads existing structure directly; it never calls
prepare on read. Generation supplies full target text plus existing profile and
current ancestor summaries, using the public_summary records as its sole summary
cache. Ancestor references adapt those records into the existing readiness and
prompt-context contracts; historical internal caches are not copied or rewritten.
Generate reuses current output; refresh replaces only the requested target and
prepares parents only when missing/stale. Each dispatch checks configured model
capacity against actual full prompt, output reserve and 1024-token overhead.
No primary truncation or research fallback. Readability uses quantity checks plus
model abstention: whole-book vocabulary diversity is not a short-passage test.
Strict output and document-language validation precede persistence; failures never
overwrite an existing successful summary. Metadata retains source hash/version,
prompt/schema, timestamps and bounded ancestor provenance, not parent source text.

Notes are stored under document artifact metadata.note, public summaries under
metadata.public_summary. No new hierarchy or task-layout payload. Mutations use
repository-owned atomic callbacks: file lock/atomic replacement or PostgreSQL
document row lock/JSONB update. Notes validate source token/revision under lock;
server UUID identity and latest mutation replay make network retry idempotent.
Delete advances a tombstone epoch to reject delayed creates. Initial file creation
and structural reparse advance a separate source epoch. Ordinary metadata writers
preserve these two owned fields, preventing delayed legacy saves from erasing them.
Confirmed structural replacement clears both; no-op/title-only/failure preserves
them. Source/version/epoch guards reject delayed writes. Explicit rename rekeys
new action identities and provenance without changing note text or summary prose.

Evidence: `test_summary_note_contracts.py`, `test_note_interaction_service.py`,
`test_summary_interaction_service.py`, disposable PostgreSQL
`test_summary_note_postgres.py` and four conditional-commit cases in
`test_summary_note_lifecycle.py`. Live `accept_summary_note_books.py` uses the
verified private backups, authorized full deletion/reupload/Common Fast parse,
and actual provider/API checks for both books at all five levels. The 2026-10-21
06:29:38 and 06:44:38 UTC reports pass ten target workflows, persisted reads and
repeat-generation reuse. Raw files, DB rows and existing derived files are backed
up under data/test-books/summary-note-20261021 with immutable manifest hashes.

#### Governance And Final Validation Record (2026-10-21)

Inspected unchanged global memory: AGENTS.md, proposal.md, high-level-design.md,
docs/modules/index.md; inspected progress.md and design/checklist pairs for
section_tasks, api_schemas, app, main, db, document_structure, config, scripts and
Deep_Reader_UI. No proposal/HLD/index or unrelated checklist was modified.

Modified controlled markdown and reasons:
- section_tasks design/checklist: finalized contracts, implementation/lifecycle
  evidence and closure of original tasks plus SN-14..16.
- api_schemas design/checklist: actual strict wire contracts and SN-08/09 status.
- app design/checklist: no-prepare Summary reads, Note dispatch and SN-10/11 status.
- main design/checklist: exposed routes/status mapping and SN-12/13 status.
- db design: atomic callbacks, preserved annotation fields and rename transaction.
- document_structure design: file locks, epochs and repository mutation boundary.
- config design: configured Summary service composition, no provider for Note.
- scripts design: private backup/delete/reparse/live acceptance and fixture cleanup.
- Deep_Reader_UI design/checklist: implemented menus/drawers/guards and SN-17..20 evidence.
- progress.md: backend owner counts synchronized (section_tasks 92, app 43,
  api_schemas 34, main 40); frontend progress remains owned by its checklist.

Validation: final 33 contract/service/route/PostgreSQL tests and four enhancement
lifecycle tests pass, plus seven existing interaction/schema/store/rename regression
scripts. Real provider acceptance passes both books x five targets. Actual API
browser acceptance passes both books x five targets x 1280px/390px, with fault
injection supplementing real reads/writes. Screenshots reviewed, TypeScript/Vite
build passes (existing bundle warning), diff check passes. New/changed checklist
items carry Timestamp: 2026-10-21. Immutable nine-file backup manifest and current
raw SHA-256 checks match; test notes removed, useful real summaries retained.
This closes the 20-task track only, not unrelated future checklists. No Git commit
or unrelated worktree cleanup was performed.

#### Original Planning Rationale

The following planning rationale is retained as history, superseded by the
implemented contract above. The delivery index in this module checklist resumes
13 existing unchecked tasks and adds missing backend/UI tasks; it does not claim
summary/note public APIs are implemented. SN-01/06 must settle strict field names,
statuses and note multiplicity before SN-08/09 freeze wire schemas. Optional
title/tags are proposals, not approved implementation requirements.

For summary, include all resolved primary-target text and labelled ancestor
summaries; never substitute an ancestor summary for the target text or silently
truncate primary evidence. Check the actual assembled input/output budget before
dispatch and return a clear refusal when it cannot fit. Book-level uses its full
primary text plus the already-loaded document profile; do not regenerate the
profile. Reads never prepare ancestors. Explicit generate prepares only missing/
stale required ancestors, reuses current artifacts and does not touch descendants.
Refresh must specify target-versus-ancestor invalidation rather than unconditionally
refreshing the whole tree. Define a single adapter/source-of-truth policy between
internal ancestor cache and public summary artifacts before persistence; no cyclic
self-summary preparation. A failed replacement must not destroy a usable current
summary. Summary remains distinct from insight's historical payload.summary field.

For notes, SN-06 must choose one-note versus multiple-note semantics, exact maximum
text length, Unicode counting units, whitespace/empty behavior, and optional field
limits. Do not ship an unspecified maximum or let frontend/backend count differently.
SN-09 must encode stable validation/conflict responses. Service and schema enforce
limits before writes; UI counters are only assistance. Define revision/conflict
handling so a stale editor cannot silently overwrite a newer note; retrying a
create/save must not silently duplicate notes. Notes are plain user-authored text
unless a later explicit formatting decision changes that contract; safely render
untrusted content. No LLM, prompt, ancestor preparation or hidden autosave is allowed.

All five target levels use server-validated IDs and source/version information,
not titles or UI positions. Block targets are not a new hierarchy. Document deletion
and confirmed structural replacement must include these new resources in the
existing full-document cleanup and confirmation inventory, including user notes.
No-op/title-only changes and failed enhancement retain them. In-flight saves must
not resurrect cleared interactions. This integrates the confirmed lifecycle policy,
not an implicit permission to erase notes during an ordinary read/generate call.

SN-14 owns note service behavior, SN-15 lifecycle integration, SN-16 real-backed-book
backend acceptance; UI SN-17..20 own adapters, summary view, note editor and end-to-end
acceptance. Existing app tasks include configured dependency wiring, not merely
method stubs. No new endpoints or SQL tables are prescribed as already implemented.

> 本節規劃兩個新增 reading interactive actions：`summary` 與 `note`。它是
> documentation-governance planning，不代表目前 API schema、route、service、
> persistence、或 UI 行為已實作。 **[Maintainer-Provided] + [Future Direction]**

Action split:

1. `summary` is an LLM-generated reading interaction for concise compression and recall.
   It answers "what should the reader retain from this target?" rather than producing
   interpretive close reading. It must remain separate from `analysis` / `insight`,
   even if both reuse target resolution, language guidance, prompt-context assembly, and
   common artifact infrastructure. **[Maintainer-Provided] + [Future Direction]**
2. `note` is a user-authored interaction artifact. It does not call the LLM, does not
   generate prompts, and should behave like a bounded annotation attached to the same
   hierarchy-aware targets as other reading interactions. **[Maintainer-Provided] +
   [Future Direction]**
3. Both actions should support document/chapter/section/task-unit/content-block targets
   through the existing id-based reading-target contract. Neither action may introduce
   content blocks as a persisted hierarchy level or use UI row position/title text as
   backend identity. **[Code-Confirmed] + [Future Direction]**

Summary action prompt/context design:

1. The selected target's full resolved text remains the primary source for the summary,
   subject to existing readable-content preflight and token-budget limits. For
   `content_block`, the full target text means the selected block text, not sibling or
   parent raw text. **[Maintainer-Provided] + [Future Direction]**
2. Lower-level summary generation should include higher-level summary context rather
   than concatenating raw ancestor text. Desired context chain:
   `content_block -> task_unit summary -> section summary -> chapter summary ->
   document profile`, `task_unit -> section summary -> chapter summary -> document
   profile`, `section -> chapter summary -> document profile`, `chapter -> document
   profile`, and `document/book -> document profile only`. **[Maintainer-Provided] +
   [Future Direction]**
3. Existing ancestor-summary infrastructure should be reused for readiness, staleness,
   prompt references, and bounded metadata, but public `summary` artifacts must have
   their own interaction type, schema version, prompt instruction version, and route
   semantics. Internal ancestor-summary cache and public summary action should share a
   compatible payload contract where possible, while avoiding two divergent summaries
   for the same target. **[Code-Confirmed] + [Future Direction]**
4. On explicit `summary generate` / `summary refresh`, the service may ensure required
   higher-level summaries first. It should reuse current higher-level summary artifacts
   when fresh; generate missing or stale higher-level summaries only as an explicit
   derived-resource preparation step; and record bounded context provenance without
   storing raw ancestor text. **[Future Direction]**
5. Summary prompt output should be strict JSON and should contain compact fields such
   as a short headline, concise summary body, key retained points, and optional
   continuity hints. It should not request insight-style complexity, tension, or
   evaluative payoff unless a later variant explicitly adds those fields.
   **[Future Direction]**
6. Summary must preserve the no-external-research default and document-language
   guidance already used by reading interactions. External research, if ever allowed,
   must remain opt-in and provenance-labelled. **[Code-Confirmed] + [Future Direction]**

Summary action persistence and read/write boundaries:

1. Summary read must never generate. It returns the current persisted summary artifact
   or `not_generated`. **[Code-Confirmed] + [Future Direction]**
2. Summary generate should reuse a completed current artifact unless refresh is
   requested, mirroring existing read/generate/refresh semantics for analysis and quiz.
   **[Code-Confirmed] + [Future Direction]**
3. Summary artifacts should be persisted as reading interaction artifacts with
   `interaction_type=summary`, hierarchy-aware target metadata, source hash/structure
   version when available, prompt instruction version, schema version, and ancestor
   summary context metadata. **[Future Direction]**
4. Summary preparation must not mutate `/documents/task-layout`, profile diagnostics,
   parser metadata, hierarchy truth, content-block segmentation, or UI state.
   **[From HLD] + [Future Direction]**

Note action contract:

1. Notes are user-authored payloads attached to hierarchy-aware reading targets. They
   should support create/read/update/delete or save/list/delete semantics through an
   explicit route family; exact route naming remains a schema/route task.
   **[Future Direction]**
2. Note payload should include bounded text, optional title, optional tags, timestamps,
   and target metadata. It should not include raw target content, LLM prompt text,
   hidden UI drawer/menu state, or generated analysis/quiz/critical-thinking payloads.
   **[Future Direction]**
3. Maximum text length must be enforced at schema/service boundary before persistence.
   Initial planning should choose explicit limits, such as a per-note character cap and
   optional per-target note count cap, with stable validation errors for oversized
   notes. **[Maintainer-Provided] + [Future Direction]**
4. Notes should be stored as user-authored interaction artifacts or annotation artifacts
   keyed by document/target identity plus note id. They must not collide with current
   single-artifact actions like analysis, quiz, summary, or critical-thinking sessions.
   **[Future Direction]**
5. Note writes are explicit user mutations. Reads must not call LLMs, prepare documents,
   reparse structure, regenerate summaries, update task-layout, or mutate profile
   diagnostics. **[From HLD] + [Future Direction]**

Recommended implementation decomposition:

1. Define `summary` and `note` schema vocabulary and interaction-type constants.
2. Add summary payload/service contract and prompt instruction version.
3. Add summary generator with higher-level summary context and strict JSON validation.
4. Add summary artifact persistence and read/generate/refresh orchestration.
5. Add note payload validation and storage semantics with explicit length limits.
6. Add app and route exposure only after service contracts are covered by regressions.
7. Add frontend handoff only after backend route contracts are stable.
