# api_schemas.py Detailed Design

## Rename command request identity (2026-11-11, implemented)

Rename requests require command_id (UUID) and expected_source_token (64 lowercase hex characters), in addition to doc_name/action/new_doc_name. These fields are only valid for rename. DocumentRenameContextResponse exposes doc_name and source_token from the pure rename-context GET route. The existing operation_id is the server operation identifier, distinct from caller command identity. Retry repeats the same payload; a new explicit command obtains current context and a fresh UUID. Existing accepted/conflict/recovery response and upload contracts remain unchanged.

## Main rename recovery response (2026-11-11, implemented)

`DocumentLifecycleActionResponse` adds `recovery_required` (default false) and
optional `operation_id`. A rename whose durable outcome needs reconciliation uses
HTTP 503 / failed / storage_failure with recovery_required=true and its operation
ID plus target name. The caller can repeat the same explicit rename to recover;
reads do not mutate. These fields do not claim that effect flags are a complete
rollback report. Upload response fields/reason vocabulary are unchanged; all size
limits map to the existing 413/file_too_large contract.


## Vocabulary assessments and completion wait (2026-11-09)

Saved hint and partial hint payloads add nullable `assessment` with candidate,
accepted, filtered, rejected and omitted-sentence counts, plus typed safe issue
code/count pairs. Validation enforces domain count consistency and accepted item
count. Older responses may omit the field; absence means unknown coverage. Pure
projections explicitly serialize issue tuples as JSON arrays under strict schemas.
No model output, raw rejected text or provider error is exposed in assessments.

The completion wait endpoint below reuses `LearningOperationRequest` and
`LearningOperationResponse`; it adds no source-authority or retry fields. Runtime
waiting/disconnect behavior belongs to learning/main, not schema validation. Schema,
file/PostgreSQL HTTP and actual API Reader checks cover saved/partial projection
and the additive route. Existing error and committed-result proof rules remain.

## Language Learning Reading Mode Contracts (LLRN-14, 2026-10-28)

**[Code-Confirmed schemas/projections; production routes delivered by LLRN-16]** [Canonical design](section_tasks/module-detailed-design.md#language-learning-reading-mode-plan-2026-10-28)；[task dependency queue](section_tasks/module-checklist.md#language-learning-reading-mode-queue-2026-10-28)。`api_schemas.py` 保持公開 request/response import facade；同一 owner 的 `api_learning_schemas.py` 集中嚴格 DTO，`api_learning_projection.py` 只做已驗證 domain snapshot → public response 的純轉換。Schema 本身不持有 route、DI、provider/storage 呼叫或 user identity；既有普通翻譯契約不變。

### Fixed transport surface

以下為 LLRN-16 已在 `main.app` 註冊的正式契約命名；實際 main/OpenAPI、file/真 SQL HTTP 驗證已通過，但 source 註冊不等同部署。成功接收命令使用 HTTP 200 typed outcome，並不代表生成或提交完成；錯誤使用下述獨立 error union。

| Method / path | Request | Response |
|---|---|---|
| GET `/reading-preferences/options` | 無 body | `ReadingOptionsResponse` |
| GET `/reading-preferences/workspace` | 無 body；scope 由 server 注入 | `ReadingWorkspaceResponse` |
| PUT `/reading-preferences/workspace` | `ReadingWorkspaceSaveRequest` | `ReadingWorkspaceResponse` |
| POST `/documents/reading-preferences/read` | `ReadingBookReadRequest` | `ReadingBookResponse` |
| POST `/documents/reading-preferences/apply` | `ReadingBookApplyRequest` | `ReadingBookResponse` |
| POST `/documents/reading-interactions/learning/read` | `LearningReadRequest` | `LearningReadResponse` |
| POST `/documents/reading-interactions/learning/generate` | `LearningGenerateRequest` | `LearningSubmissionResponse` |
| POST `/documents/reading-interactions/learning/operations/read` | `LearningOperationRequest` | `LearningOperationResponse` |
| POST `/documents/reading-interactions/learning/operations/wait` | `LearningOperationRequest` | `LearningOperationResponse` |
| POST `/documents/reading-interactions/learning/operations/reconcile` | `LearningOperationRequest` | `LearningSubmissionResponse` |
| POST `/documents/reading-interactions/learning/operations/partial` | `LearningPartialReadRequest` | `LearningPartialReadResponse` |
| POST `/documents/reading-interactions/learning/commands/read` | `LearningCommandLookupRequest` | `LearningCommandLookupResponse` |

### Settings, mode list and registry

所有 DTO 使用 `strict=True`、`extra=forbid`，不把 bool/string/float 當 revision，不 trim identity 或 silently normalize 語言。Document identity 拒絕路徑、`.`/`..`、空白包邊與控制碼，最多 255 字元；command/mutation ID 最多 128，偏好 mutation ID 同時遵循 domain identifier 規則。這是 transport identity 界限，不是正文／模型字數上限。

Workspace save 僅含 nullable `native_language`、`expected_revision`、`mutation_id`。Options 從現有 registry 提供 codes 與每語言 proficiency scheme/version/ordered levels/labels/reading descriptions，不維護第二份語言列表。Book read 僅提交 `doc_name`，回應含 book/workspace revisions、母語、source override/resolved language/origin、saved active/effective mode、帶 label/description/settings version/eligibility/reasons 的模式清單，以及已保存設定描述。來源無效／未知 saved mode 時 effective mode 為 null，不能偷偷改成 normal。

Book apply 僅提交 `doc_name`、兩個 expected revisions、`mutation_id`、以 `mode_id` 區分的 `selection` 及可選 `source_language_override`。Normal selection 只有 `mode_id=normal, version=1`；language_learning selection 加 language_code/scheme_id/scheme_version/level_id，重用 registry selection validation。未知 mode、跨模式欄位、版本不符或額外 payload 拒絕。普通模式不要求學習欄位。

`to_settings(previous)` 是純 merge：previous 必須是 server 讀取的同書／同 expected book revision，保留其它模式及未知持久化設定；不允許 client 回傳任意 JSON 或保留憑證。Override 省略保留原值，顯式 null 清除並回到 existing profile。LLRN-16 已將 merge 結果交既有 `ReadingPreferencesService.save_book`，該服務仍在 repository 寫入邊界校驗兩個 revisions、保留規則及 eligibility。母語不在 book apply 中重複提交；同語禁止依 server workspace/profile 驗證，schema 不以 client 語言宣稱來源授權。Mode/settings 一次套用；單純候選選取不 mutation。

未知／失效 saved settings 可讀，但回應只有 mode/version/supported=false/selection=null 描述，不公開或讓 client 重交原始 payload；未變更的 domain 值由 server merge 保留。切回普通模式仍可行。這些 pure projections 不修復或補寫資料。

### Current-page commands and source expectations

Read 要求唯一 `content_block_ids`，每次 1–50 個，與 learning 的 100 component-key query bound 一致。Generate 要求 doc_name、content_block_id、component（translation/hints）、command_id、`expected_source_token`、expected_workspace_revision、expected_book_revision。Intent 是 get_or_generate／regenerate／retry；不接受 raw source、internal source binding、parent plan、任意 namespace/user/workspace ID、source/target language 或模型參數。

`learning_source_token` 是 SHA-256，輸入固定為 canonical JSON `["learning-api-source-v1", source.to_dict()]`（UTF-8、sort_keys、無格式空白），綁定完整 server scope/document/hierarchy、epochs/hash/structure version。Public request 只回傳這個 opaque token，不接收 parent IDs。Token 只作比較，不是簽章或權限憑證。LLRN-16 已透過 learning source-binding port 從 server-scoped repository 解出當前 block，呼叫 `request.verify_source(actual_source)`，再把該 actual source 傳給 learning；learning claim/commit 仍重新驗證，不能從 token 還原或相信 client source。

Operation handle 僅 `{doc_name, operation_id}`，讀取、reconcile、partial 與 retry 都帶此 handle。Retry 必須同書、指定原操作且使用新 command ID；unknown 是否需要 `acknowledge_possible_duplicate_cost=true` 仍由 durable domain state 決定。省略確認不是自動批准，會依原 policy 回應 reconcile。Schema 可拒絕嵌套 handle 與外層書籍不一致；無法只靠 JSON 判斷某 operation 實際屬於哪本書，LLRN-16 已使用 scoped lookup，projection 也對照 authoritative operation binding，不能全域搜尋後跨書返回。

### Results, receipts and recovery states

Read response 綁定同書 settings、source token、目前 block 及結果語言／能力級別。Sentence projection 用 Unicode code-point 半開區間與原文 text，保留全部空白、CRLF、組合字和 emoji；空白 gap 的 sentence_id 為 null。後端驗證完整區間覆蓋、譯文完整有序 sentence IDs 與 hint span/source form，不能用 UTF-16 offset 替換。

每 block 的 translation/hints 各含 state、nullable saved、operation handle/state。已保存且驗證通過才 ready；`items=[]` 是有效 ready hints。沒有結果時 claimed/running 投影為 pending；partial、failed、stale、unknown、unavailable 與 missing 分開。Saved ready 可同時有 running/failed/unknown refresh；失敗不能抹掉舊 saved 或另一 component。Read/storage failure 走 error union，不可冒充 missing。

Submission 是 durable acceptance/reuse/reconcile outcome，不載入或冒充完成的正文。Claimed/coalesced/resuming 不帶 committed result revision；reuse 必須從 durable command receipt 取 revision，succeeded 必須有成功 operation proof。`submission_response` 對照同書 command/operation；command lookup 的 receipt 是原接受結果，舊 claimed receipt 不因稍後 succeeded 而重寫成完成，前端另讀 operation/result。失去 acknowledgement 可返回 unknown + 無 operation handle，caller 使用原 command ID lookup。

Operation response 分開 current projected state 與 durable_state，並帶 source_current、settings_current、owns_current_slot、completed/total batches、nullable committed result_revision、outcome_uncertain 與安全 failure_code。歷史 succeeded 可因來源改變投影 stale，不能當目前結果；running/unknown 不得把不確定費用標成 false，且無完整 checkpoints 不得聲稱 succeeded。

Partial read 的 offset 非負、limit 1–100，回應為明確有序 batch list/next_offset，不帶 result revision 或 ready claim。即使全部 batches 已 checkpoint，仍須 final commit 才是 saved success；空 hints batch 仍保留 batch index。Pure partial projection 的輸入必須來自 service/repository 的 current-binding guarded `read_partial`，不能接受任意 provider output。

### Error union and route obligations

LLRN-16 supporting additive reason：`PreferenceReason` 新增 `learning_mode_not_active`，
與 preference domain 一致，表示生成時未選用該模式；不把 same_language／profile failure
混用。原 DTO／11 routes 路徑與 request fields 不變。Partial response 現由 app 的
`read_partial_snapshot` 提供一致 status/batches，並行變動返回 409；失效 active mode 的
`effective_mode` 為 null，不默默將持久化 active mode 改成 normal。

`LearningErrorResponse.detail` 以 kind 區分 invalid_request、conflict、unavailable、ineligible、not_found、unknown；每種有有限 code union。Eligibility 附 domain stable field/reason/message；unknown 明確要求 lookup_command 或 read_preferences，不能推斷 rollback。Error 沒有 stack、provider body 或任意 debug JSON。Admission denial 是 submission outcome，reason 是 domain policy 的 bounded stable token，不偽造用戶 quota。

LLRN-16 的 route 已對應 400/422 invalid selection/capacity、404 missing target、409 conflict、503 unavailable/unknown，以及 scoped service exceptions；不能用 catch-all 轉 missing。FastAPI `RequestValidationError` 也必須轉成此 content-free invalid_request union，避免預設 validation body 重送 client input；isolated harness 已驗證 422 shape，production route-local handler 已由 LLRN-16 接線。Schema/投影不導入全域 exception handler，也不改其它 API 的錯誤格式。

### Verification (2026-10-28)

`scripts/test_learning_api_schemas.py` **56 tests passed**：嚴格 JSON/Python validation、所有 registry 語言／分級、未知 mode/version、normal 無學習欄位、保留 inactive/unknown settings、權威同語拒絕、source token 全 identity/epoch 綁定、跨書 handles、bounded reads、Unicode exact projection、empty hints、refresh/partial/unknown、committed revision proof、error union，以及 11 個 transport operations 的 isolated FastAPI OpenAPI/HTTP contracts。另以 temporary file repositories + 現有 app/service + 固定 provider 驗證 read 零寫入／零模型、生成保存／reuse、command lookup、partial checkpoint 和未知 provider outcome 的 public projection。

154 項 preference/proficiency/shared/policy/file repository 回歸及五份既有 API schema scripts 通過；`api_schemas.py` 僅 additive re-export，舊 DTO 不變。未修改 DB/schema/parser/LLM 演算法，因此本輪未新增 SQL、真模型、真書或 browser acceptance；LLRN-10/11 語義品質未通過的紀錄維持。LLRN-15 DI 已交付；LLRN-16 production routes/error mapping 已交付，UI 仍待實作，完整模式不可用。

## Persisted Translation DTO Plan (2026-10-26)

**[Code-Confirmed]** Supporting TRUI-01/05, not another counted task.
[Minimal transport plan](main.module-detailed-design.md#persisted-translation-api-plan-2026-10-26).
Read input has document/block identity only; output has backend registry codes,
nullable existing-profile source default, checked source identity and bounded saved
language sections. Generate retains optional source_language, rejects unregistered
source/target codes and accepts a bounded replay ID, never original text/parent IDs.
Saved entries expose target-code grouping, resolved source, text/status, timestamps
and revision/commit binding. Invalid model/commit outcomes cannot claim persisted.
No-change/changed-body response fields must separate mandatory translation/summary
invalidation from the user's Keep/Clear choice for other metadata. Do not widen all
existing interaction envelopes or reuse a blanket generated-invalidated flag to
claim unrelated retained data was deleted. TranslationReadRequest/Response and
TranslationSavedRecord are implemented; generate supports optional bounded mutation_id
and persisted/record evidence. Public request codes use the actual registry while
the standalone service retains broader language tags. ReadingContentRefreshResponse
adds body_changed, summaries_invalidated and translation_block_ids without widening
other interaction status enums. Route/schema and frontend DTO tests pass.

Reuse correction (2026-10-26): generate may return an already committed record for
the selected target language. Its source_language and mutation_id are original
provenance, not necessarily the new request values. Preserve payload/record parity,
target identity and effective hash checks; do not reject valid reuse as a mismatched
new mutation. No new fields or force-regeneration parameter are required.

## Block Translation Contract (2026-10-26)

**Historical TRANS-API-02 schema scope, superseded by the persisted DTOs above.**
The script/region-tag statement below describes the initial contract. Current public
read/generate schemas use exact registered language codes; the standalone text service
retains its broader tag support. Current generation also accepts `mutation_id` and
returns `envelope.persisted` plus a saved record or null. The UI retry/highlight fixes
do not add fields or change backend storage identity.

TRANS-API-02 supporting schemas: TranslationInteractionRequest requires doc_name,
content_block_id and target_language; source_language defaults to None. Reject extra
fields/source text/parent IDs, empty or malformed language codes, missing identities
and unsafe document path names. Preserve supported script/region language tags.
Dedicated translation envelope keeps partial/untranslatable local to this action,
without widening analysis/quiz/session artifact statuses. Response validates nonempty
text for completed/partial and null text for untranslatable. See the main module's
Block Translation API for the minimal response binding and error statuses.

## Source Review Block Commands (2026-10-26)

SRBLOCK corrects the UI mutation target to one block. ReadingContentMutationRequest
accepts review_block_edit/delete/keep with mandatory block_id and region_id. Edit
requires nonblank bounded text and explicit boolean discard_interactions; Delete
and Keep reject text/consent fields. Cleanup-record IDs and unknown fields are
forbidden. No caller origin flag or offsets. ReadingBlockResponse optionally adds
review_repair with can_edit/can_delete/can_keep/blocked_reason; older servers remain
read-only in the UI. The existing reading-content route/refresh schema is reused.
The old region commands below remain compatible but are no longer UI entry points.

## Source Region Repair DTOs (2026-10-26)

Implemented support for domain-owned SRREPAIR-04, counted only in its checklist.
ReadingContentMutationRequest adds region_edit, region_apply and region_delete with
required region_id, source_token, expected_revision and mutation_id. Region commands
reject block/cleanup target IDs. Edited text must be nonblank and at most 10,000
Unicode code points; Apply requires an explicit boolean discard_interactions. Draft
Save/Delete forbid that field, and Delete forbids text. Domain validates again under
the write lock; no caller-supplied offsets, application proof or manual-edit bypass.

SourceReviewRegionResponse optionally includes strict RegionRepairResponse with
text/edited_at, applied_text/applied_at and can_edit/can_apply/can_delete/blocked_reason.
Missing optional fields remain compatible with older clients/servers. Internal
canonical spans and ownership never enter public DTOs. ReadingContentRefreshResponse
adds cleared_block_ids (default empty) for cross-block artifact/Note invalidation;
the existing paged reader block/tombstone and source/revision binding remain unchanged.
The existing reading-content routes handle transport without a second repair endpoint.

## Cleanup Record DTOs (2026-10-25)

RCR-03 adds strict `ReadingContentRefreshResponse` / `ReadingRefreshBlockResponse`
and optional `ReadingContentResponse.refresh`. GET can return the standalone minimal
refresh shape. Reuse `TaskUnitContentBlockResponse` for visible body content and the
existing fixed-20 page model; do not expose internal ownership/audit payloads.
Validate unique block IDs, block/body identity, visible-versus-tombstone consistency,
current/mutation revision agreement, complete page size and no stale-scope body data.
Include hierarchy/order/next-unaffected anchor, unit hash, backend manual flag and
the actual interaction invalidation flags. Null fields may be omitted on the wire;
missing lazy artifact/availability fields do not assert not_generated. Full contract
and behavior belong to document_structure RCR-03, not a second schema-owned task.

CLEAN-RESTORE extends ReadingContentMutationRequest with cleanup_edit/cleanup_restore
and an optional 64-hex cleanup_record_id, required by the service for those actions.
Existing source_token, expected_revision and mutation_id remain mandatory. Body/region
targets and discard_interactions are not accepted for record mutations. Explicit
restoration-text edits retain the manual 10,000-character limit; this is independent
of the removed LLM cleanup proposal/quote limits. Original long quotes can restore
without resubmitting edited text.

ReadingBlockResponse includes server-owned manually_edited. Cleanup audit records
expose immutable quote, saved text/time, record_id, status, can_restore/restore_reason
and verified block links with Unicode char_from/char_to and can_navigate. No client
offset authority or UI-derived permission. Legacy UI capability fields are optional
only for read-only compatibility with older servers. Actual API serialization and
mutation validation are covered by test_cleanup_history.py; owner task CLEAN-RESTORE.

## Standalone Cleanup DTO Plan (2026-10-23)

**[Code-Confirmed: SCLEAN-06, 2026-10-24]** Supporting main SCLEAN-06, counted once
there. Three routes are implemented in [transport plan](main.module-detailed-design.md#standalone-cleanup-transport-plan-2026-10-23);
SCLEAN-07 supplies the namespace-scoped runtime service for supported storage.
Unsupported/missing runtime ports remain unavailable; no DTO change or deployment
is implied by composition.
Use dedicated cleanup DTOs, not ReparseDocumentStructureRequest/Response with
invented parser_mode, structure_changed or discard_interactions values.

Preflight needs only the route document identity. Response exposes execution_available,
allowlisted reason, cleanup impact (generated invalidation only on actual change;
notes/original retained), and opaque operation_id/confirmation_token when available.
Bind revisions, source, policy/prompt and capacity server-side; keep raw hashes,
budgets, provider configuration and executable plans off this public response.
The availability response is also the UI capability check; no general introspection
API or duplicated task-layout payload is required.

Execute request accepts only bounded strict operation_id/confirmation_token. The
result and lookup projection separate operation state/phase and commit certainty
from CleanupOutcome (status/counts/safe reason/content-change/revision), current
reading readiness, remaining review count and the allowed recovery action. Failed
planning is not successful content mutation; unknown commit is not failed planning.
Replay, superseded state, stale consent, zero-applied count and no-change revision
invariants need schema/route tests. No raw exceptions, source text, plans, arbitrary
thresholds, offsets or provider details. Exact DTO enum/limits are frozen by SCLEAN-01/06.
Unknown fields are forbidden. Legacy enhance responses remain compatible and keep
their structural committed/success meanings; cleanup-only DTOs must not inherit
those ambiguous semantics.

### Implemented Contract

`SmartCleanupPreflightRequest` is an empty strict object; the body may be omitted.
`SmartCleanupExecuteRequest` requires only a 32-character lowercase hexadecimal
operation_id and 43-character URL-safe confirmation_token, both strict strings.
No action/confirmed/parser/discard/namespace/provider fields are accepted. Repeat
POST uses these same handles for backend-directed layout continuation; a fresh
preflight creates a new attempt. GET accepts the operation ID in its route only.

`SmartCleanupPreflightResponse` exposes execution_available, allowlisted reason,
fixed impact (generated invalidation only on content change; notes/manual edits/
original/structure preserved), nullable operation_id/confirmation_token and
expires_in_seconds (1..300). Unavailable responses cannot carry handles; available
responses require all handles and no failure reason. No model/capacity hash or
prompt/budget fields are exposed.

`SmartCleanupOperationResponse` contains operation_id, phase, commit, nullable typed
CleanupOutcome, allowlisted reason, retry_action, nullable reading_ready and
pending_review_count, plus durability. Phase/commit/retry vocabulary follows the
existing domain operation; bindings are never serialized. reading_ready is true
only for finished+committed, false for layout, otherwise null. Review count is null
when no trustworthy current review snapshot is available, not zero. Unknown state
requires unknown commit, null outcome and reconcile; changed content requires a
confirmed commit. CleanupOutcome retains its strict count/status/revision guards.
No structural success/change_kind fields are invented for cleanup-only work.

`SmartCleanupErrorResponse` has only an allowlisted detail. OpenAPI declares typed
state-or-error unions on 404/409/503 and a sanitized error on 422. The scoped route
wrapper never echoes invalid input/token/source or raw exception payloads. All new
DTOs forbid extras and coercion; existing enhance DTO behavior is unchanged.
Evidence and the 56 file/SQL HTTP cases are counted in main SCLEAN-06, not a second
schema checklist task. Live/browser acceptance remains SCLEAN-10.

## Optional Cleanup Contracts (2026-10-23)

Supporting app CLEAN-06: preflight/reparse/result requests forbid unknown fields.
allow_ocr_cleanup is a strict boolean default false; the existing reparse adapter
rejects true outside llm_enhanced. This remains compatibility behavior, not the
planned standalone capability's eligibility rule. Preflight ocr_cleanup is nullable for legacy/failed readiness and
otherwise exposes enabled, eligibility/reason, fixed scope/policy, reading_revision,
conditional generated-output invalidation and note-preservation facts. Existing
cleanup_scope/cleanup_condition still describe structural interaction deletion.

Reparse adds nullable typed CleanupOutcome plus phase (layout/complete/superseded),
retry_stage (generation/layout), safe layout_error, review_required and nonnegative
pending_review_count. Structural change_kind/committed/interactions_cleared retain
their old meaning; cleanup-only content change is not a structural commit.
Operation durability now allows persisted_commit as well as process_local.
status=completed proves receipt commit; consumers must inspect phase before opening.
After Undo/newer edits, phase=superseded has historical cleanup outcome but current
review counts; it does not authorize recovery writes. Missing optional fields never
mean cleanup succeeded. App CLEAN-06 tests strict request coercion rejection,
legacy defaults, actual routes and restart/rollback semantics.

## Reading Content Contract (2026-10-21)

ReadingContentMutationRequest forbids unknown fields and restricts action to edit,
delete, undo, discard_region. It requires a nonnegative strict integer revision,
64-hex source token and bounded mutation ID; text is limited to 10000 code points
and service validation rejects blank edits. There is no manual block insertion.
Additive management review_required and note source_stale flags default false.
SRMAP-04 (2026-10-25) adds strict ReadingContentResponse, SourceReviewRegionResponse,
ReadingBlockResponse, ReviewManualChangeResponse and ReadingBlockPageResponse.
GET and POST serialize through the same response contract; internal source geometry,
canonical spans and persistent mapping records are excluded. Optional structure
version and manual-change timestamp remain absent when unavailable. Region fields
separate coverage, body inclusion, mapping reason and related deletion count;
historical block IDs are not a claim that all blocks survive. Current navigation
targets carry chapter/section/unit/block IDs in bounded 20-block pages. Existing
mutation fields/actions remain unchanged. Source preview returns bounded PNG bytes.
Evidence and task ownership: document_structure SRMAP-04 and test_source_review_api.

## Summary And Note Contracts (2026-10-21)

Summary has dedicated read/generate/refresh request types and a strict shared
envelope response carrying headline/body/key_points only when completed. Note
uses read/save/delete requests and a nullable single-note response with target,
target_token, server note_id, revision, timestamps and max_chars=10000. Save
requires explicit expected_revision/mutation_id; writes forbid extra fields,
blank text and oversized Unicode text. Corresponding routes are now implemented
in main; schema evidence: test_summary_note_contracts.py, route evidence in
test_note_interaction_service.py and test_summary_interaction_service.py.

## FULL-03 Capacity Contract (2026-10-20)

EnhancementPreflightResponse now includes nullable EnhancementCapacityResponse.
It exposes fits/exceeded/unavailable/resource_limited, safe reason codes, selected
model, context/input/output limits, output/margin/framing reservations, required
and available input tokens, count method/identity, full-source coverage and policy/
prompt versions. Unknown measurements are null; no prompt or private fingerprint.
Preflight readiness is source readiness, not capacity fit. Token requires both
execution availability and fit. Consent recheck 413/503 errors preserve string
detail and add the same filtered capacity object. Main/app own status/auth, not
schemas. Completion and tests belong to app FULL-03; UI integration is pending.

## Enhancement Consent Schemas (2026-10-19)

Preflight contract and PostgreSQL reader implemented; execution remains gated.
Supporting work inside ENH-APP-01. Preflight accepts
document identity, `parser_mode=llm_enhanced`, `toc_hint` and compatible legacy intent.
Response describes `ready`, a machine-readable `reason`, current structure version,
cleanup scope/counts (nullable when unknown) and a short-lived `confirmation_token`
only if ready. No source text or private paths are exposed. Token is server-bound to
document namespace, source/canonical identity, structure/interaction revisions and
selected options, not a client-asserted boolean or hash.

Enhanced reparse additionally requires `discard_interactions: true` and the issued
token; never default destructive consent to true. Missing consent returns
`409 confirmation_required`; stale/expired state returns a distinguishable conflict
requiring renewed preflight. Invalid source readiness returns actionable preparation
guidance without invoking extraction. Successful enhanced execution now carries committed
structure version so consumers distinguish replacement from later layout failures.
Current DTOs expose the preflight fields above and strict-boolean discard consent
(default false) with a bounded token on reparse. Optional structure_version is populated
after enhanced commit and remains null for other legacy flows. Preflight resolves readiness
and impact; unsupported file storage fails explicitly. `execution_available=false`
is distinct from ready: a valid token alone cannot bypass missing migration/profile
guards or enable OCR fallback. PREP-03 tests route-to-storage and browser contract parity.
Tests cover 409 missing/invalid/expired/stale consent, 503 gated execution and 422
invalid input. Do not alter the one-operation enhancement policy.

## Enhancement Semantics Correction (2026-10-19)

Existing EnhancementIntent enum values/default remain wire-compatible, not two
user-facing operations. Both enhanced requests now optimize Common results and
may propose labelled semantic navigation. The planner owns this policy; request
mode guards/forwarding remain unchanged. Source/generated origin comes from the
accepted boundary metadata, not the intent value. The two-purpose interpretation
in historical NATIVE-04 notes is superseded by document_structure's unified correction.

## READER-01 Public Provenance (2026-10-19)

Task-layout responses now expose nullable toc_hint, toc_observation,
enhancement_intent, implicit_single_body and declaration_unconfirmed through
ParseProvenanceResponse. String values use Literal allowlists; absent legacy
evidence remains null. SectionTaskLayoutResponse adds nullable title_origin
(source/implicit/generated). No prompt, source text or full boundary map is exposed.
Evidence: `scripts/test_reader_provenance.py` exercises real route serialization,
unknown evidence and exact-range origins without persistence mutation. This
supersedes the older NATIVE-05 public-projection staging note below.

## NATIVE-05 Runtime Status (2026-10-19)

Existing typed enhancement intents are now executable through explicit preparation
and reparse. The NATIVE-04 staging guard is removed; validation/provider failures
remain errors rather than successful fallback writes. No additional public response
fields or inferred generated-origin guarantees are introduced. Internal v2
provenance/origin persistence is documented by document_structure; public projection
mapping still needs an explicit contract before UI consumption.

## NATIVE-04 Enhancement Intent (2026-10-19)

Prepare, prepare-task-layout and reparse request schemas accept EnhancementIntent:
restore_source (default) or generate_reading_outline. Generation requires explicit
llm_enhanced and cannot be combined with common/manual/layout-only execution.
Schema acceptance is not application readiness: generated runtime execution returns
400 reading_outline_application_not_available until NATIVE-05. Strict versioned
candidate models are internal planning DTOs, not new public response fields.
Evidence: structure-plan and API contract tests.

## NATIVE-03 Request Contract (2026-10-19)

PrepareDocumentRequest, PrepareTaskLayoutRequest and ReparseDocumentStructureRequest
now expose shared `TocHint` enum `auto|present|absent`, default auto. Invalid values
are validation errors. It is an advisory printed-TOC declaration, independent of
PDF extraction mode; it cannot invent source structure. Non-auto hints are rejected
with manual-structure reparse. No new public response/provenance fields are claimed.
Evidence: `test_common_parse_policy.py`, `test_pdf_extraction_mode_contract.py`.

## 1. Module Purpose

LAYOUT-02 (2026-10-16): `PrepareTaskLayoutRequest.layout_only` defaults false;
true skips raw/structure preparation, retaining explicit task-unit persistence and
readback. It cannot combine with force_rebuild or page-evidence requests (400).
PDF mode is not executed in layout-only mode, so successful metadata remains unknown
with null effective_mode. Ordinary layout `refresh_task_units=true` now returns 409,
as do missing/stale/invalid layouts, with recovery guidance in string `detail`.
Read split options validate existing cache, never invoke resolution. Success DTOs
and management structured-ready semantics are unchanged. Owned by app LAYOUT-02.

`api_schemas.py` 定義 REST API 的 request/response schema contract（Pydantic models），是 API 層與 orchestration 層之間的資料邊界。 **[Code-Confirmed]**

## 2. Position in Overall Architecture

### PDF Extraction Contract (2026-10-15)

PDF-MODE-01 adds the shared `PdfExtractionMode` enum (fast/ocr/auto, default fast)
to both preparation requests, independently of parser/base/free_qa settings.
`PdfExtractionMetadataResponse` separates requested mode from effective mode and
unknown/not_applicable/applied/reused provenance; effective mode requires applied
or reused provenance and must match a supplied requested mode. Legacy assets remain
unknown, never relabeled fast. `PrepareDocumentResponse` and the dedicated
`PrepareTaskLayoutResponse` expose this metadata; ordinary layout DTOs do not.
`PdfExtractionFailureResponse` exposes doc_name/reason/detail/pdf_extraction.
422 also permits standard FastAPI validation detail arrays. Shared enum/error types
live in the IO-free `doc_loaders/pdf_extraction_contract.py`; schemas do not run loaders.
PDF-MODE-05 now integrates the mode executors/reuse and removes the temporary 501
gate. Installed engines and source quality still determine runtime success. Evidence:
`scripts/test_pdf_extraction_mode_contract.py` (11 tests); status owned by main PDF-MODE-01.

- API Layer

## 3. Key Files

| File | Responsibility | Notes |
|---|---|---|
| `api_schemas.py` | REST DTO 定義與輸入校驗 | 含 prepare/ask/task-layout/summary/quiz/reparse schemas **[Code-Confirmed]** |

## 4. Main Responsibilities

1. 定義 endpoint payload contract。 **[Code-Confirmed]**
2. 對關鍵請求做基礎 validation（例如 chapter_id/chapter_title at-least-one）。 **[Code-Confirmed]**
3. 約束 task-layout public response 為 chapters-first 結構。 **[Code-Confirmed]**
4. 定義 `ProfileStructureDiagnosticsResponse` mixed-source 語義說明。 **[Code-Confirmed]**

## 5. Non-Responsibilities

1. 不負責 business logic。 **[From HLD]**
2. 不負責 persistence 寫入。 **[From HLD]**
3. 不應承擔 parser strategy 決策。 **[From HLD]**

## 6. Important Data Structures / Contracts

- `PrepareDocumentRequest/Response`
- `DocumentListItemResponse`
- `DocumentListResponse`
- `DocumentManagementItemResponse`
- `DocumentManagementListResponse`
- `DocumentUploadMetadataRequest`
- `DocumentUploadResponse`
- `DocumentLifecycleActionRequest`
- `DocumentLifecycleActionResponse`
- `GetDocumentTaskLayoutRequest`
- `DocumentTaskLayoutResponse`
- `SummarizeChapterRequest`, `ChapterQuizRequest`
- `ManualStructureAnchorRequest`
- `ManualStructureEntryRequest`
- `ManualStructurePlanRequest`
- `ManualStructureValidationRequest`
- `ManualStructureValidationResponse`
- `ReparseDocumentStructureRequest`
- `ProfileStructureDiagnosticsResponse`
- `ReadingInteractionTargetRequest`
- `ReadingInteractionTargetResponse`
- `ReadingInteractionResponseEnvelope`
- `AnalysisInteractionReadRequest`
- `AnalysisInteractionGenerateRequest`
- `AnalysisInteractionRefreshRequest`
- `AnalysisArtifactPayloadResponse`
- `AnalysisInteractionResponse`
- `QuizInteractionReadRequest`
- `QuizInteractionGenerateRequest`
- `QuizInteractionRefreshRequest`
- `QuizArtifactItemResponse`
- `QuizArtifactPayloadResponse`
- `QuizInteractionResponse`
- `CriticalThinkingSessionReadRequest`
- `CriticalThinkingQuestionGenerateRequest`
- `CriticalThinkingAnswerSubmitRequest`
- `CriticalThinkingEvaluationRetryRequest`
- `CriticalThinkingEvaluationResponse`
- `CriticalThinkingSessionPayloadResponse`
- `CriticalThinkingSessionResponse`
- `ArtifactAwareInteractionMetadataResponse`

Critical-thinking submit and retry requests carry the shared `ReadingInteractionTargetRequest` plus `session_id` so route mapping can preserve hierarchy-aware session-store lookup through app orchestration. They do not define a route-local global session lookup contract. **[Code-Confirmed]**

### 6.1 Reading Interaction Route Integration Contract

The shared reading interaction schema family is now verified against the public REST route set for insight, quiz, and critical-thinking sessions. **[Code-Confirmed]**

1. Insight routes use `AnalysisInteractionReadRequest`, `AnalysisInteractionGenerateRequest`, `AnalysisInteractionRefreshRequest`, `AnalysisArtifactPayloadResponse`, and `AnalysisInteractionResponse`. **[Code-Confirmed]**
2. Quiz routes use `QuizInteractionReadRequest`, `QuizInteractionGenerateRequest`, `QuizInteractionRefreshRequest`, `QuizArtifactItemResponse`, `QuizArtifactPayloadResponse`, and `QuizInteractionResponse`. **[Code-Confirmed]**
3. Critical-thinking routes use `CriticalThinkingSessionReadRequest`, `CriticalThinkingQuestionGenerateRequest`, `CriticalThinkingAnswerSubmitRequest`, `CriticalThinkingEvaluationRetryRequest`, `CriticalThinkingEvaluationResponse`, `CriticalThinkingSessionPayloadResponse`, and `CriticalThinkingSessionResponse`. **[Code-Confirmed]**
4. All route families share `ReadingInteractionTargetRequest`, `ReadingInteractionTargetResponse`, `ReadingInteractionResponseEnvelope`, and `ArtifactAwareInteractionMetadataResponse` instead of defining route-local target, status, provenance, or UI-state variants. **[Code-Confirmed]**
5. Schema validation remains a boundary check only. It does not perform hierarchy lookup, prompt construction, persistence mutation, LLM calls, or frontend drawer/inline state management. **[Code-Confirmed] + [From HLD]**
6. `ReadingInteractionTargetRequest` now accepts `target_type=content_block` only when `task_unit_id` and `content_block_id` are present, while allowing optional parent `chapter_id` / `section_id` consistency ids. The shared target schema is used by insight, quiz, and critical-thinking route families, and route regressions confirm those families preserve `content_block_id` through app dispatch and shared envelopes without route-local target variants. Schema validation remains boundary-only; resolver source slicing, service generation, and artifact persistence are owned outside `api_schemas.py`. **[Code-Confirmed]**
7. Frontend per-block menus should build `content_block` targets from the task-unit content response by mapping `content_blocks[].block_id` to `ReadingInteractionTargetRequest.content_block_id` and carrying the response-level `task_unit_id` plus optional `chapter_id` / `section_id`. `content_blocks[].metadata.source_hash`, when available, may be passed as optional source evidence; rendered row position, block text, pagination index, and UI menu state are not valid target identity. **[Code-Confirmed]**

## 7. Public API vs Internal DTO Boundary

| Layer | Scope |
|---|---|
| Public API Schema (`api_schemas.py`) | 對外契約，應穩定、可版本化 |
| Internal DTO (`section_tasks/document_task_layout.py`) | 協調層與 service 層中間模型，可保留 transitional fields |

原則：internal transitional fields 不應直接暴露為 public API。 **[Code-Confirmed] + [From HLD]**

## 8. Main Flows Involving This Module

1. API request parse/validate flow。
2. API response serialization flow。

（此模組不直接參與 prepare/parse/artifact 算法流程） **[Code-Confirmed]**

## 9. No-Heavy-Payload Contract

`task-layout` 與 diagnostics 相關 schema 不應暴露 heavy payload，包括但不限於：
- `raw_text`
- `section.content`
- `summary.content`
- `quiz.items`
- 問答逐題全文內容

**[Code-Confirmed] + [From HLD]**

`DocumentListResponse` follows the same lightweight API principle for document discovery:

- returns document candidates only
- does not expose raw text
- does not expose chapter/section/task-unit hierarchy
- does not expose task-unit content or `content_blocks`
- does not include diagnostics/profile payload

**[Code-Confirmed]**

## 10. Cache/Validity Reason Code Surface (Current)

目前 schema 已支持 validity/reason 類欄位，但完整 reason-code taxonomy 仍待文件化統一。 **[Code-Confirmed] + [Needs Confirmation]**

方向：保留 API 可觀測 invalidation reason code。 **[From Proposal] + [Needs Confirmation]**

## 11. Persistence / Side Effects

- read persistence：否
- write persistence：否
- mutate structured document：否
- generate runtime projection：否（僅 schema）
- call LLM：否
- diagnostics only：否（只定義 diagnostics payload 形狀）

## 12. Known Legacy / Compatibility Behavior

No known legacy compatibility responsibility（僅 DTO 契約層）。 **[Code-Confirmed]**

## 13. Current Risks

1. risk：schema docstring 與實作行為漂移
- why：使用者理解成本上升
- guardrail：schema regression + doc sync

2. risk：internal transitional fields 誤暴露為 public API
- why：可能破壞 hierarchy-first外部契約
- guardrail：保持 task-layout response 僅 chapters-first映射

3. risk：validation 規則與 coordinator 語義不一致
- why：出現 400/500 邊界錯亂
- guardrail：request validator + integration tests

## 14. Open Questions for Maintainer

1. `profile_diagnostics` 是否要升級成獨立 API contract 文檔章節？
2. cache invalidation reason code 若 API 化，是否放在 diagnostics 或 recommendation metadata？
3. 是否要在 schema 層標記更多 compatibility/deprecation 註釋？

## 15. Suggested Next Documentation Improvements

1. 增加 endpoint-to-schema mapping 表。
2. 增加 error status mapping 表（validation/business/runtime）。
3. 補 no-heavy-payload 保證清單。

## 16. Future Direction Note: Rich Task-Unit Content Schema Evolution

> 本節同時記錄「已落地的 schema normalization」與「後續 future-direction」。  
> 已落地項目為 backward-compatible additive evolution，不引入 API breaking change。 **[Code-Confirmed]**

1. 已正式定義 top-level `TaskUnitContentBlockResponse`，作為 task-unit on-demand rich-content API 的官方 block contract。 **[Code-Confirmed]**
2. `TaskUnitContentResponse` 已正式收斂為 `content_blocks`（preferred future interaction payload）+ nullable `content`（compatibility/debug payload）雙軌 additive 形狀。 **[Code-Confirmed]**
3. `content` 仍保留，未標記 deprecated；但預設回應可為 `null`，需透過顯式 `include_raw_content=true` 取得 legacy raw content。 **[Code-Confirmed]**
4. block optional 欄位（`block_type` / `artifact_ids` / `metadata`）在目前 adapter 路徑下可為 `null`，且 response serialization 形狀固定。 **[Code-Confirmed]**
5. 上述 schema normalization 不包含 persistence migration、不包含 task-layout heavy payload 擴張、不包含 retrieval/LLM integration。 **[Code-Confirmed]**

6. future `TaskUnitContentResponse` 可由 simple `content: str` 演進為更完整 structured content blocks / segments。 **[From Proposal] + [Inferred]**
7. reading-interaction request schema already exposes `content_block_id` for `content_block` targets as the first public-contract slice. Full resolver/source-context/persistence semantics remain future work. **[Code-Confirmed] + [Future Direction]**
8. future block-level metadata 可擴展為 artifact metadata（discovery/availability）與 evidence/quote/annotation target metadata。 **[Inferred]**
9. future content blocks 可作 QA evidence target、quote target、annotation target、retrieval grounding target。 **[From Proposal] + [Inferred]**
10. content blocks 是 API interaction target，不是 persisted hierarchy level；不取代 hierarchy-first contract `chapters[].sections[].task_units[]`。 **[Code-Confirmed] + [From HLD]**
11. evidence/grounding metadata 不等同 parser authority，亦不得成為 persistence truth source。 **[From HLD] + [Inferred]**
12. `task-layout` response 邊界不變：保持 lightweight metadata/projection，不返回 heavy content payload；rich content 仍走 on-demand content API path。 **[Code-Confirmed] + [From HLD]**
13. content-block artifact target metadata 已以 `ArtifactTargetRefResponse` 正式化為 response-only pass-through contract（metadata only，不是 artifact persistence truth / existence proof）。 **[Code-Confirmed]**
14. task-unit content response 的 artifact target metadata glossary 目前僅允許：`source_hash`、`content_block_id`、`quote_span_start`、`quote_span_end`、`schema_version`。 **[Code-Confirmed] + [Maintainer-Confirmed]**
15. content endpoint context 下的最小 target constraints：`content_block` level 必須含 `task_unit_id + content_block_id`；`task_unit` level 必須含 `task_unit_id`。其他 level 保留 enum 表達能力，但不宣稱本 endpoint 支援 artifact write semantics。 **[Code-Confirmed] + [Maintainer-Confirmed]**
16. `target_level` 使用 shared single-source enum（`shared.artifact_target_model.ArtifactTargetLevel`）驗證，非法值在 schema/shared boundary fail-fast；不做 silent coercion/silent fallback。 **[Code-Confirmed] + [Maintainer-Confirmed]**
17. metadata glossary keys 由 `api_schemas.ARTIFACT_TARGET_METADATA_GLOSSARY_KEYS` 單一常量維護，schema validation 與 endpoint mapping 共用同一來源，避免雙份定義漂移。 **[Code-Confirmed]**

## 16.1 Document Management, Upload, And Lifecycle Schemas

> 本節記錄 file-management / upload 的 public schema 邊界。Lightweight upload
> metadata / response DTOs are implemented and used by the explicit upload route.
> Lightweight management status and lifecycle action DTOs are also defined, while
> destructive lifecycle route behavior remains future work. **[Code-Confirmed] +
> [Future Direction]**

1. File management should expose lightweight document-source candidates and lifecycle
   status, not raw text, hierarchy, task-unit content, profile payloads, retrieval
   records, or interaction artifacts. **[From HLD] + [Future Direction]**
2. `DocumentUploadMetadataRequest` defines the route-adjacent metadata DTO for explicit
   multipart uploads: optional display title plus explicit conflict policy vocabulary
   `reject`, `replace`, or `create_distinct`. File bytes remain FastAPI multipart input,
   not a Pydantic field. **[Code-Confirmed]**
3. `DocumentUploadResponse` returns backend-owned document identity plus lightweight
   source metadata: source filename, source type, source size, optional source hash,
   optional upload time, source status, preparation status, optional conflict policy,
   and failure reason when applicable. **[Code-Confirmed]**
4. `DocumentManagementItemResponse` and `DocumentManagementListResponse` implement the
   lightweight document-management list/status schema. They expose backend-owned
   document identity, source label, source filename/type/size, source status,
   preparation status, optional timestamps, and optional last error only.
   **[Code-Confirmed]**
5. Management schemas keep the no-heavy-payload contract and must not turn task-layout
   into a document management payload. They forbid extra fields, preventing raw text,
   hierarchy, task-unit content, content blocks, profile payloads, retrieval records, or
   interaction artifacts from being serialized through the management list.
   **[Code-Confirmed]**
6. Conflict handling vocabulary is explicit at the schema boundary and implemented by
   the upload route: `reject`, `replace`, and `create_distinct`. **[Code-Confirmed]**
7. `DocumentLifecycleActionRequest` defines the route-adjacent request DTO for future
   explicit lifecycle mutations. Supported action vocabulary is `archive`, `restore`,
   `delete`, and `rename`; `rename` requires `new_doc_name`, non-rename actions reject
   `new_doc_name`, and optional expected source/hash evidence is metadata only.
   **[Code-Confirmed]**
8. `DocumentLifecycleActionResponse` defines the lightweight response DTO for future
   lifecycle mutations. Supported status vocabulary is `accepted`, `rejected`,
   `conflict`, `failed`, and `not_implemented`; non-accepted terminal statuses require
   a typed reason. The response can distinguish source effects, structured-artifact
   effects, derived-resource effects, archive state, delete state, and rename target
   without exposing raw document payloads. **[Code-Confirmed]**
9. Lifecycle schema presence alone does not implement archive/delete/rename behavior.
   Future destructive routes must remain explicit, route-level, confirmation-aware, and
   delegated to storage/service cleanup policy rather than hidden behind list/read/open
   flows. **[Code-Confirmed] + [Future Direction]**
10. Maintainer-scoped lifecycle MVP currently targets only `delete` and `rename` route
   implementation. Both are implemented through explicit lifecycle routes. `archive`
   and `restore` remain reserved schema vocabulary and must not be surfaced as
   implemented actions until explicitly reprioritized. **[Code-Confirmed] +
   [Maintainer-Provided] + [Future Direction]**
11. `POST /documents/lifecycle/delete` uses `DocumentLifecycleActionRequest` and
   `DocumentLifecycleActionResponse`. The route requires `action=delete` and
   `confirm=true`, returns typed rejected states for missing/unconfirmed/wrong-action
   cases, and keeps responses lightweight with effect flags instead of raw text,
   hierarchy, profile payloads, retrieval records, or artifacts. **[Code-Confirmed]**
12. `POST /documents/lifecycle/rename` uses `DocumentLifecycleActionRequest` and
   `DocumentLifecycleActionResponse`. The route requires `action=rename` and
   `new_doc_name`, returns typed rejected/conflict states for missing/wrong-action/target
   conflict cases, and keeps responses lightweight with the accepted `new_doc_name` plus
   effect flags and recovery metadata described above. **[Code-Confirmed]**
13. Failure reason vocabulary covers unsupported file type, file too large,
   duplicate/conflict, unsafe filename, storage failure, missing source, and validation
   failure. Failed upload responses require a reason; successful upload responses must
   not include one. **[Code-Confirmed]**
14. Upload success must not imply preparation success. Preparation readiness belongs to
   explicit prepare/reparse contracts and should be represented as status, not hidden
   side effect. The upload response preparation-status vocabulary therefore excludes
   `ready` and is limited to `not_prepared`, `failed`, `stale`, or `unknown`.
   **[Code-Confirmed]**
15. Upload and lifecycle schemas forbid extra fields, preventing route-local expansion of
   raw text, task-layout payloads, prepare flags, hierarchy, content blocks, profile
   payloads, retrieval records, or interaction artifacts. **[Code-Confirmed]**
16. `POST /documents/upload` uses these schemas for route-adjacent metadata and typed
    responses. The route stores/registers raw source candidates only; post-upload
    prepare/open/retry orchestration remains separate future work. **[Code-Confirmed]**

## 16.2 Future Direction: Summary And Note Interaction Schemas

> 本節規劃新增 `summary` / `note` interactive action 的 public schema 邊界；
> 不代表目前 schema 已實作。 **[Future Direction]**

1. Summary should reuse the shared `ReadingInteractionTargetRequest`,
   `ReadingInteractionTargetResponse`, and response envelope pattern, while defining a
   dedicated `SummaryArtifactPayloadResponse` and read/generate/refresh request family.
   **[Future Direction]**
2. Summary payload should expose compact generated fields only, such as headline,
   body, key retained points, optional continuity hints, schema version, and prompt
   instruction version. It should not reuse `AnalysisArtifactPayloadResponse` or expose
   raw source text / raw ancestor summaries. **[Future Direction]**
3. Note schemas should represent user-authored text with explicit max length validation,
   optional title/tags, target metadata, note id, created/updated timestamps, and stable
   validation errors for oversized text. **[Future Direction]**
4. Note request/response schemas should not include LLM prompt fields, generated
   interaction payloads, raw target content, drawer/menu state, draft editing state, or
   frontend scroll/selection state. **[Future Direction]**

## 17. Future Direction Note: Manual Structure Override API Schema

> 本節記錄 source-agnostic manual structure override 的 public schema direction；不代表目前 implementation。 **[Maintainer-Provided] + [Future Direction]**

1. manual structure schemas should apply to any document type, not only OCR/PDF. **[Maintainer-Provided]**
2. public request naming should avoid OCR-only terms; prefer `manual_structure`, `manual_entries`, `manual_structure_mode`, or `manual_structure_plan` over OCR-specific vocabulary. **[Future Direction]**
3. schema should support validation/preview request and commit reparse request separately. **[Future Direction]**
4. manual entry schema should minimally include `title`, `level`, and one supported anchor form. **[Future Direction]**
5. anchor schema should be explicit and typed, for example `char_range`, `page_range`, or later `printed_page_number`; clients must not rely on implicit title matching as the sole locator. **[Future Direction]**
6. response schema should expose validation result, normalized entries, warnings, errors, preview hierarchy shape, and parse provenance preview without raw text/heavy content payload. **[Future Direction]**
7. commit reparse response may reuse/extend `ReparseDocumentStructureResponse`, but must identify `parser_mode=manual_structure` or equivalent source-agnostic mode. **[Future Direction]**
8. task-layout response schema should not be expanded with manual structure edit payload; task-layout remains lightweight projection. **[From HLD] + [Future Direction]**
9. user-defined manual structure is limited to two levels for now: `level=1` chapter and `level=2` section. `level>2` must be rejected by validation instead of folded. **[Maintainer-Provided] + [Future Direction]**
10. schema validation should distinguish malformed payload, unsupported anchor type, out-of-range anchor, overlapping range, empty projected range, invalid level sequence, unsupported depth, and stale source evidence. **[Future Direction]**
11. Implemented checkpoint: request-side manual structure schema now exists as `ManualStructureAnchorRequest`, `ManualStructureEntryRequest`, `ManualStructurePlanRequest`, and `ManualStructureValidationRequest`. It is source-agnostic, supports typed `char_range` / `page_range` anchors, enforces current two-level `chapter -> section` maximum depth, and does not add endpoint/runtime behavior. **[Code-Confirmed]**
12. Implemented checkpoint: validation/preview response schema now exists as `ManualStructureValidationResponse` with normalized entry, issue, preview chapter/section, and provenance preview DTOs. It is source-agnostic, keeps preview payload lightweight, validates the planned issue taxonomy, and does not add endpoint/runtime behavior. **[Code-Confirmed]**
13. Implemented checkpoint: commit reparse request schema can now represent `parser_mode=manual_structure` with a required `manual_structure` plan. Existing `common` / `llm_enhanced` modes remain accepted and reject manual plans; this checkpoint is schema-only and does not add endpoint/runtime behavior. **[Code-Confirmed]**
14. Manual structure commit route orchestration exists for explicit manual reparse, but page-backed manual commit remains incomplete until backend page-boundary evidence can validate and materialize `page_range` into hierarchy content. **[Code-Confirmed] + [Future Direction]**

## 18. Page-First Manual TOC API Support

> 本節支援 UI TOC editor 的 page-first anchor UX。
> 目前已完成 task-layout existing-structure anchor evidence response schema；page-backed manual validation/commit semantics 仍屬後續 checkpoint。 **[Code-Confirmed] + [Future Direction]**

1. Public manual-structure schemas should keep `char_range` as the universal fallback anchor and treat `page_range` as available only when backend page-boundary evidence is explicit and validated. **[Future Direction]**
2. `ManualStructureAnchorRequest` already has typed `page_range` vocabulary, but schema presence alone must not imply commit support. Validation/commit responses must distinguish "schema-valid but unsupported by available source evidence" from malformed payload. **[Code-Confirmed] + [Future Direction]**
3. Task-layout response schema now exposes lightweight existing-structure anchor evidence for UI prefill via `AnchorEvidenceResponse` on chapter/section nodes, without raw text, full page text, OCR boxes, or heavy layout payload. **[Code-Confirmed]**
4. Existing-structure anchor evidence should be optional per chapter/section and should support at least:
   - anchor type (`page_range` or `char_range`)
   - page indices or display page labels when validated
   - character offsets when no reliable page evidence exists
   - source/evidence availability state and warning code
5. The schema boundary must preserve hierarchy-first response shape: anchor evidence is metadata on `chapters[]` / `chapters[].sections[]`, not a second hierarchy source. **[From HLD] + [Future Direction]**
6. Any future `page_range` request/response schema must remain source-agnostic; PDFs are the first expected pageable source, not an OCR-only special case. **[Maintainer-Provided] + [Future Direction]**
7. `AnchorEvidenceResponse` currently supports `anchor_type`, `status`, `reason`, char offsets, page indices, and page labels as optional lightweight metadata. **[Code-Confirmed]**

## 19. Batched Task-Unit Content API

> 本節記錄 frontend content fan-out optimization 的 schema boundary。Batch schema 已落地為 backward-compatible additive API evolution；single task-unit content schema 仍保留。 **[Code-Confirmed]**

1. Batched content request allows the UI to request one selected section's ordered `task_unit_ids` through a single on-demand content API call instead of issuing one request per task unit. **[Code-Confirmed]**
2. `BatchTaskUnitContentRequest` keeps explicit render options `segmented` and `include_raw_content`, matching the existing single task-unit content endpoint semantics. **[Code-Confirmed]**
3. `BatchTaskUnitContentResponse.contents[]` returns ordered per-task-unit content responses that preserve the existing `TaskUnitContentResponse` shape rather than inventing a second content-block contract. **[Code-Confirmed]**
4. Request validation rejects empty task-unit id lists and duplicate ids at schema boundary; unknown ids and hierarchy-incompatible legacy documents fail fast through the same coordinator lookup semantics as the single endpoint. **[Code-Confirmed]**
5. Initial batch semantics are all-or-nothing; there is no partial-success item schema. **[Code-Confirmed]**
6. The existing single task-unit content endpoint remains the compatibility and fallback path. **[Code-Confirmed] + [Future Direction]**
7. Batch content is still on-demand rich content, not task-layout payload; `/documents/task-layout` must remain lightweight and must not return task-unit content or `content_blocks`. **[Code-Confirmed] + [From HLD]**
8. Batch schema evolution must not introduce persistence mutation, profile diagnostics write-back, parser authority, artifact write semantics, root `sections[]` flow, or `structure_nodes` as main flow. **[From HLD]**

## 20. Reading Target Interaction API Schema Exposure Audit

> 本節記錄 reading interaction API schema planning 與目前 schema exposure audit。Shared target/envelope schemas and artifact-aware metadata exist, while payload-specific schema families for generic reading-interaction endpoints are not yet complete. **[Code-Confirmed] + [Future Direction]**

### 20.1 Current Schema Exposure

1. `api_schemas.py` currently exposes `ArtifactAwareInteractionMetadataResponse` as metadata-only provenance for future artifact-aware interaction responses. It intentionally does not embed lower-level artifact payloads. **[Code-Confirmed]**
2. `api_schemas.py` now defines `ReadingInteractionTargetRequest` for generic `document|chapter|section|task_unit` target identity. It trims ids, normalizes `book` to `document`, rejects title-primary locator fields via `extra="forbid"`, and validates required/forbidden id combinations per target level. **[Code-Confirmed]**
3. `api_schemas.py` now defines `ReadingInteractionTargetResponse` and `ReadingInteractionResponseEnvelope` as shared response metadata for future reading-interaction route payloads. The envelope validates interaction type, status, artifact/session ids, provenance fields, reason-required failure statuses, critical-thinking-only statuses, and artifact-aware metadata without embedding UI behavior state. **[Code-Confirmed]**
4. `api_schemas.py` now defines public payload-specific request/response schemas for `analysis` artifacts, target-agnostic `quiz` artifacts, and `critical_thinking_session` lifecycle states, including read/generate/submit/retry requests, completed payloads, missing state, insufficient-content, and failure-state response validation. **[Code-Confirmed]**
5. Existing `SectionQuizResponse` and `ChapterQuizResponse` belong to the legacy section/chapter quiz endpoints and should not be treated as the full schema surface for the new target-agnostic quiz interaction API. **[Code-Confirmed]**

### 20.2 Future Schema Boundary

1. Public schemas should represent a generic reading target with `target_type=document|chapter|section|task_unit` and the required id for that level: `doc_name`, `chapter_id`, `section_id`, or `task_unit_id`. **[Maintainer-Provided] + [Future Direction]**
2. Optional parent ids may be included only for consistency validation and must not become title-based fallback. **[Maintainer-Provided] + [Future Direction]**
3. Read schemas and generate schemas must be separate. Read responses return persisted artifact/session state or a missing/not-generated status without implying generation. **[Maintainer-Provided] + [Future Direction]**
4. Analysis schemas should expose validated structured fields for summary, reasoning/interpretation, parsing/explanation, status, target metadata, artifact metadata, and validation/generation failure information when applicable. **[Maintainer-Provided] + [Future Direction]**
5. Quiz schemas should expose `short_answer`, `multiple_choice`, and `true_false` item types only, include answer payloads, enforce item count limits, and support insufficient-content status. **[Maintainer-Provided] + [Future Direction]**
6. Critical-thinking schemas should cover session generation, answer submission, and evaluation retry with statuses `question_generated`, `insufficient_content`, `answer_submitted`, `evaluation_failed`, and `completed`. **[Maintainer-Provided] + [Future Direction]**
7. All successful generated-artifact response schemas should carry schema version, prompt instruction version, context mode, source structure version, source hash or equivalent provenance where available, and target metadata. **[Maintainer-Provided] + [Future Direction]**
8. Validation should reject malformed target ids, unsupported quiz types, excessive quiz count, missing critical-thinking answer where required, and invalid model-output shapes. **[Maintainer-Provided] + [Future Direction]**
9. Interaction schemas must not expand `/documents/task-layout` with heavy artifact payloads and must not expose raw text by default. **[From HLD] + [Future Direction]**

### 20.3 Frontend-Required Reading Interaction Data Shape

> 本節是 frontend UI 對接所需的 public DTO planning。Shared target/envelope schemas are implemented; payload-specific insight/quiz/critical-thinking schemas remain future work. **[Code-Confirmed] + [Future Direction]**

1. `ReadingInteractionTargetRequest` carries a hierarchy identity object rather than loose route/body fields. Minimal fields are `doc_name`, `target_type`, and the one required id for the selected level (`chapter_id`, `section_id`, or `task_unit_id` where applicable). **[Code-Confirmed]**
2. Target titles are not accepted in request schema as locators. Response-side display title metadata is allowed through `ReadingInteractionTargetResponse`, but it is not target authority. **[Code-Confirmed] + [From HLD]**
3. `ReadingInteractionResponseEnvelope` includes `target`, `interaction_type`, `status`, optional `artifact_id` or `session_id`, `generated_at`, `updated_at`, optional `schema_version`, optional `prompt_instruction_version`, optional `source_structure_version`, optional `source_hash`, `reason`, and optional `artifact_context_metadata`. **[Code-Confirmed]**
4. Common read statuses include `not_generated`, `completed`, `insufficient_content`, `stale_target`, `generation_failed`, and `validation_failed`. Critical-thinking session responses additionally support `question_generated`, `answer_submitted`, and `evaluation_failed`. **[Code-Confirmed]**
5. Insight/analysis response is represented by `AnalysisInteractionResponse`, which composes `ReadingInteractionResponseEnvelope` with compact `AnalysisArtifactPayloadResponse` fields: `summary`, `reasoning`, `interpretation`, optional `explanation`, and optional `key_points`. Completed analysis responses require payload; missing, insufficient-content, stale-target, generation-failed, and validation-failed responses must not carry payload. **[Code-Confirmed]**
6. Quiz response is represented by `QuizInteractionResponse`, which composes `ReadingInteractionResponseEnvelope` with bounded drawer-oriented `QuizArtifactPayloadResponse.items[]`. Each `QuizArtifactItemResponse` carries `item_id`, `item_type`, `prompt`, optional `options[]`, `answer`, and optional `explanation`; allowed item types are `short_answer`, `multiple_choice`, and `true_false`. Completed quiz responses require payload; missing, insufficient-content, stale-target, generation-failed, and validation-failed responses must not carry payload. **[Code-Confirmed]**
7. Critical-thinking response is represented by `CriticalThinkingSessionResponse`, which composes `ReadingInteractionResponseEnvelope` with `CriticalThinkingSessionPayloadResponse`. It validates `question_generated`, `answer_submitted`, `evaluation_failed`, and `completed` lifecycle shapes; failed evaluation preserves `question` and `submitted_answer` and marks retry eligibility; completed sessions require evaluation feedback and may carry numeric score, suggested refinement, strengths, and improvements. **[Code-Confirmed]**
8. Frontend-local UI state such as drawer open state, selected menu item, pending textarea text, optimistic answer draft, or inline expanded/collapsed state must not be represented as backend truth schemas. **[From HLD] + [Maintainer-Provided]**

### 20.4 Artifact-Aware Interaction Metadata Schemas

1. Generated artifact response schemas should optionally expose artifact-aware context metadata: `artifact_context_mode`, `referenced_artifact_ids`, referenced artifact target levels/types, coverage counts, and deduplication/abstraction hint flags. **[Maintainer-Provided] + [Future Direction]**
2. Metadata should distinguish primary source evidence from secondary lower-level artifact references. **[Maintainer-Provided] + [Future Direction]**
3. Missing lower-level artifacts should serialize as empty/omitted secondary context metadata, not as an error. **[Maintainer-Provided] + [Future Direction]**
4. API schemas should not expose full lower-level artifact payloads inside higher-level artifact responses by default. **[Maintainer-Provided] + [Future Direction]**
5. Referenced artifact ids are observability/provenance metadata and must not become client-side hierarchy authority. **[From HLD] + [Future Direction]**

### 20.5 Frontend Exposure Completion Plan: Schema-Owned Tasks

> 本節把 frontend exposure 12-task plan 中由 `api_schemas.py` 擁有的任務固定為 implementation backlog。Shared target/envelope schema has an initial implementation, but the task remains open until it is verified against the first public REST route integration and payload-specific schemas. **[Code-Confirmed] + [Future Direction]**

| Task ID | Task | Scope | Completion Evidence |
|---|---|---|---|
| FE-INT-01 | Freeze shared reading interaction contract for route integration | Confirm `ReadingInteractionTargetRequest`, `ReadingInteractionTargetResponse`, `ReadingInteractionResponseEnvelope`, status vocabulary, and artifact metadata work as the single public schema foundation for all three route families. | Route-level request/response tests for insight, quiz, and critical-thinking all use the shared schemas without adding route-local target/status variants. |
| FE-INT-02 | Define insight/analysis payload schema | Add payload-specific request/response schemas for compact inline insight/analysis rendering. | Implemented by `AnalysisInteractionReadRequest`, `AnalysisInteractionGenerateRequest`, `AnalysisInteractionRefreshRequest`, `AnalysisArtifactPayloadResponse`, and `AnalysisInteractionResponse`; schema tests cover `summary`, `reasoning`, `interpretation`, optional `key_points`, shared envelope composition, missing state, insufficient content, and generation/validation failure. |
| FE-INT-03 | Define quiz artifact payload schema | Add drawer-oriented quiz request/response schemas with strict item type/count validation. | Implemented by `QuizInteractionReadRequest`, `QuizInteractionGenerateRequest`, `QuizInteractionRefreshRequest`, `QuizArtifactItemResponse`, `QuizArtifactPayloadResponse`, and `QuizInteractionResponse`; schema tests cover `short_answer`, `multiple_choice`, `true_false`, answers/explanations, max count, insufficient content, and invalid item shapes. |
| FE-INT-04 | Define critical-thinking session payload schema | Add session-oriented request/response schemas for read, question generation, answer submission, failed evaluation, completed evaluation, and retry eligibility. | Implemented by `CriticalThinkingSessionReadRequest`, `CriticalThinkingQuestionGenerateRequest`, `CriticalThinkingAnswerSubmitRequest`, `CriticalThinkingEvaluationRetryRequest`, `CriticalThinkingEvaluationResponse`, `CriticalThinkingSessionPayloadResponse`, and `CriticalThinkingSessionResponse`; schema tests cover `session_id`, question, submitted answer, evaluation feedback/score/refinement, status transitions, answer preservation on evaluation failure, and invalid missing-answer payloads. |

Schema-owned tasks must not expand `/documents/task-layout`, expose raw target content by default, accept frontend UI state as backend truth, or use title fields as target locators. **[From HLD] + [Maintainer-Provided]**
