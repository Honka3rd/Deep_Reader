# Deep_Reader_UI Detailed Design

## Source invalidation during interaction generation — R4 (2026-11-11)

Implemented correction for review b2517c2: distinguish ordinary surface close from a
hard interaction reset. Close/reopen retains the current generation task (R1).
`resetInteractions()` establishes a new source-invalidation epoch, including source
save/Undo, repair/reload and uncertain layout resets. Invalidate all tracked work
conservatively: block edits can affect section/chapter/document interactions whose
requests carry no source hash. New opens read current persisted state before any
new generation; they never attach to a previous epoch's promise.

Both the controller and Summary drawer capture the epoch before asynchronous work
and guard subsequent generation, success, failure and completion UI writes. Abort
obsolete HTTP to release resources, but treat the epoch check as the correctness
boundary even when transport ignores cancellation. Old task cleanup must not remove
new tasks with the same target key. Content refresh notifications that report body,
generated artifacts or summaries invalidated all cross this boundary. Note drafts
keep their existing lifecycle. No backend persistence/API change or billing claim.

Validation: `scripts/test-interaction-source-reset.mjs` failed before the fix for
all four kinds at 1280/390 and four actual controller/drawer probes. After the fix,
16 cases pass: eight App editor-save/reopen journeys and eight controller/task cases,
including transport that ignores abort, Summary kept mounted, delayed read/failure,
and old cleanup with a new task under the same key. Each App journey checks R1
close/reopen deduplication both before and after source invalidation. The same suite
passes with an immutable English book excerpt; original backup hashes are unchanged.

`npm run check`, the full four-kind activation suite at both widths, 21 learning
controller groups, reading-settings/display/lifecycle at 1280/390 and four R2/R3
production API tests across file/real PostgreSQL pass. The temporary test database
was removed. Desktop/mobile screenshots inspected. Evidence: `/tmp/audit-b2517c2/`.
Only UI source and owned documentation changed; no backend/schema or R2/R3 production
changes. Existing bundle-size warning remains. The 448-test previous revision result
is historical, not a new full-backend run.

Limits: this epoch is a local invalidation boundary, not a backend source revision.
It depends on observed source/layout reset notifications; unobserved remote edits
and cross-tab/process deduplication are outside this guarantee. Hard reset abandons
old task tracking; ordinary surface close does not. No paid model, production
mutation, Docker deployment or push. External re-review of the new SHA is required.

## External review corrections for e7073bb (2026-11-10)

R1 preserves dispatched interaction work across surface teardown with identity-scoped
session ownership and guarded observers. Reads may be cancelled before generation;
reopening pending work attaches to the existing generation. Explicit retry reconciles
saved state first after failure; no automatic paid retry.

R2 distinguishes old operation settings from the freshly verified page settings.
A settings-stale operation on the current source and current slot exposes Retry;
retry first rereads current page/source/settings and operation identity. Explicit
reconcile fences a still-active stale operation before a replacement command is
submitted. A second operation read must confirm terminal durable state and continued
slot/source ownership. New retry uses current revisions and the old operation handle;
uncertain cost requires the existing confirmation dialog. Source mismatch, missing
operation and replaced slots never authorize generation. No retry on mode return.

R3 keeps deleted preferences unavailable on read and shows an explicit reapply hint
for a canonical reimport. Save and apply uses the existing backend CAS command,
which clears the tombstone and advances revision even for unchanged Normal settings.
Language learning still requires registry-valid language/level and different native
language; unknown blockers remain blocking. No hidden read repair or new import API.

Validation: all three red regressions failed before production changes (R1 produced
four HTTP requests instead of two; R2 had no stale retry; R3 blocked explicit apply).
Summary independently reproduced R1 before sharing the same task owner. Green:
21 controller groups, interaction activation/settings/display/lifecycle browser suites
at 1280/390, and actual production preference/learning API mode-switch browser tests
with temporary file storage and an offline provider. `npm run check` passes with the
existing bundle-size warning. Backend regression: 448 file/real PostgreSQL tests,
including explicit stale-worker fencing, uncertain-cost consent, pure reimport reads,
CAS reapply and old-source rejection; disposable test DB removed. Two hash-verified
English/Chinese excerpts reuse aligned results with one offline call each; backups
unchanged. Desktop/mobile layouts inspected. Evidence: `/tmp/audit-e7073/`.

Limits: no paid provider/semantic evaluation, production data mutation or full PDF
upload/reimport browser journey. Interaction deduplication ends at App unmount and
is not cross-tab/backend-wide. External APPROVED is required before Docker or push;
this correction does not claim deployment acceptance.

## Reader reading-mode control (2026-11-10)

Implemented after maintainer-authorized documentation governance. A compact button
beside the reading-content heading combines current mode and the switching entry:
`Mode: Normal reading` / `Mode: Language learning`, with a full Reading mode accessible
name. Use the existing
authoritative reader preference observation; loading, failed reads and unavailable
saved modes remain explicit, never guessed as normal. The control is present in
ordinary and fullscreen reading and wraps within narrow screens.

Click opens the existing mode list/detail settings dialog directly over the reader,
without navigating to Library. Candidate selection is still a preview; Save and
apply validates and persists through the existing preference service/CAS contract.
After a confirmed book save, close the dialog, reread the applied mode and resume
the same section/page. First-time learning setup, workspace language, registry
levels, unsupported/ineligible modes and unknown-save recovery reuse existing
settings behavior. Back to reader/Escape preserves dirty-draft confirmation and
restores focus to the mode button. No optimistic mode activation or implicit save.

The reader owns this settings session by document identity. Opening it suspends
new current-page learning work; accepted backend work is not claimed cancelled.
Close/reopen, Library/other reader dialogs, book changes and unmount abort obsolete
observations and guard late saves. A fresh authoritative mode read precedes resumed
learning requests. Keep the selected section/page and source container mounted;
mode-specific content can change its height. Opening/closing settings in ordinary
reading preserves the reading scroll position. Merely
opening settings does not generate translations; compatible saved results follow
the existing shared-translation reuse contract. Backend hierarchy and task-layout
remain authoritative and no backend API/schema or preference lifetime is changed.

Acceptance: type/build and actual-App browser tests at desktop/mobile widths cover
normal/learning labels, mode roundtrip without Library, initial setup, preview and
dirty cancel, save/read failure and unknown recovery, duplicate save, stale responses
on owner change, keyboard/focus, fullscreen, long labels and overflow. Check relevant
Library settings and learning display/lifecycle regressions; report mock versus
real API evidence separately.

Implementation: `ReaderModeControl` renders the shared settings feature in the
reader header. `ReaderModeBoundary` reuses its current preference observation and
suspends page work while the dialog is open. The shared settings controller accepts
a document-scoped owner and optional close-on-confirmed-book-save behavior; Library
retains its original behavior. Request identity includes owner and target, so late
responses cannot update a different book. Reader notifications defer to the settings
dialog while it owns focus; source-edit/Note departure guards run before opening.

Validation: `npm run check` passes (existing bundle-size warning). New
`scripts/test-reader-mode-control.mjs` runs the actual App with production
preference/learning routes, temporary file persistence and an offline provider at
1280/390. Mode roundtrip, draft/keyboard/focus, duplicate save, known rejection,
unknown-save readback, failed/late reads, delayed save after owner replacement,
page/source-container retention, fullscreen and long-label wrapping pass. Across
both widths the fixture makes two provider calls total (translation plus vocabulary);
later observations/switches reuse results. Existing reading-settings, Library
reading-mode labels, learning display and learning lifecycle browser suites pass
at both widths. Desktop/mobile screenshots were inspected. A read-only smoke on
the local Madame Bovary API verifies the normal-mode label and settings open/close
without preference mutation or generation requests. Evidence: `/tmp/reader-mode-control/`;
no live-book mode save or paid-model acceptance is claimed.

## Shared translation presentation (2026-11-09)

Ordinary translation keeps its existing language dialog, saved-translation drawer,
session inline placement and highlight behavior. Its API now projects complete
sentence-aligned results as one aggregate paragraph; the frontend neither joins
sentence rows nor writes a duplicate result. Learning mode keeps sentence pairs
and independent vocabulary. Compatible saved translations are reused in both
directions; switching modes alone does not regenerate them. Legacy paragraph-only
records remain ordinary-readable but do not fabricate learning sentence alignment.
Acceptance uses existing display suites and test-shared-translation-api.mjs against
actual routes/temporary storage with an offline provider at 1280/390.

Mode-switch acceptance (2026-11-10): the user opened the existing English-to-Chinese
learning result in Madame Bovary, then switched to Normal reading and opened a saved
translation. Server logs show `learning/read` and `translate/read` returning 200,
with no translation-generation requests or batch events during the operation
window. Separate API/SQL checks verified the reference paragraph's aggregate equals
its two aligned translations and retains the same operation/revision. The ordinary
read took 666.2 ms in this run; this is not a response-time guarantee.

Reproducible steps, request IDs and evidence limits belong to the backend
[manual acceptance](../Deep_Reflective_Reader/section_tasks/module-detailed-design.md#manual-reuse-acceptance-2026-11-10).
The logs omit request bodies and cannot identify the clicked paragraph by themselves.
This user run covers learning-to-normal reuse; both directions remain covered by
the earlier automated test. Vocabulary generation/failure is an independent
component and is not evidence of translation regeneration.

The existing target-language drawer grouping is presentation state. Backend exact
language-pair/source/version keys determine generation reuse; the frontend must not
treat a displayed section as authority to merge backend keys. Returning to ordinary
mode restores the existing session display/highlight behavior, not another persisted
translation. Hierarchy/content and saved results remain backend-owned projections.


## Library reading-mode label freshness (2026-11-09)

Implemented under maintainer authorization: every Library row has a mode-status
surface on first open, without requiring a prior settings visit. Visible rows read
the existing authoritative book-preferences endpoint with at most two concurrent
requests. Offscreen rows wait until visible; Library rendering never generates
learning content or writes preferences. Unknown/loading/read failure is explicit,
never guessed as Normal reading. A failed read offers a deliberate retry.

Row observations are owned by the Library lifecycle. Closing, refreshing, leaving
for settings, changing document identity or unmounting aborts obsolete reads and
suppresses late results. Reopening and returning from settings reread visible rows,
so workspace eligibility changes and newly applied modes cannot leave stale labels.
This supersedes the old settings-controller-only last-checked summary cache; no
browser storage or backend list/schema change is introduced. Filtering a row out
also discards its observation; returning to the visible list reads it again.

Validation: `npm run check`; `test-library-reading-modes.mjs` at 1280/390 covers
first open, scrolling, close/reopen, Refresh, late results, identity replacement,
filter removal/return, failed-read retry, unsupported modes and malformed identity.
The 200-book fixture initially reads only 7 desktop / 3 mobile visible rows, with
at most two concurrent reads. Reading-settings, Library identity and readiness
regressions pass; settings return includes changed workspace eligibility. Desktop
and mobile screenshots inspected. Actual local API smoke displays mode labels via
read-only requests; no model generation or preference writes were performed.

## Persistent current-page learning progress (2026-11-09)

Implemented under maintainer authorization: keep an overall progress bar and completed
translation/vocabulary task count outside the collapsible paragraph details.
Counts represent confirmed finished components, not estimated model time or token
progress; empty saved hints and no-content components count as complete. Refreshing
an old saved result does not imply the current attempt completed. Initial saved-data
reads remain visibly indeterminate; failures and unknown outcomes stay distinct.

The same derived state controls forward pagination (Next and Last page) and its
visible reason. Block queued/executing work, pending completion after a wait timeout,
and unconfirmed submissions/observation failures. Confirmed terminal failures do
not require successful regeneration before leaving. Explicit Check progress uses
existing recovery; no timer polling, automatic retry, backend cancellation or
backend concurrency guarantee is introduced. Previous/First, Library and hierarchy
navigation retain existing behavior. Mode verification temporarily guards forward
navigation so a fast click cannot bypass the current-page initialization.

State belongs to learning-reader; ReaderModeBoundary passes the projection to the
reader pagination controls without duplicated state or App-level synchronization.
Source/settings identity and abort guards remain authoritative. A local
`awaitingCompletion` flag records that an observation is unresolved; it never
invents a durable operation state. A validated terminal observation releases the
forward guard even if the subsequent saved-result read fails. Confirmed terminal
unknown is shown as needing attention, while a lost submission receipt remains
blocked pending explicit recovery. Known rejected submissions also release it.

The progress panel reserves summary/hint height so active-to-failure updates do
not move the reader's scroll position. Native disabled Next/Last controls also
check the guard in their handlers. No page state is copied into App or persisted.

Validation: `npm run check` passes (existing bundle-size warning). The controller
suite passes 19 groups, including timeout, lost observation, refresh and terminal
failure rules. `test-learning-reader.mjs` passes at 1280/390 with held mode check
and completion, collapsed progress, keyboard toggling, 15/16 then 16/16 completion,
disabled Next/Last with no off-page commands, page/source/book transitions and
receipt recovery. Display/lifecycle suites pass at both widths, including scroll,
focus and retained saved results; collapsed desktop/mobile screenshots inspected.
Actual production learning routes with temporary file persistence and offline
provider pass Reader/reload checks at both widths (four calls total, no repeat on
reload). No live model calls or backend deployment changes are part of this UI fix.

## Vocabulary filtering display and completion waiting (2026-11-09)

The typed LearningService consumes the implemented optional assessment on saved
and partial hint responses, validating count coherence without judging candidates
in the browser. Saved vocabulary shows retained items and a compact expandable
selection summary with human-readable reasons. Validation omissions are explicit;
zero retained items with omissions do not display “No vocabulary hints needed”.
Legacy results without assessments remain readable. Partial checkpoint hints stay
separate from committed results; the aggregate summary is shown for saved results.
All model-derived strings remain text, with no HTML interpretation.

Generation acceptance now leads to one `operations/wait` observation, followed by
one validated saved read on completion. The frontend has no periodic status timer
or automatic repeat wait. Pending timeout offers Check progress; an explicit check
reads/reconciles the operation and may wait once again. Leaving the page aborts the
observer and suppresses stale updates, preserving unresolved identity-scoped
command handles for recovery. It does not cancel accepted server work or infer a
rollback. Backend wait details are owned by app/main designs. No UI state becomes
backend truth, and no read/task-layout path acquires a hidden mutation.

Verification: controller suite passes 16 groups; reader, display and lifecycle
suites pass at 1280/390, including no repeated reads, filtered/all-invalid hints,
keyboard details, reload reuse, late responses and source/mode switching. Type/build
checks pass with the existing bundle warning. Actual API/temporary file/offline
provider Reader tests pass at both widths (four generation calls, reload adds none).
Desktop/mobile screenshots were inspected. Both backed-up real-book captured outputs
plus mixed control pass actual API/SQL/restart/Reader replay without live model calls;
see the section_tasks 2026-11-09 evidence. These checks do not certify model semantics.

## Language Learning Reading Mode Plan (2026-10-28)

**[Code-Confirmed: LLRN-17/18/19 implemented; LLRN-21 verified]** [Canonical design](../Deep_Reflective_Reader/section_tasks/module-detailed-design.md#language-learning-reading-mode-plan-2026-10-28)；[task dependency queue](../Deep_Reflective_Reader/section_tasks/module-checklist.md#language-learning-reading-mode-queue-2026-10-28)。LLRN-22 已由 scripts 完成系統驗收；依 2026-10-29 maintainer 指示，模型語義評分不屬此 gate。

本節設定 UI 已依 LLRN-16 實作的後端設定 APIs 交付；LLRN-18 已接入當頁生成與恢復，LLRN-19 已交付逐句呈現。後端設定、對齊資料及 operation APIs 的 authoritative 定義與交付由 backend owner 負責。UI 不定義後端 schema 或把本地 state 當真實資料。

Library 每書提供「閱讀模式」入口，開啟可擴充的模式清單與專屬設定區；每書一次只啟用一個模式。原文校正/能力屬語言學習設定，母語入口集中於 workspace 偏好。Options 由實際 backend registry/proficiency contract 提供，不硬編語言/分級，不以 localStorage 分散保存。原文等於母語時不提供學習該語言選項，設定不足時保留普通閱讀及明確原因。

已實作的 LLRN-18 feature model/controller/service 負責目前 reader 頁 read-before-generate、bounded concurrency、dedup、AbortSignal、source/settings/request-generation 防晚到。啟用模式後閱讀目前頁是使用者選定的明確生成流程；一般 content/task-layout reads 和 Library save 本身仍是無生成行為。Read failure、pending/unknown 不能當 missing；恢復使用 LLRN-16 已實作的 command/operation contract。App.tsx 保持組裝，view 不放 policy/API。

已實作的 LLRN-19 學習顯示：source sentence → mother-tongue translation 交替，段落詞語提示置於 Insight 前。普通 Translate/View translations 入口及普通 inline translations 在該模式隱藏；原有結果與 session language selection 保留，退出模式恢復。翻譯/提示有獨立 loading/failure/refresh，舊結果到新提交前保留，原文始終可讀。長句自然換行、RTL/語言標記、穩定 sentence keys、keyboard/focus/scroll 納入驗收。已接受 server 工作不可由 HTTP abort 推論取消；關頁停止排隊和 UI 更新，重開回查。

### LLRN-22 saved-result reuse correction (2026-10-28)

Live SQL/browser inspection found a completed operation's old preference revision
could turn a current reusable saved result into `stale` after changing levels or
native-language settings and returning. `learning-reader/session.ts` now trusts
the authoritative current `ready` + saved + `operation_state=succeeded` read when
there is no unresolved session command. Initial/reload/check preserves Saved;
explicit refresh sends `regenerate` with current source/settings revisions. It
does not poll that historical completed operation as an admission prerequisite.
Active, partial, failed, unknown operations and unresolved commands retain their
existing observation/reconciliation/consent paths. No UI-derived cache authority,
backend contract change or automatic retry is introduced.

`test-learning-controller.mjs` adds a settings-roundtrip regression (15 groups
total): no initial generation or historical operation poll, followed by one
explicit refresh using current revisions. Type/build and all 10 suites in
`accept-learning-reader.mjs` pass; private report `/tmp/llrn22-ui/report.json`
is copied into the scripts-owned validation bundle. The scripts-owned
[LLRN-22 acceptance](../Deep_Reflective_Reader/scripts/module-detailed-design.md#llrn-22-live-acceptance-and-open-quality-gate-2026-10-28)
checks Saved and exact rendered rows at 1280/390 using real learning HTTP/SQL and
captured live output replay after this fix; replay makes zero new provider calls.
Original live output quality remains rejected, and this UI correction does not
close the scripts-owned LLRN-22 checkbox or alter backend progress counts.

### Learning reader paragraph compatibility correction (2026-11-09)

The segmented content endpoint returns prose with `block_type: "paragraph"`.
The learning reader previously selected and rendered only `"content"` blocks,
silently excluding a real paragraph-only page even when the saved effective mode
was `language_learning`. Earlier browser fixtures used only `"content"` and did
not expose this contract mismatch.

`pageSources` now includes the current page's backend-provided blocks by block ID
and document/task-unit ownership, regardless of their presentation type label.
Missing, duplicate or foreign identities still reject the page; the existing
learning read validates server source/settings before any missing-result generation.
The sentence/hint renderer likewise uses the validated block result and exact
source text, rather than a `block_type` whitelist. Backend hierarchy remains the
authority; this change adds no API, persistence or source normalization behavior.

Evidence: the paragraph-only `test-learning-reader.mjs` failed before the fix and
passes afterward at 1280/390, including paired translations, vocabulary, ordinary
translation menu suppression, pagination, stale responses and command recovery.
`test-learning-controller.mjs` passes 16 groups, including mixed/omitted type labels
and invalid paragraph identities. Existing display/lifecycle suites and
`npm run check` pass; the existing large-bundle warning remains.
Desktop/mobile paragraph screenshots were inspected. A read-only browser smoke
against local Madame Bovary confirms 9 paragraph blocks, 12 sentence rows and HTTP
200 preference/learning reads at both widths. Generate requests were intercepted
before reaching the backend: this proves mode activation with real content, not
new live-model translation quality. No user preferences or source were changed.

### Extensible mode list and interaction ownership

**[Maintainer-Confirmed direction; settings/controller/display implemented in LLRN-17/18/19; UI acceptance verified in LLRN-21]** 模式以可擴充清單呈現，初期僅普通閱讀與語言學習。以下是依此方向收斂的 UI 設計；取代先前對話中的固定兩選項 RadioGroup 建議；settings list/detail 已實作，reader mode controller 組裝已由 LLRN-18 交付，LLRN-19 已交付逐句呈現。

- **入口**：Library 書籍列的「閱讀模式」按鈕，旁邊顯示目前已套用模式名稱。開啟有書名的閱讀模式 Dialog；桌面為左側模式清單、右側模式詳情/設定，窄螢幕為清單 → 專屬設定的單欄流程，提供返回清單。避免 Dialog 疊 Dialog。
- **清單列**：模式名稱、一句用途描述，以及「使用中／待設定／不可用」等狀態；不可只用顏色傳意。選取候選列與「使用中」標記分開。不可用模式可查看說明/修正入口，但不能套用；同語的語言學習不提供啟用選項。
- **套用**：選取、鍵盤移動或查看模式只改 local candidate，不寫偏好、不生成、不變目前閱讀模式。完成該模式必要設定後按「儲存並套用」；後端成功回應才更新使用中標記。失敗維持目前模式與可重試 draft，取消不修改已套用設定。普通模式不要求填寫語言學習欄位。
- **複雜設定**：共用 Dialog 負責書籍身份、清單、navigation、儲存狀態、焦點及 dirty-draft 保護；每個模式的 feature 擁有 typed 設定 model/controller/view、驗證展示及必要的內部分步流程。切換候選/返回/關閉前保護未儲存修改，只有目前面板 mount；不預先啟動所有模式的 effects 或資料請求。
- **閱讀互動**：共用 reader 保持 hierarchy/content/navigation/scroll 所有權，依已確認模式組裝該模式的呈現與 action 元件。語言學習 feature 自己管理逐句資料、詞語與刷新；避免把所有模式分支堆入 App 或單一巨型 controller。不要求未來模式全部使用同一張設定表單或 action layout。
- **擴充契約**：前端以穩定 mode ID 對應本地 typed settings/reader feature；可選項與 eligibility 依屆時實作的後端 mode catalog。後端不下發可執行 UI、任意 JSON 表單或 component 名稱。新增模式明確註冊對應 feature 與受驗證的 settings 契約，本期不建立動態插件平台。
- **不支援的模式**：舊客戶端不能呈現的 mode ID 顯示不可用提示，保留 server 設定，不自動覆寫為普通模式或嘗試生成。仍可讀基本原文，顯式選擇支援模式才保存切換。
- **生命週期**：candidate/draft 按 document + mode 隔離；已套用模式切換後卸載舊 controller、取消排隊、抑制晚到結果。已接受的 backend 工作仍按 operation contract 回查。列表需支援鍵盤、清晰焦點與目前詳情關聯；若每列有額外操作，採可存取 list/navigation 語義，不把複雜互動硬塞入 listbox option。

### LLRN-17 implemented settings UI (2026-10-28)

`ReadingPreferencesService` uses `RestClient` for the implemented options/workspace/book
read/apply routes, with AbortSignal, no-store and checked DTO identities/coherence.
Registry responses supply every language and proficiency scheme/version/level;
`Intl.DisplayNames` supplies labels only. No browser preference storage, parser/profile
write, learning generation, new API or backend schema is introduced.

`features/reading-settings/{model,controller,view,index}` owns the settings lifecycle.
A local typed `features.tsx` registry resolves mode ID/version to its panel and validation;
`language-learning/view.tsx` owns original-language/level fields. Only the selected panel
mounts. Unsupported modes/versions remain visible and unapplied until an explicit supported
selection is saved. Normal reading requires no learning fields; saved learning settings
are retained by the existing backend contract. Known editable eligibility reasons can be
resolved in the draft. A canonical document returned by the book settings API may
retain a deletion tombstone after same-name reimport: explicit Save and apply may
reactivate it using current book/workspace revisions. Unknown reasons stay blocking.

Library provides Workspace preferences and a Reading mode action per book. It replaces
its dialog with the settings surface, so no Library/settings modals are stacked. Desktop
uses a list/detail split; mobile uses list → detail with Back to modes. Candidate preview
and the server's applied/effective mode are separate. Same-language learning explains its
unavailability and offers no level/apply action. Library reads preferences for visible
rows through a queue capped at two concurrent reads; offscreen rows wait until visible.
Every row has an explicit loading, current-mode or retryable failure label. Returning
from settings rereads visible rows, including workspace eligibility changes. The old
last-checked summary cache is removed. A fresh settings entry independently rereads
authoritative state; Library observations never supply settings drafts.

Source override is a reading preference. The API exposes only the effective source: when
an existing override is cleared, the UI does not know the hidden profile language. It
instructs the reader to explicitly save Normal reading with Use saved profile, then reopen
Language learning using the read-back source and applicable scheme. A missing profile can
instead be corrected by explicitly choosing a registry language. No profile repair is hidden
inside a read or form action.

Save is explicit and guarded against duplicate submission. Each request has a controller,
generation and target boundary; close, target change and unmount abort obsolete requests
and suppress all late results, notifications and finally updates. In-flight/unknown saves
are not described as rolled back. Conflict/unknown outcomes block resubmission until
Read current settings succeeds; this read retains the draft for review and updates the
server revision, never automatically replaying a mutation or claiming a recovered receipt.
Known rejection retains draft and applied state. Book/workspace transitions, candidate
switching, mobile back and close protect dirty input in the same dialog; browser unload
also warns. Explicitly leaving an in-flight save requires rereading on reopening.

The dialog has a bounded scroll area, long-title scrolling, labeled inputs, list navigation,
contained keyboard focus and a return to the originating Library button. Transient messages
use the shared AppNotification inside the modal boundary and above its action bar, so
persistent failures cannot cover the recovery action. Only the active surface notifies.

Validation: `npm run check`; `scripts/test-reading-settings.mjs` at 1280/390 against the
actual StrictMode App and controlled APIs (catalog generated from the actual backend
registry, 19 languages), including pure model/DTO guards, dirty navigation, same-language,
source/level changes, unsupported mode preservation, malformed identity, duplicate click,
CAS/rejected/unknown outcomes, held read and held save across close/reopen, workspace
invalidation, keyboard/focus and no learning calls. A 200-row Library refresh took 253ms /
239ms in that historical environment with no preference-read fanout (the visible-row
label reads added on 2026-11-09 supersede that read policy); the existing table is not newly
virtualized and this is not a guarantee for arbitrarily large libraries. Screenshots at both
widths were inspected. Actual LLRN-16 routes/lifespan with temporary file repositories also
passed settings save, page reload/reuse and native-language invalidation at both widths;
provider access was guarded and never called. User databases and books were not changed.
Library identity/readiness suites passed; the identity fixture was updated to answer the
already-existing enhancement-status read. The older workflow suite stopped waiting for its
hierarchy fixture; it also retains obsolete pre-one-click interaction assumptions, so a
full legacy Library pass is not claimed. Vite reports the existing large-chunk warning.

The evidence above covers LLRN-17 settings only. LLRN-18 supplies the current-page
controller below, extended by LLRN-19 sentence-pair rendering. LLRN-20 backend and
LLRN-21 UI acceptance are recorded separately; LLRN-22 now covers system acceptance only.

### LLRN-18 implemented current-page controller (2026-10-28)

`features/learning-reader` owns the page model, request session, React controller and
compact status view. `LearningService` uses the implemented LLRN-16 routes through
RestClient, validates response identities, state coherence, code-point source coverage,
translation alignment and hint spans, and never retries a mutation automatically.
App supplies the current document, modal suspension/source refresh boundary and an explicit
in-memory session owner for unresolved commands. ReaderContentView supplies only the
currently displayed task-unit page's content blocks; hierarchy/pagination/content fetching
remain with their existing owners. No task-layout, source, profile or preference write is added.

A readonly book-preference check confirms the effective mode before mounting the registered
reader feature. Normal/unavailable/unsupported-version modes do not call learning APIs.
Library/settings, enhancement, cleanup and source editing suspend this controller. Returning
to the same page first discards its previous mode confirmation and rereads settings; candidate
preview/save alone never queues generation. Reading an already selected learning mode is the
explicitly authorized read-or-generate flow. LLRN-18 originally supplied progress above the
existing original text. LLRN-19 now adds paired text, hint rendering and ordinary translation
entry-point suppression; its display evidence is recorded separately below.

Read every current-page block in chunks of at most 50, validate all chunks against the
confirmed book/workspace revisions and displayed source before dispatching anything, then
queue each translation/hints component independently. Only a verified `missing` without
an unresolved session command may auto-generate. Saved results, including empty hints,
are reused; whitespace-only sources make no generation call. The session admits at most
two concurrent generation/monitoring workflows and deduplicates queued/running actions by
block + component. This bounds frontend workflows, not server jobs accepted before leaving.
No next-page or whole-book generation is scheduled.

Commands carry a new UUID, the server source token and expected book/workspace revisions;
the UI sends no raw source, guessed language, model or sentence boundaries. HTTP success or
an acceptance receipt is not completion: operation status and a fresh validated saved-result
read establish the committed revision. Translation and hints have separate states; failed
refreshes retain the prior saved result until a verified newer commit. Partial checkpoints
are held separately from saved output (at most the first 50 checkpoint batches are read for
this status surface); checkpoint completeness alone never implies a committed result.

One bounded completion wait replaces the former one-second/20-read polling loop.
Completion triggers one saved read; pending timeout pauses with Check progress.
Failed/unknown/stale states stop observation; read failures never become missing.
Check progress rereads current data, looks up uncertain commands, then reads the
operation and can explicitly reconcile a current claimed/running/partial operation
before one further wait. It never automatically replays a paid generate call.
Unknown command receipts that remain absent stay unknown. A receipt records original
acceptance; completion observation verifies present state. Source/settings/superseded-operation checks prevent
foreign or obsolete results from being attached to this page.

The controller exposes explicit refresh and retry for the LLRN-19 action view. Refresh of a
confirmed successful operation sends a new `regenerate` command. Retry rechecks the bound
failed/unknown operation, uses a new command ID and supplies the original handle; uncertain
outcomes require an explicit duplicate-cost acknowledgement before dispatch. LLRN-19 now exposes the corresponding per-component actions and a Cancel-first
confirmation for retries with uncertain cost; unknown commands without a verified operation
still offer only Check progress. No timeout or network failure loops into a new paid command.

On page/book/source/settings change, suspension or unmount, abort requests and completion observers
and stop the old queue; generation/identity checks also suppress late state updates. StrictMode
probe effects cannot dispatch duplicate work. Aborting HTTP does not cancel an accepted
backend operation. Unresolved command identities are retained only by the App reader session,
scoped by document, source token, component and book/workspace revisions, and cleared after
verified commit. No text/results or preferences are saved in browser storage; a full reload
uses the backend's saved/operation projection. The compact status list has stable block and
component identities, bounded height and keyboard-scroll access; original text stays readable.

Validation: `npm run check`; `scripts/test-learning-controller.mjs` passes 14 grouped cases
for DTO/source guards, read-before-generate, 50-block batching, two-workflow bound, dedup,
abort/late responses, empty/saved reuse, independent failures, bounded partial polling,
command recovery, duplicate-cost consent, explicit regeneration and changed bindings.
`scripts/test-learning-reader.mjs` passes the actual StrictMode App at 1280/390 with eight
blocks per page, held read/generate/preference replies, page/book/source/mode transitions,
Library suspension, aborted-command receipt lookup, keyboard recovery/scroll and no unseen
page generation. Desktop/mobile screenshots were inspected. `test-learning-reader-api.mjs`
starts the production learning routes/lifespan with disposable file repositories and an offline
fixed provider: both widths, two visible blocks, four commands/provider calls, reload/reuse
and no third-page generation pass. Fixture server drains and removes temporary storage.
Existing reading settings, reader identity/recovery and Library readiness suites pass.
Vite retains its large-chunk warning. No user database migration, live provider, semantic
quality, SQL browser acceptance or complete LLRN-20/21/22 acceptance is claimed by those controller tests.


### LLRN-19 implemented sentence-pair display (2026-10-28)

The stable ReaderModeBoundary/context supplies one active page controller to the existing
reader tree. `LearningBlockBody` renders the authoritative ordered sentence IDs as adjacent
source/translation rows; source whitespace gaps remain exact and receive no invented
translation. Each block renders once. Until a verified learning read is available, the
original text remains readable. No frontend sentence splitter, language classifier, model
call or new backend schema is introduced. Source sentences use the confirmed book language;
translations/meanings use the confirmed native language. The API does not expose per-sentence
language classification, so the UI does not invent it for foreign quotations. `dir="auto"`
and bidi isolation support mixed/RTL text; long unbroken strings wrap. Source and generated
content remain React text, including literal HTML-like input.

`LearningBlockDetails` renders translation controls, then vocabulary, then the existing
block Insight. Vocabulary includes source form, optional distinct lemma, word/phrase/idiom
kind, optional part of speech and native-language contextual meaning. A saved empty hints
result explicitly says no hints are needed at this level. Partial checkpoint rows/hints
are labeled Partial, and missing rows keep their source without fabricated output. A saved
result takes precedence over partial refresh checkpoints. Failed/pending/unknown refreshes
retain the saved rows and meanings until a verified new commit. Stable sentence and hint
occurrence keys preserve DOM identity across status/result changes.

A compact per-component status/action distinguishes queued, pending, saved, empty, partial,
failed, unknown, stale and read failure. Confirmed saved data offers Refresh; a failed
operation offers Retry; unknown operations offer a Cancel-first duplicate-cost confirmation
and a separate Check progress. The confirmation is bound to page/settings/component/operation
identity, is keyboard-contained and restores focus on close. Unknown commands without a
verified operation cannot reach a paid retry button. The session revalidates binding/state
before any command and deduplicates busy actions. Busy controls use aria-disabled and handler
guards so keyboard focus is not discarded by native button disabling; queued state is
published immediately. State changes retain the same action element and reader scroll
container. Navigation/source highlights mark source fragments only, never translated text;
removing a highlight does not replace translated rows. Detailed page progress is collapsed
behind a native keyboard-accessible disclosure with the existing bounded scroll region.

Only a confirmed normal mode exposes Translate/View translations and ordinary inline
translations. Pending/unavailable/suspended or learning modes do not expose those controls;
the ordinary saved-read observer receives an empty visible set, cancelling obsolete reads.
App-owned ordinary language selections and saved records are not deleted. Returning to
normal rereads the mode and restores the prior session display choices and existing focus
rules. The provider/reader tree stays mounted while confirmations/results arrive, rather
than replacing the whole reading subtree. Suspension clears prior learning snapshots so
same-page reentry cannot briefly display an unverified old result.

Evidence: `npm run check`; `test-learning-display.mjs` passes actual StrictMode App at
1280/390 with exact Unicode/gap preservation, alternating rows, vocabulary-before-Insight,
empty/partial/failed/unknown/read-error states, mixed/RTL/long content, escaped HTML, stable
source DOM and scroll during held refresh, keyboard action focus, duplicate-cost Cancel/
confirm, one new bound retry, no ordinary reads/duplicate rendering in learning mode, and
normal-mode session preference restoration. A mounted presentation probe verifies source-only
highlight/dismiss and stable translated nodes. Desktop/mobile full-page and paragraph
screenshots were inspected. The LLRN-18 controller (14 groups) and page lifecycle suites
pass; `test-learning-reader-api.mjs` additionally asserts actual backend sentence/translation/
vocabulary DOM at both widths with disposable file storage and a fixed offline provider:
only four commands/provider calls for two visible blocks, with reload reuse. Ordinary inline
translation/focus and session lifecycle suites pass, as does source-review navigation.
No live model, SQL browser, deployed migration or translation-quality acceptance is claimed;
LLRN-20 backend and LLRN-21 UI gates are separately verified; current LLRN-22 system evidence belongs to scripts.
Vite retains the existing large-bundle warning.

### LLRN-21 reader lifecycle and accessibility acceptance (2026-10-28)

`scripts/accept-learning-reader.mjs` provides a reproducible UI gate: type/build
plus ten controller/browser suites, stopping on a failed suite and recording
exit status, durations, source hashes and logs. The real StrictMode App is tested
at 1280 and 390 pixels. `test-learning-lifecycle.mjs` adds a continuous Library
mode-selection → rejected/retained draft → successful apply → page generation →
refresh → source edit → Undo → normal-mode journey with controlled HTTP. It
records request bodies/counts and Playwright traces with DOM snapshots/screenshots.

The new journey dispatches exactly seven distinct generation commands per width:
two initial components, one explicit translation refresh, two after edit and two
after Undo. The three source bindings remain distinct; no next-page generation
occurs. Duplicate save/refresh clicks cannot submit twice. Saving while Library
or settings is open does not generate. An empty vocabulary result is rendered
explicitly. Returning to normal removes paired output and restores ordinary
translation entry points; existing inline/session suites verify display preferences.

A third **test-only** mode is registered in the browser's existing local feature
registry, with a two-step panel. The test verifies keyboard list selection,
candidate versus applied state, desktop list/detail, mobile Back, dirty leave/
cancel, focus containment and panel disposal after confirmed departure. It is
preview-only: it is never saved, published in the catalog or added to the backend
selection contract. Supporting a real new mode still requires its typed contract.

This exposed and fixed a shell defect: the leave-confirmation branch unmounted
the selected feature, so Keep editing reset its local step. The active settings
panel now stays mounted inside `.reading-settings-panel` during confirmation,
with `hidden` and `display: none`; its controls are excluded from accessibility
navigation and Tab order. The confirmation remains in the same dialog. Cancel
retains the panel's state; confirmed close/mode change still disposes it. No
additional modes or background controllers are mounted.

Verification passed: `npm run check` and `test-learning-controller` (14 grouped
cases), `test-reading-settings`, `test-learning-reader`, `test-learning-display`,
`test-learning-lifecycle`, `test-learning-reader-api`, `test-translation-inline`,
`test-translation-session-lifecycle`, `test-reader-guards`, and
`test-source-review-navigation`. Existing cases cover held/reordered responses,
rapid page/book/mode changes, unmount/reopen and command recovery, read failure,
unknown outcomes/consent, two-workflow bound, unsupported modes, partial output,
mixed/RTL/HTML-like content, source-only highlights, focus and bounded scrolling.
Desktop/mobile screenshots were visually inspected for mode navigation, long-text
wrapping, paired text and scroll containment. Keyboard/DOM semantics are tested;
this is not a universal assistive-technology or WCAG conformance certification.

`test-learning-reader-api` additionally runs production learning routes/lifespan
with temporary file storage and a fixed offline provider. Two visible blocks use
four commands/calls across both widths, including reload reuse; the unseen third
page is not generated. Its fixture drains and removes temporary storage. Other
browser cases use controlled HTTP, including the edit/Undo journey; they do not
claim real SQL browser or paid-model acceptance. No user books, preferences,
profile, database migration or backend schema changes were made. The existing
Vite large-chunk warning remains. LLRN-22 semantic issues remain unresolved.

Reproduce from `Deep_Reader_UI` with Playwright/Chrome available:

```sh
UI_URL=http://127.0.0.1:5173 node scripts/accept-learning-reader.mjs
```

Optional `PLAYWRIGHT_MODULE` and `CHROMIUM_PATH` select local installations;
`LEARNING_ACCEPTANCE_DIR` selects evidence output (default `/tmp/llrn21`). Final
private evidence is copied to
`Deep_Reflective_Reader/data/test-books/learning-reader/llrn21-20261028/`:
`report.json`, per-suite logs, `requests-{1280,390}.json`,
`trace-{1280,390}.zip`, mode/reader and paired-display screenshots. Existing suites
are rerun acceptance evidence, not counted as new implementation or backend tasks.

## One-click Interaction Activation (2026-10-27)

Selecting Insights, Summary, Quiz or Critical thinking explicitly authorizes a
read-or-generate command for that target. Read saved output/session first. Only a
successful, identity-matched `not_generated` response with no payload starts one
generation request automatically. Show loading/generating continuously; display
existing artifacts and critical-thinking answers/evaluations without regeneration.
Failed/malformed/mismatched reads and stale/insufficient/validation states do not
count as missing. Failures require explicit retry; there is no retry loop. Retry
of an unknown generation outcome reads persisted state before generating again.
Explicit retry re-reads first, reuses recovered output, and may retry a known
failed generation; the existing explicit new-question action may replace a policy-
stale critical-thinking session. Neither exception applies to initial activation.
Refresh remains an explicit replacement command; Note remains read-only on open.

Deduplicate in-flight activation for the same action/target. Abort obsolete reads
and suppress subsequent generation and late UI results on close, target/document
change, reset or unmount. Once dispatched, Insights/Summary/Quiz/Critical thinking generation
is owned by the mounted App interaction controller, independently of the visible
observer. Within the same source epoch, close/reopen joins that pending promise without another read/generate;
completion never reopens hidden UI. Failed transport outcomes remain tracked until
explicit retry reads persisted state. Unmount releases tracking and aborts HTTP,
which does not prove server rollback; this is session-local deduplication, not a
backend exactly-once or cross-tab guarantee. React Strict Mode must not duplicate requests. Passive menu opening, navigation and
layout loading never generate. Read endpoints remain read-only; the feature
controller orchestrates the separate existing read/generate APIs. UI state never
becomes hierarchy or artifact truth. This supersedes older two-click descriptions.

Validation: npm run check passes; test-interaction-activation.mjs passes at 1280/390 with actual React Strict Mode and controlled APIs: four one-click flows, saved sessions/answers, 11 read-failure/mismatch states per action, no automatic retry, explicit retry reconciliation, close during read/generate, deduplication, target/document switching and Note draft protection. Summary stale-to-explicit-Refresh also passes a focused rerun at both widths. test-reader-guards.mjs passes 42 identity cases and lifecycle recovery; test-hierarchy-interaction-menus.mjs passes both widths with the real 451-unit layout and mocked saved interactions. Desktop/mobile screenshots inspected. No live model generation or document mutation was performed.

Additional regression limitation: test-reader-recovery.mjs completed its interaction
checks, then stopped at disabled Enhance. Its fixture does not handle the current
reparse-structure/status activity read. The full legacy suite is not claimed as
passing; enhancement behavior is outside this change.

## Irreversible Ignore All OCR Flags (2026-10-27)

**[Code-Confirmed, 2026-10-27]** Add Ignore all beside the Flagged areas heading, enabled only for a
verified OCR document with pending regions and no busy/refresh-failure state. It
covers every pending region in the document, including other pagination pages.
Before dispatch, show the total count, ask the user to carefully inspect OCR text
against original pages, and explicitly state that ignoring all cannot be undone.
Explain that this dismisses flags, not body text or original images. Cancel is the
initial focus. Respect dirty drafts; issue one discard_all_regions command with
source/revision and mutation ID. Preserve the confirmation on failure with visible
error and same-command retry; disable duplicates, abort on document change, and
ignore late results. Merge authoritative response without task-layout/body reload.
Keep the dialog, selected tab and cleanup history; pending count becomes zero.


Validation: IgnoreAllRegionsDialog and ReadingContentActions implement whole-document confirmation and same-command retry. npm run check and test-ignore-all-regions.mjs, test-source-review-navigation.mjs and test-cleanup-history.mjs pass at 1280/390. Browser coverage includes 25 regions across pages, Cancel-first focus, duplicate clicks, failed/lost-response retry, one logical commit, no body/layout reload, disabled empty action/Undo, unmount and late-response suppression; screenshots inspected.

## OCR Review Action Visibility (2026-10-27)

**[Code-Confirmed, 2026-10-27]** Show the document-level reading-content action toolbar only when the
reading-content response explicitly reports `ocr_review_available: true`.
Missing/false capability, loading, native PDF and non-PDF documents must not show
review or toolbar Undo buttons. An OCR PDF remains eligible after all flagged
areas are resolved. Auto mode alone is not proof that OCR ran. The backend derives
the capability from persisted source-bound extraction evidence; the UI does not
infer it from filenames, selected mode or review counts. Preserve ordinary body
Edit/Delete dialogs and their Undo/refresh handling. Document changes and stale
responses retain existing cancellation/identity protection; no extra requests.
After closing review, restore keyboard focus when the verified OCR entry returns
from the overview read, without taking focus from another active control.
Validation: build, 1280px/390px visibility, request-lifecycle, source-review navigation
and cleanup-history suites pass; desktop/mobile screenshots inspected.


## Extracted Text Review Terminology (2026-10-27)

**[Code-Confirmed, 2026-10-27]** This module compares extracted reading text with original
page evidence, supports correction/confirmation of eligible related blocks, and
exposes OCR cleanup history. The entry is **Review extracted text**, the dialog is
**Extracted text review**, and the source-region tab is **Flagged areas**.
These replace the ambiguous Review source / Source review / Source regions labels.
The existing **OCR cleanup history** label remains descriptive of its own content.

Flagged areas can include unresolved and ignored entries, and areas with no linked
body text. The label must not promise that every item is an error, a pending task,
an editable paragraph, or recoverable missing text. A short dialog description
explains comparing against the original page, correcting/confirming text, and
reviewing cleanup changes. Empty-state and refresh/confirmation copy use the same
terms. Preserve backend region/status/identity contracts, existing DOM IDs/classes,
tab order, requests, mutations, navigation, focus and scroll behavior.

This is presentation-only governance. Backend hierarchy remains authoritative;
task-layout remains lightweight, and opening/switching views introduces no write.
Internal source-review identifiers remain implementation names; this wording
change does not create a separate feature or alter backend schemas.

Validation: TypeScript/Vite check and existing source-review navigation, state and
cleanup-history browser suites pass. Desktop (1280px) and mobile (390px) screenshots
confirm that the title, purpose description and tabs fit. Keyboard tab switching,
draft retention, pagination, navigation, snapshot guards and cleanup history flows
remain covered by those suites.

## Persisted Block Translation Plan (2026-10-26)

**[Code-Confirmed: TRUI-07..11]** Integrated with backend TRUI-01..06.
[Shared design](../Deep_Reflective_Reader/section_tasks/module-detailed-design.md#persisted-block-translation-plan-2026-10-26)
and [task order](../Deep_Reflective_Reader/section_tasks/module-checklist.md#persisted-block-translation-queue-2026-10-26).
The persistent read/generate API now supplies this UI's saved-result contract.

Delivery uses features/block-translation/{model,controller,view,index} and typed
TranslationService; App only wires normal-block commands and server invalidation.
The feature controller owns generation after Modal submission. Aborted/unknown
commands keep only in-memory identity until a readonly reconciliation; known rejected
commands release that identity. No browser storage, polling or frontend timeout.
Main-reader edit now consumes bounded refresh responses rather than reloading layout;
identical text closes the editor with no invalidation/refresh. Source-review consent
wording also distinguishes mandatory summary/translation deletion from other metadata.

Historical initial delivery: build, mounted StrictMode/controller/DTO tests and existing refresh/pagination tests
pass. Actual App/current API/configured LLM/disposable PostgreSQL tests pass using
immutable Madame Bovary/Red Roulette excerpts at 1280/390, including two-language
reload, initial same-language regeneration (superseded by reuse below), no-op/change Keep, cancel,
duplicates and stale navigation. Private backend evidence:
data/test-books/persisted-translation/20261026T070158Z-06e6f6e0. Production Docker
was not rebuilt during that acceptance; Vite was running at http://127.0.0.1:5173.

Reuse correction (2026-10-26): Modal opening reads saved results from the backend.
Submitting an existing valid To opens/focuses that saved section with zero generate
requests when From is absent or differs from To; an equal selected pair is rejected
before saved-result reuse. Only a missing To generates with valid distinct languages.
This describes the existing UI shortcut, not current backend cache identity:
since shared sentence translation, explicit backend generation validates the exact
source/target key and source versions. Returned saved records retain their original
source language and mutation ID, which need not match the new request. DTO checks
still require target/source-hash and committed record/payload agreement. No force
regeneration API/UI is added. 80 focused backend tests, mounted controller/DTO tests,
build and real-book App/API/model/SQL at 1280/390 pass; evidence:
backend data/test-books/persisted-translation/20261026T113558Z-5d5c71b0.

### Interaction

Add Translate only to the normal content-block interactive menu, not parent-level
or source-review repair menus. Open a titled MUI Dialog/Modal with searchable
Autocomplete controls labelled Translate from/to. From uses the server's existing
profile language; missing/unknown leaves it empty, requiring selection. To is required
and empty on a fresh open; re-entry may prefill an unresolved command's language pair
for retry. No locale/profile default is invented for To.
Both inputs are selection-only, not freeSolo, searchable by language name or code.
Options come from backend registry codes, never an independent frontend allowlist.
Disable submit while identity/options/selections are missing or a submit is active.
To options exclude the selected From code. Changing From clears To if they become
equal; choosing To first follows the same rule. Clearing From restores all registry
options. The model, controller submit path and DTO service reject equal selected codes
before saved reuse or HTTP generation. Reconciliation clears an equal To from legacy
unresolved commands. API rejection `translation_same_language` is a definite failure
with a language-selection message, not an unknown outcome to retry unchanged.
View translations still reads existing records, including historical equal-language
records; this policy introduces no cleanup or deletion.

Submit closes the Modal immediately and transfers the operation to the feature
controller. Keep pending feedback in the shared notification pattern; open the
drawer only after confirmed persisted completed/partial success, label partial
output, and never replace saved results on failure/untranslatable. Cancel before
submission starts no model request. Include View translations for saved records
so reopening does not require generation; Add translation reopens the same-block
language Modal from the drawer. Submitting a saved target reuses its section only
after the distinct-language selection guard.

### Inline Translation Display (2026-10-26)

**[Code-Confirmed: initial delivery]** The saved-translation drawer has `Translations`
and `Reading display` tabs. Multiple saved languages render below the original body,
after the block's visible Insight when present. Labels, partial status, paragraph
breaks, direction and wrapping are preserved; task-unit pagination bounds rendered
blocks. Initial per-block/current-section selection is superseded by the session
contract below; it must not be described as the final preference lifetime.

#### Session preference contract (2026-10-26)

**[Code-Confirmed: UI-TRINLINE-02, following maintainer contract]** Reading display
selects target languages for the entire active application session. Session means
one mounted App/page lifetime: drawer close, block/page navigation, section changes
and book changes preserve the selected codes. A full page reload, closing the tab or
unmounting App ends the session and resets them. No localStorage, sessionStorage,
backend preference write or account/custom-tab binding is introduced in this phase.
Future longer-lived/custom-tab preferences are a separate requirement.

Language preference is separate from saved translation content. Use one feature-owned
language selection, not per-block checkboxes or a saved-text cache as preference
truth. A checked language applies to matching existing saved records as they are read
and to subsequent successfully persisted generation results, without another checkbox
click. Unchecking hides that language across blocks while leaving saved data intact.
Missing/deleted/invalidated records never clear the selected language preference;
new valid translations in that language automatically become visible again.

Reading display lists the current backend registry codes together with already
selected session codes, so a preference can be disabled even when this block has no
matching saved translation. It explains session-wide behavior and does not promise
to generate missing translations. Generation remains an explicit Translate action.

Saved content remains bound to document, block and source identity. Scope/source
changes drop obsolete snapshots and cancel pending reads, while retaining language
preferences. Fetch saved translations only for blocks on the currently rendered
reader page when at least one language is selected, using the existing read-only
translate/read API with at most two concurrent background reads. Do not fetch all
sections/books, change task-layout or generate translations on display. Cache verified
results for the current loaded scope, including empty reads, to avoid repeated reads
on ordinary rerenders. Drawer reads and successful generation reconcile that cache;
a late background read must not replace a newer explicit result or resurrect an
invalidated record.

Leaving a reader page or scope, removing/changing a block, disabling all selected
languages or unmounting cancels obsolete background reads. Guard responses against
request/source/scope changes even if cancellation is ignored. Read failures are
explicitly retryable and must not be silently interpreted as no saved translations;
retain a last verified snapshot when available. Server translation/cleared-block
invalidation removes affected snapshots even with the drawer closed, without clearing
session preferences. Restoring source text does not restore old snapshots; only a
fresh authoritative read or successful generation may supply content.

**[Code-Confirmed: UI-TRINLINE-02]** Display locations
are exclusive: a session-selected language is omitted from the drawer's Translations
panel and shown only inline; unchecking moves it back. Reading display keeps the
checkbox available regardless of where the text is shown. If every saved language
is inline, explain that location instead of claiming no translations exist.

Highlight/focus follows the target language's actual rendered destination. After
new generation or saved-target reuse, an inline-selected target closes the language
Modal and focuses/scrolls its inline section without opening an empty translations
drawer; other targets open/focus their saved drawer section. Changing a highlighted
language's checkbox transfers its highlight to the other location. Preserve highlight
through reload while the source-valid record exists, clear it when that record is
removed or its source/scope changes, and never steal keyboard focus behind an open
modal. Closing the drawer allows a pending inline destination to receive focus.
Only the current document/block/language destination may be highlighted.

The render order remains body -> optional visible block Insight -> matching saved
translations. A task-unit Insight above child blocks does not change that order.
Keep preference/cache/request ownership in the translation feature; App only composes
reader props. A small preference boundary should permit a later lifecycle owner
without implementing storage or a generic settings system now.

Acceptance gates: session selection across blocks/pages/sections/books; later matching
generation; unchecking globally; no matching record and partial output; failed-read
retry; bounded current-page requests; source invalidation, delayed/aborted reads and
same block IDs in different books; full reload resets preferences. Preserve existing
drawer focus/recovery behavior. Run build, mounted controller and actual-App browser
checks at 1280/390 and inspect screenshots. Backend/SQL/model acceptance is not implied
by UI fixture tests.

Delivery evidence: `npm run check`, `test-translation-controller.mjs`,
`test-translation-session-lifecycle.mjs`, `test-translation-inline.mjs`,
`test-translation-focus.mjs` and `test-translation-recovery.mjs` pass. Actual-App
controlled-HTTP checks at 1280/390 verify cross-block/section/book preference, same-ID
book isolation, global deselection, explicit future generation, reload reset, read
failure/retry, exclusive placement and inline/drawer focus transfer. Held-read tests
verify two-worker concurrency, explicit-result precedence, source/scope cancellation,
late-response suppression and invalidation retaining language choices. Desktop/mobile
settings and inline-highlight screenshots are inspected. This phase changes frontend
only; no preference storage, backend schema/API or live-book mutation is added.

### MVC And Lifecycle

Use a translation model/controller/view and typed RestClient service. App remains
a composition shell; do not grow quiz/thinking controllers with translation storage.
The feature controller outlives the submitted Modal. Modal-only reads cancel on
close; submitted generation has explicit feature ownership. Unmount, document/block
switch, source changes or supersession abort requests and invalidate generation IDs.
Check target/source identity before every state/notification/finally update. A delayed
response must not reopen an old drawer. HTTP abort does not prove rollback: re-entry
reconciles backend records before retry, without polling or browser persistence.
No fixed frontend timeout, hidden prepare/profile generation or full-layout reload.

### Failure Recovery (2026-10-26)

**[Code-Confirmed]** A failed or unconfirmed generation must not lock both language
selectors. `Try again` opens the language Modal and reads saved results first; it
does not generate automatically. A failed read exposes `Retry` and keeps inputs
disabled until a valid read succeeds. Once ready, the user may change either language,
retry the original pair, or open an already saved target without model work.

Unknown commands retain in-memory identity keyed by document, block, source language
and target language, with the effective-content hash. Retrying the same unresolved
pair reuses its `mutation_id`; changing a pair uses a separate command identity.
Selecting or saving another target must not discard the original pending command.
This command key is distinct from backend storage's one-record-per-block/target rule.
A successful read removes pending entries when their content hash changed or a valid
saved target is present. Confirmed save/reuse clears pending entries for that target;
explicit source invalidation clears the selected block's pending entries.
An empty saved read alone does not prove an earlier request stopped or rolled back.

The typed service distinguishes definite rejection from unknown outcome:

| Response | Retry identity policy |
| --- | --- |
| HTTP 4xx except 408 | Release the rejected command; retain normal conflict/reload handling |
| 502 with `translation_provider_failed` or `translation_invalid_output` | Release the command and offer retry |
| 503 with `translation_capacity_unavailable` or `translation_profile_unavailable` | Release the command and offer retry |
| Network failure/abort, 408, generic gateway 502/504, `translation_service_unavailable`, or unverifiable success payload | Retain identity until reconciliation; keep language choices available after a successful read |

Known rejection and unknown outcome both preserve saved translations. Provider timeout
is not evidence that a language pair is forbidden, and the UI recovery fix does not
claim to eliminate backend/provider timeouts. Closing or switching targets still
aborts owned requests and prevents late success/failure from reopening old surfaces.

### Drawer And Source Changes

Use a bounded, scrollable drawer keyed by selected block, with stable sections keyed
by target language code. Show language name/code, source code, timestamp, status and
plain translated text. The selected or newly generated target appears first, has
`aria-current` and a visible highlight, and receives keyboard focus; other sections
sort by language code. The scroll container returns to the selected section at the
top. There are no per-language collapse controls in the current implementation.

`Reload translations` reads saved results and retains the active language through
loading and failed-read retry. Once the read succeeds, retain its highlight only
when that language still has a valid saved record; otherwise clear it. A fresh
open, including close/reopen, resets the active language instead of restoring an old
highlight. An obsolete read cannot restore that old selection. Reload does not
generate or replace translations. Reuse/new generation preserves sibling languages.
Reopen/reload reads the backend, not localStorage/sessionStorage. Registry size bounds
section count. Render loading/empty/failure states explicitly; use one AppNotification
within the active modal/drawer surface.

Backend compares submitted text with stored effective body inside its write boundary.
Same-text success must not clear translations/summaries, flip edit flags or trigger
layout reload. Real changes delete every saved language for each changed block from
backend storage, including its replay receipts; this is not just hidden UI content.
Unaffected blocks retain their translations. Real changes also clear affected summaries;
Keep/Clear applies only to other interaction metadata. Update editor confirmation
wording to make this distinction explicit. Consume server changed/cleared scope to
close or invalidate affected views without dropping unrelated Notes/drafts. Block
deletion clears all its interaction state. Never publish in-flight obsolete results.
The backend owns dependency scope; UI must not guess which ancestor summaries survive.
`App.tsx` forwards `translation_block_ids` and cleared-block scope to the translation
controller even when `generated_invalidated` is false under Keep. The controller closes
an affected drawer/Modal, cancels its request and clears its pending commands. The
backend commit guard rejects results generated from an obsolete source. Undo does
not restore deleted translations; translating the new/restored body requires a new
explicit user action. See the [source lifecycle design](../Deep_Reflective_Reader/document_structure/module-detailed-design.md#selective-atomic-cleanup).

### Acceptance

Verify search/selection, registry rejection, profile defaults, one POST per submit,
Modal-close-before-completion, persisted-success drawer, multiple saved languages,
same-language reuse without generation, refresh/reopen, partial/untranslatable and commit errors.
Exercise delayed responses, rapid target changes, close/unmount, no-op/change/delete
policies, keyboard/focus and long text at 1280/390. Final App/API/SQL tests use isolated
copies of immutable Madame Bovary/Red Roulette backups. Mocked contracts are useful
regressions but not final persistence acceptance; no production books are mutated.

Correction evidence (2026-10-26, preceding implementation runs): `npm run check`,
`scripts/test-translation-controller.mjs`, `scripts/test-translation-focus.mjs` and
`scripts/test-translation-recovery.mjs` passed. Focus and recovery browser checks use
1280/390 widths and controlled service/HTTP outcomes. They cover reload highlight,
failed-read retry, removed language, keyboard/long-text scrolling, close/reopen and
stale reads; provider/gateway failure, editable From/To, separate command identities,
saved-target reuse and return to the original retry ID. These are frontend regressions,
not a new live-model/SQL acceptance run or a guarantee of provider availability.
Implementation: `src/features/block-translation/{model.ts,controller.ts,view.tsx}`,
`src/services/TranslationService.ts` and the refresh integration in `src/App.tsx`.

## Block-Level Source Review Correction (2026-10-26)

The maintainer corrected the operation boundary: every `.source-review-block`
owns one three-dot menu with Edit, Delete and Keep it. A region is a grouping and
source-image/navigation context, never the batch-edit target. The region-wide
editor described below is historical and removed from the current UI.

- Expand Related blocks lazily; render at most 20 blocks. Edit the complete current
  text of the selected block, not concatenated region text or a guessed substring.
- Consume per-block `review_repair` from the backend. A protected sibling never
  disables another block. Repeated review edits do not set body manual-edit flags.
- Edit uses explicit Keep/Clear interaction consent; Delete confirms removal of
  that block and its Notes/interactions only. Keep it confirms the text is correct,
  removes that block from Extracted text review across its linked regions, and warns that
  review editing is no longer available and this decision cannot be undone. Body
  text and interactions remain intact; the main reader's Edit remains available.
- Keep confirmation is persisted by the server, not browser storage. Original
  evidence is retained. No body insertion, OCR/model calls or automatic Prepare.
- Merge authoritative target DTOs/tombstones and kept membership into existing
  keyed rows. Preserve modal/tab/expansion/scroll. Refill a bounded page only when
  deletion/Keep leaves a page gap; edits do not re-fetch the review list or layout.
- SourceReviewBlockRepair owns block menu/editor presentation; ReadingContentActions
  owns command/retry lifecycle; useSourceReviewBlocks owns paged related reads and
  local merges. Definitive failures and unknown outcomes remain distinct.

SRBLOCK-02 is accepted against Madame_Bovary_261021_155934, Page 3, 20 canonical
related blocks, on 1280/390 viewports with real API and disposable SQL. Final report:
backend data/test-books/srrepair07/20261026T044959Z-7b1702d7. Repeat edits preserve
other blocks and row DOM; Keep/cancel/replayed response loss/reopen and single-block
Delete pass. Source and backups are unchanged; the disposable schema is removed.
Mounted editor, navigation/state and reader refresh regressions plus build pass.
Screenshots were checked on desktop/mobile. Current API is rebuilt/healthy; a live
read-only browser confirms block menus and Keep confirmation, without book writes.
Old region-wide acceptance does not establish this correction.

## Guarded Source Region Repair (2026-10-26)

Historical region-wide delivery. The block-level correction above supersedes its
UI action boundary; shared CAS, transaction, source and interaction safeguards remain.

Maintainer-approved successor to the read-only SRMAP UI. Backend authority
and shared execution order live in the
[domain design](../Deep_Reflective_Reader/document_structure/module-detailed-design.md#guarded-source-region-repair-2026-10-26)
and [checklist](../Deep_Reflective_Reader/document_structure/module-checklist.md#guarded-source-region-repair-2026-10-26).
SRREPAIR-05/06 consume the implemented optional per-region repair DTO, not legacy
block IDs/manual flags alone. Older servers stay read-only.

- Keep Flagged areas and cleanup history as separate horizontal tabs with stable
  region/record identity. Reuse existing editor/confirmation primitives, not a second
  body-mutation implementation. Model transforms DTOs; controller owns requests and
  drafts; view renders; typed services own transport.
- Distinguish Original evidence, saved repair draft and applied text. Save changes
  only the draft; Apply explicitly updates the body. Offer region Edit and Delete
  text through the established menu; Delete requires confirmation and only removes
  the server-mapped region text, not unrelated text in a shared block.
- Render server can-apply/can-delete and reason. Main-reader manual edit/delete
  disables region body mutation. Region/history draft saves and applications never
  toggle a frontend manual flag. Unknown/partial/stale mappings retain preview and
  valid navigation/Ignore without guessed write targets. Ignore is not text deletion.
- Edit asks Keep/Clear for interaction data including Note; complete block deletion
  clears that block's interactions. Use backend invalidation results, never assume
  retained generated data is still current. Preserve recovery history and explicit
  Ignore semantics; deleting all related text does not automatically Ignore.
- After commit, merge only authoritative affected block DTOs/tombstones and related
  region states. Preserve tab, row expansion, selection, focus, drafts and scroll;
  do not remount Extracted text review or fetch full task-layout. Navigation/highlights use
  verified current targets, never JavaScript slicing of canonical source offsets.
- Cancel obsolete requests and guard generation/document/source/revision on every
  response. Keep same-command unknown-outcome reconciliation distinct from read
  refresh retry after a confirmed commit. Protect dirty drafts; no sessionStorage,
  localStorage, fixed request timeout, implicit Prepare or library polling.
- Use centralized notifications, stable classes/labels/data identities and bounded
  lazy region/block lists. Validate 1280/390 viewports, keyboard/focus, rapid target
  changes and long lists; actual real-state acceptance belongs to SRREPAIR-07.

### Implemented Controls And Lifecycle

`SourceRegionRepair` renders the existing MUI menu/dialog primitives. The nested
editor has one scoped draft, distinct Save draft and Apply, a Keep/Clear interaction
confirmation, and explicit Delete region text confirmation. Original remains the
immutable preview. Delete preserves the region row and its saved draft for recovery.
Editable text counts Unicode code points with the existing 10,000-character limit.
Missing capabilities and protected mappings never become writable client-side.

`ReadingContentActions` owns revision-bound commands through ReadingContentService,
same-ID unknown-outcome retry and the existing committed-body refresh callback.
Definitive 4xx rejection releases the retry identity; uncertain transport outcomes
retain it until identical replay or an explicit authoritative reload. No browser
persistence, timeout, polling or implicit preparation. Single active error surface
is inside the current editor, not duplicated behind its modal focus boundary.

Region mutation responses update the existing keyed rows and related block list.
Draft Save performs no GET/body refresh. Apply/Delete/region Undo use the existing
20-block refresh paging, including tombstones, reappearing blocks and plural cleared
interaction targets. They never request task-layout or section content. Cleanup
history eligibility is invalidated after body changes and read lazily on tab entry;
that read merges only history, retaining Flagged areas DOM, expansion and scroll.
Failed related-block reads hide unverified text; failed navigation after an accepted
read retains verified review data and requires a fresh check on another navigation.

Flagged areas render at most 20 rows; related blocks retain their existing 20-item
pagination. Editor exit restores its connected region trigger with preventScroll
and restores the outer scroll offset. Cleanup cancels pending focus frames. Dialog
close/document change aborts requests; generation/source/revision checks reject late
publication. Editor unmount/target switch cannot close or dirty a replacement editor.

Supporting changes stay within typed service, review components/hooks and the existing
reader refresh model. App only forwards authoritative plural interaction invalidation.
No new body editor path, manual block insertion or frontend manual-edit truth.

### SRREPAIR Acceptance

UI-owned 05/06 and shared final acceptance 07 are complete. Actual App/API/disposable
SQL workflows pass at 1280/390 on both hash-verified dark-water and Madame Bovary OCR
backups. Private reports live under backend data/test-books/srrepair07/
20261026T040032Z-963f487a and 20261026T040631Z-ee8127e8. Assertions cover immutable
source, repeated application, Delete/recovery, same-ID response-loss replay, retained
row DOM/scroll/focus, lazy history refresh, no region-triggered layout/section reads
and authoritative protection after main-reader Edit. Test schemas are removed.

test-source-region-editor, test-source-review-navigation, test-source-review-state,
test-cleanup-history and test-reading-refresh browser suites pass; actual real-book
browser acceptance is the sixth suite. Screenshots are inspected on both viewports.
npm run check passes with the existing large-bundle warning. The final long-book
acceptance uses a fixed build to exclude development hot-reload interference.
No production writes or backend deployment; full governance/validation receipt is
in the linked document_structure design, without duplicating task ownership.

## Repeatable Cleanup Repair (2026-10-25)

RCR-04/05 follow backend RCR-01/03; RCR-02 was withdrawn before implementation.
RCR-04 local refresh and RCR-05 isolated real-state acceptance are complete. Canonical backend design:
[Repeatable Cleanup Repair](../Deep_Reflective_Reader/document_structure/module-detailed-design.md#repeatable-cleanup-repair-2026-10-25).

Save changes only the record draft. Restore explicitly applies it, including after
an earlier Restore: backend replaces its own previously applied fragment. Never
disable solely because status=restored, concatenate text or set a local edit flag.
The backend remains authority for can_restore and permanent body manually_edited.
Use concise state for a saved draft not yet applied; already restored is not a
permanent prohibition. Keep immutable quote distinct from draft and applied text.

After apply, fetch/merge only server-specified affected block DTOs in the existing
reader controller; preserve selection/focus and recalculate local pagination if
needed. Support cross-block and hidden-to-visible cases, authoritative ordering and
interaction invalidation without fetching task-layout or a whole section. Refresh
history status separately. Saved success followed by refresh failure keeps a targeted
read-retry handle, never a fresh Restore. Abort/generation/identity guards, draft
discard confirmation and same-ID unknown-outcome retry remain mandatory.

Backend RCR-03 supplies optional mutation `refresh` plus paged read-only
reading-content refresh queries. RCR-04 consumes the nested payload, not legacy
top-level blocks. Bind every page to the mutation/source/revision, merge in canonical
order using the server's next-unaffected anchor, and apply tombstones or reappearing
blocks. Update cached sibling owner/hash metadata using `unit_source_hash`; absence
of lazy interaction availability means unknown, not not_generated. Apply server
generated-invalidation/purged-block flags without deleting unaffected Notes.
`scope_current=false` acknowledges an older commit but cannot update current text;
409 or unavailable refresh needs explicit reconciliation, never a guessed patch.
Frontend wiring and controlled-transport browser checks are complete under RCR-04;
RCR-05 additionally accepts the isolated incident-book API/SQL/browser workflow.

### Implemented Refresh Lifecycle

ReadingContentService owns the typed refresh DTO/query. readingRefreshModel validates
page bindings and performs pure merge; the reader controller fetches pages and owns
cancellation/publication. App only connects the callback and existing interaction
invalidation. Cleanup Restore and Undo of cleanup/record restoration use this path;
ordinary manual Edit/Delete and Ignore retain their existing refresh workflow.
This task does not claim every reading mutation has been migrated.

The controller rereads page zero against the committed mutation binding, including
on retry, then collects every 20-block page before publishing any body change.
Reject stale source/revision/structure/document bindings, duplicate IDs, invalid
hierarchy/visibility and missing insertion anchors. Merge only loaded groups using
server block order/next-unaffected anchors; do not fabricate unloaded sections or
concatenate replacement strings. Later normal navigation loads current content for
unloaded units. Update sibling source hashes and invalidate lazy generated hints
according to backend flags, without inferring missing availability as not_generated.

History draft Save refreshes history only. Restore retains batch expansion/item page,
reads current history separately, and never derives eligibility from status=restored.
The caption distinguishes a saved draft from the immutable quote without claiming
the draft is the currently applied body; only server can_restore permits another apply.
Confirmed mutation plus refresh failure retains an in-memory read-retry handle;
Retry reading does not issue POST. Missing/stale refresh capability blocks patching
instead of silently falling back to full layout. Lost mutation responses retain the
same mutation ID. No browser persistence is added.

Generated interaction state is invalidated on confirmed restoration even if its
body refresh fails. Unaffected Note drawers/drafts remain. Reader selection and current
page are retained while measurement recalculates; navigation focus is applied once
per navigation request. History submission restores its connected action's focus
after controls re-enable, with frame cancellation/generation guards on teardown.
Document/section switch, layout supersession, request cancellation and unmount cannot
publish delayed partial pages or restore stale focus. Non-navigable history remains.

Retain the existing history even when navigation is unavailable, as the maintainer
withdrew the proposed filtering/deletion rule to preserve accidental-deletion recovery.
Do not infer permanent absence from an empty client cache or can_navigate=false.
No new browser persistence, general inline body editor or manual block insertion.

Validation must prove Edit -> Restore -> Edit -> Restore updates the same fragment,
flag remains false, no duplicate text, zero task-layout fetches and correct targeted
body content. Then body interactive Edit makes history overwrite unavailable. Test
stale/cancel/response-loss/retry, retained non-navigable history and desktop/mobile layout on an
isolated copy of the incident book. Earlier CRUI tests asserted one-shot behavior
and must be revised, not cited as new acceptance.

### RCR-05 Delivery And Governance Receipt

UI-owned `scripts/accept-repeatable-cleanup.py` captures document 68 through a
read-only, repeatable-read source transaction and preserves its full SQL rows,
source PDF and existing profile under ignored `data/test-books/rcr05`. A unique
disposable PostgreSQL schema loads only those backup rows, preserving IDs,
revision 5, saved draft, prior application and Undo evidence. Desktop/mobile runs
reset from the backup, not by reparsing or synthesizing the incident. Source-path
null remains intact. Source and backup hashes are checked again after teardown.
The capture helper preserves the scoped DSN search_path when adding read-only mode.

The actual App uses production FastAPI reading-content/task-layout/content routes
and the real repository/domain logic through an allowlisted local test bridge.
Library listing and enhancement status are transport fixtures; generated interaction
availability is disabled and the existing backed-up profile is reused. No model,
OCR, Prepare or production mutation occurs. Preparation is forbidden in the test
coordinator. The bridge and disposable schema are closed even after failure.

Final evidence: `Deep_Reflective_Reader/data/test-books/rcr05/20261025T150648Z-06b7c9e3/`
contains manifest, immutable inputs, SQL/browser reports and four screenshots.
At 1280/390, block `19652:content:2` accepts three successive record applications.
Save changes only the draft; each Restore replaces its own text exactly once,
retains manually_edited=false, and triggers one targeted read with zero additional
layout or section-content requests. Both SQL reading-state copies agree. Five real
non-navigable records remain visible with navigation disabled. Body interactive
Edit sets the backend flag, disables UI Restore and makes a direct API Restore
return 409; the rejected request leaves the entire SQL fingerprint unchanged.
Canonical hierarchy, extraction rows, source PDF/profile and backups are unchanged;
the temporary schema is confirmed absent. Screenshots show fitting desktop/mobile
controls and scroll-contained history. This is source acceptance, not deployment.

Supplemental verification: `/tmp/rcr05-backend.log` reports 264 passing file/SQL
tests from test_cleanup_history and test_reading_refresh. `/tmp/rcr05-history.log`,
`/tmp/rcr05-refresh.log` and `/tmp/rcr05-requests.log` cover response loss, read retry,
Note retention, hidden/cross-block cases, 25-block pagination and stale lifecycle
responses using controlled fixtures. `/tmp/rcr05-build.log` passes TypeScript/Vite
with the existing bundle-size advisory. `/tmp/rcr05-real.log` records the real-state
run. Earlier failed harness runs are retained as failed reports, not acceptance.

Inspected markdown: AGENTS; proposal/high-level-design; module index/progress;
both development standards; UI, document_structure, scripts and db designs and
checklists; module-memory-loader and implementation-validation skills. Modified
markdown: UI design for this harness/evidence receipt; UI checklist for RCR-05
completion; document_structure design/checklist for shared acceptance and 4/4 status.
No progress.md change or additional backend completion count. Maintainer-authorized
governance only; historical RCR-04 receipts below describe their earlier scope.

Implementation-validation (UI implementation/acceptance): scope, architecture,
boundaries, documentation, testing and memory PASS; diff and script syntax checks
pass. Added only the two UI-owned acceptance scripts; no production source changes
or file removals in this task. Memory loading and validation themselves modify no
files. Pre-existing worktree changes remain untouched. RCR is complete at 4/4,
UI-owned tasks 2/2; no remaining task in this corrective track.

### RCR-04 Delivery And Governance Receipt

Evidence: `/tmp/rcr04-history.log` records actual App browser tests at 1280/390
with controlled HTTP: repeated saved/unsaved-draft application replaces text, no
layout/section reloads on Restore, response-loss same-ID replay, read-only retry
after a saved mutation, current-page/expansion/focus retention, unsaved Note retention,
server edit-protection presentation, 23-row pagination and retained non-navigable
history. `/tmp/rcr04-refresh.log` records a mounted controller with 25 affected
blocks, all-pages-before-publication, replay without duplication, six stale/failure/
cancel/switch/unmount scenarios and 13 pure merge/contract checks, including hidden
and cross-unit recovery. Backend flags are consumed, never inferred or written by UI.

Regression evidence: `/tmp/rcr04-review-state.log`, `/tmp/rcr04-review-navigation.log`
and `/tmp/rcr04-requests.log` pass; `/tmp/rcr04-build.log` passes TypeScript/Vite
with the existing large-chunk advisory. Desktop/mobile screenshots are
`/tmp/cleanup-history-1280.png` and `/tmp/cleanup-history-390.png`; review confirms
bounded scrolling, fitting controls and no horizontal overflow. Existing Vite at
127.0.0.1:5173 was reused. No production book/API mutation, LLM/OCR, Docker restart
or isolated real-state acceptance at that delivery; RCR-05 was then unchecked.
The subsequent acceptance receipt above closes it without changing this test scope.

Inspected markdown: AGENTS, proposal, high-level-design, module index/progress,
both development standards, UI and document_structure design/checklist, memory,
checklist-sync and validation skills. Modified markdown: UI design for implemented
refresh lifecycle and evidence, UI checklist for RCR-04 completion/shared 3/4;
document_structure design/checklist only to align the cross-module integration
status/shared count. Historical test receipts remain historical. No progress.md edit.
These are maintainer-requested governance updates; no global architecture changes.

Module-memory-loader: module-specific UI implementation, required memory loaded,
no conflicts; memory loading itself modified nothing. Implementation-validation:
scope, architecture, boundaries, documentation, testing and memory PASS; git diff
--check passes. Validation itself modifies no files. Added readingRefreshModel and
test-reading-refresh; edited ReadingContentService, reader controller/view,
ReadingContentActions, useCleanupHistory, CleanupAudit, App and test-cleanup-history.
No files removed. Unrelated pre-existing changes retained. Next work at that delivery
was RCR-05 isolated real-state acceptance, now completed in the receipt above.

## Per-Item Cleanup Recovery (2026-10-25)

Supersedes the CLEAN-08 claim that only last-batch Undo is available. Source-region
review remains read-only; this editor changes a cleanup record's restoration text,
not arbitrary body text. Body Edit/Delete remain the only general block editor.
No manual block insertion, model call, Prepare or reparse occurs in this workflow.

Five UI tasks: typed contract, record editor, Restore, navigation, acceptance.
Backend support is implemented in document_structure CLEAN-RESTORE. Absent record
capabilities on older deployments mean read-only history, never local guesses.
Rows expose original quote separately from editable restoration text, stable record
ID, block links with canonical block-relative Unicode char_from/char_to, status and
server unavailability reason. Browser UTF-16 offsets are never sent as write authority.

Keep batch pagination and paginate expanded deletion rows, since cleanup proposal
count/quote/deletion-ratio limits have been removed. Only the selected row mounts a
bounded multiline editor. Save persists restoration text; Restore submits that text
atomically, including unsaved edits. Original evidence never changes. Editing a
restored record does not silently modify its body; explicit Restore reapplies that
record's own fragment under RCR. General body changes still use body interactive Edit.

One controller owns drafts, busy locks, request aborts, mutation IDs and generation
guards. Drafts are in memory only; server-saved record text survives reopening.
Confirm discarding dirty drafts on close/page/collapse/navigation. Use the existing
dialog notification surface, stable classes/data-record-id and MUI icon tooltips.
Source/reading revision conflicts keep the draft and require explicit reload.
Unknown transport outcomes replay the same mutation ID; no optimistic insertion.
Confirmed restore must reload authoritative review and only affected body blocks;
the historical implementation's full-layout refresh is replaced by RCR-04.
a failed refresh offers read-only retry and never repeats the committed restore.
Saving record text does not invalidate generated interactions or rebuild layout.

Navigation rereads the specific block and current revision, then uses the existing
reader navigation controller and measured pages. No PDF-page-to-reader-page guess,
substring search or whole-book fetch. Fully cleanup-hidden blocks can Restore but
cannot navigate until visible; manually deleted blocks cannot Restore or navigate.
Changed/reparsed/unmapped targets report unavailability. Request completion after
close/switch cannot update state, reopen UI or move the reader.

Backend `manually_edited` is persistent document JSONB state, returned on reading
blocks. Restore eligibility is separately computed by the server and rechecked in
the transaction. UI never derives the flag from local editor use. Body Edit sets it;
Undo does not erase it. Editing restoration text or restoring a disjoint cleanup
record does not set the whole-block flag.

### CRUI Delivery And Governance Receipt

All five CRUI tasks are complete. `CleanupAudit`, `useCleanupHistory` and
`cleanupHistoryModel` own view, lifecycle and presentation rules; the existing
reading-content controller/service remains the sole mutation transport. No new
generic write route, browser persistence, body editor or backend layout mutation.

Validation: three browser suites (`test-cleanup-history.mjs`,
`test-source-review-navigation.mjs`, `test-reading-content-requests.mjs`) pass at
1280/390. Editor screenshots inspected at both sizes. New history coverage includes
23-record pagination, immutable evidence, dirty discard, saved text, lost-response
replay, duplicate submit, server edit guards, conflicts, reader focus, cancellation
and legacy fallback. Backend regression passes 266 tests; the existing audit suite
passes 29 tests. These include file/PostgreSQL rollback, stored edit flags after Undo,
cross-block restore and artifact/Note semantics. Read-only backup-book shared-engine
checks ran; browser transport is controlled fixtures, not live-book restoration or
live LLM acceptance. `npm run check` passes with the existing bundle-size warning.
No production book mutation, Docker rebuild or deployment was performed.

Inspected markdown: AGENTS.md, proposal/high-level-design/module index/progress,
both development standards, UI design/checklist, document_structure design/checklist,
main design/checklist, api_schemas design/checklist, db design/checklist and scripts
design/checklist. Modified markdown: UI design/checklist for the five delivered
frontend tasks; document_structure design/checklist for CLEAN-RESTORE behavior and
acceptance; main/api_schemas/db detailed designs for the implemented transport,
response and atomic JSONB contract. Evidence is the source and tests named above.
No progress aggregate was changed for this UI track; existing unrelated work remains.

## Request Wait Policy Follow-up (2026-10-24)

**[Code-Confirmed; WAIT-01 delivered 2026-10-24]** At this development stage the
frontend must not impose a fixed elapsed-time deadline on API requests. This replaces
the 120-second client wait policy described in historical cleanup/enhancement delivery
sections; both timer-triggered aborts are now removed. Scope is application
request deadlines generally, with the changed sites in
`features/smart-cleanup/controller.ts` and `features/structure-enhancement/view.tsx`.
Shared backend incident/design and task order:
[WAIT](../Deep_Reflective_Reader/llm/module-checklist.md#offline-preflight-follow-up-2026-10-24).

- WAIT-01 removes timer-triggered aborts for preflight, execute, lookup/resume and
  reading refresh. Do not replace 120 seconds with a larger constant, synthetic
  failure, deadline-triggered retry or a shared RestClient default timeout.
- Keep AbortController for owner destruction, dialog close, target changes,
  superseded requests and explicit cancellation. Preserve generation/identity checks,
  duplicate-submit guards and per-document recovery handles. A deliberate longer-lived
  owner must be explicit; removing a timer must not strand stale component callbacks.
- A live pending request stays pending until response, actual transport failure or
  lifecycle cancellation. Existing compact progress and accessible Close/Cancel remain
  available; cancel means stop waiting, not backend rollback. Do not create new alert
  panels, fake progress percentages or claim the backend job stopped.
- Browser/proxy/backend failures can still occur. With a submitted mutation handle,
  preserve unknown state and reconcile through the existing operation API before retry.
  Preflight cancellation without a handle must not become a cleanup execution or an
  invented recoverable operation. Preserve known failure versus unknown outcome.
- Consent expiry, polling intervals, debounce, upload queue scheduling and UI timers
  are not request timeouts; retain their purpose. In particular the backend's
  five-minute cleanup consent validity remains enforced. Long request waits do not
  extend it or authorize execution with an expired token. Preserve conservative UI
  expiry and obtain fresh consent when required; do not redesign the API in this task.
- This policy does not remove provider/OCR/server/proxy timeouts or bypass capacity
  limits, and does not promise an HTTP connection can survive browser closure.

Acceptance: advance controlled browser time beyond 120 seconds while actual API
responses are held; verify pending remains cancellable, then valid success is accepted.
Exercise cleanup and structuring, close/switch/unmount, out-of-order completion,
duplicate clicks, expired consent, external gateway/network failure, refresh/reopen
and recovery without duplicate generation. Run desktop/mobile browser tests and build.
Historical timeout tests become lifecycle/transport-failure recovery tests; retain
their safety assertions rather than deleting coverage with the timer.

### WAIT-01 Delivery

Cleanup retains its existing owner-scoped controller, generation check, busy lock,
consent interval and recovery handles; only deadline wiring and timer-only fallback
state were removed. Structuring now binds execution and subsequent reading refresh
to the dialog's AbortController and request generation. App forwards that signal to
existing layout/title/content services. Close, unmount and document switch invalidate
late results/finally effects. A cancelled submitted operation retains its consent
locator as unknown; a cancelled post-commit refresh remains reloadable. Closing does
not clear saved content or claim backend rollback. Cancel is available during every
wait, including reading preparation. Unacknowledged legacy layout preparation is
not marked complete; an explicit reading retry may repeat that idempotent preparation,
but never the LLM operation. The current backend flow retains authoritative readiness.

Source audit found no other API deadline/AbortSignal.timeout/Promise.race mechanism;
the zero-delay upload queue and consent/polling intervals are unchanged. No shared
RestClient timeout, backend/provider changes, automatic retry or capacity bypass.

Verification: npm run check; cleanup transport contract (36 malformed variants);
enhancement confirmation/outcomes and standalone cleanup/recovery/combined cleanup
browser suites at 1280/390. Controlled time passes 120 seconds during held preflight,
execution, content refresh and lookup; requests remain pending and valid replies are
accepted. Existing expiry, network/503, duplicate submit, draft retention, close,
document switch, reload and reconciliation checks pass. Desktop/mobile long-wait
screenshots are unclipped with usable Cancel/Close. Actual FastAPI/isolated PostgreSQL
cleanup tests use a deterministic provider; no live LLM or production book writes.
Cold/offline real-book tokenizer acceptance remains WAIT-05. Backend progress unchanged.

WAIT-05 acceptance (2026-10-24): `test-cleanup-books.mjs` now has an explicit
preflight-only mode. It reads navigation from the immutable full-book backup and
forwards readiness to the actual local HTTP service, without executing cleanup.
At 1280/390, held responses remain cancellable after 120001 ms, release into valid
consent, and cannot reopen a closed dialog. Navigation is a fixture projection,
not acceptance of production task-layout rendering. Existing five wait/recovery
browser suites, the 36-case adapter check and build pass. Full evidence, storage
scope and timing belong to [scripts WAIT-05](../Deep_Reflective_Reader/scripts/module-detailed-design.md#wait-05-delivery-2026-10-24),
not a duplicated UI completion checkbox. No UI runtime changes or deployment.

## Independent Smart Cleanup UI Plan (2026-10-23)

**[Code-Confirmed 2026-10-24]** SCLEAN-08 delivers the independent entry and
confirmed normal path; SCLEAN-09 delivers recovery/reconciliation and review reuse.
SCLEAN-06 registered the three standalone backend routes/strict DTOs on 2026-10-24;
SCLEAN-07 now supplies backend runtime/readiness composition. Unconfigured or
unsupported-storage preflight returns
503 with execution_available=false, not permission to enable the UI. No deployment
is claimed. Frontend integration is detailed below. Authoritative
[backend design](../Deep_Reflective_Reader/document_structure/module-detailed-design.md#independent-smart-cleanup-2026-10-23)
and [transport plan](../Deep_Reflective_Reader/main.module-detailed-design.md#standalone-cleanup-transport-plan-2026-10-23)
define readiness, consent and outcomes. This supersedes a UI where cleanup is
available only through smart structuring; historical CLEAN evidence below remains.

Add a document-level Smart cleanup command alongside Smart structuring in the
existing repair/action surface, using the existing icon library and tooltip. Do not
add block-level cleanup, a new editor page, arbitrary scope switches or plan editing.
An accepted Common result can use the command; imported/unparsed files cannot.
Backend preflight, not parser mode/filename/rendered noise, decides execution.
Native/unknown evidence, capacity overflow and old-server unsupported responses
show concise reasons without triggering OCR, Prepare or enhancement. Availability
does not require task units to exist; backend owns explicit missing-layout handling.

A small cleanup feature controller/service owns standalone preflight, confirmation,
operation handles and reconciliation. It reuses reading refresh/audit/Undo components,
not a copied enhancement state machine or direct fetches from App. Confirm possible
deletion and generated-output invalidation, preserved source/notes and Undo; never
ask to discard all interactions for cleanup alone. Protect unsaved edits/note drafts
before starting or replacing views. One confirmed action plans/applies and prepares
reading automatically; never expose a normal-path manual Prepare requirement.

Show running, no changes, applied/partial, skipped, failed, unknown and superseded
results truthfully using backend phases and counts. Completed is not fully repaired
OCR. Do not display model budget/JSON/raw diagnostics or infer success from HTTP 200.
After failed/partial automatic cleanup, offer Smart cleanup again as a NEW cleanup
operation, without triggering reparse. After lost response use read-only lookup;
resume committed layout only when instructed by backend. Page refresh, polling,
duplicate click, switch-away/back or draft restoration must not start a new LLM call.

Reload content/interaction availability only on authoritative content or readiness
change; preserve stable standalone hierarchy selection/IDs. Reload server review
state, not a local cleared flag. Existing collapsible audit/page references/Original,
remaining-region Ignore and latest-operation Undo are reused. Fully resolved regions
disappear only when server proves coverage; partial/textless regions stay. A later
edit can supersede Undo/recovery; preserve current work rather than overwrite it.
Store only per-document opaque recovery handles, never plans or an independent
content snapshot as backend truth. Older servers do not receive fallback reparse
requests. Executing cleanup requires a supported backend's affirmative preflight;
opening the entry may inspect unavailable/older servers without starting cleanup.

Acceptance: actual App/API/disposable storage at 1280/390 for Common-only cleanup,
enhance automatic cleanup, independent retry after failure, partial/no-op, concurrent
edit, draft protection, lost response/restart, layout-only recovery and Undo. Verify
no new split/profile/OCR calls, no duplicate commits or hidden Prepare, no clipping/
overlap and stable selection. SCLEAN-10 owns final real-book verification once.

### SCLEAN-08 Delivery (2026-10-24)

`features/smart-cleanup` owns presentation, lifecycle/consent and pure user-facing
reason mapping; `SmartCleanupService` owns encoded document paths, minimal request
bodies and validated transport responses. Neither consumes a split plan nor calls
the enhancement service. `shared/cleanupOutcome.ts` supplies the existing shared
outcome validator/messages to both entry paths; App only composes the dialog and
the shared `refreshReadyReading` callback. The reader refresh accepts an optional
AbortSignal without replacing backend hierarchy/revision truth.

The Repairs menu can check cleanup with an accepted document even when task-layout
projection fails because units are absent. Common/LLM repair actions retain their
own readiness gates. Cleanup preflight sends `{}` to the implemented document path;
execution sends only opaque operation ID and confirmation token. Consent validates
all four preservation promises and content-change-only generated invalidation.
Malformed/old responses fail closed; no raw provider/model/source metadata appears
in DOM. Eligibility, source provenance and capacity are never guessed in the UI.

One Clean text action executes cleanup and backend readiness; only authoritative
reading-ready results refresh the reading view. Changed content reuses generated
interaction invalidation and review/content refresh, retains hierarchy selection
and mounted Note drafts. An active block editor disables the entry to avoid replacing
unsaved text. Compact modal status/consent replaces scattered alerts; the screen's
notification is suppressed while the cleanup dialog is active. Confirmation focuses
Cancel and remains bounded/scrollable on mobile.

On close/unmount/document change, requests and timers are cancelled and generation
checks reject late effects. Double clicks cannot submit twice. Confirmation expires
conservatively from request start. Before execute, session storage saves only the
document-scoped opaque handles; storage failure prevents execution. HTTP failure,
timeout, aborted wait, committed-but-unready or failed reading refresh do not prove
rollback and retain the handle. Reopen/refresh blocks a new request. At SCLEAN-08
delivery, GET reconciliation, layout resume, read-refresh retry, automatic-cleanup
retry integration and full Undo/review recovery acceptance were deferred; these are
now delivered by SCLEAN-09 below.
Definitive terminal results clear the handle. No manual Prepare or fallback reparse.

Evidence: `scripts/test-smart-cleanup-contract.mjs` validates valid/partial/unknown
responses, rejects 31 malformed variants, checks encoded identity, minimal bodies,
AbortSignal and safe error mapping. `scripts/test-smart-cleanup.mjs` runs real App
components through FastAPI/isolated PostgreSQL via `accept-cleanup-flow.py`, with a
controlled provider and explicit negative transport fixtures. Accepted Common and
zero-unit fixtures prove cleanup does not require enhancement or manual Prepare.
1280/390 checks cover all normal outcomes, native/unknown/capacity/legacy gates,
consent/expiry/cancel, note preservation, manual-edit guard, duplicate submit,
response loss/reopen/refresh blocking and late mutation response after navigation.
No split calls/Prepare requests; browser runtime errors absent. Screenshots saved in
`/tmp/smart-cleanup-{consent,completed,partial,failed,no_changes,unsafe}-{1280,390}.png`;
mobile consent and desktop partial-result captures were visually inspected, with
browser assertions for dialog bounds. `npm run check`, enhancement-confirmation
regression and the existing actual-API/SQL `test-cleanup-flow.mjs` pass at both
widths, including legacy enhancement recovery/Undo. This is not live-model/book
acceptance; SCLEAN-10 was still required at this delivery, and deployment was not performed.

Governance receipt: inspected AGENTS.md, proposal.md, high-level-design.md, module
index, backend progress and shared SCLEAN order, both development standards, and
this UI design/checklist. Modified only the UI design and checklist markdown:
design records implemented ownership/lifecycle/remaining recovery boundaries;
checklist records evidenced SCLEAN-08 completion and UI-local 1/2/shared 8/10 counts.
No backend progress or checkbox was changed. Validation: source/API contract review,
type/build, adapter and browser checks above, git diff --check and scoped ownership
review pass. No forbidden architecture changes or new backend mutations on reads.
The only aggregation caveat is the untouched backend index's older 7/10 snapshot;
owner checkboxes established shared 8/10 at SCLEAN-08 delivery. Validation itself modified no files.

### SCLEAN-09 Delivery (2026-10-24)

The same `features/smart-cleanup` controller owns every standalone entry and retry.
Enhancement's completed result offers Smart cleanup after failed/partial/skipped
automatic cleanup; it only hands off to this feature, without another split or
duplicated cleanup policy. A new operation always requires fresh preflight/consent.
Provider planning, validation, commit and readiness remain backend responsibilities.

`recovery.ts` persists only per-document opaque operation ID/confirmation token in
session storage, validates old handles and compare-removes the expected ID. Invalid
storage fails closed. Opening/reopening with a handle uses read-only GET lookup,
never a new preflight or implicit execute. Unknown/planning/committing results offer
Check stored state. An explicit Resume reading click first rechecks GET and only
POSTs the existing consent when phase=layout and retry_action=resume_layout. Lost
resume acknowledgements follow the same recovery path. No hidden Prepare, reparse,
profile creation or OCR fallback is introduced by the UI.

`SmartCleanupService` validates phase/commit/outcome consistency and retains valid
operation DTOs returned with 409/503 rather than flattening them into network errors.
Timeout, abort or HTTP failure is not proof of rollback. At this historical delivery,
requests have lifecycle cancellation, a 120-second client wait limit, generation guards
and a synchronous duplicate-action lock. WAIT-01 above retires that fixed wait limit;
it retains lifecycle and actual transport-failure recovery. Client cancellation does
not claim to stop backend work. A live request remaining pending is not itself failure.
Invalid/expired consent requires another preflight, never silent regeneration.

Terminal/superseded recovery refreshes the current backend task-layout projection
and selected section contents before forgetting the handle. The reader validates
responses against this new projection, retains matching chapter/section IDs and
clears selections that no longer exist. App publishes the layout after the content
read so changing the projection cannot abort its own refresh. Review/availability
refresh follows server truth; generated views reset where needed, while Note and
critical-thinking drafts remain. A failed reading refresh retains the handle and
offers Retry reading, which rechecks GET rather than replaying a stale result or
starting cleanup. Concurrent edits and Undo supersession never restore old content.

Existing reading-content review, collapsible audit/Original, remaining-region Ignore
and latest-operation Undo are reused without a second implementation. Fully covered
regions disappear through server state; textless/unmapped regions remain visible.
Ignore removes the server review entry and refreshes the same panel. Later edits
disable/conflict Undo without overwriting current work. Manual block-edit protection
continues to prevent cleanup from discarding an unsaved editor draft.

Evidence: `npm run check` passes (existing bundle-size warning).
`test-smart-cleanup-contract.mjs` rejects 36 malformed variants and checks GET/no-store,
minimal mutation payloads, scoped recovery storage, and authoritative 409/503 DTOs.
Actual App/FastAPI/disposable PostgreSQL suites `test-smart-cleanup-recovery.mjs`,
`test-smart-cleanup.mjs`, `test-cleanup-flow.mjs` and `test-cleanup-review.mjs` pass at
1280/390. The recovery suite covers failed/partial automatic cleanup -> independent
consented retry, lost acknowledgement, service recreation/page refresh, content
refresh failure/timeout, layout-only continuation, concurrent edits, supersession,
stable IDs, Note draft preservation, review retention/resolution, Ignore, Undo and
Undo conflict. Assertions check provider/commit counts and reject hidden Prepare
fallback; stale responses after navigation do not update the view. Screenshots
`/tmp/smart-cleanup-recovery-audit-390.png` and
`/tmp/smart-cleanup-recovery-conflict-1280.png` were visually inspected with no modal
clipping/overlap. Controlled providers, Note-read fixtures and a synthetic preview
image mean this is integration/lifecycle acceptance, not live-model, real-book or
OCR-image quality acceptance. SCLEAN-10 remained necessary at this delivery. No production data was
changed and Docker was not restarted; the local source preview runs on port 5175.

Governance receipt: inspected AGENTS.md, proposal.md, high-level-design.md, module
index, backend progress/shared SCLEAN ordering, both development standards and UI
design/checklist. Modified markdown: only this design (implemented ownership,
recovery boundaries and validation evidence) and UI checklist (SCLEAN-09 completion,
timestamp and UI 2/2/shared 9/10 counts). Backend-owned progress remains 7/8; the
untouched backend index's 7/10 is its earlier snapshot, not current shared completion.
Validation evidence is above; scoped architecture/ownership and whitespace checks
are required before closing this task. Validation itself is read-only.

### SCLEAN-10 Acceptance Support (2026-10-24)

Final shared acceptance is owned once by
[scripts SCLEAN-10](../Deep_Reflective_Reader/scripts/module-detailed-design.md#sclean-10-delivery-2026-10-24),
not an additional frontend task. `test-cleanup-books.mjs` now checks standalone
recovery, fresh confirmation and an actual desktop cleanup against the live provider
through isolated FastAPI/PostgreSQL; 390px checks current-state recovery, audit,
collapse, reading reload and Undo. All three backed-up OCR books exercise standalone
and combined entry paths; Sample2 also covers injected automatic failure followed
by live independent retry. Native TXT/Fast PDF remain no-cleanup controls.

The final Sample2 desktop result and mobile expanded audit screenshots were visually
inspected with no modal clipping. Every applied deletion was checked against its
source page. Sample/dark-water retained noise is explicitly recorded as safely
rejected, not cleaned. This does not validate general OCR quality or guarantee zero
false deletions for all documents. Production Library data and Docker services were
not changed; isolated image/API smoke is not deployment. No runtime UI code changed
in this task. UI remains 2/2; the shared track is 10/10 after the owner acceptance.

Governance: inspected the global memory/standards and UI/scripts/structure module
design/checklist pairs; updated UI design for this evidence/limitations and UI
checklist's shared aggregate only. Backend acceptance is counted in scripts, not
again in UI. Build, adapter, actual-book browser and scoped whitespace validation
pass; the scripts receipt records full regression evidence.

## Optional OCR Cleanup Experience (2026-10-22)

The following describes existing enhance-integrated delivery, not an exclusive UI
entry policy. Standalone entry and recovery are delivered above by SCLEAN-08/09.

**[Code-Confirmed 2026-10-23]** CLEAN-07/08 consume the implemented CLEAN-06
contract and explicit reading-content audit/Undo projection. Canonical requirement:
[cleanup plan](../Deep_Reflective_Reader/document_structure/module-detailed-design.md#optional-ocr-cleanup-plan-2026-10-22).

One confirmed smart-structuring flow: split in memory -> optional OCR cleanup ->
single final commit -> automatic layout -> open. Consent distinguishes potential structural deletion of all
interactions from cleanup-only invalidation of generated outputs with notes retained.
Original extraction is preserved; cleanup is reversible. No new normal-path manual
Prepare or mandatory cleanup retry. Show concise stage/status text, not raw model
budgets, hashes, JSON or backend diagnostics. Do not infer OCR from rendered text.

Only after final commit succeeds, cleanup failure/partial result shows a nonblocking
warning and continues reading. Split-candidate readiness is not saved success;
final storage failure preserves the old document and shows an operation error. Report no-change honestly. Refresh, network loss and document
switch reconcile through backend operation/revision truth; ignore obsolete responses.
Only server-confirmed content changes reload content/availability and affected open
interaction views; structure change still follows existing identity reset. Old server
responses without cleanup fields remain compatible, not falsely labelled cleaned.

Expose cleanup counts and audit through the existing reading-content review surface,
with collapsible source details and source-page references where available. Use
existing single-level Undo extended by the backend; explain unavailability/conflict
without overwriting subsequent manual edits. No manual block insertion. Preserve dirty
draft protection, confirmed deletes, collapsed Original preview and explicit Ignore.
Automatic cleanup success alone cannot hide unresolved regions or disable guards.
In div[aria-labelledby="reading-content-dialog-title"], reload authoritative review
state after commit: backend-confirmed fully resolved regions leave the active queue
without another Ignore Region action. If none remain, remove the review warning;
keep audit/original evidence accessible. The newer Read-Only Source Review Navigation
contract distinguishes proven never-in-body evidence from unknown mapping and manual
deletion; do not unconditionally retain or hide every textless/unmapped row.
Undo reloads restored region states.
Backend CLEAN-04/06 provide these review/Undo semantics. CLEAN-08 now implements
the expanded audit display and controlled API/browser acceptance of resolved/remaining
queues. CLEAN-10 now adds real-model/book reading acceptance as described below.
UI cannot invent cleanup spans, persist independent content truth or call OCR/profile
on read. Cover 1280px and 390px, long audit lists, empty states, rollback, retry and
reload through actual API; heavy audit data stays outside task-layout.

### CLEAN-10 Supporting Browser Evidence (2026-10-23)

`scripts/test-cleanup-books.mjs` mounts the actual App against the isolated live-book
FastAPI/SQL result prepared by the backend acceptance harness. Sample, Sample2 and
the reported garbage sample pass at 1280px/390px: read, source-review audit, empty
history or quote/page/reason, no horizontal panel overflow and read after reload.
Sample2 additionally checks expand/collapse and mobile Undo against persisted
revision and restored content. Screenshots were inspected. Library listing alone
is stubbed empty to avoid production access; book reads/mutations are real.
Browser runs do not generate another plan or invoke manual Prepare. Controlled
CLEAN-07/08 suites remain responsible for consent, failures, conflicts and resolved/
remaining regions; live source-review queues are empty, not invented fixtures.
The backend owns the single CLEAN-10 completion count. No new frontend runtime
feature, independent content truth or backend schema claim is introduced here.

### CLEAN-07 Consent And Recovery Delivery (2026-10-23)

`StructureRepairService` requests cleanup-aware preflight. Only its implemented
`ocr_cleanup` metadata enables the extra consent and binds `allow_ocr_cleanup`
on execution/result lookup. Validate policy/scope, notes-preservation semantics and
reading revision; incompatible metadata disables confirmation. Missing metadata
uses the legacy structural-only response path, never a claimed cleanup success.
No model, token, source hash or raw cleanup reason is rendered.

Keep structural `change_kind`, `committed` and `interactions_cleared` separate from
`cleanup.reading_content_changed`. Validate cleanup counts/revision and the coherent
`phase`, layout-required, retry-stage and remaining-review flags before acting.
`phase=complete` permits reading refresh; `phase=layout` means already committed,
not a generation failure. Resume with the original consent to finish backend layout,
without frontend Prepare or another LLM call. A failed resume reconciles stored state
and never claims the committed changes were unapplied. `superseded` retains current
content; missing/contradictory results stay unknown and block blind regeneration.
Temporary precommit failures retain the existing explicit retry flow.

Session storage keeps only a per-document operation locator: original consent,
TOC hint and minimal readiness metadata. It is not completion truth. Reopening
after refresh checks the backend result, including durable committed receipts after
worker restart. Unknown results cannot authorize a new operation. Successful local
recovery clears the locator; no document text or drafts are persisted in this record.
Draft retention is in-memory only, not a promise to survive page refresh.

On confirmed cleanup-only change, reset stale generated-interaction views, refresh
the selected content and existing review state, and retain the mounted Note draft.
No-change/skipped/failed cleanup without a structural change does not refetch content
or clear drafts. Structural changes retain full identity/draft reset; title-only
changes retain the established label refresh. If page refresh left layout unavailable,
read the now-ready backend projection before reporting reading recovery complete.
Read failure offers read-only retry, not generation. Successful recovery clears stale
error notifications; background notifications are hidden while the modal is open so
they cannot cover mobile recovery controls. Dirty content edits require the existing
discard confirmation before opening enhancement. Document-scoped callbacks and
obsolete-response guards remain in force.

Validation evidence: `npm run check`; `scripts/test-enhancement-confirmation.mjs`
and `scripts/test-enhancement-outcomes.mjs`; `scripts/accept-cleanup-flow.py` driving
`test-cleanup-flow.mjs`. The cleanup harness serves production routes/coordinator,
real layout/storage and disposable PostgreSQL with a controlled provider/candidate.
It covers completed/partial/skipped/failed/no-change cleanup, duplicate clicks,
stale consent, final-write rollback, response loss/unknown, refresh plus worker
restart, layout resume including actual content opening, read-only retry and Undo
supersession. A separate note-read fixture in that harness verifies unsaved Note
retention and cancelled discard; it is not a saved-note storage acceptance claim.
Legacy suites cover conditional deletion, title/no-op draft retention, transient
retry, capacity/expiry guards and late document-switch responses. Desktop/mobile
1280px/390px screenshots and horizontal overflow are checked. No live model,
production-book mutation, deployment or CLEAN-10 real-book acceptance is claimed.

Governance receipt: inspected AGENTS.md, backend proposal.md, high-level-design.md,
docs/modules/index.md, progress.md, app design/checklist, document_structure checklist
and both UI module documents. Modified only the two UI markdown files: this design
records implemented wire/recovery behavior and limits; the checklist records CLEAN-07
evidence/date/progress. Backend progress and earlier dirty work are unchanged by this
task. Validation uses implementation-validation: scope, hierarchy ownership, read-only
projection, documentation and diff-whitespace checks; only the existing build-size
warning remains. No files are modified by the validation phase itself.


### CLEAN-08 Audit And Undo Delivery (2026-10-23)

The existing Extracted text review dialog now loads cleanup history only when opened,
using GET reading-content with include_cleanup_audit=true and cleanup_offset.
The server returns ten newest-first applied batches per page, total/offset/limit,
revision/time, applied/skipped/remaining counts, undoable status and exact deleted
quotes with safe reason/page-index projections. Internal operation IDs, canonical
offsets, policy hashes and provider output are not exposed. Empty history is distinct
from a legacy server lacking the optional field. No new source-of-truth store exists.

Each batch expands independently and quotes scroll within a bounded area. Only one
Undo slot exists: undo_action determines whether the command is Undo OCR cleanup
or Undo last content change. A later manual edit prevents undoing the older cleanup
batch; the UI does not target arbitrary historical batches or overwrite newer edits.
Revision/source token and stable mutation ID stay bound to the explicit POST. Immediate
double clicks are excluded; a lost response retries the same mutation ID. Conflicts
retain drafts/current state and offer Reload. After a confirmed mutation, generated
interaction views at all levels are invalidated locally and authoritative content/review
state is reread; mounted notes are retained. A failed reading refresh offers read-only
retry, never another Undo. Fully resolved regions disappear without Ignore; textless,
unmapped or partially resolved regions remain. Undo restores their authoritative queue.

Integration exposed a stale layout stamp after Undo. Explicit overlay mutations now
advance only a previously matching reading_revision cache stamp within the same
transaction. Canonical units, hierarchy IDs and other cache keys are unchanged;
missing/stale layout is not blessed. Snapshot projection validates the resulting
reading content before commit. No task-layout GET mutation or automatic reparse is added.

Evidence: scripts/test-cleanup-audit.py passes 29 cases (five new, 24 imported
overlay regressions). Reading/SQL overlay/API/preparation regression passes 237
cases; suites overlap and are not summed. accept-cleanup-flow.py with
CLEANUP_BROWSER_SCRIPT=test-cleanup-review.mjs passes production App/routes/disposable
PostgreSQL at 1280px/390px: original quote/page, empty/collapsed/23-record pagination,
resolved versus textless/unmapped regions, restored content after Undo and page reload,
duplicate/lost response, conflict, preserved Note draft and failed-read recovery.
The long history and note-read data are explicit fixtures, not live-model outcomes.
CLEAN-07 browser regression, TypeScript/build and diff-whitespace checks pass;
desktop/mobile screenshots were inspected. Extra accept_reading_review.py real-PDF
OCR rerun is unverified because the local OCR/renderer is unavailable. No paid LLM,
production book changes or deployment; CLEAN-09/10 remain pending.

Governance/validation receipt: inspected AGENTS.md, proposal.md, high-level-design.md,
docs/modules/index.md, progress.md; both UI documents; document_structure design/checklist,
main design/checklist, app and document_preparation designs. Modified UI design/checklist
for CLEAN-08 implementation/evidence/date and 8/10 combined progress; main and
document_structure designs for the actual read-only API projection and explicit-mutation
cache stamp; app design removes the obsolete unconditional post-Undo not-ready claim.
No backend checklist/progress totals changed. Source changes are limited to the review
components, App refresh callback, reading-content projection/mutation and route adapter;
tests use isolated fixtures. No files removed; prior unrelated dirty work preserved.
implementation-validation: scope/architecture/boundary/documentation/memory PASS;
testing WARNING only for unavailable extra real-PDF OCR regression and the existing
bundle-size warning. Validation itself modifies no files. Next task is CLEAN-09.

## Read-Only Source Review Navigation (2026-10-25)

Historical delivered scope. SRREPAIR above supersedes the permanent single-editor
restriction as a planned feature; current UI remains read-only until SRREPAIR-05/06.
The historical acceptance below is not evidence that region writes are implemented.

**[Maintainer-Provided] + [Code-Confirmed, 2026-10-25]** Current manual-review
requirements supersede the earlier inline Extracted text review editor. Backend mapping
authority and visibility rules are specified in
[Source Review Mapping](../Deep_Reflective_Reader/document_structure/module-detailed-design.md#source-review-mapping-and-read-only-workflow-2026-10-25).
The UI consumes the implemented SRMAP-04 DTO. Tasks and acceptance are
[SRMAP-05/06/08](module-checklist.md#read-only-source-review-backlog-2026-10-25).

### Historical Single Editing Location

- Extracted text review stays read-only for body text: original image, latest associated
  blocks with IDs, manual-change status/time and Go to content. Remove inline
  block Edit/Delete menus and editor selection wiring from SourceReviewBlocks.
  Reuse the existing Reading content three-dot actions as the sole editing entry;
  do not implement another editor or a character-range highlighting system.
- Ignore remains an explicit confirmed review decision, not a body edit. Retain
  the existing successfully loaded Original prerequisite. Success removes the row
  from the active list only after server confirmation; preserve backend evidence
  and audit. Manual edit/delete, including deletion of every linked block, never
  means automatic Ignore. Verified automatic cleanup resolution remains separate.
  Ignore acknowledges review without deleting retained body text; confirmation must
  not promise text removal or exclusion from generation.
- A region can intersect one or several blocks; several regions can share a block.
  Show these as related blocks, not wholly owned text. A partial intersection still
  navigates to the actual block. Editing/deleting there affects that entire block
  and follows the existing confirmation, not an implicit region-wide operation.
- No manual add/insert blocks, OCR candidate recovery into new blocks, substring
  editor, range selection or automatic text reconstruction in this track.

### Navigation And Freshness

Use server-verified chapter/section/task-unit/block IDs and current revisions, never
quote searches in the browser. Present ordered related targets when more than one
exists; default to the first surviving target. Resolve the Reading content page
using the existing pagination model after fetching the relevant content. A PDF
physical page number is not a reader page. Wait for pagination/layout measurement
before focusing the block; resize must not redirect to a different block. No full
book prefetch, implicit Prepare, profile rebuild or mutation through task-layout.

Remember review selection by document/region identity while navigating to the body.
Reopening, reselecting a row (including the same row) or explicit refresh loads current
server text/status; do not retain stale text merely because the ID is unchanged.
Body save/delete uses the existing single authoritative layout/content refresh path;
review refresh must not duplicate that operation. On version conflict, refresh and
re-resolve; deleted/stale targets cannot send empty-ID requests or fall back to an
unrelated block. If every linked block was manually deleted, keep the row with
deletion status/time and disable navigation until a surviving target exists.

Label related body as manually modified/deleted with backend-provided latest UTC
timestamp formatted for the user. Do not imply all associated text was corrected.
Smart cleanup, parsing and Ignore are not manual body modification. Undo refreshes
current state and audit; do not fabricate timestamps or recover purged Note data.

### Visibility And Interaction Lifecycle

OCR-KEEP replaces blanket hiding of never-in-body evidence: quality-excluded legacy
text and nonblank recognition gaps remain server-visible, without invented targets.
Only verified non-body evidence without recognition loss may be omitted by the backend.
Do not filter by empty block_ids, empty raw_candidate, a failed
request or unavailable mapping in JSX. Unknown/partial mapping stays visible with a
concise reason; expose only verified navigation targets. Originally mapped but
manually deleted blocks are not never-in-body. Library/reader warning and interaction
guard state must come from the same backend eligibility policy. Old-server responses
remain conservative and cannot be interpreted as already filtered or fully mapped.

Reading content editing retains 10000-code-point validation and explicit Save.
Ask whether to keep or clear the edited block's interactions, explicitly including
Note; saved old generated results cannot be labelled current after changed content.
Delete confirmation explains the block's interactions/Note are removed. Other notes
are unaffected. Undo restores text only, not deleted interactions. No manual mutation
count limit. Protect drafts on navigation/close; failed writes keep drafts and stable
mutation IDs. Review itself has no editing draft to synchronize.

Use existing MVC/service boundaries and centralized notifications. Lazy-load bounded
block lists and images; Original expands/collapses and releases resources appropriately.
Give controls/rows stable classes or labels and region/block data identities. Abort
and generation-guard reads on close, switch, supersession and unmount; clear stale
selection/menu state. Strict Mode must not duplicate requests or mutations. Preserve
keyboard focus and scroll containment on desktop/mobile and long lists.

### SRMAP Delivery And Acceptance

`SourceReviewBlocks` is read-only; `sourceReviewModel` derives navigation eligibility
and display labels, and `useSourceReviewBlocks` owns bounded reads, selection,
refresh and cancellation. `ReadingContentService` consumes the additive server DTO
and treats absent legacy fields conservatively. Only server-verified surviving
targets can navigate. The existing reader controller selects the target section;
the view finds the block in measured pagination, maintains the anchor through
resize, then focuses/scrolls it. Content batches have no snapshot token, so navigation
does a bounded target read and rechecks source/version/revision, hierarchy identity
and displayed target text before publishing or focusing. It never treats PDF
page_index as a reader page.

Same-row reselection and explicit refresh fetch new content. Snapshot guards check
both dispatch-time and latest source/version/revision before publishing or
navigating; close/switch/unmount cancels work. A shrinking block list clamps and
refetches a valid page. Deleted-linked counts and manual UTC timestamps are shown
without claiming resolution; all-deleted rows remain, with navigation disabled.
Cleanup audit is retained only for the same snapshot; a newer related-block read
invalidates old audit rather than showing stale remaining/Undo state. Ignore copy
explicitly states that retained body text is unchanged.
Reading content remains the sole editor, with existing draft protection and a
single authoritative layout/content refresh after a successful mutation.

Validation: `npm run check` passes with the existing bundle-size warning.
`test-source-review-navigation.mjs` covers actual App/StrictMode, multiple/shared/
partial/unknown/legacy targets, measured pagination and resize, same-row refresh,
failure/conflict, aborted/superseded reads and bounded-list shrinkage at 1280/390.
`test-source-review-state.mjs` covers twelve delayed-response snapshot races across
refresh/navigation/content/verification and nine display-normalization comparisons. Existing
reader pagination/request tests and 42 reader identity guards pass.

`scripts/accept-source-review.py` runs actual App/FastAPI/disposable PostgreSQL for
synthetic and `READING_TEST_REAL_CASE=sample2` inputs at 1280/390. Both pass body
navigation, Keep/Clear Note, manual edit/delete timestamps, failed save/retry,
last-block deletion retention, explicit Ignore, text-only Undo, source immutability
and exactly one layout refresh per mutation. Twelve body/review/deletion screenshots
were inspected. Real mode uses hash-verified accepted book/page text with constructed
review annotations and no injected canonical_span; Original preview is a UI fixture.
It is not a naturally flagged positive region or fresh OCR/model-quality test.
Private evidence is retained under backend `data/test-books/srmap08/`.

No production book mutation, backend deployment or Docker restart was performed.
Legacy records without persisted exclusion proof stay conservative until an explicit
backend write/backfill; the UI never performs that migration on open or refresh.
SRMAP-05/06/08 are accepted (3/3); the shared backend/UI track is 8/8. Independent
review's stale navigation/audit and misleading Ignore-copy findings were fixed;
eleven controller cases, final browser reruns, type/build and diff checks pass.

## OCR Review And Content Correction (2026-10-21)

**[Code-Confirmed]** Shared queue: backend doc_loaders
`module-checklist.md` OCR-REVIEW-01..06. UI delivery is part of OCR-REVIEW-04/05/06.
Do not infer source quality from
rendered strings or convert extraction failures into success locally.

The baseline exposes review-needed indication and failed-page/region previews.
Current target visibility is defined in Read-Only Source Review Navigation above;
quality-excluded text and nonblank recognition gaps remain in the queue under OCR-KEEP.
Empty or unavailable mappings are not sufficient proof to hide a row.
Opening review is read-only, without OCR/profile/LLM generation. Replaced source
bytes must invalidate preview identity. Task-layout stays lightweight.

Block editing uses server identity/revision, explicit Save and a 10000-code-point
limit. Confirm Delete; provide explicit Undo and dirty-draft protection. Failed
or conflicting writes retain the draft. On successful mutation reload authoritative
content and affected interaction availability; do not emulate cache invalidation
in UI state. Original extraction remains immutable. Edit asks whether to clear
interactive data including Note; block Delete removes its Note with confirmation.
Automatic span cleanup retains its separate note-preservation policy. Block generation selection with
a warning for server-reported unresolved source until reviewed; backend preflight
remains authoritative. Do not assume that low quality
necessarily means no usable content. Desktop/mobile acceptance must cover preview,
warning persistence, save/delete/undo/conflict and failed-request recovery.

Edit/Delete are interactive actions in the existing content-block three-dot menu.
No manual addition/insertion of blocks is supported. Extracted text review is a separate
document-level surface listing physical pages/regions; Original displays the
server-validated crop. Ignore requires a successfully loaded image and a second
confirmation; ignored content is not fabricated. Undo is single-level. The app
reloads authoritative content and resets interaction views after successful changes.
Unsubmitted edits are protected on close/document navigation and beforeunload.
Network failures reuse the mutation ID; stale conflicts retain the draft and offer Reload.

Implemented endpoints: GET/POST /documents/{doc_name}/reading-content and
GET /documents/{doc_name}/source-review/{region_id}/image. Reading state is not
task-layout state and cannot become backend truth. Library uses review_required.
Production-editor browser harness test-reading-content-live.mjs passes at 1280/390px
against the isolated real-backup review API, including original image, failed Save
retry, dirty-close rejection, concurrent-edit conflict, delete confirmation and undo.
This component/API acceptance does not claim a production Library re-import.

## Format-Neutral Enhancement Readiness (2026-10-21)

**[Code-Confirmed]** EXTRACT-09 is implemented against backend EXTRACT-08.
See [the shared plan](../Deep_Reflective_Reader/document_preparation/module-detailed-design.md#format-neutral-extraction-plan-2026-10-21).
Render backend readiness and recovery truth; never fabricate identity or infer
capacity from file size. TXT identity failure must not recommend OCR. Distinguish
source preparation needed from model capacity exceeded using implemented response
contracts only. Show whether intelligent splitting can run and an actionable reason,
not raw hashes/model budgets. Preserve confirmation, stale-response cancellation,
retry/reconciliation and automatic layout/open flow without manual Prepare.
Desktop/mobile tests cover TXT ready, migration required, malformed source, capacity
rejection, successful enhancement and failed operations preserving existing reading.
No new public backend fields or supported formats are claimed by this design.

## Bounded Content Loading (2026-10-21)

CONTENT-04: section reads split task-unit IDs into sequential batches of at most 20,
retaining response identity/order validation. AbortController cancels obsolete work
on section selection, reset, document/layout change and unmount. No next batch is
issued after abort. Request IDs still reject late results. Backend hierarchy remains
truth; no UI-generated IDs, prepare calls or hidden interaction generation. Backend
CONTENT-01 must independently make bulk clients efficient. Browser acceptance must
select the largest body section in each rebuilt real book, not only front matter.

## Summary And Note UI (2026-10-21)

SN-17..20 implemented and accepted. `SummaryNoteService` validates server target
identity and response shape; all five existing menus expose independent Summary
and Note actions. Single-section merged rows retain separate chapter/section
targets. `SummaryNoteDrawer` reads on opening, cancels obsolete requests, and is
keyed by action/target. Summary renders headline/body/key_points, generates missing
output on explicit Summary selection (2026-10-27 contract), retains explicit Refresh,
preserves saved output on failure, and scrolls independently.
Note is one plain-text record per target, max 10000 Unicode code points; no title/
tags/autosave. Explicit Save supplies revision, stable retry ID and target token;
Delete requires confirmation. Dirty close/document navigation and page unload are
guarded. Failure keeps the draft; source/conflict errors never overwrite server
content. Structural commit clears selection; label-only refresh leaves it intact.

Acceptance: `test-summary-note-live.mjs` uses actual backed-up/rebuilt books and
API at 1280px/390px across all five levels, saved real-provider summaries and real
note writes. It verified the then-current read-only opening contract (superseded for missing
summaries on 2026-10-27), persisted reopen, over-limit emoji,
failed save/refresh retention, cancelled/confirmed deletion and no runtime errors.
Screenshots reviewed; `npm run check` passes with the existing bundle-size warning.
The user explicitly authorized runtime deletion/rebuild of these two books; the
backend fault/rename/conditional-reparse tests run in disposable storage.

### Historical Planning Rationale

The following original requirements are superseded by the implemented contract
above. The original summary/note
backend backlog remains open; follow its finalized contracts rather than inventing
routes. Existing ancestor-summary caches and analysis.payload.summary are not the
standalone Summary action. All five hierarchy-aware target levels use backend IDs,
including canonical content_block_id, never row position or title matching.

Extend existing action menus with Summary and Note only when their corresponding
backend contracts are available. Opening is read-first and must not generate,
prepare, reparse or save anything. Keep action/target state isolated and discard
late responses after navigation. Task-layout remains a lightweight projection;
fetch summary/note payloads on demand through their dedicated service adapters.

Summary uses a dedicated scrollable drawer with current saved output, explicit
Generate/Refresh, pending and recoverable failure states. Long book summaries must
not break the reader pane's scroll layout. Do not mix interpretive insight content
into the summary. Preserve existing output during a failed refresh and report
insufficient/stale/capacity failures without claiming success or automatically
switching models. The exact payload fields depend on the implemented backend DTO.

Note uses the agreed single/multiple-note editor/list contract; do not decide that
policy implicitly in the frontend. Use plain text initially, explicit Save,
confirmed Delete, character counter matching backend Unicode units, and visible
validation without truncating drafts. Preserve drafts on save failure and warn
before dirty close/target/document switching. Stale updates must not silently
overwrite newer server content. UI state/drafts are not persisted backend truth;
no hidden autosave or LLM calls. Contract tasks must finalize exact limits first.

Confirmed structural replacement follows the existing full-clear workflow and must
include saved notes in impact confirmation. Unchanged/title-only outcomes preserve
notes and drafts; failures preserve them. Existing draft-reset behavior must be
extended deliberately, not inferred from a generic HTTP success. Browser acceptance
uses backed-up real books and actual isolated API/storage at desktop/mobile sizes;
generation quality requires separately recorded real-model evidence.

## Enhancement User Flow Revision (2026-10-20)

Maintainer-confirmed via grill-me; combined track 8/8 accepted (100%) as of
2026-10-21, UI ENH-UX-01/02 and ENH-REV-05 delivered. Backend ENH-REV-01..04
are implemented: see document_structure "Enhancement Autonomy And
Change Semantics", app "Enhancement Outcome Handoff" and db "Conditional Enhancement
Commit". Historical FULL completion does not cover this revision.

### User-Facing Readiness

ENH-UX-02 delivered: the technical `[aria-label="Model capacity"]` panel is removed,
replaced with concise readiness:
"Ready for smart structuring" when the server confirms ready/executable/full-source
fit and valid consent. Never show model names, required/available token values,
output reserves or maximum-output limits in normal UI. Do not merely hide them with
CSS, move them into tooltips or retain diagnostic content in the accessible DOM.
No advanced capacity panel is requested. Backend capacity DTOs remain available
for server decisions, diagnostics and tests, not as user-facing specification sheets.

Show actionable plain-language refusal categories: document too long for this
operation; service configuration unavailable; document exceeds current processing
limits; connection/check failed; or backend update required. A failed capacity check
is not proof of a long document. Use existing UI language conventions and no raw
server internals. The eligibility guard remains server-owned; removing numbers
must not remove safety checks or enable legacy/missing-capacity results.

Implementation evidence (2026-10-21): `view.tsx` renders readiness as a success
Alert with role=status only while confirmation is valid. `capacity.ts` maps model
overflow, service configuration, processing limits and unsupported/missing responses
to plain-language refusals. Preflight connection/check failure does not echo raw
exception details or masquerade as overflow. Technical capacity data is neither
rendered nor moved into hidden markup/tooltips; response types remain unchanged.
Destructive-consent inventory remains separate and is not removed by this task.

`test-enhancement-confirmation.mjs` passed at 1280px/390px, checking HTML and the
accessibility tree for absent diagnostics, readiness presence, expired/blocked
readiness absence and zero generation on rejected preflight. Desktop/mobile
screenshots inspected; `npm run check` passed with the existing bundle-size warning.
Real FastAPI/disposable PostgreSQL browser acceptance also passed both widths,
preserving automatic layout preparation and zero normal-path layout 409.
No paid provider call, production-data mutation or backend API change.

### One-Step Operation

Recovery clarification (2026-10-20): temporary LLM HTTP/provider failures are
retryable; retry cost is not a gate in this phase. A contextual Retry resumes the
failed stage. Before retrying an unknown generation outcome, reconcile server work/
result identity to avoid duplicate commits, not to enforce a spend limit. Completed
generation is reused when downstream preparation/read fails. Over-capacity requests
never start; valid no-op does not trigger automatic regeneration. Cost controls are
future work. No manual Prepare action is required in the normal enhance flow.

One confirmation authorizes enhance and its required reading recovery:
checking -> confirmation -> generating -> confirmed result -> preparing reading
layout (only after structural change) -> reload -> ready. Do NOT attempt the known
unready task-layout read first or require a normal-path "Prepare reading layout"
click. The frontend calls the existing explicit layout-only preparation API;
task-layout reads still perform no hidden persistence mutation.

ENH-UX-01 delivered on 2026-10-21: `reload()` calls `onPrepareLayout` before
`onCommitted`, with a preparing phase. A document-scoped in-memory checkpoint
records successful preparation, so read failure retries only reading. A recovery
single-flight set prevents overlapping recovery for the same document; late preparation
completion retains the checkpoint without loading a different active document.
Preparation and read failures explicitly report that the structure is already saved.
Recovery state lasts only for this mounted app, not a durable server-job record.

Validation: `npm run check` passed (existing large-bundle warning);
`test-enhancement-confirmation.mjs` passed at 1280px/390px including preparation
failure, read failure, read-only retry, duplicate click, timeout reconciliation and
document switches during preparation. `test-enhancement-live.mjs` through backend
`accept_enhancement_browser.py` passed both widths against actual FastAPI/disposable
PostgreSQL, asserting every task-layout read returns 200 and one automatic
layout-only prepare follows each successful commit. Desktop/mobile preparation
screenshots inspected. No paid provider, user-document mutation or Docker restart.
This initial recovery slice is now extended by ENH-REV-05 below; capacity-panel
simplification is delivered by ENH-UX-02.

Disable duplicate commands during generation/preparation; show the current operation
stage. Preparation failure is "structure saved, unable to prepare reading", not
"LLM failed". Recovery retries only layout-only prepare/read, never generation, OCR
or full raw preparation. Ignore late responses for another document; retain recovery
state for the original document. If preparation already succeeded and only reload
failed, avoid a forced rebuild; repeated preparation must be non-forced/idempotent.

### Three Result Branches

Implemented backend result meanings, consumed by ENH-REV-05:
- unchanged: show "No structure changes"; retain IDs, interactions, drafts and reader
  position. No version advance, layout rebuild or automatic regeneration.
- titles_only: reload navigation labels while retaining target IDs, body selection,
  content, saved interactions and local drafts. Do not call the generic destructive
  structural reset path; unchanged structure_version is expected.
- structure_changed: clear old target-bound caches/drafts only after confirmed
  atomic backend replacement; all persisted interactions are cleared by backend.
  Automatically prepare layout-only, then read current hierarchy and resume reading.

Confirmation warns conditionally: changing divisions/levels/parents clears all
document interactions; unchanged or navigation-title-only results preserve them.
Use a neutral "Enhance structure" command, not unconditional "Discard and enhance".
The result is unknown before generation; do not promise improvement or preselect
a non-destructive outcome. Known notes must be included in the conditional warning.
Missing outcome fields must not be guessed from HTTP 200, chapter count or
version equality: require a compatible backend before enabling the new branch flow.

Timeout/abort is unknown, never a rollback claim. Same-version label commits/no-op
require the authoritative reconciliation contract implemented by app ENH-REV-03;
the former version-increase heuristic is removed. Reconcile
timeouts and then resume the appropriate stage, including generation retry when
needed. Do not automatically regenerate a valid no-op or a confirmed committed plan.

Execution order: ENH-UX-01 -> ENH-UX-02 -> structure ENH-REV-01 -> db ENH-REV-02 ->
app ENH-REV-03 -> structure ENH-REV-04 -> ENH-REV-05 -> structure ENH-REV-06.
Only three items are UI-owned; backend progress.md excludes these.

### ENH-REV-05 Delivery (2026-10-21)

`outcome.ts` accepts only same-document enhanced responses with consistent
`change_kind`, `committed`, `interactions_cleared` and `reading_layout_required`.
Old/missing/contradictory successful responses remain unknown, never destructive.
Preflight requires `cleanup_condition=structure_changed_only` alongside full-source
fit and unexpired consent. The confirmation explains conditional whole-document
cleanup, including notes, using a neutral Enhance command.

Unchanged displays a result notice without layout reads or preparation. Title-only
uses `refreshEnhancedTitles` and `refreshSelectionLabels`, retaining loaded content,
selection IDs, displayed insights and target-keyed drafts. Content request guards
use reset/request generation and document identity after dispatch, allowing pending
content to survive a label-only layout object replacement. Structural results call
`clearEnhancedDocument` once, then explicit layout-only preparation and read;
separate local-clear/prepared checkpoints prevent repeat cleanup or preparation
after downstream failure. Switching documents retains recovery state without
applying late responses to the new active document.

Closing the critical-thinking drawer now retains target-keyed drafts in App memory,
otherwise closing it to access enhancement would discard them before comparison.
This supersedes historical close-to-discard behavior. Drafts are not persisted across
page reloads; existing document reset/submission/new-question semantics still apply.

`StructureRepairService.enhancementResult` calls the implemented result endpoint
with original document, consent and TOC hint. Completed results enter the same typed
branch handler; running/unknown remains blocked, failed retryable/not-started allows
explicit same-operation retry, and terminal failure requires fresh preflight/consent.
Temporary provider failures expose Retry smart structuring without cost restrictions.
Expiry is rechecked before dispatch. No version comparison authorizes recovery.
The backend registry is process-local; restart/eviction/wrong-worker unknown results
cannot be resolved by the UI and do not silently start another operation.

Validation: `npm run check`; confirmation, outcomes and reader-recovery browser
suites at 1280px/390px; desktop/mobile screenshot inspection. Outcome tests assert
saved insight and unsent draft retention, unchanged zero reads, same-version title
refresh without content refetch, retry and authoritative reconciliation. The existing
real API/disposable PostgreSQL browser harness was updated only to expect conditional
cleanup: structural replacement then no-op, preserved snapshot and zero layout 409.
No paid LLM, production data or Docker restart. ENH-REV-06 semantic/end-to-end quality
acceptance is now complete in document_structure; live model quality remains unevaluated.

ENH-REV-06 integration evidence: `test-enhancement-lifecycle-live.mjs` runs through
the disposable backend browser harness at 1280px/390px. It covers real three-way
responses, unchanged zero layout reads, title-only no preparation, transient retry
with identical consent, lost title/structural response reconciliation and downstream
preparation failure recovery without another generation call. All normal task-layout
reads return 200. This supporting UI test adds no new feature or UI task; backend
document_structure owns acceptance. Screenshots are inspected separately from live
model semantic quality, which was not evaluated.

## Full-Source Enhancement Capacity UI Plan (2026-10-20)

FULL-04 is implemented against backend FULL-03's additive capacity contract.
The combined track is 5/5 complete (100%); FULL-05 synthetic integrated acceptance
passed on 2026-10-20. Live model semantic quality remains unverified.
Consume server
decisions, never infer model capacity from file size, character counts or UI defaults.
Keep the single LLM enhance action and conditional cleanup confirmation lifecycle.

After preflight, a capacity rejection disables generation and explains that the full
document exceeds the selected model's allowance, no LLM request was sent and existing
structure/interactions remain unchanged. Distinguish model capacity, unavailable
configuration, local processing limits and network/preflight failure. Never label a
failed HTTP fetch as proof the document is too long. Present only readiness or an
actionable refusal. ENH-UX-02 removes model/token diagnostics from the user interface;
backend measurements remain internal diagnostic evidence, not a normal UI requirement.

Offer close/continue Common reading; do not silently truncate, switch models, retry,
start OCR or clear interactions. Re-evaluation after an explicit relevant configuration
or document change requires fresh preflight. No new model-settings screen is assumed.
Explain safe preservation only for confirmed pre-dispatch/pre-commit rejection; retain
existing timeout/unknown-outcome reconciliation. Only a confirmed body-structure
replacement clears interactions and frontend target caches; title/no-op outcomes
follow the implemented retention rules above.

Model/policy/source changes invalidate confirmation. Preserve duplicate-click guards,
document-scoped pending states and stale-response protection. Missing capacity fields
from an older backend must not be presented as a verified fit; request compatible
backend deployment before enabling this new capacity-confirmed path.
Acceptance: mocked service states plus desktop/mobile keyboard checks; long localized
messages wrap without overlap, cancel sends no generation, rejected execution keeps
reader state is preserved; transient generation failure remains retryable without
cost gates. No UI state becomes backend truth;
task-layout remains a lightweight hierarchy projection without hidden mutation.

Implementation evidence: `src/features/structure-enhancement/capacity.ts` owns
capacity messages and typed pre-dispatch rejection recognition. `view.tsx` requires
`capacity.status=fits`, `full_source=true`, execution readiness and unexpired consent.
Missing/unsupported decisions fail closed. The current implementation's model/token
panel was obsolete UX and has been removed by ENH-UX-02. No browser-side budget
calculation is introduced.
Preflight transport failure cannot claim model overflow. Execution 413/503 is treated
as a definite capacity rejection only when its status, detail and capacity status match
the backend contract; generic failures remain unknown. Rejection drops consent and
requires explicit preflight; backend revalidation remains authoritative for configuration
changes after confirmation. Cancel returns to the existing reader without mutation.

Validation (2026-10-20): `npm run check` passed; mocked
`scripts/test-enhancement-confirmation.mjs` passed at 1280px and 390px, covering
missing capacity, unconfirmed full source, unavailable/local/model limits, long model
names, estimates/output labels, network failure, changed policy, expired/stale consent,
keyboard cancel, duplicate submission, pending document switches and timeout/reload
reconciliation. Tests use no paid LLM calls or user-document writes. Live backend
end-to-end capacity acceptance belongs to FULL-05, not this UI-only validation.
FULL-05 supporting evidence: `scripts/test-enhancement-live.mjs`, driven by backend
`accept_enhancement_browser.py`, now passes at 1280px/390px through real FastAPI and
isolated PostgreSQL, including capacity/resource/configuration rejection and consent
invalidation on source/model/policy changes. Success uses a deterministic provider
and full synthetic source, not a paid model. Screenshots verify mobile scrolling.

## LLM Enhance Destructive Confirmation (2026-10-19)

ENH-UI-01 implemented against the backend preflight/consent contract. Public backend
execution is now capability-gated by ENH-PREP-03; UI must honor execution_available=false.
Keep one `LLM enhance` command, not two planner-intent actions.
See `../Deep_Reflective_Reader/document_preparation/module-detailed-design.md` for
the six-task execution order and accepted-source reuse contract.

Clicking enhance first requests the read-only preflight, then opens a
confirmation dialog. Always ask, even when local interaction caches are empty;
backend scope/counts are authoritative and unknown is not zero. Show the document
name and explain that changing chapter/section divisions, levels or parent grouping
clears existing insights, quizzes/answers, critical-thinking sessions, other persisted
interactions and local drafts, including book-level data. No-op and title-only
results preserve them. Explicitly name affected notes. The source file is retained.

Use a neutral "Enhance structure" title/command with the conditional warning above;
default focus on Cancel. Escape/cancel does not submit generation or delete anything.
There is no user-selectable keep-stale-interactions option for structural replacement.
Rejection/failure before commit preserves data; cleanup occurs only with successful
body-structure replacement, not merely any successful LLM response.
Unready source disables confirmation and offers separate explicit Prepare guidance,
never automatic OCR or preparation.
Also require execution_available=true before enabling the submit action; backend
preflight readiness alone does not mean the structure-only commit path is enabled.

Bind the dialog to document identity/version, selected hint and server consent token.
Document/hint changes, expired tokens or changed interactions require new preflight
and confirmation. Disable duplicate submission while pending; never reuse a global
confirmation flag. Ignore late responses for another document. After confirmed
structural commit, clear old IDs, interaction caches and drafts, prepare layout-only,
then reload. Label-only/no-op retains them under ENH-REV-05. Cancel or definite
pre-commit failure retains them. A timeout/abort is an unknown outcome: reconcile
backend state before retrying, never claim rollback or blindly duplicate active work. A
post-commit layout reload error offers reload, not another enhancement operation.

Acceptance includes keyboard/mobile dialog behavior, zero-count/unknown-count scope,
cancel with no mutation, stale tokens and concurrent new interactions, source not
ready, duplicate clicks, failed generation/commit, late responses and timeout versus
committed-but-reload-failed states. UI state never becomes backend consent authority.

Implementation: `features/structure-enhancement/view.tsx` owns the confirmation
lifecycle; `StructureRepairService` exposes separate preflight and consent-bound
submission calls. The old generic repair method refuses llm_enhanced, preventing a
second frontend entry without confirmation. App only resets targets/content/drafts
after a matching successful response or a later authoritative structure version.
After structural commit the same frontend operation explicitly prepares layout
with layout_only=true, force_rebuild=false and no page evidence before reading.
The former read-first/expected-409/manual-prepare path is retired as a UX design.
Preparation failures expose recovery without regenerating the LLM plan; the
task-layout endpoint itself stays read-only.

Cancel is initially focused after the menu/dialog transition. Preflight failures,
unknown counts/notes, zero interactions, server gating and expired tokens are
explicit; zero does not waive consent. TOC hint is snapshotted with the token.
409/422 or a document-matched success=false response invalidates consent and requires
another explicit check. Other ambiguous errors and lost connections retain local data
and block re-submission. WAIT-01 removed the historical 120-second client timer;
external timeout/unknown-outcome recovery remains required. Aborting waiting
is not server cancellation. Pending/unknown/reload state survives dialog close and
document switches in this mounted app; this is not a durable server-job registry.

Read-only reconciliation compares backend structure_version with the submitted
version. A newer version permits reloading current truth (not a claim that this
particular request succeeded). An unchanged/missing version cannot prove rollback
or completion: the UI stays blocked and offers another check. The backend currently
has no job-status endpoint; browser restart/cross-tab reconciliation is not claimed.
The frontend does not invent such an API. ENH-REV-03/05 must implement the recovery
contract before claiming reliable unknown-outcome retries; paid-call cost is not a
reason to prohibit transient-failure recovery.

Evidence: `scripts/test-enhancement-confirmation.mjs` covers 1280px/390px consent,
Cancel/Escape focus, unavailable/unready/expired, stale consent, definite 502 failure,
duplicate submission, document switch, lost connection, actual client timeout and
reload failure/retry with mocked APIs only. `test-reader-recovery.mjs` now confirms
before enhancement and preserves existing reading on rejection. Both desktop/mobile
dialog screenshots were inspected; typecheck/build passes. No live provider calls,
backend activation, migration or user-book mutation. This task updates only UI-owned
memory; backend progress.md remains backend-only. Track: UI 1/1, combined 6/6 (100%)
under isolated engineering acceptance. test-enhancement-live.mjs, driven by the
backend accept_enhancement_browser.py harness, also passes both widths against
actual FastAPI routes and disposable PostgreSQL. Only planner/unit resolution are
deterministic substitutes. No deployment or live paid-generation claim.

## LAYOUT-02 Integration (2026-10-16)

Backend app LAYOUT-02 is implemented. Ordinary Open remains a single read-only
task-layout request; 409 readiness guidance is shown through existing error handling,
without silently calling prepare. Post-upload explicit Prepare now calls
`DocumentPreparationService.prepareReader` -> `/documents/prepare-task-layout`
with progressive splitting, matching reader requests. Success is reported only
after the backend returns persisted/readback layout, not merely structured-ready.
Failures stay retryable and do not announce success.

The explicit repair and TOC-commit callbacks use `layout_only=true` preparation to
rebuild units on the committed hierarchy without parsing raw source or switching
PDF mode. Normal document navigation still uses fetchTaskLayout. Management `ready`
is not upgraded to layout-ready by UI inference. Automatic upload preparation,
complete Open gates, cancellation and mode selection remain pending UI tasks.

Evidence: `scripts/test-library-identity.mjs` intercepts all APIs and verifies no
prepare on read 409, visible guidance, explicit Library preparation failure/retry/
success, and the layout-only service payload; existing identity/race cases and
`npm run check` pass. This supports app LAYOUT-02, not additional UI task completions.

## Module Purpose

`Deep_Reader_UI` is the Web UI client for Deep Reflective Reader.

The first-phase responsibilities are:

- consume task-layout responses from the backend
- render Chapter / Section navigation
- keep task units as internal section content-addressing data
- fetch selected section content on demand by requesting its task-unit content in backend order
- render segmented content blocks
- provide frontend loading, error, empty, and initial states
- provide a responsive reading layout
- keep navigation state local to the frontend runtime
- provide a document-name search control constrained to backend API candidates

## Architecture Boundary

- Backend hierarchy is the source of truth.
- `/documents/task-layout` is a lightweight projection for navigation.
- Section is the visible reading/navigation unit for this slice.
- Task units are backend content-addressing units retained inside each section.
- Task-unit content is fetched on demand from the task-unit content endpoint when a section is selected.
- The frontend prioritizes rendering `content_blocks`.
- Task units are not rendered as primary hierarchy nodes.
- Content blocks are not hierarchy nodes.
- Frontend state is not backend truth.
- The frontend must not perform hidden backend mutation.
- The frontend must not extend or reinterpret backend API contracts.
- Future artifact, annotation, and question interactions are out of scope for this phase.
- Backend schema/API changes are out of scope for this module slice.
- Manual TOC editing uses explicit validation and hard-reparse submit paths, not hidden task-layout mutation.
- Selecting an already prepared backend document should prefer the read-centric task-layout route; prepare-then-read is a fallback or explicit repair path, not the default meaning of "open document."

## First Vertical Slice

```text
Document input
  -> Task Layout
  -> Chapter
  -> Section
  -> Section Selection
  -> On-demand Task-Unit Content Fetches
  -> Aggregated Content Blocks
```

### UI State

- `docName`: controlled combo-box input value.
- `layout`: latest task-layout response for the loaded document.
- `selectedSection`: frontend-local selected section reference with retained `task_units`.
- `contentBlocks`: aggregated `content_blocks` from the selected section's task units.
- `layoutStatus`: initial / loading / success / error.
- `contentStatus`: initial / loading / success / error.
- `errorMessage`: visible failure detail for the relevant request.

### Component Responsibility

- App shell owns state, API calls, and top-level layout.
- Document search owns doc-name input, backend candidate lookup, empty-result list fallback, and API-returned option selection. Selecting an API-returned option triggers layout loading without a separate submit button.
- Hierarchy navigation renders backend chapter and section projection.
- Selectable section tree items update local selection state while retaining internal task-unit ids.
- Reader panel fetches each task-unit content for the selected section on demand and renders aggregated `content_blocks`.
- Empty-state views handle missing document, missing hierarchy, missing task units, and missing content blocks.

### Chapter / Section Tree Navigation (2026-10-26)

`.chapter-list` now hosts MUI X `SimpleTreeView` rather than the earlier nested
MUI `List`/`ListItem` presentation. The visible tree remains at most two levels:
chapter and section. A chapter with one section retains the existing merged row,
section reading target and chapter interaction menu; task units are never tree nodes.
Backend hierarchy/order/IDs and `/documents/task-layout` remain authoritative.

`NavigationTreeItem` uses MUI `useTreeItem` and the supported TreeItem components
for registration, expansion, roving focus and arrow-key navigation. Section selection
is controlled by the existing reader selection and dispatches the same section and
ordered task-unit IDs. Clicking an already selected section still re-reads it;
mouse selection dispatches once, while keyboard selection follows the tree callback. The root exposes `treeitem`, selected and disabled semantics.
Empty sections cannot start reading. Chapter/section action menus retain their
original target identities; menu and inline-insight keyboard events do not bubble
into tree navigation. Inline insight regions remain outside the selectable label.

Chapters start collapsed, except the selected section's parent when the tree mounts.
MUI group transitions unmount collapsed children. Expanding/collapsing is local UI
state, keyed by document, with no content fetch, persistence, profile regeneration
or hierarchy mutation. Selection survives collapse/reopen. Same-document render
updates preserve expansion; switching documents resets it. This bounds initial
section DOM growth without introducing task-unit nodes or changing API payloads.

Long chapter/section titles wrap completely, including unbroken strings. Rows grow
vertically, use 16px child indentation and keep a fixed-size action column. Unit
counts and subtitles occupy lines below the title, so counts no longer squeeze
long titles horizontally. `.chapter-list` remains the bounded vertical scroll
surface; the header and reading pane preserve their existing scroll containment.

Validation: `npm run check`; `scripts/test-hierarchy-tree.mjs` at 1280/390/320px
covers keyboard and mouse navigation, disabled leaves, selected-state preservation,
menu focus restoration, full-title bounds, no horizontal/page overflow, document
reset, no expansion API calls or browser-storage writes. A fixture containing
100 chapters/1,000 sections mounts 100 initial tree items and 110 after expanding
one chapter; one local development run measured 480ms navigation/render and 67ms
last-chapter expansion. These timings are observations, not a universal performance
guarantee. Desktop/mobile screenshots were inspected. Existing hierarchy header,
interaction-menu, pane-resize, pagination and reader identity/recovery regressions
also pass (42 identity guard cases). Interaction-menu tests use
real Madame Bovary layout metadata with fixture content/interaction reads; they
verify the 451-unit merged row and chapter/section targeting without production
mutation or LLM calls.

### Reading Content Pagination

Reading content pagination is a frontend-only viewport projection. It exists to prevent
very long chapters or sections from overwhelming the reading surface, especially on
small screens or when a user uses larger type. It should improve reading rhythm without
changing backend hierarchy, backend task-unit semantics, or task-layout projection.

The backend should continue to provide ordered task units and ordered `content_blocks`.
It should not calculate how many task units belong on one UI page, because page capacity
depends on frontend-only runtime conditions such as reader pane width and height, font
size, line height, browser zoom, device density, and responsive layout. A UI page is not
a backend page, not a document hierarchy node, and not parser authority.

The reader-content feature owns pagination as local UI state:

- `content_blocks` remain the render/input units received from backend task-unit content.
- frontend pages are derived from the current `content_blocks` plus current viewport and
  typography conditions.
- section selection resets the current reader page to the first page for that section.
- viewport resize, responsive breakpoint changes, or future reader typography changes
  should trigger repagination.
- page state must not be persisted as backend truth and must not mutate task-layout,
  task units, content blocks, profile diagnostics, or parser artifacts.

The first implementation should use block-level pagination: `content_blocks` are the
smallest non-splittable page units. A frontend page may contain blocks from multiple task
units, and one task unit may span multiple frontend pages. If a single content block is
larger than the available reader page, the initial behavior may keep the block intact and
allow that page to overflow locally; block-internal soft splitting can be a later
enhancement after the basic page model is stable.

The page packing rule should be that the current rendered task-unit/content-block group
does not make `.reader-content-block-list` contain hidden, unreachable, or clipped content
that requires the whole reading panel to scroll. Pagination should choose the largest
ordered group that fits the visible reader page. The explicit exception is when one
individual task unit or one non-splittable content block is itself longer than the
available page. In that case the UI should render that single unit/block as the page
content and allow local scrolling for that oversized page, rather than combining it with
additional units that would make the overflow worse.

The expected evolution is:

1. start with deterministic block-level pagination using a conservative estimate or
   simple measurement strategy;
2. add DOM measurement pagination based on actual rendered block heights;
3. repaginate on reader-pane resize and future typography changes;
4. preserve stable block/task-unit identifiers so future reading progress, annotation,
   or selection features can map back to backend-provided content targets without making
   frontend page numbers backend truth.

### Scroll Containment

The Reader UI should keep document-level scrolling disabled during normal app use. The
browser `html` / `body` surface is the viewport shell, while `.reader-layout` is the
bounded reader workspace.

This matters because the top app shell combines a fixed-height viewport, a sticky
topbar, grid padding, and independently constrained navigation/content panes. If
`html`, `body`, `#root`, or `.app-shell` use only `min-height` without a bounded height
and overflow policy, the reader workspace can push the full document taller than the
viewport and create page-level scrolling. That makes pagination and TOC editor range
workspaces feel unstable because scrolling occurs outside the reader workspace.

Expected layout behavior:

- `html`, `body`, and `#root` occupy the viewport and do not become the primary scroll
  surface.
- `.app-shell` is a viewport-height flex container with `min-height: 0`.
- `.reader-layout` is the non-scrolling desktop workspace container for the hierarchy
  and content panes. It should be height-bounded by `.app-shell`, use `min-height: 0`,
  and avoid becoming the primary scroll surface during normal desktop reading.
- The hierarchy pane should constrain its own height and let `.chapter-list` own
  hierarchy overflow. The hierarchy header remains visible while the chapter/section
  list scrolls inside its parent.
- Long document-level inline insight content in the hierarchy pane should own a bounded
  local scroll surface, so generated book insights do not force document-level scroll
  or become clipped by the fixed reader workspace.
- The content pane should constrain its own height and let the reader-content internal
  surface own content overflow. The right pane should not push `.reader-layout` taller
  than the viewport.
- child panes such as `.navigation-pane` and `.content-pane` should size from the
  available `.reader-layout` space rather than independently adding another
  `calc(100vh - ...)` height on top of grid padding.
- oversized single reader pages may still scroll locally inside
  `.reader-content-block-list-oversized`, but ordinary page overflow should remain
  contained by reader-content pagination/inner content surfaces rather than by
  `.reader-layout`.

Desktop target behavior:

- `.reader-layout` does not scroll.
- `.chapter-list` is the primary scroll surface for long chapter/section navigation.
- `.reader-content` or its inner content list is the primary scroll surface for
  oversized reader content, with ordinary paginated content still avoiding scroll when
  it fits the visible reader page.
- If no suitable parent element currently exists for either scroll surface, the UI may
  introduce a thin structural wrapper whose only role is layout containment. That
  wrapper must not add hierarchy semantics, backend state, task-layout mutation, or a
  second source of truth.

Responsive/mobile behavior may remain more flexible: single-column layouts may still
use local pane scrolling where needed, but should preserve the same principle that
document-level scrolling is not the normal reading surface.

This is a frontend-only viewport contract. It must not change backend hierarchy,
task-layout payloads, task-unit content APIs, profile diagnostics, parser artifacts, or
manual TOC commit semantics.

### Feature Module Structure

The Reader UI is organized by user-facing responsibility:

- `src/features/book-search/`: top document-search entry feature.
- `src/features/file-management/`: document library surface for read-only catalog loading, empty/error/list states, and explicit document-open row actions.
- `src/features/hierarchy-navigation/`: lower-left chapter/section hierarchy navigation feature.
- `src/features/reader-content/`: lower-right section content rendering feature.
- `src/features/toc-editor/`: right-pane manual TOC editing feature for source documents whose table of contents is missing, incorrect, or not automatically recognized.

Each feature follows a lightweight MVC-style split:

- `model.ts`: pure UI/domain helpers such as option normalization, hierarchy counts, display labels, parser-mode derivation, heading selection, and content-block aggregation.
- `controller.ts`: React state orchestration hooks and side-effect coordination for that feature.
- `view.tsx`: MUI/DOM rendering only.
- `index.ts`: public feature exports.

`App.tsx` remains the shell/composition layer. It wires feature controllers, feature views, top-level repair controls, route selection, and layout panes without owning low-level REST request construction or content aggregation logic.

### Route Structure

React Router provides document-scoped routes:

- `/documents/:docName`: default reader route. The right pane renders `reader-content`.
- `/documents/:docName/toc-edit`: manual TOC editor route. The right pane is fully replaced by `toc-editor`; the reader-content DOM is not mounted for this route.

The `toc-edit` route is not a cold-start entry point. It requires a loaded task layout in frontend state. If a user directly opens or refreshes `/documents/:docName/toc-edit` without a loaded layout, the right pane must show a guarded state such as "Load document first" and must not automatically prepare or load the layout. The user can return to `/documents/:docName` or the default entry flow.

The TOC editor entry trigger belongs in the hierarchy navigation header, near the document title and unit count. It is enabled only when the layout has loaded successfully. Activating it routes to `/documents/:docName/toc-edit` while preserving the loaded hierarchy in the left pane as context.

### Reading Interaction UI Design

This section defines the Reader UI behavior for backend reading-interaction features:
insight, quiz, and critical-thinking session. Backend REST routes and public schemas
for generic reading interactions are now exposed for all three interaction families.
The current UI integration wires hierarchy-level document/chapter/section/task-unit
targets through `ReadingInteractionService` for insight, quiz, and critical-thinking
flows. The next backend-ready frontend slice is content-block target integration, with
target identity gates so only backend-addressable segmented rows start interaction
flows.

Reading interactions are artifact/session interactions attached to backend hierarchy
targets. They must not become frontend hierarchy, task-layout truth, or hidden backend
mutation. The target identity must come from backend-provided ids:

- book/document target: current loaded `doc_name` / document identity
- chapter target: `chapter_id`
- section target: `section_id`
- unit target: `task_unit_id`
- content-block target: `task_unit_id` plus canonical `content_blocks[].block_id` as
  `content_block_id`, with optional parent `chapter_id` / `section_id`

Optional parent ids may be carried by future requests for consistency checks, but the UI
must not target interactions by title text or by visible row position alone.

The UI implementation should be decomposed into four large design areas:

- target/action surface: document, chapter, section, task-unit, and content-block
  three-dot menus
- inline insight surface: compact target-local insight placement and statuses
- quiz drawer: read-first quiz rendering, local practice, and generation controls
- critical-thinking drawer: session question, answer, evaluation, and retry flow

The original checklist split these areas into small implementation tasks so backend API
exposure could remain narrow and each UI behavior could be validated independently.
The follow-up content-block slice is now split separately below.

#### Interaction Entry Points

Each readable target should expose a compact MUI icon button using the vertical
three-dot `MoreVert` affordance. The button opens a MUI menu with exactly these first
actions:

- `Insights`
- `Quiz`
- `Critical thinking`

Placement:

- book/document action: in the hierarchy/navigation header near the loaded document
  title and existing document-level controls
- chapter action: at the right edge of each chapter row
- section action: at the right edge of each section row
- unit action: at the right edge of a task-unit content group header inside the reader
  content surface
- content-block action: at the right edge of each visible `ContentBlockView` row once
  the UI implementation consumes the backend `content_block` target contract

Task units remain internal content-addressing units, not primary navigation nodes. The
unit action is therefore rendered only inside reader content where task-unit grouping is
already available from the selected section content response. The hierarchy navigation
must not become a task-unit tree.

Current implementation distinction:

- `taskUnitId` identifies the task-unit group that owns interaction targeting.
- `content_blocks[].block_id` identifies a rendered content block inside that task
  unit, for example `7725:content:0`.
- The task-unit interaction menu is currently rendered at the task-unit group header,
  not beside each `content_blocks[]` row.
- A visible block id such as `7725:content:0` therefore does not currently get its own
  three-dot menu because `ContentBlockView` renders block metadata/body only and the
  frontend `ReadingInteractionTarget` model has no `content_block` target level.

Maintainer direction: business interaction should eventually happen at the visible
content-block level. Backend support for `content_block` reading-interaction targets
now exists: public target schema, hierarchy-aware block resolution, selected-block
source context, block-scoped artifact identity, generic route acceptance, and shared
response envelopes are code-confirmed. `ContentBlockView` can therefore render a
per-block `MoreVert` affordance in a later UI implementation using the block id as
target identity instead of borrowing the parent task-unit target.

Backend handoff contract for a content-block menu:

- use the loaded document `doc_name`;
- use `TaskUnitContentResponse.task_unit_id` from the owning task-unit content response;
- carry `TaskUnitContentResponse.chapter_id` and `section_id` when present;
- submit `content_blocks[].block_id` as `target.content_block_id`;
- submit `target_type: "content_block"`;
- optionally pass `content_blocks[].metadata.source_hash` as stale-target evidence.

The UI must not derive block target identity from visible row position, pagination
page number, rendered block text, menu state, drawer state, or draft answer state.
`metadata.content_block_id`, when present, is only advisory duplication; the canonical
source for request `content_block_id` is `content_blocks[].block_id`.

Canonical eligibility rule:

- A visible content block may open a block-level interaction menu only when the UI can
  construct a backend-valid `content_block` target from response-owned ids.
- The UI may present the block-level menu when the selected task-unit content response
  provides both response-owned `task_unit_id` and `content_blocks[].block_id`.
- Backend `interaction_availability` and `artifact_target_refs` are confidence/status
  hints for the menu, not a hard display gate. Some `segmented=true` render payloads
  may omit those fields while still carrying enough target identity for a read-first
  backend validation attempt.
- If a block is render-only, the UI may omit the block menu or show a disabled
  affordance with local explanatory copy; it must not silently fall back to the parent
  task-unit target while visually implying a block-specific interaction.
- Failed `content_block_id not found` responses should be surfaced as target-invalid or
  stale-target UI states, not retried as title/position based targets.

The menu should be target-aware. Opening it records a frontend-local
`interactionTarget` object derived from the current row/group. Selecting a menu item
starts a read-first interaction flow for that target; it must not immediately generate,
refresh, prepare, reparse, or mutate task-layout.

#### Insight Inline Behavior

Insight is expected to be short enough to live inline beneath the target that requested
it. When the user selects `Insights`, the UI should insert an inline insight region
directly below the target row or group and indented one visual level deeper than that
target:

- book/document insight: below the document header or first hierarchy header area,
  indented relative to document controls
- chapter insight: below the chapter row, before its section rows
- section insight: below the section row, before the next sibling section
- unit insight: below the task-unit group header/content block group inside reader
  content

The inline region should be compact, dismissible/collapsible, and status-aware. It
should support:

- loading persisted insight
- generate once on an explicitly selected action after a valid not-generated read
- explicit retry after failure
- explicit refresh
- completed insight payload
- insufficient-content state
- stale-target state
- generation-failed state

The inline insight region must not push raw source text, child artifact payloads, or
task-unit content into `/documents/task-layout`. It may show bounded metadata such as
status, generated time, target level, and artifact reference summary when backend API
schemas expose those fields.

Only one inline insight per target should be open at a time. Multiple targets may keep
their own collapsed insight summaries locally, but frontend collapse/open state is not
backend truth and must not be persisted as artifact state.

#### Quiz Drawer Behavior

Quiz interaction needs more space than inline insight because it includes question
items, answer reveal, local answer inputs, generation/refresh controls, and validation
states. Selecting `Quiz` should open a right-side MUI drawer anchored to the current
reader workspace.

The drawer should display the current target breadcrumb, for example:

```text
Book > Chapter > Section > Unit
```

The quiz drawer should use read-first behavior:

- first load persisted quiz artifact for the target through the generic quiz read route
- automatically generate once when that explicit selection reads `not_generated`
- provide an explicit retry after a failed activation
- provide an explicit `Refresh quiz` action only when overwriting/regenerating is
  intended
- show completed quiz items with their type-specific UI
- show insufficient-content, stale-target, validation-failed, and generation-failed
  states distinctly

The drawer may support local answer practice before backend answer-submission exists.
Those local answers are UI state only. They must not be described as persisted quiz
results unless a backend endpoint explicitly supports saving or evaluating them.

Existing `/documents/section-quiz` and `/documents/chapter-quiz` routes are legacy
section/chapter quiz generation routes and should not be treated as the full
target-agnostic quiz interaction API. The new drawer should wait for generic
read/generate quiz routes before implementation, or use a clearly marked temporary
compatibility adapter if the maintainer explicitly chooses that bridge later.

#### Critical-Thinking Drawer Behavior

Critical thinking is a session-shaped interaction and should also use the right-side
drawer. It must not be rendered inline because the user needs room for prompt reading,
answer writing, evaluation feedback, and retry controls.

The first-version drawer flow is:

1. Read current/persisted critical-thinking session state for the target through the
   generic critical-thinking read route.
2. If a valid read reports missing, generate a question as part of the same explicit selection.
3. After question generation, show the single generated question and an answer input.
4. On submit, send one user answer for evaluation through the submit route.
5. If evaluation succeeds, show feedback, score, and suggested refinement.
6. If evaluation fails, preserve the submitted answer and expose `Retry evaluation`.

The UI should recognize these first-version statuses:

- `not_generated`
- `question_generated`
- `insufficient_content`
- `answer_submitted`
- `evaluation_failed`
- `completed`
- `stale_target`

Critical-thinking drawer state may cache typed answer text locally while the drawer is
open, but that draft is not backend truth. Closing the drawer should warn only if there
is unsent local answer text; it should not imply a backend session was created unless
the generation route already succeeded.

#### Shared Interaction State Model

The next UI slice should introduce a dedicated feature folder:

- `src/features/reading-interactions/model.ts`: target identity, status normalization,
  target breadcrumb helpers, inline insight placement helpers, menu action definitions,
  and drawer mode derivation
- `src/features/reading-interactions/controller.ts`: target menu state, inline insight
  state, drawer state, read-first request orchestration, explicit generate/refresh,
  critical-thinking submit/retry, and stale response suppression
- `src/features/reading-interactions/view.tsx`: target action menu, inline insight
  region, quiz drawer, critical-thinking drawer, compact status views
- `src/features/reading-interactions/index.ts`: public feature exports

REST usage is centralized in `ReadingInteractionService` rather than placed directly in
views. The first implementation covers insight read/generate/refresh and maps the
shared backend target/envelope contract into inline insight state. Drawer interactions
for quiz and critical-thinking should extend the same service/controller boundary
without using legacy section/chapter quiz routes as the generic drawer API.

#### Remaining Frontend API Wiring Plan

The frontend API wiring is intentionally split into three major steps:

1. Insight inline API wiring: completed. `ReadingInteractionService` supports
   insight read/generate/refresh, and the inline insight surface maps backend
   analysis envelopes into local view state.
2. Quiz drawer API wiring: in progress. Quiz request/response API types and
   `ReadingInteractionService` methods are implemented, and generic quiz responses
   can be mapped into drawer view state. Drawer open now performs read-first quiz
   loading followed by one generation only for missing output after explicit Quiz
   selection (UI-ACTIVATE-01); explicit retry/refresh wiring remains available.
   Critical-thinking flow remains separate.
3. Critical-thinking drawer API wiring: in progress. Request/response API types and
   `ReadingInteractionService` methods are implemented, and backend session responses
   can be mapped into drawer view state. Drawer open now performs read-first
   critical-thinking session loading followed by one question generation only for
   missing output after explicit selection (UI-ACTIVATE-01); submit/retry remain explicit. This remains separate from quiz
   wiring because it has a session lifecycle, draft-answer state, submit semantics,
   and retry semantics.

The remaining two major steps are split into smaller implementation tasks:

Quiz drawer API wiring:

- add quiz request/response API types and `ReadingInteractionService` methods for
  `readQuiz`, `generateQuiz`, and `refreshQuiz` (completed)
- map `QuizInteractionResponse` into `QuizViewState`, including strict item ids,
  item types, options, answers, explanations, metadata, error/reason fields, and
  status normalization (completed)
- wire drawer open to read-first quiz loading while preserving local practice answers
  and reveal toggles as frontend-only state (completed)
- wire explicit `Generate quiz` and `Refresh quiz` controls to service calls, keeping
  legacy `/documents/section-quiz` and `/documents/chapter-quiz` out of the drawer API
  (completed)

Critical-thinking drawer API wiring:

- add critical-thinking request/response API types and `ReadingInteractionService`
  methods for read, generate-question, submit-answer, and retry-evaluation (completed)
- map `CriticalThinkingSessionResponse` into `CriticalThinkingViewState`, including
  session id, question, submitted answer, evaluation feedback, score, suggested
  refinement, status, metadata, and retryable error state (completed)
- wire explicit selection to read-first critical-thinking activation; a separate
  generate-question command follows only a valid missing read (completed, UI-ACTIVATE-01)
- wire explicit generate-question, submit-answer, and retry-evaluation controls while
  keeping unsent draft answers frontend-local and sending the shared target object plus
  backend session id for submit/retry calls (completed)

Both remaining steps must keep stale response suppression in the controller, keep REST
calls out of view components, keep `/documents/task-layout` unchanged, and avoid
persisting drawer open state, local quiz practice answers, or unsent critical-thinking
draft text as backend truth.

#### Backend-Ready Content-Block Frontend Task Split

The backend now exposes enough `content_block` target support for a guarded frontend
integration. The implementation should be split into eight tasks:

1. Extend frontend interaction target identity to include `content_block`, including
   deterministic target keys, breadcrumbs, API target serialization, parent ids, and
   optional `source_hash` evidence.
2. Preserve response-owned content-block owner context in reader-content aggregation:
   each rendered block must retain its owning task-unit response id, optional parent
   chapter/section ids, block id, and optional source evidence.
3. Define and implement block eligibility in the reader-content model so only
   response-owned task-unit/block identities can start backend interaction flows.
4. Add `ContentBlockView` three-dot menu affordance for eligible blocks only, using
   the existing Insights / Quiz / Critical thinking action vocabulary and MUI menu
   behavior.
5. Route content-block menu actions through the existing reading-interaction
   controller so insight stays inline and quiz / critical-thinking continue to use the
   shared drawer without new route families.
6. Consume optional `interaction_availability` slots from task-unit content responses
   for menu/status hints without storing artifact payloads, source text, or UI state in
   task-layout.
7. Normalize content-block error states across inline insight and drawers, especially
   `stale_target`, `insufficient_content`, `generation_failed`, and resolver
   not-found responses.
8. Add frontend validation coverage and smoke checks proving document/chapter/section/
   task-unit flows still work, content-block flows use explicit backend ids, and
   `/documents/task-layout` remains lightweight.

These tasks are frontend implementation tasks only. They must not require backend API
shape changes, hidden prepare/reparse calls, task-layout mutation, persisted drawer
state, local quiz answer persistence, or unsent critical-thinking draft persistence.

Recommended frontend state:

- `interactionMenuTarget`: current target for the open three-dot menu
- `openInlineInsightsByTarget`: local map of target key to inline insight view state
- `interactionDrawer`: closed / quiz / critical-thinking with target key
- `interactionArtifactsByTarget`: local cache of read responses, keyed by target level
  and id
- `interactionRequestStatus`: read / generate / refresh / submit / retry status
- `draftCriticalThinkingAnswer`: local unsent answer text for the active session

All of this state is frontend-local projection/control state. It must not mutate
backend hierarchy, task-layout, profile diagnostics, parser metadata, or artifact
persistence except through explicit future interaction mutation routes.

#### Frontend Data Structures For Minimal API Exposure

The UI should define a small internal model before backend route wiring begins. This
model is a frontend contract for rendering and request orchestration; it is not a
claim that backend schemas already exist. Future backend APIs can use this model as a
guide for minimal exposure: return enough target identity, status, artifact metadata,
and interaction payload to render the feature, while keeping hierarchy ownership and
artifact persistence in the backend.

Recommended target model:

```ts
type ReadingInteractionTargetLevel =
  | "document"
  | "chapter"
  | "section"
  | "task_unit"
  | "content_block";

type ReadingInteractionKind =
  | "insight"
  | "quiz"
  | "critical_thinking";

interface ReadingInteractionTarget {
  targetLevel: ReadingInteractionTargetLevel;
  docName: string;
  documentId?: string;
  chapterId?: string;
  sectionId?: string;
  taskUnitId?: string;
  contentBlockId?: string;
  parentChapterId?: string;
  parentSectionId?: string;
  displayTitle: string;
  breadcrumb: string[];
  sourceStructureVersion?: string;
  sourceHash?: string;
}

type ReadingInteractionTargetKey = string;
```

`ReadingInteractionTargetKey` should be a deterministic frontend key derived from
`targetLevel` plus backend ids. It is only a cache/rendering key. It must not become a
backend identifier and must not be derived from mutable display title text alone.

Recommended status model:

```ts
type InteractionStatus =
  | "idle"
  | "loading"
  | "not_generated"
  | "generating"
  | "refreshing"
  | "submitting"
  | "retrying"
  | "completed"
  | "insufficient_content"
  | "stale_target"
  | "generation_failed"
  | "validation_failed"
  | "evaluation_failed";

interface InteractionMetadataView {
  artifactId?: string;
  sessionId?: string;
  generatedAt?: string;
  updatedAt?: string;
  targetLevel: ReadingInteractionTargetLevel;
  referencedArtifactSummary?: Array<{
    artifactId: string;
    artifactType: string;
    targetLevel: ReadingInteractionTargetLevel;
  }>;
}
```

Recommended view state model:

```ts
interface InsightViewState {
  targetKey: ReadingInteractionTargetKey;
  target: ReadingInteractionTarget;
  status: InteractionStatus;
  content?: string;
  metadata?: InteractionMetadataView;
  expanded: boolean;
  errorMessage?: string;
}

interface QuizItemView {
  itemId: string;
  itemType: "short_answer" | "multiple_choice" | "true_false";
  prompt: string;
  options?: string[];
  answer?: string;
  explanation?: string;
}

interface QuizViewState {
  targetKey: ReadingInteractionTargetKey;
  target: ReadingInteractionTarget;
  status: InteractionStatus;
  items: QuizItemView[];
  answersByItemId: Record<string, string>;
  revealByItemId: Record<string, boolean>;
  metadata?: InteractionMetadataView;
  errorMessage?: string;
}

interface CriticalThinkingViewState {
  targetKey: ReadingInteractionTargetKey;
  target: ReadingInteractionTarget;
  status: InteractionStatus;
  question?: string;
  draftAnswer: string;
  submittedAnswer?: string;
  evaluation?: {
    feedback: string;
    score?: number;
    suggestedRefinement?: string;
  };
  metadata?: InteractionMetadataView;
  errorMessage?: string;
}

interface InteractionDrawerState {
  kind: "quiz" | "critical_thinking" | null;
  targetKey?: ReadingInteractionTargetKey;
}
```

Recommended cache/request model:

```ts
interface InteractionArtifactCache {
  insightsByTarget: Record<ReadingInteractionTargetKey, InsightViewState>;
  quizzesByTarget: Record<ReadingInteractionTargetKey, QuizViewState>;
  criticalThinkingByTarget: Record<
    ReadingInteractionTargetKey,
    CriticalThinkingViewState
  >;
}

interface InteractionRequestState {
  targetKey: ReadingInteractionTargetKey;
  kind: ReadingInteractionKind;
  operation: "read" | "generate" | "refresh" | "submit" | "retry";
  status: "pending" | "succeeded" | "failed";
  requestId: string;
}
```

Backend API exposure should stay minimal and target-aware. The UI should need:

- a generic target envelope that echoes `targetLevel`, document identity, backend ids,
  and optional parent ids for consistency checks
- separate read, generate, refresh, submit, and retry operations so read paths never
  trigger LLM work
- normalized status and reason/error fields that map into `InteractionStatus`
- artifact/session metadata such as artifact id, session id, generated time, updated
  time, and backend-owned source version/hash when available
- bounded referenced-artifact summaries, not raw child artifact payloads
- quiz item type and prompt/option/answer/explanation fields needed for rendering
- critical-thinking question, submitted answer, evaluation feedback, optional score,
  and suggested refinement when the session reaches those states

The UI should not require backend APIs to expose raw source text, expanded child
content, full artifact dependency graphs, task-layout mutations, drawer state,
collapsed/expanded state, local quiz practice answers, or unsent critical-thinking
draft text.

#### Interaction API Boundary

Backend routes support separate read and write paths:

- read persisted interaction artifact/session
- generate or refresh insight
- generate or refresh quiz
- generate critical-thinking question session
- submit critical-thinking answer for evaluation
- retry failed critical-thinking evaluation

Read endpoints must remain read-only. Explicit selection of Insights, Summary, Quiz
or Critical thinking authorizes the read-or-generate activation described above.
Passive menu opening, document/section navigation and layout reads never generate.
Refresh, answer submission and evaluation retry remain separate explicit commands.

Frontend API wiring should only use backend-implemented public contracts. The current
`src/types/api.ts` reading-interaction types mirror the implemented insight analysis
target/envelope/payload schema used by `ReadingInteractionService`; quiz and
critical-thinking type/service expansion should be added with the same evidence before
their drawer controls call backend routes.

### TOC Editor Page Contract

The TOC editor exists to support documents where automatic table-of-contents recognition is missing, failed, or needs correction. It must not be treated as a visual-only tree editor; user edits must be validated before they can produce an explicit hard reparse.

The TOC editor has two draft source modes:

- `from scratch`: the default mode. It starts with an empty editable tree and is optimized for the common failure case where the original TOC is missing, unreadable, or not automatically recognized.
- `edit existing`: a secondary mode. It seeds the editable tree from the currently loaded task-layout hierarchy for the rarer case where an existing structure mostly works but needs correction.

Switching modes resets the in-memory editor draft and validation state. It must not mutate task-layout or backend hierarchy until the user completes frontend validation, backend validation, and explicit hard-reparse commit.

The right pane is a single integrated TOC editor, not a collection of permanent subpanels. Its main screen uses two coordinated work surfaces inside one editor:

- editable TOC tree: supports `chapter -> section` editing, including add, delete, rename, and reorder operations
- source anchor workspace: shows source context and lets the user assign anchors to the selected chapter/section item

`section` is optional in the UI editing experience. A chapter-only book is represented as the special case where every chapter contains exactly one section. For display, the UI may use the single section title as the chapter display name, matching the existing reader rendering convention. Persistence and backend submission must still preserve the backend hierarchy contract rather than introducing root `sections[]` or deeper hierarchy levels.

The TOC editor supports both `page_range` and `char_range` anchors. The anchor workspace is page-first when the loaded task-layout exposes reliable backend `anchor_evidence` with page boundaries, and remains character-range based when page evidence is unavailable or when the user explicitly switches to character anchors:

- pageable source with reliable page evidence: default to page-number entry and submit page-backed anchors through the manual-structure flow
- pageable source without reliable page evidence: show the parsed evidence gap and fall back to `char_range`
- non-pageable source: use `char_range`
- advanced/manual override: allow the user to choose `char_range` even when page entry is available

`edit existing` mode should seed each chapter/section with default anchors from the current parsed structure when that evidence exists. For pageable documents, the displayed defaults should be the parsed page numbers. For non-pageable documents, or when page evidence is unavailable, the displayed defaults should be the parsed `char_start` / `char_end` range. If the current layout does not expose reliable anchor evidence, the editor should leave the anchor empty and surface validation guidance rather than inventing positions.

`page_range` support depends on backend page-boundary mapping and manual-structure validation/commit support. The UI must not claim page anchors are authoritative when backend page evidence is missing; it must fall back to `char_range` or require explicit user override.

The TOC editor must use an independent feature folder:

- `src/features/toc-editor/model.ts`: editable TOC tree model, from-scratch draft construction, existing-layout draft construction, chapter-only normalization, anchor/range validation helpers, manual-structure request projection
- `src/features/toc-editor/controller.ts`: route guard, draft source mode, selected TOC item, dirty state, frontend validation, backend validation, and commit confirmation orchestration
- `src/features/toc-editor/view.tsx`: integrated TOC editor surface
- `src/features/toc-editor/index.ts`: public feature exports

### TOC Validation And Commit Flow

Manual TOC editing must use a three-stage flow:

1. Frontend validation:
   - enforce the two-level maximum (`chapter -> section`)
   - allow chapter-only as the one-section-per-chapter special case
   - require usable titles
   - require anchors for submitted items
   - validate `char_start < char_end` for `char_range`
   - validate page order and source page availability for future `page_range`
   - reject missing, overlapping, or out-of-order anchors where they would make projection ambiguous
2. Backend validation:
   - call the non-mutating manual-structure validation endpoint
   - display backend issues and preview evidence
   - do not persist hierarchy or mutate task-layout during validation
3. Commit / hard reparse:
   - require successful frontend and backend validation
   - use an explicit reparse commit route
   - show a MUI confirmation dialog before commit

Entering the TOC editor may show a low-disruption notice that submitting edits will trigger hard reparse. The destructive warning belongs at commit time: the confirmation dialog must clearly state that hard reparse replaces the current structure and derived QA, summaries, quiz artifacts, and similar generated outputs will not be preserved.

Validation results and hard-reparse warnings should use MUI popup/dialog/snackbar-style interactions rather than occupying permanent space in the main editor surface.

### API Boundary

Task layout:

```http
POST /documents/task-layout
Content-Type: application/json
```

```json
{
  "doc_name": "<doc_name>",
  "refresh_task_units": false,
  "task_unit_split_mode": "progressive"
}
```

Document-open policy:

- For an API-returned document candidate, the UI should treat document selection as an
  existing-layout read first.
- The preferred route for existing layout display is `POST /documents/task-layout`.
- `POST /documents/prepare-task-layout` should be reserved for first-time prepare,
  explicit repair/retry, or fallback when the read-centric layout route reports that the
  layout is missing or unavailable.
- The UI should not rely on prepare-then-read as a hidden way to repair cache,
  structured-document, language, profile, or OCR state during ordinary document open.
- This policy avoids surprising OCR/LLM-backed backend work during normal document
  selection while preserving backend hierarchy as the source of truth.

TOC edit page-default handshake:

- Ordinary reader entry must request task layout without page-evidence opt-in so normal
  document opening does not trigger backend source-evidence/OCR work.
- Entering TOC edit mode is the explicit UI intent that may request page-backed anchor
  defaults. Before rendering edit-existing page defaults, the UI should refresh the
  current layout through `POST /documents/task-layout` with
  `include_anchor_page_evidence: true`.
- The refreshed layout remains a lightweight task-layout projection. The UI consumes
  only `anchor_evidence` metadata on chapter/section nodes and must not request or store
  raw text, page text, OCR geometry, or content blocks through task-layout.
- `edit existing` mode may prefill page inputs only when backend `anchor_evidence`
  reports available `page_range` evidence. Displayed page defaults are 1-based UI page
  numbers derived from zero-based backend page indices; manual-structure submissions
  must convert them back to zero-based `page_range` indices.
- If the opt-in layout refresh fails or returns unavailable page evidence, the TOC
  editor should keep page defaults empty and fall back to `char_range` guidance rather
  than inventing page positions.
- This handshake is read-only until the user explicitly validates and commits a
  manual-structure hard reparse.

Task-unit content, issued once per selected section:

```http
POST /documents/{doc_name}/task-units/content
```

```json
{
  "task_unit_ids": ["<task_unit_id>"],
  "segmented": true,
  "include_raw_content": false
}
```

The frontend requests the selected section's task units in backend layout order with one batch content request and renders the flattened `content_blocks` in that same order. It does not request duplicated raw content by default.

Single task-unit content remains the compatibility/fallback endpoint:

```http
GET /documents/{doc_name}/task-units/{task_unit_id}/content?segmented=true
```

- The batch request preserves current render options, especially `segmented=true` and default raw-content suppression.
- The batch response preserves per-task-unit content ordering and reuses the backend `TaskUnitContentResponse` shape so reader aggregation logic remains content-block-first.
- Batch content must remain an on-demand content API path. It must not add task-unit content or `content_blocks` to `/documents/task-layout`.
- The frontend must not reinterpret batch response order as hierarchy truth, persist content as backend state, or trigger hidden prepare/reparse/profile mutation from section selection.

### Loading / Error / Empty Behavior

- Initial state shows no hierarchy until a document is loaded.
- Layout loading disables the document search control and marks the navigation area busy.
- Layout error shows the backend error detail and keeps the user on the document input.
- Empty layout shows a no-content state if the backend returns no chapters, no sections, or no task units.
- Selecting a section starts content loading for that section's task units.
- Content error is scoped to the reader panel.
- Empty content shows a no-content-blocks state when `content_blocks` is empty.

### Error Notification Policy

The UI consumes several backend error payload shapes, including FastAPI `detail`,
operation-level `error`, task `reason`, string `errors[]`, and structured validation
`errors[]` issue objects. Backend API normalization is out of scope for this UI module
slice, so the frontend should normalize these payloads at the REST/client boundary for
display only.

Error presentation should avoid consuming permanent main-screen workspace, especially in
the TOC editor where source range editing has limited space. Runtime API failures,
validation summaries, and commit failures should use dismissible MUI popup interactions
such as `Snackbar` + `Alert`. Destructive hard-reparse confirmation remains a `Dialog`.

Main panes may keep compact state placeholders when the primary workflow cannot
continue, but detailed backend or validation text should be surfaced through the
dismissible notification layer. Validation issue lists may be opened from a popup or
dialog when multiple issues need inspection, instead of being permanently rendered in
the range workspace.

## Frontend Technical Structure

- Framework: React with TypeScript.
- UI library: MUI for app shell, combo box, buttons, navigation controls, chips, loading indicators, and alerts.
- Build tooling: Vite with `@vitejs/plugin-react`.
- Dev server: Vite dev server with `/api/*` proxy to the backend target from `DEEP_READER_API_URL` or `http://localhost:8000`.
- Routing strategy: React Router with document-scoped reader and TOC editor routes as described above.
- State approach: component-local React state in `src/App.tsx`; no external state library.
- REST service organization: `src/services/RestClient.ts` owns shared request/error handling; endpoint groups are exposed through typed service classes.
- Type organization: `src/types/api.ts` mirrors the consumed backend response fields.
- Feature organization: `src/features/book-search/`, `src/features/hierarchy-navigation/`, and `src/features/reader-content/` own MVC-style model/controller/view files.
- Shared component organization: `src/shared/components/StateView.tsx` owns cross-feature loading/error/empty state rendering.
- Styling approach: MUI theme in `src/theme.ts` plus scoped layout CSS in `src/styles.css`.
- Testing approach: TypeScript validation and production build through `npm run check`, HTTP smoke through the Vite dev server, and manual browser/API validation against the backend.

### REST Service Classes

Backend REST usage is centralized behind service classes:

- `DocumentCatalogService`: `GET /documents`
- `DocumentManagementService`: current read-only document-management boundary over `GET /documents`; maps backend-owned `doc_name`, title, and source into the future file-management item model without inventing source hash, preparation status, profile-derived display fields, upload metadata, lifecycle state, or reading history
- `TaskLayoutService`: `POST /documents/task-layout`; `POST /documents/prepare-task-layout`
- `TaskUnitContentService`: `POST /documents/{doc_name}/task-units/content` for selected-section batch read; `GET /documents/{doc_name}/task-units/{task_unit_id}/content?segmented=true` as single-task compatibility/fallback
- `StructureRepairService`: `POST /documents/reparse-structure` for common / LLM structure repair
- `ManualStructureService`: `POST /documents/manual-structure/validate`; `POST /documents/reparse-structure` with `parser_mode=manual_structure`
- `DocumentPreparationService`: `POST /documents/prepare`

These services preserve the existing backend API contract and keep manual-structure validation/commit explicit. `DocumentManagementService` is intentionally narrow until backend management/upload contracts exist; it does not call upload, delete, archive, prepare, reparse, task-layout, OCR, profile, retrieval, or reading-interaction routes.

### File Management Dialog

The current file-management implementation is a read-only document library dialog. The
topbar `Library` action opens the dialog and loads document candidates through
`DocumentManagementService`, which currently adapts the existing lightweight
`GET /documents` API. The dialog owns frontend-local loading, empty, error, and loaded
states and renders row-level `Open` actions.

The visible library table should not treat backend `source` as a primary user-facing
column. Source is operational metadata and is usually less useful to readers than
document identity and reading-oriented metadata. The default list should prefer display
fields such as document title, author/work identity when available, document language,
profile-derived genre/type/topic signals, preparation/readiness status, and a compact
profile summary or description. Until backend management/profile APIs expose those
fields, the UI may show available title/doc name plus preparation status and keep source
metadata hidden behind details or an advanced filter. The UI must not invent profile
metadata when the backend has not provided it.

The `Open` action is intentionally read-centric: it calls `POST /documents/task-layout`
for the selected backend document candidate and does not fall back to
`/documents/prepare-task-layout`. If no current layout exists, the user sees the backend
error and can use explicit preparation/repair flows when those are available. This keeps
file-management opening separate from upload, prepare, reparse, OCR, profile, retrieval,
and reading-interaction generation.

The dialog now exposes upload controls as an embedded explicit backend subflow. It
supports file selection, advisory `.txt` / `.pdf` extension validation, selected,
validating, uploading, uploaded, failed, cancelled, and duplicate-conflict confirmation
states. The upload action calls `POST /documents/upload` through
`DocumentUploadService`; upload success refreshes the catalog and offers explicit
Prepare/Open/Upload another actions.

Duplicate upload handling is two-step and user-confirmed. The first upload attempt uses
`conflict_policy=reject`; when the backend returns `duplicate_conflict`, the UI opens a
confirmation dialog. Only after the user confirms does the UI submit the same selected
file again with `conflict_policy=create_distinct`, allowing the backend to save a unique
source filename and backend-owned `doc_name` such as `Book (1)`. The UI must not
silently replace an existing source and must not invent the unique name locally.

This upload subflow stores raw source candidates only through the explicit backend
upload route. It must not prepare/reparse documents, run OCR/profile/retrieval, call
task-layout, or generate interactions as a hidden side effect. Backend validation remains
authoritative for supported type, safety, size, conflict policy, and final document
identity.

Boundary validation confirms the upload controls are explicit state:
`useFileManagementController` lists documents through `DocumentManagementService`,
records selected file metadata in memory, performs advisory extension validation, calls
`DocumentUploadService` for upload only after the user action, and handles duplicate
confirmation before requesting `create_distinct`. The document library `Open` action
still uses the read-centric task-layout path rather than hidden prepare or upload side
effects.

Library usability should include an advanced filter area for large future collections.
The basic search should remain simple, while advanced filters can later include
preparation/readiness status, language, profile-derived type/genre/topic, upload or
updated time, source type, and lifecycle state once those fields are exposed by the
backend. Filter state is frontend-local and must not become backend truth.

The library should also expose a recent-reading shortcut area below the main list that
shows the three most recently opened/read documents. The first implementation may track
recent entries locally from successful reader opens. If a future backend exposes
server-side reading history or `last_read_at`, the UI can replace or reconcile the
local list with that backend-owned history. Clicking a recent book should enter the
same read-centric document open path as the normal row `Open` action.

### Document Combo Box

The first slice consumes the backend document list API for document-name candidates. It still does not implement a full document library UI.

`DocumentSearch` uses MUI `Autocomplete` and consumes the backend `GET /documents` list/search API:

- opening the combo box calls `GET /documents?limit=200`
- users may type to filter the loaded API options locally
- the loaded document must be selected from API-returned options, and selection immediately starts the layout load flow
- the UI does not issue per-keystroke search requests
- successful document names are not added as local-only options

This control remains a document-name entry surface, not a document library UI. It does not create local-only selectable documents.

## Future Direction: File Management And Upload UI

The Reader UI should introduce a file-management module as the parent surface for
document inventory, document status, and explicit document lifecycle actions. File
upload is a child workflow inside file management, not a separate primary app mode.
This keeps the user's mental model simple: the user manages documents in one place,
and upload is one way to add a document candidate to that managed library.

The file-management surface should be planned as a document library rather than as an
extension of the current document combo box. The current combo box remains a compact
reader entry control backed by `GET /documents`; the future file-management module may
reuse the same lightweight discovery contract initially, but it should add richer UI
state around source status, preparation readiness, explicit prepare/open actions,
upload progress, and failure recovery only after backend contracts exist.

Planned feature boundaries:

- `src/features/file-management/` owns the document library view, filters, status
  presentation, row actions, recent-reading shortcuts, and composition of upload as an
  embedded workflow.
- `src/features/file-upload/` may exist as an internal sub-feature only if upload
  state becomes complex enough to justify isolation; it should still be rendered from
  file management.
- `src/services/DocumentManagementService.ts` should own future management REST calls
  such as list/status/delete/rename when backend routes exist.
- `src/services/DocumentUploadService.ts` should own future upload REST calls when
  backend upload routes exist.
- Existing reader services must remain focused on reading: `TaskLayoutService` reads
  layout, `TaskUnitContentService` reads content, and `DocumentPreparationService`
  explicitly prepares a selected document.

File-management UI state should distinguish:

- catalog loading, loaded, empty, and failed states
- per-document reader-facing display metadata, preparation status, and last error when
  backend exposes those fields
- advanced filter state for future large-library use
- recent-reading shortcuts derived from successful reader opens or future backend
  reading-history metadata
- upload idle, selected, validating, uploading, uploaded, failed, and cancelled states
- duplicate upload conflict state requiring confirmation before creating a distinct
  backend-owned source name
- explicit post-upload actions such as prepare, open existing layout, or retry upload

The upload flow should be explicit and non-surprising:

1. The user opens file management.
2. The user chooses one or more supported files.
3. The UI performs advisory client-side checks such as extension and size, while the
   backend remains validation authority.
4. The upload call stores or registers the raw source and returns a lightweight
   document candidate/status response.
5. If the backend reports a duplicate conflict, the UI asks the user for confirmation
   before retrying with the backend `create_distinct` policy.
6. After a confirmed distinct upload, the backend returns the final unique source
   filename and `docName`; the UI displays those backend-owned values.
7. The UI refreshes the document catalog and offers explicit next actions.
8. Upload success must not silently call prepare, reparse, task-layout, profile
   generation, OCR, or interaction generation.

The future UI data shape should be small and backend-owned:

```ts
type DocumentManagementItem = {
  docName: string;
  title?: string | null;
  author?: string | null;
  language?: string | null;
  profileLabel?: string | null;
  profileSummary?: string | null;
  sourceFilename?: string | null;
  sourceType?: "txt" | "pdf" | "unknown";
  sourceSizeBytes?: number | null;
  sourceHash?: string | null;
  sourceStatus?: "available" | "missing" | "unsupported" | "failed";
  preparationStatus?: "not_prepared" | "ready" | "failed" | "stale" | "unknown";
  uploadedAt?: string | null;
  updatedAt?: string | null;
  lastReadAt?: string | null;
  lastError?: string | null;
};
```

This frontend shape is a planning target only. It must not be treated as an implemented
backend schema until `api_schemas.py`, routes, and app orchestration expose matching
contracts.

Boundary rules:

- Backend document identity remains authoritative.
- Upload creates or updates a raw source candidate only through explicit backend upload
  routes when they exist.
- Same-name upload must first fail/ask through the `reject` conflict path, then use
  `create_distinct` only after explicit user confirmation; unique naming is backend-owned.
- Opening a document still uses the read-centric task-layout flow when a layout exists.
- Preparing or repairing a document remains an explicit user action.
- File-management filters, sort order, selected rows, drawer state, upload progress,
  recent-reading shortcuts, and optimistic UI rows are frontend state only unless a
  backend history/profile API explicitly owns the corresponding fields.
- The UI must not invent prepared status, source hash, structure version, or backend
  lifecycle state.
- The UI must not invent document profile fields. When profile-derived fields are
  unavailable, it should show neutral fallback display such as title/doc name rather
  than fabricating author, genre, language, or summary values.
- The current lifecycle MVP should expose only delete and rename once backend routes
  exist; archive/restore should not appear in the UI until explicitly reprioritized.
- The UI must not hide destructive actions behind row selection. Delete requires an
  explicit user confirmation step before the frontend calls the backend delete route.
- After `status=accepted` and `deleted=true`, file management removes the deleted
  `docName` from its rows and recent-reading shortcuts, notifies book search through
  the App callback, then reloads the Library catalog. Book search removes the matching
  candidate and clears the input only if it matches that name. Failed/rejected delete
  responses do not trigger this invalidation. **[Code-Confirmed]**
- Both list controllers use request sequence checks. Accepted deletion invalidates
  older requests so their late successes or failures cannot restore stale entries or
  overwrite newer list state. These are frontend caches, not backend deletion authority.
  The already-rendered reader content is not automatically closed by this change.
  **[Code-Confirmed]**
- Rename should be a visible row action or dialog flow that submits `new_doc_name` to the
  backend and displays backend conflict/failure reasons without inventing local backend
  identity state.

## Library Entry Investigation And Planned Correction (2026-10-15)

The investigation below is the pre-fix baseline. The automatic post-upload preparation decision
below supersedes the earlier manual-only post-upload direction for future work;
it is NOT implemented. Backend upload remains raw-source intake only. No new
hidden prepare/reparse or write-on-read behavior is authorized.

Confirmed source/log evidence:

- `App.openManagedDocument` assigns `docName` and loads layout, but does not populate
  book-search options. Library and book-search controllers load independent lists.
- `BookSearchView` uses that same string as both reader identity and search input.
  If absent from search options, its Autocomplete `value` becomes null while
  `inputValue` still contains the name. Installed MUI `useAutocomplete` resets an
  unfocused non-freeSolo input to the selected label (empty for null), emitting
  `onInputChange(..., "reset")`. The view forwards every reason to `setDocName`.
  This explains why Library opens can clear the identity while selecting a loaded
  combo-box option works. This is a source-traced causal chain, not a completed
  browser reproduction; browser event-sequence verification remains LIB-05.
- Layout response handling uses its captured requested name and may still install
  a valid layout after the shared input has cleared. Reader content then uses the
  current empty `docName`; `TaskUnitContentService` has no empty-name validation.
  Logs contain repeated POST `/documents//task-units/content` 404s, e.g. request
  `0ce3668362b34cca805c82a027496f74`, and after the 24.8-second request
  `a58eff86b34d49f9bfb13d82b823137d`.
- This is the batch content route: task-unit IDs belong in `task_unit_ids` in the
  JSON body. The missing URL segment is the document name; there is no evidence
  from this URL alone that task-unit IDs were lost too.
- `selectSection` catches failed requests. Interaction target builders return null
  for an empty document name. Thus identity loss explains the apparent inactive
  UI without assuming an uncaught exception or a dead JavaScript runtime.
- Post-upload, row and recent-item Open controls have no readiness gate. Upload
  only sets `prepareStatus=initial`; preparation is manual and lacks request
  generations/abort handling. Success/failure may therefore race with replacement
  upload state. `DocumentManagementService` still calls `/documents` and drops
  readiness fields, although backend `/documents/management` already exposes
  `preparation_status`. That status means structured availability, not proof of a
  valid persisted task-unit layout.

Planned state/behavior contract:

1. Separate `searchInput` from committed `activeDocumentName` (backend `doc_name`,
   not `document_id` or title). All content/interaction requests use the committed
   identity paired with the accepted layout and navigation request generation.
   Library, recent items, search and route restoration share identity handling;
   absent/truncated search options cannot invalidate a known backend identity.
   Autocomplete reset/blur events must not clear the active reader document.
2. Validate nonempty document names and task-unit IDs before requests. Reject
   stale layout/section combinations and ignore responses from previous document
   generations. Show recoverable content errors; retry/switching documents must
   work after 404 without a reload or dummy IDs. Do not synthesize hierarchy IDs.
3. After upload success, use the returned final unique `doc_name` to start one
   visible preparation workflow. Suggested local states: queued, preparing,
   ready, failed, cancelled; retain a per-operation token and AbortController.
   Closing/reopening a dialog or rerendering must not resubmit. Duplicate-name
   confirmation still precedes distinct upload; no prepare on rejected upload.
4. Explicit preparation must cover readiness for reading, not just a raw upload.
   Prefer the existing explicit `prepare-task-layout` orchestration for this visible
   workflow (base preparation plus task-unit layout); preserve `common`, non-force
   defaults. LAYOUT-02 owns separating its write work from the read route. Avoid
   blindly chaining duplicate base prepares. Do not enable Open on a 200 response
   with failed flags, or on an unvalidated/incomplete layout.
5. Cancel before dispatch prevents submission; cancel after dispatch aborts client
   waiting where supported and invalidates late UI results. It does not promise
   server cancellation/rollback or refunded LLM/OCR costs. No cancellation/job API
   currently exists. Prevent blind overlapping retry; reconcile backend readiness
   before retry/resume. Never auto-open on late completion after cancellation.
6. Connect the existing management endpoint and preserve its actual status fields.
   All Open entry points share a readiness/identity guard. Unknown, missing,
   preparing, failed and cancelled states cannot bypass the guard. Structured
   `ready` can proceed to a read-only layout validation; a missing/stale layout
   directs the user back to explicit prepare, not implicit repair. Existing
   prepared documents do not rerun preparation merely because a modal opens.

Execution order is maintained in the checklist's dated Execution Order table:
LIB-01/02 first, backend mode contract/fast/quality/OCR/reuse/propagation next,
LAYOUT-01..03, then LIB-06/03/04/05. The 2026-10-16 backend routing dependency
adjustment places PDF-AUTO-01 before the final PDF-FIX-03 rerun.
This supersedes the earlier numeric LIB order and respects backend dependencies.
Backend dependencies now include main PDF-MODE-01, loader PDF-MODE-02/03 and
PDF-FIX-02/03, preparation PDF-MODE-04/05, and app LAYOUT-01..03.
PDF-AUTO-01 replaces PDF-FIX-01 as deferred auto-only improvement. Combined total:
17 tasks (6 UI, 11 backend), not eleven as in the superseded plan; current completion
is recorded only in the owning checklists.
Older checked tasks remain historical delivery evidence, not proof these bugs
are fixed. The OCR-01..08 track is not reopened or inflated by this follow-up.

### LIB-01 Delivered Identity Isolation (2026-10-15)

`App` now owns the committed reader `docName` and synchronous active identity ref;
book-search owns only `searchInput` and catalog options. Autocomplete input/clear
events affect the draft only; reset/blur cannot mutate reader identity. A committed
name is included in display options if absent from the capped/stale catalog, without
inventing backend catalog entries or persisting frontend state as hierarchy truth.

Library rows/recent/upload-result Open and search selection share one layout load;
route reload and browser history use the same loader. Each navigation invalidates
older layout/content requests, clears the old hierarchy/content, and closes the
selected interaction. Layout installation is generation-guarded; repair/TOC reload
paths cannot install another document's late layout. Deleting the active document
clears its reader state and invalidates pending layout work. Requests use committed
identity, never search text. Search/Open errors no longer implicitly call prepare;
explicit preparation remains available separately, with readiness improvements pending.

Evidence: `scripts/test-library-identity.mjs` uses Playwright with all `/api/**`
traffic mocked, covering empty/stale catalogs, Library-first open, typing/clear/
selection, slow A then fast B, reload and Back. Asserted complete document URL and
matching task-unit IDs; no page errors/unexpected API calls. Run against local Vite
with `node scripts/test-library-identity.mjs`; optional `PLAYWRIGHT_MODULE`,
`CHROMIUM_PATH`, `UI_URL` select an existing test runtime/browser/server. This is
identity regression evidence, not the full LIB-05 workflow or live-backend acceptance.
`npm run check` passes (existing bundle-size advisory remains). This evidence closes LIB-01 only.

### LIB-02 Delivered Request Guards And Recovery (2026-10-15)

App retains the requested `doc_name` together with each accepted layout snapshot;
content loading receives that scope, not editable search text or opaque document_id.
`shared/readerIdentity.ts` validates nonempty opaque IDs and current hierarchy
membership without synthesizing identifiers. Section selection must reference the
accepted chapter/section/task-unit objects; copied/previous snapshots cannot be
combined with the active document even if their IDs happen to match.

TaskUnitContentService rejects blank/duplicate IDs before transport, skips an empty
valid batch, and validates returned task-unit identity/order. The content controller
also rejects conflicting optional document/chapter/section response metadata.
Request generation and captured document/layout scope guard both successes and
errors. A new section/document invalidates old results; HTTP/network/malformed
response errors remain recoverable, with an explicit Retry content icon. Repeated
selection and navigation remain usable; retry does not prepare or generate anything.

Before reading/generating interactions, App checks the target against the accepted
current hierarchy; block targets must also be present in loaded content. Each
interaction service method validates required identities before transport. An
ordinary content failure does not disable valid chapter/section/document targets.
Server hierarchy remains authoritative; UI checks do not replace backend validation.

Evidence: `scripts/test-reader-guards.mjs` runs with all API calls intercepted and
tests 42 invalid-identity/stale-selection cases with zero transport, HTTP 404/network
failure retries, wrong-unit/document responses, delayed section/document results,
and an insight read after content errors. It uses the same optional Playwright
environment variables as `test-library-identity.mjs`. These focused tests and
`npm run check` validate LIB-02, not live-backend integration or the pending LIB-05
upload/readiness/cancellation matrix. PDF mode single-dispatch/independent-executor
acceptance is in loader design section 18.1; frontend delivery is recorded below.

### Planned PDF Mode Controls (2026-10-15)

Original approved design (implementation status follows below): PDF uploads expose
a single-choice fast/OCR/auto control before submission, default fast. Non-PDF
uploads hide this control. `pdf_extraction_mode` is distinct from the existing
base/common preparation/parser settings. Fast means native-only extraction with
no automatic OCR escalation; OCR means full-page recognition; auto retains the
existing conservative selective behavior for mixed/unknown sources.

LIB-06 snapshots the mode alongside an operation generation before upload; after
success LIB-03 uses the backend-returned unique name and that snapshot for one
explicit preparation. Upload itself remains raw-only. Do not expose enabled mode
choices against a backend that ignores the field. Pending operations cannot be
retargeted by changing the selection, dialog close/reopen or duplicate-name flow.
New upload defaults fast; existing-document reprepare may initialize from known
backend provenance, otherwise fast with no claim about the prior extraction mode.

Fast empty-text failures offer an explicit OCR/auto retry, never automatic costly
fallback. Empty-page warnings do not certify complete text coverage. Existing PDFs
can be explicitly re-prepared with a chosen mode; backend owns rebuild/invalidation
and truthful readiness. Cancelling stops waiting/late UI updates, not server work;
reconcile before retry to avoid overlapping prepares. Failed replacement preserves
previous valid backend artifacts; opening those requires verified readiness, not
optimistic local success. All Open gates remain shared. Stage displays use only
available backend evidence, not fabricated percentage progress or job APIs.

### Delivered PDF Mode Controls (2026-10-16, LIB-06)

PDF drafts expose Fast/OCR/Auto, defaulting to Fast; TXT drafts omit the control
and the preparation mode field. Upload snapshots mode locally with its generation
and retains it with the backend-returned unique doc_name after duplicate confirmation.
Upload itself sends only raw data. The current explicit Prepare action forwards
`pdf_extraction_mode` through `DocumentPreparationService` and `TaskLayoutService`
to `/documents/prepare-task-layout`, without force rebuild. Automatic post-upload
dispatch/cancellation remains LIB-03, not part of this delivery.

Library listing now reads `/documents/management` for source type, availability
and preparation status; search still uses `/documents`. Available PDF rows expose
an explicit Prepare PDF dialog. Each new dialog defaults to Fast without claiming
that it reflects historical extraction provenance. Pending upload/preparation locks
mode changes, competing preparation and rename/delete actions. Duplicate confirmation
also locks the upload selection. Closing/reopening Library retains the pending mode;
it does not cancel or submit another request. Draft cancellation invalidates upload
generation; in-flight cancellation remains a separate future task.

Preparation validates returned layout shape and matching requested/effective mode
with applied/reused provenance. Fast failures leave retry under explicit user control;
no OCR/auto escalation occurs. Failed existing-PDF preparation reads task-layout
without mutation before offering Open for surviving valid data. Settled preparation
of the active document triggers a read-only reader reload to discard stale unit IDs.
Management ready is displayed as Structure prepared, not certified reader readiness.
Full shared Open gates remain LIB-04; this task does not certify other Open entries.

Evidence: `scripts/test-library-pdf-modes.mjs` passed with intercepted APIs at 1280px
and 390px; screenshots inspected for mode control/dialog fit. Coverage includes fast
default, duplicate unique name, explicit OCR retry, auto pending lock and close/reopen,
TXT omission, and failed reprepare with read-only old-layout reconciliation. Existing
`test-library-identity.mjs`, `test-reader-guards.mjs` and `npm run check` also passed.
These are browser contract tests, not live OCR/LLM runs. UI follow-up is 3/6 (50%);
combined execution index at LIB-06 delivery was 12/17 (70.6%). LIB-03 delivery follows.

### Delivered Upload Preparation And Cancellation (2026-10-16, LIB-03)

This supersedes LIB-06's manual post-upload dispatch limitation, not its mode
contract. Accepted upload now queues exactly one visible, non-force
`/documents/prepare-task-layout` request using the backend-returned unique name
and captured mode. Rejected/duplicate-conflict uploads never queue preparation;
confirmed distinct upload does so only after acceptance. Upload remains a raw-only
endpoint; neither catalog loading nor Open implicitly prepares documents.

`FileUploadResult.prepareStatus` adds queued/cancelled to loading/success/error
(visible preparing/ready/failed states). A synchronous busy guard and unique
operation object prevent duplicate dispatch. The operation retains its snapshot,
queue timer and AbortController; only the current operation may change result
state or release its busy guard. Dialog close/reopen does not resubmit. The
controller cleanup clears queued timers, aborts waiting and invalidates results.
Upload-result Open is disabled until preparation returns a validated layout or
an explicit reconciliation confirms readable layout. Nothing auto-opens.

Cancel before dispatch clears the queue; explicit Prepare may subsequently retry.
Cancel after dispatch aborts the fetch and invalidates its operation, but reports
that server preparation may continue. Network loss is also treated as uncertain.
Such documents retain a mounted-controller uncertainty marker: Check readiness
performs only `/documents/task-layout`. If unreadable, it stays cancelled/unknown
without submitting another expensive request; if readable, Open becomes available.
Mode is locked while reconciliation is required. An existing-PDF prepare dialog
for the same uncertain name also reconciles instead of bypassing this guard.
A replacement upload has its own snapshot and cannot be changed by an old result.

There is no backend job-status or cancellation API. Therefore a negative readiness
read cannot establish that server work has stopped, and this workflow deliberately
does not offer a blind replacement prepare while uncertainty remains. The marker
is not persisted across page reloads and is not server truth. Full readiness gates
for rows/recent shortcuts remain LIB-04; automatic cancellation during raw upload
and backend rollback are not claimed. HTTP preparation failures remain explicit
retry states; Fast never automatically escalates to OCR/Auto.

Evidence: `scripts/test-library-auto-prepare.mjs` (intercepted APIs, controlled
timers, 1280px/390px) covers queue cancellation, actual fetch abort, unique names,
mode snapshots, duplicate dispatch guard, read-only reconciliation, replacement
uploads, late results and dialog reopen. Pending/ready screenshots inspected.
Updated `test-library-pdf-modes.mjs`, `test-library-identity.mjs`, existing
`test-reader-guards.mjs` and `npm run check` passed. These do not claim live OCR/LLM
acceptance. UI follow-up is 4/6 (66.7%); combined index is 13/17 (76.5%).
LIB-04 delivery follows.

### Delivered Shared Library Open Gates (2026-10-16, LIB-04)

All Library entry points (rows, recent shortcuts, upload result and prepare-dialog
Open) use controller `canOpenDocument` plus `validateDocumentOpen`; App navigates
only after validation. Management `preparation_status=ready` permits a read check,
not an assumption that task-unit layout exists. Missing/unknown/stale/failed status
does not enable Open. A validated preparation or explicit reconciliation may provide
fresh backend evidence for the current result/dialog even before catalog refresh.
Recent shortcuts are labels/identities only; their historical opening is not readiness.
A recent identity absent from the current catalog has no status-based permission.

Pending upload/preparation and lifecycle operations block Open. Local cancelled or
failed preparation records survive discarding the upload panel within the mounted
controller, so a catalog row cannot bypass them with an older structured-ready flag.
Explicit preparation/reconciliation clears a block only on backend-readable evidence;
failed replacement may still expose previously valid data after read-only confirmation.
This updates matching upload-result state too, avoiding conflicting entry-point status.
Deleted/renamed old names are blocked; renamed recent shortcuts are removed pending
an explicit opening of the new backend identity. None of these local guards is
persisted hierarchy truth or a claim about server job status.

Open performs `/documents/task-layout` with refresh/evidence disabled. A malformed
layout, HTTP failure or network failure keeps Library open, reports unavailable
readiness, and never invokes prepare. Where raw source is available, the prepare
dialog is offered for a separate explicit action. TXT now has a Prepare document
entry with no PDF selector/mode field; PDF retains its existing mode control.
Unavailable raw sources do not imply that existing prepared layouts are unreadable;
structured-ready entries still undergo the same read check.

Each pending Open has an AbortController identity. Closing Library or prepare dialog,
refreshing catalog, starting preparation/lifecycle/upload work or changing drafts
invalidates it. Only the active check may resolve; App additionally compares the
reader navigation generation before navigating. The normal reader then loads its
own fresh layout using existing generation guards; no readiness result is installed
as backend truth. Repeated clicks during checking are disabled. Catalog reads stay
read-only and do not probe/prepare every row.

At narrow widths, Library rows use two text columns with a full-width wrapping
action row, preserving access to Open/Prepare without a clipped action column.
Evidence: mocked-API `test-library-readiness.mjs` passed at 1280px/390px and screenshots
were inspected. Coverage includes readiness statuses, recent shortcuts, close during
read check, missing/malformed layout, explicit TXT prepare and cancelled-upload row
bypass prevention. Existing identity, PDF mode, automatic preparation and reader
guard suites passed, as did `npm run check` and `git diff --check`. No live OCR/LLM
run is claimed. UI follow-up is 5/6 (83.3%); combined index is 14/17 (82.4%).
LIB-05 acceptance follows.

### Library Browser Acceptance (2026-10-16, LIB-05)

Run `npm run test:library` in `Deep_Reader_UI` against the local Vite server
(default `http://127.0.0.1:5173`). Optional `UI_URL`, `PLAYWRIGHT_MODULE` and
`CHROMIUM_PATH` select the server and an installed Playwright/Chromium runtime.
`scripts/test-library-suite.mjs` runs six suites sequentially, fails immediately
on any nonzero exit, and limits each child suite to 180 seconds. It does not start
or mutate the backend. All `/api/**` calls are intercepted; unexpected routes fail
the test rather than reaching real documents, preparation or interaction generation.

| Suite | Verified coverage |
|---|---|
| `test-library-identity.mjs` | Cold Library entry, empty/stale search options, prepared rows, typing/clear, delayed A/B navigation, reload/Back, URL and content batch identity, explicit prepare failure/retry |
| `test-library-pdf-modes.mjs` | Fast default, duplicate-confirmed unique name, explicit OCR retry after no-text, Auto snapshot, pending mode lock/close/reopen, TXT omission, failed reprepare with surviving layout |
| `test-library-auto-prepare.mjs` | Queue cancellation before dispatch, actual fetch abort after dispatch, double submission guard, readiness-only reconciliation, replacement upload isolation, late completion and unchanged navigation pathname |
| `test-library-readiness.mjs` | Status gates across rows/recent/upload result/dialog, missing/malformed layout, explicit TXT prepare, close during read check, no cancelled-upload row bypass |
| `test-library-workflow.mjs` | Duplicate confirmation through auto-prepare and reader content, Unicode unique-name URL plus exact content batch body, 404/retry then all three interaction read menus, preparation-time lifecycle locks, confirmed rename/delete while readiness is pending and no stale navigation |
| `test-reader-guards.mjs` | 42 rejected identity cases without transport, 404/network/mismatched-content recovery, stale section/document response isolation and interaction recovery |

The mode, auto-prepare, readiness and complete-workflow suites run at both 1280px
and 390px. Workflow fixtures use two sections to exercise section menus explicitly;
the single-section chapter-merged presentation is not changed. Page errors (including
unhandled promise rejections) and unexpected API calls must remain empty. Interaction
menu acceptance verifies read-first behavior and correct target identity, not LLM
generation/evaluation quality. Lifecycle mutation fixtures are intercepted only.

Validation: all six scripts passed together, `npm run check` passed (existing
bundle-size advisory remains), and `git diff --check` passed. Desktop/mobile workflow
screenshots in `/tmp/library-workflow-{1280,390}.png` were inspected; an expected
404 fixture notification may still be visible after successful retry. This delivery
changes tests, the npm runner entry and governed UI docs only, not product behavior.
No live PDF/OCR/LLM or real-data destructive test is claimed.

This follow-up's UI checklist remains 6/6 (100%). After the final backend
PDF-FIX-03 acceptance recorded 2026-10-17, the combined execution index is 17/17
(100%); all 21 real-engine mode cases pass. The loader checklist owns this
backend completion and deployment limitations, not this UI milestone. No frontend
code, hierarchy truth or API contract changes are introduced by this index sync.

## Planned: Honest Structure And Interaction Recovery (2026-10-17)

READER-01 belongs to the new ten-task fragmented-native-PDF follow-up in
`Deep_Reflective_Reader/document_structure/module-detailed-design.md`; it is not
an extension of the completed 17-item Library/PDF-mode track. Delivered 2026-10-19.

Delivery, corrected 2026-10-19: upload/reprepare use shared Auto/Present/Absent TOC
selection, snapshotted independently of PDF extraction mode. Reparse offers Common
and a single LLM enhance action. The earlier two-intent menu was a requirement
misinterpretation and is removed. Backend task-layout supplies bounded
TOC observations, declaration uncertainty and implicit-single-body state; sections
expose exact-range matched title_origin while chapter metadata retains its existing
origin. Missing provenance stays unknown, without client inference. UI labels
generated navigation and warns about unconfirmed declarations/fallback.

Successful replacement clears interaction caches, pending response identities,
drafts and content before authoritative layout reload. Failed enhancement leaves
the accepted layout/content usable. Manual TOC commit also invalidates old targets.
Non-2xx recoverable envelopes retain statuses/reasons; policy-stale thinking offers
explicit new-question generation, never automatic generation on open. Failure
responses cannot trigger success notifications or clear answer drafts. Thinking
drafts have one owner in App rather than an effect reset on callback changes.

Validation: production build/typecheck and all seven mocked Library/reader suites
pass, including 1280px/390px screenshots, declaration snapshots, both intents,
flat/TOC fixtures, fallback, failed-reparse preservation, stale policy recovery,
repeated drawer opens and failed-submit draft preservation. Backend provenance
contracts (4) and task-layout readiness tests (11) pass. No real LLM, private-document
reparse or deployment was performed in READER-01. NATIVE-06 subsequently completed
isolated private-PDF/PostgreSQL acceptance on 2026-10-19 and reran all seven browser
suites. The combined track is 10/10; browser tests still use intercepted APIs, not
a claim of full live frontend/API/provider acceptance. See backend owning design.
The original acceptance design follows; delivery supersedes its staging notes.

Investigation confirmed backend `序` mis-segmentation and an explicit common-parser
fallback after LLM reparse. UI must not infer a physical TOC from the existence of
chapter containers, or infer no TOC solely from a title/one-chapter shape. Consume
backend-validated provenance distinguishing implicit reading containers, actual
headings and TOC evidence; any new fields require backend delivery first. A flat
article remains readable with valid target IDs and all interactive actions, but
does not show fabricated chapter/TOC labels. User TOC editing stays explicit and
does not automatically save a generated hierarchy.

Existing types already expose requested/effective mode and fallback reason;
`resolveLayoutParserMode` uses effective mode. Reparse feedback must clearly
distinguish requested LLM parsing, successful LLM application, fallback to common,
and failure preserving the prior hierarchy. Do not report fallback as an LLM
repair success. Do not expose raw prompts or source excerpts in error diagnostics.

2026-10-19 planning addition: before upload-triggered preparation and explicit
reprepare, offer a three-state TOC declaration: Auto (default), Present, Absent.
Keep it distinct from PDF fast/OCR/auto extraction. Snapshot the selection with
the operation so changing a draft cannot alter an in-flight request; a draft or
ordinary Open never mutates server state. Existing prepared documents may open
without repeating preparation or waiting for optional LLM enhancement.

For explicit LLM enhancement, offer only LLM enhance: optimize Common results with
a useful navigation/chapter/section plan. Do not expose Restore Source Structure
versus Generate Reading Outline as separate actions. Existing restore_source wire
default is compatibility-only; backend unified planning permits labelled semantic
headings without another user selection. Show generated heading origin from accepted
chapter/section provenance, not by inferring it from the request intent. Do not label
generated navigation an original TOC. Consume
backend conflict/fallback results with a recoverable warning and retain the usable
common document on failure. A single Main Body is a successful reading state,
not a repair-required error. Request inputs depend on NATIVE-03/04/05; bounded public
provenance was delivered and contract-tested with READER-01 before UI consumption.

READER-01 adds desktop/mobile tests for declaration snapshots, optional enhancement,
one LLM menu action, mismatched declarations, generated-origin labels and preserved Open
readiness after enhancement failure. Existing explicit user TOC editing remains
available and separate from AI-generated navigation.

Interaction status presentation must preserve the distinction between short text,
fragmented source, stale output and generation failure rather than suggesting every
`insufficient_content` case needs more text. Reuse supported reason/status fields;
new contracts are conditional on backend implementation. Explicit retry/refresh
recovers corrected guards/old wrong-language artifacts, without auto-generation on
menu/drawer open. After explicit reparse, clear stale local target/artifact state
and reload authoritative IDs; rejected reparse preserves current readable content.

Acceptance: desktop/mobile no-heading article, real TOC, valid preface, fallback,
failed reparse, guard recovery, old language artifact and repeated menu opens.
Backend source remains authoritative, task-layout remains lightweight/read-only,
and frontend display choices never mutate hierarchy or profiles.

## Non-Goals

The list below describes the completed first Reader slice, not the next reading
interaction design slice.

- summary UI
- quiz/critical-thinking backend mutation from drawer open or read paths
- quiz or critical-thinking drawer API calls before their planned service/controller
  wiring tasks are implemented
- free QA
- annotation
- text selection
- content-block highlight
- ask-about-selection
- artifact creation outside explicit future interaction routes
- artifact persistence UI outside explicit future interaction routes
- generic reparse UI outside the explicit TOC editor flow
- enhanced parse diagnostics UI
- authentication
- persistent reading progress

## Future Non-Goals For TOC Editor

- direct cold-start editing from `/documents/:docName/toc-edit`
- automatic layout loading from the TOC edit route guard
- hidden task-layout mutation
- treating frontend-edited TOC as backend truth before validation and explicit commit
- deeper-than-two-level hierarchy editing
- manual-structure UI support for `page_range` before page-boundary mapping is available
