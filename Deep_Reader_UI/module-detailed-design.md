# Deep_Reader_UI Detailed Design

## OCR Review And Content Correction (2026-10-21)

**[Code-Confirmed]** Shared queue: backend doc_loaders
`module-checklist.md` OCR-REVIEW-01..06. UI delivery is part of OCR-REVIEW-04/05/06.
Do not infer source quality from
rendered strings or convert extraction failures into success locally.

Show a persistent review-needed indication in Library and reader when the server
reports partial recognition. Expose failed physical pages/regions with original
image preview; an empty OCR region is not invented text or a silently absent block.
Opening review is read-only, without OCR/profile/LLM generation. Replaced source
bytes must invalidate preview identity. Task-layout stays lightweight.

Block editing uses server identity/revision, explicit Save and a 10000-code-point
limit. Confirm Delete; provide explicit Undo and dirty-draft protection. Failed
or conflicting writes retain the draft. On successful mutation reload authoritative
content and affected interaction availability; do not emulate cache invalidation
in UI state. Original extraction remains immutable, and saved notes must not be
silently deleted with stale generated interactions. Block generation selection with
a warning for server-reported unresolved source until reviewed; backend preflight
remains authoritative. Do not assume that low quality
necessarily means no usable content. Desktop/mobile acceptance must cover preview,
warning persistence, save/delete/undo/conflict and failed-request recovery.

Edit/Delete are interactive actions in the existing content-block three-dot menu.
No manual addition/insertion of blocks is supported. Source review is a separate
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
keyed by action/target. Summary renders headline/body/key_points with explicit
Generate/Refresh, preserving saved output on failure, and scrolls independently.
Note is one plain-text record per target, max 10000 Unicode code points; no title/
tags/autosave. Explicit Save supplies revision, stable retry ID and target token;
Delete requires confirmation. Dirty close/document navigation and page unload are
guarded. Failure keeps the draft; source/conflict errors never overwrite server
content. Structural commit clears selection; label-only refresh leaves it intact.

Acceptance: `test-summary-note-live.mjs` uses actual backed-up/rebuilt books and
API at 1280px/390px across all five levels, saved real-provider summaries and real
note writes. It verifies read-only opening, persisted reopen, over-limit emoji,
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
another explicit check. Other ambiguous errors, lost connections and the 120-second
client waiting timeout retain local data and block re-submission. Aborting waiting
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
- Section buttons update local selection state while retaining internal task-unit ids.
- Reader panel fetches each task-unit content for the selected section on demand and renders aggregated `content_blocks`.
- Empty-state views handle missing document, missing hierarchy, missing task units, and missing content blocks.

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
- missing/not-generated state
- explicit generate
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
- show `not_generated` without calling LLM
- provide an explicit `Generate quiz` action
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
2. If missing, show an explicit `Generate question` action.
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
   loading without generation; explicit generate/refresh UI wiring is implemented.
   Critical-thinking flow remains separate.
3. Critical-thinking drawer API wiring: in progress. Request/response API types and
   `ReadingInteractionService` methods are implemented, and backend session responses
   can be mapped into drawer view state. Drawer open now performs read-first
   critical-thinking session loading without question generation; explicit
   generate/submit/retry UI wiring is implemented. This remains separate from quiz
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
- wire drawer open to read-first critical-thinking session loading without creating a
  question session from the read path (completed)
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

The UI must not call generation from a read path, and must not hide LLM/cost-bearing
work behind menu open, target selection, document load, section selection, or drawer
open. Explicit user commands such as `Generate`, `Refresh`, `Submit answer`, and `Retry
evaluation` are the only places where interaction mutation should occur.

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
