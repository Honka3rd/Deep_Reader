# app Detailed Design

## APP-01–04 remediation (2026-11-13)

### Unit 3 answer durability and fixture reconciliation (2026-11-14, locally verified)

Submit-answer first conditionally saves answer_submitted for the loaded session,
then evaluates outside the repository mutation, then conditionally publishes the
result for that exact saved session revision. Retry evaluates the stored submitted
or failed answer and uses the same compare-and-save contract. Concurrent changes
must conflict instead of overwriting another answer/result; source/incarnation,
rename epoch and effective target identity remain valid at publication. Interruption
leaves the accepted answer readable for explicit retry. No automatic provider retry.
Transport/provider errors and malformed output become safe evaluation_failed codes,
never private exception strings. Unknown persistence acknowledgements must read
back the exact proposed session before returning success or dispatching evaluation;
otherwise fail closed without claiming acceptance. Legacy sessions remain readable;
missing source identity cannot authorize a new mutation. No read-time migration.

APP-04 remaining fixtures must supply current persisted hierarchy through repository
reads, assert zero preparation/provider/write on ordinary reads, and test filesystem
mutation at public boundaries rather than patching removed migration methods.
Acceptance includes real file/SQL and HTTP timeout/connection/malformed/reload/retry,
parallel submissions, source/lifecycle races, interrupted/save-failure paths, and
provider-time persisted-answer/no-long-transaction evidence. Complete cumulative
app regression and explicit skip accounting precede the final local review commit.

Implementation: the app saves answer_submitted before evaluation, using the exact
previous artifact as a condition; the session adapter validates source and session
revision inside mutate_document_artifacts. Evaluation and retries stay outside the
mutation. Unknown saves return a safe storage failure unless exact readback confirms
the proposed revision. HTTP publication conflicts return 409; unconfirmed storage
returns 500 without exposing provider/storage exception text or dispatching evaluation.

Evidence: `/tmp/app-fix-5b3291a3/final-report.txt`. New session publication suite:
32/32 file/real SQL methods, including HTTP timeout/connection/malformed output,
interrupted reload/retry, concurrent submit, stale result and unconfirmed save;
rollback after real SQL UPDATE, file replace failure and lost acknowledgements.
The original answer-loss reproduction now passes on both backends. Twelve related
interaction suites pass 96/96. Original eighteen app suites run in separate Python
processes: 327 executions, 321 pass, six unchanged private-source opt-in skips,
zero failures/errors. All seven formerly failing fixture suites pass independently.
The twelve original failures are reconciled; additionally discovered obsolete
prompt/output/type/legacy-fixture assumptions are corrected with original business
intent retained, not production guards relaxed. Two complete immutable backup copies
pass answer-save/timeout/retry on file/SQL with unchanged hierarchy and versions;
native extraction and deterministic providers do not establish model/OCR semantics.
Full app external review remains pending; no deployment or push.

### Unit 2 APP-02 source resolution (2026-11-14, locally verified)

Use config's shared, read-only raw-source resolver for enhancement state (also
standalone cleanup input) and the enhancement commit precheck. Qualified PDF/TXT
names, including uppercase extensions and safe Chinese/legacy punctuation, select
only the exact raw entry. Bare legacy identities require exactly one supported raw
candidate; never select one of simultaneous PDF/TXT sources by guess. Validate
names and the resolved raw-root boundary; persisted source_path is not authority.
No mkdir, migration, loader instantiation, extraction or profile generation.

Storage resolves the document first using the active repository's identity rules.
Ambiguous aliases become explicit source readiness failures. Source selection is
repeated after candidate generation, and the existing accepted-evidence/hash and
conditional commit checks remain mandatory. A missing/ambiguous/out-of-root source
at commit is source_changed, preserving current state. Consent and incarnation
fences are unchanged. Raw-path resolution does not expand file enhancement support:
the file adapter still lacks the production snapshot/lease port and remains gated.
Standalone cleanup also rereads its injected accepted-input port after planning and
before repository commit. The full consent fingerprint must still match; this
reselects the current logical raw entry instead of trusting an earlier resolved
symlink target. Source/identity failure is not_committed/source_changed; revised
input is not_committed/reading_revision_changed. This short precheck supplements,
not replaces, the repository's existing locked source/revision validation.

Acceptance: file source/accepted-evidence and unsupported-HTTP checks; real SQL
HTTP preflight, cleanup and enhancement execution/commit; qualified PDF/TXT,
uppercase/Chinese, coexistence, unique/ambiguous legacy, renamed, missing/replaced,
internal/external symlink cases; zero generation/save for rejected preflight,
unchanged storage on rejected commit. No production data or paid provider calls.

Evidence: `/tmp/app-fix-5b3291a3/unit-2-report.txt`. The new source-resolution
suite passes 14/14 methods (including real SQL HTTP commits); the original external
APP-02 assertion passes. Five selected baseline methods produced eleven expected
subtest failures, with no fixture errors. A separate cleanup source-retarget test
also failed before the post-planning check and now rejects without committing.
Ten adjacent suites execute 350 tests: 342 pass, eight private-source opt-in skips,
zero failures/errors. Inherited cases may overlap; these are execution counts.
Two complete immutable backup copies (652001/147401 canonical characters) pass
file accepted-source and real SQL HTTP readiness/reconstruction checks with zero
provider calls; replacement is rejected, stored snapshots and backups unchanged.
This source-read acceptance does not replace the eight skipped cleanup/private
source scenarios or claim live LLM/OCR semantics, UI or deployment acceptance.

Deliver in three independently validated units, with developer confirmation before
starting the next: (1) APP-01 guarded layout publication and related APP-04 layout
fixtures, (2) APP-02 shared source resolution across enhancement/cleanup/commit,
(3) APP-03 durable answer acceptance and guarded evaluation, plus remaining APP-04
read-only/migration fixtures. Final local review commit follows all units; external
approval is required before any deployment/push. Existing db approval is unchanged.

APP-01: capture server-owned source evidence before layout computation (including
explicit duplicate-ID repair). It binds document identity/incarnation, rename epoch,
structure/reading revisions, canonical and effective hierarchy text/coordinates and
source provenance. Artifact-only updates are excluded so concurrent saved artifacts
remain mergeable. Pass this evidence with layout metadata to the repository; compare
against the current snapshot inside the existing file lock/SQL transaction, before
any write. No provider/resolver work runs under this lock. Missing/stale evidence is
a conflict with zero mutation, including same-ID/same-text reparse, delete/reimport,
rename round trips, Edit/Undo and source changes. Retry is explicit preparation.
A final readback remains a readiness check, not the publication guard. Existing
valid legacy layout caches remain readable; no automatic migration/repair occurs.
Explicit preparation of a legacy snapshot without incarnation evidence fails with
`layout_incarnation_unknown_explicit_reparse_required`; reads never invent an epoch.
The token is internal server evidence, not a client-provided authorization field.

Acceptance: file/real SQL mutation and failure matrices, no pre-rejection changes,
current artifacts retained, content endpoint readback, successful explicit retry,
ordinary read zero preparation/provider/write. Related APP-04 fixtures must initialize
persisted hierarchy/current resolver identity and use generation-bound artifact writes.
All three units are locally implemented and cumulatively validated; external app
review remains pending. Earlier unit reports retain their dated checkpoint status.
`task_layout_source` owns source capture/validation, and `update_task_layout` checks
it inside the existing backend mutation. Unchanged ordered unit sources retain
stored IDs/artifacts (including SQL-assigned IDs). Duplicate-ID repair validates
ordered source equivalence and copies artifacts from the locked snapshot, never
from the proposed layout. Source checks precede all persistence.

Evidence: `/tmp/app-fix-5b3291a3/unit-1-report.txt`. Five new regression selections
fail against baseline production modules; all pass after the fix. The original
APP-01 reproduction now rejects with `task_layout_source_changed` and preserves
the new snapshot. Its diagnostic print alone was adjusted to handle no task units;
the unchanged desired-behavior assertion passes. New file/real SQL publication suite:
31/31; readiness/atomic/DB-metadata/cleanup/mapping support: 101/101; four corrected
APP-04 layout suites: 12/12; content/page/profile/hierarchy support: 36/36. No skips.
The adjacent hierarchy-write fixture now explicitly initializes an incarnation;
its content, duplicate-ID and artifact assertions are retained. Full app regression
results are recorded under unit 3 above.

Two complete immutable backup copies (652001/147401 canonical characters) pass
late same-text reparse rejection, unchanged new-snapshot bytes, explicit retry and
exact body readback. Backup manifests/hashes remain unchanged. Real native extraction
and disposable file storage, deterministic resolver, no live LLM/OCR semantic claim.
Units 2 and 3 were developer-authorized. The final local review commit covers all
three units; exact-SHA external approval is required before deployment or push.


## STRUCT-04 legacy generation integration (2026-11-13, implemented)

Section/chapter summary and quiz generation capture the accepted repository snapshot
before provider calls, resolve the task unit from its effective reading projection,
and pass source binding with the result. Cache checks require the same binding and
effective-source hash in addition to existing language/prompt/options checks.
Source conflicts return failure; never return a successfully generated but rejected
artifact as persisted success. Provider work stays outside repository transactions.

Validation: the 23-method generation suite, 18 atomic cases, 43 lifecycle cases and
120 HTTP rename cases pass across the applicable file/real SQL owners. Existing
535-test and PREP/manual support results are recorded in the document_structure
design and `/tmp/struct-fix-2c897af/`. No schema migration, read-time identity
backfill, deployment or push; latest review SHA still needs external approval.

## STRUCT-03 conditional manual commit (2026-11-13)

Implemented correction: manual orchestration obtains preparation-owned accepted evidence,
builds a complete draft and calls the conditional manual replacement repository port.
Missing accepted commit evidence returns readiness; stale source/version/incarnation
returns conflict. The app does not supply extraction provenance or use unconditional
save_reparsed_document for this route. Source verification runs again at publication
under the storage owner's lock/transaction. Existing page/char validation remains.
Validation: eight coordinator source-evidence cases, ten existing validate/route cases and the 16-method real file/SQL integrity suite pass. Source changes at the publication boundary return 409; injected persistence failures return 500 without success. Evidence: `/tmp/struct-fix-2c897af/`; whole-audit external review is pending.


## MAIN-06 pending file identity (2026-11-11, implemented)

Pending recovery must fence delete/reupload even when raw bytes are identical. A
raw-root-scoped `DocumentFileIdentity` owner persists a UUID generation per logical
name, independent of the selected canonical repository. Explicit upload publication
and raw/derived deletion advance the generation durably BEFORE effects. Their
critical sections and rename use the same POSIX lock; lock order is file identity,
rename journal, then repository preference/document/learning locks. Context reads
only project existing generations and never create storage.

Version-3 rename journals bind both original source and destination generations,
plus each move's device/inode/size/mtime/ctime, content fingerprint, and planned or
moved stage with the acknowledged destination stat. Recovery validates the captured
generations before any repository recovery or file effect. A moved file with a lost
acknowledgement is accepted only at the destination with the original device/inode,
size/mtime and hash; ctime may change during rename. A persisted moved stage also
requires the acknowledged destination stat. Upload/delete invalidate the generation
even if a filesystem reuses the inode and bytes. Changed identities leave the journal
intact and return recovery_required; no recapture of current identity is allowed.

Generation publication failing before a file effect leaves that effect unexecuted;
publication acknowledgement loss may conservatively fence a pending operation and
requires explicit inspection. This is fail-closed recovery, not a cross-resource
transaction. Old v1/v2 pending journals lack sufficient move identity and require
explicit maintenance; completed v2 receipts stay readable and are not deleted.
Legacy raw files without a generation use a null baseline until an explicit API
mutation. Out-of-band filesystem replacements and distributed hosts remain outside
the cooperating-owner guarantee. No xattr support or book-content backup is needed.

Validation: 393 backend regression executions pass without skips; final rename
matrix has 120 file/real PostgreSQL cases including 10 actual TCP HTTP cases.
The eight original HTTP repros fail with incorrect accepted responses on b6d5df24
and pass after this correction. Additional cases cover simulated inode/stat reuse,
new move-stage write/ack failures, v2 compatibility and identity-storage failure.
Nine standalone route/schema scripts pass; Python compilation and diff checks pass.
Evidence: `/tmp/audit-main-b6d5df24/`. No UI change, Docker deployment or push;
external review of the new candidate is pending.

## Rename command identity correction (2026-11-11, implemented)

Completed receipts (v3 and compatible v2) are keyed by caller command_id, bind old/new names and expected_source_token, and retain the server operation_id. The token hashes the storage scope, source name, canonical incarnation (or legacy source fingerprint), and optional raw stat identity (device/inode/size/mtime/ctime), plus the durable namespace generation in v3. New commands validate the token under the rename lock before starting. Reusing a command with another payload conflicts. Completed receipts also retain result_token for the destination incarnation/raw stat identity. A completed retry is read-only and succeeds only while its target, result token and moved resources still match and its source is absent; it never calls _start. A reverse rename or reimport requires a fresh command and current token. Pending recovery requires the same command/payload. Version-1 receipts are retained as historical evidence, never adopted as new commands; v1/v2 pending journals require explicit maintenance and fail closed. No read-time migration or receipt deletion. This remains local POSIX coordination, not protection from external filesystem replacement during a move.

## Main rename recovery orchestration (2026-11-11, implemented)

`document_rename.py` owns only the explicit rename use case. Inject the active
repository and raw/derived roots. Never open inactive structured storage. Repository
lifecycle owns preference/learning identity migration and writer fencing. Use a
POSIX lock and fsynced scoped journal for repository plus filesystem coordination;
this is forward recovery, not a cross-resource atomic transaction. Preflight all
known conflicts before effects; persist expected document epoch and resource
fingerprints before repository commit. Retry the exact request explicitly, verify
identity, complete matching file lifecycle recovery if needed, then move remaining
resources. Keep a completed receipt for idempotent lost-response retry. A pending
record blocks other rename operations in that scope. Changes outside cooperating
owners cause a recovery conflict, never guessed rollback or overwrite. Do not
claim coordination with external filesystem tools or distributed deployments.

CONFIG-03 follow-up (implemented): new commands run config's shared upload/rename
destination predicate before creating coordination state, then recheck under the
lock before journal publication. A qualified target cannot acquire a stored bare
legacy stem's source family. Pending recovery and completed receipts continue
through their recorded identity/stage/generation checks rather than new-command
occupancy checks, preserving idempotent recovery after a completed move.

Journals live in `<raw-root-parent>/.document-renames/<storage-scope-hash>/`;
`pending.json` is fsynced before repository commit, completed receipts are keyed
by caller command identity (see correction above). A new DocumentRename instance on an explicit identical HTTP
request reconciles pending work; no startup/read-time repair. Malformed journals
fail closed with recovery_required and require maintainer inspection. Changed
resource hashes or recreated epochs are never overwritten. An explicit reverse/rename cycle with a fresh command and current token starts a new operation. Legacy documents without
a document epoch use a stable source/provenance fingerprint; reads do not backfill.
Pure raw candidates have no canonical learning state and retain raw rename support.
Known file type/access and destination conflicts are checked before mutation; a
later disk or SQL failure retains recovery evidence. No cross-filesystem transaction
or host power-loss guarantee is claimed. HTTP/file/real PostgreSQL failure tests
exercise journal, repository, each move, receipt and acknowledgement boundaries.


## Shared sentence translation (2026-11-09)

Ordinary TranslationAction now delegates production generation to SharedTranslation,
using the existing learning runtime, aligned generator and canonical repository.
Its internal translation-purpose binding has zero preference revisions and keeps
exact source/language identity; ordinary translation does not require learning
eligibility. Existing learning requests and hints retain their settings checks.
Read/list and both generate entrypoints reuse compatible saved TranslationKey data.
No second aggregate is written into document artifacts. Legacy aggregate records
remain readable; canonical aligned results take precedence. Same mutation ID with
a different key is rejected before reuse. Known failures allow a new explicit retry;
unknown work is not redispatched. Repeating a known command may explicitly reconcile
safe claimed/partial work. One subscribe-before-status bounded notification wait
replaces any need for polling in the synchronous ordinary API. Timed-out requests
leave backend work intact. Source changes reject late output; partial/failed refresh
keeps the earlier complete result. Service restart resolves the new runtime owner.

Validation: 470 backend tests and actual API/React mode-switch acceptance pass; detailed evidence and limits are recorded in the shared section_tasks design.

Read/reuse clarification (2026-11-10): `SharedTranslation.catalog` accepts only
current compatible complete aligned results; `paragraph_record` derives the
ordinary DTO and retains the saved operation identity/result revision. The read
adapter exposes one record per target language for UI compatibility, while explicit
generation checks the exact source/target key before provider work. It does not
require learning preferences. An omitted source must resolve to a supported
existing default; otherwise generation requires an explicit source, even when
ordinary saved results can still be listed. No aggregate artifact copy is written.

The [manual acceptance](../section_tasks/module-detailed-design.md#manual-reuse-acceptance-2026-11-10)
records the user's learning-to-normal flow: both reads succeeded, no generation
request/batch event occurred, and the reference block retained its saved operation,
revision and SQL payloads. Request IDs and database comparison are complementary;
request logs alone omit the selected block and do not report an explicit cache hit.


## Learning completion observation (2026-11-09)

Contract and evidence moved to [learning](../learning/module-detailed-design.md#learning-completion-observation-2026-11-09).

## Language Learning Reading Mode Plan (2026-10-28)

Learning application orchestration is owned by the sibling [learning module](../learning/module-detailed-design.md).
App retains ordinary translation orchestration and consumes that shared runtime.

### LLRN-13 implementation contract (2026-10-28)

Moved to the [single owner contract](../learning/module-detailed-design.md#llrn-13-implementation-contract-2026-10-28).
This anchor is retained only for historical links; it is not a duplicate contract.

## Persisted Translation Orchestration Plan (2026-10-26)

Historical TRUI-04 implementation and legacy compatibility evidence. Current
production wiring uses [Shared sentence translation](#shared-sentence-translation-2026-11-09).
The target-only reuse, `TranslationService` generation, `commit_translation` artifact
writes and absent-source reuse described in this dated section are superseded for
the shared production path. Retain the historical tests, not a second runtime policy.

**[Code-Confirmed]** TRUI-04 extends the historical stateless action below.
Shared contract/lifecycle: [translation plan](../section_tasks/module-detailed-design.md#persisted-block-translation-plan-2026-10-26).
Keep TranslationService as the reusable provider/prompt owner. App coordinates
registry validation, existing profile default, authoritative block resolution,
generation and domain-owned repository commit; no new prompt or SQL in app.

Read current language-keyed results with zero model/profile generation, mutation or
stale-row cleanup. Opening controls loads supported registry codes and the existing
profile's default language; absent/unknown defaults are selectable, not fabricated.
Explicit source override still wins. Generate persists only completed/partial valid
results after commit-time text/version/deletion-epoch and same-language revision
checks. Untranslatable/provider/validation/commit failure retains earlier records.
Resolve an omitted/null source from the existing profile, then reject equal
source/target codes with `translation_same_language` before saved lookup or replay.
An existing valid target language is reused unchanged for a valid distinct pair,
even with a different From or mutation ID, before model calls/writes. Only a missing target
language generates; concurrent different languages
merge atomically. Same mutation replay returns the saved outcome without another
completion once committed; distinguish source conflict, transport failure and
unknown commit. Never keep an LLM request inside a repository transaction.

Storage uses existing document artifact metadata and atomic mutation
ports, with section_tasks owning records and document_structure owning source-change
cleanup; db adapters provide transaction guarantees. Verify ordinary artifact save,
rename/delete/reparse and Undo do not drop/resurrect translations unintentionally.
No frontend state, polling, timeout/cost policy or job queue is backend truth.

TranslationAction.read returns current saved records, exact registry codes and the
existing profile default without model calls or writes. Generate checks the resolved
language pair, then current saved target language, then replay; on a miss it calls TranslationService and commits
through domain commit_translation. Source binding guards still exclude stale results. Concurrent
same-command completions return the committed record, not an unsaved alternative
model response. Untranslatable does not save; failures preserve prior translations.
File and SQL CAS/merge/rollback/rename/source-race tests and live backup-excerpt
App/API tests pass. The stateless section below is historical first-phase evidence.

Recovery clarification (2026-10-26): saved reads expose current records, not durable
in-flight operation status. A missing target record cannot prove that an earlier
request was cancelled or never committed. The UI retains command-specific retry IDs
and allows another language pair; app/domain still enforce source-valid reuse,
same-command replay and guarded commit. Existing records survive provider failure.
Equal source/target pairs fail even if a target record or replay receipt exists.
An explicit valid source avoids profile lookup. For omitted/null source, a profile
read failure returns 503 even on a cache hit; absent/unknown profile may still reuse
an existing saved target, but generation requires an explicit source. Saved reads
retain historical equal-language records; no migration or deletion is introduced.
Frontend behavior is
specified in the [UI recovery design](../../Deep_Reader_UI/module-detailed-design.md#failure-recovery-2026-10-26).

## Block Translation Action (2026-10-26)

**Historical TRANS-API-01 scope, superseded by the persisted orchestration above.**
The no-write and future-frontend statements below describe that initial phase only;
current generation commits completed/partial records and UI delivery is implemented.

`TranslationAction` owns only target/profile orchestration; TranslationService
owns the prompt, capacity rules and output validation. Input is doc_name,
content_block_id, target_language and optional source_language. This is block-only:
do not accept source text or require frontend chapter/section/task-unit IDs.
Search the authoritative hierarchy's render-compatible blocks for exact identity;
reject missing/duplicate matches and use ReadingTargetResolver for effective
manual-edit/cleanup content. IDs are opaque, not parsed by prefix.

An explicit source_language wins without reading the profile. Omitted/null loads
the existing DocumentProfile.document_language (the public profile language), with
read-only path configuration: no directory creation, migration, profile generation,
OCR, preparation or ancestor summaries. Missing/unknown language requires explicit
selection; corrupt/unavailable profile is a service failure, not a language guess.

Translate the whole effective block or reject capacity before completion. Source
review flags alone do not erase/block translation: the shared translation prompt
returns partial/untranslatable for unreadable input rather than inventing text.
No artifact/session/cache/database write is introduced. After completion, reload
and compare target content, structure version, reading revision and interaction
epoch; reject deleted/reparsed/edited sources with translation_source_changed.
There is no transaction held over the model request. Return the checked snapshot's
version/revision/effective text digest, not a claim that future reads cannot change.
Frontend result lifecycle and translation persistence are separate future work.

Tasks: TRANS-API-01 in this module, then transport/DI/acceptance TRANS-API-02 in main.

## Offline Preflight Follow-up (2026-10-24)

**[Code-Confirmed 2026-10-24]** WAIT-04 adds observability, not a second cleanup engine or a job queue.
Incident, offline provider policy and execution order:
[WAIT design](../llm/module-detailed-design.md#offline-preflight-follow-up-2026-10-24).
Previously `StandaloneCleanupService.preflight` reported only total elapsed time; this
was insufficient to distinguish tokenizer acquisition from full-prompt counting in
the 157.6-second dark-water incident. Source reuse/input adaptation already have logs.

Record request-correlated start/end/failure and elapsed time for lease acquisition,
accepted input read/adaptation, capacity invocation, fingerprint/consent registration
and overall preflight outcome. Emit safe reason and execution availability on all
returns, including rejection/exception, without logging opaque consent/handle values,
private text or raw exception payloads. LLM-owned asset/init/count logs from WAIT-02
complete the timeline; app must not reimplement encoding/prompt construction to time it.
No additional source/repository read solely for logging. Logging failure must not
change operation semantics. Reuse the existing request context and timing helpers.

Preflight remains zero completion/OCR/profile work and zero document-content writes;
its short-lived in-memory consent registration is not a cleanup commit. Keep existing
shared lease, snapshot/capacity revalidation, short write transactions and unknown
outcome reconciliation. Do not hold a new DB transaction over initialization or add
request-path asset repair. Frontend removal of fixed deadlines does not disable
backend provider timeouts, consent expiry, concurrency checks or capacity refusal.
No new HTTP route or diagnostic payload is needed for these internal timings.

### WAIT-04 Delivery (2026-10-24)

`StandaloneCleanupService.preflight` retains the existing result/exception mapping;
its private `_preflight` coordinates the same lease, accepted-input reader, capacity
engine and process-local consent registry. `cleanup_preflight_logging` is a scoped
logging adapter around existing `logging_config.log_stage`, not a second timer or
tokenizer. It supplies these `cleanup_preflight_*` spans:

- `overall`: one terminal availability/reason for every returned result.
- `lease`: acquisition only, not time spent doing work while holding the lease.
- `reconciliation`: existing uncertain-attempt checks, without extra receipt reads.
- `input`: accepted-source read/adaptation and eligibility check, called once.
- `capacity`: existing engine call, including full prompt/schema assembly/counting,
  called once; nested llm-owned initialization/count logs retain the same request ID.
- `consent`: input fingerprint, token binding and bounded in-memory registration.

Existing stage IDs/parent IDs and request ContextVar correlate worker-thread logs.
Started stages emit finished/failed events with duration; unreached stages emit
nothing. Business refusals finish with `unavailable` and a safe reason; exceptions
mark the throwing span `failed`, while handled failures still produce an overall
unavailable result. Cancellation propagates and marks the overall span failed.
Stage context and the original lease unwind on every path. `execution_available`
is false until successful consent/overall completion; it is diagnostic, not new
authorization state or an HTTP field. No source, prompt, path, token, handle or
fingerprint is added to these records. Reason strings are allowlisted for logging;
unknown values map to `cleanup_preflight_unavailable` without changing use-case
results or route-owned sanitization. Existing shared logging emits exception type,
never exception message. The adapter absorbs ordinary sink exceptions for its new
records, preserving success/failure and context cleanup. Global logging and unrelated
services are unchanged; this does not promise every older log call is fault-isolated.

Validation: `scripts/test_cleanup_preflight_logging.py` adds 14 tests for paired
spans/parentage, deterministic per-stage timing, single read/count, no content writes,
capacity refusal/unavailability, actual missing-asset fallback without network,
source/identity/eligibility failures, busy lease, reconciliation and consent failures,
safe unknown reasons, sink failures, cancellation and concurrent request isolation.
Actual FastAPI middleware/worker invocation retains the existing six-field DTO
(including `impact`) and correlates every new record with `X-Request-ID`.
208 scoped regressions pass with verified local tokenizer assets, immutable-book
opt-ins and disposable PostgreSQL. Three OCR backup books retain hashes through
existing service/API tests with deterministic model responses; Madame Bovary/Red
Roulette full accepted-text counter checks also preserve backups. The disposable
database was dropped. No paid model call, production mutation, deployment or restart.

Broader legacy check: `test_upload_preparation_logging.LoggingTests.test_prepare_stages_and_cache_preserve_behavior`
still expects `structured_reused=True` from an outdated Common-policy fixture and
fails with `common_policy_mismatch`. The same failure reproduces independently in
the pre-WAIT-04 image `62776af533ea`; shared logging/parser code was not changed or
the unrelated assertion weakened. The 208 passing cases exclude that known failure.
Full incident-book cold/warm HTTP and browser acceptance remains WAIT-05; these
stage tests do not claim that final performance gate or production deployment.

## Independent Cleanup Use Cases (2026-10-23)

**[Code-Confirmed: SCLEAN-05/07, 2026-10-24]**
SCLEAN-05 owns a narrow cleanup use-case service; SCLEAN-07 integrates its shared
planning path with enhance. Canonical rules:
[Independent Smart Cleanup](../document_structure/module-detailed-design.md#independent-smart-cleanup-2026-10-23).
The service coordinates preflight/consent, source snapshot adapter, planner,
repository commit and explicit readiness recovery. Do not put prompt construction,
quote matching, SQL or a generic job framework in SectionTaskCoordinator. Existing
coordinator entry points may delegate without exposing internal planning/staging
methods to HTTP or UI. Do not call the service's standalone commit from enhance:
enhance owns the combined hierarchy/overlay commit; both reuse the same planner.

Standalone consent authorizes only safe OCR deletion and changed-content generated
invalidation, retaining notes/manual edits and structure. It cannot reuse structural
discard_interactions permission. Bound source/hierarchy/reading revision, cleanup
policy/prompt and capacity identity are checked again before dispatch/commit.
Preflight and result lookup are read-only and make no model/OCR/profile calls.
Failed/no-op cleanup does not invalidate output. Current revision and durable receipt
take precedence over browser state; a historical receipt after Undo is superseded.

Replay the same operation without regeneration; fresh explicit cleanup obtains new
consent and can call the cleanup LLM again, never the split LLM. Refresh/unknown
commit outcomes first reconcile; database read failure must not look like absence.
Process-local in-flight work is not durable; a new attempt cannot silently race a
possibly running old operation. Source/revision CAS and operation ownership guard
both standalone and combined flows. Release transactions before model calls.
Expose a small retry action (regenerate_cleanup, resume_layout, reconcile, none),
not provider failure internals. Fresh consent is required for regenerate_cleanup;
resume_layout preserves the committed result and makes zero generation calls.

Logs record operation origin (standalone/enhancement), stages/timing, version,
counts/reason, commit/rollback/replay and layout. Do not log source quotes or tokens.
Automatic cleanup failure is nonblocking only for the valid split candidate; final
combined commit failure is still fatal. Standalone failure preserves the previous
reading state. The use case, not the UI, performs any required layout continuation.

### SCLEAN-05 Delivery (2026-10-24)

`standalone_cleanup.StandaloneCleanupService` exposes preflight, execute, read-only
lookup and explicit resume_layout. Composition injects one accepted-input reader,
the repository/provider, the SAME cross-process document lease used by enhancement,
and read-only readiness / source-and-revision-guarded layout continuation ports.
No default unlocked execution, private coordinator calls, SQL, matching rules or
prompt construction are added to app. Runtime port composition is SCLEAN-07;
minimal HTTP mapping is SCLEAN-06. Neither is claimed implemented by this service.

Preflight budgets the complete remaining-view prompt through the structure-owned
engine, with zero model calls. Five-minute opaque consent handles bind namespace,
document, full input fingerprint, reading revision, prompt/policy and model-capacity
identity. They authorize only standalone cleanup, not all-interaction discard.
Execution rechecks under the document lease, the engine checks capacity identity
again before dispatch/transient retry, and app checks again before the short commit.
Precommit records are process-local, bounded to 1024 entries with one-hour inactive
retention; active work is never evicted. Restart does not authorize replaying an
uncommitted attempt. A new attempt needs fresh explicit consent, readable storage,
a newly accepted snapshot and an available shared lease. Locally uncertain writes
are reconciled first. Same-operation execution never implicitly regenerates a plan.

Durable content-only receipts now include a minimal source/structure fingerprint.
Lookup verifies namespace/document/operation, accepted source, structure version,
reading revision and actual injected readiness; historical layout flags alone are
not readiness authority. Missing legacy lineage, later edits or Undo cannot authorize
layout continuation. Missing receipts after restart stay unknown; read failures are
reported separately and are never swallowed as absence. Lost commit acknowledgements
reconcile the receipt before any continuation. Conflicts preserve concurrent work;
unconfirmed storage failures remain unknown. Fresh confirmed retries call only the
cleanup planner. Failed/partial plans retain their actual counts/status and retry
action, never report complete repair of all OCR regions.

After a confirmed commit, the service checks readiness and explicitly invokes the
layout port when required. Failure returns resume_layout; lookup itself does not
prepare. Resume after restart preserves the committed outcome and makes zero model
or cleanup-commit calls. The port must CAS its supplied accepted snapshot before
writing and reuse existing profile/source; actual preparation wiring remains SCLEAN-07.
Stage/timing logs use hashed operation IDs and safe reasons, not consent tokens,
private source text or exception payloads.

Validation: `test_standalone_cleanup_service.py` adds 32 file and the same 32 isolated
PostgreSQL use-case cases. Total 248 local and 146 disposable SQL/API/enhancement
regressions pass. Includes consent expiry/staleness, dispatch-time model changes,
duplicate/concurrent submission, provider retry, rollback/lost response, restart,
read failure versus absence, no-op/partial, Undo/supersession and layout-only recovery.
Three hash-verified backup books exercise actual accepted adapters, shared engine and
file/SQL commits using deterministic provider responses; Sample2 applies the previously
reviewed deletion. Backups and production Library remain unchanged. No live LLM,
standalone HTTP/UI availability or service deployment is claimed. SCLEAN-10 remains
the live-model/browser acceptance gate.

### SCLEAN-06 Transport Support (2026-10-24)

The use case now derives the opaque operation ID from the confirmation token's
SHA-256 prefix. Execute checks this commitment before lookup or layout continuation,
including after restart; receipts retain only the operation handle, never the token.
Previously issued independent random-ID precommit handles are not compatible with
this initial public wire contract and require fresh preflight. No standalone public
API was shipped before SCLEAN-06.

Repeat execute on a committed layout phase delegates to the same private layout
continuation while holding the shared document lease. No route-owned recovery
engine or second planner is added. Committed lookup includes current pending review
count from the accepted snapshot; unavailable current review state remains unknown.
Missing-document errors remain distinct from unknown-operation or storage-read
failure. Unknown handles check accepted document existence read-only; this neither
generates nor authorizes retry. Transport/status mapping remains in the API module.
Production port composition and automatic enhancement integration are delivered by
SCLEAN-07 below; the preceding test receipts describe their original task scope.

### SCLEAN-07 Runtime Composition (2026-10-24)

`cleanup_runtime.build_cleanup_service` injects public coordinator ports into the
existing standalone service; it owns no prompt, validation, SQL or retry policy.
Composition requires coherent snapshot and shared cross-process enhancement lease
ports; unsupported storage remains unavailable. Preparation and standalone receive
the same configured provider directly. Main installs the service without model I/O.

EnhancementState retains its coherent accepted hierarchy for the candidate adapter;
cleanup does not reload a persisted candidate. Enhance and standalone use the same
remaining-view planner, exact validator and overlay application; only their commit
contracts differ. Consent binds the new remaining-view prompt/policy as well as the
existing overlay scope. Adapter failure retains the valid split and reports failed
cleanup. Public reading_layout_ready is shared by both reconciliation paths; explicit
prepare_cleanup_layout guards accepted input before work and just before layout
write. No-op receipts retain existing IDs even when refreshing stale cache metadata.
Same-op retries do not generate, and independent retries never invoke enhancement.

Evidence/counts belong to preparation SCLEAN-07: 30 new real-layout cases, 308 local
and 190 SQL/API regressions plus 16 existing consent/operation/cache cases. Actual
default API dependency, container/provider reuse and same-document lease tested;
three backup books unchanged. No deployment, UI or live-model acceptance claim.

## Cleanup Layout Handoff Support (2026-10-23)

Supporting preparation CLEAN-05 only: explicit layout resolution delegates to the
canonical/effective mapping helper, retaining existing overlay targets. Cache metadata
includes reading_revision; cleanup-dependent mismatch is read-only not-ready until
explicit preparation. CLEAN-08 supporting mutation now advances an already matching
stamp atomically for explicit overlay changes/Undo, preserving canonical units/IDs;
a previously stale or missing stamp still requires explicit preparation. Thus Undo
of a ready document no longer makes page reload fail. Ordinary manual block/review-only
edits retain readiness. Repeated ready preparation does not write.
Public cleanup execution, consent, outcomes and automatic invocation are delivered
by CLEAN-06 below; this supporting cache change alone exposed no frontend capability.

## Optional OCR Cleanup Operation (2026-10-22)

**[Code-Confirmed]** CLEAN-06 delivered on 2026-10-23. API support is implemented;
frontend consent/integration and real-book acceptance remain separate tasks.
See [canonical design](../document_structure/module-detailed-design.md#optional-ocr-cleanup-plan-2026-10-22).
Orchestrate explicit enhance as split/validate in memory -> optional cleanup/fix
plan and in-memory application -> one atomic hierarchy/overlay commit -> automatic
explicit layout preparation/read handoff. No manual Prepare, intermediate hierarchy
save or database reload to perform cleanup.
Do not hold a database transaction open across LLM requests. Keep operation identity
and stage receipts so refresh/response loss can reconcile the single final commit
without repeating its mutations. Generation receipts are not hierarchy saves; never
expose an uncommitted candidate as the current document. Persist minimal source/version/revision-
bound phase receipts via the repository; do not claim the current process-local
registry survives restarts. Unknown outcome remains unknown until storage is checked.

Extend existing preflight, execution and result DTO/routes additively; load root
main/api_schemas memory before changes. Preserve structural change_kind/committed/
structure_version/interactions_cleared meanings. Add optional typed cleanup outcome
(status, proposed/applied/skipped counts, safe reason, reading_content_changed,
reading_revision) and phase-aware retry information. Legacy absence means unknown/
unsupported, never cleanup completed. Update reading_layout_required based on actual
effective-content freshness, including cleanup after structurally unchanged results.
Return/reload server-owned remaining review state after the final commit. Only
regions proven fully resolved by the backend leave the active review queue; no
redundant manual Ignore is required. Failed/partial/textless regions remain pending.
Undo restores batch-resolved review states with reading content. Do not equate
cleanup.status=completed with every OCR region resolved or perfect transcription.

Cleanup failure/skip retains the usable uncleaned split candidate and a warning;
only a successful final commit can report structural success. Final commit failure
rolls back hierarchy and overlay together, retaining the old stored document.
Transient cleanup provider failure may retry before commit without cost gates; on
failure, continue with the retained split candidate. After commit, do not implicitly
reload and clean/save again: recovery retries only layout/read or reconciles the
existing receipt. A new enhancement requires explicit initiation and fresh consent.
This recovery rule concerns replay of that same enhance operation. The new SCLEAN
design additionally permits a separately confirmed standalone cleanup operation
after failed/partial cleanup; it does not require another enhancement or split.
Final source/revision/consent conflicts reject the whole stale write and preserve
concurrent user work; they are not downgraded to optional-cleanup warnings.

Consent must explicitly include OCR cleanup and generated-output invalidation while
distinguishing structural replacement's all-interaction deletion. Bind reading
revision and cleanup policy identity; stale or old-scope consent cannot authorize it.
No-op structural results do not short-circuit optional cleanup. Separate capacity
checks ensure oversized cleanup requests make zero cleanup calls without blocking
a feasible structure operation. Correlated privacy-safe logs cover both stages and
layout, including counts, failure reasons, timings, replay and rollback.

### CLEAN-06 Delivery And Recovery Contract (2026-10-23)

- Strict additive `allow_ocr_cleanup` defaults false on preflight, reparse and result
  requests. The existing reparse adapter only accepts true with llm_enhanced;
  SCLEAN adds a separate entry rather than relaxing reparse into a cleanup endpoint.
  Existing clients do not implicitly grant
  deletion authority through discard_interactions. Consent v3 binds this option,
  current reading revision, cleanup scope/policy/prompt, and existing source,
  hierarchy/interaction and model-capacity identities. Preflight exposes eligibility,
  revision, scope and conditional generated-output invalidation without a model call.
- `enhancement_cleanup.py` invokes the configured structure provider through the
  independent full-source cleanup planner. No extraction/profile work or transaction
  spans generation. One transient transport retry is permitted before commit; no
  token/cost/OCR quota is introduced. Capacity/invalid-plan/permanent failures skip
  cleanup and retain the candidate. An empty internal application preserves prior
  manual/source overlays through structural replacement without authorizing deletion;
  the reported outcome remains the actual failure/skip, not a fabricated no-op success.
- The one final hierarchy/overlay transaction also stores an operation-keyed minimal
  receipt in parse_provenance.enhancement_receipts. It contains structural outcome,
  source identity/hash, resulting structure/reading revisions, counts and cleanup
  outcome, not confirmation tokens, source text or model responses. Unchanged
  structure can therefore write an operation/cleanup receipt while committed=false
  retains its original structural meaning. Cleanup revision/change lives separately.
- The opted-in POST automatically runs explicit progressive layout after commit.
  success=true proves structural commit only; phase=complete proves current layout
  readiness. Layout failure returns phase=layout, retryable=true, retry_stage=layout
  and a safe layout_error. Transient precommit structural generation instead returns
  its existing error status with retry_stage=generation. Optional cleanup failure is
  terminal for that committed operation, not permission to save/reload/clean again.
- Result polling is read-only. A matching persisted receipt returns
  durability=persisted_commit and status=completed (commit completed, not necessarily
  layout). Current layout metadata determines phase after restart/response loss.
  Reposting the same confirmed request resumes only missing layout, rechecking source
  identity under the document lease. It does not depend on the restarted process's
  signing secret or expired precommit token, because a scoped exact receipt proves
  authorization already exercised. A changed reading/source/structure state yields
  historical phase=superseded without automatic mutation or retry. Review counts
  always describe current stored state, including review restored by Undo.
- Before commit there is still only the bounded process-local operation registry,
  not a durable job queue. Missing/evicted receipts remain unknown, never permission
  to regenerate automatically. Receipt history is bounded to 1000 metadata records;
  this is not a model-use quota. Failed result-store reads return a sanitized error.
  A durable receipt never claims an uncommitted candidate was saved or a precommit
  model request survived a worker restart. Last layout exception text is not stored;
  after refresh the server proves pending layout rather than reproducing that error.
- Final storage failure rolls back receipt/hierarchy/overlay/invalidation together.
  Lost commit acknowledgement remains unknown until receipt reconciliation. Cleanup
  safety conflicts reject stale operations; they are never optional-failure success.
  Authoritative pending_review_count/review_required come from remaining regions,
  not cleanup.status. Existing reading-content Undo restores the review queue.

### CLEAN-06 Governance And Validation Receipt (2026-10-23)

Inspected: AGENTS.md; proposal.md; high-level-design.md; docs/modules/index.md;
progress.md; design/checklist pairs for app, document_structure,
document_preparation, db, section_tasks, scripts, main.py and api_schemas.py.
Earlier loader/UI memory and pre-existing working-tree edits were retained unchanged
in this task. No UI source or UI markdown changes were made by CLEAN-06.

Markdown modified and reason (paths relative to backend root):

| Files | Reason |
| --- | --- |
| app/module-detailed-design.md | Record implemented consent, orchestration, recovery limits and validation evidence. |
| app/module-checklist.md | Complete only CLEAN-06 with evidence and current timestamp. |
| main.module-detailed-design.md; api_schemas.module-detailed-design.md | Describe implemented route forwarding and additive strict contracts, not future claims. |
| db/module-detailed-design.md | Document same-transaction receipt storage and rollback, with no schema migration. |
| section_tasks/module-detailed-design.md | Separate structural success from cleanup, phase and review result fields. |
| document_preparation/module-detailed-design.md; document_preparation/module-checklist.md | Record configured-provider handoff and update the cross-module delivery pointer without changing CLEAN-05 completion. |
| document_structure/module-detailed-design.md; document_structure/module-checklist.md | Update shared ownership/delivery index to 6/10 without closing later tasks. |
| progress.md | Sync backend-only CLEAN-06 completion: 6/8, leaving UI out of backend totals. |

Evidence: test_enhancement_cleanup_api.CleanupApiTests has 27 disposable-PostgreSQL
route cases. Combined 23-suite run: 360 cases, 359 passed, one existing private-source
opt-in skipped; command scope includes outcome/preflight/operations, cleanup
contract/planner/validator/overlay/preparation, real SQL reading mutations, source/
loader/layout and PDF-mode/identity regressions. Local log:
/tmp/deep-reader-clean06-combined.log. The new cases cover strict consent/policy/
revision checks, capacity zero-call, cleanup-only note preservation, partial review,
Undo, transient retry, manual edits surviving failed cleanup, single hierarchy
commit before separate layout persistence, concurrent edits during provider work,
real rollback after receipt writes, lost acknowledgement, worker restart and safe logs.
No paid model, production book/schema mutation, deployment or browser acceptance.
Those facts are not substituted for CLEAN-09/10 acceptance.

Validation: git diff --check passes; ownership and architecture remain intact:
hierarchy is authoritative, source/profile remain immutable in cleanup, result/layout
reads do not prepare, and persistence is explicit. Implementation-validation is
read-only and changes no files. Checklist synchronization changes documentation only;
source changes belong to the preceding implementation phase. Existing unrelated
dirty files were not reverted. Remaining app cleanup tasks: 0/1; combined 6/10,
backend 6/8. Frontend CLEAN-07 is the next ordered task.


## Corrected Content Projection (2026-10-21)

OCR-REVIEW-05: get_task_unit_contents applies the reading-only overlay to the
existing authoritative snapshot. Preserve explicit edited block IDs and hashes;
do not resegment and renumber surviving blocks after deletion. An emptied unit
returns empty reading content, never resurrects original section text. Content
reads remain read-only and never prepare/profile/OCR. Task-layout remains a
lightweight hierarchy projection; review evidence has its own endpoint.

## Format-Neutral Enhancement Integration (2026-10-21)

**[Code-Confirmed]** EXTRACT-08 is implemented; see [the preparation plan](../document_preparation/module-detailed-design.md#format-neutral-extraction-plan-2026-10-21).
Both TXT/PDF must use the same accepted-source preflight/execute flow. Source routing
comes from trusted storage/loader ownership, not a PDF suffix guess. Keep source
readiness distinct from model capacity; validate full assembled prompt before LLM
calls. Preserve consent, failure reconciliation, automatic post-commit layout and
retry policy. No extraction/OCR/profile fallback. Reuse existing response contracts
where sufficient; any new reason/schema must be implemented and tested before UI
adoption. Log correlated stage/format/reason/timing without text or tokens/secrets.

## Summary And Note Orchestration (2026-10-21)

SN-10/11 delivered. Summary read/generate/refresh uses repository hierarchy reads
directly, validates supplied structure version, loads only an existing profile,
then delegates to the configured SummaryInteractionService. It deliberately does
not reuse the older target helper that calls prepare_and_load. Note read/save/delete
delegates to NoteInteractionService with the repository only: no LLM/profile or
preparation dependency. Target DTOs are validated by public schemas at the route.
Tests cover actual coordinator/routes and persisted real-book operations.

## Enhancement Outcome Handoff (2026-10-20)

### Recovery Clarification (Maintainer-Confirmed 2026-10-20)

Prioritize a correct recoverable workflow, not LLM retry cost. Token capacity is a
technical feasibility gate BEFORE generation, not a monetary budget. Do not add a
cost threshold, spend approval per retry or a blanket prohibition on retrying LLM.
Confirmed transient transport/provider failure permits retry of the LLM stage.
Unknown HTTP outcome first reconciles operation state: reuse an available result,
wait for active work, or retry generation when recovery establishes it is needed.
Operation identity, single-flight and at-most-once commit/cleanup remain correctness
requirements even when duplicate provider work cannot be ruled out.

After a confirmed structural commit, automatically prepare layout-only then read.
If preparation fails, retry preparation; if only reading fails, retry the read.
Do not rerun completed generation merely because a downstream stage failed.
The user sees one smart-structuring operation and a contextual Retry action, not a
normal-path manual Prepare step. Capacity rejection stays blocked until relevant
input/configuration changes; a valid no-op is not a transient failure to retry.
Failure classification must distinguish transient provider/network failures from
capacity, configuration, stale consent and invalid-plan outcomes. This revision
does not authorize blind retries for every HTTP 4xx/5xx or unlimited looping.
Cost budgets and optimization policies are deferred. No retry-count policy is
claimed as confirmed here; cancellation and recoverable UI state remain required.

ENH-REV-03 implemented (2026-10-21); follows structure ENH-REV-01 and db ENH-REV-02.
Public API now implements the additive outcome fields below. UI adoption remains
ENH-REV-05; this is not a claim that current UI preserves drafts on every success.
The authoritative requirement is the three-way table in document_structure design:
unchanged preserves everything without commit; titles_only commits labels without
clearing interactions; structure_changed commits hierarchy and clears ALL document
interactions. API success alone is not permission to clear frontend caches/drafts.

Implemented response fields: `change_kind` (unchanged/titles_only/structure_changed),
`committed` (false for unchanged), authoritative `structure_version`,
`interactions_cleared` and `reading_layout_required`. An unchanged result is a
successful evaluated request with no update, not a provider failure or claimed
improvement. Title-only may commit with the SAME structure_version. Structural
commit requires layout-only preparation before reading; no-op/title-only does not
invalidate an already-ready layout. Legacy Common/manual responses leave these new
outcome fields null rather than guessing a classification. `retryable` defaults false.

Read-only preflight cannot know the eventual LLM result. Show conditional cleanup
scope and obtain consent BEFORE the generation operation: all interactions are removed
only if body structure changes. Retain existing explicit token/discard field for
compatible request handling, but its meaning is authorization for conditional
structural cleanup, not a command to delete on any success. Bind consent to the
baseline structural AND label fingerprint, source, interaction revision and prompt/
model/budget policy. Recheck at commit, including title-only updates. Recovery within
the authorized operation may retry transient generation failures; changed source/
scope or stale consent requires renewal. Do not automatically retry valid no-op.

Unknown outcome reconciliation does not rely only on a larger structure_version:
title-only can commit without a version increment, and unchanged has no commit.
The minimal result contract is implemented below. An unchanged version cannot prove
failure; labels alone cannot prove completion. No cross-restart durability or general
job queue is claimed.

### Implemented Transport And Recovery

- Preflight adds `cleanup_condition=structure_changed_only`; existing cleanup_scope
  describes the maximum impact, not unconditional deletion. Existing strict boolean
  discard consent and source/hierarchy/label/interaction/capacity binding remain.
- Enhanced success exposes `change_kind`, `committed`, `structure_version`,
  `interactions_cleared`, `reading_layout_required`. The latter two are true only for
  structural replacement. Title-only commits without incrementing structure_version.
- `POST /documents/reparse-structure/result` accepts doc_name, confirmation_token,
  toc_hint and enhancement_intent (same defaults as preflight). It is read-only and
  returns status `not_started | running | completed | failed | unknown`, nullable
  original result, safe reason, retryable and `durability=process_local`.
- Preflight registers an operation keyed by a digest of token/document/options.
  Execution atomically claims it before generation. Concurrent same-token execution
  returns enhancement_in_progress; completed replay returns the original result,
  including unchanged/title-only outcomes, without generation, preparation or cleanup.
  Results are historical operation evidence, not an assertion about later document edits.
- The registry is process-local, holds at most 1024 records, and retains inactive
  records for one hour after their last execution transition/registration. Running
  records are not evicted. Registry exhaustion blocks issuing a new token; it is a
  memory bound, not a model spending policy. Lookup/replay do not extend retention.
  Result lookup/replay may outlive the five-minute consent token; starting/retrying
  generation still revalidates token expiry and all baseline/policy bindings.
- A restart, different worker or expired record returns unknown and cannot restart
  the same operation. An unexpected exception during execution is also unknown,
  because a lost DB commit acknowledgement does not prove rollback. No version-based
  inference or automatic regeneration is permitted. Cross-worker durable recovery
  remains unsupported; explicit investigation/new consent is separate from blind retry.
- Only confirmed transient provider transport/timeout or HTTP 408/429/500/502/503/504
  failures return retryable=true (HTTP 502). Retry re-enters the same guarded operation
  without cost gates; there is no automatic retry loop. Unknown provider exceptions,
  invalid plans, capacity/configuration rejection and stale consent are not transient.
  Terminal results/conflicts replay without generation; safe capacity error details
  are retained. Provider messages, prompts and tokens are never logged.
- Safe operation logs contain digest/status/change kind/version/retryability. Source
  mutation, capacity checking and whole-document cleanup remain in their existing owners.

Evidence: `enhancement_operations.py`, preflight/coordinator, reparse result and DTO/
route wiring; `llm_provider.is_transient_provider_error` plus splitter/builder safe
failure propagation. `test_enhancement_outcome_api.py` includes real disposable
PostgreSQL/HTTP tests for all three outcomes, lost response replay, transient retry,
in-flight exclusion, title-edited consent and restart/worker unknown state. Existing
tests continue to prove capacity rejection spends zero model calls and layout-only
preparation is a separate explicit write. Operation tests cover retention, memory
bounds, unknown exceptions and identity binding. Seven-suite acceptance: 124 tests,
123 passed, one private-source opt-in skipped. No paid LLM calls, production mutation
or deployment. Combined revision track now 5/8 accepted (62.5%); backend 3/5.

Frontend orchestrates the explicit prepare-task-layout write after confirmed
structural commit, then task-layout read. The read endpoint remains mutation-free.
Preparation/reload failures report "structure saved, reading preparation failed";
recovery repeats only layout-only preparation/read, never enhance/raw extraction.
Existing source-not-ready recovery still requires separate explicit intake Prepare.

## Full-Source Capacity Preflight Plan (2026-10-20)

FULL-03 is implemented. Depends on FULL-01 (llm capability semantics)
and FULL-02 (document_structure complete-source assembly and budget decision).
Reuse the planner's read-only assembly/budget function in preflight and execution;
never maintain a separate character-based approximation in the coordinator.
Source readiness and execution capacity are separate decisions. No OCR, extraction,
LLM request, artifact cleanup or persistent mutation is allowed during preflight.

Implemented additive preflight `capacity` object:
- status: fits | exceeded | unavailable | resource_limited; reason_code.
- model, context_window_tokens, optional independent max_input_tokens,
  max_output_tokens, output_reserve_tokens, safety_margin_tokens.
- required_input_tokens, available_input_tokens, count_method
  (tokenizer | conservative_estimate | unavailable), tokenizer_identity.
- full_source, source_chars, visible_source_chars, budget_policy_version.
Unknown numeric values are null, not zero. Counts describe local estimation, not
billed usage. Do not expose source text, credentials, internal paths or provider errors.
Schema/API expose these fields; frontend presentation remains FULL-04.

Issue consent only when all existing readiness/commit checks and capacity fit pass.
Bind consent to existing source/version/revision identity plus model, prompt version,
budget policy, tokenizer/count policy and effective output allowance. Recompute before
dispatch; configuration/source changes require renewed preflight and consent. Use a
bounded request-local assembly snapshot where useful, not hidden persisted derivatives.
Existing commit-time concurrency/atomic cleanup guards remain authoritative.

Preflight/consent recheck rejection codes: model_capacity_exceeded, capacity_unavailable and
local_resource_limit. Capacity/resource rejection uses the existing 413 category;
unavailable trustworthy model capability uses 503. Preserve compatibility for existing
reason handling while exposing typed safe details consistently in preflight/execution.
Preflight can return a successful diagnostic response with execution_available=false;
this is not a successful enhancement. No confirmation token on rejection.
Execution rechecks must return the same reason and never call the LLM when rejected.
Keep hierarchy, version and interactions unchanged on all pre-commit failures.

Logs correlate request/document/model/policy, source coverage, count method, full
prompt token estimate, reserves, limit and rejection code without source excerpts.
Tests must cover preflight/execution parity, late configuration changes, unknown
capability, exact/over limits, local resource ceiling and zero-call/no-write rejection.

### FULL-03 Delivery (2026-10-20)

`inspect_enhancement_capacity` delegates through the preparation pipeline to the
same builder TOC observation and splitter full-source assembly used by execution.
Source resource guards precede TOC detection. No alternative character budget,
OCR, extraction, generation or persistence is introduced. Source readiness stays
true when capacity fails, but execution_available=false and no token is issued.
Storage/commit unavailability also no longer issues consent; capacity is null when
not evaluated. Resource/unknown failures keep unknown token counts null.

Consent v2 binds existing source/revision/options plus language and a digest of
the exact prompt, prompt version, model/endpoint/limits, output allowance, counted
input/method, budget version/margins/framing and local resource policy. Tokenizer
identity is model plus counting method, not billed usage; method/count changes
invalidate consent. Process-local token signing still invalidates across restarts.
The fingerprint is private and is not returned in the public capacity DTO.

Cheap signature/expiry checks precede the document lease. Inside the lease, state
and full capacity are re-read once; there is no shared persistent prompt cache.
A request-local ContextVar guard compares the approved fingerprint immediately
before dispatch using the actual prompt/current budget. It resets on success/error.
Later policy/prompt changes reject with enhancement_configuration_changed (409);
an already-exceeded budget retains the existing planner 413 error path. Existing
commit-time source/revision and atomic cleanup checks remain unchanged.

Preflight returns HTTP 200 diagnostic states. Recheck capacity errors preserve the
legacy string `detail` and add a schema-filtered `capacity` object (413 or 503).
Missing/stale/expired consent stays 409. No raw prompt, source, signing token body,
private paths or exception content enters diagnostics. FULL-04 must distinguish
ready from execution_available and consume these fields, not estimate locally.

Validation: 139 tests discovered, 138 passed, one intentionally skipped private-source
opt-in case. Includes 11 new capacity/consent tests plus existing planner/source/API
tests and 32 PostgreSQL tests in a uniquely created/migrated/removed schema. Covers
exact/over limit, nullable failures, tokenizer fallback, config changes before and
after authorization, real successful dispatch, failed/atomic commits and unchanged
database/source on rejection. git diff --check passed. No paid model, user-document
mutation, Docker restart or frontend changes. Overall track 3/5 (60%).

## Enhancement Preflight And Consent (2026-10-19)

Preflight/consent handoff implemented; owned by ENH-APP-01. Actual enhancement
execution is capability-gated by PREP-03: PostgreSQL migration 003 and the wired
version-aware profile store are required before paid work. The source contract is in
`../document_preparation/module-detailed-design.md`, Source-Preserving Enhancement.
The former enhanced `reparse_document_structure -> prepare_and_load(force_rebuild=True)`
coupling is removed; Common reparse retains its separate preparation behavior.

Propose a read-only `POST /documents/reparse-structure/preflight`: resolve accepted
source readiness and affected interaction scope without loaders, OCR, LLM calls or
persistence mutation. Return readiness/reason, structure version, cleanup inventory
(unknown counts explicitly marked), and a short-lived server-issued confirmation
token only when ready. Never expose canonical text or filesystem paths to the UI.

Bind consent to namespace/document identity, source/canonical hashes, structure
version, interaction revision, selected parser mode/toc_hint and policy version.
Current execution requires explicit `discard_interactions: true` and that token.
ENH-REV-03 retains consent but narrows cleanup to structure_changed outcomes as above.
Missing consent is `409 confirmation_required`; expired/stale consent requires a
fresh preflight and confirmation. Legacy intent values remain accepted but never
bypass destructive-operation consent. Readiness failures identify explicit Prepare
as recovery; they do not start it automatically.

Validate before paid work and again at commit. Coordinate document-scoped single
flight so duplicate submissions cannot create concurrent paid runs. Do not hold a
database transaction throughout LLM generation. Source or interaction changes during
generation must reject replacement, preserving the latest persisted state. DB owns
revision/commit enforcement, not a UI timestamp or artifact count comparison.

Cancellation before submission is side-effect free. A browser abort or timeout after
submission does not prove server rollback: reconcile persisted version/state before
retrying; recover the failed stage while preventing duplicate commit/cleanup. Return committed version
on structural success; follow it with automatic explicit layout-only preparation
before read. A preparation/reload failure is separate from the committed transaction.
Public activation requires ENH-DB-01 guards and an atomic supported storage adapter.

DB-01 supporting handoff: resolved PostgreSQL reading targets carry the server-owned
storage structure version before generation. Conflicting client versions are rejected;
generation metadata retains this version through persistence, preventing late old
interactions from being saved against a new hierarchy. File-backed targets keep their
existing nullable-version contract.

PREP-02 supporting handoff: the internal `enhancement_candidate` context manager
uses validated consent and a document lease, invokes only the pipeline candidate
builder, and yields state/candidate without writing. The lease remains held until
the caller exits, including exceptions. Public enhanced reparse now calls the guarded
repository commit inside this context and returns its structure version. Candidate
success alone cannot clear UI data.

PREP-03 acceptance: 30 real FastAPI/disposable PostgreSQL cases and real desktop/mobile
browser flow passed. File-profile derived advice is suppressed through committed DB
provenance, not a post-commit file deletion. Layout preparation after commit remains
a separate explicit write. Production was not restarted; local startup now applies
migration 003 before starting the API. See preparation design for evidence and limits.

Implementation checkpoint (2026-10-19): `enhancement_preflight.py` implements the
read-only service contract and signed five-minute consent tokens. The binding covers
the validated source snapshot, namespace/document, structure version, interaction
revision and request options. Counts/notes remain nullable, never inferred as zero.
The token contains a binding digest, expiry and nonce, not source text. Secrets are
process-local: restart or another worker invalidates consent and requires preflight
again. Shared-worker signing/key lifecycle remains required before distributed use.

The public preflight route is wired to the coordinator's storage-backed reader.
PostgreSQL reads one REPEATABLE READ, READ ONLY snapshot covering document/version,
raw source metadata, chapters, sections, task units, content blocks and artifacts.
A fingerprint of full row payloads plus PostgreSQL xmin detects in-place edits,
including embedded interactions, not merely changes in counts. This fingerprint
is optimistic evidence; DB-01 still owns commit-time locks and all-writer enforcement.
Source bytes are validated under the configured raw intake root without loaders.
Ambiguous/missing sources, failed validation and unsupported file storage return
ready=false with a reason and no token. Unknown embedded counts remain null;
known persisted notes are explicitly flagged. Metadata-only logical duplicates
are not guessed or double-counted.

Ready means the source/impact can be confirmed, not that execution is enabled:
`execution_available=false` is explicit. Session advisory locks provide a DB-owned,
namespace/document-scoped single-flight handoff without holding a transaction during
generation. The coordinator validates before acquisition and again under the lease;
busy documents return 409, and exceptions release the lease. This does not itself
prevent interaction writers from changing state; commit guards remain required.
Enhanced reparse is gated in the coordinator: missing/invalid/expired/stale consent
returns 409. PREP-03 checks execution capabilities after validation; missing guards
return a specific 503 reason. It cannot fall through to legacy forced Prepare.
Common/manual reparse and explicit Prepare routes are unchanged by this checkpoint.

Evidence: eight live PostgreSQL tests in disposable schemas cover real route/storage
readback, unchanged persisted state, artifact create/edit/delete, embedded notes,
namespace isolation, source changes, lease exclusion/release and a concurrent insert
during snapshot reading. Nine consent tests and fourteen PDF-mode contract tests
also pass. No user documents or paid providers were used, and no service restart.
These are the earlier APP-01 checkpoint tests. PREP-03/DB-01 now exercise committed
versions and browser timeout reconciliation with actual disposable persistence.

## Enhancement Semantics Correction (2026-10-19)

The UI exposes one LLM enhance operation. Both legacy intent wire values reach the
same Common-result optimization policy in document_structure; neither implies that
all accepted headings are original or generated. Consumers use accepted per-heading
origin metadata. Existing explicit reparse, read-only projection and failure
preservation boundaries are unchanged. Historical two-purpose/staged notes are not
the current planner policy; see document_structure's dated unified correction.

## READER-01 Projection (2026-10-19)

Task-layout projects whitelisted TOC declaration/observation, enhancement intent,
implicit-single-body and declaration-unconfirmed fields from accepted provenance.
Unknown/invalid values remain null. Enhanced present/not-found or unusable evidence
is projected as unconfirmed, not a fabricated TOC. Section title_origin is matched
by exact accepted source interval and limited to source/implicit/generated; no
frontend inference, resolver call or read-time persistence mutation is added.
Evidence: `scripts/test_reader_provenance.py` (4 tests), including real route,
coordinator, isolated file-store readback and unchanged storage bytes. This is
supporting READER-01 work, not an additional backend checklist completion.

## INTERACT-03 Policy Handoff (2026-10-19)

Coordinator requests bind server prompt versions, ignoring optional client versions
as policy authority. Thinking latest/by-ID reads and block availability project
policy mismatch as stale without modifying sessions or generating questions.
Availability uses the same profile-first language selection as generation. Explicit
new-question generation retains historical sessions; submit/retry keep the stored
question identity. Existing preparation/source-resolution behavior is unchanged.
The section_tasks delivery owns this task/count; real API/coordinator tests use
temporary file storage and fake preparation/providers, with no deployment claim.

## NATIVE-05 Reparse Activation (2026-10-19)

The existing reparse handoff now executes validated v2 plans through preparation;
no additional app-level write path is added. Isolated TestClient tests exercise
real Prepare/Reparse route-to-pipeline flow for generate_reading_outline. Failed
enhancement reports failure and leaves persisted source hierarchy available.
Origin/count/coverage provenance is internal storage metadata; the lightweight
public projection remains unchanged pending explicit UI/API work. The NATIVE-04
staged rejection below is a historical snapshot, not current runtime behavior.

## NATIVE-04 Intent Handoff (2026-10-19)

Explicit reparse forwards non-default enhancement_intent to prepare_and_load.
Default restore_source preserves the existing invocation. Generated outline
application currently rejects before preparation writes; planning-only v2 candidates
must await the NATIVE-05 applier. API forwarding and staged-rejection contracts are
covered by `test_pdf_extraction_mode_contract.py`; no new task-layout read mutation.

## NATIVE-03 Reparse Handoff (2026-10-19)

`reparse_document_structure` accepts request-local optional `toc_hint` and forwards
it to explicit forced preparation. Omitted internal calls retain their existing
behavior. Read-only task-layout projection and persistence ownership are unchanged.
New Common-policy provenance is stored by the builder, not yet exposed by this
module's lightweight provenance DTO. Evidence: Common-policy preparation tests,
reparse route contract and existing structured-reuse regression. LLM prompt
consumption and extended public provenance remain subsequent task scope.

## 2026-10-15 Task-Layout Latency Investigation And Follow-up

LAYOUT-01..03 completed. Historical production evidence below is read-only; new measurements use
disposable JSON storage and fake external dependencies, not production requests.

### LAYOUT-03 Persisted Page Evidence (2026-10-16)

Ordinary layout reads never request page evidence (including readiness failures).
Explicit `include_anchor_page_evidence=true` passes the same persisted document
snapshot used by the projection to preparation's `load_task_layout_page_boundaries`.
This dedicated read checks extraction identity schema/mode, saved text SHA-256,
current source-file SHA-256 and stored page span ordering/ranges/types before
returning boundaries. It hashes source bytes only; it does not parse PDF pages,
render, OCR, call language/profile/LLM services, reload structure, or write artifacts.
It does not recompute extraction settings/engine fingerprints or change modes.

Unknown/legacy provenance, missing or changed source, unsupported identity schema,
text mismatch and invalid/missing spans yield no page evidence. Projection retains
character anchors with `page_boundaries_unavailable`; no fabricated page numbers or
read-time migration. Explicit preparation is needed to establish missing evidence.
`task_layout.persisted_page_evidence` records bounded outcome/reason and count, no
source content. Hashing is still O(source bytes), but only on explicit evidence
requests, not normal Open. The optional read snapshot does not guarantee freshness
at later commit: manual commit still uses its original source-evidence and source-hash
validation, including its existing legacy fallback/error policy.

Validation: 8 page-evidence tests; 11 readiness tests; 10 PDF-mode preparation tests
(including persisted three-page OCR spans with no additional extraction/render);
persistence-cache smoke and manual preparation/commit source-evidence regressions.
All pass. Native fixture preparation uses real PDF extraction; OCR execution is
stubbed by the existing OCR fixture, not an OCR accuracy or production-speed claim.
No production data mutation or deployment. **[Code-Confirmed / Controlled-Test Evidence]**

### LAYOUT-02 Readiness Contract (2026-10-16)

Ordinary `get_document_task_layout` reads the repository directly, validates hierarchy,
cache metadata/options and duplicate IDs, then projects persisted units. It never
calls preparation, task-unit resolution, repair or writes. Missing structure/units,
stale options/source/resolver metadata, invalid structure/hierarchy and duplicate IDs
raise `TaskLayoutNotReadyError`. HTTP returns 409 with a string `detail` identifying
the reason and directing the caller to explicit prepare with matching split options.
`refresh_task_units=true` on the read route also returns 409 without work. Storage
outages remain errors, not silently treated as cold cache. No automatic retry/prepare
is introduced into UI Open or read services.

`POST /documents/prepare-task-layout` owns base preparation followed by
`ensure_document_task_layout`: reuse or resolve/persist/repair task units, then reload
and validate repository state before success. Response projection also reads back
persisted IDs. A successful base prepare alone is not reader readiness. Management
`ready` and `/documents/prepare` retain their existing structured-document meaning;
no new management boolean is claimed. The successful prepare-task-layout projection
is the readiness evidence for the supplied split options, not a permanent guarantee.

Additive `layout_only=false` defaults to the full explicit flow. `layout_only=true`
prepares units from existing structure only, for explicit reparse/TOC commit follow-up;
it does not parse the raw source, change PDF mode, or overwrite the manual hierarchy.
Missing structure returns 409. `force_rebuild` and `include_anchor_page_evidence`
are rejected with 400 in this mode. PDF metadata is unknown with no effective mode,
not a claim that the default fast mode ran. At LAYOUT-02 delivery, optional evidence
still used the legacy source-evidence path; LAYOUT-03 above replaces that path.

Library's explicit post-upload Prepare now calls prepare-task-layout with progressive
splitting, matching ordinary UI reads; failed readback remains an error and can be
retried. After explicit structure repair/TOC commit, UI uses layout-only preparation.
Upload automation, cancellation, selectors and comprehensive Open gates remain in
their own future UI tasks. Existing documents with stale layout are not silently
migrated on Open; they require explicit preparation through the API (or the current
post-upload Prepare control where available).

Validation: 10 readiness tests cover poison-write/resolution/raw guards, cold/stale/
invalid/duplicate cases, explicit repair, persistence readback, storage failures,
HTTP 409, failed base preparation and layout-only preservation. Existing cache smoke,
5 logging tests, 11 PDF API contract tests, TypeScript/Vite build and intercepted
Playwright Library identity/409/prepare-failure/retry/success/service cases pass.
The logging baseline now observes zero resolver/language calls in all three read
states: cold and structure-only reject, layout-ready projects. The timings below
are the historical LAYOUT-01 baseline, not current cold-read behavior. All test data
is temporary or intercepted; no production DB/LLM calls or Docker deployment.
**[Code-Confirmed / Controlled-Test Evidence]**

### LAYOUT-01 Measurement Contract And Evidence

`task_layout.total` contains prepare/load, cache orchestration, profile read,
optional source-evidence loading and unit DTO spans. Cache orchestration contains
unit resolution, repository update or duplicate-ID repair spans. Preparation adds
`structured_cache_read` and `structured_runtime_read`; `embedding.single/batch`
measures SDK orchestration, including SDK batching/retries. No inputs are logged by
these spans. Cache reasons distinguish missing units/artifacts/metadata, source,
split mode, top-k or resolver version mismatch, valid cache and explicit refresh.

Every `log_stage` now records `parent_stage_id`, with ContextVar restoration after
success/failure and inheritance into worker threads. Group by request ID and span
ID; subtract only direct child durations from their parent. Remaining time includes
uninstrumented projection/normalization and logging overhead, not proven network
latency. Timings are rounded to 0.1 ms; very small residuals can have rounding error.

At INFO logging level, an idempotently installed httpx filter emits one content-free
`external_response` per POST response to embeddings, responses or chat/completions,
including non-2xx/retry responses. It records endpoint kind, status and active span,
never URL, headers, payload or exception text. This counts completed HTTP exchanges,
NOT SDK method calls or transport attempts without a response; connection failures
have no response count. SDK span failure/elapsed time remains separate. This adapter
depends on httpx's structured HTTP log event; its MockTransport regression must be
rerun when upgrading httpx. Existing httpx diagnostics are not rewritten.

`scripts/test_task_layout_logging.py` runs the real preparation pipeline/coordinator
against the same fixture/options in separate temporary JSON stores. Raw loading,
parser, profile/language and semantic resolver are test substitutes. Setup is excluded
from measured requests. One local run on 2026-10-15 produced:

| State | Total ms | Prepare/load ms | Cache ms | Residual ms | Unit resolver calls | Language stub calls | HTTP responses |
|---|---:|---:|---:|---:|---:|---:|---:|
| cold | 3.7 | 2.4 | 0.9 | 0.3 | 1 | 2 | 0 |
| structured-ready | 1.2 | 0.3 | 0.6 | 0.3 | 1 | 1 | 0 |
| layout-ready | 0.5 | 0.3 | 0.0 | 0.2 | 0 | 0 | 0 |

Other direct children (profile, evidence, DTOs) are also included in residual
calculation; cold evidence was 0.1 ms. These are reproducible orchestration baselines,
not real-book performance estimates or claims that the historical 26-second request
is fixed. They expose redundant source-language work on misses and no such work on
hits. Remote latency and PostgreSQL performance still require a deployed trace.
Five tests cover these states, nested failures, concurrency/worker correlation,
one SDK batch producing three HTTP responses (429/200/200), no-response transport
failure and sensitive-content exclusion. Existing cache and upload/preparation
logging regressions also pass. No read/write behavior changes in this task.
**[Code-Confirmed / Controlled-Test Evidence]**

Validation caveat: `test_task_layout_hierarchy_first_read.py` fails its first
cache-hit expectation because fixtures use `task_unit_resolver_v2` while runtime
requires `task_unit_resolver_v3_section_scoped`. Re-running the same test with the
HEAD coordinator loaded in memory reproduces the identical failure; no worktree
files were reverted. This pre-existing fixture mismatch is not fixed by LAYOUT-01.
Anchor-evidence DTO and structured-reuse-before-raw-load regressions pass.

The approximately half-minute report has a matching four-page `《歸來》` request
`0ffc6cfaa88a48858626bdec088308eb` in `data/logs/deep-reader.log`:

| Stage | Observed ms |
|---|---:|
| HTTP task-layout total | 26,679.9 |
| Base preparation (nested in total) | 14,982.8 |
| Raw load (nested in preparation) | 533.8 |
| Language detection (nested in preparation) | 5,248.4 |
| Profile generation (nested in preparation) | 8,514.4 |
| PDF layout evidence (nested in preparation) | 24.1 |
| After preparation to HTTP completion (residual, not one instrumented stage) | 11,697.1 |

Preparation logs report zero renders/OCR calls. The same request re-inspects the
native source after preparation and finishes a third Responses API request before
returning. This strongly matches the manual source-evidence language-detection
path below, but the remaining 11.7 seconds are not all individually timed LLM work.
Language/profile alone account for about 51.6% of total latency. A separate TXT
variant of the same title (`a58eff86b34d49f9bfb13d82b823137d`) took 24,807.5 ms:
10,849.0 ms language detection, 5,201.9 ms profile generation, 16,275.1 ms base
preparation, then two embeddings responses and another Responses call. These are
different source variants/requests, not a before/after benchmark.

Another native PDF, `家醜不可外揚` (`5b5c4803f8864d4cb4d8bb95ad3f14e6`), took
37,405.3 ms: preparation 17,818.5 ms, language 9,900.6 ms, profile 7,348.7 ms,
raw 203.0 ms and layout 13.0 ms. Its later phase includes three embeddings
responses, another raw inspection and a Responses call; zero OCR was observed.
HTTP completion log timestamps alone cannot establish each remote call's elapsed
time. Add missing stage timings before optimizing those residuals.
**[Code-Observed]**

Pre-LAYOUT-02 boundary gap: `get_document_task_layout` called `prepare_and_load(base)`
even on the nominal read route; a missing structure can therefore cause raw load,
language/profile generation and persistence. `_get_or_refresh_task_layout_document`
can resolve task units and `update_task_layout` on cache miss, and even repair/save
duplicate IDs on a hit. Optional anchor evidence is skipped only for cache hits
with `include_anchor_page_evidence=false`; a miss calls
`_load_task_layout_anchor_page_boundaries` -> manual source evidence -> raw load
and language detection again. These current behaviors contradict the intended
lightweight, no-hidden-write read contract. Earlier documentation describing
Open/read as side-effect-free is an intended boundary, not an accurate assertion
for the cold/cache-miss path. Do not preserve or expand this drift as an optimization.
**[Code-Confirmed]**

Planned tasks:

1. LAYOUT-01 (complete) adds request-correlated stage spans for task-unit resolution/embedding
   orchestration, cache reason, repository read/write and optional anchor evidence.
   Track calls and durations independently; no raw/prompt/credential logging or
   summing nested timings. Establish cold/structured-ready/layout-ready baselines.
2. LAYOUT-02 (complete; contract above) separates explicit prepare/refresh writes from ordinary layout reads.
   `/documents/prepare-task-layout` currently calls the shared get route after base
   prepare, so move/reuse explicit orchestration rather than merely disabling work
   inside get and breaking prepare. Ordinary reads of missing/stale layout return
   an actionable readiness failure without OCR/LLM/embedding/repair/writes. Explicit
   preparation owns valid task-unit IDs/cache and reports reading readiness only
   after persistence/readback. The existing management endpoint's `ready` currently
   means a structured document exists, not valid task-unit cache; preserve that
   distinction and specify any additional readiness exposure before implementation.
   Regress routes and UI service error handling together, retaining lightweight DTOs.
3. LAYOUT-03 (complete; contract above) makes page evidence opt-in on ordinary reads regardless of cache status;
   persisted hierarchy character anchors remain valid fallback. Explicit TOC evidence
   should use a boundary-only handoff without redundant language/profile LLM work,
   with source/version validation and read-only semantics. Do not remove manual
   commit validation or accept stale source evidence to save time.

The preparation checklist's existing unchecked language-cache-priority task remains
the owner of broader language cache optimization; it is not duplicated or marked
complete here. UI LIB-03/04 consume explicit readiness and must not infer server
cancellation from AbortController. These three app tasks plus five UI and three
loader tasks originally totaled eleven pending items. The later three-mode decision
supersedes that total: 17 across UI (6), loader (5), preparation (2), main (1),
app (3), all pending. PDF-FIX-01 is replaced by deferred PDF-AUTO-01. App's three
LAYOUT tasks remain valid: fast PDF extraction does not remove language/profile
latency or authorize hidden preparation. Mode pass-through belongs to main's
PDF-MODE-01 with preparation PDF-MODE-04/05; do not count it again here.

## 1. Module Purpose

`app/` 是 runtime orchestration 層，承接 API 請求後的跨模組協調：
- QA 協調（`QACoordinator`）
- section/chapter task 協調（`SectionTaskCoordinator`）

**[Code-Confirmed]**

## 2. Position in Overall Architecture

- API Layer 與 Core Layers 之間的 orchestration layer（Application service 層）

## 3. Key Files

| File | Responsibility | Notes |
|---|---|---|
| `app/qa_coordinator.py` | QA 主協調（prepare+retrieve+prompt+LLM+session） | 使用 DI container 組裝依賴 **[Code-Confirmed]** |
| `app/section_task_coordinator.py` | section/chapter 任務協調與 task-layout projection | hierarchy-first + diagnostics projection **[Code-Confirmed]** |
| `app/coordinator.py` | backward-compatible alias module | naming migration compatibility **[Code-Confirmed]** |

## 4. Main Responsibilities

1. 協調 prepare/load 與 downstream module 使用順序。 **[Code-Confirmed]**
2. 對 task layout 進行 runtime projection 組裝。 **[Code-Confirmed]**
3. 協調 summary/quiz 任務執行與 artifact 更新路徑。 **[Code-Confirmed]**
4. 輸出 enhanced recommendation 與 profile diagnostics。 **[Code-Confirmed]**
5. 維持 fail-fast 邊界（如 hierarchy inconsistency / migration required）。 **[Code-Confirmed]**

## 5. Non-Responsibilities

1. 不應成為 parser rule 定義層。 **[From HLD]**
2. 不應自行改寫 structured model schema。 **[From HLD]**
3. 不應在 task-layout read path 做 profile 隱性回寫。 **[Code-Confirmed] + [From HLD]**

## 6. Important Data Structures / Contracts

- `AskExecutionResult`
- `SectionTaskResult`
- `DocumentTaskLayout`
- `EnhancedParseRecommendationDTO`
- `ProfileStructureDiagnosticsDTO`
- `ResolvedTaskUnit`（coordinator internal runtime contract）
- `ReadingInteractionTargetDTO`
- `ReadingInteractionResponseDTO`
- `CriticalThinkingSessionStoreResult`

## 7. Coordinator Responsibility Slices

| Slice | Coordinator Scope | Must Not Do |
|---|---|---|
| Prepare orchestration | 呼叫 pipeline，整合 readiness/error | 自己實作 parser/repository 邏輯 |
| Task layout projection | assemble DTO + diagnostics | hidden persistence write |
| Summary/Quiz write path | 呼叫 service + repository 寫入 | 直接繞過 repository contract |
| Recommendation/diagnostics | projection 與 advisory訊號輸出 | 直接控制 parser 行為 |

## 8. Module Relationships

- depends on:
  - `document_preparation/`
  - `document_structure/`
  - `section_tasks/`
  - `profile/`
  - `retrieval/`, `question/`, `prompts/`, `session/`
- used by:
  - `main.py`
- reads from:
  - structured/profile artifacts（透過 pipeline/repository）
- writes to:
  - task artifacts（透過 repository）
- projection relationship:
  - task-layout + diagnostics DTO projection

## 9. Main Flows Involving This Module

1. QA ask flow（prepare_and_load -> context build -> prompt -> LLM -> session update）。 **[Code-Confirmed]**
2. task-layout flow（load/refresh layout -> build chapters-first response）。 **[Code-Confirmed]**
3. section summary/quiz flow（cache check -> resolve -> generate -> persist）。 **[Code-Confirmed]**
4. chapter summary/quiz flow（chapter_id-first target resolution）。 **[Code-Confirmed]**
5. reparse flow（explicit parser mode re-run）。 **[Code-Confirmed]**
6. reading interaction target resolution flow（document/chapter/section/task-unit, id-based hierarchy path）。 **[Code-Confirmed]**

## 10. Error Semantics (High-Level)

| Scenario | Expected Behavior |
|---|---|
| missing/invalid section target | fail-fast with explicit error |
| ambiguous chapter title (title-only) | fail-fast; encourage id-based target |
| legacy sections-only document at hierarchy-required runtime path | fail-fast migration-required semantics |
| severe hierarchy inconsistency | fail-fast; no legacy runtime mask |

**[Code-Confirmed]**

## 11. Persistence / Side Effects

- read persistence：是（透過 pipeline/store/repository）
- write persistence：是（summary/quiz/task-layout metadata 更新）
- mutate structured document：是（透過 repository update methods）
- generate runtime projection：是（task-layout + diagnostics）
- call LLM：間接（透過 task services / QA path）
- diagnostics only：部分（profile_diagnostics 為 projection）

## 12. Known Legacy / Compatibility Behavior

1. `app/coordinator.py` 保留 alias compatibility。 **[Code-Confirmed]**
2. `_find_chapter_or_raise` 為 hierarchy-only chapter target resolver：`chapter_id` 優先，`chapter_title` 僅作 hierarchy 內 secondary lookup；缺失/歧義皆 fail-fast。 **[Code-Confirmed]**
3. runtime path 不再回退 root `sections`、不再合成 legacy chapter、不再使用 legacy title/section fallback 掩蓋 hierarchy 問題。 **[Code-Confirmed]**
4. `_resolve_task_layout_sections` 對 legacy sections-only / severe inconsistency 皆採 fail-fast。 **[Code-Confirmed]**

## 13. Current Risks

1. risk：coordinator 職責過重
- why：跨層邏輯集中，易造成耦合增長
- guardrail：保持 module contract 文檔與測試邊界

2. risk：runtime fallback 收斂不完全
- why：若未持續維持 hierarchy-only fail-fast，可能在後續維護時回退為隱性 compatibility path
- guardrail：保留 hierarchy-required 測試與文檔邊界，避免重新引入 root-sections/title fallback runtime 行為

3. risk：diagnostics 與 recommendation 混用
- why：可能誤把 advisory 當 control signal
- guardrail：維持 projection-only 契約與欄位語義註釋

4. risk：chapter title ambiguity 若回退 title-only
- why：重複標題文檔（如 part/chapter 作品）會失敗或不穩
- guardrail：優先 chapter_id targeting contract

## 14. Open Questions for Maintainer

1. `section_task_coordinator` 是否需要拆分成 read/write/application service 子層文檔（先文檔化，不改程式）？
2. recommendation 與 diagnostics 是否要在 API 語義層分離文件？

## 15. Suggested Next Documentation Improvements

1. 增加 coordinator flow sequence diagrams（QA ask / task-layout / summary）。
2. 增加 fail-fast error taxonomy（migration required / inconsistency / cache stale）。
3. 補 runtime read-path vs write-path 邊界圖。

## 16. Future Direction Note: Rich Content Interaction API Preparation

> 本節為 future-task documentation/preparation，非當前 implementation。 **[Doc-Confirmed]**

1. app coordinator 未來可擴展為 content-block lookup/read API 的 orchestration boundary（request normalize、identity resolve、response assemble），但本輪僅記錄方向，不代表已實作 endpoint。 **[Maintainer-Confirmed] + [Inferred]**
2. target resolution 順序的 future contract 應為：`chapter_id/section_id -> task_unit_id -> content_block_id`；id 缺失或不一致時 fail-fast。Public reading-interaction app method signatures now accept `content_block_id` as an additive pass-through field, while actual block resolution remains unsupported until the resolver/source-context task lands. **[Code-Confirmed] + [Future Direction]**
3. 若 `content_block_id` 缺失（且該 API 要求 block 粒度），應 fail-fast，不得隱式降級為 title/全文掃描。 **[Inferred]**
4. 若 `content_block_id` 重複或產生歧義，應 fail-fast，避免非決定性 target 綁定。 **[Inferred]**
5. 若 hierarchy path 無效（chapter/section/task-unit 與 block 關聯不成立），應 fail-fast，不得容忍跨層錯配。 **[Inferred]**
6. future targeting path 不得回退為 title-only lookup，也不得回退 root `sections` 或 synthetic legacy hierarchy。 **[Code-Confirmed] + [Maintainer-Confirmed]**
7. `task-layout` API 仍保持 lightweight metadata/projection read path，不承載 heavy rich content body。 **[Code-Confirmed] + [Maintainer-Confirmed]**
8. rich content body 應透過 on-demand content API path 讀取，並與 task-layout metadata path 分離。 **[Code-Confirmed] + [Maintainer-Confirmed]**
9. app layer 在 rich-content 方向中仍是 orchestration 層，不是 parser authority，不主導 parser strategy。 **[Code-Confirmed] + [From HLD]**
10. app layer 不應擁有 artifact persistence internals；write path 仍由 service/repository 邊界承擔。 **[Code-Confirmed] + [From HLD]**
11. rich-content interaction request path 不得 hidden mutation，不得觸發 profile write-back；diagnostics 仍為 projection-only。 **[Code-Confirmed] + [Maintainer-Confirmed]**
12. Current implemented slice is limited to schema/route/app-boundary pass-through for `content_block_id`; it does not add content-block lookup, source slicing, artifact persistence, or successful block-level generation behavior. **[Code-Confirmed] + [Future Direction]**

## 16.1 Future Direction Note: Document Management And Upload Orchestration

> 本節規劃 file-management / upload 的 app-layer orchestration；不代表目前已實作。 **[Future Direction]**

1. The app layer should own document-management orchestration between API routes,
   raw-source intake, catalog/status lookup, preparation readiness, and lifecycle
   mutation policy. **[Future Direction]**
2. Upload orchestration should register or store a raw source candidate, validate source
   metadata through lower layers, return lightweight status, and stop before prepare
   unless the user explicitly starts a prepare flow. **[From HLD] + [Future Direction]**
3. File management should read source/preparation status without calling LLM-backed
   language/profile stages, OCR, reparse, retrieval build, task-layout mutation, or
   interaction generation. **[From HLD] + [Future Direction]**
4. Conflict handling should be centralized at the app/service boundary so route handlers
   and UI do not decide whether a duplicate upload replaces, rejects, or creates a
   distinct candidate. **[Future Direction]**
5. Lifecycle mutations such as archive/delete should coordinate raw source, structured
   artifacts, profile/retrieval artifacts, and reading-interaction artifacts through an
   explicit cleanup policy. **[Future Direction]**
6. User scope and raw-file ownership/copyright boundaries from the proposal must remain
   visible in orchestration design. **[From Proposal] + [Future Direction]**
7. App orchestration should expose response-ready DTOs only; frontend upload progress,
   row selection, filters, and confirmation dialogs are not app-layer state. **[Future Direction]**

## 17. Future Direction Note: Manual Structure Override Orchestration

> 本節記錄 app-layer 對 source-agnostic manual structure override 的 orchestration 邊界；不代表目前 implementation。 **[Maintainer-Provided] + [Future Direction]**

1. app layer may orchestrate manual structure validation and explicit manual reparse for any document type, not only OCR/PDF. **[Maintainer-Provided]**
2. app layer should normalize request intent, call preparation to load required raw/page evidence, call document_structure validation/projection, and return validation/reparse results. **[Future Direction]**
3. app layer must not define parser rules or manual projection semantics; those remain owned by `document_structure/`. **[From HLD] + [Future Direction]**
4. manual validation/preview should be read/analysis-oriented and must not persist hierarchy, task-layout metadata, profile diagnostics, or artifacts. **[Future Direction]**
5. manual commit reparse is an explicit mutation path and should be separate from `/documents/task-layout`; task-layout remains projection-only. **[From HLD] + [Future Direction]**
6. successful manual commit should replace the active structured hierarchy only after validation succeeds; failure should preserve the current structured artifact. **[Future Direction]**
7. app-layer error semantics should distinguish malformed request, unresolved anchors, invalid hierarchy shape, empty ranges, and stale source evidence. **[Future Direction]**
8. This direction does not add runtime endpoints, schema fields, source code, or persistence behavior in this documentation pass. **[Doc-Confirmed]**

## 18. Page-Backed TOC Anchor Orchestration

> 本節支援 UI TOC editor 的 page-first anchor UX。
> 目前已完成 task-layout coordinator DTO prefill orchestration、public API schema/route mapping、read-only page-boundary handoff、以及 page-backed manual validation/commit orchestration。 **[Code-Confirmed]**

1. The app layer should orchestrate two related flows:
   - read/projection flow: expose current structure anchor evidence for UI edit-existing prefill **[Code-Confirmed at coordinator DTO level]**
   - explicit mutation flow: validate and commit manual `page_range` anchors when source page evidence is available **[Code-Confirmed]**
2. The app layer should not compute parser semantics, page boundary mapping, or hierarchy projection itself; those remain delegated to `document_preparation/` and `document_structure/`. **[From HLD] + [Future Direction]**
3. For task-layout, `SectionTaskCoordinator.get_document_task_layout(...)` now loads optional preparation-owned page boundary evidence, delegates existing hierarchy-span projection to `document_structure.project_structure_anchor_evidence(...)`, and passes lightweight `AnchorEvidenceDTO` into chapter/section DTOs. **[Code-Confirmed]**
4. For manual commit, app loads canonical source evidence including source hash and page boundaries, rejects stale source evidence, then delegates page-anchor projection/draft building to `document_structure/`. **[Code-Confirmed]**
5. Page-backed validation/commit failures preserve the current structured artifact and do not silently fallback to common parser as a successful manual reparse. **[Maintainer-Provided] + [Code-Confirmed]**
6. `char_range` remains the source-agnostic fallback for documents with no reliable page evidence or for explicit advanced override. **[Maintainer-Provided] + [Future Direction]**
7. Current task-layout coordinator projection may load preparation source evidence to access page boundaries, but only forwards compact page boundary metadata to `document_structure/`; it does not expose raw text, page text, OCR geometry, or content blocks. **[Code-Confirmed]**
8. Public `/documents/task-layout` response schema exposes lightweight `anchor_evidence` for chapters/sections, including `page_range` when page boundaries are available and `char_range` fallback otherwise. **[Code-Confirmed]**
9. The task-layout anchor evidence path remains read-only: it does not mutate hierarchy, profile diagnostics, task-layout metadata, or artifacts. **[Code-Confirmed]**

## 19. PostgreSQL Manual Reparse Replacement Boundary

Manual `page_range` reparse validates source evidence and builds a hierarchy-only draft before persistence. PostgreSQL-backed manual commit must then use the explicit parser-level replacement boundary rather than the artifact/task-layout save method that preserves `current_structure_version` and existing hierarchy rows. **[Code-Confirmed]**

Observed failure:

- API: `/documents/reparse-structure`
- parser mode: `manual_structure`
- affected document example: `暗水幽灵`
- original failure: `duplicate key value violates unique constraint "uq_chapters_document_order"` for `(document_id, chapter_order)=(13, 0)`

Root-cause boundary:

1. Manual commit is a hard reparse mutation and should use a parser-level hierarchy replacement save. **[From HLD] + [Code-Confirmed]**
2. PostgreSQL artifact/task-layout saves intentionally preserve the current structure version and should not clear hierarchy rows. **[Code-Confirmed]**
3. Manual reparse calls `commit_manual_structure(...)` with a preparation-owned verifier; PostgreSQL validates and saves with `replace_existing_hierarchy=True` on one locked connection. **[Code-Confirmed]**
4. Validation/source/draft failures must continue to preserve the current structured hierarchy and derived resources. **[From HLD] + [Code-Confirmed]**
5. Successful manual reparse must remain explicit, transactional, hierarchy-first, and separate from task-layout read/projection. **[From HLD] + [Code-Confirmed]**

## Content Read Recovery (2026-10-21)

Maintainer requirement: profile is persistent source-level metadata, generated on
initial explicit preparation only; replacement is permitted only on explicit full
reparse. Ordinary content/layout/interaction/QA/index operations must reuse it.
Missing/corrupt existing profile must not cause silent generation or clearing.
Structure-only enhancement may retain source-level profile and suppress stale
derived advice without rewriting it. Initial/reparse enrichment is allowed;
ordinary prepare reuse must not rewrite profile bytes.

Incident: request 12025d294d31406ba8d5e5645a9c581b took 420990.5 ms.
Its 3181 blocks caused 9543 profile reads (110694 ms) and per-action full-document
loads. Selected section contains 409 units, not only the visible short paragraph.
Single-unit real probe: 1905 characters, six blocks, eighteen document loads.
Isolated same-snapshot reuse reduces full-section assembly to 3.646 seconds;
this is diagnostic evidence, not a deployed HTTP performance result.

Execution order / owner tasks: CONTENT-01 app (section_tasks supports snapshot
artifact reads), CONTENT-02 preparation, CONTENT-03 bundle_factory, CONTENT-04 UI,
CONTENT-05 scripts acceptance. Five tasks; accepted 5/5 (100%). Owner checklists
are authoritative. No parser/TOC redesign or profile-as-parser-authority changes.

Content read must load hierarchy/artifacts once per request, resolve profile once,
and reuse a request-local read-only snapshot for all interaction status checks.
Preserve policy/staleness/preflight behavior and canonical block IDs; no global
cache, mutation, LLM, OCR, embeddings or prepare on content reads. Add content-free
stage/count logs. UI uses bounded sequential batches, cancels obsolete requests,
and maintains target/order/error correctness. Backend still accepts existing bulk
clients efficiently; UI batching is not a substitute for backend correction.

Acceptance uses immutable copies of Madame Bovary.txt and the red-gamble PDF in
data/test-books/summary-note-20261021. Verify hashes before authorized product-API
deletion, verify DB/raw/derived absence, reupload only copies, Common Fast parse
and prepare layout. Test the largest body section, not only first/front matter.
Require full-section API under 10 seconds on local service, one document snapshot
and at most one profile read per request regardless of block count, zero read-time
generation/writes, unchanged profile SHA-256 across ordinary operations, and
desktop/mobile browser navigation/interaction smoke. Test missing/corrupt profile,
index rebuild, stale artifacts and cancellation with deterministic regressions.
Failures remain open; inspect and repair until live acceptance passes.

### Acceptance And Governance Record

Completed 2026-10-21: CONTENT-01..05, 5/5 (backend 4/4; UI owns its separate 1/1).
Authorized product deletion/reupload was repeated after the PDF cache-read defect
was exposed. Final reset report: private content-reset-20261021T073350.json;
acceptance reports content-recovery-20261021T073403.json and
content-recovery-20261021T074014.json under data/test-books/summary-note-20261021.
The backup manifest and its nine files remain unchanged and ignored by Git.

Madame Bovary largest section: 409 units / 3181 blocks; serial HTTP 3.562-3.976s,
final deployed concurrent-browser load 4.660-5.403s, versus incident 420.9905s.
Red-gamble PDF: 93 units / 100 blocks; serial HTTP 0.203-0.206s and final
concurrent-browser load 0.264-0.396s. All runs satisfy the stated 10-second bulk API
gate. Repeated prepare preserves layout/IDs, all five interaction reads pass,
profile SHA-256 remains unchanged. Final two-book browser tests pass 1280/390px:
Madame 6.970s mobile / 13.947s desktop under simultaneous bulk API benchmarking;
earlier serial browser pass 6.669s / 7.140s. Red-gamble about 0.6s. This distinction
avoids claiming concurrent-load measurements as isolated benchmarks.

Tests: 51 cases across content-read recovery (10), lazy PDF preparation (15),
extraction identity (9), summary service and note service (17 combined). Eight
additional script entry points pass: profile-before-structure, post-structure
enrichment, analysis read/generate, quiz read/generate, artifact store, layout
readiness, accepted-source snapshot and content endpoint. Old test doubles were
updated for accepted candidate APIs, additive provenance, generated-profile flags
and explicit temporary corrupt-fixture reseeding; no production validation was
weakened. New tests poison per-block repository writes/reads, assert constant I/O,
check current/stale policy parity including structure version, and serialize two
simultaneous initial profile requests to one build. UI build passes with existing
bundle-size warning. Browser screenshots inspected; no runtime errors or hidden
prepare/generate requests. git diff --check passes.

Markdown inspected: AGENTS.md, proposal.md, high-level-design.md,
docs/modules/index.md, progress.md; app, section_tasks, document_preparation,
profile, bundle_factory, scripts and UI detailed-design/checklist pairs.
Markdown modified and reasons:
- app design/checklist: incident evidence, snapshot-read contract, task ownership,
  acceptance and this governance record.
- preparation design/checklist: lifecycle-only profile generation and verified
  PostgreSQL canonical-text reuse, plus task completion evidence.
- bundle_factory design/checklist: remove obsolete profile rebuild/clear fallback
  and record reuse-only runtime behavior.
- profile and section_tasks designs: atomic profile persistence and request-local
  read-only artifact snapshot boundary, respectively; supporting owner tasks.
- scripts design/checklist: backed-up real-book destructive/performance acceptance.
- UI design/checklist: bounded batches, abort semantics and real-browser evidence.
- progress.md: synchronize four backend owner rows only; exclude UI aggregation.

Validation: architecture/boundaries/docs/tests PASS. No root sections authority,
hidden content/layout mutation, diagnostics profile writes, global artifact cache
or parser authority changes. Existing unrelated worktree changes preserved; no
Git commit created. Scope is this recovery track, not all repository future tasks.

## 20. Batch Task-Unit Content Orchestration

> 本節記錄 selected-section content fan-out optimization 的 app-layer boundary。Coordinator batch read 已落地，single task-unit content lookup 仍保留。 **[Code-Confirmed]**

1. `SectionTaskCoordinator.get_task_unit_contents(...)` resolves one document and one hierarchy snapshot once, then assembles content for an ordered batch of selected-section task-unit ids. **[Code-Confirmed]**
2. Batch lookup must preserve the existing hierarchy-only, id-based task-unit resolution semantics used by the single task-unit content path. **[Code-Confirmed] + [Future Direction]**
3. The app layer returns per-task-unit content DTOs in request order so the UI can render section content without issuing many concurrent browser requests. **[Code-Confirmed]**
4. Missing, duplicate, or hierarchy-incompatible task-unit ids fail fast; the batch path does not fall back to title lookup, root `sections[]`, synthetic legacy hierarchy, or `structure_nodes`. **[Code-Confirmed] + [From HLD]**
5. The batch path shares the same `segmented` semantics as the single task-unit content path; route/schema mapping controls raw-content inclusion. **[Code-Confirmed]**
6. Batch content orchestration remains read-only: no task-layout mutation, no structured hierarchy mutation, no profile diagnostics write-back, no artifact writes, and no parser strategy decisions. **[Code-Confirmed] + [From HLD]**
7. App orchestration attaches lightweight content-block interaction availability during task-unit content DTO assembly when configured interaction dependencies are present. The projection reads current analysis/quiz artifacts and latest matching critical-thinking sessions for canonical content-block targets, emits only slot metadata, and skips non-canonical segmented render blocks instead of treating them as persisted hierarchy or artifact truth. **[Code-Confirmed]**

## 21. Future Direction: Summary And Note Interaction Orchestration

> 本節規劃 app-layer 對新增 `summary` / `note` actions 的協調邊界；不代表
> coordinator methods、routes、schemas、或 persistence 已實作。 **[Future Direction]**

1. App orchestration should expose response-ready methods for summary read/generate/refresh
   only after `section_tasks` owns the summary service, prompt, validation, and artifact
   persistence contracts. **[Future Direction]**
2. Summary app methods should resolve the target once, invoke explicit higher-level
   summary preparation only for generate/refresh, and pass bounded context metadata
   through response DTOs without raw ancestor text. **[Future Direction]**
3. App orchestration should expose note read/list/save/delete methods only as explicit
   user-authored mutations. Note methods must not call LLMs, prepare/reparse documents,
   generate summaries, mutate task-layout, or write profile diagnostics.
   **[Future Direction]**
4. App response DTOs for both actions should remain free of frontend-only state such as
   drawer visibility, menu state, selected rows, draft editing state, or scroll position.
   **[Future Direction]**

## 21. Future Direction Note: Reading Target LLM Interaction Orchestration

> 本節記錄 book/chapter/section/task-unit LLM interaction 的 app-layer governance，來自 Grill-me requirements 收斂。Target resolver, REST-facing app method contracts, app-level persistence/read behavior contracts, public route wiring, and durable backend adapter assembly are implemented for analysis/insight, quiz, and critical-thinking sessions. **[Code-Confirmed]**

1. app layer exposes a target-agnostic resolver boundary for reading interactions over `document`, `chapter`, `section`, and `task_unit` targets through `SectionTaskCoordinator.resolve_reading_target(...)`. **[Code-Confirmed]**
2. Target identity is deterministic: document/book uses `doc_name`, chapter uses `chapter_id`, section uses `section_id`, and task unit uses `task_unit_id`; optional parent ids are accepted only for consistency validation. **[Code-Confirmed]**
3. Target resolution stays hierarchy-aware and fail-fast; no title-primary targeting, root `sections[]`, `structure_nodes`, synthetic legacy hierarchy, or metadata/LLM classification fallback may become authority. **[Code-Confirmed] + [From HLD]**
4. `SectionTaskCoordinator` now exposes REST-facing app methods for analysis read/generate/refresh, quiz read/generate/refresh, and critical-thinking read/generate-question/submit-answer/retry-evaluation. These methods resolve the hierarchy target once, delegate to configured service/orchestrator dependencies, and return response-safe app DTOs without raw target content. **[Code-Confirmed]**
5. Artifact read and generation are split at the app/service boundary. Read paths load persisted artifacts/sessions only and return `not_generated` when absent; they do not call LLM services or persist placeholders. **[Code-Confirmed]**
6. Generate, refresh, critical-thinking answer evaluation, and evaluation retry are explicit write paths. They are the future hooks for cost and permission gating; ordinary persisted-artifact reads stay outside the costly-action gate. **[Code-Confirmed] + [Future Direction]**
7. Initial interaction types are `analysis`, `quiz`, and `critical_thinking_session`. `analysis` and `quiz` are current artifact per target; critical thinking is multi-session per target. **[Maintainer-Provided] + [Future Direction]**
8. Critical-thinking orchestration is intentionally simple for the first version: generate one question, accept one user answer, and evaluate that answer. A generated-but-unanswered session is saved with non-completed status instead of being discarded. **[Code-Confirmed]**
9. Critical-thinking statuses include `question_generated`, `insufficient_content`, `answer_submitted`, `evaluation_failed`, and `completed`. Failed evaluation preserves the submitted user answer, is saved as a retryable session state, and can be completed through explicit retry. **[Code-Confirmed]**
9.1. Critical-thinking read semantics are read-first and current-session aware: when a read request omits `session_id`, app orchestration loads the latest persisted session for the resolved target through the session store; only absent current sessions return `not_generated`. This lets frontend target clicks reopen existing sessions without generating a new question or requiring the UI to know a session id first. **[Code-Confirmed]**
10. app orchestration should delegate target context construction to `context/`, prompt construction to `prompts/`, strict output validation to service/schema boundaries, LLM calls to provider-backed services, and artifact persistence to repository/storage boundaries. **[From HLD] + [Future Direction]**
11. Insufficient target content is a valid persisted outcome for analysis and quiz current-artifact flows, with reason metadata, so OCR noise or symbol-only task units can terminate quickly. Analysis/quiz read and explicit generate recheck persisted insufficient-content artifacts against the current resolved target so stale insufficient-content results can recover when hierarchy content becomes readable; read does not call LLM or write, while generate regenerates only after preflight passes. Critical-thinking insufficient-content question-generation results are saved as session outcomes by the app session store boundary. **[Code-Confirmed]**
12. Hard reparse invalidates/deletes derived interaction artifacts through the existing document-scoped derived-resource lifecycle; ordinary interaction read/generate methods do not own derived-resource cleanup. No historical artifact layer is required for the first version. **[Code-Confirmed] + [From HLD]**
13. Public app orchestration now covers `analysis`, `quiz`, and `critical_thinking_session` method contracts and route dispatch. Container-backed durable adapter assembly is implemented for analysis/insight, generic quiz, and critical-thinking sessions through the shared document-backed reading-interaction store. **[Code-Confirmed]**

### 21.1 Artifact-Aware Generation Orchestration

1. When generating section/chapter/document artifacts, app orchestration should request lower-level artifact summaries as optional secondary context after the target itself is resolved. **[Maintainer-Provided] + [Future Direction]**
2. The orchestration boundary should keep primary source context and secondary artifact context separate when delegating to context and prompt services. **[Maintainer-Provided] + [Future Direction]**
3. Lower-level artifact lookup should be scoped by the resolved hierarchy target and should never scan unrelated documents or use title matching as target resolution. **[From HLD] + [Maintainer-Provided]**
4. Missing lower-level artifacts should be treated as an empty secondary context, not as an error and not as a reason to auto-generate child artifacts. **[Maintainer-Provided] + [Future Direction]**
5. Generated artifact metadata should expose whether artifact-aware context was used, which lower-level artifact ids were referenced, and whether deduplication/abstraction hints were applied. **[Maintainer-Provided] + [Future Direction]**

### 21.2 REST-Facing Orchestration Plan For Frontend Contract

> 本節規劃並記錄 `main.py` route 到 `section_tasks/` service/orchestrator 的 app-layer seam。`SectionTaskCoordinator` now exposes these public method contracts and their app-level persistence/read behavior, and `main.py` routes dispatch to them for insight, quiz, and critical-thinking. Durable backend adapter assembly is code-confirmed for insight, quiz, and critical-thinking through `config/` wiring. **[Code-Confirmed]**

1. First app-layer surface is `analysis` read/generate/refresh because the frontend can render its short payload inline below book/chapter/section/unit nodes. The first vertical slice is wired through the container and public insight routes. **[Code-Confirmed]**
2. `read_analysis_artifact(...)`, `generate_analysis_artifact(...)`, and `refresh_analysis_artifact(...)` resolve the target once, delegate to the configured analysis interaction orchestrator, and return `ReadingInteractionResponseDTO`. **[Code-Confirmed]**
3. `read_quiz_artifact(...)`, `generate_quiz_artifact(...)`, and `refresh_quiz_artifact(...)` resolve the target once, delegate to the configured quiz interaction orchestrator, and return `ReadingInteractionResponseDTO`. The app layer does not decide quiz type mix. **[Code-Confirmed]**
4. `read_critical_thinking_session(...)`, `generate_critical_thinking_question(...)`, `submit_critical_thinking_answer(...)`, and `retry_critical_thinking_evaluation(...)` resolve the target once, delegate to configured critical-thinking service/store dependencies, and return `ReadingInteractionResponseDTO`. Failed evaluation preservation is validated and saved at the service/session-store boundary. **[Code-Confirmed]**
5. All operations call `resolve_reading_target(...)` before service work and fail fast for stale or inconsistent hierarchy ids. Optional parent ids validate consistency only; titles remain display metadata. **[Code-Confirmed] + [From HLD]**
6. App orchestration returns response-ready DTOs but keeps UI behavior state out of backend persistence: drawer state, menu state, inline expansion, pending answer drafts, and loading spinners remain frontend state. **[Code-Confirmed] + [From HLD]**
9. Generation, refresh, answer evaluation, and retry share a clear app-layer insertion point for future cost/quota/permission policy. Persisted-artifact reads stay outside that costly-action gate. **[Code-Confirmed] + [Future Direction]**
10. Analysis and quiz stores represent current-artifact semantics: read returns persisted current artifacts or `not_generated` without generation; persisted insufficient-content artifacts are preflight-rechecked on read and generate so readable targets are no longer masked by stale insufficient results; generate reuses completed current artifacts unless refresh is requested; refresh replaces the current artifact only after the service returns a persistable validated status. **[Code-Confirmed]**
11. `completed` and `insufficient_content` analysis/quiz outputs are persistable; `generation_failed` and `stale_target` are surfaced but not saved as current artifacts. **[Code-Confirmed]**
12. Critical-thinking session writes save generated questions, evaluation failures that preserve submitted answers, and completed retry results under the same session id. **[Code-Confirmed]**

### 21.3 Frontend Exposure Completion Plan: App-Owned Tasks

> 本節把 frontend exposure 12-task plan 中由 `app/` 擁有的任務固定為 implementation backlog。These tasks expose existing interaction foundations through `SectionTaskCoordinator` without moving service logic into the route layer. **[Code-Confirmed] + [Future Direction]**

| Task ID | Task | Scope | Completion Evidence |
|---|---|---|---|
| FE-INT-05 | Define REST-facing app orchestration method contracts | Specify and implement app methods for analysis read/generate/refresh, quiz read/generate/refresh, critical-thinking read/generate-question/submit-answer/retry-evaluation. | Implemented by `SectionTaskCoordinator` methods plus `ReadingInteractionResponseDTO`; coordinator tests prove every method resolves the target once, delegates to the correct service/orchestrator, and returns response-ready DTOs without raw target content. |
| FE-INT-06 | Define artifact persistence/read contract for public interaction APIs | Align app read/write behavior with artifact/session storage semantics: read-only reads, explicit writes, insufficient-content persistence and preflight recheck behavior, stale-target handling, and refresh replacement. | Implemented and regression-tested by app persistence contract coverage: reads do not generate, writes persist only validated outcomes, completed artifacts are reused by generate, insufficient-content artifacts are rechecked on read/generate, stale targets do not overwrite current artifacts, failed critical-thinking evaluation preserves submitted answers for retry, and hard reparse cleanup remains the repository derived-resource boundary. |
| Analysis vertical slice | Wire analysis/insight read, generate, and refresh through the app orchestration boundary. | Public insight routes call `SectionTaskCoordinator` analysis methods, the container supplies `AnalysisInteractionLLMGenerator`, `AnalysisInteractionService`, `DocumentReadingInteractionArtifactStore`, and `AnalysisInteractionOrchestrator`, and regressions prove read/write split plus compact inline response mapping. |
| Route dispatch vertical slice | Dispatch public insight, quiz, and critical-thinking routes through app orchestration methods. | `main.py` routes call the corresponding `SectionTaskCoordinator` methods for read/write operations, and route regressions prove route mapping without moving target resolution, prompt construction, service validation, or persistence internals into the route layer. |

App-owned tasks must preserve hierarchy-only target resolution, fail-fast parent-id consistency checks, no title-primary lookup, no route-owned prompt/context logic, and no task-layout/profile hidden mutation. **[From HLD] + [Maintainer-Provided]**
