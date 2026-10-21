# document_preparation Detailed Design

## Review Evidence Handoff (2026-10-21)

OCR-REVIEW-03 uses the existing generic SourceArtifact.storage_metadata handoff:
source_review is persisted in structured-document parse provenance without format
branches in this pipeline. Reviewable OCR acceptance is loader-owned; preparation
must not reinterpret a missing region as complete text. Review/edit reads never
enter preparation and never rebuild the existing profile. Fast and Auto policies
are unchanged; the explicit OCR review flag is part of its extraction identity.

## Loader Boundary Governance (2026-10-21)

**[Code-Confirmed]** LOADER-03 delivered. Preparation/cache/save no longer imports
concrete PDF/TXT loaders, reads their private paths or branches on source_format.
Both adapters supply source sessions; warm reuse verifies captured identity and
stored canonical coverage without extraction. prepare_and_load also forwards an
explicit mode; omitted session mode is Fast. source_capture/source_extract stage
logs contain timing/implementation identifiers, never source text.
New page evidence uses source_page_spans with read compatibility for pdf_page_spans.
Legacy API status is projected through PreparedDocumentAssets, never cache truth.
Typed extraction failures propagate before hierarchy save; load-only injected
compatibility readers retain their previous error collection. Source changes now
reject writes rather than merely downgrading optional outline evidence.

LOADER-03 replaces concrete-loader dispatch with a source session supplied by the
loader. Order: resolve loader -> capture identity -> validate accepted cache ->
extract once on miss -> profile under existing lifecycle authority -> structure ->
verify source unchanged -> persist provenance/evidence/diagnostics. A cache hit
must not decode/extract/OCR or regenerate profile. Format/mode validation belongs
to adapters; compatibility API fields are output projections, not cache authority.
Manual/task-layout evidence reads use persisted mappings and a public source-path
capability, never a concrete PDF class or private loader method. Missing optional
evidence is not an instruction to extract on read. Existing legacy source readers
retain explicit behavior for unprepared documents. No hidden hierarchy mutation.
Acceptance: third-format injected session, source-race rejection, identity/settings
cache misses, exact text, no eager optional evidence, unchanged three PDF strategies,
API compatibility, and read-only real-book checks from verified backups. Do not
delete user books in this refactor's acceptance; the preceding reset is recorded.
This refactor does not promise EPUB/DOCX APIs or new live LLM-enhancement results.

Delivery validation: 95 focused Docker tests, 180 PDF tests, PostgreSQL generic
extraction/inherited enhancement suite 90 cases (88 pass, two opt-in skips).
Suites overlap and must not be summed. Both backup books pass isolated cold/warm
preparation with real loaders/Common builder and controlled profile/storage;
all canonical characters retained, one extraction each and zero warm profile calls.
Live acceptance initially used an older script that forced Common/default intent;
the Red-book cache reported common_policy_mismatch, rebuilt hierarchy and invalidated
layout. No profile regeneration occurred. Explicit layout-only recovery restored
93 units. The script now has --read-only, used for final live acceptance; do not
describe the entire run as mutation-free. No book/raw file was deleted in this task.

Governance: inspected AGENTS.md, proposal.md, high-level-design.md, module index,
progress.md and doc_loaders/document_preparation/scripts designs and checklists.
Modified those three modules' design/checklist pairs to specify ownership, tasks,
implementation evidence and limitations; progress.md synchronizes only this new
backend track. Source/tests changed under implementation authority, not the sync
skill. Timestamp checks and git diff --check pass; unrelated dirty work preserved.

## Format-Neutral Extraction Plan (2026-10-21)

### Delivery Evidence

EXTRACT-01..10 complete (10/10). The existing TXT Library document was explicitly
backfilled using scripts/backfill_txt_extraction.py --doc-name "Madame Bovary"
--apply, after source/canonical coverage proof. That operation preserves structure
version/IDs, interactions and profile; it is never invoked by reads or preflight.
The CLI currently targets the supported PostgreSQL enhancement runtime; file-store
snapshot validation/readback remains tested, not a new file-store enhancement API.

New generic provenance coexists with matching legacy PDF provenance for old
consumers; conflicting records reject. PDF evidence remains an adapter sidecar,
not fabricated universal page data. Current enum support is PDF/TXT only.

Final live Madame Bovary evidence is under private data/test-books/
fa794acef055f235/20261021T083239624177Z/api-live.json and its
api-20261021T084827447743Z/result.json: direct API/model success, committed=true,
structure_changed, 5 chapters/31 sections/423 task units, three provider calls,
652001 characters preserved exactly and profile hashes unchanged. Initial live
failure exposed source-title downgrade diagnostics; semantic prompt v16 and a
located-boundary regression repair that failure without relaxing matching/slicing.
The PDF backup passed two real calls and isolated API replay/real layout (11
chapters, 30 sections, 108 units). Production hierarchy was not replaced by these
tests. Backup manifests remain immutable; evidence files stay private.

Validation: 95 focused Docker tests pass, 180 PDF Docker tests pass, PostgreSQL
enhancement regression runs 77 cases (75 pass, two opt-in skips); UI build and
1280px/390px confirmation/readiness/recovery tests pass. Suites overlap. Deployed
API/PostgreSQL are healthy; live original-book preflight is ready/executable and
capacity fits. No promise of deterministic future LLM output is made.

Governance: inspected global AGENTS/proposal/HLD/module index/progress plus design
and checklist pairs for doc_loaders, document_preparation, db, app, scripts,
document_structure and Deep_Reader_UI. Modified those seven pairs and backend
progress: extraction contract/reuse/migration owners record delivery; app/DB record
the generic validation flow; UI records source-readiness handling; scripts records
actual acceptance; document_structure records the discovered diagnostic fix.
Evidence is the named code/tests/private reports and live API response. Required
timestamps and owner counts were checked; diff whitespace validation passes.
Validation is read-only, no unrelated changes were reverted. The planning-only
governance record later in this section is historical, not this delivery's status.

**[Code-Confirmed]** EXTRACT-01..10 are implemented and verified. Prior CONTENT/PDF
completion did not close this track; the acceptance evidence below does.
This section is the forward design authority for format-neutral extraction. Older
PDF-specific implementation notes below remain historical/current-code evidence,
not a requirement that all documents use PDF identities.

### Observed Defect And Scope

**[Code-Confirmed]** TextDocumentLoader currently returns a UTF-8 string through
AbstractDocumentLoader.load. AcceptedSourceSnapshot instead requires
PdfExtractionIdentity from parse_provenance.pdf_extraction. Candidate construction
and PostgreSQL commit checks also use that key; private_book_corpus.py assumes
source.pdf and reconstructs PdfExtractionIdentity. Fixing only TXT loading therefore
cannot fix the complete enhancement path.

The reported Madame Bovary preflight returned extraction_identity_unknown with
capacity=null: this is a source-readiness rejection before capacity evaluation,
not evidence that the model cannot fit the book. The inspected TXT measured
652,001 characters / 149,663 o200k_base tokens; this is a diagnostic measurement,
not the full assembled prompt budget or a permanent model-capacity guarantee.

Deliver PDF and TXT through one contract. Reserve extension points for EPUB/DOCX;
do not implement those parsers or claim support in this track. PDF Fast/OCR/Auto
remain separate existing strategies. A TXT document must never acquire an OCR mode.

### Contract And Ownership

The common contract is implemented in doc_loaders/extraction_identity.py. The
existing field name extraction_version is retained for extractor-version compatibility.
Optional PDF mappings/providers remain on PdfExtractionResult, which adapts its
same-pass text through as_extraction_result; TXT has no artificial pages. doc_loaders owns the
format-neutral immutable extraction contracts and format adapters;
document_preparation owns accepted-source validation and preparation orchestration.
Database storage owns atomic persistence. No circular dependency from loaders to
preparation or application services is permitted.

| Contract | Required data | Semantics |
|---|---|---|
| ExtractionIdentity | schema_version, source_sha256, source_format, extractor_id, extractor_version, mode, settings_sha256 | Source hash covers original bytes. Stable canonical serialization hashes only extraction-affecting settings. Mode is adapter-defined: PDF fast/ocr/auto; TXT native. |
| ExtractionResult | identity, canonical_text, text_sha256, text_length, completion | Text digest covers the exact canonical UTF-8 text. Only validated complete output may become accepted source. Empty/unreadable/partial extraction is an explicit failure, not fabricated completion. |
| Optional result evidence | source mappings, bounded warnings/format metadata | Mappings are not mandatory for TXT, and page counts are not universal document requirements. Evidence never becomes parser authority. |
| AcceptedSourceSnapshot | namespace, document identity, structure_version, accepted extraction result, existing accepted language | Binds verified text to a specific persisted document revision; not another extraction pass. |

Identity excludes document name/path, hierarchy IDs, structure_version, model,
prompt and profile contents. Renaming does not change extraction identity; changed
bytes/settings do. Structure-only enhancement may change hierarchy/version while
preserving extraction identity and exact canonical text. Equal decoded text from
different source bytes still has distinct source identity.

Use strict schema/hash/enum validation and deterministic settings serialization.
Unknown identity schemas/formats fail closed without pretending to be PDF.
A legacy schema version and a new generic schema are distinct contracts even if
their numeric versions match; the persisted provenance key identifies the envelope.

TXT must explicitly version its UTF-8 decoding and newline policy, preserving the
current loader's universal-newline behavior. Pin BOM handling and reject invalid
UTF-8 rather than silently replacing bytes. Do not trim, normalize Unicode, repair
text or guess encodings during enhancement. Any future policy change needs a new
version/settings identity. Hash and decode the same captured bytes, not separate
reads that can observe different file revisions.

Optional mappings use zero-based Unicode character offsets [start, end), checked
against the canonical text. PDF may add one-based page locations; TXT can omit
mappings. Future adapters may supply spine/paragraph locations without inventing
pages. Mapping validation and canonical-text completeness are separate: missing
optional mapping must not imply incomplete extraction.

### Persistence, Compatibility And Lifecycle

New successful preparations will persist a generic parse_provenance.extraction
record containing identity, text digest/length and completion plus bounded optional
evidence. Reuse existing provenance storage; do not introduce a new table without
demonstrated need. Keep canonical text in its existing owned storage, not copied
into every provenance record. Preserve hierarchy-first reads and current DB
canonical-slice reconstruction when raw_text is omitted.

Read compatibility normalizes accepted legacy pdf_extraction records in memory,
retaining all original mode/version/settings hashes and provenance. Do not guess
new extractor settings or rewrite old records during reads. New writes have one
generic authority; any retained legacy record must normalize equivalently.
Conflicting generic/legacy records or malformed preferred generic records are
errors, never a reason to fall back to whichever record happens to validate.

Legacy TXT without accepted identity remains unavailable for enhancement until an
explicit migration/preparation operation validates its source. A narrowly scoped
backfill is allowed only when deterministic decoding exactly matches the complete,
verifiable accepted canonical text and the source remains unchanged at commit.
Otherwise require explicit source preparation; never manufacture proof from text
length, filename, current profile or newly computed hash alone. Identity-only
backfill preserves hierarchy IDs/version, interactions and profile bytes, and must
be atomic/restart-safe. The explicit migration entry point belongs to preparation,
not preflight, task-layout, content reads or interaction opening.

An identity/schema change does not authorize profile regeneration. Initial
preparation and explicit full reparse retain their existing lifecycle authority;
enhancement, migration, capacity checks and cache reads reuse the existing profile.
Missing/corrupt profile remains a readiness error. Do not resurrect the older
identity-bearing-profile rebuild rule in historical PDF-MODE-04 notes.

### Enhancement Flow And Failure Boundaries

1. Resolve the actual source through trusted repository/raw-source metadata and
   loader routing; no document_name + ".pdf" assumption. Conflicting/ambiguous
   sources fail explicitly; do not accept arbitrary client paths.
2. Read accepted hierarchy/provenance and normalize the generic or legacy identity.
   Verify namespace/document/revision, source hash, canonical hash/length/coverage,
   completion and existing language. Retain race checks around source file access.
3. Assemble the full-source prompt from that immutable snapshot and use the current
   model capability resolver/token budget, including instructions and output reserve.
   Unready source and capacity excess are distinct failures; either yields zero LLM
   calls. No PDF parsing, OCR, extraction or profile generation in this path.
4. LLM returns the existing split plan; local application creates IDs and validates
   exact body preservation/coverage. Candidate provenance preserves the same generic
   extraction identity. Format metadata does not determine chapter boundaries.
5. Atomic commit rechecks source, structure/labels and interaction revision against
   the same baseline. No-op/title-only retain interactions; actual structure changes
   clear all document interactions under existing confirmed-consent rules. Failure
   preserves accepted structure, interactions and profile.
6. Preserve the existing single user workflow through required reading-layout
   preparation and opening. Projection reads remain read-only; explicit orchestration
   owns writes. Do not add a manual Prepare step or an automatic re-extraction retry.

Identity-normalized consent must not invalidate merely because a legacy record is
represented through the new in-memory wrapper, but changed source/settings/revision
must invalidate it. Keep internal validation errors distinguishable and correlated
in logs (stage, format, schema, reason, duration, reuse decision); never log full
book text, credentials or confirmation tokens. UI receives actionable readiness,
not backend identity fields or verbose model/token diagnostics.

### Ordered Delivery Index

Checklist items live only in their owner modules; this table is not a second
completion tracker. Implement sequentially, checking each item only with evidence.

| Order | ID | Owner | Depends on | Deliverable |
|---|---|---|---|---|
| 1 | EXTRACT-01 | doc_loaders | none | Generic identity/result contracts and invariant tests |
| 2 | EXTRACT-02 | doc_loaders | 01 | PDF adapter, unchanged three-mode behavior |
| 3 | EXTRACT-03 | doc_loaders | 01 | Deterministic TXT adapter |
| 4 | EXTRACT-04 | document_preparation | 02, 03 | Generic preparation persistence and reuse |
| 5 | EXTRACT-05 | document_preparation | 04 | Legacy normalization and explicit safe backfill |
| 6 | EXTRACT-06 | document_preparation | 05 | Generic accepted snapshot and candidate provenance |
| 7 | EXTRACT-07 | db | 06 | Atomic commit/readback identity compatibility |
| 8 | EXTRACT-08 | app | 06, 07 | Preflight/execute, capacity and consent integration |
| 9 | EXTRACT-09 | Deep_Reader_UI | 08 | Format-appropriate readiness/recovery experience |
| 10 | EXTRACT-10 | scripts | 01..09 | Backup-only real TXT/PDF enhancement acceptance |

Implementation status: 10/10 accepted. This is separate from previously accepted Common
parse/content/profile work. Backend aggregate covers EXTRACT-01..08 and 10 only;
frontend EXTRACT-09 belongs exclusively to its UI checklist.

### Acceptance And Governance Record

Real-book closure requires backed-up Madame Bovary.txt and
红色赌盘_261014_145614.pdf under data/test-books, verified against immutable manifests.
Tests must read backups, not live data/raw paths. Preserve old PDF backup schemas;
create a versioned new manifest rather than rewriting an existing backup. Use
isolated storage/namespaces and explicit opt-in for destructive or paid runs.

Run initial extraction -> persisted/reloaded identity -> preflight -> actual LLM
plan -> local validation -> atomic commit -> automatic reading layout for both.
Assert exact canonical-text conservation, valid generated hierarchy IDs, reasonable
plan boundaries, no hidden extraction/OCR/profile calls during enhancement and
unchanged profile hashes. Do not require arbitrary chapter counts just to prove
change; evaluate meaningful subdivision on the backed-up source. A Common parse
or mocked planner pass alone cannot close EXTRACT-10. Record attempts and failure
stage, private report paths, model/settings and actual prompt budget.

Negative coverage includes missing/corrupt/unknown/conflicting provenance, TXT
BOM/newlines/invalid UTF-8, absent page evidence, same-name changed bytes, ambiguous
source routing, capacity rejection with zero LLM calls, stale consent, concurrent
source/structure changes, rollback, no-op/title-only retention and whole-document
structural cleanup. Repeat reads/preflight and migration twice to verify no writes
or profile regeneration. Existing PDF Fast/OCR/Auto tests must remain green.

Inspected memory: root AGENTS.md; proposal.md; high-level-design.md;
docs/modules/index.md; progress.md; design/checklist pairs for doc_loaders,
document_preparation, db, app, scripts and Deep_Reader_UI.
Modified memory: those six design/checklist pairs plus backend progress.md.
Reasons: loaders define shared contracts/adapters; preparation owns the cross-module
design, migration and snapshot; db owns commit checks; app owns API orchestration;
UI owns user-facing recovery; scripts owns real-book closure; progress records only
the nine backend pending items. Global architecture and registry remain unchanged.
Evidence: text_document_loader.py, abstract_document_loader.py,
pdf_extraction_identity.py, accepted_source_snapshot.py,
structure_enhancement_candidate.py, postgres_structured_document_store.py,
private_book_corpus.py and the reported TXT preflight rejection.
Validation scope: documentation links, task ownership/dependencies, unique unchecked
IDs, timestamp lines and backend-only progress synchronization. No runtime/code
changes, book deletion, model calls or implementation-test acceptance in this turn.
Validation result: PASS for documentation scope/architecture/ownership and task
metadata. git diff --check passed; a read-only check found exactly ten unique
unchecked IDs with required timestamps and resolving plan links (nine backend,
one UI). Runtime tests were not run; implementation acceptance remains pending.
Validation itself modified no files. Existing unrelated worktree changes remain
untouched; this governance update modified only the thirteen markdown files above.

## Profile Lifecycle Recovery (2026-10-21)

CONTENT-02 implements the maintainer's reuse-only rule. Initial explicit preparation
may create a profile; explicit full reparse may replace it. Existing profiles are
loaded before considering missing raw text/language, and PDF extraction identity is
not generation authority. Missing profile on an accepted document returns readiness
failure; corrupt files are preserved and reported, never silently cleared/rebuilt.
A request-local generated flag permits enrichment only for initial/rebuilt profiles.
Creation is serialized by a per-document file lock with an inside-lock existence
check. Profile storage uses atomic replacement, preserving old bytes on failure.
Structure-only enhancement continues to reuse source-level profile metadata.

Real-book acceptance also exposed PostgreSQL raw_text omission in PDF cache checks.
Use the existing accepted-source validator (source identity, contiguous canonical
section spans and full-text hash), never task-unit concatenation or a metadata-only
cache hit. Repeated same-source/mode preparation must retain IDs and prepared layout.

## Enhancement Outcome Policy Update (2026-10-20)

Maintainer-confirmed, implementation pending in document_structure ENH-REV-01/04,
db ENH-REV-02 and app ENH-REV-03. Candidate building remains side-effect-free and
source-preserving. Only actual body-structure changes invoke hard replacement and
whole-document interaction cleanup. An unchanged applied plan returns without
commit; navigation-title-only updates preserve IDs/structure version/interactions
through the DB label-update branch. Do not route both through force-rebuild merely
because the LLM returned a valid candidate. No new preparation-owned task is counted.

LLM is responsible for autonomous reasonable semantic divisions; Common's
uncertain-heading conservatism is not a restriction on generated chapter planning.
Source text, order and complete coverage remain immutable. Source/heading-origin
honesty and locator validation do not grant Common a veto over generated titles.

## FULL-03 Read-Only Capacity Handoff (2026-10-20)

inspect_enhancement_capacity accepts the trusted AcceptedSourceSnapshot and uses
the builder's shared enhancement TOC observation plus full-source splitter assembly.
It does not call prepare/load/extract/OCR/profile/LLM or storage writers. Typed safe
capacity and a private configuration/prompt digest return to app consent ownership.
Actual candidate generation still uses the same builder under the app lease and
dispatch identity guard. No candidate persistence or cleanup authority is added.
Evidence/completion counted only in app FULL-03 (139 tests, one intentional skip).

## Source-Preserving Enhancement And Consent (2026-10-19)

Status: all six engineering tasks delivered, including guarded public execution and
isolated browser/API/PostgreSQL acceptance. Production deployment is not claimed. This supersedes any
recommendation to implement LLM enhance by forcing the entire prepare pipeline.
The earlier completed parser/prompt work does not close this new entry-flow defect.

### Evidence And Withdrawn Assumptions

- The last three llm_enhanced reparse requests for the same PDF failed before the
  planner at `page=2:native_coverage_unresolved`, after about 42/38/38 seconds and
  eight OCR candidate attempts per request. This is extraction failure, not prompt
  rejection. Logs dated 2026-10-19 and coordinator failure output support this.
- The former `SectionTaskCoordinator.reparse_document_structure` called
  `prepare_and_load(force_rebuild=True)`. That handoff does not carry PDF extraction
  mode and re-enters the legacy raw-loader/OCR path. This is a known defect, not the
  desired fallback behavior. Never equate forced structure replacement with forced
  source extraction, or assume that omitted mode means the user's previous choice.
  ENH-PREP-03 now routes llm_enhanced exclusively through the confirmed candidate/commit path.
- PostgreSQL `StructuredDocumentStore.load` returns `raw_text=""`; accepted section
  content is persisted, but a complete reusable raw buffer is not guaranteed. Merely
  having an Open-able document, section text, or OCR output does not prove canonical
  source coverage. Do not concatenate task units/blocks or trust a UI copy as source.
- A one-off successful splitter test and fake-provider regression counts are not
  end-to-end reliability acceptance. Do not fix these extraction failures by adding
  prompt constraints, silently switching modes, or weakening OCR coverage checks.

### Accepted Source Snapshot

ENH-PREP-01 implementation: `accepted_source_snapshot.py` provides an immutable
snapshot and `SourceNotReady` reason/recovery contract. Storage owners supply a
consistently read document, namespace, expected ID, structure version, accepted state
and trusted source path; the validator does not construct storage configs (which
can migrate files), invoke loaders or write artifacts. Public adapters remain APP-01.
It streams source bytes in 1 MiB chunks and checks file identity/stat changes during
hashing. Commit-time revalidation remains mandatory; this read is not a lock.

Existing accepted `pdf_extraction` records provide the full canonical-text digest.
When raw_text is omitted, contiguous section slices are joined without separators;
the complete digest proves the endpoint/length, including missing-tail detection.
The snapshot records that verified length. Legacy/non-PDF records without equivalent
accepted identity are explicitly unavailable, not silently reparsed. Optional page
spans are not returned in this first implementation. Accepted older extraction
versions remain reusable. This does not invent durable raw-text or version storage.

Validation: 13 snapshot tests, 9 extraction-identity tests and existing structured
reuse regression passed locally. File JSON roundtrip and PostgreSQL-shaped empty
raw_text fixtures are covered; real PostgreSQL and route integration remain PREP-03.

Define a backend-owned, request-local `AcceptedSourceSnapshot` contract, not a new
hierarchy or history table. It contains namespace/document identity, current
structure version, source-file SHA-256, canonical-text SHA-256 and length, extraction
mode/version/settings identity, accepted language, exact canonical text and optional
validated page spans. Load through storage abstractions, never by assuming file paths.

For a file backend, validate stored raw text and its accepted evidence. For PostgreSQL,
permit reconstruction only from canonical `chapters[].sections[]` source slices whose
spans cover [0, accepted_length) contiguously, without overlap/gaps, and whose exact
concatenation matches the accepted canonical-text hash. Never join with newly invented
separators, normalize canonical text, read task-layout as content, or reconstruct from
task-unit/block summaries. If existing evidence cannot establish coverage, stop; any
future durable canonical-text storage extension needs its own storage design.

Check current source-file identity by bounded streaming hash or authoritative immutable
source revision, without PDF parsing/rendering/OCR. Check accepted text hash, completion
state and namespace. Missing source, missing/empty text, partial OCR, unknown legacy
identity, changed source, corrupt spans/hash and unsupported language are distinct
readiness failures. Return a specific recoverable reason and explicit preparation
guidance; do not load/OCR implicitly. A newer extraction implementation alone does
not invalidate proven accepted text for structure-only enhancement: preserve the
accepted extraction identity; current parser/prompt versions apply to the new plan.

### Structure-Only Execution

ENH-PREP-02 implementation: pipeline `build_enhancement_candidate` delegates to
`structure_enhancement_candidate.py`, using only the existing enhanced builder.
It validates snapshot hash/length/language before paid work, reuses the builder's
bounded Common context and v2 planner/application, then checks exact hierarchy
coverage, identity/language, validated plan provenance, title origins, and absence
of task units/interaction artifacts. Builder failure/fallback is rejected, not
reported as successful enhancement. Accepted extraction identity and source version
are attached to the new candidate; no prior interaction/profile metadata is copied.
No storage config, loaders, page/layout providers, language/profile services or
embeddings are invoked. Legacy root mirrors are removed from the candidate.

The coordinator's internal `enhancement_candidate` context manager validates consent
and retains the document lease while yielding the source state and candidate to the
repository commit owner. The context manager itself never saves. Public reparse now
executes only when DB version guards and the version-aware profile adapter are present.
Otherwise preflight reports execution_available=false and execution returns 503 before
paid work. Source path/title preservation and
structure-dependent profile invalidation at persistence remain the commit owner's
responsibility; the candidate alone is not an authorized write or new active version.

Validation: seven candidate tests with real builder/planner/application and a
deterministic provider; ten live disposable PostgreSQL tests including real
coordinator -> pipeline -> planner candidate success/failure. Both outcomes preserve
persisted hierarchy/interactions exactly and release the lease. Missing consent
prevents provider calls. Thirteen snapshot and nine consent regressions pass.
No paid LLM or semantic-quality/deployment claim. The earlier three extraction
failures are prevented on this path; production rollout and live model quality remain separate.

### ENH-PREP-03 Acceptance And Rollout

The real public route validates consent, checks storage/profile capability, retains
the document lease through generation and atomic commit, and returns committed
structure_version. Rejected candidates return success=false without clearing data;
stale/source/interaction conflicts return 409; ambiguous DB/transport failures are not
reported as rollback. There is no extraction/OCR/profile-generation fallback.

PostgreSQL is supported only with migration 003 active. JSON hierarchy storage remains
explicitly unsupported for enhancement, not silently redirected. VersionedDocumentProfileStore
uses the committed enhancement_source marker to suppress stale file-backed structure_profile
and post_structure_metadata on runtime loads, without changing the profile file or its
summary/language. Marker creation is server-owned inside the hierarchy transaction;
failed commits leave the previous profile view unchanged. Suppression remains conservative
until explicit full preparation replaces the marker. This is advisory invalidation,
not a filesystem transaction or automatic metadata regeneration.

Successful replacement deliberately removes old task units. Task-layout stays read-only
and returns 409 until a separate explicit layout_only preparation. The UI's post-commit
dialog offers Prepare reading layout only for that readiness failure; the action uses
the existing prepare-task-layout endpoint with layout_only=true, force_rebuild=false
and no page evidence. Reload/prepare failures never repeat enhancement.

Validation on 2026-10-19: 30 disposable PostgreSQL cases, including public success,
provider/validation/commit failures, new-interaction conflicts, replay prevention,
profile guards, explicit layout recovery and the opt-in reported source case. Nine
consent, seven candidate and eleven task-layout readiness tests pass; container wiring
and UI build pass. accept_enhancement_browser.py runs current UI against real FastAPI
routes and PostgreSQL at 1280px/390px: cancel, consent, committed version, read-only 409,
explicit layout preparation and ready readback. The planner and unit resolver are
deterministic fixtures; this does not measure live model quality or embedding latency.

The reported four-page source was read/copied without modifying its original DB rows
or bytes. Its legacy record lacks pdf_extraction identity and is correctly unavailable.
The isolated copy was explicitly Fast-prepared once; enhancement then preserved all
8,190 canonical characters with one deterministic planner call and zero extraction,
render/OCR calls. Do not invent provenance for existing legacy documents: they need
an explicit Prepare before enhancement, even if old text is visible.

Rollout: docker-compose includes migration 003 for fresh volumes; start_local_dev.sh
waits for PostgreSQL and applies that idempotent migration transactionally before API
startup, including existing volumes. No production migration/restart occurred during
this task. Rebuild/restart through the existing scripts to deploy. Missing migration
or profile guards keeps execution unavailable. Source intake must retain immutable
accepted bytes; arbitrary external filesystem edits are not transactionally controlled.

1. Validate destructive-operation consent and the accepted source snapshot before
   any paid call; acquire document-scoped single-flight protection.
2. Build bounded Common context in memory from that exact text and selected toc_hint;
   run the existing LLM planner/application against the same immutable snapshot.
3. Validate exact source coverage, local boundaries, title origins and hierarchy.
4. Compare applied plan with authoritative baseline (including labels). Revalidate
   source identity, hierarchy/label revision and interaction revision. No-op does
   not commit; title-only updates labels while retaining interactions; structural
   changes atomically replace hierarchy and clear all document interactions.
   Stale state returns conflict, never overwrite or automatic retry with new consent.
5. Read back authoritative outcome/IDs/version. For a structural commit, UI
   automatically calls explicit layout-only preparation BEFORE reading layout.
   Title-only refreshes labels without a destructive reset; no-op retains the reader.
   Recovery failure never means the commit rolled back or warrants another LLM call.

This path must not call raw loaders, native PDF inspection, renderers, OCR, language
detection, profile generation or embeddings. Reuse accepted language/profile where
valid; preserve the original file, canonical text and extraction provenance. Structure-
dependent profile diagnostics must be invalidated/recomputed by their existing explicit
owner, never silently retained as current or used as parser authority. Missing language
is a readiness failure, not a hidden language-detection call. Re-extraction is a separate
explicit Prepare action using fast/OCR/auto (fast default for a new selection), never
an automatic fallback of enhance; it must honor its own replacement consent.

### Consent And Destruction Boundary

Use the planned app preflight/token contract below, with backend counts and scope.
Interactive metadata means persisted interaction outputs and session state, not just
cached display labels: analysis/insight, quiz/questions/answers/evaluation, critical-
thinking sessions, and any actually persisted summaries/notes attached to replaced
targets. The implementation must inventory relational artifacts AND embedded
document/chapter/section/task-unit artifact payloads; it must not claim note support
or exact counts where unimplemented. Include all document-owned targets currently
invalidated by hard reparse, including book-level artifacts. Preserve raw file and
source identity. User-created notes, if present in affected storage, must be explicitly
identified in the warning; do not silently exempt or migrate them.

Confirmation authorizes cleanup ONLY when validated body-structure replacement
commits. Declining means cancel; no pre-clearing on click, preflight or LLM start.
Failure/stale consent and no-op preserve hierarchy/version/interactions. Title-only
updates retain target identities and interactions. Structural replacement increments
version once and invalidates all old targets, including book-level data; frontend
drafts clear only for that confirmed outcome. Deletion failure rolls back structural
replacement. Preserving interactions on a title-only result is required, not an
option to retain stale data after structural change. No partial reparse is supported.

### Execution Checklist Index

Separate engineering track: 6/6 complete (100%); backend 5/5, UI 1/1. Items have one owning checklist.

| Order | ID | Owner | Scope / dependencies |
|---|---|---|---|
| 1 | ENH-PREP-01 | document_preparation | Validate accepted source snapshot; include DB readback fixtures |
| 2 | ENH-APP-01 | app | Read-only readiness/impact preflight and consent request contract; depends on 01 |
| 3 | ENH-PREP-02 | document_preparation | Isolated structure-only candidate path; depends on 01, APP-01; no unsafe public cutover |
| 4 | ENH-DB-01 | db | Atomic snapshot/revision checks and cleanup, single-flight integration; depends on APP-01/PREP-02 |
| 5 | ENH-UI-01 | Deep_Reader_UI | Confirmation, cancel, pending/stale/failure/committed states; depends on backend contract and DB guard |
| 6 | ENH-PREP-03 | document_preparation | Full API/UI/storage acceptance and deployment gates; depends on all above |

API/schema/main adapters are supporting work inside APP-01, not extra counted tasks.
DB and file-backed supported deployments must demonstrate atomic preservation/commit
semantics; an unsupported backend returns an explicit unavailable result, never an
unsafe fallback. No stage claims completion from mock tests alone.

## Enhancement Semantics Correction (2026-10-19)

LLM enhance is one optional Common-result optimization. Existing intent values are
legacy compatible inputs to the same planner policy, not separate restoration and
outline workflows. Source reload through the current forced prepare entry is a known
defect, superseded in the planned flow above; source reuse and structure replacement
must be separate decisions. The planner uses bounded Common context and validated source
boundaries; failure preserves current hierarchy/artifacts. Historical staged/two-purpose
NATIVE-04 notes below are superseded by document_structure's current correction.

## NATIVE-05 Validated Enhancement (2026-10-19)

The staged generated-outline guard is removed. Request-local assets carry intent
to the builder; both enhanced intents use v2 validation/application. Enhanced
requests defer profile rebuilding/enrichment until structure validation/save and
return before profile/FAISS/bundle writes on rejection. Existing atomic save and
readback remain the commit boundary. A failed attempt reports failure while the
previous persisted hierarchy remains available; no Common fallback is written.
Common preparation keeps its prior profile-first ordering. Policy-aware reuse now
checks Common policy v2, hint, parser mode and intent. The historical enhanced path
bypasses full preparation reuse; this must not remain the source-loading policy for
structure-only enhancement. Reads never rebuild. Twelve application tests plus
existing Common cache/profile-order regressions provide isolated evidence; live
PostgreSQL acceptance is not claimed. Earlier NATIVE-04 staging notes are historical.

## NATIVE-04 Intent Boundary (2026-10-19)

`prepare` and `prepare_and_load` accept enhancement_intent, default restore_source.
Generated intent requires llm_enhanced and is staged: it raises
reading_outline_application_not_available before loading raw source or saving.
The v2 planner is independently usable but its safe application is NATIVE-05;
do not silently pass it through the legacy merger. Existing omitted/default intent
keeps compatibility. Evidence: structure-plan and API contract tests; no live LLM.

## NATIVE-03 TOC Declaration (2026-10-19)

Explicit callers pass `toc_hint=auto|present|absent`, independently of PDF mode.
Validation precedes source loading; request-local `PreparedDocumentAssets` carries
the hint to the builder. Reuse checks persisted `common_parse_policy` version/hint;
missing or changed identity rebuilds only during explicit preparation. Internal
omission (`None`) preserves legacy reuse. `prepare_and_load` forwards an explicit
hint for reparse. No new read mutation or profile parser authority is introduced.
Evidence: `test_common_parse_policy.py`, PDF-mode preparation and contract suites.
LLM intent/prompt changes remain NATIVE-04/05.

## 1. Module Purpose

### Task-Layout Boundary-Only Read (2026-10-16)

App LAYOUT-03 adds `load_task_layout_page_boundaries(doc_name, document=...)`:
reuse persisted PDF page spans for the caller's exact projection snapshot only.
Validate extraction identity schema/mode and saved text hash, hash the current
source bytes, validate numeric ranges/page order and reject unknown or stale evidence.
No store reload/save, raw extraction, OCR/render, language or profile generation.
Missing/invalid evidence returns an empty list for character-anchor fallback;
legacy evidence is not reconstructed or migrated by reading. Log safe outcome/reason
and page count. Manual commit's `load_manual_structure_source_evidence` remains
unchanged and retains its freshness gates. Existing evidence provenance/schema is
reused, not a new hierarchy or persistence contract. Evidence: page-evidence,
readiness, PDF-mode and manual handoff/commit regression scripts. Completion is
owned by app LAYOUT-03, not an additional preparation checklist item.

### Three-Mode Preparation (2026-10-15)

Current status: PDF-MODE-05 below supersedes the historical staged 501 gate and
pending-integration statements in PDF-MODE-01/04 delivery notes.

PDF-MODE-01 supporting boundary is implemented: `prepare` accepts a request-local
enum and validates it before scope creation/cache reuse. All PDF mode requests
currently raise typed mode_unavailable; TXT fast remains normal preparation with
not_applicable provenance, and TXT ocr/auto raises mode_not_applicable. Internal
callers omitting this argument (`None`) retain legacy execution and unknown
provenance; public routes always pass their validated default/selection. Added
PreparedDocumentAssets metadata round-trips without relabeling legacy snapshots.
This is a fail-closed contract gate, not mode-aware extraction or cache integration.
Evidence: `scripts/test_pdf_extraction_mode_contract.py`; completion owned by main.

PDF-MODE-04 identity/reuse hooks are implemented (see below); PDF-MODE-05 still
owns end-to-end propagation of `pdf_extraction_mode`
(fast default, ocr, auto) independently of preparation/parser mode. Loader design
section 18 owns extraction semantics. Raw loading, page boundaries, manual source
evidence and optional lazy layout must use consistent provenance; fast forbids
render/OCR even after an Outline rejection. Evidence reads use the prepared source
provenance without silently re-extracting it under a different default mode.

Reuse must check source fingerprint, effective mode and extraction version/settings.
Do not let the existing structured-first shortcut silently bypass a changed mode.
Legacy artifacts with unknown extraction provenance stay readable, but cannot be
claimed as a cache hit for a new explicit mode preparation. Rebuilding them is an
explicit operation, not read-time migration. Reuse/invalidation of profile, task
layout, retrieval and artifacts must follow source/structure versions and existing
hierarchy replacement rules; a mode setting is not a new hierarchy authority.
Preserve the prior valid document on extraction failure; never pre-delete it.

Request-local mode prevents concurrent jobs from sharing mutable loader settings.
Mode-aware logs distinguish native extraction, OCR, language/profile and layout
costs. Fast removes OCR/classification overhead, not language/profile LLM latency.
Automatic post-upload prepare remains a separate visible UI-triggered request;
upload/list/Open reads stay outside the mutation boundary. Client cancellation
does not imply server cancellation. Resource/quality failures cannot publish
partial readiness; optional layout diagnostics remain advisory-only.

### PDF-MODE-04 Identity And Reuse Hooks (2026-10-15)

`PdfExtractionIdentity` records source SHA-256, mode, extraction version and a
canonical JSON settings digest under schema version 1. `identity_for_loader`
includes pypdf version; OCR/auto also include engine/renderer binary hashes,
language, DPI, quality version, timeout, page limit and execution budgets. Auto
includes policy config and its legacy enable flag. Fast ignores OCR settings.
Source hashing is supplied by the request snapshot; identity construction performs
no OCR/engine execution. Missing binaries are represented as unavailable, not
proof of a successful extraction. Language pack contents are not independently
hashed; replacing traineddata in place requires explicit rebuild/version handling.

PreparedDocumentAssets carries this identity without changing public DTOs. The
structured reuse hook requires exact identity and canonical text SHA-256 under
`parse_provenance.pdf_extraction`; missing/legacy/malformed metadata, source,
mode, settings/version changes or changed text force a miss without deleting data.
Successful hits report effective mode and reused provenance. Legacy internal
requests without explicit identity retain their existing behavior and do not
backfill extraction metadata. Mode is not document/chapter identity.

The existing validated structured save boundary adds the record only for applied
extraction with a matching effective mode and unchanged canonical raw text. No
sidecar or DB schema is introduced: JSON and PostgreSQL already preserve nested
parse_provenance. Explicit identity-bearing profile/FAISS/bundle preparations
conservatively rebuild instead of trusting resources with no identity marker;
missing raw text exits before those stores are touched. This does not claim
transactional rollback across all derived-resource writes or add a new raw cache.

These are integration hooks: PDF-MODE-05 must construct the identity from the actual
snapshot, populate applied provenance only after successful extraction, and pass
bound evidence providers. Public PDF requests still return 501. Tests call the
hooks directly; they do not demonstrate an operational public three-mode pipeline.

Evidence: `test_pdf_extraction_identity.py` (9 tests) covers mode/source/settings
changes, unknown/corrupt provenance, same-mode hit, canonical settings ordering,
JSON/DTO roundtrip, save proof, and failure preservation. Repository tests use a
PostgreSQL-shaped store substitute, not a live DB; existing DB parse_provenance
serialization was inspected and unchanged. Existing structured reuse, lazy layout
(15), and API mode contract (11) regressions pass. No deployment/user-data writes.

### PDF-MODE-05 Integrated Preparation (2026-10-15)

Both explicit API routes now accept default fast or explicit ocr/auto execution.
Prepare captures source bytes once, computes the mode/settings identity, and
checks structured reuse before extraction. On a miss, the loader receives the
same bytes and selected mode. Result mode, extraction version, source hash and
page spans are validated before language/profile/build work. No mutable global
mode is introduced; request-local values and existing ContextVar execution scopes
isolate concurrent calls. TXT and internal legacy calls retain their behavior.

Fast binds native-only providers; OCR binds recognized text inventory; auto wraps
the existing selective executor using a temporary source snapshot, including its
optional layout call. Auto retains the configured OCR enable flag and policy
limitations. An unresolved page with OCR disabled fails explicitly. Optional
Outline/layout failure falls back to parsing without entering another mode.
Required extraction failure raises a typed API failure before profile/hierarchy
writes. A source change detected before structured save rejects the write.

Successful saves record extraction identity and compact page spans alongside
existing parse provenance, plus result-scoped OCR provenance/pages where supported
by the repository. Manual source evidence for such documents validates the current
source hash and saved text/spans, without OCR or language LLM calls. Unknown legacy
documents retain the old read path; stale or malformed proven evidence requires
explicit preparation, not a hidden migration. No schema migration is added.

Mode/source logging complements existing stage timings and execution counters.
Client disconnection is not server cancellation: no job/cancel endpoint or
same-document write-order guarantee is introduced. UI navigation-generation guards
remain responsible for late responses. Real DB/engine acceptance and cross-store
transactionality are not claimed; PDF-FIX-03 remains open.

Evidence: `test_pdf_mode_preparation.py` (10 cases) covers default/API fast,
prepare-task-layout response, reuse, OCR-through-save with simulated engines,
auto native input, optional fallback, manual saved evidence/source mismatch,
failure preservation and concurrent fast/auto requests. Existing mode contract
(11), identity (9), fast (9), full OCR (7), lazy layout (15), raw recovery (25),
execution pass (25), raw-load errors and structured reuse scripts pass. No live
API restart/deployment or user-document preparation was performed.

Preparation timing uses the shared `logging_config.log_stage` helper for raw loading,
language detection, profile cache/load/generation/save, structured cache/evidence/build/
save/readback, OCR persistence, profile enrichment, FAISS preparation and bundle setup.
`prepare_started` records doc_name, mode, parser mode, force_rebuild and store class;
`prepare_result` records readiness, raw character count, structured reuse and error codes
without text/error details. Stage finish is execution timing, including existing
non-blocking failure/skip paths; result flags remain the success authority. Timings are
diagnostic only and preserve cache, stage ordering and persistence policy.
**[Code-Confirmed]**

`document_preparation/` 負責準備流程編排（prepare orchestration）：把 raw document 轉成可用於 QA/task-layout 的 artifacts readiness snapshot。 **[Code-Confirmed]**

## 2. Position in Overall Architecture

- Document Preparation Layer

## 3. Key Files

| File | Responsibility | Notes |
|---|---|---|
| `document_preparation/document_preparation_pipeline.py` | 主準備流程（raw/language/profile/structured/faiss/bundle） | 包含 Step 4.5 post-structure enrichment **[Code-Confirmed]** |
| `document_preparation/prepared_document_assets.py` | readiness DTO（ready flags/path/errors） | API prepare response 主要來源之一 **[Code-Confirmed]** |
| `document_preparation/prepared_document_result.py` | prepare + loaded artifacts 組合 DTO | 給 coordinator consume **[Code-Confirmed]** |
| `document_preparation/preparation_mode.py` | `base` / `free_qa` mode contract | mode resolve + validation **[Code-Confirmed]** |

## 4. Main Responsibilities

1. pipeline step ordering（載入、語言、profile、structured、faiss、bundle）。 **[Code-Confirmed]**
2. base/free_qa mode 差異化準備。 **[Code-Confirmed]**
3. profile build 與 profile store reuse/force rebuild 控制。 **[Code-Confirmed]**
4. structured build + atomic save。 **[Code-Confirmed]**
5. post-structure metadata enrichment 並儲存更新 profile。 **[Code-Confirmed]**
6. 失敗不阻塞的錯誤收集（尤其 profile/enrichment）。 **[Code-Confirmed]**

## 5. Non-Responsibilities

1. 不負責 API request/response mapping。 **[From HLD]**
2. 不負責 task-layout 組裝與 diagnostics projection。 **[From HLD]**
3. 不應把 parser metadata 當 parser hard rule。 **[From Proposal] + [From HLD]**

## 6. Important Data Structures / Contracts

- `PreparedDocumentAssets`
- `PreparedDocumentResult`
- `PreparationMode`
- `SectionSplitterMode`（作為 structured parser mode 輸入）

## 7. Preparation Lifecycle

```text
Step 1  Load canonical raw text
Step 2  Detect document language
Step 3  Prepare document profile
Step 4  Prepare structured document
Step 4.5 Enrich profile with post-structure metadata
Step 5  Prepare FAISS artifacts
Step 6  Prepare runtime bundle
```

說明：`BASE` mode 也會走 profile（Step 3）與 enrichment（Step 4.5）。 **[Code-Confirmed]**

## 8. Step I/O Artifact Matrix

| Step | Inputs | Outputs | Persistence |
|---|---|---|---|
| 1 Raw load | `doc_name`, loader | `raw_text` | read `data/raw` |
| 2 Language detect | `raw_text`, optional existing artifacts | `document_language` | read profile/records (if exists) |
| 3 Profile prepare | `raw_text`, `document_language` | `DocumentProfile`, `profile_ready` | write `profile.json` |
| 4 Structured prepare | `raw_text`, parser mode | `StructuredDocument`, `structured_document_ready` | write `*.structured.json` |
| 4.5 Post enrich | profile + structured doc | enriched profile snapshot | write `profile.json` |
| 5 FAISS prepare | `raw_text` (+ language via parsed document) | index/records/meta | write `index.faiss`/`records.json`/`meta.json` |
| 6 Runtime bundle | index + profile | `FaissIndexBundle` | runtime cache |

## 9. Non-Blocking Policy Matrix

| Failure Point | Blocks Prepare? | `profile_ready` | `structured_document_ready` | Error Collection |
|---|---:|---:|---:|---|
| profile load/build fail | No | `false` | unaffected | add error reason |
| post-structure enrichment fail | No | keep prior profile result | unaffected | add error reason |
| structured build/save fail | Yes for structured-ready | unaffected | `false` | add error reason |
| faiss/bundle fail (free_qa path) | Yes for index/bundle readiness | unaffected | unaffected | add error reason |

**[Code-Confirmed]**

## 10. Module Relationships

- depends on:
  - `doc_loaders/`, `language/`, `document_structure/`, `retrieval/`, `profile/`, `bundle_provider.py`
- used by:
  - `app/qa_coordinator.py`
  - `app/section_task_coordinator.py`（via coordinator chain）
  - `main.py` `/documents/prepare`
- reads from/writes to:
  - structured storage
  - faiss storage
  - profile storage

## 11. Main Flows Involving This Module

1. prepare flow（核心）。 **[Code-Confirmed]**
2. structured hierarchy build flow（透過 structured builder）。 **[Code-Confirmed]**
3. profile metadata flow（pre-structure builder + post-structure enricher）。 **[Code-Confirmed]**
4. bundle/index readiness flow（free_qa mode）。 **[Code-Confirmed]**

## 12. Persistence / Side Effects

- read persistence：是（structured/profile/faiss）
- write persistence：是（structured/profile/faiss）
- mutate structured document：間接（透過 structured builder/repository 寫出）
- generate runtime projection：否
- call LLM：間接（language detector/profile builder/llm parser path）
- diagnostics only：否

## 13. Known Legacy / Compatibility Behavior

1. profile load path 支援 cache hit/reload failure rebuild。 **[Code-Confirmed]**
2. structured load/save 仍允許舊 JSON 讀取（由 model/store 層承接）。 **[Code-Confirmed]**
3. 本模組不直接管理 root sections/structure_nodes compatibility 細節（委派到 structure layer）。 **[Code-Confirmed]**

## 14. Current Risks

1. risk：prepare step 增長造成錯誤來源難追蹤
- why：多 artifact pipeline 容易有 partial success
- guardrail：維持 assets.errors 分層前綴與 step log

2. risk：profile/cache policy 與 parser 策略邊界混淆
- why：可能把 advisory metadata 變成 parse authority
- guardrail：文件化 cache-only contract + 測試約束

3. risk：base/free_qa 分支差異被新功能破壞
- why：可能出現 base mode 不完整或重複構建
- guardrail：mode-specific regression tests

4. risk：post-structure enrichment 失敗處理語義漂移
- why：若變成阻塞會改變 API 行為
- guardrail：固定 non-blocking 行為測試

5. risk：cache hit 被誤解為完整 prepare reuse
- why：`task_layout_cache_hit` 或 OCR/page text cache hit 只證明該層資料可重用，不等同於 language/profile/structured preparation 已全部短路。若 route 仍進入 prepare flow，可能再次觸發 raw/OCR 讀取、language detection、profile build，甚至間接 LLM call。 **[Code-Observed] + [Inferred]**
- guardrail：將 cache hit log 分層命名；明確區分 raw/OCR cache、structured artifact reuse、profile/language cache、task-layout projection cache。read-like caller 不應把任何單一 cache hit 當作整條 prepare pipeline 完成。 **[Future Direction]**

6. risk：base preparation 在既有文檔讀取場景未完全避免昂貴 path
- why：UI 選擇既有文檔時，若只是要讀取當前 task-layout，進入 `base` prepare 可能因 structured/profile/language reuse 條件不成立而重新載入 PDF/OCR，並間接觸發 LLM-backed language/profile 分析。 **[Code-Observed] + [Inferred]**
- guardrail：`force_rebuild=false`、common parser、既有 structured artifact 可用時，應優先 short-circuit 到既有 hierarchy；若 task-layout cache 可用但 structured reuse 失敗，應記錄為 cache-boundary inconsistency 供排查，而不是安靜走昂貴 prepare。 **[Future Direction]**

7. risk：language detection cache 無法被有效使用
- why：language detection 若未接收可用的 storage/config context，可能無法從 profile 或 retrieval records reuse 既有語言結果，只能 fallback 到 LLM detector。 **[Code-Observed] + [Inferred]**
- guardrail：language detection 應有明確 cache source priority（profile/records/source metadata -> deterministic fallback -> LLM fallback），並以測試保證既有文檔的 read-like path 不重複呼叫 LLM。 **[Future Direction]**

## 15. Open Questions for Maintainer

1. cache-first 命名標準化與實作命名遷移先標註、後落地的時程是否固定？ **[From HLD] + [Needs Confirmation]**
2. post-structure enrichment 是否需要獨立 refresh 機制（非 prepare 路徑）？
3. prepare logs 是否需要標準化為可機器解析的 event code？

## 16. Suggested Next Documentation Improvements

1. 增加 preparation pipeline lifecycle diagram（base vs free_qa）。
2. 增加錯誤分類矩陣（blocking/non-blocking）。
3. 增加 profile cache policy appendix（hash/version/rebuild triggers）。

## 17. Future Storage Independence Note

> 本節是 storage abstraction boundary preparation，非目前 implementation。 **[Maintainer-Provided] + [Future Direction]**

1. The preparation pipeline should prepare artifacts and readiness snapshots. **[Code-Confirmed] + [Future Direction]**
2. The preparation pipeline should not permanently assume file-path persistence as the only possible destination. **[Maintainer-Provided] + [Future Direction]**
3. Future storage destination should be configurable through a storage/backend policy boundary, while preparation lifecycle semantics remain owned by `document_preparation/`. **[Maintainer-Provided] + [Future Direction]**
4. This note does not introduce storage abstractions, repositories, DB schemas, dual-write behavior, read-path switches, API changes, parser changes, or runtime behavior changes. **[Doc-Confirmed]**

## 18. Future Direction Note: Manual Structure Reparse Preparation Boundary

> 本節記錄 source-agnostic manual structure override 的 preparation-layer boundary；不代表目前 implementation。 **[Maintainer-Provided] + [Future Direction]**

1. manual structure reparse may apply to any raw document source, not only OCR/scanned PDFs. **[Maintainer-Provided]**
2. `document_preparation/` should own preparation-time handoff of raw text, language, source identity, page boundaries when available, and existing structured/profile artifacts needed by validation/reparse orchestration. **[Future Direction]**
3. The preparation layer should not decide manual hierarchy semantics; projection and validation semantics belong to `document_structure/`. **[Future Direction]**
4. For source-agnostic support, manual anchors should be able to target raw-text character spans even when page evidence is unavailable. **[Future Direction]**
5. For PDF/OCR/native-page sources, preparation may provide page boundaries and OCR provenance as validation evidence, but page evidence remains advisory/supporting data rather than parser authority by itself. **[From HLD] + [Future Direction]**
6. manual validation failure must be reported as structured readiness failure for that explicit reparse attempt; it must not overwrite existing structured artifacts and must not silently fallback to common parser as a successful manual reparse. **[Future Direction]**
7. Existing `force_rebuild` semantics should remain explicit: manual structure commit is a mutation/reparse operation, while validation/preview is read/analysis-oriented and should not persist a hierarchy. **[Future Direction]**
8. This note does not introduce new API fields, parser modes, persistence behavior, or runtime route behavior. **[Doc-Confirmed]**

## 18.1 Future Direction Note: Raw Source Upload Intake Boundary

> 本節規劃 file upload 在 preparation layer 的 raw-source intake 邊界；不代表目前已實作。 **[Future Direction]**

1. Uploaded files should enter the system as raw source candidates before any preparation
   pipeline work. **[Future Direction]**
2. Preparation/upload intake should validate filename normalization, supported extension,
   source type, source size, source hash, and storage outcome before returning a document
   candidate. **[Future Direction]**
3. Upload intake must not run language detection, profile building, structured parsing,
   OCR, FAISS/retrieval build, task-layout projection, or reading interaction generation
   as a hidden side effect. **[From HLD] + [Future Direction]**
4. The existing preparation pipeline remains the explicit path from raw source to
   profile/structured/retrieval readiness. Upload success should be represented as source
   availability, not prepared document readiness. **[Future Direction]**
5. Source hash should be captured at upload/intake time so later prepare/reparse and
   stale-source checks can compare against stable evidence. **[Future Direction]**
6. Conflict policy should be explicit and app-coordinated; preparation should not silently
   overwrite an existing raw source with the same display name. **[Future Direction]**
7. User-uploaded raw files remain user-scoped and must preserve the raw-file
   ownership/copyright boundary described in the proposal. **[From Proposal] + [Future Direction]**

## 19. Page Evidence Handoff For TOC Editing

> 本節支援 UI TOC editor 的 page-first anchor UX。Manual-structure source evidence can now carry validated page boundaries; downstream `page_range` hierarchy materialization and task-layout prefill projection remain separate checkpoints. **[Maintainer-Provided] + [Code-Confirmed] + [Future Direction]**

1. `load_manual_structure_source_evidence(...)` provides a reusable source-evidence handoff containing canonical raw text, raw-text source hash, best-effort language, and validated page boundaries when available. **[Code-Confirmed]**
2. For pageable sources such as PDFs, page evidence includes stable document page indices, optional display labels/page labels, and raw-text offset ranges per page. **[Code-Confirmed]**
3. Preparation validates manual page evidence before handoff: duplicate or non-contiguous page indices, invalid ranges, out-of-bounds ranges, non-monotonic ranges, and page-text mismatches are rejected. **[Code-Confirmed]**
4. Invalid or unavailable page evidence is reported in source-evidence `errors` and dropped while preserving canonical raw text and `char_range` fallback. **[Code-Confirmed]**
5. This evidence is needed by:
   - manual-structure validation/commit for `page_range` anchors
   - task-layout anchor evidence projection for UI edit-existing prefill
6. Preparation must not decide user-defined hierarchy semantics; it only supplies evidence. Projection and hierarchy draft building remain owned by `document_structure/`. **[From HLD] + [Code-Confirmed]**
7. Page evidence remains supporting evidence, not parser authority by itself. Ambiguous, incomplete, or stale page evidence must be reported explicitly and must not authorize partial hierarchy persistence. **[From HLD] + [Future Direction]**
8. `char_range` remains available when page evidence is unavailable or rejected. **[Maintainer-Provided] + [Code-Confirmed]**

## 20. Future Direction Note: Prepare Reuse And LLM Cost Boundary

> 本節記錄本輪運維觀察後的 cache/prepare governance；不代表目前 implementation 已完成。 **[Code-Observed] + [Future Direction]**

1. A task-layout cache hit is a projection-layer reuse signal, not proof that document preparation has been fully skipped. **[Code-Observed]**
2. OCR/page-text cache hits are raw-source reuse signals. They do not by themselves guarantee language/profile/structured reuse. **[Code-Observed] + [Inferred]**
3. For read-like flows that only need the current active hierarchy or task-layout, preparation should avoid re-entering expensive OCR/LLM-backed stages when a valid structured artifact and compatible task-layout projection already exist. **[Future Direction]**
4. Language detection should receive enough storage/config context to reuse existing language evidence from profile or retrieval records before calling LLM-backed detection. **[Future Direction]**
5. Profile building/classification may use LLM as fallback, but repeated UI document selection should not implicitly rebuild profile metadata when source hash, parser mode, and structured artifact are unchanged. **[Future Direction]**
6. If `task_layout_cache_hit` is observed before a later prepare-stage OCR or LLM call for the same document request, logs should expose whether the cause was forced rebuild, structured artifact miss, source hash mismatch, parser/schema version mismatch, profile cache miss, language cache miss, or storage-backend mismatch. **[Future Direction]**
7. This note does not change parser authority: metadata and LLM classification remain advisory, and hierarchy truth remains `chapters[].sections[].task_units[]`. **[From HLD]**

## 21. Selective OCR Scheduling (2026-10-14)

The selective OCR track is defined in `../doc_loaders/module-detailed-design.md`
section 16; this module owns OCR-06. Current preparation loads layout evidence
only after structure-owned Outline projection rejects or is unavailable. Earlier
Outline precedence described source selection but did not avoid eager layout
execution; OCR-06 now implements that scheduling boundary. **[Code-Confirmed]**

`prepare(base)` retains structured-cache-first reuse. On a miss or forced rebuild,
`_load_raw_source` obtains raw text plus page boundaries through the optional
loader `load_with_page_boundary_evidence` entrypoint. PDF text and spans come from
one canonical extraction/recovery pass and stay local to this preparation call;
non-PDF loaders retain `load`, with the prior optional boundary fallback. No
pipeline-wide mutable evidence cache or readiness DTO field is added.
Language/profile ordering is unchanged. **[Code-Confirmed]**

The structure stage passes boundaries, native Outline and a lazy
`page_evidence_provider` to one builder invocation. Only `document_structure`
validates destinations, canonical ranges and source identity. Accepted Outline
skips the provider and splitter; absent/rejected Outline retains parser selection
and requests optional layout once with the existing first-30-page limit. Direct
builder callers can still provide eager evidence. Selection remains
`native_outline -> validated_ocr_toc -> current_keyword_heading_parser` with the
existing explicit enhanced-parser option, not duplicated preflight builds or LLM
calls. **[Code-Confirmed]**

Text recovery is independent: accepted Outline does not excuse missing scanned
body text. Loader-required text failures still block raw readiness; optional
layout failure preserves parser fallback and explicit diagnostics. Loader failure
logs include error type only; existing parse provenance gains `layout_evidence`
status (`skipped_validated_outline`, `provided`, `loaded`, `unavailable`, `failed`,
or `rejected_source_mismatch`). Known source mismatches reject evidence, not the
already loaded canonical text; the current parser can still consume that text.
Fresh preparation calls reload source evidence. No new upload side effect,
task-layout mutation, profile authority, API schema or durable cache is added.
Broader per-pass layout inspection reuse/budgets remain OCR-07. **[Code-Confirmed]**

Evidence: `scripts/test_prepare_lazy_pdf_layout.py` passes 15 tests locally and in
isolated Compose on 2026-10-14. Coverage includes one extraction per page/one build,
Outline validation (not presence), internal paragraph newlines, required recovery
before Outline, optional failure, source replacement, explicit enhanced parser,
force rebuild and structured reuse. Existing raw-error, profile ordering, PDF,
manual source-evidence/handoff and logging regressions pass. Tests use temporary
PDFs, in-memory stores and mocked recovery, not real LLM or scan-quality claims.
No live API deployment or end-to-end speedup is claimed. **[Test-Confirmed]**

### OCR-07 Scope Lifecycle Integration

`prepare` and `load_manual_structure_source_evidence` now open the factory's
optional `preparation_scope` around their existing synchronous workflow. PDF
loaders share bounded inspection/decision/OCR evidence and execution budgets
through raw load, boundaries, lazy layout and OCR persistence; non-PDF/custom
factories without the hook retain `nullcontext` behavior. Cache-first structured
reuse still does not call `factory.get`, load raw text or invoke profile/LLM work.
Manual evidence reads remain read-only. The scope always releases its cache;
loader handoff metadata is context-local rather than shared across worker threads.
No hierarchy/profile/task-layout persistence policy changes. **[Code-Confirmed]**

Budget values and failure semantics belong to loader design section 16.11. Text
budget exhaustion follows the existing raw-load failure mapping and prevents
partial structured writes; optional layout exhaustion retains inventory and
parser fallback. Time budgets cover external PDF work, not profile/LLM latency
or the entire prepare request. Evidence: `test_pdf_execution_pass.py` covers the
real factory handoff, manual sharing, required failure with zero saves and
optional exhaustion with successful structure creation. Existing preparation,
manual and logging regressions pass. This supports loader-owned OCR-07 only.
**[Test-Confirmed]**
