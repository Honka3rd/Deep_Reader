# scripts Detailed Design

## Reading Review Acceptance (2026-10-21)

OCR-REVIEW-03..06: test_reading_content.py and test_reading_content_postgres.py
exercise the same mutation contracts in temporary file storage and disposable SQL
schemas. Cover missing IDs, no insert, Unicode bounds, concurrent writes, receipts,
late saves/generated writes, notes, structural reparse, delete/undo and atomic rollback.
accept_reading_review.py targets the private Sample2 backup only, verifies its SHA,
runs actual OCR normally and with margin recovery fault-injected off, then tests
real source crops, source-identity refusal and four generator/context poison guards.
The partial case is fault injection on a real book, not an unmodified-book failure.
Its optional --serve port exposes only review endpoints from real FastAPI handlers
against temporary storage; never point it at production data.
Deep_Reader_UI/scripts/test-reading-content-live.mjs mounts the production editor
against that disposable API at 1280/390px. Checks failed Save retry, draft protection,
concurrent conflict, delete confirmation, real crop and ignore/undo; screenshots
are local /tmp/reading-{delete,review}-{width}.png. This is component/API acceptance,
not whole-app Library workflow automation. Original Sample and mixed-image OCR
regression use accept_chinese_direction.py and unchanged reviewed references.

### Governance And Validation Receipt (2026-10-22)

Inspected memory: AGENTS.md, proposal.md, high-level-design.md, docs/modules/index.md,
progress.md; design/checklist pairs for doc_loaders, document_structure,
document_preparation, db, section_tasks, app, scripts, shared, main.py, api_schemas.py,
and Deep_Reader_UI. Existing unrelated working-tree changes were retained.

Markdown modified in this delivery and reason:

| File (backend paths relative to Deep_Reflective_Reader) | Reason |
| --- | --- |
| doc_loaders/module-detailed-design.md | Replace future-only review policy with implemented scope, limits and real-source evidence. |
| doc_loaders/module-checklist.md | Close OCR-REVIEW-03..06 against verified evidence; timestamps 2026-10-22. |
| document_structure/module-detailed-design.md | Record immutable overlays, no insertion, revision/undo/receipt limits. |
| document_preparation/module-detailed-design.md | Record generic evidence handoff without format branches/profile regeneration. |
| db/module-detailed-design.md | Record transactional invalidation and rollback evidence. |
| section_tasks/module-detailed-design.md | Record document-wide guards, source revision and note preservation. |
| app/module-detailed-design.md | Record stable-ID read projection and empty-unit behavior. |
| main.module-detailed-design.md | Record implemented routes, error boundaries and read-only preview. |
| api_schemas.module-detailed-design.md | Record strict mutation request and additive flags; no response-DTO overclaim. |
| scripts/module-detailed-design.md | Record reproducible acceptance, exclusions and this governance receipt. |
| ../Deep_Reader_UI/module-detailed-design.md | Record existing-menu actions, no add, source review and verified UI behavior. |
| ../Deep_Reader_UI/module-checklist.md | Replace stale future-only reference, without duplicate task counts. |
| progress.md | Synchronize backend queue 2/6 to 6/6 and loader count 53/4 to 57/0. |

Evidence: 32 focused file/SQL/cache tests pass (24 reading mutation + 8 policy),
seven artifact-store script checks pass; PDF discovery 184 run / 182 pass / two
opt-in skips. Separate summary/note discovery 120 run / 28 pass / 92 opt-in skips
is not a claim that skipped cases passed. Real Sample/Sample2 and browser tests
are described above. Frontend TypeScript/build passes with the existing bundle-size
warning. git diff --check passes; direct checklist count is 57 checked / 0 unchecked.

implementation-validation: implementation task, cross-module OCR review boundary.
No root sections/structure_nodes authority, profile write-back or hidden layout write
was introduced. Source tree/manual edits are scoped; no files removed. New files:
ocr_review.py, reading_content.py, ReadingContentActions.tsx, reading mutation
file/SQL tests and backup/browser acceptance harnesses. No commit or deployment.
Checklist-progress-sync modified only the owned loader checklist and progress;
implementation code changes were outside that governance skill. Validation itself
modified no files. Overall WARNING only for bounded coverage/component acceptance,
existing build-size warning and unperformed production deployment, not failing tests.

## Chinese Margin Recovery Acceptance (2026-10-21)

Supporting evidence for doc_loaders OCR-REVIEW-01/02; no duplicate checklist count.
`accept_chinese_margin.py --corpus data/test-books/ocr-review-20261021` verifies
backup SHA-256 before/after real OCR, all three physical pages, confirmed blank
page 2, representative page-3 body phrase, exact margin `新千年版序 003`, component
evidence and source offsets. It does not measure whole-document accuracy or test
partial-success import. Output is opt-in via --output; no API/LLM/profile calls.
Sample2 extraction passes (450/0/644 characters, 7.50 seconds). Original six-page
Sample and real-image mixed-layout regression pass with unchanged reference CER.
28 focused tests and 182 PDF tests (180 pass/two skips) pass in read-only-source,
network-disabled Docker. The independent synthetic-PDF real-engine Fast/OCR/Auto
matrix passes 18/18, including warm reuse and failure-preserved storage. General
review UI/block mutation remains future work.

## Chinese Direction OCR Acceptance (2026-10-21)

Owner queue: doc_loaders OCR-DIR-01..07, not a duplicate scripts task group.
`test_chinese_ocr_strategy.py` covers ordered columns, blank/horizontal images,
non-Chinese bypass, pinned-model failures, model identity/Fast isolation, source
offsets, garbage/confidence/coverage rejection, mixed bands and crop invariance.
Its sideways synthetic case checks implausible transcription size, not actual
upside-down character recognition. Do not claim universal rotation support.

Private immutable source/manifest: data/test-books/ocr-direction-20261021. The
sample is ignored by git and Docker context. `accept_chinese_direction.py` runs
real Poppler/Tesseract without LLM/profile/API/persistence and verifies source hash
before/after. Reference CER is limited to reviewed TOC titles with enumeration
normalization, not decorative headings, punctuation or the entire book. Numeric
labels and 12/17 column ordering are exact checks. Page 3 has a representative
horizontal phrase check only. Decorative 目錄 errors remain explicitly recorded.

Commands from backend root (Docker provides pinned engines/models):
```sh
python -m scripts.accept_chinese_direction --corpus data/test-books/ocr-direction-20261021 --mixed-fixture --output data/test-books/ocr-direction-20261021/acceptance-ocr.json
python -m scripts.accept_chinese_direction --corpus data/test-books/ocr-direction-20261021 --mode auto --expect-native-coverage-rejection --output data/test-books/ocr-direction-20261021/acceptance-auto-boundary.json
python scripts/accept_pdf_modes.py --synthetic-only --pdf /unused.pdf
```

OCR reference: 4.82%/1.85% CJK title error against an 8% ceiling, exact numeric
associations, mixed real-image header/body pass and Fast zero subprocesses. Auto
is an expected native_coverage_unresolved rejection, not positive extraction.
The independent three-mode matrix passes 18/18 real-engine cases (synthetic PDF
sources), including warm reuse and unchanged storage on failure. PDF unit suite:
180 run, 178 pass/two skips. Focused source/preparation/content tests and Chinese
tests are overlapping suites, not additive distinct-test totals.

Tests use disposable container storage; source tree/data must remain read-only
except the explicit private report directory. When mounting source at /source,
use a tmpfs at /source/data for legacy test profile locks and PYTHONPATH containing
/source and /source/scripts. Initial failures without these test-environment
settings were corrected by isolation, not by changing production persistence.
No live Library document or profile was deleted/replaced during this task.

## Loader Boundary Governance (2026-10-21)

**[Code-Confirmed]** LOADER-04 accepted. Final Docker runs: 180 PDF tests pass;
95 focused loader/preparation/content/source/LLM-contract tests pass; PostgreSQL
generic extraction/inherited enhancement tests: 88 pass, two opt-in skips.
Local focused combined run: 121 pass; standalone structured-reuse script passes.
These suites overlap. Obsolete tests now verify SourceSession handoff instead of
pdf_request tuples, source verification without extraction instead of forbidding
factory resolution, and rejection on source replacement instead of saving stale
text. Manual evidence now reuses saved spans without an additional extraction.

accept_loader_boundary_books.py uses verified immutable backup bytes, real loaders
and Common builder, controlled language/profile providers and in-memory storage.
Private loader-boundary-20261021T133550.json: Madame Bovary 652001 characters,
4 chapters/36 sections; Red-book 147401 characters, one section. Both exact digests
match prior accepted sources, one cold extraction and zero warm extractions/profile
calls; Library profiles unchanged. No actual EPUB decoder or live LLM tested here.
Final content-recovery-20261021T133544.json uses --read-only: all five interaction
reads pass, profile hashes stable, full-section HTTP reads 0.306-0.651 seconds.
An earlier invocation without that flag forced Common/default options on Red-book,
triggering common_policy_mismatch and layout invalidation. Explicit layout-only
recovery restored 93 units; do not claim the earlier invocation was read-only.
Use --read-only for existing Library acceptance; rebuild tests belong to isolated
storage or an explicitly authorized destructive reset. The flag rejects combination
with --allow-destructive-rebuild.

LOADER-04 covers request-local extraction sessions, cache reuse without extraction,
third-format injection without pipeline changes, source changes before commit,
optional-evidence failure and Fast/OCR/Auto isolation. Reuse the immutable books
under data/test-books/summary-note-20261021; verify backups before any run. This
refactor uses isolated/read-only real-book acceptance and must not delete Library
books or replace profiles. Test synthetic third formats without claiming production
upload support. Record exact commands/counts and limitations before completion.

## Generic Extraction Real-Book Acceptance (2026-10-21)

**[Code-Confirmed]** EXTRACT-10 is accepted; see [the preparation plan](../document_preparation/module-detailed-design.md#format-neutral-extraction-plan-2026-10-21).
Remove PDF-only assumptions from the acceptance corpus adapter, not from historical
immutable backups. Version new manifests; continue reading old PDF fixtures.
Madame Bovary.txt and 红色赌盘_261014_145614.pdf must be tested from verified private
backups under data/test-books, in isolated storage. Exercise actual LLM enhancement,
persisted-source readback, commit and automatic reading layout for both formats.
A Common-only/content-read result is insufficient. Capture conservation, profile
immutability, call counts and failure-stage evidence; require all negative cases
and existing PDF strategy regressions from the master design. Paid/destructive runs
are explicit opt-in; the subsequent implementation request authorized the runs below.

Delivery evidence: private fa794acef055f235/20261021T083239624177Z/api-live.json
records Madame Bovary direct API plus actual model, atomic commit and real layout:
three calls, 5 chapters, 31 sections, 423 units, exact 652001-character conservation
and unchanged profile hashes. The first live API run failed on an unlocated
source-title downgrade; semantic v16 now reports the offending quote index before
application, retaining the guard. Private api-20261021T084827447743Z captures the
final requests/responses/result. Red-book 085747b65094d52f/20261021T083612482751Z
records two actual calls, 11 chapters/30 sections and API replay/108 real-layout
units. Both tests use immutable backups and disposable PostgreSQL schemas.
The user authorized live acceptance in the implementation turn; no Library book
was deleted. The explicit TXT provenance migration is the only Library mutation.
Final suite: 95 focused tests, 180 PDF tests, 75 PostgreSQL regressions plus two
opt-in skips; desktop/mobile confirmation tests and UI build pass. These overlapping
suites must not be reported as a distinct-test total or a universal model guarantee.

## Content Recovery Acceptance (2026-10-21)

CONTENT-05 uses accept_content_read_books.py with optional explicit
--allow-destructive-rebuild. Existing immutable backup manifest must verify before
any deletion; backup books are the sole reupload source. The reset confirms raw,
profile and DB document absence, then Common Fast and progressive layout readiness.
Acceptance compares layout before/after repeated prepare, tests largest section
three times under a 10-second HTTP limit, reads all five interaction kinds, and
compares profile SHA-256. Reports remain private under the existing test corpus.
UI test-content-read-live.mjs verifies bounded batches, large-section latency,
selection cancellation, block actions and zero prepare/generate calls at 1280/390px.

## Summary And Note Real Books (2026-10-21)

SN-16/20 evidence: `accept_summary_note_books.py` supports explicit backup,
--operation rebuild --allow-destructive-rebuild, and --operation accept
--allow-live-model. Only the user-named Madame Bovary.txt and red-gamble PDF are
eligible. Backups live in data/test-books/summary-note-20261021; immutable manifest
hashes cover raw copies, current DB rows and existing derived files. Verify first,
delete via product lifecycle API, verify raw and DB absence, upload backup bytes,
Common Fast prepare, then progressive layout preparation/read and five-level
Summary/Note checks. Upload 201 is success; task-layout is POST. No original-source
fallback during reupload. Live output is private; reports contain persisted summary
responses and safe result counts. Do not rerun destructive rebuild casually.
Final backend acceptance 06:44:38 UTC passes ten workflows. Frontend companion
test-summary-note-live.mjs exercises actual APIs at desktop/mobile and saves ignored
screenshots. Synthetic contracts/faults supplement, never replace these live tests.
After acceptance, --operation cleanup-notes deletes only exact fixture-authored
notes using current revision/token, refuses changed user content, and keeps real
summaries. Final cleanup report confirms ten fixture notes removed and backup
hashes unchanged. Run accept before browser checks to seed their fixture notes.

## Real-Data Acceptance Policy (2026-10-21)

User-facing feature acceptance must use representative real books. Synthetic/unit
fixtures remain useful for contracts, privacy and fault injection, but cannot alone
close a real-document/model incident. A skipped real-data test is not a pass. For
LLM-dependent changes, capture explicitly authorized live-model evidence and compare
the applied outcome against the accepted source, not merely HTTP status or mocks.
Valid output is not automatically useful navigation; report both result and limits.

`accept_enhancement_real_book.py --backup-doc-name NAME` explicitly captures an
immutable PDF, accepted snapshot and original hierarchy before testing. Subsequent
`accept_enhancement_real_book.py --case-id CASE --allow-live-model
--require-structure-change` reads only that backup, captures each paid
request/response privately, forbids extraction, validates exact body preservation,
requires actual structural change when requested, and verifies production snapshot
and backup identity remain unchanged. Production document reads are blocked during
generation. It exits nonzero on failure. Run from
the backend after sourcing `../scripts/load_local_env.sh`, with explicit PostgreSQL
storage backend and local DSN. No keys are placed in command arguments or reports.

Private corpus: `Deep_Reflective_Reader/data/test-books/<hashed-case-id>/`, ignored
by Git and Docker through existing data exclusions. The old root directory has
been migrated with its attempt history. `source.pdf`, `snapshot.json` and
`original-document.json` form the independent fixture; `manifest.json` checks all
three hashes, followed by extraction/source/hierarchy consistency validation.
Existing differing backup bytes are never overwritten. Library/raw-file deletion
cannot remove this separate fixture. Missing/corrupt backups fail closed; no live
Library fallback is allowed. Current capture supports the accepted PDF test path.
Historical `accepted-source.json` remains evidence, not the new test input.
UTC attempt folders retain numbered requests/responses, candidate and safe report.
Private directory/file permissions are 0700/0600. Never commit books, extracted
prose, prompts, responses or credentials. Commit only harnesses, hashes/counts and
sanitized failure shapes. Use authorized local books; do not download test books.

`verify_real_book_enhancement_api.py --attempt-dir PATH --allow-live-embeddings`
requires `DEEP_READER_TEST_POSTGRES_DSN`. It replays an accepted real-model response
against a copy of that same accepted source in a disposable schema, through real
preflight/reparse/prepare-layout/read APIs and the real task-unit resolver. Its
embedding calls are paid and explicitly enabled; planner replay spends no extra
generation calls. Check structural commit, full seeded artifact deletion, nonempty
units, exact source and unchanged original DB; always remove the disposable schema.
Replay also loads the backed-up hierarchy, never the original Library snapshot;
the original repository read methods are guarded against accidental access.
Original interactions are never cleared by either harness. Both test paths can
operate after the original Library entry/raw PDF is deleted, provided the backup
is intact. PostgreSQL is still required for isolated API integration, not for
retrieving the original book.

Backup-only validation (2026-10-21): four corpus regression tests pass, including
deleting the temporary original PDF and disabling original repository reads.
Live case `085747b65094d52f`, attempt `20261021T052028529811Z`, succeeds with
18 chapters/35 sections in three calls and exact 147401-character preservation.
Existing accepted output also replays from the backed-up hierarchy through real
API/disposable PostgreSQL/layout with 107 units. Original production reads are
blocked during both tests; no actual Library deletion was performed.

Semantic-plan acceptance supersedes the historical ENH-REF ID-copying runs below.
The live harness captures the actual request-local output allowance and native
response schema, including local quote-repair requests. Replay patches the same
provider entrypoint as production, not an unused legacy completion method. Require
independent live runs on the final code and an accepted-response API/layout replay;
failed development attempts remain private evidence, never reported as passes.
Regression fixtures cover ID-free output, exact quotes, repeated openings, Unicode
paragraph indentation, unsupported cuts, full body reconstruction and bounded
repair with source/configuration/capacity guards. Full-source first dispatch and
local correction dispatch have distinct assertions. Original book storage remains
read-only; automatic OCR/extraction during enhancement is forbidden in the harness.

Final SEM evidence: case `085747b65094d52f`, v15 attempts
`20261021T044839792942Z` and `20261021T044933593384Z` pass independently (3/33 and
4/27 chapters/sections; 2 and 3 calls). All 147401 source characters and original
production state are unchanged. First response passes isolated real API/PG/layout
replay with 107 task units and seeded artifact deletion. Final backend lifecycle
suite runs 180 cases (179 pass, one named opt-in skip); live harnesses are separate.
Model semantic grouping varies and remains subject to user review, not inferred
from exact body coverage alone. Only the final-code runs count toward SEM-04.

ENH-REF evidence: final live implementation succeeded in two independent real-book
runs, with two provider calls each (initial invalid ID plus accepted correction).
The 12-chapter/25-section result passed API/PG/layout replay with 106 units. Six
focused repair tests and safe-reference/metrics suites pass 21 cases. Failure
injection covers unknown IDs, duplicate chapter/section starts, false source-region
claims, exhaustion, no retry on valid/no-op/provider/internal outcomes, expanded
capacity limits and confirmation identity. PostgreSQL tests verify no intermediate
writes during correction and unchanged storage on exhaustion. The document_structure
checklist owns this delivery; do not count a duplicate scripts feature task.

## ENH-REV-06 Acceptance Entry (2026-10-21)

`accept_enhancement_lifecycle.py` runs nine backend test modules, the UI build,
three mocked browser suites and expanded real API/disposable PostgreSQL browser
acceptance. It requires an explicit test DSN, running UI and installed Playwright;
missing integration prerequisites fail closed. Only the named private-source opt-in
test may skip. `accept_enhancement_browser.py` now verifies unchanged row/xmin parity,
title-only ID/payload/snapshot retention and structural cleanup while both viewports
exercise transient retry and authoritative lost-response recovery. Its synthetic
scenario controls exist only on the disposable HTTP adapter, never in product API.
The detailed matrix, reproduction, 155-test count and semantic-quality limits are in
document_structure's ENH-REV-06 design section. Completion belongs to that checklist;
no additional scripts task is counted and no paid model is evaluated.

## ENH-PREP-03 Acceptance (2026-10-19)

test_enhancement_postgres_preflight.py runs 30 cases with explicit
DEEP_READER_TEST_POSTGRES_DSN and ENHANCEMENT_SOURCE_DOCUMENT. Without the latter,
the opt-in source case is skipped, not passed. Original DB/source are read-only;
copies live in temporary storage/disposable schemas and are explicitly prepared if
legacy extraction identity is absent. Enhancement forbids raw/OCR re-entry. The
reported four-page fixture preserved 8,190 canonical characters.

accept_enhancement_browser.py serves only enhancement/layout routes for its disposable
document, forwards to real FastAPI/PostgreSQL and runs UI's test-enhancement-live.mjs
at 1280px/390px. Requires running UI_URL (default 5173), PLAYWRIGHT_MODULE and optional
CHROMIUM_PATH. Planner/unit resolver are deterministic fixtures. All tests and opt-in
case passed; screenshots inspected. No paid provider or production deployment claim.

Root scripts/start_local_dev.sh waits for PostgreSQL, then applies migration 003 with
psql ON_ERROR_STOP and --single-transaction before starting API. Fresh Compose volumes
mount the same migration; existing-volume upgrades do not rely on initdb. Failure
stops startup. SQL is exercised in disposable schemas, shell syntax passes, installed
Compose supports --wait. Startup was not run against user services in this task.
Completion belongs to document_preparation, not a new scripts checklist item.

## NATIVE-06 Acceptance Entry (2026-10-19)

`accept_native_recovery.py` aggregates 160 existing contract/regression cases and
adds opt-in real-PDF hierarchy/interaction round trips in temporary storage plus
live PostgreSQL replacement/rollback/artifact checks in a disposable schema.
All groups passed on 2026-10-19 for both reported local PDFs. The exact matrix,
counts, reproduction and limits are recorded in document_structure's NATIVE-06
section; completion belongs there, with no extra scripts checklist item counted.
No private source text is committed, no real LLM is called, and no production
document is modified. Missing options report not run; failures exit nonzero.

## READER-01 Contracts (2026-10-19)

`test_reader_provenance.py` covers four cases: legacy/invalid provenance stays
unknown, real-route allowlisted serialization excludes source/prompt payloads,
enhanced declaration uncertainty, and exact-range section origins through the
coordinator/file store/API with unchanged stored bytes. The fixture explicitly
uses the current resolver identity. Eleven task-layout readiness tests also pass.
Frontend owns seven mocked browser suites, including new reader-recovery cases
on desktop/mobile; see its module design/checklist. No live provider or deployment
is claimed, and NATIVE-06 remains the separate final acceptance task.

## INTERACT-03 Cache Recovery Tests (2026-10-19)

`test_interaction_policy_identity.py` adds eight real-file-store tests, including
real API/coordinator calls with fake preparation/providers. Four suites pass 38
cases; 12 standalone interaction/store/cleanup regressions pass. Reuse fixtures
now explicitly stamp current policy; client-version expectations now assert server
authority. Snapshot comparisons prove interaction reads do not write file storage.
See section_tasks delivery for exact coverage and the unchanged failing legacy
root-sections-only task-layout availability fixture. Live DB/LLM and deployment
remain outside this task's evidence.

## INTERACT-02 Language Tests (2026-10-19)

`test_interaction_output_language.py` adds 12 fake-provider cases for shared prompt
policy, precedence/fallback, explicit variants, conservative generated-field checks,
no retry, matching outputs and thinking answer preservation. Alongside INTERACT-01
and NATIVE-01 baselines, 30 unittest tests pass. Thirteen standalone interaction
regressions also pass; the exact set and guard limitations are recorded in the
section_tasks delivery note. No real LLM or deployment is part of this evidence.

## INTERACT-01 Guard Tests (2026-10-19)

`test_fragmented_interaction_preflight.py` adds eight cases, including all 15
service/target-level combinations with fake generators and exact input preservation.
The final NATIVE-01 expected failure is now a passing assertion. Six suites pass
61 cases; standalone artifact validity, analysis, quiz, thinking and target resolver
scripts pass. Existing metadata-only PDF diagnostics accept both reported local
fast-extracted sources without changing files, invoking OCR or calling a provider.
See section_tasks detailed design for counts, guard limitations and cache follow-up.

## NATIVE-05 Application Tests (2026-10-19)

`test_structure_plan_application.py` covers 12 cases: single-body execution without
Common merging, mapped anchors/exact occurrences, generated origins/word-boundary
guards, source heading evidence, contradictory bookmarks, complete source coverage,
failed-attempt preservation, mode-aware reuse and real Prepare/Reparse routes with
isolated dependencies. The NATIVE-01 cross-line contract now passes through the v2
resolver rather than the compatibility matcher. Only INTERACT-01 remains expected
failure. Eight unittest suites total 92 cases (91 pass, one expected failure).
Six standalone scripts pass: heading normalization, v1 region-plan compatibility,
TOC projection, hard-reparse artifact cleanup, structured reuse and Common profile
ordering. Projection expectations now retain source prefix; PDF outline positive
fixtures have real standalone source headings instead of inline prose prefixes.
Explicit enhanced requests are tested as taking precedence over Outline projection.
The profile-order test double accepts the existing extraction argument. No private
document, paid provider, live database or deployment is used for these checks.

## NATIVE-04 Planning Tests (2026-10-19)

`test_structure_plan_contract.py` adds 11 isolated tests for v2 outcomes, strict
fields/origins, exact source references, omitted ranges, intent/TOC/language prompts,
provider failure versus invalid output, request defaults and no-write staged runtime
rejection. API contracts cover all three intent handoffs. Combined seven suites:
80 cases, 78 pass, two existing expected failures. Legacy LLM region-plan smoke
passes. Providers are fake; no OCR, live LLM or private-document mutation occurs.

## NATIVE-03 Regression Delivery (2026-10-19)

`test_common_parse_policy.py` covers typed hints, single-body readiness, exact
source preservation, real-preface/heading controls, TOC/layout bypass and policy
cache invalidation/reuse. Three NATIVE-01 expected failures are now ordinary
passing assertions; the semantic guard and LLM anchor cases stay expected failures.
API contracts also verify independent TOC/PDF options and reparse forwarding.
Combined six-suite run: 68 tests, 66 passing and two expected failures. Standalone
heading normalization, LLM region-plan, TOC projection and structured reuse checks
pass. Tests use synthetic sources and isolated dependencies, not private-document
mutation, OCR subprocesses or paid LLM requests.

## 1. Module Purpose

`scripts/` 目前承擔主要 regression/smoke 測試入口（以 Python 腳本與 shell 腳本為主），覆蓋 hierarchy、task-layout、artifact persistence、profile metadata、real-document 流程等。 **[Code-Confirmed]**

## 2. Position in Overall Architecture

- Tests Layer（package-like scripts folder）

## 3. Key Files

| File | Responsibility | Notes |
|---|---|---|
| `scripts/test_task_layout_*.py` | task-layout projection/availability/persistence 回歸 | hierarchy-first 與 diagnostics **[Code-Confirmed]** |
| `scripts/test_hierarchy_*.py` | hierarchy write/read/target resolution 回歸 | section/chapter/task-unit 路徑 **[Code-Confirmed]** |
| `scripts/test_document_profile_*.py` | profile/parser_metadata/post_structure metadata 回歸 | schema/fallback/semantics **[Code-Confirmed]** |
| `scripts/test_post_structure_metadata_enrichment.py` | post metadata enrichment 統計與風險判斷 | title uniqueness/shape/task-unit coverage **[Code-Confirmed]** |
| `scripts/test_language_*` | script/discourse/registry 能力回歸 | multi-language policy **[Code-Confirmed]** |
| `scripts/test_rest_*.sh` | REST smoke/manual 验证輔助 | 非單元測試 **[Code-Confirmed]** |
| `scripts/test_upload_preparation_logging.py` | Request correlation, upload/preparation stages, reuse, failure propagation, and content exclusion | Temporary files and fake dependencies; no LLM calls **[Code-Confirmed]** |
| `scripts/test_document_delete_route.py` | Explicit deletion, missing raw source, DB failure/retry, both catalog readbacks | Temporary files and fake PostgreSQL store; no LLM calls **[Code-Confirmed]** |
| `scripts/test_postgres_document_delete.py` | Real PostgreSQL cascades and namespace isolation | Requires `DEEP_READER_TEST_POSTGRES_DSN`; creates/drops a unique test schema, never deletes existing document rows **[Code-Confirmed]** |

## 4. Main Responsibilities

1. 提供主要 regression 保護（目前未見獨立 `tests/` 目錄）。 **[Code-Confirmed]**
2. 提供真實文檔流程 smoke scripts。 **[Code-Confirmed]**
3. 驗證 legacy fallback 收斂與 hierarchy-only 進度。 **[Code-Confirmed]**

## 5. Non-Responsibilities

1. 不負責 production runtime。 **[Code-Confirmed]**
2. 不應承擔 API 文件。 **[Code-Confirmed]**

## 6. Important Data Structures / Contracts

- 無單一核心 DTO；主要消費各 module contract
- 關鍵測試群組契約：task-layout、profile metadata、hierarchy persistence

## 7. Module Relationships

- used by: 開發者/CI
- depends on: 全專案 runtime modules

## 8. Main Flows Involving This Module

1. regression test flow。 **[Code-Confirmed]**
2. real-document REST smoke flow。 **[Code-Confirmed]**

## 9. Persistence / Side Effects

- read/write persistence：部分（測試流程會讀寫 artifacts）
- external service：部分（需 OpenAI/embedding 時）

## 10. Known Legacy / Compatibility Behavior

1. 多個測試專門覆蓋 legacy 相容讀取與 fallback 收緊退場。 **[Code-Confirmed]**

## 11. Current Risks

1. risk：scripts 作為主要測試層，命名與分群易擴散
- why：維護與定位成本升高
- guardrail：維持 module docs 索引與測試群組規範

2. risk：部分測試依賴外部 API/環境
- why：本地可重現性不穩
- guardrail：區分 pure unit-like vs integration smoke

## 12. Open Questions for Maintainer

1. `scripts/` 是否長期作為正式 test layer，還是將來遷移到 `tests/`？ **[Needs Confirmation]**
2. 是否需要定義 smoke test 最小集與 full 集？ **[Needs Confirmation]**

## 13. Suggested Next Documentation Improvements

1. 建立 scripts 測試分類索引（unit-like / integration / real-doc smoke）。

## 14. Future Direction Note: Reading Interaction Regression Coverage

> 本節記錄 future tests for reading interactions；不代表目前 tests 已存在。 **[Future Direction]**

1. Regression coverage should prove read routes for persisted interaction artifacts do not auto-generate or call LLM when artifacts are absent. The first analysis/insight route vertical slice covers read/generate/refresh route dispatch separation, compact response mapping, HTTP status mapping, and a read-only poison coordinator check proving the exposed insight read route returns `not_generated` for absent artifacts without non-read coordinator access. **[Code-Confirmed] + [Future Direction]**
2. Target resolver coverage should verify document/chapter/section/task-unit ids, optional parent consistency checks, and fail-fast rejection of title-primary or legacy fallback targeting. **[Maintainer-Provided] + [Future Direction]**
3. Analysis coverage validates strict JSON success, insufficient-content success, invalid-output failure, current-artifact read behavior, explicit read/generate split semantics, document-backed current-artifact store persistence, public insight route mapping, no-auto-generation read-route behavior, and insight HTTP status mapping for malformed, missing target, stale target, generation failure, validation failure, recoverable terminal status, and success cases. **[Code-Confirmed]**
4. Quiz coverage validates configured max counts, allowed type enum, fewer-than-max generation, answer payload validation, insufficient-content behavior, current-artifact read behavior, explicit read/generate split semantics, public quiz route mapping, legacy section/chapter route separation, and route-level rejection of invalid request count or generated item type shapes. **[Code-Confirmed]**
5. Critical-thinking coverage validates question generation persistence, generated-but-unanswered status, answer submission, evaluation failure recovery, retry behavior, completed status, public route mapping, no-generation read behavior, and route-level malformed submit rejection before dispatch. **[Code-Confirmed]**
6. Public route regression coverage now spans the complete frontend-facing interaction exposure set: insight read/generate/refresh, quiz read/generate/refresh, and critical-thinking read/generate-question/submit-answer/retry-evaluation. **[Code-Confirmed]**
7. Context strategy coverage verifies full-target mode within model budget, semantic compact fallback when the target exceeds capability, provenance metadata, and no title fallback for empty resolved content. **[Code-Confirmed]**
8. Artifact persistence coverage verifies hierarchy-aware replacement cleanup for interaction artifacts, including file-backed structured payload cleanup and PostgreSQL parser-level replacement payload sanitization. **[Code-Confirmed]**
9. Artifact-aware context coverage should verify higher-level generation can include compact lower-level artifact summaries as secondary context without replacing primary source context. **[Maintainer-Provided] + [Future Direction]**
10. Deduplication coverage should verify higher-level quiz generation receives lower-level quiz coverage signals and avoids repeating equivalent questions when possible. **[Maintainer-Provided] + [Future Direction]**
11. Critical-thinking abstraction coverage should verify higher-level critical-thinking prompts can build on lower-level session focus areas while still producing one first-version question. **[Maintainer-Provided] + [Future Direction]**
12. Persistence coverage verifies generated higher-level artifacts record referenced artifact ids/metadata, and hard reparse cleanup removes referenced-artifact metadata from replacement structured payloads. **[Code-Confirmed]**
13. Common artifact mapping coverage verifies `analysis`, `quiz`, and `critical_thinking_session` reading interaction artifacts map to one shared `CommonArtifact` entity shape with hierarchy-aware targets and no category-specific serialization roots. **[Code-Confirmed]**
14. Document-backed analysis artifact store coverage verifies current analysis artifacts persist under document task artifact metadata as shared `CommonArtifact` payloads and can be read back by deterministic artifact id. **[Code-Confirmed]**

## 15. Selective OCR Validation (2026-10-14)

Track definition: `../doc_loaders/module-detailed-design.md` section 16.
The OCR-01 through OCR-07 entries below are delivery-time evidence; current final
acceptance and limitations are recorded under OCR-08 at the end of this section.
OCR-01 contract tests in `test_pdf_ocr_decision.py` are pure standard-library
tests: enum values, required
reasons, valid source/page identity, immutable result, metadata serialization and
independent text/layout purposes. These are supporting evidence for OCR-01, not
an extra delivery task. All 7 tests passed on 2026-10-14, along with existing
PDF loader inspection and page-evidence regressions. **[Code-Confirmed]**

OCR-02 adds `test_pdf_page_inspection.py` (18 passing tests on 2026-10-14).
Synthetic pypdf fixtures exercise native text reuse, multilingual quality signals,
blank/vector distinction, drawn versus unused images, clipped overlap union,
matrix composition/rotation, inline images and conservative nested-Form fallback.
Poisoned image decoding and subprocess paths verify no image decoding/render/OCR;
additional cases cover missing geometry/resources, empty text with text operators,
extraction failure, hidden text, annotations, geometry limits, immutable metadata
without raw text and source replacement. Existing PDF loader inspection,
page-evidence, Outline and OCR-01 contract tests also pass. Tests support OCR-02
only; runtime routing and OCR-08 acceptance remain pending. **[Code-Confirmed]**

The 18 inspection tests also passed in a disposable Compose container with
pypdf 6.17.0, in addition to local pypdf 6.9.2. Only temporary synthetic files
were written; project source was mounted read-only. **[Code-Confirmed]**

OCR-08 owns final acceptance after runtime integration. Offline fixtures must
cover native Chinese/Latin, scans, mixed native/scanned pages, hidden OCR text,
garbled mappings, short headers, blank/illustration pages, nested image forms,
rotation/vertical text, missing engines, budget exhaustion and stale-source reuse.
Use render/OCR call counts as deterministic cost assertions; separately validate
raw text/order/offsets, hierarchy purity and fallback. **[Future Direction]**

The reported four-page `《歸來》` PDF is a real-document smoke target, together
with Outline/native, horizontal and vertical scanned fixtures when available.
Compare cold preparation and warm reuse separately on the same runtime/settings;
report stage durations, OCR calls and extraction quality, not only total latency.
Baseline: 51,352.1 ms layout evidence within a 63,140.3 ms task-layout request on
2026-10-14 (request `ee99839b50204673a79bf602c32c90cf`). No fixed speedup promise.
Use disposable document copies/storage for explicit preparation, never overwrite
user artifacts just to benchmark. Missing real fixtures/engines are recorded as
unverified, not passed. **[Code-Observed] + [Future Direction]**

OCR-03 adds `test_pdf_ocr_policy.py`: 22 tests cover native Chinese/Latin/numeric/
mixed text, blank pages, images versus confirmed scanned/non-text pages, sparse
headers, corrupt/duplicate hidden text, incomplete or conflicting evidence,
source/page identity, separate text/layout needs, exact threshold boundaries and
configuration fingerprints. Probe tests cover disabled/unavailable/exhausted/zero
budgets and unchanged uncertainty; synthetic PDF handoff poisons subprocess calls.
Visual checks are supplied fixtures, not real detector output. The suite passes
locally and in isolated Docker; existing OCR decision (7), page inspection (18)
and loader inspection regressions also pass. Supporting tests count only toward
OCR-03, not a separate progress item or OCR-08 acceptance. **[Code-Confirmed]**

OCR-04 adds `test_pdf_layout_routing.py`: 14 tests pass locally and in an isolated
Compose container on 2026-10-14. Real synthetic PDF streams verify one native
extraction, zero render/OCR/image decoding for native/blank/disabled/zero-candidate
paths, prefix rather than selected-count limits, explicit uncertain native
fallback, retained horizontal/vertical/rotated/mixed candidates and page-local
failure recovery. Renderer failure can use embedded images; unavailable engine,
decode failure, timeout and empty OCR retain inventory/native text. Rendered
coordinate evidence uses image pixels rather than PDF point dimensions. Raw text,
boundaries, input bytes and raw OCR provenance remain unchanged. The existing
loader metadata fixture now accepts pypdf's visitor keyword in its test double.
Earlier policy/inspection/decision, loader, page-evidence, Outline and OCR fixture
regressions pass. **[Code-Confirmed]**

Read-only local `《歸來》` layout smoke retained 8,187 native characters over four
pages with zero render/OCR calls, measured at 346.1 ms with poisoned expensive
paths. Native-only uncertain evidence remained explicitly uncertain. This does
not count as OCR-08 end-to-end cold/warm acceptance or deployed speedup; neither
the live API nor stored documents were changed. **[Code-Observed]**

OCR-05 adds `test_pdf_raw_recovery.py`: 25 tests pass locally and in isolated
Compose on 2026-10-14. Coverage includes mixed-page recovery, exact native
whitespace, scan/native/blank boundary parity, one inspection per page, explicit
OCR enablement and prefix limits, full-page replacement without duplicated headers,
native coverage rejection, per-page quality failures, retry/state clearing,
source replacement/change, unavailable engines, timeout/OSError, corrupted text,
uniform rendered blank proof, render cardinality, coordinate-independent raw text,
opaque graphics state/background exceptions and single-glyph duplicate metrics.
OCR and rendering are controlled fixtures, not real-scan accuracy claims.
**[Code-Confirmed]**

Existing loader tests now reject partial-book success with a one-page budget and
expect page-local quality failure; their native page double implements the visitor
contract. The real Outline fixture contains a raster cover and unresolved final
page: disabled recovery must fail, then a mock recovers only page indexes 0/123.
The unchanged builder preserves front matter before the first two chapters once
the recovered cover no longer shares an empty-page offset. The test does not call
Tesseract or modify user artifacts. Earlier PDF policy/inspection/layout/decision,
page evidence, OCR fixtures, upload/preparation logging, manual source-evidence
and preparation-handoff regressions also pass. **[Code-Confirmed]**

Read-only `《歸來》` smoke returns 8,200 canonical characters with 8,187 stripped
native characters across four pages; each boundary slices back to the exact page.
OCR is disabled and subprocesses poisoned. Full cold/warm and real-scan quality
acceptance remain OCR-08; no deployed speedup is claimed. **[Code-Observed]**

OCR-06 adds `test_prepare_lazy_pdf_layout.py`: 15 tests pass locally and in an
isolated Compose container on 2026-10-14. Temporary native/scanned PDFs exercise
one canonical extraction per page, shared raw/span offsets, one build/save,
accepted/rejected/missing Outline, optional provider failure, hash mismatch,
changed-source refresh, required body recovery before Outline, cache reuse and
force rebuild. Direct builder checks cover eager-provider compatibility, internal
paragraph separators, invalid ranges/destinations and enhanced splitter call
counts. Rendering/recovery and LLM dependencies are mocked where applicable.
**[Test-Confirmed]**

The 86 earlier PDF unittest cases pass, as do standalone Outline, page-evidence,
TOC, loader inspection/layout fixtures, raw-error mapping, structured reuse,
profile-before-structure ordering, manual source-evidence/handoff and logging
scripts. Raw-error/order test doubles now follow the private `_load_raw_source`
text-plus-boundaries handoff; public response/error contracts remain unchanged.
This is supporting evidence for preparation-owned OCR-06, not a new task or
OCR-08 full cold/warm performance acceptance. No live API or user artifact was
changed. **[Test-Confirmed]**

OCR-07 adds `test_pdf_execution_pass.py` with 25 tests. On 2026-10-14, all 111 PDF
unittest cases pass locally and in isolated Compose; the 15 lazy-layout tests
and 11 standalone PDF/TOC/preparation/manual/logging regression scripts also pass
locally. Tests run actual budget wrappers with controlled subprocess outputs,
temporary PDFs/binary identities and injectable clocks, never production OCR/LLM
or user artifacts. Coverage includes bounded LRU storage and defensive copies,
per-pass and worker isolation, source/settings/binary/policy invalidation, unknown
identity misses, separate purpose counters, failed attempts, zero limits, clipped
timeouts, late completion rejection, required failure atomicity, optional inventory
fallback, request-correlated content-free logs and real factory scope integration.
**[Test-Confirmed]**

Earlier raw-recovery/loader memory tests explicitly open one preparation scope
and provide known fixture executable identities; standalone calls no longer
implicitly reuse the loader's previous OCR snapshot. Layout timeout tests now
expect `layout_analysis_time_budget_exhausted` when the budget caps the subprocess,
while other optional failures retain their prior fallback. These are supporting
tests for OCR-07, not extra checklist tasks. Cold/warm performance and real-scan
quality acceptance remain OCR-08; no deployed improvement is claimed.
**[Test-Confirmed]**

### OCR-08 Acceptance Results

Completed the scoped acceptance on 2026-10-14. `accept_selective_ocr.py` uses real
PDF inspection, Poppler rendering, Tesseract, the preparation pipeline, hierarchy
builder and JSON store. It substitutes deterministic language/profile and unused
retrieval dependencies from the existing preparation test fixtures. Thus it tests
the complete PDF-to-structured-readback path, NOT HTTP, PostgreSQL, real LLM costs
or the broader historical TOC/vertical-reading accuracy roadmap.
**[Test-Confirmed]**

Each case copies the source (or a declared one-based page excerpt) into a temporary
directory. A store adapter ignores production storage targets and confines all
structure writes to one temporary JSON file. Source SHA-256 is checked unchanged.
Successful preparation asserts ordered exact page spans, hierarchy-only JSON,
raw-text persistence parity and one save. Native-only cases additionally compare
the complete text with independent native extraction. Warm reuse poisons raw-loader
selection and profile generation, asserts zero external processes and verifies
the saved bytes are unchanged. Rejected cases assert no structure was saved and
report warm reuse as unavailable, not a fictitious cache success.
**[Test-Confirmed]**

Actual subprocess calls are counted independently and cross-checked against
purpose/stage execution logs. Stage timings come from existing logger arguments;
nested durations must not be summed as if independent. Fixture transcript checks
and source hashing outside `prepare` are excluded from its timing. The generated
horizontal scan asserts its full five-line transcript after whitespace folding;
the mixed native/scan/blank fixture also verifies exact native/blank spans and
accepted Outline with no optional layout OCR. Three harness regression tests cover
native cold/warm persistence, disabled recovery/no writes and excerpt/source
isolation. Missing engines remain hard failures for the synthetic real-engine run,
not silent skips. **[Test-Confirmed]**

Measured in disposable Compose with read-only source mount: Python 3.11.17,
pypdf 6.17.0, Tesseract 5.5.0, Poppler 25.03.0, 200 DPI, OCR explicitly enabled.
Real sources use `eng+chi_sim+chi_tra`; synthetic fixtures use `eng`.
The following are observations from the final run, not latency thresholds or
statistical benchmarks. Cold means no structured artifact, not cold OS disk caches.

| Fixture | Cold / warm ms | Raw / layout stage ms | Text render / OCR | Layout render / OCR | Result |
|---|---:|---:|---:|---:|---|
| Reported four-page `《歸來》` | 789.9 / 1.9 | 428.5 / 5.5 | 0 / 0 | 0 / 0 | Exact 8,200 canonical characters, four page spans |
| Generated native Latin | 8.7 / 0.8 | 2.0 / 0.6 | 0 / 0 | 0 / 0 | Exact 60 native characters |
| Generated horizontal scan | 4593.4 / 1.4 | 2901.4 / 1682.5 | 1 / 4 | 1 / 2 | Full known transcript, 299 characters |
| Generated native/scan/blank + Outline | 2717.4 / 1.6 | 2708.1 / skipped | 1 / 4 | 0 / 0 | Exact native/blank spans, transcript, Outline hierarchy |
| `暗水幽灵.pdf`, physical page 8 only | 10184.8 / 1.8 | 6526.0 / 3637.9 | 1 / 4 | 1 / 2 | 318 characters; runtime quality gate only |
| `國富論lite.pdf`, physical page 1 only | 28667.9 / unavailable | 28660.4 / not reached | 1 / 4 | 0 / 0 | `page_quality_gate_failed`; no partial hierarchy |

Every OCR-recovery case also executes one engine-version metadata command; all
successful warm reads execute zero commands. The original `《歸來》` SHA-256 is
`1d0f1b559d73bb1a533924d7fc3a3aa2c3748f8aeb55886adde2b36794ed0862`;
canonical text SHA-256 is
`146ac32ec4521c296cc6537c66879e2e3371ae1af088282bc51f4585dc61222c`.
The script emits source/excerpt hashes, settings, counts and timings as JSON lines
on stdout; existing pipeline progress prints are redirected to stderr.
**[Code-Observed]**

Regression matrix executed locally and in isolated Compose: 111 PDF tests,
15 lazy-preparation tests and 3 acceptance-harness tests (129 total).
Eleven existing standalone PDF/TOC/preparation/manual/logging scripts also pass
locally. Matrix ownership/evidence:

| Cases | Executed evidence |
|---|---|
| Native Chinese/Latin, actual horizontal scan, mixed/blank, cold/warm | Acceptance harness and real-engine synthetic run above |
| Hidden/garbled text, short headers, illustration/vector pages, nested Forms, rotations | `test_pdf_page_inspection.py`, `test_pdf_ocr_policy.py`, `test_pdf_layout_routing.py`, `test_pdf_raw_recovery.py` |
| Quality rejection, disabled/unavailable engine, prefix/zero/exhausted budgets | Raw-recovery/layout/execution tests; real rejected excerpt above |
| Exact order/offsets, Outline acceptance/rejection, fallback, source mismatch | Raw-recovery, lazy-preparation, Outline/page-evidence/TOC tests |
| Source/settings/binary invalidation, worker isolation, bounded cache/logging | `test_pdf_execution_pass.py` |

Reproduction from repository root (the real-engine command requires Tesseract,
Poppler and the configured language packs; the measurements above used Docker):

```bash
PYTHONPATH=Deep_Reflective_Reader Deep_Reflective_Reader/.venv/bin/python -m unittest discover -s Deep_Reflective_Reader/scripts -p 'test_pdf_*.py'
PYTHONPATH=Deep_Reflective_Reader Deep_Reflective_Reader/.venv/bin/python -m unittest discover -s Deep_Reflective_Reader/scripts -p 'test_prepare_lazy_pdf_layout.py'
PYTHONPATH=Deep_Reflective_Reader Deep_Reflective_Reader/.venv/bin/python -m unittest discover -s Deep_Reflective_Reader/scripts -p 'test_selective_ocr_acceptance.py'
PYTHONPATH=Deep_Reflective_Reader Deep_Reflective_Reader/.venv/bin/python Deep_Reflective_Reader/scripts/accept_selective_ocr.py --synthetic --pdf 'Deep_Reflective_Reader/data/raw/《歸來》白事／喪葬／鄉村儀式素材整理稿.pdf' --sample 'Deep_Reflective_Reader/data/raw/國富論lite.pdf' 1 --sample 'Deep_Reflective_Reader/data/raw/暗水幽灵.pdf' 8
```

Limitations: real scan excerpts have no independently transcribed ground truth;
runtime quality acceptance is not measured character accuracy. Full scanned books,
real vertical/rotated transcript accuracy and artistic TOC segmentation remain
unverified; synthetic orientation/fallback coverage is not a substitute. The
rejected `國富論lite` excerpt remains unsupported, not fixed by faster routing.
No live API was rebuilt/restarted and no user artifacts were modified. The old
51,352.1 ms layout observation is historical diagnostic evidence, not a matched
baseline permitting an end-to-end speedup ratio. Historical preparation checklist
items remain open. OCR-08 completes the eight-item routing/guard/cost acceptance
track, not every PDF accuracy or deployment concern. **[Code-Observed]**

## 16. Three-Mode PDF Acceptance Harness (2026-10-16)

`accept_pdf_modes.py` supports loader PDF-FIX-03, not an additional checklist task.
It uses real Poppler/Tesseract, synthetic native/scanned/mixed/blank/corrupt files,
zero-budget controls, and a required local 13-page bilingual fixture. Successful
cases verify ordered spans and mode-specific persisted reuse; typed failures
verify no storage mutation. Unexpected typed failures are recorded and execution
continues through the remaining modes; the aggregate result exits nonzero.

Run from `Deep_Reflective_Reader` in an isolated environment with the three OCR
language packs and renderer installed:

```bash
python scripts/accept_pdf_modes.py --pdf 'data/raw/20250515-为什么中式思维既不产生科学，也不产生民主.pdf'
```

`--synthetic-only` skips the actual file, not the required full acceptance gate.
JSON stdout records versions, source hash, timings, process counts, failure detail
and assertion outcomes; diagnostics go to stderr. No extracted text is printed.
Language/profile collaborators are test stubs; persistence is temporary JSON,
not production PostgreSQL. Use read-only source mounts and disabled networking.
Renderer fixtures use indirect content streams and explicit page bounds; parser-
only PDF fixtures are not assumed renderable. Fast parity trims page-edge spaces
to match the executor contract. Optional layout OCR is counted separately from
extraction, preventing false mode-boundary failures.

2026-10-16 result: 19/21 cases pass; actual-file OCR/auto quality rejection leaves
PDF-FIX-03 unresolved. Loader design 18.4 owns runtime versions, measurements,
limitations and remaining work. Local PDF regression suite: 162 passed. No extra
completed scripts task is claimed. **[Code-Observed]**

Follow-up with observed-script-v3: 20/21 real-engine cases pass. Full OCR now
recognizes all 13 pages and preserves English page 10; auto still fails native
coverage after recognition, with unchanged saved hierarchy. Loader design 18.5
records the new timings and the isolated page-1 exception-chain reproduction.
`test_pdf_ocr_quality_scripts.py` now has eight cases, including CJK whitespace
positives, mixed-script noise negatives and quality-version identity invalidation;
the full PDF suite passes 165 tests. The acceptance harness still exits nonzero
and PDF-FIX-03 remains open. No additional completed task is counted.

Auto fail-fast follow-up: four additions to `test_pdf_raw_recovery.py` cover
coverage failure stopping later pages, atomic retry, blank-render conflicts,
cached OCR revalidation and safe actionable error messages. The PDF suite now
passes 169 tests. A focused real-engine fast-then-auto run on the full 13-page
source preserves saved fast JSON and verifies 1 render/4 OCR calls before the
expected page-1 coverage failure. Loader design 18.6 records timing and scope.
This focused regression passes, but it is not a full matrix rerun or successful
Auto acceptance. No acceptance expectation was changed to count failure as success.

PDF-AUTO-01 adds `test_pdf_native_visibility.py` (eight cases). The full PDF suite
passes 177 tests with fontTools 4.67.0 installed/on PYTHONPATH. A separate poisoned-
subprocess Auto run through `accept_pdf_modes.accept` verifies the actual 13-page
source in temporary JSON with zero rendering/OCR, exact page spans and warm reuse.
Loader design 18.7 owns supported constructs, runtime/timing evidence and limits.
Only PDF-AUTO-01 is completed; full-matrix PDF-FIX-03 remains pending. No supporting
scripts task is counted separately.

Final PDF-FIX-03 acceptance recorded 2026-10-17 supersedes the intermediate pending
states above: `accept_pdf_modes.py --pdf <actual-source.pdf>` passes all 21 cases
using real render/OCR engines. The existing image was run read-only with network
disabled, a writable temporary filesystem, repository mounted read-only, and
fontTools 4.67.0 mounted at `/fonttools` with `PYTHONPATH=/fonttools`.
`DEEP_READER_STRUCTURED_STORAGE_BACKEND=file` selects disposable JSON storage;
language/profile remain stubs. All 177 `test_pdf_*.py` cases and 18 tests from
`test_prepare_lazy_pdf_layout`/`test_selective_ocr_acceptance` pass. Loader design
18.8 records source hash, versions, timings, process counts and limitations.
The unchanged harness verifies source preservation, atomic failures, ordered
spans and mode-specific warm reuse. No separate scripts task or deployment is
claimed; only the owning loader PDF-FIX-03 checklist item is closed.
