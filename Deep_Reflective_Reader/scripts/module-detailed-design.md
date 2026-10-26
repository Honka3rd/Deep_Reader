# scripts Detailed Design

## Persisted Translation Acceptance Plan (2026-10-26)

**[Code-Confirmed]** TRUI-06 owns backend acceptance for the
[translation queue](../section_tasks/module-checklist.md#persisted-block-translation-queue-2026-10-26).
Existing test_translation_action/service and accept_translation_action_books prove
the earlier foundation; action tests now follow the persistent contract. New tests assert persistence after new
service instances/reload, same/different-language concurrency, replay after lost
response, rollback after SQL write and delayed generation after edit/delete/reparse.
Use real file and disposable PostgreSQL adapters, not mocks to certify atomicity.

Exercise all registry entries plus rejected arbitrary/unknown/subtag codes; both
profile-default and explicit source; completed/partial/untranslatable/provider and
commit failures. No-op and actual edit Keep/Clear must follow the full matrix:
same effective body preserves all state; changed text purges translations/summary
dependencies while retaining other kinds on Keep; deleted block purges its Notes
and all other interactions. Verify draft-only changes/Ignore/Keep-review, same-text
repair application, Undo/rename/reparse, hidden cleanup blocks and legacy-save races.
Document-profile source/summary remains unchanged; do not count regenerated profiles
as reuse or fallback reads as success.

Use immutable Madame Bovary/Red Roulette backups already in data/test-books, verify
hashes first/last, and mutate only disposable copies. Live model block calls must
produce at least two stored target languages and prove repeat selection reuses the
saved record without a model call, while failed generation of a missing language
preserves old results. Existing variant-code live scripts require alignment to registry-only
interactive selection without weakening generic-service language tests. Record private
reports without committing book text or credentials. UI TRUI-11 separately proves
actual browser flow, keyboard/focus, lifecycle races and 1280/390 screenshots.

Delivery evidence: test_translation_persistence and its PostgreSQL subclass cover
real storage reload, same/different-language and same-command concurrency, post-write
SQL rollback, unchanged document timestamps, source races, no-op/Keep/Clear, selective
summary cleanup, rename and legacy writers. Combined command runs 408 tests with
zero skips, including opt-in cleanup fixtures from data/test-books/cleanup-20261023.
accept_persisted_translation_books --live --browser calls the configured provider
against bounded immutable book excerpts, actual App/API and disposable SQL, with
no production mutation. Report/screenshots: data/test-books/persisted-translation/
20261026T070158Z-06e6f6e0. Profiles/backups are unchanged and schemas removed.
Frontend model/controller, refresh and pagination tests and production build pass;
literary accuracy and Docker deployment are not certified by these checks.

Reuse follow-up (2026-10-26): 80 focused tests pass on file/disposable SQL, including
new-ID/different-From reuse, no profile/model/write on hit, partial reuse and new
generation after actual body change. Mounted controller/DTO and build pass. Updated
real App/API/model/SQL acceptance at 1280/390 verifies zero generate requests on repeat
To selection, unchanged records, direct generate-API reuse and missing-language
generation: data/test-books/persisted-translation/20261026T113558Z-5d5c71b0.
Backups unchanged, schemas removed; no production deployment or book mutation.

## Offline Preflight Follow-up (2026-10-24)

Scripts owns WAIT-03 packaging/setup and WAIT-05
acceptance; token counting stays in llm and use-case timing stays in app.
[Canonical design](../llm/module-detailed-design.md#offline-preflight-follow-up-2026-10-24)
and [ordered tasks](../llm/module-checklist.md#offline-preflight-follow-up-2026-10-24).

WAIT-03 supplies a small explicit tokenizer-asset preparation/verification command
used by the root Dockerfile and local-dev setup. Reuse the model/encoding mapping
owned by WAIT-02, not a second hardcoded selection. Acquire only supported required
assets, verify known encoding hashes/version, fail clearly on acquisition/corruption,
and apply bounded network timeouts in this setup path. No private API key needed.
Install data in a reproducible runtime-readable location outside ephemeral `/tmp`,
without baking books, keys or host-specific paths into the image. Startup verifies
offline availability; it must not download during import/preflight or silently fix
missing files. Missing dev assets get an explicit setup action plus the provider's
documented unavailable/fallback behavior. Merely warming a running container is not
acceptance: recreated containers must work without the previous cache. Image build
and production service restart/deployment remain distinct operations.

### WAIT-03 Implementation (2026-10-24)

`scripts.prepare_tokenizer_assets` consumes the llm-owned manifest, deduplicates
the five supported models to their required encoding, and verifies exact size and
SHA-256. Valid existing assets require no network. Explicit acquisition uses HTTPS,
a 15-second socket timeout and a 180-second total-transfer check between reads
(an in-flight socket operation may add up to its timeout). Oversized, truncated or
wrong-hash content is rejected. Same-directory staging, fsync and atomic replacement
avoid publishing partial files; failed acquisition preserves the previous destination
and removes temporary files. Files are installed mode 0644. No credentials are used.

`--verify-only` never acquires or repairs assets: it checks the file and exercises
the actual offline counter for every supported model. Recipe, encoding hash and
installed tiktoken version are reported; no books, keys or raw network errors are
logged. Unsupported installed-library mappings or unavailable tokenizer dependencies
fail explicitly. Existing runtime counting still has its labelled conservative
fallback if an application deliberately bypasses the supported startup wrapper.

Root Dockerfile runs acquisition before copying the full application, installing
assets at `/opt/deep-reader/tokenizer-assets` through the existing
`DEEP_READER_TOKENIZER_ASSET_DIR` contract. This is image content, not a data-volume
or `/tmp` cache. The entrypoint `python -m scripts.tokenizer_startup` performs
offline verification before exec-ing the supplied command, preserving signals and
exit status. Missing/corrupt assets prevent API startup; no startup network repair.
`scripts/start_local_dev.sh` builds this image and waits for API health before
launching the UI. No DI/provider/domain policies or model limits change.

For direct host Python development, from the repository root:

```bash
bash scripts/prepare_local_tokenizer.sh
bash scripts/prepare_local_tokenizer.sh --verify-only
```

The helper uses `Deep_Reflective_Reader/.venv/bin/python` (or the explicit executable
path `DEEP_READER_PYTHON`). Dependencies must already be installed. Its default
asset directory is the counter's `llm/tokenizer_assets/`, ignored by git and Docker;
`--asset-dir` or the existing environment variable can select a different directory.
Use that same environment variable for subsequent runtime commands. From the
backend root, a direct API launch with the gate is:

```bash
.venv/bin/python -m scripts.tokenizer_startup .venv/bin/uvicorn main:app --host 127.0.0.1 --port 8000
```

Preparation is explicit setup; imports, preflight and normal startup cannot download.
Docker context excludes nested private env files, host venvs, books and local assets.
Existing dependency layers can be reused; tokenizer acquisition is independently
verified. Tests live in `test_tokenizer_asset_setup.py`; real asset and recreated-image
checks require `DEEP_READER_TEST_TOKENIZER_ASSETS` and
`DEEP_READER_TEST_TOKENIZER_IMAGE`. No production restart is part of this task.

Validation: 145 tests pass with actual local assets, immutable-book opt-ins and the
rebuilt image enabled, including 16 new packaging/setup/container cases. Root
Dockerfile build produced `deep-reader-tokenizer:wait03` (`62776af533ea`), acquiring
fresh verified encoding bytes while reusing unchanged dependency layers. Local
tiktoken 0.12.0 and image tiktoken 0.14.0 both pass actual offline counting/parity;
their distinct library versions remain in capacity identity. Two earlier container
downloads failed integrity checks and were rejected; an explicit build retry
succeeded. The initial 60-second local setup deadline was insufficient on this
network; the implemented 180-second setup-only deadline completed and passed.

New/recreated non-root containers run with `--network none`, read-only root and
empty tmpfs, and count all supported models without ephemeral cache. Missing/corrupt
asset containers refuse child execution. Actual default-entrypoint Uvicorn boots
offline and returns HTTP 200 `/health` with a dummy, noncredential key and no mounted
production data; the disposable container was removed. A separate image counter/
budget suite passes 20/21 (one private-book opt-in skipped, covered on host). Host
shell integration tests are not image tests because root development scripts are
deliberately not shipped in the API image. Unit faults cover truncation, wrong hash,
oversize, network/deadline/replace errors, cleanup, idempotence and private-safe
errors. Local-dev harness proves failed API startup does not launch UI. No model,
OCR/profile generation, production data mutation or production service restart.
Full incident-book HTTP timing and browser waiting remain WAIT-05.

Reproduce the scoped setup suite from the backend root after building the image:

```bash
PYTHONPATH=.:scripts DEEP_READER_TEST_TOKENIZER_ASSETS=llm/tokenizer_assets DEEP_READER_TEST_TOKENIZER_IMAGE=deep-reader-tokenizer:wait03 .venv/bin/python -m unittest test_tokenizer_asset_setup
```

### WAIT-05 Acceptance Contract

WAIT-05 extends existing private-book and browser harnesses, not a parallel test
platform. Before acceptance, preserve immutable raw PDF AND accepted OCR snapshot,
page/evidence/hierarchy/reading state needed to reproduce the full `暗水幽灵` case
(176196 characters at incident time), under `data/test-books/` with hash manifest.
Do not substitute `暗水幽灵_sample` or the smaller SCLEAN noise fixture for this
full-book incident. Reports contain hashes/counts/timings, never public full text.
Use backups and disposable storage; no production deletion, reparse, extraction,
cleanup commit or profile regeneration is authorized by this follow-up.

Required acceptance matrix:

- New image/container, empty ephemeral tokenizer cache, external network egress denied
  after image setup (allow the isolated database connection): first and warm full-book preflight complete using packaged data.
  Observe actual tokenizer and HTTP behavior, not a stubbed counter; repeat after
  container recreation. Prove zero tokenizer HTTP/completion/OCR/profile calls.
- Isolate asset-missing/corrupt/unsupported-model cases: deterministic unavailable
  or documented conservative fallback, never an implicit download or false ready.
  Cover concurrent initialization and capacity identity changes before execution.
- Record input/adaptation/initialization/count/overall timing, image/dependency and
  encoding identities. Initial test-environment gate: five fresh-process cold and
  five same-process warm full preflights each under 10 seconds; label hardware/load
  and observed distribution.
  This test-harness deadline is not a frontend/API timeout or a production SLA.
  The prior 0.304-second capacity probe is baseline evidence, not a promised bound.
- App at 1280/390: hold responses and advance virtual time beyond 120 seconds,
  then release success; independently verify lifecycle abort, late results, duplicate
  clicks, external failures, expired consent and refresh/reconciliation. No model
  calls/commits merely to test browser waiting; retain existing cleanup fault suites.

Keep byte-for-byte source/profile and semantic hierarchy/reading-state preservation
checks. This performance fix requires actual full-book preflight, not paid cleanup
generation or a changed semantic cleanup prompt. Existing SCLEAN live-model results
remain separate. Store scoped test results privately, report missing prerequisites
as Needs Confirmation, and never equate removing the UI timer with fixing cold start.

### WAIT-05 Delivery (2026-10-24)

`accept_wait_preflight.py` reuses `private_book_corpus`, the production cleanup
route/use case, accepted-source resolver, cleanup adapter and actual
OpenAILLMProvider token-count/capability methods. It does not generate a cleanup
plan or instantiate a credentialed SDK client. The immutable full incident-book
case is `data/test-books/7270c4d2c581f525`: 176196 characters, 261 page mappings,
221 actual OCR spans, original PDF/snapshot/hierarchy, and a separately hash-backed
`profile.json`. Existing manual edits and cleanup exclusions remain in reading state.
The existing corpus serializer needed one scoped fix: dataclass serialization alone
cannot encode typed Pydantic OCR evidence. Explicit `model_dump(mode="json")` now
round-trips it and validates restored context without weakening immutable hashes;
the six-case corpus suite covers the new typed-evidence round trip.

Each worker copies the accepted hierarchy/profile into disposable file storage,
reads the source only from the backup, and runs the actual Uvicorn HTTP route on
loopback. This deliberately uses a test-local lease, not the production SQL lease;
it measures the read/adaptation/counting incident path, not production database
latency or cross-process locking. Separate actual API/SQL tests cover that composition.
Forbidden completion, OCR/extraction, preparation, profile building/saving, cleanup
commit, repository saving and tokenizer registry/download methods are counted and
must remain zero. External socket connections are denied and counted. All source,
profile and hierarchy bytes and semantic reading state must be unchanged afterward.

Image `deep-reader-wait:wait05` (`9c0a3963e9ff`) reuses WAIT-03's verified assets and
the existing acceptance Dockerfile with current source. Five independently recreated
containers run with `--network none`, a read-only root and fresh empty `/tmp` cache.
The first performs one cold plus five warm requests; four more perform one cold
request each. The test asserts actual cold/memory tokenizer events, precise counting,
HTTP 200 readiness, one accepted-input read/count per request and zero forbidden calls.
Startup verification execs a new worker, so each first request initializes its own
counter; merely warming an earlier process cannot satisfy this gate.

Measured HTTP times (five samples each, milliseconds):

| State | Minimum | Median | Maximum |
| --- | --- | --- | --- |
| Cold | 413.7 | 543.4 | 664.6 |
| Warm | 125.2 | 136.3 | 380.2 |

All are below the 10000-ms test gate, not an API deadline or production SLA. The
Docker Linux/aarch64 worker uses Python 3.11/tiktoken 0.14.0; the macOS ARM host
uses Python 3.12.2/tiktoken 0.12.0. Reports record exact platform/CPU/encoding/image
identity and stage distributions plus correlated adapter/asset/init/count logs.
Cold initialization is 235.1-407.9 ms; warm counting is 18.8-23.9 ms. Full private
evidence is `data/test-books/7270c4d2c581f525/wait05-20261024/offline.json` and
`browser.json`, with copied regression logs and inspected 1280/390 screenshots.

The extended existing `test-cleanup-books.mjs` preflight-only mode renders the
backup navigation fixture and sends readiness to the actual local HTTP service.
At both viewports a response held past 120001 ms stays pending/cancellable, then
succeeds; closing a second held request prevents late reopening. No cleanup is
executed. This is real-book preflight acceptance, not a claim of production layout
API coverage. Five existing suites (cleanup, recovery, combined cleanup, enhancement
confirmation and enhancement outcomes) separately pass desktop/mobile long waits,
expiry, transport failures, duplicate submission, unmount/switch/late responses and
unknown-outcome reconciliation. They use controlled providers/disposable SQL or
explicit transport fixtures, not live models. The 36-case malformed transport suite
and `npm run check` pass; the existing large-bundle warning remains.

Final backend suite: 227 tests pass with no skips, with actual assets/image, private
book and disposable SQL opt-ins enabled. Missing/corrupt assets, concurrency,
capacity identity changes and rollback/reconciliation guards remain enforced.
An early harness assertion used `warm` where the counter logs `memory`; corrected
to the implemented event contract, then all ten container requests reran successfully.
The unrelated legacy prepare-cache failure documented in WAIT-04 remains outside
this suite; no parser assertion was relaxed. Test DB `wait05_acceptance_20261024`
and all disposable containers were removed. Original source/profile/hierarchy checks
remain unchanged. No original-book mutation, live LLM request, production restart
or acceptance-image deployment occurred.

Reproduce from the backend root after building the existing acceptance Dockerfile
against WAIT-03 (or a current asset-packaged image):

```bash
PYTHONPATH=.:scripts .venv/bin/python -m scripts.accept_wait_preflight \
  --case-id 7270c4d2c581f525 --image deep-reader-wait:wait05 \
  --report data/test-books/7270c4d2c581f525/wait05-repeat/offline.json
```

For the host browser mode use `--browser` instead of `--image`, set an empty
`TIKTOKEN_CACHE_DIR`, and supply `UI_URL`, `PLAYWRIGHT_MODULE` and `CHROMIUM_PATH`
as required by the existing browser harness. Reports must remain under the private
corpus. No case can silently substitute a sample or a different accepted-text length.

Governance/validation record: `module-memory-loader` ran in module-specific
implementation mode with scripts as owner and app/llm/UI as supporting modules.
Inspected `AGENTS.md`, both `docs/standards` development standards, proposal,
high-level-design, module index, progress, and scripts/app/llm/UI design/checklist
pairs. That loading phase modified no files and found no conflicting authority.
Updated scripts design/checklist for implementation evidence and completed WAIT-05;
llm design/checklist for the accepted shared track; UI design for supporting browser
scope; progress for backend-only 4/4 and scripts 63 actual completed items (excluding
the literal checklist template). No other markdown is changed by WAIT-05.
`checklist-progress-sync` changed only the relevant checklist/progress fields during
its synchronization phase. `implementation-validation` is read-only: source/test
scope (one new acceptance script, corpus helper/regression and existing UI harness),
architecture, module boundaries, memory, timestamps, tests and diff checks pass.
No files were removed; unrelated existing worktree edits were retained. Runtime
API/domain/DB semantics are unchanged. The documented legacy regression/build
warning and lack of production deployment remain explicit limitations.

## Independent Cleanup Acceptance Plan (2026-10-23)

**[Code-Confirmed 2026-10-24]** SCLEAN-10 supplies bounded final acceptance of the
[new capability](../document_structure/module-detailed-design.md#independent-smart-cleanup-2026-10-23).
Extend existing backup-only helpers and private reports; do not rebuild a parallel
test platform. Hash-check original backup and extracted canonical/page artifacts
before/after. Use disposable file/SQL storage and explicit opt-in live credentials;
no Library deletion, profile rebuild or production reparse is authorized by this plan.

Required matrix: real Common -> standalone cleanup (splitter poisoned to fail if
called); enhance split -> same cleanup engine before one commit; deliberately failed
automatic cleanup -> fresh standalone retry against the resulting stored hierarchy;
second cleanup sees only remaining eligible text; raw/canonical/page/profile hashes
and standalone hierarchy/unit/block IDs unchanged. Compare all five target content
projections and generated-output invalidation versus saved notes/manual edits.
Fault cases include transport failure, malformed/unsafe plan, capacity zero-call,
native/unknown provenance, commit rollback, source/edit/reparse race, duplicate
request/lost acknowledgement/restart, stale consent, layout-only recovery and Undo.
Task-local tests accompany SCLEAN-01..09; the final task does not defer their coverage.

Reuse immutable Sample/Sample2/dark-water backups under data/test-books, including
the known mixed-readable/garbled fragment as a false-deletion sentinel. Real model
runs must include at least one reviewed nonempty safe deletion; all skipped is not
positive acceptance. Manually inspect every applied span against original pages,
record retained garbage and reject meaningful-text loss rather than weaken guards.
Madame Bovary TXT/Red-book Fast backups are zero-cleanup-call controls, not proof
of another full-book generation run. Preserve failed attempt evidence privately.

Actual frontend/API at 1280/390 must verify independent entry/retry, collapsible
audit, review queue, reading after refresh and Undo with screenshots. Record code/
prompt/policy/model versions, stage latency, plan/applied/rejected counts and reasons,
provider/splitter/OCR/profile call counts, one-commit evidence and hash checks.
Smoke the proposed routes against an isolated image built from current code so an
old Docker image cannot be mistaken for acceptance of new source. Production
rebuild/restart is a separate explicitly authorized operation. Never claim existing
CLEAN-10 combined-flow passes demonstrate the standalone entry without separate evidence.

### SCLEAN-10 Delivery (2026-10-24)

`accept_ocr_cleanup_books.py` now accepts `--entry standalone|combined|retry` while
retaining backup-only extraction/control helpers. Standalone uses actual runtime
composition, FastAPI and disposable PostgreSQL, poisons the structural builder,
executes two fresh consents and checks replay/restart without another commit.
It verifies prior-deletion gaps do not reenter the cleanup input, stable hierarchy/
unit/block IDs, exact canonical/page preservation and five-level effective content.
`retry` performs a real split, injects one automatic cleanup failure, commits the
usable split and then invokes the same independent service with live cleanup output.
No second split occurs. Combined cleanup still has one final enhancement commit.

The existing real-book browser helper now restores standalone handles, checks
current results, obtains fresh consent and actually executes one desktop cleanup;
mobile verifies recovery, review, collapsed audit, reload and Undo. Every fresh
operation has one receipt commit, including no-op receipts; GET/replay adds none.
Private request/response/result/deletion evidence includes stage latency, model,
prompt/policy versions and runtime-source hashes. Preparation/profile entry points
are poisoned; no extraction/profile regeneration or production Library writes occur.
The fixture documents have no persisted profile: actual profile-row preservation
is covered by SQL regressions, not claimed as a live-book profile hash comparison.

Final private evidence: `data/test-books/cleanup-20261023/sclean10-review-20261024.json`.
All seven runs use gpt-4.1-mini, structure prompt v17 and cleanup prompt v3:

| Case / entry | Final private attempt | Applied result |
| --- | --- | --- |
| Sample / standalone | sample/20261024T051716227798Z-standalone | No safe deletion; three fresh calls including browser execution |
| Sample2 / standalone | sample2/20261024T051647129042Z-standalone | One deletion; subsequent API/browser cleanup applies none |
| Dark-water / standalone | noise/20261024T051736708300Z-standalone | No safe deletion; three fresh calls including browser execution |
| Sample / combined | sample/20261024T051315612333Z-combined | Safe rejection, one final commit |
| Sample2 / combined | sample2/20261024T051250640995Z-combined | One reviewed deletion, one final commit |
| Dark-water / combined | noise/20261024T051346020445Z-combined | Safe rejection, one final commit |
| Sample2 / failed auto then standalone | sample2/20261024T051759848215Z-retry | Usable split preserved, independent retry removes one span without resplitting |

Every applied span was manually checked against Sample2 original page 1: offsets
[386,389) contain corrupt recognition at the printed SARS acronym; neighboring
readable Chinese remains. The same span is removed once per isolated flow and Undo
restores the previous projection. Supplemental final Sample2 run
`sample2/20261024T052734641917Z-standalone` also passes an explicit equality assertion
against the deletion state before the latest applied cleanup, not merely an Undo
success response. Its only applied span is the same reviewed range. This does not
reconstruct the missing acronym.
Sample's mixed-readable sentinel and dark-water noise are retained; quote_not_found
and unsafe_text_boundary reject proposals rather than weakening guards. No observed
meaningful-text loss in these cases is not a general semantic guarantee. Final live
review queues are empty; unresolved/textless regions and Ignore use controlled tests.
Original backup/document hashes and page maps remain unchanged. TXT Madame Bovary
(652001 chars) and Fast PDF Red-book (147401 chars) remain zero-cleanup-call controls,
not fresh full-book generation acceptance. Standalone first-call provider times were
3.485 / 5.341 / 5.291 seconds for Sample / Sample2 / dark-water; all timings remain
per-stage evidence, not a performance SLA. Structural planning may retry validation;
the reports retain those calls without treating them as repeated cleanup/commit.

`scripts/Dockerfile.cleanup-acceptance` reuses an explicitly selected local dependency
image and copies current source without private books or keys. Build from repository
root with:

```bash
docker build --pull=false --build-arg BASE_IMAGE=deep_reader-api \
  -f Deep_Reflective_Reader/scripts/Dockerfile.cleanup-acceptance \
  -t deep-reader-cleanup-acceptance:sclean10 .
```

Run only with a disposable test DSN and a
dummy OPENAI_API_KEY; the image's default suite uses deterministic providers.
Isolated image 83cead1c68d3 passed 39/41 tests with two private-source opt-in skips;
separate actual Uvicorn HTTP smoke on temporary port 18001 verified the three
routes and unavailable-storage 503 without consent. The production container was
not restarted or rebuilt. An initial image run exposed test-state pollution:
`patch.object(State, ..., create=True)` deleted the original dynamic attribute on
exit. Removing create=True and asserting identity restoration fixes test isolation,
not runtime cleanup behavior. Another initial run lacked the dummy import-time key;
this was a harness prerequisite, not a provider request failure.

Regression: 387 tests pass with the private-source opt-in enabled, including
file/SQL atomicity, notes/profile preservation, manual edits, capacity/provenance,
source races and recovery. The image suite overlaps these tests and is not added
to that count. Type/build and 36 malformed-adapter response checks pass; the existing
bundle warning remains. Real-book screenshots at 1280/390 were inspected, including
final Sample2 desktop execution and mobile audit, without modal clipping. Detailed
browser fault regression evidence is recorded in the owning checklist.

Governance receipt: inspected AGENTS.md, proposal.md, high-level-design.md, module
index, progress.md, both development standards and scripts/document_structure/UI
design-checklist pairs. Modified markdown: this scripts design (reproduction,
evidence and limitations), scripts checklist (SCLEAN-10 evidence/date), shared
structure design/checklist (final acceptance pointer/10-of-10 index), backend
progress (8-of-8 backend-only aggregate), UI design/checklist (shared completion
reference, no new frontend task). No API/domain/runtime policy was changed; no
duplicate cleanup engine was introduced. Validation is read-only after these edits.

## Optional Cleanup Real-Book Acceptance (2026-10-23)

Supporting CLEAN-10, counted once in document_structure/module-checklist.md.
`accept_ocr_cleanup_books.py` separates immutable backup/extraction from enhancement:
`--backup-noise <source>` preserves the reported optional fixture; `--extract` runs
real OCR into hash-bound artifacts; `--controls` checks TXT/Fast cleanup exclusion;
`--case sample|sample2|noise --allow-live-model --browser` requires an explicit
DEEP_READER_TEST_POSTGRES_DSN and uses isolated SQL, actual planners/provider/API
and the production layout resolver. No production document name is exposed to the
temporary browser API. UI_URL selects a separate Vite instance; PLAYWRIGHT_MODULE
and CHROMIUM_PATH may point to installed browser tooling. Temporary SQL schemas,
API server and browser are closed by the harness. Start/stop Vite separately.

Run with PYTHONPATH=Deep_Reflective_Reader:Deep_Reflective_Reader/scripts and the
project venv from repository root; inject secrets with scripts/load_local_env.sh.
Use the existing OCR image with network disabled for extraction when the host has
no renderer/OCR. The repository source mount is read-only except the private corpus
output directory. Never overwrite an immutable backup to make a test pass.
Private reports under data/test-books/cleanup-20261023 include source/document
hashes, full model requests/responses, result/deletion records and 1280/390 screenshots.
`accepted` proves automated assertions only; CLEAN-10 completion additionally
requires manual review of every final applied deletion and at least one safe nonempty
case. See the structure design for exact attempts, timing, failures and limitations.

private_book_corpus.load_book now reconstructs typed cleanup evidence from the
hash-verified document for legacy snapshots without cleanup_context. All preexisting
identity/structure comparisons remain; stored non-null context must exactly match
the reconstructed value. No fixture rewrite or production source access is allowed.
Regression fixtures cover missing legacy context, immutable backup survival and
corruption rejection. `test_cleanup_identity.py` covers the actual SQL block-ID bug
discovered by live cleanup, rather than relying on JSON roundtrips that omit blocks.

### CLEAN-10 Governance Receipt

Inspected: AGENTS.md, proposal.md, high-level-design.md, docs/modules/index.md,
progress.md; design/checklist pairs for document_structure, scripts, shared,
section_tasks, db and Deep_Reader_UI. Scope is the next existing CLEAN-10 task,
including fixes found during its acceptance, not a new product backlog.

Markdown modified in this task:

| File (backend paths relative to Deep_Reflective_Reader) | Reason and evidence |
| --- | --- |
| document_structure/module-detailed-design.md | Final live attempts, prompt v2, semantic limitations, SQL identity defect and exact validation reasons. |
| document_structure/module-checklist.md | Close CLEAN-10 against reviewed final evidence; date 2026-10-23. |
| shared/module-detailed-design.md | Pure generated identity rebinding contract verified by focused tests. |
| section_tasks/module-detailed-design.md | Both explicit normalization paths use that contract, preserving hierarchy authority. |
| db/module-detailed-design.md | Generated block IDs follow allocated SQL unit IDs in the same transaction. |
| scripts/module-detailed-design.md | Reproduction, private immutable corpus, scope and this governance receipt. |
| ../Deep_Reader_UI/module-detailed-design.md | Actual-book 1280/390 reading/audit/Undo evidence, without new runtime UI semantics. |
| ../Deep_Reader_UI/module-checklist.md | Refresh shared-track reference only; UI remains 2/2 and is not counted again. |
| progress.md | Backend-only CLEAN aggregate 7/8 to 8/8; no UI aggregation. |

Validation evidence: 288 unittest cases, 287 pass/one inherited opt-in skip; 28
focused cases overlap that suite except prompt coverage, so counts are not summed.
Shared block (19 assertions), normalizer and hierarchy resolver standalone scripts
pass. The old normalizer fixture was updated from obsolete root sections to chapters;
its optional old Library snapshot is absent and is NOT a real-book pass. The separate
immutable live corpus supplies real-book acceptance. Both controlled consent/flow
and audit/review browser suites pass at 1280/390, as do all three live-book browser
runs. npm run check passes with the existing large-bundle warning. Private
manual-review.json distinguishes rejected development attempts from final outcomes.
No files removed, source/profile changes to production, Git commit or deployment.

Implementation-validation scope: document_structure with supporting shared,
section_tasks, db, scripts and UI acceptance. Architecture/boundary checks retain
hierarchy-first authority, explicit writes, immutable canonical source and no
profile regeneration. Overall result is WARNING for bounded model-semantic
accuracy and retained noise, not failed acceptance assertions. Governance sync
changes only checklist/progress; implementation edits occurred outside that skill.
Final read-only validation modifies no files; unrelated preexisting work is retained.

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
