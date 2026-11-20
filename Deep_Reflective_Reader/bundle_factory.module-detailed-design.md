# bundle_factory.py Detailed Design

## FACTORY-02 validated reuse (2026-11-20, locally verified)

Every get_or_load lookup must validate the persisted publication at the requested
physical config using the exact supplied canonical text and current fingerprint
configuration, then attach the current valid existing profile. Cold and warm follow
the same load_validated boundary, including previous-snapshot journal reads. No
unversioned object hit is accepted as evidence; pending/invalid data cannot fall
back to memory. Missing canonical text yields explicit readiness even on a hit.

The conservative policy reloads on every lookup. The bounded LRU retains the latest
validated returned bundle per logical name for API compatibility; it is not a
read shortcut. Same-name roots replace that retained slot but never supply each
other's returned content. Capacity0/1, recency, invalidate and clear remain memory
only. Callers must not assume bundle object identity across successful lookups.
The cost is disk IO on warm reads, with zero parse/embedding/profile generation,
write-back or migration. load_prepared remains independent of this LRU.

ensure_index_ready first attempts the same coherent read (not unlocked active-file
fingerprint checks); only an index-readiness failure can enter FACTORY-01 guarded
candidate repair. Profile readiness errors do not trigger rebuilding. Acceptance
covers external Alpha/Beta/profile and cross-root reproductions, exact whitespace,
configuration mismatch, malformed/current profile, corrupt IDs, journal snapshots,
missing files/source evidence, cache lifecycle and zero-generation controls.


## FACTORY-01 guarded explicit publication (2026-11-19, locally verified)

All retained explicit index builds must use retrieval's staged, validated three-file
publication and journal boundary; never clear active files before work. Existing
profiles are reuse-only. Parse, embedding, profile admission, candidate save and
fingerprint failures retain the previous trio and profile; first-build failure is
not ready. Source validation runs before work, after expensive work and at both
publication checks. No model work is held inside the index publication lock.

The factory cannot derive lifecycle authority from a raw string. Retain public
ensure_index_ready, but require a server-owned AcceptedSourceGuard for any build.
Its namespace and exact canonical text must match the request/config; its live
checks fence raw replacement and current accepted incarnation/version. Missing
proof yields index_not_ready:explicit_prepare_required:source_guard_required before
parse/embed/write. Initial hierarchy/profile creation remains explicit preparation.
The existing guard is a read/revalidation boundary, not a new SQL/filesystem
transaction or an interlock with arbitrary external writers.

Unchanged valid reuse remains available without build authority. load_prepared
continues its independent validated, generation-free path. FACTORY-02 enforces the same publication/storage/profile admission on every reuse.
Acceptance: true file stores/FAISS/facade, old-trio hashes and validated reload,
normal replacement, first build, source/lifecycle changes, profile failures and
journal rollback/interruption. Preserve warm preparation and existing controls.


Local cumulative evidence (2026-11-20): `/tmp/factory-unit2-5e8e142/report.txt`.
Formal70/70 executions, zero failures/errors/skips: reuse11, guarded publication20,
preparation runtime24, exact index identity10 and profile lifecycle5. Real PostgreSQL
runs in disposable schemas. Adapted external probe6/6 retains all original product
assertions and explicit guard setup; before unit2 its three cache assertions failed.
Original baseline9 failures and the guard-admission fixture diff are preserved in
unit1 evidence. Prepared controls5/5 and four immutable-book file/SQL sample scenarios
pass, now including external publication/current profile and zero-generation reuse.
No live model, HTTP/TCP, UI or deployment acceptance. Exact-SHA external review pending.



## PREP runtime reuse-only handoff (2026-11-13)

Implemented supporting entry: load_prepared verifies persisted index/records/fingerprint
through retrieval's coherent read contract, then attaches the existing profile.
It does not extract, parse, embed, publish, or use an unversioned cached bundle.
Preparation/QA uses this path; existing explicit ensure_index_ready callers retain
separate build authority only with accepted-source guard evidence. Missing/corrupt assets report explicit readiness errors.

## Reuse-Only Profile Policy (2026-10-21)

CONTENT-03 supersedes the former missing/corrupt-profile rebuild fallback.
ensure_profile_ready now only loads the persisted profile or raises actionable
profile_not_ready/profile_unreadable. Index clearing/rebuilding cannot clear,
generate, save or enrich profiles, even with supplied raw text. Initial preparation
and explicit full reparse belong to document_preparation, not runtime bundles.

## 1. Module Purpose

`bundle_factory.py` 負責 FAISS runtime bundle 的建立、快取、重建與 profile/index artifact readiness 保證。 **[Code-Confirmed]**

## 2. Position in Overall Architecture

- Peripheral / Support Layer（retrieval bundle orchestration）

## 3. Key Files

| File | Responsibility | Notes |
|---|---|---|
| `bundle_factory.py` | `BundleFactory` | cache + ensure_profile_ready + ensure_index_ready **[Code-Confirmed]** |

## 4. Main Responsibilities

1. 管理 doc-scoped bundle cache（LRU-like OrderedDict）。 **[Code-Confirmed]**
2. Reuse validated assets; explicit guarded candidates repair missing/invalid indexes. **[Code-Confirmed]**
3. Load the existing profile or report readiness failure; never rebuild it. **[Code-Confirmed]**
4. Publish index/records/fingerprint through retrieval staging and journal; profile is read-only. **[Code-Confirmed]**

## 5. Non-Responsibilities

1. 不負責 API route 協調。 **[Code-Confirmed]**
2. 不負責 structured hierarchy parser。 **[Code-Confirmed]**

## 6. Important Data Structures / Contracts

- `BundleFactory`
- `FaissIndexBundle`
- `FaissStorageConfig`
- `DocumentProfile`

## 7. Module Relationships

- depends on: `retrieval/`, `profile/`, `fingerprint_handler.py`, `config/`; explicit build admission consumes the server-owned `document_preparation.accepted_source_guard.AcceptedSourceGuard` interface without invoking the preparation pipeline.
- used by: `bundle_provider.py`, `app/qa_coordinator.py`

## 8. Main Flows Involving This Module

1. ensure index/profile ready flow。 **[Code-Confirmed]**
2. cache load/reuse flow。 **[Code-Confirmed]**
3. Legacy records require explicit preparation or a guarded candidate; no raw-string-only rebuild. **[Code-Confirmed]**

## 9. Persistence / Side Effects

- read/write persistence: index/records/meta; profile is read-only.
- side effects：cache mutation、clear/remove artifacts

## 10. Known Legacy / Compatibility Behavior

1. Missing position metadata yields readiness unless the caller supplies accepted-source build authority. **[Code-Confirmed]**

## 11. Current Risks

1. risk：cache 命中與 artifact 狀態漂移
- why：可能讀到 stale bundle
- guardrail: FACTORY-02 reloads through load_validated on every lookup and attaches the current valid profile; the retained LRU is not read authority. Runtime load_prepared remains independent.

2. risk: missing/corrupt profile prevents a complete runtime bundle.
- why: downstream prompts require stable document metadata.
- guardrail: explicit readiness error; only initial preparation/full reparse owns recovery.

## 12. Open Questions for Maintainer

1. bundle cache eviction 是否需要更多可觀測 metrics？ **[Needs Confirmation]**

## 13. Suggested Next Documentation Improvements

1. 補 bundle readiness sequence diagram。
