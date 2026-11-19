# retrieval Detailed Design

## RETRIEVAL-03 ordinary output reserve (2026-11-19, locally verified)

Cold builder and persisted store reload obtain the ordinary completion output
allowance from the injected provider's public get_default_output_budget contract.
Pass it separately from configured targets to the shared budget resolver. Both
paths reserve exactly that allowance, with existing input/context caps and ratios;
QA deducts output once downstream. No re-embedding or stored-format migration.
Providers without a declared allowance retain the existing configured-target
fallback; malformed declarations fail explicitly, never silently under-reserve.

Acceptance: near-limit red/green and cold/warm consistency, no automatic retry or
smaller provider allowance; bounded real-input copies and scoped regressions.
Cumulative external review required before deployment/push.


Local evidence: `/tmp/retrieval-unit3-51813cd/report.txt`. Original near-limit
reproduction 3 methods /1 failure -> 3/3 unchanged; configured reserve 500 -> 1638,
actual dispatched output remains 1638. Expanded Responses boundary exposed a
newline-count underestimate; final prompt admission now rejects that candidate
before answer dispatch/session update. New suite 10/10, cumulative 15 suites
188/188 executions, zero skips, including real PostgreSQL and verified offline
encoding. Two full immutable-book copies retain hashes; 8 bounded cold/warm QA
flows pass for both endpoints. Groups overlap earlier evidence, not unique cases.
3200 combined tokens cannot accommodate the 3276 Responses default output allowance;
this setting now fails safely, without changing configured caps or provider output.
No live models, semantic/full-book QA, full server/TCP, UI or deployment acceptance.
All three retrieval units locally verified; cumulative exact-SHA external review
pending. No deployment, push or next-module work before that review.

## RETRIEVAL-02 search diagnostic privacy (2026-11-19, locally verified)

The unconditional SearchMetadata repr dump is removed from FaissIndexBundle.search.
Search remains a data-returning operation: full text/source/chapter/position,
ranking, scores, top_k/deduplication and query embedding counts are unchanged.
Do not redact returned hits or alter context to suppress logs. Existing bounded
context/stage diagnostics remain; no new per-hit/content diagnostic is required.
Regression captures stdout, stderr and actual configure_logging file sinks at
INFO, ERROR and DEBUG for direct search and the production ask function in isolated ASGI.
Assert synthetic body/query/source markers absent from diagnostics while context
and prompt retain full hits, answer remains unchanged, embedding/answer calls stay
fixed. Retain empty results, sparse-ID ranking, neighbor and dedupe controls.
Use temporary sinks, restore logging/environment, and no network/paid provider.
This is retrieval-owned output removal, not a global logger or HTTP error refactor.
At the unit 2 checkpoint, RETRIEVAL-01 remained intact and RETRIEVAL-03 awaited confirmation.

Local evidence: `/tmp/retrieval-unit2-51813cd/report.txt`. Production change is
removal of the search-result print only. External privacy 2 methods /4 failed
subassertions -> 2/2; original 9 search controls remain green. New formal suite
13 methods /10 failed subassertions -> 13/13, covering INFO/ERROR/DEBUG, all three
sinks and unchanged body/query/source handoff. Scoped regression 74/74, zero skips:
privacy 13, ID integrity file/real PostgreSQL 24, context capacity 9, question output
15 and embedding logging 13. Counts overlap external and focused groups.
Two full hash-verified immutable-book copies supply bounded English/Chinese samples
for 8 direct/ASGI QA flows. Full returned/context/prompt text preserved; diagnostic
sinks exclude sample/query/source markers. Controlled providers/vectors; no live
model, semantic QA, full server lifespan/TCP, UI or Docker acceptance.
Unit 2 checkpoint: RETRIEVAL-03 awaited developer confirmation. No commit,
deployment, push or next-module work; latest cumulative SHA needs external review.

## RETRIEVAL-01 index/record identity (2026-11-19, locally verified)

One retrieval-owned admission boundary validates the persisted FAISS ID map,
JSON record keys and each record's integer node_id as the same unique set, with
key/node_id equality per row. Non-contiguous IDs remain valid; count/dimension,
exact canonical fingerprint and slice checks remain independent requirements.
Reject duplicate JSON keys and integer-key aliases before dictionary construction
can hide collisions. Unsupported ID representation fails closed; no inferred ID
renumbering or read-time repair. Validated active reads, journal previous snapshots
and staged candidate reload use this same gate. Invalid runtime data reports
explicit_prepare readiness; explicit Prepare may rebuild. Invalid candidates must
leave active index/records/meta bytes and journal state unchanged.

Regression: real FAISS/store/pipeline, shifted keys, swapped rows, duplicate FAISS
IDs/record IDs/JSON keys, alias keys, noninteger IDs, valid non-contiguous IDs,
interrupted publication previous snapshots, explicit recovery and zero warm work.
Retain source/fingerprint/coordinate guards, real publication rollback tests and
old trio hashes. Controlled embeddings are protocol evidence, not semantic quality.
At the unit 1 checkpoint, logging and output budgeting remained separate later units.
No deployment/push/next module; final cumulative SHA requires external review.

Local evidence: `/tmp/retrieval-unit1-51813cd/report.txt`. Original external
5 methods / 4 failed subassertions -> 5/5; new file matrix 12 methods /17 failed
subassertions -> file/real PostgreSQL 24/24. Cumulative selected regression 78/78,
zero skips, including PREP runtime 24, exact identity 10, context capacity 9 and
completion admission 11. Groups overlap; external flock control is separate.
Two full immutable-book copies (652001/147401 canonical chars; 700/147 chunks)
preserve valid warm zero-generation, reject swapped records without writes, then
recover with exactly one explicit rebuild. Hashes preserved; deterministic vectors
are not semantic QA. Unit 1 locally verified; units 2/3 and external review pending.
No commit/deploy/push or next-module work at this unit gate.


## LANG-01/02 detector boundary (2026-11-17, locally verified)

NodeProvider rejects missing detector evidence before invoking the chunk parser or downstream embeddings/publication. It raises a fixed document_language_not_ready ValueError without source/output content. Valid canonical detector values and persisted language reuse retain current behavior. No runtime generation, index migration or relaxation of staged publication.

Verification: New boundary suite 13/13 (real file/PostgreSQL); external detector 4/4 and normalization 3/3; adjacent/focused total 207/207, zero skips after documented runner cwd correction. Two full hash-verified book copies across file/SQL retain accepted assets under three invalid rebuild replies. `/tmp/lang-unit1-847f0d6/report.txt`. Unit 1 evidence retained; LANG-03 is separately verified in language/section_tasks. External language approval pending.


## PREP-05 coordinate-sensitive validation (2026-11-13)

Implemented supporting correction: validated loads check versioned exact canonical
fingerprints and every record's positional fields, integer bounded offsets and
canonical slice. NodeRecord stores stripped chunk text, so slice comparison uses
that existing chunk normalization; the whole-document fingerprint never strips.
Whitespace between chunks and parser overlap remain legal. Invalid evidence is
index_not_ready:explicit_prepare_required, never read-time repair or generation.
Validation applies equally to staged candidates and the complete previous snapshot
selected by a pending journal. A previous snapshot is usable only for its own exact
canonical text/configuration; no mixture with active metadata is permitted.

Validation: `scripts/test_preparation_index_identity.py` passes 10 test methods
(66 matrix subcases across file/real PostgreSQL), with actual SentenceSplitter,
FAISS and deterministic embeddings. External reproducer changes from two failures
to zero; PREP-01–04 regressions remain green. Two immutable-backup book copies
validate 1527/298 exact record slices and zero warm generation/write-back.
Evidence: `/tmp/prep05-fix-a8c0ca1/`; external approval is pending. No deployment,
remote push, paid model calls or next-module audit is included.

## PREP-04 staged index publication (2026-11-13)

Implemented supporting contract: prepare index/records/fingerprint together in isolated
candidate storage, validate reload and fingerprint, then revalidate source before
publication. Never clear the active files before parsing/embedding. A per-index
advisory lock coordinates publication and validated reads. Persist an fsynced
rollback journal and complete previous-file copies before replacing active files;
remove the journal only after all three new files and directory entries are durable.
Exceptions roll back; process interruption leaves durable previous-state evidence.
The lock is held on the existing directory inode, so reads create no lock files.
Read-only loads use the previous complete snapshot while a pending journal exists;
explicit recovery/publication restores it. Reads do not mkdir, rename or repair.
Initial failure has no ready index; replacement failure keeps the old trio reloadable.
This is a local three-file publication boundary, not a SQL/profile transaction or
an interlock with arbitrary external writers. Production retrieval policy is unchanged.

## CONFIG-02 explicit write ownership (2026-11-12)

FaissIndexStore.save creates the destination parent before index/record writes.
Config construction and index reads do not create or rename directories. Production
config factories use the active repository's source identity, preserving PDF/TXT
isolation without changing the index/records schema or embedding behavior.

## FULL-01 Capacity Logs (2026-10-20)

Builder/store budget logs distinguish shared context_window from optional max_input;
the config resolver retains existing combined QA budget semantics. No index schema,
embedding, persistence or retrieval behavior changed. Evidence: llm FULL-01 tests
and config/container regressions; no additional checklist completion counted here.

## 1. Module Purpose

`retrieval/` 管理節點解析後的向量索引建置、持久化、重載與檢索，形成 `FaissIndexBundle` 供 QA 與 context 模組使用。 **[Code-Confirmed]**

## 2. Position in Overall Architecture

- Document Preparation Layer + Runtime Retrieval Core

## 3. Key Files

| File | Responsibility | Notes |
|---|---|---|
| `retrieval/node_provider.py` | 原文切 chunk + 位置 metadata 附加 | 產出 `ParsedDocument` **[Code-Confirmed]** |
| `retrieval/parsed_document.py` | parsed document DTO | nodes + language **[Code-Confirmed]** |
| `retrieval/faiss_index_builder.py` | 從 parsed nodes 建 FAISS bundle | embedding batch + token budget resolve **[Code-Confirmed]** |
| `retrieval/faiss_index_store.py` | FAISS index/records save/load/clear | records schema 檢查 **[Code-Confirmed]** |
| `retrieval/faiss_index_bundle.py` | runtime query bundle | search + profile attachment **[Code-Confirmed]** |
| `retrieval/node_record.py` | persisted node record contract | chunk/char/neighbor metadata **[Code-Confirmed]** |
| `retrieval/search_metadata.py` | retrieval hit DTO | score/source/chapter/position **[Code-Confirmed]** |

## 4. Main Responsibilities

1. 把 raw text 轉換為可檢索節點與位置 metadata。 **[Code-Confirmed]**
2. 建立與保存 FAISS index + records。 **[Code-Confirmed]**
3. 重載 index 並提供 query search contract。 **[Code-Confirmed]**
4. 將 profile 附掛在 runtime bundle。 **[Code-Confirmed]**

## 5. Non-Responsibilities

1. 不負責 structured hierarchy parser。 **[Code-Confirmed]**
2. 不負責 summary/quiz artifact write。 **[Code-Confirmed]**
3. 不負責 API mapping。 **[Code-Confirmed]**

## 6. Important Data Structures / Contracts

- `ParsedDocument`
- `NodeRecord`
- `FaissIndexBundle`
- `FaissIndexBuilder`
- `FaissIndexStore`
- `SearchMetadata`

## 7. Module Relationships

- depends on: `doc_loaders/`, `language/`, `embeddings/`, `llm/`, `config/`
- used by: `bundle_factory.py`, `bundle_provider.py`, `context/`, `app/qa_coordinator.py`

## 8. Main Flows Involving This Module

1. prepare/index flow：node parse -> build -> save。 **[Code-Confirmed]**
2. runtime load flow：load index/records -> bundle search。 **[Code-Confirmed]**
3. query flow：embed query -> FAISS search -> SearchMetadata。 **[Code-Confirmed]**

## 9. Persistence / Side Effects

- read persistence：是（index.faiss + records.json）
- write persistence：是
- mutate structured document：否
- call LLM：間接（透過 language detector capability path）

## 10. Known Legacy / Compatibility Behavior

1. `FaissIndexStore.has_position_metadata` 偵測 legacy records schema；普通讀取回報 readiness，只有明確 preparation 可重建。 **[Code-Confirmed]**
2. `NodeRecord.position()` 保留 legacy metadata accessor。 **[Code-Confirmed]**

## 11. Current Risks

1. risk：records schema drift
- why：load path 可能失敗或 silently 缺欄
- guardrail：unique FAISS/map/record identity + canonical fingerprint/slice validation；explicit preparation rebuild only

2. risk：embedding/token budgets 與模型能力不一致
- why：context/quality 成本波動
- guardrail：token budget resolver + capability logs

3. risk：node text fallback 定位偏移
- why：char offsets 可能不精準
- guardrail：保留 metadata 欄位與 downstream容錯

## 12. Open Questions for Maintainer

1. records schema 是否需要版本欄位（目前靠欄位存在判斷）？ **[Needs Confirmation]**

## 13. Suggested Next Documentation Improvements

1. 補 index artifact schema 文件（index + records + meta + profile）。
