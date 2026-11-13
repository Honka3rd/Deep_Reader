# db Detailed Design

## PROFILE audit integration (2026-11-13)

PROFILE-01 supporting port: expose a short read-only profile publication snapshot on the
real owner SQL connection. A SHARE table lock fences lifecycle updates and initial-row
insertion (including absence); load the exact namespace/name on that connection. No LLM
work or SQL writes occur inside this boundary, and filesystem profile replacement is not
a SQL transaction or rollback claim. Existing source/incarnation fences remain
unchanged.

Validation: profile persistence 32/32 file/real SQL methods, original six assertions
6/6, PREP runtime 24/24, exact-index 10/10 and 142 related regressions pass.
Atomic artifact 18/18 and generated-source 23/23 preserve existing fences; config
35/35 passes with valid profile fixtures. No new module audit or deployment.
Evidence: `/tmp/profile-fix-fc1dd285/final-report.txt`; external review pending.

## STRUCT-04 generation fences (2026-11-13, implemented)

Dedicated artifact callbacks validate generation bindings against the current row
inside STRUCT-01's transaction. Rename advances artifact_generation_epoch in both
metadata and snapshot payload in its existing transaction, after checking transform
source invariants. Historical source/learning epochs and receipts remain intact.
No schema migration or read repair is needed.

Validation: the 23-method generation suite, 18 atomic cases, 43 lifecycle cases and
120 HTTP rename cases pass across the applicable file/real SQL owners. Existing
535-test and PREP/manual support results are recorded in the document_structure
design and `/tmp/struct-fix-2c897af/`. No schema migration, read-time identity
backfill, deployment or push; latest review SHA still needs external approval.

## STRUCT-03 conditional manual replacement (2026-11-13)

Implemented correction: the manual repository port locks the current document row and
loads, validates, replaces and verifies readback on the same owner connection.
Source/version/incarnation checks run before replacement and again before commit;
failure rolls back hierarchy, metadata and derived rows. Replacement remains a hard
reparse with a new structure version/epoch; source evidence is selectively retained,
not old interactions. The file adapter uses its corresponding per-document lock.
Validation: eight real PostgreSQL integrity methods (28 matrix subcases), the two original SQL manual reproductions and existing duplicate-chapter-order/repository boundary regressions pass. Failures after actual SQL replacement roll back the old hierarchy; source changes before commit also roll back. Evidence: `/tmp/struct-fix-2c897af/`; no schema migration. Whole-audit external review is pending.


## STRUCT-01 transactional artifact updates (2026-11-13)

Implemented supporting store port: acquire documents ROW EXCLUSIVE then target FOR UPDATE,
load the latest hierarchy using that connection, apply the domain mutation and invoke
save(replace_existing_hierarchy=False) with the same connection. Reload the returned
persisted hierarchy before commit so SQL-assigned IDs agree with storage. Any callback,
save or readback failure rolls back; no filesystem lock or second write connection.
Existing incarnation/version, annotation preservation and source-transition guards
remain active. Generation runs before this transaction, never inside it.

Validation: `scripts/test_atomic_hierarchy_artifacts.py` passes 18 file/real
PostgreSQL cases. The same paused-read concurrency assertions fail on both baseline
adapters and pass after the correction. 29 existing suites execute 535 tests:
529 pass, zero failures/errors, six private-source opt-in skips. The two task-layout
fixtures now explicitly prepare before reading and use the current resolver version;
all original hierarchy/artifact/cache assertions remain. The legacy summary/quiz cache scripts originally failed on both baseline and
unit-1 code due to root-sections-only fixtures. STRUCT-04 modernizes those fixtures
and current merge semantics, preserving all original assertions; both now pass. Evidence: `/tmp/struct-fix-2c897af/`.
This completes only the local STRUCT-01 unit, not external approval of STRUCT-01–04.


## CONFIG-01 qualified identities (2026-11-12)

SQL document names retain source suffixes. The active repository resolves legacy
aliases from names in its own namespace plus matching raw-source candidates;
residual structured JSON is irrelevant. Target construction remains pure. New
imports write their explicit identities; ordinary snapshots saved through an alias
must resolve to that snapshot's identity, retaining all existing epoch checks.
No table migration or read-time data repair is introduced.

## Ordinary save incarnation fence (2026-11-11, implemented)

The shared annotation-preservation function now validates incoming/current document incarnation before replacing provenance, within the existing SQL write transaction and locks. Explicit reparse retains its new-epoch behavior. Real PostgreSQL HTTP regressions exercise rename, delete/reimport, delayed old commands and ordinary stale saves; SQL-only documents still require no local files. No schema migration or inactive-backend mutation is introduced.

## MAIN-02 repository identity (2026-11-11, implemented)

Ordinary metadata/artifact saves carrying an existing document epoch must reject
a now-missing identity instead of recreating a renamed/deleted document. Explicit
reparse/import remains the creation path, preserving the R3 reimport contract.

The SQL artifact adapter exposes a hashed storage identity from its configured
DSN/namespace, separate from inherited file paths. Explicit rename recovery has no
file lifecycle to replay; the application reconciles canonical document epochs
after the existing SQL transaction. No migration or SQL transaction change.

## Shared sentence translation (2026-11-09)

No new tables or migration: ordinary and learning translation use the same
learning_sentences/cache_entries/operations/commands/checkpoints storage. A bounded
block/type/cache-cursor selection loads only complete translation keys and their
required current/saved operation evidence; it excludes vocabulary and unrelated
book history. Existing transaction scope and preference→document lock order remain.
Ordinary translation-purpose binding does not require a native-language preference;
source validation and dispatch/commit fences still run within the existing transaction.
There is no additional aggregate paragraph row. This removes cross-mode duplication,
not the existing durable checkpoint/operation evidence needed for recovery.

Validation: 470 backend tests and actual API/React mode-switch acceptance pass; detailed evidence and limits are recorded in the shared section_tasks design.


## Local reading-mode schema deployment correction (2026-10-29)

The local API was running the learning routes against a database containing only
001..003. All seven tables from 004/005 were absent; reading workspace preferences
therefore returned `preference_storage_unavailable` (503). Missing tables remain
a storage failure, not empty settings or a read-time schema repair.

The existing database was backed up with `pg_dump -Fc`, restored and verified in
a disposable PostgreSQL instance before applying the unchanged 004/005 SQL in
one explicit transaction. The local `Madame Bovary` preference read, workspace
read and learning read now return 200 without restarting the API or generating
content. Counts and content fingerprints of all 12 pre-existing tables match;
the seven added tables remain empty after reads. No book/profile/settings content
was rewritten. Private backup and evidence:
`data/maintenance/preference-schema-20261029/` (backend-relative).

[Local startup](../../scripts/start_local_dev.sh) now applies reviewed 003..005
with `psql -X`, `ON_ERROR_STOP` and one transaction before API startup. Compose
initialization mounts 001..005 for fresh volumes; existing volumes require the
explicit startup upgrade, because PostgreSQL initdb scripts do not rerun there.
Fresh initialization, backup restore, upgrade/reapply and failed-startup controls
are verified by the scripts owner. This is a local schema deployment correction;
the earlier isolated acceptance reports remain historical evidence.

## Learning composition ports (LLRN-15, 2026-10-28)

`PostgresLearningRepository` 可注入同一 `documents` store 與 `preferences` repository，
建構時驗 scope/DSN；未注入仍保留原 constructor compatibility。公開 `document_store` /
`preference_repository` 供組裝核對；store 的 `namespace` / `dsn` 與 preferences 的 `dsn`
只供 server composition，不能下發或記錄 DSN。交易內仍借用同一 connection，無第二筆
write transaction、schema 變更或 constructor DDL/read。Config 使用實際 SQL store identity，
不用 SQL artifact repository 繼承的 file namespace。真 SQL container tests 驗證共享實例、
非 default scope、保存/reuse/lifecycle、缺 migration 明確失敗且零自動建表。

## Application orchestration verification (LLRN-13)

Learning SQL adapter 沿用共用 repository behavior 的 bounded block/operation reads、transaction-local effective-binding validator、每次 dispatch 的集中 admission 與明確 invalidation。沒有 SQL schema/migration 變更，也不在 read 補寫；callback 借用已鎖定偏好 snapshot，不另開 write transaction。Provider 執行始終在已提交 dispatch 的交易外。

`scripts/test_learning_orchestration.py` 有 40 項 file + 41 項真 PostgreSQL cases，SQL 額外驗證刷新 final persist 執行 SQL 後拋錯，全交易 rollback 保留舊 saved，且不重送模型；雙 client claim/reconcile、source/settings/profile race、lost ack 也在 disposable schema 驗證。另 290 項既有回歸通過。此證據不是 production migration、route/runtime composition 或獨立 profile 儲存的原子性保證。

## Document lifecycle transaction integration (LLRN-12)

`learning_lifecycle.py` 提供 transaction-local rename/delete 協調，`PostgresReadingPreferencesRepository.apply_document_lifecycle` 借用 caller 的同一 connection，無獨立 commit 或 DDL。Lock order 為 workspace preferences → canonical document → learning rows；rename 同交易遷移設定、舊名 tombstone、所有 sentence/key/result/operation/checkpoint/receipt references。Active operations 轉 stale，保留 unknown 費用，entry CAS revision 推進。全書歷史遷移使用五個集合查詢，成本隨書籍歷史增長；這是明確 lifecycle 維護，不能等同一般有界 saved-read。Delete 的 learning FK cascade 與 preference tombstone 同交易提交／回滾；不存在文件的 retry 不建立偏好。

`save` 初次／reparse 使用新 document epoch，普通 metadata save 保留 authoritative epoch/reading state，並使用真實目前 hierarchy 比對 effective source。`mutate_reading_content`、cleanup transaction、legacy save 同步保存 content epoch；改文作廢所有語言／級別，Undo 不能復活。舊 learning rows 保留歷史但 read/commit guards 拒絕使用。Cleanup snapshot 依 documents.current_structure_version 精確讀寫，不再選取／覆蓋其它歷史版本。沒有新 migration、read-time repair、LLM 或 profile 寫入。

File/SQL 共用矩陣及 SQL rollback／rename-late-writer、delete-preference-save concurrency 見 `scripts/test_learning_lifecycle.py`；真 PostgreSQL 使用 disposable schema。File coordinator 的恢復限制見 [owner design](../document_structure/module-detailed-design.md#implemented-source-and-document-lifecycle-llrn-12)。LLRN-13 application orchestration 已交付；LLRN-15 composition 已交付，LLRN-16 API/lifespan 已交付，UI 尚未交付。

## Language Learning Reading Mode Plan (2026-10-28)

**[Code-Confirmed: LLRN-07/08 complete]** [Canonical design](../section_tasks/module-detailed-design.md#language-learning-reading-mode-plan-2026-10-28)；[task dependency queue](../section_tasks/module-checklist.md#language-learning-reading-mode-queue-2026-10-28)。偏好及學習 artifact/operation SQL 已交付；LLRN-12 lifecycle 與 LLRN-13 application integration 已交付；LLRN-15 runtime DI 已交付，LLRN-16 production API 已交付，UI 仍待後續任務。

兩項獨立責任：reading_preferences 的 SQL adapter/migration，以及學習 artifact/operation 的 SQL adapter/migration。Domain ports 不帶 SQL；資料保持 namespace/document/hierarchy ownership，生成前 atomic claim，provider 在短交易外，結果與 receipt 原子提交並驗證 source、設定與 fencing。沿用 common artifact typed payload 原則；以 bounded rows/query 支援當頁，不為每語言/等級建表，也不預先承諾固定表數或擴大成全書 JSON。需明確 migration/rollback、唯一約束及 real SQL failure/concurrency 證據；不得以 read/import 執行 DDL。

### Implemented preference SQL persistence (LLRN-07)

`postgres_reading_preferences_repository.PostgresReadingPreferencesRepository(dsn, scope)` 實作既有 reading_preferences port。建構只保存配置；read 使用唯讀 transaction，缺資料回傳 revision=0 的 typed defaults，不 INSERT、DDL、repair 或載入全書庫偏好。未部署 migration 時回報 storage error，不冒充 missing。`reading_preferences/commands.py` 集中 file/SQL 共用 command signature、最後 receipt replay 與 revision guards；typed codec、mode/registry/proficiency 規則仍由原 domain 擁有。

顯式 `migrations/004_reading_preferences.sql` 新增兩張表：

- `reading_workspace_preferences`：exact namespace primary key、固定 workspace scope kind、schema_version=1、非負 BIGINT revision、可空 native_language 及最後 receipt。
- `reading_book_preferences`：namespace/document_id composite primary key、schema_version=1、revision、typed/versioned settings JSONB、deleted tombstone、最後 receipt；namespace FK 指向 workspace。使用 C collation 保留 exact identity，不去副檔名、不正規化 Unicode、不複製正文或按語言建表。Tombstone settings 必須是空的普通模式。SQL constraints 管物理形狀與唯一性，完整 payload/registry 語意由 domain codec 驗證；未知 persisted modes/versions 可讀並保留，不能新建或啟用。

偏好 document_id 是 port 接收的 authoritative logical identity，未與 documents 的 numeric surrogate ID 混用。刻意不設 document FK/cascade：缺省偏好可先存在，tombstone 必須在實際書籍刪除後保留，以拒絕遲到寫入。LLRN-12 已由 structured document store 同交易串接 rename/delete（見下節）；runtime composition 已由 LLRN-15 交付。

每次 mutation 以同一 connection/transaction 執行 workspace `INSERT ... ON CONFLICT DO NOTHING`（只在顯式寫入）→ workspace `FOR UPDATE` → 讀取/驗證需要的 book rows → 保存 settings/revision/receipt → COMMIT。Workspace row lock 涵蓋首次寫入競爭；同 workspace 寫入序列化，不同 namespace 獨立。最多讀一個 workspace 與兩個 book rows；沒有 workspace-wide snapshot。Book save 同時核對 workspace/book revision；rename 兩端 CAS、source tombstone、destination 及雙端 receipt 一次提交，反向 rename 也遵循同一鎖序。未提供可任意繞過鎖的 public raw writer、獨立 helper commit 或 provider 工作。

同值 save 不增 revision，但保存最後 command receipt；同命令 replay 不重寫設定。僅保證最後 receipt，非永久操作歷史；book replay 仍先驗 workspace revision，rename replay 必須兩端 receipt 仍相符。Delete 保留單調 tombstone，重新建立需最新 revision。資料壞損、SQL 失敗與 conflict 分別回報，已確認 server rejection 回滾整筆交易；COMMIT 階段連線失聯/結果不明回報 `PreferenceCommitUnknown`，由 read 與同命令 reconcile，禁止盲目 retry。Logs 僅 operation/outcome/duration，不包含 namespace、書名、設定、DSN 或私有 SQL 錯誤。

#### Explicit migration and downgrade procedure

由部署者選定正確 database/search_path，在維護流程執行 001..003 後，以 `psql -X --set ON_ERROR_STOP=1 --single-transaction --file db/migrations/004_reading_preferences.sql` 套用 004；連線由外部安全環境配置。Forward glob 僅取 migrations 頂層 SQL。004 可對其已完成的同版本 schema 重跑；不是任意舊/壞 schema 自動修復器，也不自動匯入既有 file preferences。

Downgrade 前停用偏好讀寫、備份兩張表及驗證可還原；顯式以同樣 single-transaction/ON_ERROR_STOP 方式執行 `db/migrations/rollback/004_reading_preferences.sql`。此檔會**刪除兩張偏好表及其資料**，不碰原 document/hierarchy/profile 表；不使用 CASCADE，未來依賴需先顯式降版。要保留設定，重新套用 004 後按 workspace → book 順序還原先前備份，再啟用服務。DDL 本身的交易失敗可回滾；成功 DROP 的資料不會自動復原。沒有 read/import/constructor DDL，也未修改既有啟動流程。

#### LLRN-07 validation

`scripts/test_reading_preferences_postgres.py` 36 項真 PostgreSQL 專測通過，包含直接沿用 file contract assertions、receipt/file parity、schema 升級/重跑/失敗回滾/降版/備份還原、原 documents 保留、scope/Unicode identity、無資料純讀、未知版本、CAS、兩程序首次競爭及同命令 replay、不同書保留、workspace race、rename/delete race、第二筆 SQL 失敗及 deferred COMMIT constraint rollback、commit lost acknowledgement、程序在 COMMIT 前後退出、target-bounded queries（另有 200 筆無關紀錄）及 log privacy。本輪連同 file preferences、learning repository/contracts/policy、語言分級及既有 SQL translation/reading-content 回歸共 250 tests passed，無 skips。另實際重啟 disposable PostgreSQL container 後，新連線仍可讀回 workspace/book settings 及 replay receipt。測試只用 disposable PostgreSQL container/random schemas，未套用使用者資料庫 migration；完整 runtime/API/UI 和 learning SQL 不在本項。

### Implemented learning SQL persistence (LLRN-08)

`PostgresLearningRepository(dsn, WorkspaceScope, admission=...)` 實作 LLRN-06 的 `LearningRepository` port；建構不連線或建表。共用 `document_structure/learning_repository_behavior.py` 的來源／設定驗證、claim/replay、checkpoint/commit assembly 和 LLRN-05 純策略。File/SQL 各自提供受保護的 storage context hooks，沒有公開任意 state writer；共用 codec 僅允許列舉的 typed DTO。本 adapter 不承包 provider/worker/API；LLRN-13 已協調，LLRN-15 已接線 runtime DI，production API 已由 LLRN-16 交付。

`migrations/005_learning_persistence.sql` 定義五張按 resource/key/operation 分開的表，不為語言／等級建表，也不存整書 learning JSON：

| Table | Identity and payload |
| --- | --- |
| learning_sentences | document_row_id + resource_id；唯一 sentence resource/typed spans，供多個 cache 重用 |
| learning_cache_entries | document_row_id + cache_id；resource FK、entry revision、last fence、result revision 及 typed current entry（含 saved output/current operation 投影） |
| learning_operations | document_row_id + operation_id；cache FK、fence/result revision、typed operation/plan/checkpoint digests；每 cache 的 fence 與非空成功 result revision 唯一 |
| learning_commands | document_row_id + command_id；跨 key 唯一 command receipt，可空 operation FK，保存 claimed/reuse/coalesce/denied 等結果 |
| learning_checkpoints | document_row_id + operation_id + batch_index；operation FK、typed partial payload，按 batch index 排序／分頁 |

每表有 schema_version=1、JSON object 形狀與對應非負 revision/index 等約束；完整 type/field、hash、scope、receipt、payload assembly 規則由共用 codec 驗證。複合主鍵／唯一索引支援 document/key lookup 和 checkpoint range；沒有任意 client SQL identifier。document_row_id 由當下 exact `(namespace, document_name)` 查詢取得，是現有 documents numeric ownership；所有 learning rows 有 document FK/ON DELETE CASCADE，child FK 保證同文件 resource/cache/operation 歸屬。Source binding 仍保留 authoritative logical document 與 hierarchy IDs，不把 SQL ID 塞入 transport。

#### Transactions and source/settings coordination

每次 mutation 使用同一 connection：`SET TRANSACTION ... READ COMMITTED` → `PostgresReadingPreferencesRepository.lock_snapshot(document_id, connection=...)` 取得 workspace row lock／當下 book 設定 → documents row `FOR UPDATE` → `PostgresStructuredDocumentStore.load_in_transaction(target, connection=...)` 載入 canonical hierarchy → 有界 learning state → 純策略／版本檢查 → 只寫改變的 rows → COMMIT。兩個 owner helper 都要求 caller-owned active transaction，不另開連線或 commit。首次 workspace 缺省 row 只可能在顯式寫入協調中建立；失敗時一起 rollback，不增加設定 revision。

固定鎖序 preferences → document → learning；現有偏好與正文寫者不能穿過 revalidation 與 COMMIT。即使 cache row 尚不存在，document lock 也防止重複 claim；同 namespace 目前沿用 workspace lock 序列化，沒有宣稱高吞吐量或逐 key 平行度。不同 key 保存自己的 rows，不覆蓋其它 cache；claim 的 sentence/entry/operation/receipt、checkpoint marker/payload、final output/succeeded history 都各自在一筆短交易內提交。Provider IO 必須在成功 dispatch 返回之後、交易外執行。

同命令 replay 返回 durable receipt，不重新取得執行權；同 key 不同命令 coalesce，未知結果需 reconcile／明確 retry，新 fence 拒絕舊 writer。成功保存的舊結果在 refresh pending/failed 時保留。提交前驗 current source/epoch、workspace/book revision、mode/language/proficiency、operation/fence/entry revision、batch alignment/digest/provenance 及最終結果組合。SQL/constraint 錯誤回滾所有 rows；COMMIT 連線失聯等未明結果回報 `LearningCommitUnknown`，先 lookup/reconcile，不能自動重送 provider。Read/constructor 不 expire、續租、清理或執行 DDL。Logs 記 stage、duration、counts、outcome，不記正文、key、書籍身份或 DSN。

#### Bounded reads and remaining lifecycle work

Saved reads 使用同一 `REPEATABLE READ READ ONLY` snapshot 同時讀來源與 learning rows，避免來源讀完後又取到另一版本結果。`db/learning_rows.py` 只載入請求 keys/references、其 current operation、保存結果對應的 successful operation，以及必要時單一明確查詢的歷史 operation/command；不遍歷整書所有 command/refresh/level 歷史。一次最多 100 keys/references，集合查詢無逐 key N+1；receipt/resource/checkpoint 只跟隨這個有界依賴集合。完整結果驗證會讀取所選 operation 的所有已存 batch payloads，成本與該結果大小相關，並非固定 bytes 保證。

`read_checkpoints` 另走 SQL batch-index range + LIMIT 1–100，僅驗證當頁 payload/digest 與目前 source，不載入其它 checkpoint payload。完整 canonical hierarchy/effective source projection 仍沿用既有 document loader，會讀取該書 hierarchy；本項不宣稱整個 source projection 已 SQL 分頁或完成大型書庫效能驗收。相同未知 schema/type、遺失 receipt、壞 digest 是 storage failure，不能當 missing 或 read-time repair。

現有內容修改／Undo／reparse 的 source/epoch guard 已阻擋舊 key 讀取與晚到提交；document DELETE 由 FK 清除 SQL-owned rows，刪除後重建的 numeric identity 不會復用舊結果。LLRN-12 已補齊全語言／全級別 source binding 作廢、rename references 與偏好交易協調；保留歷史證據而不要求每次改文物理刪除 learning rows。Runtime 尚未組裝完整模式。

#### Migration and validation evidence

在 001..004 之後，由部署者對正確 database/search_path 顯式以 `psql -X --set ON_ERROR_STOP=1 --single-transaction --file db/migrations/005_learning_persistence.sql` 套用；同版本可重跑，不修復任意舊 schema。沒有改動啟動流程或自動匯入 file sidecar。實作依賴 LLRN-07 的偏好 schema/transaction helper，任務 DAG 已補上此依賴。

Downgrade 需先停 learning writers 並備份五表，顯式執行 `db/migrations/rollback/005_learning_persistence.sql`（同樣 single-transaction/ON_ERROR_STOP）；會刪除 learning rows，包括昂貴生成結果。保留資料需重新套用 005，再按 sentences → entries → operations → commands/checkpoints 還原備份。Rollback 檔不在 forward glob 中，不使用 CASCADE，不碰原 document/profile/preference 表；未來依賴先各自降版。交易內的失敗 DDL 可 rollback，已成功 DROP 不會自行恢復資料。

`scripts/test_learning_postgres_repository.py` **47 tests passed**；279 項 aggregate 初輪 278 通過，file/SQL 對照 fixture 因兩 backend 初始 epoch 不同失敗；統一該 fixture 的來源身份後重跑全部 47 SQL 專測通過。合計 **279 個不同測試**有通過證據、無 skips（47 新 SQL + 232 既有）。另實際重啟 disposable PostgreSQL container，保存結果、成功 receipt、running refresh 及 replay 均保留，未重新授權 dispatch。

涵蓋六程序同 key claim（一勝五 coalesce）、同 command replay、不同 key 保留、四程序 dispatch（一勝三 conflict）、source/preferences 真 row-lock wait、read snapshot 競爭、CAS/fence/source/settings/stale worker、partial/empty hints、失敗 refresh、receipt/checkpoint/deferred COMMIT rollback、lost ack、claim/dispatch/result COMMIT 前後 `os._exit`、跨 scope/書、corruption/遺失 receipt、bounded queries/page、直接 file/SQL 對照、005 升降版/失敗 rollback/昂貴結果備份還原、FK/unique constraints 與 log privacy。全部為隔離 PostgreSQL/random schemas；未改真書、未部署使用者資料庫、未呼叫真 provider 或重啟正式服務。

## Translation And Same-Text Persistence (2026-10-26)

Supporting TRUI-02..06, counted in the shared translation queue, not a new DB task.
Translations reuse document_task_artifacts metadata and mutate_document_artifacts;
the domain resolves hierarchy ownership and executes model work outside the lock.
Same/different-language CAS and replay merge fresh metadata under the document row
lock. No DDL, broad mutation API or independent nested commit is added.

Reading mutations distinguish content revision from a neutral replay receipt. Same-
text submissions preserve documents.updated_at and never run artifact purges. Real
Keep edits delete summary fields from hierarchy JSONB and summary-kind artifact rows
without deleting Quiz/other kinds; changed-block translation metadata is cleared in
the same transaction. Existing consented Clear and deleted-block Note cleanup remain.
Disposable-SQL tests verify concurrent writers, post-write trigger rollback, no-op
timestamps and source-dependent lifecycle. All are part of 408 passing backend tests.

## Source Region Repair Persistence (2026-10-26)

Supporting domain-owned SRREPAIR-02/03; no duplicate DB task or new DDL. Existing
reading-state JSONB and structured snapshots hold region_drafts, region_applications,
source_region_id ownership, affected/cleared block IDs and durable mutation receipts.
Domain stages exact range changes; the existing document lock and transaction commit
overlay, audit, Undo, interaction epochs, document metadata and snapshot together.
No SQL-side mapping policy, independent helper transaction or model work under lock.

The artifact purge now accepts all cleared canonical block IDs. It remains scoped
to the current document and resolves physical content-block targets through existing
metadata/identity rules using parameterized SQL. Full removal and explicit Clear
purge Notes as well as other block interactions; unaffected Notes remain. Undo does
not resurrect purged artifacts or rewind permanent manual flags/interaction epochs.
File and disposable PostgreSQL tests cover multi-block Note rows, rollback after
metadata/artifact work, duplicate replay and concurrent revision conflicts.

## Cleanup Record Persistence (2026-10-25)

RCR-03 stores `refresh_source_token` beside the existing mutation audit revision,
affected block IDs and invalidation metadata in reading-state JSONB/snapshot. The
existing transaction persists this binding with body/artifact changes; response
projection failure inside the mutation rolls back. No new table, DDL, SQL adapter
rewrite or read-time migration. Old receipts without the binding remain replayable
but cannot claim a current targeted refresh. GET uses one current repository snapshot
and checks source/revision in the domain service without writing or repairing state.

RCR-01 extends the same JSONB with cleanup_record_applications and source-edit
record ownership. Applied state and overlays participate in the same Undo snapshot;
saved drafts and permanent manual block flags do not. Reapply replaces a verified
owned overlay, never appends another copy. Legacy ownership inference is domain-owned
and read-only until an explicit successful mutation. No DDL/read-time migration.
The proposed RCR-02 retention pruning was withdrawn before implementation; preserve
history, active overlays and idempotency/Undo evidence regardless of navigation state.
RCR-01 SQL validation passes within the 281-test regression: both persisted payloads
carry the same applied-text ownership, reapplication rolls back after injected SQL
failure, and concurrent writes cannot overwrite newer revisions. Domain helpers
use the existing transaction port; no database DDL or SQL repository rewrite needed.

CLEAN-RESTORE uses existing document reading-state JSONB and structured snapshot;
no new table/column or DDL migration. Persistent block_edit_flags, cleanup_record_links
and cleanup_record_edits are server-owned. Body Edit sets the flag in the same
transaction as body/artifact changes; Undo does not erase the historical flag.
Cleanup draft edits and record restores do not set the whole-block flag.

The existing mutate_reading_content transaction revalidates source/revision and
restoration eligibility before writing state, audit, snapshot and artifact effects.
Draft saves preserve generated data and Note; Restore invalidates generated data
while retaining Note. `_commit_cleanup_state` binds links against final persisted
task-unit IDs before saving cleanup-only state. Reads never repair/backfill JSONB.
Mapping/protection rules stay in document_structure, not SQL code.

Disposable PostgreSQL tests in test_cleanup_history.py verify both JSONB copies,
Undo flag persistence, post-SQL failure rollback, replay/concurrency, unchanged raw
source and artifact/Note behavior. These pass within the 266-test regression; no
production data migration or book writes were performed. Counted under CLEAN-RESTORE.

## Standalone Cleanup Commit (2026-10-23)

**[Code-Confirmed: SCLEAN-04, 2026-10-24]** Storage implementation and supporting file
adapter parity under the structure-owned repository contract. See the
[active design](../document_structure/module-detailed-design.md#independent-smart-cleanup-2026-10-23).
Expose one narrow internal content-only cleanup commit, not a public generic plan
writer and not a forged unchanged enhancement. Under a short lock/transaction,
revalidate source/hierarchy fingerprint and reading revision; stage safe deletions;
atomically write reading overlays/audit/Undo/review state, generated invalidation,
ready-layout revision stamp and minimal operation receipt. Never replace hierarchy,
renumber units/blocks, advance structure_version or rewrite raw source/profile.

Notes and concurrent manual work must survive. New effective content advances
reading_revision once; no-op/failure must not invalidate generated output or bump
content revision. A receipt-only terminal outcome may be stored separately from a
content change. Replay the same operation idempotently; later edits/Undo supersede
historical results and do not authorize reapplication. Roll back receipt and content
together on write failure, and reject late generation writes at stale revisions.
Provider/planning work is outside transactions. Share staging primitives with the
existing combined enhancement commit, retaining its one-transaction semantics;
do not let storage orchestrate layout or call back into preparation/app.

Keep existing audit/read/Undo routes and relational hierarchy as truth. File and
PostgreSQL contracts must agree on rollback, replay and no-resurrection. No schema
change is presumed necessary; any required migration must be explicit in SCLEAN-04,
not runtime table creation or a lazy read repair.

### SCLEAN-04 Delivery (2026-10-24)

Both repositories implement `CleanupRepository.commit_cleanup` and read-only
`lookup_cleanup`. The file adapter scopes its configured base directory to one
namespace (default `default`, explicit constructor override supported). PostgreSQL
uses its existing configured namespace and locks the document row with FOR UPDATE.
Neither entry manufactures an enhancement candidate or changes hierarchy/version.
File storage uses its existing lock and one atomic replacement; SQL uses one
connection/transaction and the shared `_commit_cleanup_state` helper. SQL snapshot
reading state and generated metadata now agree with relational state after cleanup.
No migration, provider request, profile generation or layout orchestration is added.

`document_structure/cleanup_commit.py` owns common locked-baseline comparison,
source/evidence/view checks, policy binding, stage revalidation and receipt handling.
It calls `validate_cleanup_input`, not duplicated deletion rules. Both standalone
and combined enhancement reuse `ocr_cleanup_overlay.stage_validated_cleanup` for
overlay application, audit, review resolution, Undo and changed-only invalidation.
Ready revision stamps reuse `reading_content.advance_layout_revision`. A carried
stamp is not proof of source/options/resolver cache validity; the existing readiness
owner must still check that in SCLEAN-05/07. Missing units are never built in storage.

Under the lock, compare the accepted document snapshot and version, recompute the
effective view/manual protections, verify authoritative OCR/page coordinates, and
rerun exact shared validation. Caller-supplied staging must equal the recomputed
result. Page labels remain preserved provenance, not executable coordinates.
Source-file bytes are checked with the existing bounded/race-aware hash utility
before and just before final commit; there is no file extraction. Raw-file lifecycle
is not part of the SQL transaction: this does not promise cross-resource atomicity.
Changes after the final source check remain the source owner's lifecycle concern.

Minimal standalone receipts live in reading-state JSON under
`standalone_cleanup_receipts`, keyed by namespace/operation digest, storing command
digest plus typed binding/outcome/readiness, never book text or raw plan. They commit
with content, audit, review and generated invalidation. A no-op/rejected plan may
persist only a receipt and does not advance reading revision or clear generated
output. Same-command replay writes nothing; changed-command reuse conflicts. Undo
or a newer reading/structure revision rejects reapplication as superseded. Lookup
returns historical evidence, not permission to overwrite current state; the app
must reconcile current state separately. A 1000-receipt bound rejects new commits
without eviction/reapplication. Lost acknowledgement is logged as unconfirmed, not
assumed rollback; lookup recovers durable evidence. Fresh consent remains SCLEAN-05.

Validation: 22 shared file cases and 27 inherited/additional real PostgreSQL cases;
216 local regressions plus 114 disposable SQL/API/enhancement regressions pass.
Coverage includes partial/no-op, same-operation and competing-operation concurrency,
manual/reparse races, failed final writes/receipt writes, lost acknowledgement,
stale workers, notes/profile/IDs preservation, revision stamps, five-level deleted
content behavior and Undo. SQL rollback compares pre/post row revisions as well as
payloads; notes/generated rows and a nonempty profile are independently checked.
Three hash-verified backups run through temporary file storage and isolated SQL;
Sample2's previously reviewed deletion commits/reloads/replays/undoes with source
unchanged. Initial real-book failure exposed extra page_label metadata; coordinate
comparison was corrected and both backend suites rerun. No new live LLM calls,
production-library mutation, service restart or frontend/API capability claimed.

## Allocated Unit And Block IDs (2026-10-23)

Supporting CLEAN-10: when an explicit hierarchy/layout save allocates a numeric
task-unit ID different from the incoming ID, rebind exact generated content-block
IDs with TaskUnit.with_unit_id and update the same row's metadata within the SAME
transaction. Custom block IDs remain intact. No table/schema migration, source
rewrite, second commit or read-triggered repair is introduced. Existing stale stored
rows are not silently migrated. Real SQL tests cover a large compatibility block,
effective-content projection without duplication and cleanup Undo; three backed-up
OCR books pass the live API/layout/content-resolver path after this correction.

## Cleanup Operation Receipts (2026-10-23)

Supporting app CLEAN-06: optional internal operation metadata on commit_enhancement
persists a minimal enhancement_receipts map in document provenance and the existing
snapshot within the SAME hierarchy/overlay transaction. No new table/migration,
candidate staging save or model call is added. Records bind backend operation hash,
accepted extraction/text identity, resulting structure/reading revisions, structural
change kind, section count and actual cleanup outcome; no source text, prompt, token
or provider response. Up to 1000 receipts are retained; eviction never authorizes
automatic regeneration. Existing no-operation structural no-op still performs no
writes. Opted-in no-op may persist receipt/overlay without changing structure_version.

Empty cleanup applications retain existing manual/source overlays through structural
replacement when optional planning fails; receipt outcome remains failure/skip.
Receipt replay is checked before mutation, and changed revisions conflict. Final
receipt-write failure rolls back hierarchy, notes/generated invalidation and overlay
with it. Lost acknowledgement is reconciled via persisted evidence by app; storage
does not run layout or claim durable precommit generation. Real PostgreSQL rollback,
cleanup-only retention, late edit, restart and replay tests are in CLEAN-06.

## Cleanup Preparation Support (2026-10-23)

Supporting preparation CLEAN-05: structural enhancement preserves trusted source
page/review/actual-OCR evidence, including the optional-cleanup-failed/no-application
path. Explicit task-layout updates reject reading-revision mismatch. On late metadata
saves after a reading edit, compare against the locked actual hierarchy before allowing
unit replacement; do not construct a fake current hierarchy from incoming stale units.
These are internal consistency checks, not new schema or API fields. Disposable SQL
tests exercise effective-to-canonical unit persistence, stable IDs, empty-section Undo,
failed cleanup, metadata retention and concurrent-edit rejection.

## Optional Cleanup Atomic Commit (2026-10-23)

Supporting structure CLEAN-04; no additional completion checkbox here.
commit_enhancement accepts optional internal CleanupApplication. Under existing
locks it checks receipts, source/version/confirmation, stages the validated overlay
on the in-memory candidate, then commits hierarchy/reading state/invalidation once.
No save-reload-clean-save sequence and no provider calls inside the transaction.
Unchanged/title-only cleanup preserves IDs/version and note rows, clearing generated
rows and metadata only when content changes. Structural replacement preserves the
new staged overlay through a private same-transaction save option, while retaining
the established all-interaction cleanup. External hard-reparse defaults still clear
old reading overlays. Final write failure rolls back the whole operation.
Internal EnhancementCommitResult.cleanup is optional; CLEAN-04 itself exposed no
public DTO. CLEAN-06 now adds explicit-consent public orchestration as described above.

Receipt replay returns the stored historical cleanup outcome and current snapshot
without duplicate writes; command mismatch conflicts. Canonical spans do not depend
on the temporary candidate IDs that PostgreSQL replaces. Source-review decisions
and undo/audit state persist with reading metadata. File-store parity and 27 actual
disposable-schema tests cover rollback after hierarchy/cleanup writes, concurrent
requests, generated rows versus notes, all-level projection, stable identity,
review resolution and Undo. Combined 304 cases: 303 pass, one existing opt-in private
source case skipped; no production schema/data mutation or live-model acceptance.

## Atomic Reading Mutations (2026-10-21)

OCR-REVIEW-05 uses mutate_reading_content with SELECT FOR UPDATE on the document
inside one PostgreSQL transaction. Persist reading overlays in metadata provenance;
clear generated document/hierarchy artifacts and generated artifact rows atomically.
Preserve note metadata, raw source, hierarchy identity and document_profile. No new
table is required. Optimistic revision/source checks run under the lock. Replays
are no-ops. Hard reparse clears overlays; ordinary metadata writes preserve them.
Disposable-schema tests cover concurrent writers, late saves, stale generation,
notes, reparse and an injected task-unit update failure after the document update,
proving rollback of the entire mutation. No production rows are touched by tests.

## Generic Extraction Commit Checks (2026-10-21)

**[Code-Confirmed]** EXTRACT-07 is implemented; see [the preparation plan](../document_preparation/module-detailed-design.md#format-neutral-extraction-plan-2026-10-21).
Replace PDF-key-specific candidate/current-source comparisons and provenance
whitelisting with the shared normalized extraction contract. Preserve atomic source,
hierarchy, label and interaction guards, existing no-op/title-only retention and
whole-document cleanup on structural change. Persist/read back generic provenance
through existing storage; no new table is presumed. Compatibility normalization
must not write during reads. Explicit identity backfill must atomically preserve
version/IDs/interactions/profile and reject concurrent changes. Test PostgreSQL
readback with omitted raw_text and existing file-store paths, not only mocks.

## Atomic Summary And Note Mutations (2026-10-21)

Supporting SN-07/14/15, no new table/migration. Repository callbacks acquire
documents ROW EXCLUSIVE table lock before FOR UPDATE row lock, read current
hierarchy and metadata in the same transaction, validate and update only
document_task_artifacts JSONB. This serializes notes and cleanup, avoids stale
editor overwrite, and follows enhancement lock ordering. Generic metadata saves
preserve note/public_summary; only explicit callbacks or structural replacement
own those fields. Identity/source checks remain domain-owned, not parser rules.
Document rename preserves numeric hierarchy IDs/version, rekeys new interaction
identities, updates source metadata/name and snapshot labels atomically. Source
file moves are separately compensated by the lifecycle route if DB rejects.
Evidence: disposable PostgreSQL note/summary tests, rename-route rollback test,
and conditional enhancement no-op/title-only/structural/failure tests.

## Conditional Enhancement Commit (2026-10-20)

Implemented in ENH-REV-02 (2026-10-21): the classification contract
in `../document_structure/module-detailed-design.md`, "Enhancement Autonomy And
Change Semantics", determines persistence. ENH-REV-02 owns this change; previous
hard-reparse cleanup remains restricted to actual structural replacement.

| Validated change | Persistence | Interactions |
|---|---|---|
| unchanged | No new version, hierarchy write or replacement event | Retain all |
| titles_only | Atomic label update on existing matching nodes | Retain all |
| structure_changed | Atomic whole-document hierarchy replacement, version +1 | Clear all |

Title-only implementation must preserve chapter/section/task-unit/block identities,
body spans, parent links, source_structure_version and interaction payloads. Updating
labels must not use the hard-reparse sanitizer, invalidate artifacts indirectly by
advancing the structure version, or discard drafts through an API side effect.
Keep authoritative labels and current JSONB snapshot/projections consistent in one
transaction; do not introduce a second hierarchy truth. A title update is not an
initial_parse/hard_reparse event; do not forge a version-increment event to fit an
old enum. If label revision/audit storage is required, define it explicitly in this
task with migration/tests rather than repurposing structure_version.

Revalidate source identity, baseline hierarchy AND labels under the existing lease/
commit locks. A concurrent label edit must invalidate stale consent even though
structure_version is unchanged; app ENH-REV-03 binds a baseline fingerprint that
includes labels. Compute classification against the revalidated baseline, not a
client-provided change kind. Rollback preserves previous labels and all artifacts.
For no-op, returning old authoritative version/IDs is sufficient; do not mutate
document snapshots simply to record an attempt. Safe operational logs are separate.

For structure_changed, retain current all-document cleanup and late-writer guards;
no selective reparse, local cleanup or partial artifact migration is authorized.
For titles_only, old-version interaction writers remain legitimate because target
structure did not change; prevent a delayed hierarchy/snapshot write from restoring
old labels. Tests must prove both artifact retention and label freshness.

Implementation evidence (2026-10-21): `PostgresStructuredDocumentStore.commit_enhancement`
classifies against locked authoritative readback after validating the candidate,
source identity, version and confirmation fingerprint. The existing fingerprint
already covers chapter/section labels and row revisions, so concurrent label edits
invalidate confirmation without a new schema column or migration. All three branches
retain current lease, table-lock and version-guard prerequisites.

- `unchanged` returns existing authoritative IDs/version without UPDATE/INSERT/DELETE,
  snapshot rewriting, cleanup, profile invalidation or parse events.
- `titles_only` patches chapter/section titles and section container titles on existing
  rows and the current JSONB snapshot in one transaction. Snapshot structural parity
  is validated before accepting positional label correspondence, independent of its
  original generated IDs. Unknown embedded metadata, task units/blocks, interaction
  payloads, profile data and structure version are retained; no parse event is created.
- `structure_changed` uses the existing full replacement/cleanup transaction and
  increments structure_version once. Source bytes are rechecked after writes.

`EnhancementCommitResult` extends the internal snapshot with `change_kind` for the
next app task. Public API exposure, conditional consent, and UI no-op/title-only
recovery are NOT claimed here; ENH-REV-03/05 still own those behaviors.
Ordinary metadata saves now reload authoritative chapter/section labels while holding
the document row lock. A delayed current-version interaction save may succeed, but
cannot restore old labels in either relational rows or the JSONB snapshot. Explicit
parser/hard-reparse saves retain their separate ownership of structural replacement.

Validation: `scripts/test_enhancement_conditional_commit.py` runs 43 disposable-schema
PostgreSQL tests (42 pass; one private-source opt-in skipped), including 11 new
conditional-commit cases plus existing guard/route regressions. Row payloads and xmin
prove zero no-op writes and preserved artifacts/units/blocks for title-only updates.
Tests cover label readback, stale confirmation, label/artifact writers, source changes,
snapshot mismatch, rollback and delayed metadata saves. The 20 pure comparator tests
and manual-reparse repository boundary regression also pass. Tests make no live LLM
calls or production document/schema changes.

## Enhancement Commit Guards (2026-10-19)

ENH-DB-01 delivered as a PostgreSQL repository operation. PREP-03 invokes it from the
confirmed public route only when migration/profile guards are available. The caller
must hold the document enhancement lease across candidate generation and commit.
The original pre-ENH-REV-02 implementation revalidated the candidate, accepted extraction identity, source
hash, structure version and interaction fingerprint before replacement. A shared
connection keeps cleanup, hierarchy replacement, one version increment and readback
inside one transaction; any failure rolls everything back.

Commit takes fixed-order, NOWAIT table locks and a document row lock. This deliberately
conservative implementation can reject unrelated-document writers too; locks cover
only the short commit, never LLM generation. Conflicts require renewed confirmation,
not an automatic paid retry. Cleanup removes old relational/embedded interactions,
task units/blocks and obsolete snapshots; raw-source provenance and source-level
profile summary/language survive. DB-backed derived profile structure is invalidated.

Migration `003_interaction_version_guards.sql` rejects stale relational artifact and
block writes. Embedded writes validate the loaded structure version, hierarchy IDs
and changed artifact versions. Server-owned versions travel from resolved reading
targets through generation policy metadata into CommonArtifact; delayed old work
cannot be relabelled with the current version on save. Missing guards reject commit.

Evidence: 22 disposable PostgreSQL schema tests cover cleanup/readback, rollback,
source changes, consent revision changes, lock conflicts, missing guards and real
artifact-store stale resurrection. Migration has not been applied to user schemas.
Source files are checked before and after writes, but are not transactional DB data;
deployment must prevent out-of-band source replacement. PREP-03 adds a readonly
profile invalidation projection from the committed, server-owned enhancement_source
marker. Runtime profile adapters suppress stale file-derived structure advice without
pretending filesystem writes are atomic. File hierarchy storage remains unsupported.
The live suite now contains 30 cases including public routes and an opt-in source
copy; no production migration was executed here.

Supporting ENH-APP-01 delivered: `load_enhancement_snapshot` reads all current
hierarchy, embedded metadata, raw-source and relational artifact rows in one
REPEATABLE READ READ ONLY transaction. The revision fingerprint includes full row
payloads and xmin; this is not a monotonic durable revision or commit guard. Exact
interaction count is null where embedded representations cannot be deduplicated;
known notes are flagged. `enhancement_lease` uses a session advisory lock scoped by
namespace/document, supports multiple processes, and releases on failure/close.
It does not lock interaction writers. Eight live disposable-schema tests validate
this handoff, including concurrent reads and lease exclusion; commit coverage is above.

ENH-DB-01 extends existing transactional hard replacement, not a
new hierarchy/history model. PostgreSQL currently loads `raw_text=""`; accepted
canonical source reuse therefore needs verified spans/hash/length reconstruction
through the preparation contract, not an assumption of persisted full raw text.

For a structure_changed replacement, before deleting anything, compare the confirmed source identity, structure version
and interaction revision under appropriate transaction locks. Every affected artifact
writer, including session updates and embedded payload changes, must participate in
revision/locking semantics; counts alone cannot detect edits. Revalidate source-file
identity via the owning source boundary and prevent replacement races during commit.
Mismatch rolls back with a conflict and requires renewed consent, not automatic retry.

Inventory relational artifacts and embedded document/chapter/section/task-unit
payloads, including book-level interactions. Clean candidates must not carry old
embedded artifacts into the new hierarchy. A structural replacement commit atomically replaces the
hierarchy, increments its version once and clears all consented derived interactions,
answers/evaluations/sessions and affected task units/blocks. Explicitly identify any
persisted user notes in the impact inventory; do not assume note support exists.
Preserve source files and accepted extraction provenance. Candidate rejection,
persistence errors and stale tokens preserve previous hierarchy and interactions.

Document-scoped single-flight integration must avoid duplicate paid work without
holding DB locks during generation. File-backed deployments must provide equivalent
atomic behavior or explicitly report enhancement unavailable. Test concurrent artifact
writes, competing enhances, failed commits, embedded artifact resurrection, and
successful cleanup/readback. Read-only preflight must not mutate storage.

## 1. Module Purpose

`db/` is the golden source for DB-era persistence planning and isolated Phase 1 DB validation implementation. It records maintainer-approved DB identity, versioning, reparse, and derived-resource lifecycle rules, and now includes the first PostgreSQL Phase 1 DDL shape plus an isolated core hierarchy persistence validation slice. **[Maintainer-Confirmed] + [Code-Confirmed]**

The SQLite validation slice remains isolated. PostgreSQL store/repository adapters now
also support explicitly selected runtime persistence, including permanent document
deletion. Backend selection remains owned by `config`. **[Code-Confirmed]**

## 2. Position in Overall Architecture

- Future DB persistence planning / governance layer
- Isolated Phase 1 core hierarchy persistence validation layer
- Cross-module design source for future schema and storage implementation work

The existing file-backed runtime remains valid until DB readiness, rollout, and validation gates are explicitly satisfied. **[From Proposal]**

## 3. Key Files

| File | Responsibility | Notes |
|---|---|---|
| `db/module-detailed-design.md` | Golden source for DB-era identity, versioning, reparse, derived-resource lifecycle, and logical schema proposal policy | Documentation only; no SQL/schema implementation approval |
| `db/module-checklist.md` | Tracks DB-era planning status and future implementation tasks | Future tasks remain unchecked until implemented with evidence |
| `db/storage-contract-inventory.md` | Inventory of current storage contracts before DB migration design | Documentation-only preparation |
| `db/storage-contract-design.md` | Conceptual storage contract boundaries before schema/backend implementation | Documentation-only preparation |
| `db/structured-document-jsonb-evaluation.md` | Phase 1 `StructuredDocument` JSONB-first evaluation plan | Evaluation planning only; no schema approval |
| `db/structured-document-jsonb-evaluation-readiness.md` | Readiness audit for future Phase 1 JSONB evaluation | Audit/planning only; no implementation impact |
| `db/phase-1-schema-design.md` | Implementation-ready schema design reference for Phase 1 logical tables/entities, candidate fields, relationships, ownership boundaries, and constraint candidates | Documentation only; no SQL, ORM, repository interface, migration, runtime behavior, or backend selection |
| `db/migrations/001_phase_1_core_hierarchy.sql` | PostgreSQL Phase 1 migration shape | Covers the executable production-target DDL surface for `documents`, `raw_source_metadata`, `document_profile`, `parse_events`, `chapters`, `sections`, `task_units`, `content_blocks`, `artifacts`, and optional `structured_document_snapshots`; does not enable runtime rollout |
| `db/sqlite_validation/phase_1_core_hierarchy_schema.sql` | SQLite validation-only schema mirroring the core hierarchy subset | Used only by the isolated local validation adapter for accepted hierarchy write/read checks; not a production migration and not the full Phase 1 DDL surface |
| `db/phase_1_core_schema.py` | Applies the isolated SQLite validation schema to a SQLite connection | Validation helper only; no production runtime switch |
| `db/sqlite_core_document_store.py` | Minimal SQLite-backed accepted hierarchy write/read adapter | Uses DB-generated IDs on readback; not a production repository abstraction or backend selection mechanism |
| `db/postgres_structured_document_store.py` | PostgreSQL-backed current StructuredDocument persistence bridge | Supports runtime structured save/load/exists/location and lightweight document list/search for the selected PostgreSQL backend |
| `db/postgres_structured_document_artifact_repository.py` | PostgreSQL-backed artifact repository adapter | Delegates structured document load/save, task-layout cache updates, and lightweight document list/search to PostgreSQL store |
| `shared/common_artifact_model.py` + `section_tasks/reading_interaction_common_artifact.py` | Common DTO and adapter for reading interaction artifacts | Implementation support for mapping `analysis`, `quiz`, and `critical_thinking_session` to one artifact entity shape before broader durable artifact repository rollout |

## 4. Main Responsibilities

1. Define DB-era identity policy before schema design. **[Maintainer-Confirmed]**
2. Define document-level `structure_version` semantics. **[Maintainer-Confirmed]**
3. Define current-state-only hierarchy persistence policy for early DB design. **[Maintainer-Confirmed]**
4. Define hard reparse transaction and derived-resource invalidation policy. **[Maintainer-Confirmed]**
5. Define minimal parse event provenance requirements. **[Maintainer-Confirmed]**
6. Preserve boundaries from existing repository memory: hierarchy-first, no hidden mutation, metadata advisory-only, and no backend/schema authority. **[From Proposal] + [From HLD]**
7. Derive the first logical relational persistence model from documented domain semantics. **[Maintainer-Confirmed] + [Doc-Confirmed]**
8. Provide an isolated executable validation slice for new-document core hierarchy persistence, without enabling production runtime DB reads or writes. **[Code-Confirmed]**

## 5. Non-Responsibilities

1. Does not own API routing or backend selection; PostgreSQL runtime storage is exposed through store/repository adapters selected by `config`. **[Code-Confirmed]**
2. Does not select a final DB backend beyond previously documented PostgreSQL JSONB evaluation assumptions. **[From Proposal]**
3. Does not redefine `StructuredDocument` hierarchy truth. **[From HLD]**
4. Does not make database schema a parser, hierarchy, artifact, profile, retrieval, or raw-document authority. **[From Proposal]**
5. Does not preserve current Python-generated `unit_id` as production identity. **[Maintainer-Confirmed]**
6. Does not introduce immutable hierarchy history, staging hierarchy tables, or candidate promotion workflows in the first DB design. **[Maintainer-Confirmed]**

## 5.1 Implemented Phase 1 Core Validation Slice

The implemented slice validates the smallest DB-backed hierarchy path for new documents. **[Code-Confirmed]**

Implemented scope:

- defines a PostgreSQL Phase 1 migration shape for `documents`, `raw_source_metadata`, `document_profile`, `parse_events`, `chapters`, `sections`, `task_units`, `content_blocks`, `artifacts`, and optional `structured_document_snapshots`
- applies a separate SQLite validation-only schema for the isolated local adapter
- persists an already accepted `StructuredDocument` hierarchy with `current_structure_version = 1`
- stores `namespace` and `document_name` on `documents` for namespace/document isolation
- stores raw-source metadata only, not raw bytes or raw text
- appends an `initial_parse` parse event
- reads current hierarchy back through `Document -> Chapter -> Section -> TaskUnit`
- exposes DB-generated IDs in the read model instead of current Python-generated `unit_id`
- keeps `documents` free of raw text; `source_path` readback comes from `raw_source_metadata.source_location`
- validates the SQLite hierarchy slice with `scripts/test_db_phase_1_core_hierarchy_persistence.py`
- validates the PostgreSQL migration shape with `scripts/test_db_phase_1_postgresql_migration_shape.py`
- enforces same-document parentage for static hierarchy and content-block parent links with PostgreSQL composite foreign keys
- enforces parse-event row-shape invariants with PostgreSQL CHECK constraints while keeping `documents.current_structure_version` authoritative
- enforces numeric and span representation integrity with conservative PostgreSQL CHECK constraints without inferring or repairing parser output
- defines application-managed `updated_at` ownership for PostgreSQL rows without timestamp triggers or hidden lifecycle mutation
- supports parser-level PostgreSQL hard reparse replacement by clearing current hierarchy/derived rows inside the save transaction before inserting the accepted candidate hierarchy
- keeps PostgreSQL repository task-layout/artifact updates separate from hard reparse by preserving `current_structure_version` on non-parser repository saves and reloading DB-generated task-unit ids after task-layout cache writes
- sanitizes parser-level replacement payloads through the repository boundary so stale structured-document task artifacts and referenced-artifact metadata are not reinserted during PostgreSQL hard reparse saves
- maps reading interaction artifacts (`analysis`, `quiz`, `critical_thinking_session`) onto a shared `CommonArtifact` DTO shape with `artifact_type`, hierarchy-aware target metadata, payload, metadata, and provenance fields
- validates current-artifact semantics for analysis and quiz at the service orchestration boundary: read returns only current persisted artifacts or `not_generated`, while generate/refresh is the explicit write path for completed or insufficient-content artifacts

Out of scope:

- production runtime DB read/write switch
- profile persistence
- content-block persistence
- artifact persistence
- durable reading-interaction artifact repository read/write wiring beyond the common entity mapping contract
- JSONB parity snapshot implementation
- migration of existing JSON outputs
- public/domain identity introduction

## 5.2 Explicit Runtime Document Deletion

`PostgresStructuredDocumentArtifactRepository.delete_document` delegates to
`PostgresStructuredDocumentStore.delete`, using the same normalized logical name and
configured namespace as reads/writes. Parameterized `DELETE ... RETURNING id` runs in a
transaction and returns whether a row existed. It performs permanent deletion, including
non-active records, rather than leaving a hidden row available to later reads or writes.
**[Code-Confirmed]**

Existing `ON DELETE CASCADE` ownership constraints remove raw-source metadata, profile,
parse events, chapters, sections, task units, content blocks, artifacts, snapshots,
OCR runs and OCR pages. No new migration is needed. A DB error rolls the DB transaction
back; physical raw/FAISS files are handled separately by the explicit lifecycle route.
This is document deletion, separate from hard-reparse replacement semantics.
**[Code-Confirmed]**

`scripts/test_postgres_document_delete.py` applies both migrations in a unique temporary
schema and verifies all owned rows disappear, missing retries return false, and another
namespace's same-name document plus another same-namespace document survive.
**[Test-Confirmed]**

## 6. DB-Era Identity Strategy

### 6.1 Default Identity Rule

DB-generated primary keys are the default internal identity and relational link foundation. **[Maintainer-Confirmed]**

They should be used for:

- document rows
- chapter rows
- section rows
- task-unit rows
- content-block rows
- artifact rows
- internal joins and foreign keys
- persistence consistency

### 6.2 Public / Domain Identity Rule

Do not create a separate public/domain identity for every hierarchy surface by default. **[Maintainer-Confirmed]**

At the current stage, there is no explicit business requirement for a separate public document identity. `documents.id` is sufficient for the first DB-backed API and application use. **[Maintainer-Confirmed]**

A separate public/domain identity may be introduced only when there is a concrete long-lived external reference requirement beyond the DB row lifecycle. Candidate areas include:

- public document identity
- artifact target snapshots
- content-block validation metadata

Concrete external stability requirements for public document identity may include public sharing, external SDK/API usage, cross-system integration, permanent external references, or multi-tenant public URLs. Until such a requirement exists, public document identity remains a future extensibility point rather than a Phase 1 requirement. **[Maintainer-Confirmed]**

Separate public/domain IDs carry generation, uniqueness, persistence, mapping, and lifecycle costs. They must be deliberate, not automatic. **[Maintainer-Confirmed]**

### 6.3 Reference / Import Identity Rule

Existing file-based JSON data is reference/test data only and will not be production-migrated as stable API-visible identity. Current Python-generated `unit_id` values are reference/import identity evidence only. **[Maintainer-Confirmed]**

Future DB work must not depend on current Python-generated `unit_id` as the production task-unit identity strategy. **[Maintainer-Confirmed]**

### 6.4 API Identity Rule

Early API references may use DB-generated IDs unless a stable public API requirement is explicitly introduced later. **[Maintainer-Confirmed]**

For the first DB-backed API, the document DB-generated identifier is acceptable as the API-facing document identifier for internal APIs and application use. A separate public document key, UUID, or slug should not be introduced in Phase 1 without a concrete external stability requirement. **[Maintainer-Confirmed]**

API-visible DB IDs are valid only within the current document structure version. If a client uses an ID from an older structure version, the system may return a stale/version error when enough metadata exists, otherwise it may fail as not found. **[Maintainer-Confirmed]**

## 7. Structure Version Policy

### 7.1 Authority

`documents.current_structure_version` is the authoritative current hierarchy version. Parse events are provenance records only and must not become an event-sourced version authority. **[Maintainer-Confirmed]**

### 7.2 Scope

`structure_version` is scoped to the whole document hierarchy snapshot, not to individual chapter, section, or task-unit rows. **[Maintainer-Confirmed]**

The first successful parse creates `structure_version = 1`. Each successful hard reparse advances the version to `2`, `3`, and so on. **[Maintainer-Confirmed]**

### 7.3 Version Type

`structure_version` is a simple monotonic integer per document. It is advanced only after a successful hierarchy replacement transaction. **[Maintainer-Confirmed]**

It is not derived from content hash, parser hash, or structure hash. Those hashes may exist as separate metadata for validation, cache behavior, or observability, but they are not the hierarchy version value. **[Maintainer-Confirmed]**

### 7.4 Hierarchy Row Versioning

Chapter, section, and task-unit rows do not store separate `structure_version` values in the first DB design. **[Maintainer-Confirmed]**

The early model is current-state-only:

- hierarchy rows represent only the current accepted structure
- no historical hierarchy snapshots are retained by default
- no child-level reparse exists
- row-level structure versions can be reconsidered only if future historical or partial-reparse requirements appear

## 8. Current-State-Only Hierarchy Model

Early DB design should model only current hierarchy persistence and transaction cutover. **[Maintainer-Confirmed]**

It should not introduce:

- immutable historical hierarchy snapshots
- old row aliases
- staging hierarchy tables
- candidate hierarchy states
- promotion workflows
- non-current hierarchy persistence areas

Candidate hierarchies are built outside durable DB storage, then validated before the transaction starts. **[Maintainer-Confirmed]**

## 9. Content-Block Persistence Policy

Content blocks are lazy materialized resources linked to task units. They are not embedded inside `StructuredDocument` JSONB as the default persistence strategy. **[Maintainer-Confirmed]**

Once computed, content blocks should be persisted separately and relationally linked. **[Maintainer-Confirmed]**

Content blocks should store:

- `document_id`
- task-unit DB foreign key
- `source_structure_version`
- source hash
- block index or deterministic block metadata
- segmentation version
- span metadata when applicable
- schema/version metadata

Derived `source_structure_version` is for provenance and application-level defensive validation. It is not a DB-enforced cross-table lifecycle constraint in the first design. **[Maintainer-Confirmed]**

## 10. Artifact Target Policy

Artifact targets should not rely on old Python-generated `unit_id`. **[Maintainer-Confirmed]**

Early artifact targets should use DB foreign keys plus validation metadata where applicable:

- target table / target type
- target DB id
- `source_structure_version`
- source hash
- quote span if applicable
- schema/version metadata

Artifacts are derived interaction outputs and must not become hierarchy truth, parser authority, or a way to recreate hierarchy identity. **[From HLD] + [Maintainer-Confirmed]**

## 11. Hard Reparse Policy

### 11.1 Meaning

Hard reparse is a document-level hard refresh / debug recovery operation. It is used when parse quality is too low or when the user rejects the current parsed structure. **[Maintainer-Confirmed]**

It is not a normal production versioned update, and early DB design should not preserve previous derived rows or hierarchy rows as historical records by default. **[Maintainer-Confirmed]**

### 11.2 Candidate Validation

The candidate hierarchy must be built and validated before any destructive DB changes occur. If validation fails, nothing is deleted and the existing current hierarchy remains usable. **[Maintainer-Confirmed]**

### 11.3 Transaction Order

The hard reparse transaction order is:

1. Replace current hierarchy rows with the validated candidate hierarchy.
2. Advance `documents.current_structure_version`.
3. Explicitly delete all derived rows for the document.
4. Write the minimal parse event record.
5. Commit.

All steps occur in one transaction, so external readers must not observe intermediate state. **[Maintainer-Confirmed]**

### 11.4 Derived Resource Cleanup

Successful hard reparse physically deletes all content blocks and all artifacts for the document. **[Maintainer-Confirmed]**

Cleanup is by document ownership, not by `source_structure_version` matching:

- delete content blocks where `document_id = ?`
- delete artifacts where `document_id = ?`

This prioritizes exhaustive document-level hard refresh and avoids leaving stale derived rows behind due to inconsistent metadata. **[Maintainer-Confirmed]**

Cleanup should be an explicit service-layer lifecycle step. Database `ON DELETE CASCADE` may be used defensively to prevent orphans, but it must not be the primary mechanism defining hard reparse semantics. **[Maintainer-Confirmed]**

## 12. Parse Event Provenance

Minimal parse event records are required for both `initial_parse` and `hard_reparse`. **[Maintainer-Confirmed]**

Parse events are append-only provenance metadata. They are not hierarchy history, full audit logs, compliance records, event sourcing, or a retained version-history system. **[Maintainer-Confirmed]**

### 12.1 Event Types

Required event types:

- `initial_parse`
- `hard_reparse`

### 12.2 Initial Parse Event

Initial parse event fields should include:

- `document_id`
- `previous_structure_version = null`
- `new_structure_version = 1`
- `trigger_source = upload / prepare`
- parser mode or parser strategy if available
- `parsed_at`

### 12.3 Hard Reparse Event

Hard reparse event fields should include:

- `document_id`
- `previous_structure_version`
- `new_structure_version`
- `reparse_reason`
- `trigger_source`, such as `internal_debug`, `user_requested`, or `low_parse_score`
- parser mode or parser strategy if available
- `reparsed_at`
- invalidated artifact count
- invalidated content-block count

The successful parse/reparse transaction updates `documents.current_structure_version` and writes the matching parse event together. **[Maintainer-Confirmed]**

### 12.4 Retention Policy

Parse event records should be retained for the lifetime of the document and deleted when the document is deleted. **[Maintainer-Confirmed]**

Retention rules:

- retain parse events while the document exists
- delete parse events when the document is deleted
- do not keep parse events after document deletion as independent audit records
- do not introduce separate archival, TTL, or compliance retention in Phase 1
- keep parse events document-scoped provenance only

This policy reflects that parse events are minimal provenance explaining how accepted structure versions were created, why `structure_version` advanced, and why derived resources may have been cleared. They are not compliance audit logs or a long-term operational history system. **[Maintainer-Confirmed]**

## 13. Validation Semantics

Application reads may validate:

```text
derived.source_structure_version == documents.current_structure_version
```

If a mismatch appears, the row should be treated as stale or invalid defensive data. **[Maintainer-Confirmed]**

This check remains application-level in the first DB design. DB-level triggers or cross-table constraints may be reconsidered later if stale derived rows become a real operational problem. **[Maintainer-Confirmed]**

Timestamp updates follow the same explicit-write principle. `documents.updated_at` may use a DB default for initial insert, but later document mutations must set it explicitly in the repository/service update statement. `artifacts.updated_at` remains nullable until a mutable artifact update occurs, and that mutation must set the timestamp explicitly. Phase 1 does not use PostgreSQL triggers or ORM hooks for timestamp mutation. **[Maintainer-Confirmed] + [Code-Confirmed]**

## 13.1 Manual Reparse Save Boundary Under PostgreSQL

The PostgreSQL store has separate semantics for parser-level replacement saves and artifact/task-layout metadata saves. Replacement saves should clear current hierarchy and derived rows in the transaction, advance `documents.current_structure_version`, and append a `hard_reparse` parse event. Non-replacement repository saves should preserve the current version and existing hierarchy identity for artifact/task-layout updates. **[Code-Confirmed]**

The manual `manual_structure` reparse path uses the conditional `commit_manual_structure(...)` repository boundary with an accepted-source verifier so existing PostgreSQL documents are saved with replacement semantics instead of the non-replacement artifact/task-layout save boundary. This avoids inserting a validated manual draft over existing hierarchy rows protected by uniqueness constraints such as `uq_chapters_document_order(document_id, chapter_order)`. **[Code-Confirmed]**

Original confirmed failure example:

- document: `暗水幽灵`
- PostgreSQL `documents.id`: `13`
- existing chapter rows: `chapter_order` includes `0`
- manual draft first chapter order: `0`
- failure: duplicate key on `uq_chapters_document_order`

Boundary requirements:

1. Keep the current-state-only hierarchy model; do not add row-level hierarchy versions or historical hierarchy tables for this bug. **[From HLD]**
2. Keep artifact/task-layout repository saves non-replacement so ordinary interaction writes do not become hidden hard reparses. **[From HLD] + [Code-Confirmed]**
3. Route manual structure commits under PostgreSQL to an explicit hard-reparse replacement boundary. **[Code-Confirmed]**
4. Ensure the replacement path remains transactional: existing hierarchy/derived rows are replaced only after validation and draft materialization succeed; failed persistence rolls back with the old hierarchy still usable. **[Maintainer-Confirmed] + [Code-Confirmed]**

## 14. Relationship to Existing Planning Documents

This document supersedes any interpretation that current Python-generated `unit_id` should be centered as production DB identity. Existing JSON files remain useful as reference, fixtures, and evaluation material only. **[Maintainer-Confirmed]**

This document does not invalidate Phase 1 StructuredDocument JSONB evaluation planning. It narrows identity and lifecycle assumptions that future DB planning must use before schema design. **[Doc-Confirmed]**

## 15. Non-Goals for First DB Design

1. No production migration of existing JSON identity.
2. No automatic dual-ID system for every hierarchy node.
3. No immutable hierarchy snapshots.
4. No retained retired artifact rows.
5. No retained retired content-block rows.
6. No content-block reuse across hard reparse.
7. No staging hierarchy persistence.
8. No row-level hierarchy versioning.
9. No DB-enforced lifecycle constraint for derived `source_structure_version`.
10. No full audit/history system.
11. No schema/table/column design in this document.
12. No ORM, repository interface, migration, fixture, or runtime behavior implementation in this document.

## 16. Logical DB Schema Proposal (Phase 1)

> This section is a logical persistence model proposal only. It does not define SQL, DDL, physical indexes, database-specific types, ORM models, repository interfaces, migration scripts, fixtures, runtime behavior, or backend selection. **[Maintainer-Confirmed] + [Doc-Confirmed]**

### 16.1 Scope

Phase 1 logical DB planning covers the persistence domains needed to represent the current-state document hierarchy and its directly related derived resources. It follows repository memory rather than current Python file layout details. **[From Proposal] + [From HLD] + [Maintainer-Confirmed]**

Included logical persistence domains:

1. Document and structure lifecycle.
2. Current structured hierarchy.
3. Minimal parse event provenance.
4. Lazy content-block materialization.
5. Artifact persistence and target references.
6. Advisory profile snapshot placement.
7. Raw-source metadata placement.
8. Backend policy placement.

Excluded from Phase 1 logical schema proposal:

1. SQL / DDL / PostgreSQL-specific implementation.
2. ORM models.
3. Repository interfaces.
4. Migration scripts.
5. Runtime read-path switch.
6. Immutable hierarchy history.
7. Staged candidate hierarchy persistence.
8. Production migration of existing JSON identities.
9. Retrieval index physical storage.
10. Raw document binary/blob storage.

### 16.2 Logical Persistence Domains

| Domain | Owner | Authority | Phase 1 Logical Role |
|---|---|---:|---|
| Document identity and lifecycle | `document_structure` for structured semantics; broader upload ownership remains separate | Authoritative for current structure version | Holds document row identity, current `structure_version`, document name/namespace identity, and optional public identity if later required. |
| Current hierarchy | `document_structure` | Authoritative hierarchy truth | Represents current `Document -> Chapter -> Section -> TaskUnit` rows only. |
| Parse events | `db` planning + `document_structure` semantics | Provenance only | Records accepted initial parse / hard reparse events matching document state. |
| Content blocks | `shared` contract + `document_structure` hierarchy link | Derived, not hierarchy authority | Stores lazy materialized content blocks linked to current task-unit rows. |
| Artifacts | `document_structure` persistence semantics + `shared` target contract | Interaction output, not hierarchy authority | Stores summaries/quizzes/interaction outputs and their validated target references. |
| Profile snapshot | `profile` | Advisory only | Candidate DB surface for profile metadata snapshots; not parser authority. |
| Raw-source metadata | raw document ownership remains separate; `db` records metadata boundary only | Source reference metadata only | Tracks file-backed raw source location and descriptive metadata; does not store raw bytes. |
| Backend policy | `config` | Not domain authority | Selects file/DB/coexistence policy later; does not define data meaning. |

Retrieval artifacts and raw document bytes remain separate migration tracks. Raw-source metadata may be tracked in DB, but canonical uploaded raw bytes remain file-backed for the first DB rollout. **[From Proposal] + [From HLD] + [Maintainer-Confirmed]**

### 16.3 Proposed Logical Entities

| Entity | Owner | Responsibility | Authority |
|---|---|---|---|
| `Document` | `document_structure` semantics; raw ownership remains separate | Current document identity, namespace/name isolation, current `structure_version`, and DB-generated document identifier for first DB-backed APIs. Public document identity remains a future extensibility point only. | Authoritative for current structure version. |
| `Chapter` | `document_structure` | Current chapter node under a document. | Hierarchy truth as part of current structure. |
| `Section` | `document_structure` | Current section node under a chapter. | Hierarchy truth as part of current structure. |
| `TaskUnit` | `document_structure` + `shared` DTO contract | Current task-unit interaction container under a section, with compatibility content payload as needed. | Hierarchy interaction container; not a content-block hierarchy parent in the persistence-authority sense. |
| `ParseEvent` | `document_structure` lifecycle semantics | Minimal provenance for accepted initial parse and hard reparse. | Provenance only; not source of current version authority. |
| `ContentBlock` | `shared` content-block contract | Lazy materialized render/interaction segment linked to a task unit. | Derived resource, not hierarchy truth. |
| `Artifact` | `document_structure` artifact persistence semantics + `shared` artifact DTOs | Common persisted interaction output entity with `artifact_type`, target metadata, type-specific payload, and lifecycle/provenance metadata. | Interaction output only. |
| `ArtifactTarget` | `shared.ArtifactTargetRef` semantics | Validated target reference metadata for artifact attachment. | Target metadata; not hierarchy authority. |
| `ProfileSnapshot` | `profile` | Advisory document-scoped profile metadata snapshot, including profile payload, version metadata, source parser/preparation mode, language/script metadata, title/author/source metadata, optional source structure version, and advisory diagnostics when produced. | Advisory only. |
| `RawSourceMetadata` | raw document ownership remains separate | Metadata-only reference to file-backed raw source, including source location/path/object key, original filename, MIME type, file size, checksum/fingerprint when available, upload time, and ownership scope when applicable. | Source reference metadata only; not raw-byte storage. |
| `StorageBackendPolicy` | `config` | Future backend selection / rollout / coexistence policy. | Configuration only, not persistence meaning. |

Entity names are logical labels, not table names. They do not prescribe physical naming, columns, constraints, or implementation types. **[Doc-Confirmed]**

### 16.4 Entity Responsibilities

#### `Document`

Responsibilities:

- Own the DB-generated internal document primary key.
- Preserve document name / namespace isolation currently represented by file paths and storage configs.
- Store authoritative `current_structure_version`.
- Use the DB-generated document identifier for first DB-backed internal APIs and application use.
- Keep separate public document identity as a future extensibility point only if a concrete external stability requirement appears.
- Provide the lifecycle anchor for hard reparse cleanup and parse events.

Non-responsibilities:

- Does not contain raw document bytes by default in this Phase 1 model.
- Does not make profile, retrieval, or artifacts authoritative.
- Does not derive current version from parse events.
- Does not introduce a public document UUID, key, slug, or permanent external identifier in Phase 1.

#### `Chapter`

Responsibilities:

- Represent current accepted chapter hierarchy under one document.
- Preserve ordering and chapter-level display metadata needed for hierarchy parity.
- Link to sections through current-state-only relationships.

Non-responsibilities:

- Does not store row-level `structure_version` in the first design.
- Does not preserve old aliases after hard reparse.
- Does not own artifact payloads directly.

#### `Section`

Responsibilities:

- Represent current accepted section hierarchy under one chapter.
- Preserve ordering and section-level display metadata needed for hierarchy parity.
- Link to task units through current-state-only relationships.

Non-responsibilities:

- Does not reintroduce root `sections[]`.
- Does not serve as a flat primary source independent of chapters.
- Does not store row-level `structure_version` in the first design.

#### `TaskUnit`

Responsibilities:

- Represent the current task-unit interaction container under a section.
- Preserve task-unit order and current task-unit content payload needed for Phase 1 parity.
- Serve as the parent link for lazily materialized content blocks.

Non-responsibilities:

- Does not use current Python-generated `unit_id` as production DB identity.
- Does not create a separate domain ID by default.
- Does not make content blocks a persisted hierarchy level.

#### `ParseEvent`

Responsibilities:

- Record accepted `initial_parse` and `hard_reparse` events.
- Store previous/new structure version values.
- Store trigger source, parser mode/strategy if available, timestamp, and hard-reparse invalidation counts.
- Retain events for the lifetime of the document and delete them when the document is deleted.

Non-responsibilities:

- Does not define current document version.
- Does not retain historical hierarchy snapshots.
- Does not become compliance audit or event sourcing.
- Does not survive document deletion as an independent audit record.
- Does not require separate archival, TTL, or compliance retention in Phase 1.

#### `ContentBlock`

Responsibilities:

- Persist lazy materialized content-block resources separately from `StructuredDocument` JSONB.
- Link internally to the owning document and task-unit DB row.
- Store `source_structure_version`, source hash, block index / deterministic block metadata, segmentation version, span metadata, and schema/version metadata.
- Support application-level stale validation against `Document.current_structure_version`.

Non-responsibilities:

- Does not become hierarchy truth.
- Does not survive hard reparse by default.
- Does not require cross-version reuse in Phase 1.

#### `Artifact`

Responsibilities:

- Persist interaction output payloads in one common logical artifact entity for Phase 1.
- Store document ownership, `artifact_type`, target metadata, type-specific payload, and `source_structure_version`.
- Store source hash / span metadata where applicable, creation time, and other lifecycle/provenance metadata needed by artifact validity checks.
- Support explicit deletion for the whole document on successful hard reparse.
- Preserve payload metadata needed for cache validity and future validation.

Non-responsibilities:

- Does not create, rename, or re-own hierarchy identity.
- Does not become parser authority.
- Does not survive hard reparse by default.
- Does not split into category-specific logical tables such as `summary_artifacts`, `quiz_artifacts`, or `answer_artifacts` in Phase 1.
- Does not prevent later category-specific tables if artifact schemas stabilize, type-specific constraints become important, query patterns require dedicated tables, or generic payload validation becomes insufficient.

#### `ArtifactTarget`

Responsibilities:

- Represent target type and resolved DB target id for artifact attachment.
- Store validation metadata such as `source_structure_version`, source hash, quote span, and schema/version metadata where applicable.
- Allow application logic to classify malformed / unresolved / stale / source-mismatched targets later.

Non-responsibilities:

- Does not allow repository writes without hierarchy-aware validation.
- Does not depend on current Python-generated `unit_id`.
- Does not turn content blocks into hierarchy nodes.

#### `ProfileSnapshot`

Responsibilities:

- Represent the first DB-backed advisory profile persistence track as a document-scoped logical entity linked to `Document`.
- Preserve profile metadata as a snapshot payload, not as normalized hierarchy truth.
- Store minimum logical metadata: `document_id`, profile payload / profile snapshot, `profile_version` or `schema_version`, `generated_at`, source parser / preparation mode when available, language / script metadata when available, title / author / source metadata when available, `source_structure_version` when generated after structure creation, and advisory diagnostics when currently produced.
- Support freshness/provenance metadata conceptually without blocking hierarchy persistence.

Non-responsibilities:

- Does not control parser structure or act as parser authority.
- Does not define hierarchy truth, artifact availability, retrieval authority, or raw document authority.
- Does not create, mutate, rename, or delete chapters, sections, task units, content blocks, or artifacts.
- Does not block `StructuredDocument` hierarchy persistence.
- Does not receive diagnostics profile write-back through hidden read-path mutation.

#### `RawSourceMetadata`

Responsibilities:

- Track metadata for the canonical file-backed raw uploaded document.
- Link raw-source metadata to the owning document.
- Store raw source location / file path / object key, original filename, MIME type, file size, checksum / fingerprint when available, `uploaded_at`, and ownership scope when applicable.
- Keep raw document storage as a separate track from first-rollout DB-backed structured outputs.

Non-responsibilities:

- Does not store raw document bytes in the DB for the first rollout.
- Does not make raw source storage part of the first DB-backed structured-output lifecycle.
- Does not weaken raw file ownership, deletion, retention, or copyright boundaries.
- Does not prevent future object storage or DB-backed raw storage from being evaluated separately.

#### `StorageBackendPolicy`

Responsibilities:

- Represent future file/DB/coexistence rollout policy in configuration.
- Select backend implementations when runtime work is later approved.

Non-responsibilities:

- Does not define schema meaning.
- Does not define hierarchy, profile, retrieval, raw document, or artifact semantics.

### 16.5 Logical Relationships

```text
Document
  -> Chapter
    -> Section
      -> TaskUnit
        -> ContentBlock

Document
  -> ParseEvent

Document
  -> Artifact
    -> ArtifactTarget

Document
  -> ProfileSnapshot

Document
  -> RawSourceMetadata
```

Relationship rules:

1. `Document -> Chapter -> Section -> TaskUnit` is the current hierarchy truth path.
2. `ContentBlock` links to `TaskUnit` but is not a hierarchy level and is not embedded in `StructuredDocument` JSONB by default.
3. `Artifact` belongs to `Document` and may have one or more target references depending on future artifact semantics.
4. `ArtifactTarget` resolves to document/chapter/section/task-unit/content-block targets only after hierarchy-aware validation.
5. `ParseEvent` belongs to `Document` and records provenance for accepted versions, but the document row remains version authority.
6. `ProfileSnapshot` belongs to `Document` and remains advisory.
7. `RawSourceMetadata` belongs to `Document` as metadata-only reference to file-backed raw bytes.
8. Hard reparse is document-scoped: replace current hierarchy, advance document version, delete document content blocks and artifacts, write parse event, commit.

### 16.6 JSONB vs Relational Placement Rationale

| Data | Proposed Placement | Rationale |
|---|---|---|
| Document lifecycle fields | Relational logical entity | `current_structure_version`, document identity, and namespace isolation are frequently referenced lifecycle anchors. |
| Chapter / section / task-unit hierarchy | Relational logical entities | Hierarchy lookup, target validation, ordering, and fail-fast behavior need explicit relationships and current-state replacement semantics. |
| Optional full `StructuredDocument` snapshot | JSONB, validation-only / parity aid | JSONB can preserve model round-trip parity and help debug missing or incorrectly mapped fields, but it must not become hierarchy authority over relational current hierarchy semantics. |
| Task-unit content string / opaque payload | Relational field or payload field under `TaskUnit` | Phase 1 parity requires opaque content preservation; it is task-unit payload, not identity. |
| Content blocks | Relational logical entity with metadata payload as needed | Lazy materialized resources need independent lifecycle, task-unit linkage, version/source metadata, and hard-reparse deletion. |
| Artifact payload | One common `Artifact` logical entity with relational lifecycle/target fields plus type-specific payload | Phase 1 stabilizes the common artifact lifecycle first: document ownership, hierarchy/content-block target reference, `artifact_type`, `source_structure_version`, source hash/span metadata where applicable, creation time, hard-reparse invalidation, and non-authority boundary. Generated payload shape may vary by artifact type, but it should initially remain inside the common artifact entity rather than split into category-specific tables. |
| Artifact target metadata | Relational target fields plus structured metadata payload | Target type and resolved DB id need checkable references; source hash/span/version metadata may remain flexible. |
| Profile snapshot | JSON payload under profile-owned document-scoped entity | Profile metadata is advisory and snapshot-like; relationalizing it early would add little authority and may imply parser control. The first profile track should store `document_id`, profile payload, version metadata, generation/source metadata, optional language/script/title/author/source metadata, optional `source_structure_version`, and advisory diagnostics when produced. |
| Backend policy | Configuration, not domain persistence | `config` owns selection/rollout, not semantic data storage. |
| Retrieval index | Outside Phase 1 logical schema | Retrieval remains derived, partially DB-candidate, and track-separated. |
| Raw-source metadata | Metadata-only DB reference to file-backed raw source | First DB rollout should track `document_id`, raw source location / file path / object key, original filename, MIME type, file size, checksum / fingerprint when available, `uploaded_at`, and ownership scope when applicable. |
| Raw document file/blob | File-backed storage outside first DB rollout | Raw uploaded documents are canonical user-owned source files with larger storage, deletion, retention, and copyright boundaries. Moving raw bytes into DB would expand the migration scope too early; future object storage or DB-backed raw storage can be considered separately. |

The placement rule is: relationalize identity, ownership, ordering, lifecycle, and target validation; keep variable advisory/generated payloads flexible where they are not authority. **[Inferred]**

Phase 1 may keep an optional full `StructuredDocument` JSONB snapshot alongside the relational current hierarchy only as a validation/parity artifact. The relational current hierarchy remains the operational read/write model, `Document.current_structure_version` remains authoritative, and the snapshot is not parser authority, runtime hierarchy truth, or a fallback read path. If relational rows and the JSONB snapshot conflict, the conflict is a validation failure, not dual-authority behavior. **[Maintainer-Confirmed]**

### 16.7 Ownership Boundaries

1. `db/` owns logical schema memory, SQL adapters and explicit migrations; domain rules remain with their owning modules and runtime backend composition remains in `config`. The earlier planning-only boundary is superseded by implemented PostgreSQL adapters, including LLRN-07 preferences.
2. `document_structure` owns hierarchy semantics, current-state hierarchy persistence meaning, parse lifecycle semantics, and artifact persistence semantics against validated hierarchy targets.
3. `shared` owns DTO vocabulary for `TaskUnit`, `TaskUnitContentBlock`, `ArtifactTargetLevel`, `ArtifactTargetRef`, and artifact payload shapes.
4. `config` owns future backend selection, rollout flags, and DI policy, not domain semantics.
5. `document_preparation` owns preparation lifecycle and should eventually write through storage policy without assuming file paths as the only destination.
6. `profile` owns advisory profile snapshot semantics as a Phase 1 logical entity; it remains document-scoped and non-authoritative.
7. Raw document bytes remain file-backed for the first DB rollout; only raw-source metadata belongs in the Phase 1 logical DB proposal.
8. `retrieval` and raw document byte storage remain separate tracks and should not be silently pulled into Phase 1 hierarchy schema.

### 16.8 Open Design Questions

No unresolved Phase 1 logical schema confirmation items remain. **[Maintainer-Confirmed]**

### 16.9 Phase 1 Logical Schema Non-Goals

1. No SQL or DDL.
2. No physical table names beyond logical entity labels.
3. No column list.
4. No indexes or constraints.
5. No ORM model design.
6. No repository interface design.
7. No migration script.
8. No fixtures.
9. No runtime behavior change.
10. No backend selection.
11. No public API contract change.
12. No category-specific artifact tables in Phase 1.
13. No DB storage of raw document bytes in the first rollout.

## 17. Current Risks

1. risk: future DB implementation may accidentally introduce a second identity system for every hierarchy node.
- why: domain IDs add generation, mapping, uniqueness, and lifecycle costs.
- guardrail: use DB-generated IDs by default; add public/domain IDs only for concrete external stability requirements.

2. risk: schema design may preserve current `unit_id` as production identity.
- why: existing JSON fixtures expose `unit_id`, but they are not production data.
- guardrail: treat current `unit_id` as reference/import evidence only.

3. risk: hard reparse cleanup may be hidden behind cascade behavior.
- why: invalidating derived rows is a domain lifecycle decision.
- guardrail: keep explicit service-layer cleanup and parse-event counts.

4. risk: parse events may drift into event sourcing or history.
- why: version provenance can be mistaken for authoritative state.
- guardrail: keep `documents.current_structure_version` authoritative.

5. risk: logical schema proposal may be mistaken for implementation approval.
- why: entity labels can look like table design.
- guardrail: keep this section explicitly logical and require a separate implementation task before SQL, ORM, repository, migration, or runtime work.

## 18. Open Questions for Maintainer

Open questions for Phase 1 logical schema are listed in section 16.8. No additional unresolved confirmation items are identified for the previously captured DB-era identity and hard-reparse policy.

## 19. Suggested Next Documentation Improvements

1. When schema work is explicitly requested, derive an implementation checklist from this document before proposing tables.
2. Before any DB runtime work, reconcile this document with `document_structure`, `config`, `document_preparation`, `shared`, and `section_tasks` module checklists.
3. If a public API stability requirement appears, add a targeted public/domain identity decision record before implementation.

## 20. Future Direction Note: Reading Interaction Artifact Persistence

> 本節記錄 reading interaction artifacts 對現有 common artifact model 的映射；不代表目前 implementation。 **[Maintainer-Provided] + [Future Direction]**

1. Reading interaction outputs should use the existing common logical `Artifact` entity rather than category-specific artifact tables in the first version. **[Maintainer-Provided] + [From HLD]**
2. Required initial `artifact_type` values are `analysis`, `quiz`, and `critical_thinking_session`. **[Maintainer-Provided] + [Future Direction]**
3. `analysis` and `quiz` are current artifact per validated target; normal read returns the current persisted artifact or missing/not-generated. **[Maintainer-Provided] + [Future Direction]**
4. `critical_thinking_session` is multi-session per validated target. A generated question is persisted immediately even if the user never answers. **[Maintainer-Provided] + [Future Direction]**
5. Critical-thinking session payload/status should support `question_generated`, `insufficient_content`, `answer_submitted`, `evaluation_failed`, and `completed`. **[Maintainer-Provided] + [Future Direction]**
6. Artifact targets must remain hierarchy-aware: document/book maps to the document target, chapter to chapter id, section to section id, and task unit to task-unit id; optional parent ids may be used only for consistency validation. **[Maintainer-Provided] + [Future Direction]**
7. Artifact metadata should include source structure version, source hash where available, schema version, prompt instruction version, context mode, token estimate/budget, and evidence ids. **[Maintainer-Provided] + [Future Direction]**
8. Insufficient-content is a valid persisted artifact/session status with reason metadata, especially for OCR noise or symbol-only targets. **[Maintainer-Provided] + [Future Direction]**
9. Hard reparse invalidates/deletes these artifacts with other derived resources; PostgreSQL replacement deletes current artifact rows in the hard-reparse transaction, and repository replacement saves sanitize replacement structured-document payloads before persistence. No historical artifact table or orphan-preserving history is required for the first version. **[Maintainer-Provided] + [From HLD] + [Code-Confirmed]**
10. The artifact model must not become hierarchy truth, parser authority, task-layout payload authority, or a backend session cache. **[From HLD] + [Future Direction]**

### 20.1 Referenced Artifact Metadata

1. Higher-level artifacts may reference lower-level artifacts used as secondary generation context. **[Maintainer-Provided] + [Future Direction]**
2. Reference metadata should be stored as artifact metadata/payload fields such as `referenced_artifact_ids`, referenced artifact types, referenced target levels, artifact context mode, coverage counts, and deduplication/abstraction hint flags. **[Maintainer-Provided] + [Future Direction]**
3. Referenced artifacts remain interaction memory only. They do not become source text, hierarchy truth, parser authority, or a replacement for hierarchy-aware target validation. **[From HLD] + [Maintainer-Provided]**
4. Hard reparse cleanup still invalidates/deletes both generated artifacts and their reference relationships because all referenced artifacts are derived from the replaced hierarchy. **[Maintainer-Provided] + [From HLD]**
5. Absence or pruning of lower-level artifact references must not make a higher-level artifact invalid when the primary source context is sufficient. **[Maintainer-Provided] + [Future Direction]**
