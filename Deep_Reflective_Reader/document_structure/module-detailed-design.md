# document_structure Detailed Design

## PROFILE-05 publication coordination (2026-11-13, locally implemented)

Public file store save and save_and_load must acquire the same document .lock as repository lifecycle writes and profile snapshots. Only a private serialization primitive may write without reacquiring it. Repository mutations already hold this lock and keep their existing atomic writer; they must not nest public store save. save_and_load returns readback before unlocking so Prepare cannot adopt a competing incarnation.

`scripts/test_profile_prepare_publication.py` passes 18/18 file/real PostgreSQL cases; external publication races change from 2 failures to 2 passes. Original six 6/6, persistence 32/32, runtime 24/24, atomic artifacts 18/18, generated-source 23/23, PREP 142/142 and SQL support 16/16 pass; seven scripts pass. See `/tmp/profile05-fix-2c1ece0c/final-report.txt`. Local validation only; latest SHA requires external review.

## PROFILE audit integration (2026-11-13)

PROFILE-01 supporting port: StructuredDocumentStore exposes a short profile publication
snapshot context using the same per-document file lock as repository lifecycle writes.
It loads the latest document or explicit absence without saving or generating. Profile
owns validation/publication; this port does not make profile hierarchy authority.
Public save now holds that same lock; save_and_load reads back before release.
Repository mutations keep their existing locked atomic writer and never reenter
public save. Learning lifecycle journal application/recovery already takes the
repository document lock. The private store serialization helper is called only
inside its public locked writers.

Validation: profile persistence 32/32 file/real SQL methods, original six assertions
6/6, PREP runtime 24/24, exact-index 10/10 and 142 related regressions pass.
Atomic artifact 18/18 and generated-source 23/23 preserve existing fences; config
35/35 passes with valid profile fixtures. No new module audit or deployment.
Evidence: `/tmp/profile-fix-fc1dd285/final-report.txt`; external review pending.

## STRUCT-04 generation binding (2026-11-13, implemented)

Generated summary/quiz writes carry a server-created snapshot binding in artifact
metadata: document identity/incarnation, structure and reading revisions, rename
generation epoch, target identity and exact effective hierarchy digest. Capture
before provider work; validate against latest state inside the existing mutation
lock/SQL transaction. Same-text no-ops retain validity; changed text, Undo, reparse,
rename (including round trips) and reimport reject old writers. Non-null generated
fields in combined writes obey the same rule. Legacy unbound artifacts remain
readable but are not valid caches or authorized new writes; no read-time repair.
Effective projected text is the provider source, never a persisted hierarchy.
Rename rotates a separate artifact generation epoch inside its existing durable
boundary without replacing learning/source epochs. Ordinary saves cannot rewind it.

Validation: `test_generated_artifact_source.py` passes 23 methods across file and
real PostgreSQL (including the four real coordinator entrypoints, same-text/Keep/
Clear, Undo/reparse/reimport, rename round trips, concurrent late/new writers and
file journal recovery). The two original late-summary assertions change from red
to green. Atomic artifacts remain 18/18; the original 535 executions are 529 pass,
zero failures/errors and six unchanged private-source opt-in skips. The two old
summary/quiz cache scripts now use hierarchy/epoch/current-layout fixtures and pass
all original assertions; merge-only artifact updates are not used to clear fixture
state. Lifecycle tests pass 43/43; all 120 real HTTP rename cases (including TCP)
pass. Evidence and suite-level overlap accounting: `/tmp/struct-fix-2c897af/`.
Two full immutable-backup copies preserve canonical text/index slices and reject
late results; deterministic providers are not live LLM or OCR semantic acceptance.
Local verification only; external approval of the new review SHA is still required.

## STRUCT-02/03 manual reparse correction (2026-11-13)

Implemented current contract (supersedes the historical manual override notes below):
manual drafts preserve every canonical character through ordered, non-overlapping
sections. Unlisted chapter introductions and section gaps become implicit sections;
text outside explicit chapter ranges becomes implicit chapters. Validate complete
coverage before any persistence, for both character and validated page anchors.

Manual commit requires server-owned accepted-source evidence. Its conditional
repository port loads and validates the current incarnation, structure version,
reading revision and logical source identity under the file lock or SQL transaction.
It retains only accepted extraction/page/OCR evidence and source metadata, not old
reading overlays or interaction artifacts. A new hierarchy advances its version
and incarnation. Stale or missing evidence fails closed; unaccepted sources need
explicit preparation before commit, while preview remains non-mutating. No client
provenance is accepted. Same canonical text retains the existing valid QA index.

Validation: `test_manual_structure_integrity.py` passes 16 methods / 56 matrix
subcases across file and real PostgreSQL. Covers char/page gaps, TXT and accepted
PDF Fast/Auto/OCR, zero-generation QA/manual/page reads, source replacement/deletion,
version/incarnation/reading conflicts, forged draft provenance and incomplete drafts,
and publication/persistence failures with old-data preservation and successful retry.
The four original external manual assertions change from failing to passing. Two
immutable-backup book copies preserve 652001/147401 characters and reuse 1527/298
exact FAISS record slices after manual reparse. Evidence: `/tmp/struct-fix-2c897af/`.
PDF mode/OCR provenance cases use deterministic accepted evidence, not live OCR.
STRUCT-01–04 are now locally verified; external review of the new SHA is pending.


## STRUCT-01 atomic artifact updates (2026-11-13)

Implemented unit correction: dedicated section/chapter/task-unit artifact, task-layout and
metadata operations acquire the current document inside one owner write boundary,
validate targets, merge only supplied fields and persist before releasing it.
File uses the existing per-document lock and final atomic replacement; PostgreSQL
uses the store's row lock and the same connection for read/merge/save. Model calls
and layout computation stay outside this boundary. Combined artifact updates merge
non-null kinds; explicit clearing remains with content/reparse/annotation owners.
Layout replacement retains current artifacts for surviving identical unit identities
and content; it cannot restore stale artifact payloads supplied by a layout worker.
The final file rename is the publication point; SQL errors before transaction commit
roll back all participating writes. This does not add a cross-resource transaction.
Acceptance covers concurrent same/different sections, chapter/unit kinds, metadata/
layout, persisted return values, and real file/SQL failures. STRUCT-02/03 manual replacement and STRUCT-04 source binding supplement this
boundary; atomic latest-state merging alone does not validate a delayed source.

Validation: `scripts/test_atomic_hierarchy_artifacts.py` passes 18 file/real
PostgreSQL cases. The same paused-read concurrency assertions fail on both baseline
adapters and pass after the correction. 29 existing suites execute 535 tests:
529 pass, zero failures/errors, six private-source opt-in skips. The two task-layout
fixtures now explicitly prepare before reading and use the current resolver version;
all original hierarchy/artifact/cache assertions remain. The legacy summary/quiz cache scripts originally failed on both baseline and
unit-1 code due to root-sections-only fixtures. STRUCT-04 modernizes those fixtures
and current merge semantics, preserving all original assertions; both now pass. Evidence: `/tmp/struct-fix-2c897af/`.
This completes only the local STRUCT-01 unit, not external approval of STRUCT-01–04.


## CONFIG-01 explicit identities (2026-11-12)

File repository reads use the common lossless name resolver. Exact qualified names
remain separate; legacy aliases require a unique compatible owner. Explicit imports
use their own exact names rather than aliasing a new document onto another source.
Ordinary snapshot saves keep incarnation checks under the exact artifact lock.
Lifecycle journal paths and locks always use exact canonical names; a retired bare
identity cannot be rebound to another qualified book.
Ambiguous legacy identities remain visible in discovery and fail closed on access;
resolution never migrates or rewrites files. Path/lock containment remains required.

## SCHEMA-01 storage boundary (2026-11-11, implemented)

StructuredDocumentArtifactRepository validates the supplied logical name, preserves its source suffix, and requires resolved
artifact/lock paths to remain below its resolved configured base directory.
Existing symlinks escaping that root are rejected; discovery skips unsafe entries
before reading their contents. Save, reparse, load, delete and rename use this guard.
The low-level StructuredDocumentStore still accepts explicit internal file paths;
it is not a public name-to-path router. Ordinary-save incarnation fences remain
unchanged; explicit reparse is required when a test replaces an entire document.
This guards request-controlled paths and pre-existing links, not hostile concurrent
OS-level link replacement outside cooperating repository ownership.

## Incarnation comparison before annotation preservation (2026-11-11, implemented)

Ordinary save compares incoming and persisted interaction_epoch while holding the source write lock, before preserving annotations or writing anything. A mismatch, including a missing incoming epoch against an identified document, raises ReadingContentConflict. Legacy documents with no epoch on either side remain writable; loaded current snapshots retain their identity. Explicit initial import/reparse remains the path for a new incarnation. The shared annotation-preservation guard also applies within the SQL transaction. An epoch-less metadata worker must reload instead of adopting the current incarnation. Regressions compare the complete new-book payload after rejecting a snapshot retained across rename and reimport.

## Main rename integration ports (2026-11-11, implemented)

Ordinary metadata/artifact saves carrying an existing document epoch must reject
a now-missing identity instead of recreating a renamed/deleted document. Explicit
reparse/import remains the creation path, preserving the R3 reimport contract.

The active artifact repository exposes its lifecycle storage identity and an
explicit matching rename recovery port. File recovery verifies old/new IDs under
the existing preference lock before applying a journal; it cannot recover another
operation accidentally. PostgreSQL uses its existing transaction and requires no
file recovery. Canonical rename still delegates to installed file lifecycle or SQL
store; source epochs, old-name tombstones, migrated references and late-writer
fences are unchanged. These ports let the main application coordinate optional
raw/FAISS moves without bypassing persistence owners.


## Shared sentence translation (2026-11-09)

The learning repository also owns aligned ordinary translations. Its bounded
read_translations projection returns only validated complete TranslationKey results
for the requested current source block, with the successful operation metadata.
Pagination uses ordered cache IDs; stale source versions are excluded without writes.
File and SQL share projection behavior. Translation-purpose commands validate exact
source/registry keys independently of native-language/mode settings; hints cannot
use this purpose. Existing hierarchy/epoch invalidation, rename/delete migration,
CAS/dispatch fences and checkpoint/result consistency continue to apply. Aggregate
paragraph strings are derived by section_tasks and never duplicated into document
artifacts. Historical paragraph-only artifacts remain unchanged and readable.

Validation: 470 backend tests and actual API/React mode-switch acceptance pass; detailed evidence and limits are recorded in the shared section_tasks design.


## Vocabulary assessment persistence (2026-11-09)

Hints `BatchPayload` and `LearningHints` carry optional shared assessments. Both
file and SQL adapters use the same typed codec and guarded repository behavior.
Checkpoint digests include present assessments; final commit must exactly match
the durable batches' aggregated assessment as well as items and provenance. A
forged summary is rejected before replacing the saved result. Source/settings,
fence/CAS and transaction boundaries are unchanged; no SQL migration is needed.

Legacy encoded batches/results without assessment decode with None. Encoding None
omits the field, preserving historical checkpoint bytes/digests rather than causing
read-time repair. Codec state validation also checks saved-vs-checkpoint assessment
coherence. New all-filtered results remain persisted ready with explicit omissions;
legacy missing reports remain unknown. File/PostgreSQL regressions and actual
API/SQL real-book output replay cover save/reuse/restart, commit mismatch and old
digest compatibility; see section_tasks' 2026-11-09 validation evidence.

## Language Learning Reading Mode Plan (2026-10-28)

**[Code-Confirmed foundation; LLRN-06/12 complete, 2026-10-28]** [Canonical design](../section_tasks/module-detailed-design.md#language-learning-reading-mode-plan-2026-10-28)；[task dependency queue](../section_tasks/module-checklist.md#language-learning-reading-mode-queue-2026-10-28)。已交付 hierarchy-aware source resolution、storage-neutral repository port 與 POSIX file adapter；db 已完成 LLRN-08 SQL adapter；LLRN-12 已交付來源失效與 document lifecycle ports；LLRN-13 application orchestration 已完成；LLRN-15 runtime composition 已完成，LLRN-16 API/lifespan 已完成，UI 仍待後續任務。

### Learning repository and source ownership

LLRN-15 組裝啟用 file lifecycle 後的回歸補正：`FileLearningRepository._write_context`
取得 owner preference snapshot lock 後先拒絕 deleted tombstone，避免 rename/delete
先完成時遲到 writer 再開啟已移除的檔案。返回 `learning_document_missing`，不轉成
storage missing/recreate；鎖序與讀取行為不變。既有 concurrent rename test 及新增
container deterministic late-write test 已在 file/SQL 驗證。

- `learning_repository.py` 定義 `LearningRepository`、來源分頁/read 結果、明確的 partial `BatchPayload` 與受驗證的 `OperationWrite`。Scope 由 server 注入 `WorkspaceScope`；document ID 必須與已存文件精確一致，拒絕名稱 alias、跨 workspace/document reference。沒有 client parent IDs 或 raw source 寫入入口。
- `learning_source_resolver.py` 從 `chapters[].sections[].task_units[]` 解出 ownership，重用既有 `project_document` / `original_blocks` 的 effective reading projection，包含已套用正文/OCR/cleanup overlays；不拿原始 OCR text 冒充 effective text，不讀 root sections/structure_nodes。拒絕重複 hierarchy/block IDs、矛盾 parent IDs、已被投影的持久化輸入；已刪 block 不復建。
- Source binding 帶 scope/document/chapter/section/unit/block、既有 structure version、interaction/document epoch、block content epoch 及 exact effective hash。Sentence source/resource reference 必須與當下來源相符；`save_sentences` 驗證 exact projection 後才存，不自行切句、prepare、修正文句或寫 profile。相同 resource 重存不寫檔。
- `read_sources` 是 backend 來源分頁；cursor 綁 namespace/document、原來源 token 及 reading revision，正文 overlay 改動後拒絕舊 cursor。`read_batch` / `read_sentences` 接收最多 100 個 keys/references；來源及 checkpoint 分頁 limit 為 1–100。這是查詢數量上限，不是模型 token/字數上限。Read 只返回請求範圍，不生成、續租、補檔、遷移或清 cache。
- Saved reads 用當次 source snapshot 驗證來源、句子與完整結果；已失效的 key 返回 `stale`，無對應已存結果才 `missing`。Corrupt/unsupported/IO failure 明確拋錯，不轉 missing。`lookup_command` / `lookup_operation` 是保留的歷史狀態，不證明舊 source 仍可顯示；`read_checkpoints` 另驗目前 source，過時 partial 不返回。讀回應仍需後續 controller 核對 request/source identity。

### Application integration ports (LLRN-13)

`resolve_blocks(document_id, block_ids)` 一次 authoritative source snapshot 解析最多 100 個指定 IDs，保留請求順序、拒絕重複／缺失，不存切句。回傳有界，但底層仍載入整份 hierarchy，不能聲稱 SQL source row pagination。`inspect_operation` 返回 durable operation 與目前 entry revision；歷史 operation 已被取代時 revision 為空，不能恢復成 active owner。

Claim／guarded operation writes 接收 internal `validate_binding` callback，在既有 preference → document → learning 鎖／短交易內，以已鎖定設定重驗有效 profile/mode/proficiency；callback 不取得 owner locks、不寫入或呼叫模型。Dispatch 以原 durable command 再評估相同 admission policy，拒絕時不產生 provider permission。顯式 `invalidate` 只在實際 binding mismatch 時進行 domain stale transition，儲存錯誤不能當作 mismatch。這些行為由 file/SQL 共用，沒有新 persisted schema。

專測與 rollback/concurrency/lost-ack 證據見 [LLRN-13 learning design](../learning/module-detailed-design.md#llrn-13-implementation-contract-2026-10-28)。Read source stale 與 operation stale 保持區分，只有來源有效的舊操作可走明確 retry。

### Shared behavior and SQL coordination (LLRN-08)

`learning_repository_behavior.py` 抽出兩種 adapter 共用的 source/scope/read bounds、claim/replay、設定保護、operation transitions、checkpoint/final assembly；仍直接呼叫 section_tasks 的既有純策略，不複製兩份演算法。Storage adapter 只提供 protected `_read_context` / `_write_context` hooks：讀取一個已驗證 snapshot；寫 context 在來源與偏好鎖內傳入 state 及 transaction-local persist callback。File 保留原單一 sidecar replace，原有 45 項 file cases 全數回歸通過；不新增公開任意 state 寫入 port。

`learning_file_codec.py` 保留相容檔名與 schema=1，typed codec 同時供 SQL 單筆 payload 使用；SQL 僅組裝請求 keys 的必要依賴集合來驗證，不把 file 全書 snapshot 搬入 JSONB。`LearningCommitUnknown` 同時表示 file replace 或 SQL COMMIT 結果未明，均需 reconcile。PostgreSQL store 新增公開 `load_in_transaction(target, connection=...)`，只借用 caller-owned active transaction 讀取既有 canonical hierarchy，不自行 commit、加第二筆交易或查 profile/LLM。

LLRN-08 的 SQL 細節與 47 項真 DB 驗證見 [db design](../db/module-detailed-design.md#implemented-learning-sql-persistence-llrn-08)；兩 backend 沿用相同結果／來源語意。LLRN-12 在此基礎上補齊來源 epoch、rename references 與偏好協調（見下節）。既有 file 歷史驗收證據如下保留。

### File layout, codec and locking

`FileLearningRepository(base_dir, documents, preferences, admission=...)` 只依賴兩個 file owners 的公開協調介面及 LLRN-05 domain policy。Scope 必須與 document repository 的 namespace 相符。Constructor 及所有 reads 不建立 directory、lock、預設 snapshot 或 temp；write/recovery 才可建立所需目錄/lock。

每本書使用 `(workspace kind, namespace, document_id)` SHA-256 命名的獨立 `.learning.json`，schema_version=1。Snapshot 保存一次 sentence resources（spans/reference，不複製原文）、各 component/level cache entries、operation history、跨 key 的 command receipts，以及 operation-bound checkpoint payloads。正文/document metadata 不塞入學習結果；translation key 不含級別，hints 不依某次翻譯修訂。相同 sentence resource 可被多語/多等級共用。

`learning_file_codec.py` 只允許明確列舉的 dataclass tags 與精確欄位/型別，沒有動態 import、任意 class instantiation、coercion 或 read-time migration。檢查 scope、resource/key hash、operation/fence/history/claim receipt 關聯、checkpoint digest/句子 partition、saved result 與已成功 operation/checkpoints 的一致性。拒絕壞 JSON、重複 JSON key、未知 schema/type/欄位及缺失 receipt；不能因 receipt 遺失而無聲恢復執行權。Result model provenance 與所有 batches 一致，final payload 必須等於已存 checkpoints 的 ordered assembly。

所有學習寫入採固定鎖序：**workspace preferences → document artifact → learning sidecar**。`FileReadingPreferencesRepository.locked_snapshot(document_id)` 在既有 preference flock 內返回同一版 workspace/book；`StructuredDocumentArtifactRepository.locked_document(doc_name)` 在既有 source lock 內返回 authoritative document，兩者均不保存內容、不允許在 context 內重入自己的寫方法。Learning file 再使用穩定的 POSIX flock；source/settings writers 使用原有鎖，因此不能穿過 claim/commit 的 revalidation 與 snapshot 替換之間。LLRN-12 lifecycle 協調沿用同一鎖序。

寫鎖內核對 actual source、book/workspace revisions、active mode、native/source language、儲存的 learning level（hints）及 registry。這是 persisted setting guard，不是 LLRN-09 effective settings/profile service：原文 fallback 到既有 profile、完整 eligibility 仍須由 LLRN-09/13 在 backend 解析，不把 preference 中的 language_code 當 parser/profile 真值；本 adapter 不偵測語言或重建 profile。

### Claim, operation and commit behavior

- `claim` 在鎖內執行 LLRN-05 `plan_request`，同 key 跨 command/process 合併；同 document command ID 跨 components 唯一，原 receipt 永久保留於本期 snapshot。相同命令重放只返回原 receipt；衝突命令拒絕。新的 entry、operation、sentence resource 與 receipt 只經一次 snapshot replace 提交；拒絕/reuse/coalesce 也保存明確 receipt，不產生 provider work。
- `write_operation` 只接受 typed `OperationWrite`：dispatch、renew、checkpoint、commit、fail、expire，各 action 拒絕多餘/缺失欄位。先驗 entry revision、operation ID/fence；dispatch/checkpoint/commit/renew 另驗當下 source/settings。舊 writer 或重複 dispatch CAS 拒絕。成功返回 dispatch 的 caller 才可在鎖外送一次 batch；lookup running 不能當作再次送出的許可。
- `BatchPayload` 明確區分 translation/hints，並不冒充 complete result。翻譯 batch 必須完整對上 plan 指定 ID 順序；hints 必須位於該 batch 的句子/原文 span，允許 empty hints。Adapter 再驗資料型別/定位、prompt provenance，將 payload 與 marker 一起提交。LLRN-10/11 仍負責 provider/schema/prompt、模型容量分批、語義與混語適用性。
- `commit` 重驗完整 source/key、source/settings guards、base result revision 及 ordered checkpoint assembly，saved output 和 succeeded operation/result receipt/history 一起替換。新結果未成功前保留舊 saved；翻譯與 hints 獨立，失敗不清其它 component。
- `fail` / `expire` 可記錄已知失敗或失聯，即使偏好或 block 已改變；它們不安裝結果或允許 provider IO。Unknown 不自動接管；明確 retry 沿用 LLRN-05 的 consent、目標 operation 與新 fence。相同 fence 找回的合法結果可 checkpoint/reconcile；來源或設定已變則拒絕。來源失效及 document delete/rename 的協調見下節 LLRN-12；歷史 operation 不等於目前可用輸出。

### Durability and recovery boundary

Sidecar 提交使用同目錄 temporary file → file fsync → atomic replace → directory fsync。只寫**一個 learning snapshot**；source/preferences 在整段保持鎖定但不寫入，因此這不是三個檔案的寫入交易。Replace 前已知 IO 失敗保留舊 snapshot；replace 開始後錯誤回報 `LearningCommitUnknown`，先查 durable receipt/operation/result，不盲重送 provider。Claim/dispatch/checkpoint/commit 的 lost acknowledgement 都能從正式 snapshot 核對。

程序中止遺留的 temp 永遠不是 read source。`recover(document_id)` 是明確維護命令：取得 sidecar lock、驗正式 snapshot 後只刪除該 snapshot 的 orphan temps 並 fsync directory；不採納較新的 temp、不建立缺失結果、不改 operation 狀態或重新生成。正式 snapshot 損壞時拒絕 recovery 並保留證據，不能拿 temp 猜結果。正常讀取/constructor 不做 cleanup。提交日誌只記 stage、duration、entry/operation counts 與 outcome，不記正文、document identity、keys 或 provider secret。

此版 file adapter 的 query **回應**有界，但為了既有 JSON backend，每次需要讀取該書 structured document/learning snapshot，成本隨書籍及生成歷史增長；不是 SQL 局部查詢或大型資料集吞吐量保證。歷史暫不裁剪，不引入 256 筆或其它猜測生成用量上限；未來 cleanup 必須保留防重放 tombstone/協議。測試證明本機 POSIX 多程序/程序中止與 replace 邊界，未宣稱所有 filesystem、斷電、新建父目錄或分散式鎖的 durability，也不承諾 exactly-once provider billing。

### Implemented source and document lifecycle (LLRN-12)

**[Code-Confirmed, 2026-10-28]** `learning_lifecycle.py` 在 canonical document 寫入邊界比較 effective block text 與 hierarchy ownership。真正改文、刪除、還原都以同一來源寫入推進 opaque content epoch；既有明確 mutation 的新 token 保留，legacy raw save 未給新 token 時補入新值。同文、OCR draft、review Keep/Ignore、僅 cleanup history 修改保留既有 epoch。Keep 只控制普通互動保留，不豁免學習來源有效性。

- 改文原子作廢該 block 的所有 sentence resources、語言譯文、各級別 hints、partial/operation binding；無關 block 保留。採 **持久化 source binding 失效**，不逐筆物理刪除歷史 learning rows/sidecars。已存 payload、checkpoints、command receipts 保留作重放／費用歷史；saved read 返回 stale，source/commit guards 拒絕晚到結果。歷史 operation lookup 可能仍是 running/succeeded，不可當成顯示或派工授權；讀取不做清理或改狀態。
- Undo 即使回到相同文字也使用新 content epoch。普通 metadata save 保留 authoritative document epoch、reading overlay/epochs；延遲 worker 不能帶回舊版本。初次保存／reparse／刪除後重建取得新 document epoch，不復用舊書結果。Source guard 不更改 parser hierarchy 或 profile。
- `rename_learning_state` 遷移全書所有歷史版本／等級、sentence/resource/key references、結果定位、operation binding、batch plans、checkpoints/digests、receipts。譯文、提示意思／原文 span 與 provenance 不重生、不改寫。Shared `SentenceResource.with_document_id` 重算 deterministic IDs；沒有實存 resource 的 denied/no-content receipt 只遷移 source identity，保留舊 opaque digest 作歷史墓碑，不能作可用 resource。Entry CAS revision 推進；active operation 轉 stale，未確認費用保留。舊 writer 不可寫入改名後的資料。
- 刪書移除該書全部 learning 資料，偏好留下 revision tombstone；workspace 母語不變。改名同時搬移設定、舊名稱留下 tombstone，既有 destination 拒絕，不覆蓋別本書。

#### File coordination and explicit recovery

`FileLearningDocumentLifecycle(documents, learning, preferences)` 明確安裝到 canonical file document repository，constructor 不寫檔。三個 owner 必須同 scope、同注入實例。來源 edit/reparse 仍是單一文件替換；rename/delete 使用 **preferences → sorted document locks → sorted learning locks**，三個 owner 各提供公開 lifecycle staging/path/lock 介面。

跨檔案寫入前持久化 workspace `.preferences.lifecycle.json`，只包含 owner 推導的文件內容與 before SHA-256，不接受任意路徑。各檔案 temp/fsync/replace（刪除則 unlink）及 directory fsync 後才清除 journal。中斷或 acknowledgement 不明返回 `LearningCommitUnknown`；有 journal 時相關 source/learning/preferences reads/writes fail closed，不把部分狀態當成功或 missing。明確 `recover()` 在同鎖序內驗證 journal schema、各 owner payload、before/desired hash，向前完成同一操作；重複 recovery 無副作用。壞 journal／外部衝突保留現場，不猜測回滾、不自動付費重送；reads 不修復。

這是合作 owner 的 POSIX 多檔案恢復協議，**不是瞬間跨檔案原子 replace、跨 filesystem/SQL 交易或硬體斷電保證**。LLRN-15 Runtime 已組裝 coordinator，pending journal 須由明確維護操作呼叫 recovery；啟動與 reads 不自動修復；未注入的獨立 legacy file repository 不具有此跨檔案協調。一般 reads 仍不建立鎖檔／補寫。全書 rename 需載入該書完整歷史，成本隨歷史量增加，非有界 saved-read 路徑。

#### PostgreSQL coordination and verification

SQL source mutation/cleanup 的 epoch 與正文在同一交易保存；初次／reparse epoch 由 store 建立。`db.learning_lifecycle` 在 rename/delete 先取 preference workspace lock，再取 document/learning locks；同一 connection 遷移／清除 learning 及偏好，任何階段失敗全數 rollback。Rename 五個集合查詢載入全書歷史再重建 references，沒有每筆獨立 commit；一般 saved reads 仍依 key 有界。Cleanup 僅更新目前 structure version 的 snapshot，避免重解析後選到歷史快照。未 migration 的舊 store 不在此補建表，部分 schema 不合法時拒絕。

專測 `scripts/test_learning_lifecycle.py` **43 tests passed**，覆蓋 file/真 PostgreSQL 的 no-op、Keep/Clear、多語／B1/C2、無關 block、OCR draft/apply/Ignore、cleanup removal/restore/history、Undo、legacy save、reparse、改名歷史與非空 hints、刪除／重建、並行 late writer/偏好保存、SQL rollback，以及 file 程序退出／新實例恢復／壞 journal／外部衝突。既有 reading/cleanup/task-layout/notes 回歸亦驗證。只使用 temporary fixtures 與 disposable PostgreSQL，未改 parser／生成策略、未新增真 provider 呼叫或修改使用者書籍；LLRN-10/11 語義品質未通過的紀錄不變。LLRN-13 application orchestration 已完成；LLRN-15 runtime DI 已交付，LLRN-16 production API 已交付，UI 仍待後續任務。

### Verification (2026-10-28)

`scripts/test_learning_file_repository.py` 最終 **45 tests passed**。包括真檔案 reload、純讀零寫入、來源/overlay/parent/scope、分頁與 cursor revision、same-ID replay/history、6 程序同 key（1 claim/5 coalesce）、4 程序 dispatch（1 success/3 CAS conflict）、source/preferences 寫入競爭、replace 前後故障注入與 `os._exit`、lost ack、orphan recovery、receipt/codec corruption、checkpoint/result 原子性、刷新保留、空 hints、未知結果/manual retry fencing，以及不洩漏正文的提交日誌。

先執行本 suite（當時 44 項）加 LLRN-01..05 基礎及既有 reading_content/source_review_blocks：**206 passed**；最後補上日誌驗證後重跑完整 45 項專測通過。共 **207 個不同測試**有通過證據（45 新增 + 162 既有），無 skips。全部使用 temporary fixtures，不呼叫真 LLM、不修改真書或重啟部署；未改 parser/切句演算法，沒有新增真書/provider acceptance。SQL/全來源生命週期/實際 API/UI 驗收仍由後續 owner 完成。

## Irreversible Ignore All OCR Flags (2026-10-27)

**[Code-Confirmed, 2026-10-27]** Add discard_all_regions to the existing reading-content mutation API.
Under the repository transaction, validate source/revision, OCR eligibility and
resolve all current pending regions server-side. Reject unexpected target/text/
interaction fields. Record all IDs as permanent ignored decisions, outside the
Undo snapshot, and in the audit with one idempotent receipt/revision. Existing body,
original source, hierarchy, artifacts and cleanup history stay unchanged. Preserve
the preceding Undo operation, but exclude this action from undo_action and ensure
any later Undo cannot reopen permanently ignored regions. Single-region Ignore
retains its existing reversible behavior. Empty pending sets reject a new command;
replaying a committed mutation returns the authoritative snapshot. No schema
migration, OCR rerun, model call or loop of independent region mutations.


Validation: discard_all_regions validates OCR/source/revision and commits permanent decisions outside Undo. 64 focused file/API/disposable-PostgreSQL tests pass, including rollback after document write, concurrency, idempotency, all-page scope, previous/future Undo and unchanged source/body. Source-review/repair/cleanup regression runs 204 tests with 202 passing and 2 opt-in private-backup cases skipped. Local API rebuilt and health/OpenAPI verified; no live document flags changed.

## OCR Review Availability Projection (2026-10-27)

**[Code-Confirmed, 2026-10-27]** Reading-content snapshots expose `ocr_review_available`, a read-only
presentation capability. Require a valid PDF extraction identity matching canonical
text. Full OCR is eligible; Auto requires matching actual OCR spans or source-bound
OCR review regions (including empty OCR regions). Native/fast, non-PDF, unknown,
invalid or stale extraction evidence is ineligible. The capability is independent
of unresolved counts, so resolving/ignoring all regions preserves access to history.
Reuse extraction identity and OCR eligibility contracts. Do not perform extraction,
write metadata, or change body-edit permissions, mutation guards or task-layout.
Validation: 17 focused availability/reading-content tests pass, including strict API
schema, snapshot purity, stale evidence, empty OCR regions and legacy full-OCR identity.
Local API rebuilt and healthy; 12 existing document reads returned explicit true/false
capabilities, including completed OCR documents without outstanding regions.


## Source-Dependent Interaction Lifecycle Plan (2026-10-26)

**[Code-Confirmed]** TRUI-02/03 implement
[persisted block translation](../section_tasks/module-detailed-design.md#persisted-block-translation-plan-2026-10-26).
reading_content.mutate now validates and compares effective text inside the write
boundary. Same-text edits store only neutral replay receipts, without body revision,
flags, Undo/audit changes or SQL document timestamp changes. mutation_refresh exposes
body_changed, summaries_invalidated and translation_block_ids. Actual body changes
share clear_source_dependents across edits/repair/cleanup/Undo; summaries use the
document-wide summary-kind fallback while translations use exact changed block IDs.
Keep preserves other kinds; legacy saves preserve current retained artifacts and
cannot resurrect cleared translations/summaries. Per-block content epochs are not
rewound by Undo. File/SQL/real-App acceptance is included in 408 passing regressions
and the TRUI private report linked from the shared design; no profile mutation.

### Same-Text Gate

After identity/ownership/authorization and expected revision/source checks, resolve
current effective body inside the repository mutation lock/transaction and compare
the submitted text in the same representation. The client may avoid redundant work
but is not the authority. Use current overlay/cleanup-applied content, not original
OCR or a previously loaded editor snapshot; reject stale requests before comparing.
Do not trim/collapse punctuation/whitespace to invent equality. Byte-identical text
must return a truthful no-change result without advancing reading revision, manual
flag, Undo, change timestamps or interaction epochs and without artifact cleanup.
Keep existing operation replay guarantees; an optional receipt must be content-neutral.
Expose enough no-change information for targeted UI settlement without layout reload.

Apply the comparison to main-reader/source-review block Edit and body-applying repair
paths. Draft Save is not a body mutation. A repeated cleanup Restore or region Apply
that yields the same effective body must not trigger body-dependent cleanup; draft/
history acknowledgements can retain their separate existing semantics. Do not loosen
manual-edit/source-review protections or permit deleted-block updates via equality.

### Selective Atomic Cleanup

**[Code-Confirmed clarification, 2026-10-26]** `reading_refresh.changed_blocks`
compares effective projected content before/after the mutation. For every changed
or deleted block, `reading_content_artifacts.clear_source_dependents` removes its
entire `metadata.translations[block_id]` entry, including all target languages and
replay receipts, in the persisted mutation. This is not merely stale marking or UI
hiding. Unchanged blocks retain their translations; identical-text submission does
not trigger this cleanup under either Keep or Clear.

On actual change, always remove changed-block translations and affected summaries,
even when Keep other metadata is selected. On block deletion remove all that block's
interactions including Note, and affected ancestor summaries. Shared summary cleanup
must cover public_summary, ancestor_summary_artifacts and derived TaskArtifacts.summary
resources, without discarding TaskArtifacts.quiz or unrelated fields under Keep.
Cover descendants that consumed invalidated ancestor summaries. Use precise verified
dependencies where available; otherwise clear summary-kind resources document-wide,
never the entire document's artifacts merely to clear summary fields. The fixed
document profile and its summary remain untouched under the existing profile policy.

Keep existing user-confirmed clearing scope for other kinds. Retention is not freshness
certification: do not stamp retained outputs with new source provenance. Clear modes
and source repair must share one policy helper rather than inventing per-UI variants.
Any changed text produced by OCR cleanup/Restore or other body mutation obeys the same
rule; unchanged review Keep/Ignore and draft saves do not invalidate translations.

Commit body, revision/flags, cleanup and necessary audit/epochs together via existing
repository boundaries. No model calls in the transaction, no read-time cleanup/write.
Legacy saves and delayed generators must not resurrect purged records; test deletion
followed by Undo and edit followed by restoring old text. Keep different-language
translation records on unaffected blocks and preserve unrelated metadata.
Return actual body-change and affected interaction scope to UI; frontend does not
infer invalidation from HTTP success or a global refresh flag.

In particular, `translation_block_ids` may be nonempty while
`generated_invalidated=false` under Keep. The UI closes the affected translation
surface and cancels its request using this explicit scope. A late generation also
fails backend `translation_records.commit_translation` binding/revision checks;
frontend cancellation alone is not the persistence safeguard. Undo or restoring
earlier text does not restore purged translations, and no automatic retranslation
runs on read. The user must explicitly request translation for the current body.

Evidence: `reading_content.py` (same-text gate and atomic policy application),
`reading_content_artifacts.py`, `reading_refresh.py`,
`../section_tasks/translation_records.py`, and
`../scripts/test_translation_persistence.py` (same-text, Keep/Clear/delete,
unaffected-block retention and generation after change). This clarification records
the existing implementation; the documentation update does not change runtime behavior.

## Block-Level Source Review Correction (2026-10-26)

Maintainer correction supersedes the region-wide editing boundary below. Source
regions group related blocks; writes select exactly one verified current block.
Do not reject a valid selected block because a sibling is protected, the full region
is partially covered, or the concatenated region exceeds the human-edit limit.

`source_review_block_repair` owns fresh region/block relation validation and
per-block eligibility. Reading-content commands `review_block_edit`,
`review_block_delete`, `review_block_keep` require region_id + block_id, source token,
revision and mutation ID. Edit requires nonblank text <=10,000 Unicode code points
and explicit discard_interactions. Delete/Keep forbid text and interaction choice.
The repository revalidates everything under its existing lock/transaction.

Reuse the existing block overlay and artifact/epoch/receipt transaction. Backend-only
overlay origin `source_review` distinguishes these edits from main-reader edits;
the client cannot supply it. Durable main-reader flags and delete/Undo audit still
prohibit review Edit/Delete. Review edits remain repeatable and do not set that flag.
Legacy source-span repair and cleanup Restore must not overwrite block overrides.
Canonical hierarchy and source evidence remain unchanged; no schema migration.

Keep it stores a durable `review_kept_blocks` decision outside Undo. It does not
delete body text, source evidence, Notes or generated artifacts, and cannot be used
to bypass an edit restriction. Exclude the block from all review groups; when every
verified block in a region is confirmed, that region leaves pending review. Mapping
and canonical evidence remain retained. The decision is scoped to current canonical
block identity and does not disable ordinary main-reader Edit. Do not conflate Keep,
Delete body text and Ignore region.

Read DTOs carry per-block `review_repair` capabilities/reason. Mutation responses
carry the selected block and existing targeted refresh/tombstone binding; Keep has
no affected body blocks or generated invalidation. Source review reconciles kept
membership without rebuilding layout. Replays include current capabilities.

Tasks: backend SRBLOCK-01; UI/incident acceptance SRBLOCK-02 in the UI checklist.
Historical SRREPAIR receipts do not count as this corrected acceptance.

Validation: 400 backend tests pass without skips, including 15 new single-block
file/API/SQL cases and post-write rollback checks. Four private book backups and
the saved Madame Bovary snapshot remain unchanged. Actual App/API/SQL acceptance
on Page 3 is owned by UI SRBLOCK-02; private final report is
data/test-books/srrepair07/20261026T044959Z-7b1702d7. Current API was rebuilt through
the existing Compose/env startup path, without restarting PostgreSQL or the UI.

Governance receipt: inspected AGENTS, proposal, HLD, module index, progress, both
standards, UI/document_structure design and checklists, and api_schemas design and
checklist. Modified those two owned design/checklist pairs and api_schemas design
to replace region-wide UI authority with single-block commands and Keep semantics;
historical implementation evidence is retained and labelled superseded. Backend
progress synchronization is separate and excludes UI completion. Checklists carry
Timestamp: 2026-10-26; source, API, transaction/browser tests and live read-only
inspection support the claims. No source/model/parser authority or hidden read
mutation is introduced. Validation itself changes no files; unrelated edits remain.

## Guarded Source Region Repair (2026-10-26)

Historical region-wide implementation. Its public commands remain for compatibility;
the active Source review UI uses the corrected block-only commands above.

Maintainer-approved successor to the read-only SRMAP workflow. Execution order and
owner checkboxes are in [SRREPAIR](module-checklist.md#guarded-source-region-repair-2026-10-26).
This is a new seven-task track, not a reopening or re-count of completed SRMAP/RCR
acceptance. SRREPAIR-02..06 now implement the guarded writes, typed API and UI;
the mapping-only receipt below remains historical evidence for SRREPAIR-01.

### Ownership And Coordinates

Reuse the source-review resolver for exact canonical region/block intersections.
Complete region coverage never means ownership of whole associated blocks. Return
hierarchy IDs, canonical absolute half-open spans and block-relative **canonical**
offsets; offsets count Python Unicode code points, not UTF-16 or normalized display
characters. OCR whitespace normalization must not turn source offsets into DOM offsets.
Validate extraction identity/hash, hierarchy and block source evidence. Ambiguous,
partial, overlapping or unmaterialized block coverage is not writable. Verified
whitespace separators between blocks may be included without inventing a block.

The domain owns eligibility and repair planning; repositories own the transaction;
routes expose only needed DTOs/actions; UI consumes authoritative state. Source
evidence stays immutable and reads never backfill, prepare or regenerate profiles.
No manually created blocks, loader-assigned block IDs or client-side text matching.

### Editing, Application And Protection

- Keep original OCR, saved repair draft and last applied replacement distinct.
  Saving a draft does not change body text; explicit Apply changes the exact region
  span. Reapplication replaces the same region's prior replacement, never appends.
- Region repair never sets or clears durable body `block_edit_flags`. Any affected
  block manually edited in the main reader prohibits region Apply/Delete, including
  after Undo. Manual deletion also prohibits both operations and cannot recreate a
  block; retain audit-based protection after a deletion is undone.
- Delete means remove the region's mapped body text, not its evidence/history or
  entire related blocks. Ignore remains a separate explicit review decision without
  body deletion. Keep repair evidence for accidental deletion; no automatic Ignore.
- A cross-block operation validates every affected block and commits all or none.
  A whole block removed from reading also loses its interactions, including Note.
  Edit asks whether to clear interactions; retained generated data must obey existing
  stale/invalidation rules. Surviving-block Notes follow the explicit choice.
- Store region-owned application proof separately from cleanup-history ownership.
  Share low-level source/revision/manual-protection/overlap helpers where identical,
  not a second implementation of cleanup policy or an untrusted caller-origin bypass.
  Only an exact proven own prior replacement can be superseded. Foreign region,
  cleanup-history and manual overlays conflict; no silent overwrite.
- Use the existing repository mutation boundary: source token, reading revision,
  idempotent command identity, current eligibility, overlays, audit, interaction
  changes and receipt commit together. Check again under lock, not only in preflight.
  Unknown outcomes require same-command reconciliation; abort is not rollback.
- Cross-block replacement appears once at its canonical start using the existing
  source-overlay projection. Do not distribute replacement prose heuristically or
  create new IDs. Empty-block visibility and artifact removal require explicit tests.

### Implemented Repair Contract

`source_region_repair` owns fresh intersection resolution, eligibility, draft and
owned-overlay staging. `reading_repair_guards` shares manual protection, overlap
and exact span comparison with cleanup history; neither caller gains a bypass flag.
`ReadingContentService.mutate` owns command orchestration and the existing repository
port owns the lock/transaction. No OCR/LLM, new quota, migration or source rewrite.

The existing reading-content POST accepts three explicit region-only commands:
`region_edit` saves a nonblank draft; `region_apply` applies explicit text or the
saved draft with mandatory boolean `discard_interactions`; `region_delete` removes
the exact source range without text or interaction-choice fields. All require region
identity, source token, expected reading revision and mutation ID. Block/cleanup IDs
are forbidden. Explicit edited text is limited to 10,000 Unicode code points; this
is an existing human-edit limit, not a new LLM cleanup quota.

Reading-state JSONB stores `region_drafts` separately from `region_applications`.
Applications bind canonical quote/hash, exact block intersections, replacement and
UTC time; source overlays carry `source_region_id`. Reapplication replaces only the
single exactly proven owned overlay, including an empty deletion or restoration of
the original quote. Draft saves preserve body, artifacts and Undo. Applications join
source overlays in Undo; drafts, manual flags and interaction epochs are not undone.
Cleanup staging also includes region applications in its Undo snapshot.

Under the same transaction, all affected targets are checked before projection and
artifact changes. Removed blocks and explicit Clear choices purge hierarchy-owned
interactions and advance each block's epoch, preventing late Note writes. PostgreSQL
also purges physical artifact rows for the full cleared-ID set, not just one block.
Metadata and structured snapshot persist together; projection or SQL failure rolls
back all changes. Replay returns the durable receipt; stale replay cannot patch body.

Region `repair` exposes only text/edit time, applied text/time, can_edit/can_apply/
can_delete and blocked_reason. Internal offsets/proof remain server-side. Mutation
`refresh` reuses the bounded 20-block DTO/tombstone contract and adds plural
`cleared_block_ids`; all current region statuses are returned by the existing review
snapshot. UI merges targeted body changes and region rows without rebuilding layout.
Older servers without repair capabilities remain read-only. Navigation may remain
available even when mutation is denied; Delete does not imply Ignore.

SRREPAIR-07 uses hash-verified private backups under `data/test-books`, including
`暗水幽灵_sample` and the available Madame Bovary OCR snapshot. Exercise mutations
only in disposable storage, with real API/SQL/browser acceptance and supplemental
fixtures for partial/cross-block/foreign-overlap cases absent from natural evidence.
No production books are deleted or reparsed. Missing real coverage remains explicit.

### SRREPAIR Final Acceptance And Governance Receipt

SRREPAIR-01..07 complete, counted once in their owner checklists (backend 5/5,
UI 2/2). The final combined suite passes 425 tests without skips, including 26
new mutation tests with real disposable PostgreSQL. Supplemental repairs use four
verified backup books and labelled accepted-page/subquote fixtures; no fabricated
claim that the loader naturally flagged those fixtures. The noise backup SHA-256
matches `暗水幽灵_sample.pdf` (3f047af95bc21a8dd370422769a2befb06c8c76f29985b5b55bec07863e42312).
All 16 backup-file hashes remain unchanged. Tests cover foreign overlap, cross-block
all-or-none/physical Note purge, rollback after SQL work, replay/concurrency, Undo,
permanent manual edit/delete protections and source immutability.

`Deep_Reader_UI/scripts/accept-source-region-repair.py` imports saved rows/profile
from private backups into unique SQL schemas and exposes actual FastAPI routes to
the actual App. It cannot proxy arbitrary routes or run preparation/model work.
Desktop 1280 and mobile 390 workflows pass on natural eligible regions of both
暗水幽灵 (101 regions, 38 eligible single-block regions) and Madame Bovary OCR
(38/11). Both repeated replacements occur exactly once; draft Save leaves body
untouched; Delete preserves recovery evidence; response-loss replay does not advance
revision twice; body Edit disables region writes. Every commit checks canonical text,
unchanged relational source rows, manual flags and matching persisted reading-state
copies. All test schemas are removed and all backup hashes are unchanged.

Private reports/screenshots/request counts:
- `data/test-books/srrepair07/20261026T040032Z-963f487a/` (dark-water final).
- `data/test-books/srrepair07/20261026T040631Z-ee8127e8/` (Madame Bovary final).

Browser assertions prove stable connected region rows, scroll/focus retention,
zero region-triggered task-layout/section reads, bounded related/region lists,
targeted hidden-to-visible updates and lazy history revalidation. Editor, history,
reading-refresh, source-review state/navigation suites also pass; desktop/mobile
screenshots are inspected and npm run check passes with the pre-existing bundle-size
warning. Earlier failures caught stale-read display and test selectors outside a
nested modal's accessibility scope; both were corrected. A long acceptance run was
interrupted by development hot reload, then passed against the fixed built preview.
No production book changes, OCR/LLM calls, Docker restart or deployment are claimed.

Inspected markdown: AGENTS.md, proposal.md, high-level-design.md, docs/modules/index.md,
progress.md, both development standards, document_structure/UI/db design/checklists,
api_schemas design/checklist and repository skill instructions. Modified markdown:
this design and UI design replace planned-only behavior with implemented ownership,
API/lifecycle details and acceptance; their checklists record only evidence-backed
task completion; db design records plural atomic artifact purge; api_schemas design
records minimal DTO/command rules; progress.md synchronizes backend-only 5/5 and the
direct recount (84 checked/18 historical unchecked, fenced template excluded).
Every changed checklist retains its final Timestamp: 2026-10-26 line. Global design,
registry, standards and unrelated historical tasks are unchanged.

Implementation validation: scope, architecture, boundaries, tests and memory pass;
no hidden read mutation, parser authority change, new schema or UI-owned backend truth.
git diff --check and task timestamp/count checks pass. Validation itself changes no
files; governance updates modify only the seven markdown files listed above. Source
changes belong to implementation (region domain/shared guards, reading-content and
refresh integration, SQL purge/schema, typed UI service/review editor/hooks/refresh
and App invalidation), with three new UI acceptance scripts and backend mutation
tests. Unrelated dirty-worktree changes are preserved.

### SRREPAIR-01 Delivery And Governance Receipt

Implemented internal `source_region_repair.region_repair_targets(document)` and
`source_review_blocks.region_source_intersections(document)`. Navigation and repair
share `_verified_block_spans`/`_build_mapping`; the existing persisted v2 mapping and
HTTP response are unchanged. Repairs resolve fresh source intersections rather than
treating cached block IDs as write proof. The contract includes source token,
structure/reading revision, canonical hash, offset basis, region spans, exact hierarchy
targets and can_apply/can_delete/blocked_reason. It is not a mutation authorization.

Reject projected documents, unverified/partial/ambiguous targets, manual edits and
deletions (including retained Undo audit), inactive review and changed/overlapping
reading overlays. Preserve navigation evidence when mutation is denied. Exact
record-owned repeat application was pending at this mapping-only checkpoint;
the implemented contract above supersedes that limitation.

Validation: 22 new tests pass; the combined mapping/repository/API/history/refresh
suite passes 399 tests without skips with disposable PostgreSQL enabled. Natural
evidence read checks cover four private backup books/39 regions, with all 16 backup
hashes unchanged. A separate saved incident snapshot contains 101 regions/60 exact
mapped targets; its manifest/source/rows/profile hashes remain unchanged. One test
initially assumed a raw_text mirror and was corrected to use the existing canonical
text contract, with a dedicated no-raw-mirror regression. Missing SQL configuration
in the initial broad run was resolved before the successful complete rerun.
These are read-contract and existing-write regression checks, not new region-write
or browser acceptance. No UI implementation, new API/schema, OCR/LLM execution,
production book mutation or Docker deployment is claimed.

Inspected memory: root AGENTS.md; proposal.md; high-level-design.md; docs/modules/index.md;
progress.md; both document_structure and Deep_Reader_UI design/checklist files;
docs/standards/frontend-development.md and backend-development.md. Governance edits:
both module designs specify the accepted successor and mark read-only constraints
historical; both owner checklists allocate seven nonduplicated tasks; backend progress
records backend tasks only. Old completed receipts/timestamps remain unchanged.
Evidence: code and scripts/test_source_region_repair.py; final regression output at
`/tmp/deep-reader-srrepair-01-tests.log`. Checklist timestamp/ownership checks and
`git diff --check` pass. Existing unrelated worktree changes are preserved.

## Repeatable Cleanup Repair (2026-10-25)

Maintainer-approved correction, tracked by RCR-01/03/04/05. Incident read-only evidence:
document 68 / task unit 19652 / block 19652:content:2 has manually_edited=false.
Revision 3 saved a cleanup draft, revision 4 restored it, revision 5 saved a newer
draft. Body still has the earlier replacement; already_restored blocks reapplication.
This is not a frontend cache-only issue or an accidentally set edit flag.

### Record-Owned Application

Draft text and applied text have separate lifecycles. cleanup_edit saves a draft;
cleanup_restore explicitly applies it. First application removes that exact deletion;
subsequent applications replace the same record's prior replacement, never append a
second copy or overwrite the whole block. Unchanged repeat application is safe and
uses existing mutation-ID replay rules. Original quote/source spans never change.

Persist cleanup_record_applications keyed by record ID in existing reading-state
JSONB, with canonical span, immutable quote and applied text. Tag owned source_edits
with cleanup_record_id; an own replacement is exempt from overlap rejection only
when ownership, span, quote and prior applied text all agree. Foreign overlaps,
manual body edit/delete, source or mapping changes still reject. Original-quote
restoration needs an application receipt even when no replacement overlay is needed.
Draft saves must not erase application ownership. Undo snapshots include applications
and overlays together, but saved drafts and permanent block_edit_flags remain outside
Undo. File and PostgreSQL share the domain logic and existing atomic repository port.

Only explicit body interactive Edit sets manually_edited. No history Save,
Restore or reapply sets or clears it. Manual deletion is a separate protection.
Legacy restored records without ownership are adopted only on an explicit write
when the latest non-draft audit is that exact record's Restore and its Undo snapshot
proves precisely which deletion was removed and replacement added. Reads can evaluate
that proof without backfill; ambiguous legacy changes stay unavailable.

### History Retention Unchanged

The maintainer withdrew the proposed no-navigation/no-save/no-display rule before
implementation: preserve recovery evidence for accidental deletion. RCR-02 is
cancelled, excluded from progress, and introduces no pruning or filtering. Existing
history remains available even when navigation is unavailable. Restore/navigation
still require their respective exact server eligibility; cleanup-hidden blocks can
be recovered, whereas stale/unmapped targets are never guessed. The earlier boundary
question is resolved by withdrawal; no answer is needed.

### Targeted Refresh Contract

RCR-03 implements optional `refresh` on the existing reading-content mutation
response. Legacy top-level blocks/review fields remain compatible. The first page
contains at most 20 affected blocks; subsequent read-only pages use the same GET
route with `refresh_mutation_id`, `expected_revision`, `source_token` and
`block_offset`. Refresh queries cannot combine with review/history/block filters.
The standalone refresh response contains no whole-document/layout/review payload.

The server binds scope to the mutation audit, current source token and reading
revision. It returns mutation/current revisions, document/structure identity,
optional record ID, `scope_current`, `block_page` and affected block DTOs. Each
block carries authoritative hierarchy IDs, canonical `block_order`, next visible
unaffected `insert_before_block_id` (null means append), `manually_edited`,
`unit_source_hash`, and either the shared reader `content_block` DTO or a
`deleted=true` tombstone. These visibility tombstones also cover cleanup-hidden
blocks; they do not imply permanent manual deletion. Cross-block records include
all linked blocks; Undo compares projected text/visibility, including span overlays.
No client-side text splicing or ID/offset inference is permitted.

Generated invalidation is reported by `generated_invalidated`, matching the existing
document-wide generated-artifact policy; `cleared_block_id` identifies explicit
block interaction purging. Notes remain unless that block was explicitly purged.
Lazy interaction availability/artifact references are not advertised in refreshed
render DTOs. Their absence is unknown, not not_generated; RCR-04 must apply the
invalidation flags and preserve unaffected state. `unit_source_hash` also updates
the owner/hash of already cached sibling blocks without fetching their body text.

Same-ID replay acknowledges a committed write. If a later revision/source change
exists, it returns `scope_current=false` and no body patches, not stale content or
a second write. Targeted GET rejects stale bindings with 409; legacy audit without
the new source binding yields refresh unavailable rather than invented scope.
Only successful explicit mutations record the binding. Reads never backfill it.

The new GET loads one repository snapshot, indexes canonical hierarchy, then projects
only units containing this page's blocks through the same helper as full reading
projection. It never calls full review snapshot, profile, Prepare, layout or LLM.
The hierarchy index is still document-wide; this is bounded response/unit projection,
not an O(1) database block lookup. Existing mutation snapshot work remains compatible
and is not claimed to be fully bounded by the refresh page size.

RCR-04 frontend integration is implemented in its owning UI module: merge only
affected reader DTOs, preserve order/focus and recalculate local pagination without
full task-layout/section reload for cleanup Restore/Undo. RCR-05 now accepts the
isolated real-incident workflow at desktop/mobile widths. Mutation
success and refresh failure remain separate; retry reads, never repeat a committed
write with a new mutation ID merely because UI refresh failed.

Implementation status belongs to owner checklists. Prior CLEAN-RESTORE/CRUI results
remain historical receipts, not acceptance of repeat application/local
refresh. RCR-05 uses an isolated, hash-verified copy of the real incident state.

RCR-05 acceptance receipt (2026-10-25): UI-owned accept-repeatable-cleanup.py and
test-repeatable-cleanup-real.mjs preserve document 68 / block 19652:content:2 IDs,
revision 5, legacy draft/application/Undo proof in disposable SQL. Actual API/App
at 1280/390 prove three replacements, draft/body separation, unchanged manual flag,
targeted reads without layout/section reload and retention of five non-navigable
history records. Body Edit then makes Restore return 409 without any SQL change.
Document/snapshot reading JSONB match; source/profile/backups and canonical rows
remain unchanged; disposable schema removed. Private final evidence is under
data/test-books/rcr05/20261025T150648Z-06b7c9e3. UI design holds full test/governance
receipt and fixture limitations. This shared acceptance update changes no backend
implementation or schema, introduces no deployment claim and completes RCR 4/4.

### RCR-01 Delivery And Governance Receipt

Implemented in cleanup_history, reading_content and the shared cleanup Undo snapshot.
Application ownership is persisted alongside existing reading state; a successful
legacy adoption also normalizes its Undo snapshot so Undo preserves the previous
application's ownership and allows another explicit reapply. Reads do not mutate.
Existing transport already consumes server can_restore; targeted refresh and improved
applied/draft presentation remain RCR-03/04, not part of this delivery.

Evidence: final regression `/tmp/rcr-regression.log` reports 281 tests passing with
disposable PostgreSQL and backup-book shared-engine checks enabled; existing audit
suite `/tmp/rcr-audit.log` reports 29 tests passing. test_cleanup_history adds repeated
application, own/foreign overlap, original-quote restoration, legacy saved-after-restore,
legacy Undo, cross-block reapplication, body-edit protection, concurrent reapply and
post-SQL rollback checks. SQL verifies application and overlay ownership in document
JSONB and structured snapshot with no manual block flag. Production document 68's
snapshot was read in a read-only transaction; local projection of revision 5 makes
record 7d588760645951da3c241c2a8f2d695e07b47d8d9a5ff432b2a55781e222d9b0
restorable with no snapshot change. This is read-only compatibility evidence, not
a production Restore or RCR-05 real-book end-to-end acceptance.

Inspected markdown: AGENTS, proposal, high-level-design, module index/progress,
both development standards, document_structure/UI/db/api_schemas/scripts module
designs and checklists. Modified markdown: document_structure design/checklist to
correct one-shot semantics, define the four-task queue and record RCR-01 evidence;
UI design/checklist to replace full-layout refresh requirements with targeted refresh
and separate pending acceptance; db design to document JSONB ownership and atomic
Undo. The withdrawn no-navigation/no-retention proposal is explicitly cancelled,
not implemented or counted complete. Validation: tests above and git diff --check
pass; no frontend code/build/browser run, deployment, production mutation or progress.md
edit in this task. Existing unrelated workspace changes are retained.

### RCR-03 Delivery And Governance Receipt

Implemented reading_refresh plus shared unit projection in reading_content, the
existing main route variants and strict api_schemas models. Added
scripts/test_reading_refresh.py for real HTTP serialization, file/disposable SQL
storage, repeat apply, cross-block/hidden recovery, Undo, 25-block pagination, stale
source and revision, legacy proof absence, schema contradictions and rollback.
The source-change fixture uses a persisted source-binding epoch, since normal SQL
save/mutation intentionally cannot change hierarchy titles. Tests assert that the
binding really changed while reading revision stayed equal before testing rejection.

Evidence: `/tmp/rcr03-regression.log` reports 354 tests OK with PostgreSQL and the
backed-up cleanup corpus enabled; `/tmp/rcr03-audit.log` reports 29 tests OK.
`/tmp/rcr03-endpoint.log` and `/tmp/rcr03-blocks.log` confirm both standalone
task-unit-content/shared-block suites pass. These latter suites use script entry
points, not unittest discovery. No live LLM/OCR, production mutation, Docker restart
or frontend/browser acceptance was performed; RCR-04/05 remain unchecked.

Inspected markdown: AGENTS, proposal, high-level-design, module index/progress,
both development standards, document_structure/main/api_schemas/db/scripts/UI designs
and checklists, and memory/validation skills. Modified markdown: document_structure
design/checklist for implemented scope, evidence and RCR-03 completion; main and
api_schemas designs for existing-route queries and typed DTO behavior; db design for
atomic audit binding without DDL; UI design for the implemented integration contract
and UI checklist shared progress only (owned UI tasks remain 0/2). No progress.md
edit. The cancelled RCR-02 retention change remains unimplemented.

Implementation-validation: scope, architecture, boundaries, documentation, tests
and memory PASS against the evidence above; git diff --check passes. Validation
itself modifies no files. Source additions are reading_refresh and its tests;
source edits are reading_content, main and api_schemas; no removals. Unrelated
pre-existing workspace changes are retained. This is source/test completion, not
deployment or real-incident frontend acceptance.

## Cleanup Record Recovery (2026-10-25)

Maintainer policy supersedes numeric cleanup limits in historical CLEAN/SCLEAN
receipts: no 4096-character, 10-percent, 128-item or 512-character quote cap, nor
derived 64-character/schema allowance. Response-byte/model capacity and exact-source,
manual-protection, overlap and nonempty-body guards remain. Prompt versions are
canonical v3 / remaining v6; this does not assert improved model accuracy.

CLEAN-RESTORE extends reading-content, not the LLM engine. A domain-owned helper
resolves each immutable audit deletion into canonical block spans; IDs come from
current authoritative hierarchy and are validated against canonical text identity.
Opaque record identity derives from its cleanup operation/revision/item, never model
output. Exact canonical char_start/end remain the insertion anchor; exposed char_from
and char_to are block-relative Unicode code points before display normalization.
Multiple overlapping block links are allowed; uncovered non-whitespace or ambiguous
mapping disables restoration rather than guessing. Missing legacy evidence is unavailable.

Explicit cleanup_edit stores restoration text separately from immutable quote;
cleanup_restore removes only the selected active deletion and optionally adds a
record-owned canonical-span replacement. RCR supersedes the original one-shot rule
with reapplication of that record's own replacement. It never writes raw_text or creates new blocks.
An original block hidden only by cleanup can reappear. A manually edited/deleted
target, a later manual edit even if undone, stale source or foreign overlapping
replacement cannot be automatically overwritten/reinserted.
Restoring another cleanup record in the same block is not a whole-block edit.
For a cross-block edited restoration, replacement appears at its canonical start,
as in the existing source-edits projection; all original target links are retained.

Both commands use source token, reading revision and mutation ID under the existing
repository transaction. Record saves preserve Undo and generated data; Restore
advances reading state, records audit, recomputes cleanup-resolved review regions,
invalidates generated data but retains notes, and supports ordinary single-level Undo.
Original deletion evidence and saved restoration text survive Undo. Record and
navigation capabilities are read-only projections; no read-side repair/backfill.
Current source/block checks are repeated inside the write transaction.

The backend owns `manually_edited`: explicit body Edit persists
`reading_content.block_edit_flags[block_id]=true` in the existing document JSONB
and structured snapshot in the same database transaction as text/artifacts. This
flag survives Undo. Legacy edit audit/current overlays provide a read-only fallback;
no GET backfill, new table/column or frontend-derived edit flag. History draft edits
and Restore do not mark a whole block edited or prevent restoring disjoint records.

Final block links are pinned during explicit materialization/cleanup writes, including
the PostgreSQL cleanup-only path using the committed hierarchy snapshot. Reads never
backfill links. Legacy records can derive exact links only when canonical identity
and complete coverage are verifiable; changed pinned links disable restoration.

Delivery: `cleanup_history.py` owns this reusable domain behavior; repositories retain
locking, CAS/idempotency and persistence. `test_cleanup_history.py` exercises file and
PostgreSQL paths, API flags, hidden/cross-block recovery, repeated/concurrent writes,
immutable evidence, stored flags after Undo and injected failure after SQL updates.
The combined regression passes 266 tests and the existing cleanup audit suite passes
29 tests. SQL checks confirm documents JSONB and structured snapshot agree; draft saves
retain generated artifacts and Note, while Restore removes generated artifacts only.
Read-only backup-book shared-engine checks are included, not live-book Restore or
model-quality acceptance. UI CRUI-01..05 are separately counted in the UI checklist.

## Source Review Mapping And Read-Only Workflow (2026-10-25)

Historical delivered SRMAP scope. The guarded SRREPAIR design above supersedes the
permanent read-only/single-editor restriction; until its UI tasks land, the running
Source regions UI remains read-only. Mapping, evidence and explicit Ignore rules
below remain applicable; old acceptance does not establish region-write safety.

OCR-KEEP supersedes SRMAP's blanket hiding of proven never-in-body evidence:
quality-rejected text and nonblank recognition gaps remain visible, including
legacy records. New extraction retains nonempty OCR in canonical text with exact
source spans. The shared resolver still owns block mapping; no guessed links,
read-side recovery or second editor. Existing books need explicit re-extraction,
not metadata backfill, to recover discarded body text. See loader OCR-KEEP design.

**[Maintainer-Provided] + [Code-Confirmed, 2026-10-25]** This is the current contract
for manual OCR source review. Implementation order and acceptance are tracked in
[SRMAP-01..08](module-checklist.md#source-review-mapping-backlog-2026-10-25).
Earlier CLEAN/SCLEAN acceptance is historical evidence, not completion of this track.

### Observed Baseline And Superseded Behavior

The pre-SRMAP implementation calculated only transient intersections, treated all
pending regions as review-needed, automatically Ignored last-block deletions and
offered an inline Source review editor. SRMAP replaces those behaviors with durable
source-bound relations, shared conservative eligibility, manual-change audit and
read-only navigation. Historical inline-editor acceptance is not current evidence.

Incident evidence: the read-only investigation of `暗水幽灵` found 29 unresolved
regions with no block mapping and zero-length canonical page ranges; 24 had empty
candidates, and the other five candidates were absent from canonical text. This
illustrates excluded OCR evidence, not permission to hide every unmapped region.
Existing successful mapping fixtures do not establish real-book mapping coverage.

Requirements superseded by SRMAP at that time: editing/deleting inside Source review; automatically
Ignoring a region when its last associated block is manually deleted; showing every
unmapped/textless region forever; treating a region as ownership of entire blocks;
and adding character-range highlighting/live edit-position tracking. Manual block
creation/insertion remains out of scope. Verified automatic smart-cleanup resolution
is separate: existing exact-coverage, successful-commit and Undo rules still apply,
and must never be labelled a manual Ignore or manual edit.

### Mapping Authority And Persistence

- `doc_loaders` owns immutable extraction identity, source page/region evidence and
  whether OCR candidates actually entered canonical text. It never assigns block
  IDs. Reuse accepted evidence; do not rerun extraction to populate a read response.
- `document_structure` owns source-range resolution, mapping semantics and a single
  reusable projection. A region may intersect several blocks and a block may
  intersect several regions. Partial overlap is valid. `complete` means the region's
  text is covered, not that every associated block belongs entirely to that region.
- Bind internal records to document scope, extraction identity/canonical hash,
  structure version, stable region ID, trusted half-open source spans, mapping
  policy version and backend-assigned hierarchy/block IDs. Persist verified links
  at explicit structure/task-unit materialization writes after final IDs exist.
  Retain link evidence for manually deleted blocks within the current structure
  epoch; no cross-reparse block aliases or new hierarchy level.
- Before units/blocks exist, preserve canonical evidence with a pending mapping
  state. An empty block list alone never proves exclusion. Reads may calculate a
  conservative legacy projection, but cannot backfill, prepare or persist links.
  Legacy backfill must be an explicit, revision-checked write with diagnostics;
  ambiguous/missing evidence stays unresolved, never fuzzy-matched or guessed.
- Manual edits retain source correspondence despite changed display text; manual
  deletions retain tombstones/links. Hard structural replacement invalidates old
  block links and builds new links from verified source evidence in its write flow;
  no-op/title-only preserves still-valid links. Undo restores current visibility,
  not obsolete hierarchy IDs or purged interaction data.
- File and PostgreSQL repositories implement the same domain contract. Persist
  mapping, overlays, revisions, audit and required artifact changes atomically;
  validate source/version and expected reading revision under the existing lock.
  No LLM/OCR/network work in a DB transaction. Physical storage representation stays
  behind repositories; no new table is mandated merely to expose navigation.

### Visibility, Resolution And Generation Are Separate

The implemented domain distinctions are:

| Evidence/state | Routine review | Navigation / generation policy |
| --- | --- | --- |
| Verified region intersects surviving body blocks | Visible while unresolved | Return exact surviving targets; retain applicable guards |
| Quality-excluded legacy candidate or nonblank recognition gap | Visible while unresolved | No invented target; retain source preview and conservative guards; explicit re-extraction for recovery |
| Source-bound confirmed blank render, empty candidate and verified empty page | Hidden | No target; omission alone never qualifies |
| Missing, ambiguous, stale or not-yet-materialized mapping | Visible with safe reason | Do not guess a target or weaken existing guards |
| Originally linked blocks later manually deleted, including all blocks | Visible until user Ignore | Show deletion/time; navigate only to surviving linked blocks |
| Explicitly Ignored or verified fully resolved by committed cleanup | Not in active queue | Retain evidence and truthful audit; apply existing Undo policy |

Hidden is not resolved, ignored or user-confirmed. Never fabricate audit entries or
delete original image/evidence on a read. Distinguish mapping coverage from the
region's body-inclusion history and from review disposition. For partial mappings,
keep the row and expose only verified targets. Shared blocks update every affected
region's manual-change projection without declaring the whole region corrected.
Use the same backend eligibility policy for review lists/counts, Library/reader
warnings and interaction guards; UI hiding alone is insufficient. Excluded raw
candidates must not leak into analysis, quiz, thinking, summary or ancestor context.
Existing safeguards for unresolved potentially included text remain conservative;
this track does not introduce a broader target-specific generation policy.
Explicit Ignore acknowledges the review item; it does not delete linked body text.
Removing that text requires the existing Reading content block action. Confirmation
copy must not promise exclusion from reading or generation merely because of Ignore.

### Manual Changes, Audit And Minimal Read Contract

Reading content's existing block actions are the only editor. Editing asks whether
to clear that block's interactive data, including Note; Keep preserves saved data
but cannot present old generated results as current for changed text. Delete clears
the deleted block's interactive data, including Note, with confirmation. Other
blocks' and hierarchy-level notes are not collateral deletions. Reuse existing
hierarchy-aware generated-output invalidation and late-writer guards. Undo can
restore text, not purged interactions. No product limit on manual edit/delete count;
retain idempotency and bounded reads without silently dropping replay protection.

Manual edit/delete/Undo audit carries server UTC time, affected block IDs, mutation
identity and reading revision. Derive related-region manual modification/deletion
status from verified links and audit, including all linked regions when a block is
shared. Preserve latest manual-change time across reload/restart; automated cleanup,
reparse and manual Ignore are not body edits. Undo shows the restored current state
with its manual-change history, not a false claim that the region was never touched.
Manual changes, even deletion of every related block, never auto-Ignore a region.

Extend the existing read service narrowly, not a second editor or arbitrary mapping
API. The projection contains region ID, mapping/visibility reason, ordered surviving
chapter/section/unit/block targets, deleted-linked-block indication and latest manual
change state/time, bound to source/structure/reading revision. Source coordinates and
full internal provenance remain backend details. Page/limit block content separately;
do not place full text or region images into task-layout. SRMAP-04's strict response
schemas and existing region/block query parameters implement this contract.

Source review reselect/reopen/explicit refresh reads current data. Successful body
mutations trigger the existing read-only layout/content refresh; never prepare or
reparse. Reader page is derived from current UI pagination and verified block IDs,
not PDF page_index. If targets disappear, refresh/reconcile instead of navigating to
another block or sending an empty ID. Browser design is owned by the
[UI module](../../Deep_Reader_UI/module-detailed-design.md#read-only-source-review-navigation-2026-10-25).

### Implementation And Recovery

`source_review_blocks.py` owns the pure resolver and `source-review-blocks-v2`
binding. `source_review_mapping` provenance binds canonical hash/length, extraction
and review/page evidence, structure epoch and hierarchy identity/content/titles.
Title changes matter because heading suppression can alter derived blocks. Custom
block source hashes, quote offsets and display-normalization evidence must agree
with the actual canonical slice. Duplicate quotes, corrupt bounds and stale proof
remain unknown, not guessed. Loader rejection records may explicitly say
`body_inclusion=excluded`; the loader still owns no block IDs or review decisions.

Initial file save, artifact writes and final-ID PostgreSQL hierarchy writes stage
the same mapping helper. Title-only SQL commits refresh mapping on the committed
hierarchy in their existing transaction; snapshots and document metadata agree.
Unchanged/replayed commits remain write-free. A failed mapping/snapshot write rolls
back the whole write. Canonical hierarchy and original block evidence remain intact
under reading overlays, so edits/deletions/Undo preserve source correspondence.

Legacy exclusion requires affirmative blank-render evidence and a proof written
from the actual canonical document. OCR-KEEP invalidates v1 exclusion classifications;
quality-excluded candidates and empty recognition gaps stay visible after backfill.
Without it, Library and reader both remain conservative; an empty block list or
old zero-length page record alone never silently removes review guards. The internal
`ReadingContentService.rebuild_review_mapping(doc_name, expected_revision,
source_token)` is an explicit CAS-checked metadata-only recovery command, not an
HTTP migration or a GET/task-layout side effect. It preserves reading revision,
overlays, notes and generated artifacts. No production backfill was run in this task.

Existing GET/POST `/documents/{doc_name}/reading-content` responses use strict
`ReadingContentResponse` transport schemas. Region fields add `body_status`,
`mapping_reason`, `related_deleted_count` and optional `manual_change={status,at}`;
the snapshot adds optional `structure_version`. `block_ids` retains historical
links, while region-scoped `blocks` contains ordered surviving targets with
chapter/section/unit/block IDs and current text. `mapping_status` remains
complete/partial/unavailable. Both general and region text queries return at most
20 blocks with `block_page`; a block-ID query returns its target only. Internal
canonical spans, geometry and mapping provenance are not exposed. Preview remains
the existing image endpoint. No second mutation API or task-layout text payload.

Manual events record UTC time, affected block IDs and mutation ID. Region badges
derive modified/deleted/restored state from those events and current overlays;
cleanup and Ignore alone do not create a manual badge. The existing block editor
retains Keep/Clear Note policy, targeted deletion, replay/conflict protection and
text-only Undo. Manual deletion never calls automatic region resolution.

### SRMAP Acceptance

`test_source_review_mapping`, `test_source_review_integrity`,
`test_source_review_mapping_repository`, `test_source_review_blocks` and
`test_source_review_api` cover exact many-to-many/partial mapping, exclusion proof,
stale source/title/custom blocks, file/SQL parity, CAS/replay/rollback, atomic notes,
bounded HTTP reads and guard consistency. The volume regression performs 1002
actual manual edits without a quota or automatic Ignore. Existing cleanup,
generation-target and reading regressions remain part of final validation.

`scripts/accept_source_review_mapping.py --backend both` accepts four hash-verified
backup corpora in temporary file storage and disposable PostgreSQL: eight storage
runs, with all 16 backup files unchanged. Report:
`data/test-books/srmap07/20261025T053519848970Z.json`. The full 176196-character
dark-water backup has 39 original regions, unknown before an explicit proof write
and excluded afterwards; it is not the earlier 29-region incident snapshot.

There is no naturally linked pending region in these backups. Positive tests
therefore construct review evidence from accepted real-page text and a strict
subquote, without injected canonical spans; the report labels this distinction.
Real-source shared/partial mappings, edit/delete/Undo, notes and explicit Ignore
are verified. UI acceptance uses the same real-page fixture through actual
App/API/disposable SQL at 1280/390; Original images are UI fixtures, not a new OCR
preview acceptance. See the UI receipt for screenshots and lifecycle coverage.
No live LLM/OCR, production document mutation, deployment or universal mapping
accuracy claim is part of SRMAP acceptance.

Final backend validation: 252 source-review/reading/notes/OCR/file/PostgreSQL tests
pass without skips. A separate 120-case cleanup/API/SQL suite also passes without
skips after enabling the immutable-book opt-in. Counts are separate suite runs,
not a claim of 372 unique tests. Source/architecture/whitespace checks pass.

### SRMAP Governance And Validation Receipt

Task type: implementation plus authorized maintainer synchronization. Read AGENTS.md,
proposal.md, high-level-design.md, docs/modules/index.md, progress.md, both root
docs/standards development files, and design/checklists for document_structure,
Deep_Reader_UI, db, scripts, doc_loaders, main.py and api_schemas.py. Earlier app/
preparation/section_tasks memory remains the supporting boundary context.

Markdown changed in this implementation pass:

- document_structure/module-detailed-design.md: replace planned/baseline statements
  with mapping/visibility/API/recovery implementation, acceptance and honest limits.
- document_structure/module-checklist.md: close five backend owner items and the
  shared 8/8 index using final evidence; timestamp changed items 2026-10-25.
- Deep_Reader_UI/module-detailed-design.md: record read-only navigation, revision/
  lifecycle guards, manual history and actual browser/real-source acceptance.
- Deep_Reader_UI/module-checklist.md: close three UI owner items with current evidence
  and timestamps; remove pending-SRMAP wording from the historical pointer.
- api_schemas.module-detailed-design.md: replace obsolete untyped-response claim
  with implemented filtered/strict additive DTO and bounded page contracts.
- main.module-detailed-design.md: document existing routes' bounded reads and Ignore
  semantics, without adding another editor or mapping-write HTTP endpoint.
- doc_loaders/module-detailed-design.md: record rejected-candidate source evidence
  and supersede old manual quota/Note/visibility statements with owner contract.
- progress.md: synchronize backend five-task completion and recount only the target
  module (74 checked/18 historical unchecked); do not aggregate UI work.

Evidence: owner checklists, cited implementation/test paths, private srmap07/srmap08
reports, final API/SQL/browser logs and inspected 1280/390 screenshots. Source files
were modified by implementation; the checklist/progress synchronization and final
implementation-validation phases made no source changes. Existing unrelated dirty
work was retained; no commit, production data operation or deployment was performed.

implementation-validation: scope, architecture, ownership, documentation, tests and
memory PASS within SRMAP scope. No root-sections/structure_nodes authority, hidden
task-layout writes, profile regeneration or loader format branching was introduced.
Warnings/limits: pre-existing UI bundle-size warning, no naturally flagged positive
review region in current backups, mocked preview images, no fresh OCR/LLM acceptance.
Historical unchecked tasks are not closed or newly validated by this receipt.

## Offline Counting Integration (2026-10-24)

Supporting llm-owned [WAIT-02](../llm/module-checklist.md#offline-preflight-follow-up-2026-10-24),
not an additional task: SemanticPlanProvider forwards the optional tokenizer identity;
plan_prompt_budget includes it only with a successful precise count. Both split and
cleanup already hash the complete budget decision, so a recipe/asset/library change
or transition from conservative fallback invalidates previous capacity consent.
Public capacity DTOs and prompts remain unchanged. Budget policy, schema/transport
accounting, reserves and byte-upper-bound refusal stay here; local encoding assets,
mapping, initialization and logging stay in llm. No request-path download or new
parser/persistence authority. The 129-test WAIT-02 suite covers unchanged limits,
complete schema accounting, fallback and no-dispatch identity rejection; packaging
and full incident-book HTTP acceptance are separate pending tasks.

## Independent Smart Cleanup (2026-10-23)

**[Maintainer-Provided] + [Code-Confirmed: SCLEAN-01..10, 2026-10-24]**
Backend and UI capability plus bounded final acceptance are delivered.
SCLEAN-01..10 supersede the exclusive
enhancement entry in CLEAN-01..10; the completed CLEAN work remains historical
implementation/acceptance evidence. No runtime deployment is implied by this update.

SCLEAN-10 uses all three immutable OCR backups, live gpt-4.1-mini, actual API/SQL
and 1280/390 browser checks for standalone, combined and failed-auto retry. Sample2
has a manually reviewed nonempty deletion and Undo; other unsafe proposals are
rejected, not treated as perfect cleanup. Runtime prompt/policy and guards were not
relaxed. Reproduction, exact attempts, isolated-image smoke, source preservation
and semantic limitations are owned by the
[scripts acceptance](../scripts/module-detailed-design.md#sclean-10-delivery-2026-10-24).
This pointer does not duplicate task ownership or claim production deployment.

### SCLEAN-01 Contract Delivery (2026-10-23)

**[Code-Confirmed]** `cleanup_input.py` defines immutable `CleanupInput` and
`CleanupViewPart`. Accepted extraction identity, exact canonical hierarchy coverage,
snapshot fingerprint, reading revision, OCR/page evidence, protected ranges and
prior-deletion gaps are bound together. The existing hierarchy DTO is frozen as
canonical JSON internally; `document()` returns a detached copy. This is not a new
persistence source or public payload. Common hierarchy without task units is valid;
no new split, parser-mode switch or HTTP `accepted` boolean is required. Explicit
OCR extraction identity remains sufficient evidence; Auto requires actual matching
OCR spans. Missing/unknown/native eligibility never authorizes a commit command.

`cleanup_operation.py` separates app-owned consent binding, pure `StagedCleanup`,
content-only `CleanupCommitCommand`, receipt/result and phase/commit/retry state.
Consent includes the complete input fingerprint plus prompt/policy/capacity identity;
standalone consent cannot reuse structural discard permission. An unknown outcome
requires reconciliation, layout recovery preserves the committed result, no-op
receipts retain reading revision, and changed results advance it exactly once.
The narrow `CleanupRepository` protocol declares commit and read-only receipt lookup;
SCLEAN-04 implements storage/CAS/replay enforcement as documented below. Dataclass construction
is not authorization or proof of semantic safety; SCLEAN-03/04 revalidate deletion
safety and current state. Adapters must establish coherent accepted storage input.

Compatibility: existing `prepare_enhancement_cleanup`, `CleanupBinding` and
`CleanupApplication` remain unchanged. SCLEAN-02 adapts accepted snapshots and
in-memory candidates; SCLEAN-03 provides the shared engine, SCLEAN-07 migrates runtime callers. Do not
manufacture an enhancement candidate for standalone cleanup or commit twice.
No new HTTP route, provider call, database mutation or UI capability is delivered here.

Validation: 26 new contract tests, including an opt-in read-only test of all three
hash-verified Common/OCR backups under `data/test-books/cleanup-20261023`; 120 tests
pass in the contract/planner/validator/file-overlay suite, plus 21 explicitly selected
preparation/evidence/layout tests. Run with `PYTHONPATH=.:scripts`; enable real input
checks with `DEEP_READER_TEST_CLEANUP_INPUT_BOOKS=data/test-books/cleanup-20261023`.
The initial broad runner encountered the existing script import-path requirement and
missing `DEEP_READER_TEST_POSTGRES_DSN`; log suppression also invalidated assertLogs
tests, so final passing runs retain logging. PostgreSQL acceptance is not claimed.
No live generation or production-data change; full end-to-end acceptance remains SCLEAN-10.

### SCLEAN-07 Combined Commit Integration (2026-10-24)

Production enhancement now uses CleanupInput and remaining-view prompt v3, just like
standalone cleanup. CleanupApplication optionally carries that immutable input plus
accepted-baseline/candidate digests; legacy applications retain their existing command
hash. Combined staging checks source/version/revision and both digests under the
repository lock, revalidates through validate_cleanup_input and delegates to the SAME
stage_validated_cleanup used by standalone commits. No copied matcher/overlay logic,
model call in storage, second standalone commit or persisted-candidate cleanup pass.
Protected edits, prior-deletion gaps and cumulative limits therefore apply to both
entry paths. Compatibility canonical planning remains only for legacy direct callers.
Implementation/test evidence and ownership are in preparation SCLEAN-07.

### SCLEAN-03 Shared Engine Delivery (2026-10-23)

**[Code-Confirmed]** `ocr_cleanup_planner.py` exposes `cleanup_input_prompt` and
`plan_cleanup_input` for either adapter's immutable input. The remaining-view prompt
is `ocr_cleanup_prompt_v3`; policy is `ocr_cleanup_remaining_policy_v1`. Each part
is separate quoted data: active deletions are absent, manual replacements are
context-only, protected/native Auto spans are ineligible. Quotes and adjacent
disambiguation context cannot cross part boundaries. The complete effective view
is budgeted, including schema/transport accounting, without truncation. Mixed-text
retention, uncertainty and injection protections share the existing prompt body.
Both new and legacy entry points use one provider dispatch/capacity/cancellation/
transient-retry implementation. The legacy canonical v2 prompt remains byte-stable
until runtime migration; it is not the new remaining-view policy.

`validate_cleanup_input` checks the supplied input fingerprint, resolves exact
quotes only within remaining parts, and calls the same item/overlap/boundary/size
validation core as legacy enhancement. Deleted copies cannot cause false ambiguity;
multiple remaining copies, including manual context, still require disambiguation.
No fuzzy matching, joined gaps, manual edits or native Auto deletions are accepted.
Existing source-origin heading protection remains; for Common without boundary
origins, conservatively retain each section's first nonempty line. Malformed supplied
boundary evidence is rejected, never treated as absent. This fallback may retain
first-line noise and is not a new parser classification or hierarchy write.

The active prior deletions plus new validated batch share the 4096-character and
10% total/nonspace caps and all-content/all-main-body guards. A cumulative-limit
violation rejects the new batch without undoing old state. Other invalid items can
be rejected independently while safe items remain. Pure resolution reports counts
and safe reasons, never commit success or semantic certainty. Consent/policy binding
and under-lock revalidation remain app/repository responsibilities in SCLEAN-04/05.
No source, hierarchy, reading revision, review queue or storage mutation occurs here.

Validation: 20 new shared-engine tests; 160 contract/adapter/planner/validator/file
overlay tests plus 21 selected preparation/evidence/layout regressions pass. Tests
cover prompt snapshots, mixed-readable sentinels, manual/gap/ambiguity cases,
cumulative limits, zero-call capacity/ineligibility, fingerprint mismatch and safe
logs. Three hash-verified Common/OCR backups pass read-only prompt/mapping checks;
Sample2's previously reviewed deletion replays at identical canonical/page offsets.
No new live model requests, PostgreSQL acceptance or deployment is claimed.
SCLEAN-07 still owns automatic enhance runtime wiring; SCLEAN-10 owns live acceptance.

### SCLEAN-04 Shared Commit Support (2026-10-24)

**[Code-Confirmed]** Structure-owned `cleanup_commit.py` provides locked-snapshot
checks, shared-validator revalidation, minimal standalone receipts and pure staging
to both storage adapters. It does not duplicate matching/deletion policy or call
models. The existing overlay application is extracted to `stage_validated_cleanup`:
legacy combined enhancement and independent cleanup call this same function for
source-span application, audit, review resolution, Undo and generated invalidation.
Existing combined commits still stage before their sole final write, never save and
reload a candidate for cleanup. `reading_content.advance_layout_revision` shares
revision-stamp handling with manual edits, without declaring a stale cache ready.

File and SQL ports expose only commit and receipt lookup; SQL/file atomicity, replay,
supersession and test evidence are owned by the [db delivery](../db/module-detailed-design.md#sclean-04-delivery-2026-10-24).
Source-file checking reuses the bounded hash utility only, not preparation/extraction.
The two cleanup entry paths must continue sharing these primitives: app coordinators
must not acquire their own prompt, quote-resolution, overlay or invalidation rules.
Public consent/operations remain SCLEAN-05/06, new enhancement-input wiring remains
SCLEAN-07, and frontend entry/recovery remain SCLEAN-08/09. No standalone HTTP API
or user-visible action is delivered by this storage task.

### SCLEAN-05 Consent And Recovery Support (2026-10-24)

**[Code-Confirmed]** `cleanup_input_capacity` budgets the same full remaining-view
prompt and schema as dispatch, returning an internal fingerprint of prompt, policy
and measured model limits. `plan_cleanup_input` can enforce that expected identity
before provider dispatch and transient retry; legacy callers remain compatible.
No app-owned prompt/budget formula or second cleanup engine is introduced.

`CleanupInput.source_structure_fingerprint()` binds namespace/document, accepted
extraction/text identity and structure version without including mutable reading
overlays or task-layout metadata. Content-only receipts persist this digest alongside
their existing revision-bound result. The app compares it to current accepted input
before replay/readiness recovery. Older receipts lacking this field are readable but
cannot authorize continuation. This additive receipt field requires no SQL migration.
Shared quote validation and overlay application remain unchanged; source-level
orchestration evidence is recorded in the [app delivery](../app/module-detailed-design.md#sclean-05-delivery-2026-10-24).

### Scope And Superseded Decisions

Smart cleanup is one reusable capability with two callers: explicit cleanup of an
already Common-parsed document, and automatic cleanup after a valid LLM split plan
has been applied in memory. Common parsing/import/read does not itself call an LLM
cleanup operation. A failed/partial automatic cleanup may be retried independently
by the user without generating a new split plan. Both paths use identical policies.

Remove these old requirements from the active design: cleanup requires a newly
successful LLM split; cleanup input must be an EnhancementCandidate; a user must
re-enhance to obtain another cleanup attempt; and structural success is the sole
operation result. The existing allow_ocr_cleanup-on-reparse wire contract may remain
as a compatibility adapter, but must not define standalone cleanup eligibility.
Do NOT remove the prior safeguards: precommit cleanup in the combined workflow,
one final atomic save, immutable extraction, exact matching, source/revision binding,
manual-edit protection, notes preservation, Undo and nonblocking optional failure.

First release is document-scoped, OCR-only, delete-only. No text rewriting, missing
character reconstruction, structural edits, web search, manual block insertion,
new extraction or profile generation. Known OCR evidence enables cleanup; Auto
requires actual recovered spans. Native TXT/Fast and unknown provenance skip with
a safe reason and zero model calls. PDF type and profile language are not eligibility
authority. Relaxing entry coupling does not relax semantic/deletion protections.
Pending source-review regions must not reuse the interaction-generation block to
disable this corrective action. Cleanup may operate on proven mapped OCR text;
textless/unmapped regions remain pending and cannot be fabricated or auto-dismissed.

### Ownership And Minimum Exposure

| Owner | Sole responsibility | Must not own or expose |
| --- | --- | --- |
| document_structure | Cleanup input/proposal/outcome contracts, prompt/planner, exact resolution, pure overlay staging and repository mutation contract | HTTP, UI, loaders, SQL orchestration or fake enhancement creation |
| document_preparation | Adapt an accepted stored snapshot or validated in-memory hierarchy to cleanup input; explicit layout readiness handoff | Cleanup consent/jobs/transactions; invoking split/OCR/profile to obtain cleanup input |
| app | Two use-case coordinations, consent, operation identity, freshness, execution/reconciliation and postcommit readiness | Prompt rules, span matching, SQL details or duplicate provider implementation |
| db / file repository | Owner-contracted atomic commit, compare-and-swap, receipts, invalidation and rollback | LLM requests or reading-triggered repair; hierarchy replacement for standalone cleanup |
| main.py / api_schemas.py | Thin authenticated/namespace-scoped transport and strict public DTOs | Plan validation, storage access policy or raw executable plan endpoints |
| Deep_Reader_UI | Explicit action/confirmation, typed service adapter, server-state rendering and recovery | Computing spans, inferring OCR, authoritative content or automatic regeneration on refresh |
| scripts | Deterministic/fault/real-book acceptance and private evidence | Production library reset or runtime policy changes merely to pass tests |

Reuse existing modules and extraction/provider/repository protocols. Introduce only
the narrow cleanup input adapter and use-case boundary that remove real coupling;
no generic job engine, plugin framework, provider inheritance hierarchy or public
plan-edit/apply API. Shared provider DI belongs to composition, not a chain through
the splitter's private fields. App invokes preparation/structure/repository ports;
the planner never imports app/main or receives a repository. Do not expand the large
SectionTaskCoordinator with prompt, mapping or persistence implementations.

### Two Execution Paths

Standalone: accepted stored hierarchy -> read-only eligibility/capacity preflight
-> user confirmation -> coherent immutable cleanup snapshot -> cleanup plan ->
exact validation/staging -> atomic reading-state commit -> automatic readiness/
content refresh. Loading the existing document is legitimate here; it is not the
forbidden intermediate save of an enhancement candidate. A validated Common body
is sufficient even when task units have not yet been created; empty/invalid source
is rejected without invoking full Prepare. Preparation may explicitly create missing
layout after cleanup, reusing the existing profile, never rebuilding it.

Combined: accepted source -> split plan -> validate/apply in memory -> adapt that
same candidate to the shared cleanup capability -> stage accepted deletions -> one
atomic hierarchy/overlay/receipt commit -> automatic layout. No candidate reload,
HTTP self-call or second cleanup commit. Cleanup provider/plan/application failure
retains the uncleaned valid split candidate, but final transaction failure rolls back
the entire combined write. A source/revision conflict is not a harmless cleanup skip.
No database transaction remains open during model calls in either path.

### Internal Contracts And Repeated Cleanup

Proposed immutable CleanupInput contains accepted extraction identity/canonical
digest, hierarchy snapshot/fingerprint, reading revision, authoritative OCR/page
evidence, protected manual regions and the current effective reading view with
exact canonical mappings. It must not require parser_mode=llm_enhanced or a new
split result. An enhancement adapter proves candidate readiness; a standalone
adapter proves existing hierarchy readiness. A boolean asserted by HTTP is no proof.

LLM sees eligible remaining source and necessary readable context, not prior deleted
text offered as fresh work. Preserve protected manual replacement context without
allowing its deletion. Keep disjoint mapped spans explicit; do not concatenate across
deleted gaps into a falsely contiguous quote. Map exact quotes back to canonical
coordinates and reject ambiguous, unmappable, cross-boundary or protected edits.
Do not persist effective text over canonical hierarchy/source. Source/structure/
reading revision, prompt/policy versions and consent are bound by the backend, not
invented by the model. The plan remains delete quote plus optional adjacent context
and allowlisted reason; IDs, offsets and execution authority stay internal.

Prefer complete eligible remaining text; count the whole assembled prompt against
the configured model capability before calling it. Overflow makes zero model calls
and yields a concise reason, not hidden truncation. No cost/OCR quotas are added.
Do not weaken exact matching or existing size/all-body/anchor guards. Policy must
also check active prior deletions plus the proposed batch so repeated runs cannot
bypass an aggregate safety limit. Freeze this behavior and prompt version in SCLEAN-03.
Mixed readable/garbled fragments retain readable words, names and punctuation;
uncertain meaning is retained. Prompt compliance alone is not a safety proof.

### Writes, Readiness And Recovery

Standalone commit changes reading state only: preserve chapter/section/unit/block
IDs, labels, source slices, structure_version, extraction/page maps and profile.
Only an actual effective-content change advances reading_revision and invalidates
generated interactions/ancestor caches document-wide. Notes and manual edits remain;
late generation saves at old revisions are rejected. A no-op/failed cleanup never
clears interactions or increments content revision; a minimal operation receipt
may still record its terminal outcome without claiming a content change.

This note-preservation guarantee belongs to cleanup-only content changes. Combined
enhancement with actual structural replacement still follows the existing separately
confirmed all-interaction deletion policy; cleanup success/failure cannot alter it.

For a ready layout, update the revision stamp within the explicit mutation while
retaining canonical units. Missing/stale layout is recovered automatically by the
explicit use case; no read endpoint persists or asks the user to click Prepare.
Postcommit layout failure is distinct from cleanup failure: resume layout without
calling either LLM planner. Audit/Undo remain the existing single-level model, not
a new arbitrary rollback API. Undo restores reading/review state, not old structure
or generated artifacts, and never overwrites a subsequent manual change.

Keep operation phase, commit certainty and CleanupOutcome separate. Terminal
cleanup statuses remain not_applicable/no_changes/completed/partial/skipped/failed;
unknown outcome is reconciliation state, not completed or retry permission. Full
source-review resolution still requires trusted exact coverage and commit; partial,
unmapped and textless regions remain visible. Success is not perfect transcription.

Repeat the same operation identity to reconcile/replay without another mutation or
paid generation. Explicitly trying cleanup again obtains a fresh preflight/consent
and operation identity against current content; it may generate a new cleanup plan
but cannot trigger splitting. Retry transient provider failures within the owned
operation; do not treat a capacity/semantic rejection as a transport failure. Do not
lock the user's content during model work; recheck source/hierarchy/revision under
the final short write lock. Concurrent cleanup/reparse/edit may invalidate the plan.
Persist minimal committed receipts for refresh/worker restart; do not claim current
process-local in-flight work is durable. Lost precommit state stays unknown until
reconciliation establishes a safe new explicit attempt. No UI polling regenerates.

### Public Surface And Acceptance

Expose only preflight, execute/resume and read-only operation lookup; proposed wire
contracts are owned by main.py/api_schemas.py. The UI submits document identity and
opaque consent/operation handles, never raw source, provider credentials, plans,
offsets, guard thresholds, parser mode or discard_interactions for standalone cleanup.
Reuse reading-content audit/Undo instead of adding cleanup-specific history/mutation
routes. Logs correlate origin, stage, prompt/policy, durations, counts, safe rejection
reasons, commit/replay and layout; no raw book text, response or confirmation token.

SCLEAN-01..10 are ordered in this module's checklist, counted once at their owners.
Required end-to-end cases: Common -> standalone cleanup with no split; enhance ->
same planner before single commit; failed automatic cleanup -> independent retry;
repeated cleanup of remaining text; partial/no-op/over-capacity/unknown provenance;
concurrent edits/reparse, lost responses/restart, rollback, layout-only retry and Undo.
Backed-up Sample/Sample2/noise provide real OCR/model/API/SQL/browser acceptance;
Madame Bovary TXT and Red-book Fast remain zero-call negative controls. Require at
least one manually reviewed nonempty deletion and review all final applied spans;
do not count all-skipped or mocked runs as live acceptance. Verify all five interaction
content levels, preserved source/profile hashes and stable standalone identities.
Deployment/version smoke checks are distinct from source-test completion; old
running containers must not be described as having the new capability.

### SCLEAN Documentation Governance Receipt

Task type: documentation/governance only. Module-memory-loader inspected AGENTS.md,
proposal.md, high-level-design.md, docs/modules/index.md and progress.md, plus
design/checklist pairs for document_structure, document_preparation, app, db,
main.py, api_schemas.py, scripts and Deep_Reader_UI. Source inspection confirmed
the current reparse-only allow_ocr_cleanup validation and candidate-dependent
prepare_enhancement_cleanup handoff; new route/service claims remain Future Direction.

| Modified markdown (backend paths relative to Deep_Reflective_Reader) | Reason |
| --- | --- |
| document_structure/module-detailed-design.md | Canonical dual-entry design, single-responsibility boundaries, safety/failure contracts; replace exclusive-entry wording and outdated helper integration statement. |
| document_structure/module-checklist.md | Shared dependency order and two owner tasks, no duplicate counts. |
| document_preparation/module-detailed-design.md; module-checklist.md | Two adapters and combined-flow/layout integration, two owner tasks. |
| app/module-detailed-design.md; module-checklist.md | Independent use-case/retry ownership, one owner task; distinguish new cleanup from replay of enhancement. |
| db/module-detailed-design.md; module-checklist.md | Content-only transaction/file parity, one owner task; no hierarchy replacement. |
| main.module-detailed-design.md; main.module-checklist.md | Three minimal proposed routes and one transport task. |
| api_schemas.module-detailed-design.md | Strict minimal proposed DTOs, preserve legacy enhance adapter without claiming new exposure. |
| scripts/module-detailed-design.md; module-checklist.md | Real-book/fault/browser/current-image acceptance and one final owner task. |
| ../Deep_Reader_UI/module-detailed-design.md; module-checklist.md | Independent action/confirmation/recovery, two owner tasks; backend remains authoritative. |
| progress.md | Add backend-only SCLEAN 0/8 without rewriting historical CLEAN completion. |

All ten new checkboxes are unchecked and dated 2026-10-23. Historical completion
receipts remain as evidence, not active exclusive-entry restrictions. No source,
tests, API schema code, configuration, data or running service is changed. No new
markdown files or deleted files. Validation checks unique task ownership, dependency
order, date metadata, local links/anchors, backend-only counts, unchanged non-markdown
diff and git diff --check. Runtime/live-book tests are intentionally not rerun for
this documentation-only task; their future requirements are explicit in SCLEAN-10.
Checklist-progress-sync owns only the progress/checklist phase; implementation-validation
is read-only. Scope/architecture/boundary/documentation checks apply to this design,
not unimplemented API behavior or production deployment.
Validation result: PASS for documentation governance. Ten unique unchecked tasks,
eight backend/two UI, owner counts/dependency order/date metadata and 25 new local
links/anchors checked; non-markdown tracked diff unchanged and git diff --check
clean. No implementation tests or deployment were performed or claimed.

## Optional OCR Cleanup Plan (2026-10-22)

Historical CLEAN delivery and invariants follow. Exclusive enhance-only entry and
candidate-only input are superseded by Independent Smart Cleanup above; completed
test receipts do not constitute SCLEAN implementation or deployment evidence.

**[Maintainer-Provided] + [Code-Confirmed]** CLEAN-01..06 contracts, planner,
validator, reversible overlay, atomic storage, preparation and public orchestration
are implemented. UI CLEAN-07/08 and deterministic cross-stage CLEAN-09 are also
complete as of 2026-10-23; CLEAN-10 live acceptance is now complete within the
bounded corpus below. Delivery subsections
below retain their original scoped evidence, not the current task status.
App design records explicit consent, durable commit recovery and automatic layout;
the current cross-stage acceptance matrix and limitations are recorded below.
Authoritative scope (corrected 2026-10-22): LLM proposes a cleanup/fix plan; backend
applies it to the validated in-memory split candidate BEFORE persistence. Commit
the candidate and accepted cleanup together once, then prepare reading layout.
Do not save, reload from the database to clean, and save again. Cleanup failure must
preserve the usable uncleaned split candidate for that commit. This is not permission
to rewrite canonical extraction or weaken structural validation.

### CLEAN-10 Live Acceptance (2026-10-23)

`scripts/accept_ocr_cleanup_books.py` uses immutable SHA-verified backups under
`data/test-books/cleanup-20261023`, real OCR extraction in the pinned Docker image,
the configured gpt-4.1-mini provider, actual split/cleanup planners, FastAPI,
disposable PostgreSQL schemas and the production layout resolver. No production
Library mutation, extraction during enhancement, profile rebuild or deployment.
Structural prompt remains v17; cleanup prompt is now `ocr_cleanup_prompt_v2`.

Initial live testing exposed a semantic false deletion: a proposal included readable
Chinese beside corrupted Latin OCR in a parenthesis. That run is rejected evidence,
not a pass. V2 explicitly preserves readable words, adjacent lines and enclosing
punctuation, and narrows or omits mixed-content proposals. Exact-location and
4096-character/10% deletion guards are unchanged; semantic certainty is not proven
by a successful exact match or by these three samples.

Actual SQL/layout acceptance also exposed stale generated block IDs: assigning a
new unit ID without rebinding its compatibility block caused a full-unit block to
be treated as custom content and appended beside segmented blocks. Shared
`TaskUnit.with_unit_id`, both unit normalization paths and SQL ID allocation now
rebind only exact generated IDs, within the existing explicit write transaction.
Custom IDs/text remain unchanged. This is not a read-time repair or migration of
old stored units. Unit tests and real SQL/Undo regression prevent duplicate body
projection; all five interaction target levels read effective text consistently.

Final private attempt directories and results:

| Case | Attempt | Cleanup | Target checks | First reparse / cleanup provider |
| --- | --- | --- | --- | --- |
| Sample | sample/20261023T030348165685Z | skipped/no_safe_items; 0 deletions | 88 | 18.640s / 1.389s |
| Sample2 | sample2/20261023T030347436965Z | partial; 1 applied, 2 skipped | 9 | 7.707s / 2.085s |
| Dark-water garbage sample | noise/20261023T030459209503Z | skipped/no_safe_items; 0 deletions | 19 | 7.492s / 3.067s |

Durations are server/provider log measurements, not OCR time or latency guarantees.
Sample/Sample2 used two structural calls and one cleanup call each; noise used one
of each. All preserve complete canonical split coverage, source hashes and page
maps, commit once on replay and render/read/reload at 1280px/390px. Sample2 audit
expansion/collapse and mobile Undo restore content without changing hierarchy;
its persisted enhancement receipt becomes superseded after Undo.

Manual inspection covers every applied deletion in final runs: one three-character
corrupt OCR fragment on Sample2 page 1, without deleting surrounding readable text.
The source image contains an acronym at that position; removing its corrupt OCR
does NOT restore the missing acronym or guarantee faithful transcription. Sample
keeps the previously endangered readable phrase because validation rejects the
proposal, NOT because v2 reliably stopped the model proposing it. Prompt-only
semantic protection remains imperfect; never bypass exact/scope safeguards or
claim a general zero-false-deletion guarantee. Replaying exact validation of the
saved responses reports quote_not_found for Sample and noise, and
unsafe_text_boundary for Sample2's other two proposals. Heavy garbage remains
in noise: a broad proposed span fails exact matching; it was not automatically
cleaned. Aggregate deletion caps remain unchanged, but are not claimed as the
reason this final attempt was rejected.
No final sample has pending source-review regions, so automatically resolved/partial
region semantics are covered by controlled API/SQL/browser tests, not claimed as a
new live-region-resolution case. No new interaction answers are generated; the
five-level check validates the actual content resolver feeding those actions.

Madame Bovary TXT (652001 characters) and Red-book Fast (147401) backup controls
both return not_applicable with zero cleanup capability/provider calls. These are
real-content eligibility controls, not fresh full-book generation runs. Private
reports retain requests/responses, source manifests, screenshots and manual review;
do not check copyrighted text or model payloads into Git. Regression suite: 288 run,
287 pass, one inherited private-source opt-in skip; explicit live runs are separate.

### Current Evidence And Ownership

- `doc_loaders/extraction_identity.py` records generic extraction identity;
  `pdf_document_loader.py:open_source` carries OCR diagnostics and page spans.
  `document_preparation_pipeline.py` persists extraction and source_page_spans.
- `semantic_structure_plan.py` currently accepts boundaries only, with exact quotes
  and backend-owned positions. It has no cleanup action contract.
- `reading_content.py` provides explicit block overlays, revision checks, audit and
  single-level undo. These are foundations, NOT an existing source-span cleanup API.
- Structure owns cleanup semantics, resolution, effective-content projection and
  repository mutation contract; preparation owns accepted-source handoff; app owns
  operation sequencing/API; UI consumes server outcomes. Read supporting DB,
  section_tasks, main/api_schemas and scripts memory before implementation there.
- PDF native Outline/Bookmarks are source evidence, unlike profile classification.
  Existing Outline-first validation remains intact; no separate TOC redesign is
  authorized here. Cleanup cannot erase titles/bookmark anchors merely because
  OCR recognition is uncertain.

### Workflow And Failure Isolation

Use separate requests initially: full-source structure planning -> validate/apply
splits in memory and classify change -> optional OCR cleanup planning -> validate/
apply reading overlay to the same candidate -> one atomic commit -> explicit
layout-only preparation if required -> read. Splitting instructions may warn against
using OCR garbage as semantic evidence, but splitting must retain every source character.
Never make optional cleanup JSON a required part of the structure response/schema.

Cleanup can be explicitly invoked on an accepted Common result, or chained inside
the confirmed enhance workflow for proven OCR content. Common parsing/import/read
alone does not start cleanup. Use stored extraction
evidence, not filename, profile language or a guess from odd-looking text. Explicit
OCR identity qualifies; Auto qualifies only with trustworthy actual recovered-page/
span evidence. Missing legacy/Auto provenance means skip with a reason, not rerun
extraction. Do not introduce PDF-type branches into preparation or generic consumers.

Optional cleanup has its own complete-prompt capacity check, timeout and cancellation
boundary. Full accepted text is preferred; no hidden truncation, OCR, web search or
profile rebuild. Capacity overflow skips cleanup without invalidating a feasible
structure request. A cleanup-only provider/JSON/validation/storage failure yields a
warning and retains the uncleaned candidate for the final commit, then layout.
Structural validation failure skips cleanup; failure of the final atomic commit
remains an operation failure, not an optional-cleanup warning. Unknown commit outcomes require reconciliation,
not fabricated success. Failure to prepare/read layout is a separate recoverable error.

### Proposed Contracts And Exact Resolution

Implemented internal envelope under CLEAN-01: `BoundCleanupProposal` separates
`binding` (backend scope/source/version preconditions) from `proposal` (model
`schema_version` and `items`). Backend binds these values; the model never invents
IDs, hashes or offsets. Detailed field and limit definitions follow below.

Model item: `action=delete`, exact `quote`, optional exact adjacent
`prefix_quote`/`suffix_quote` for disambiguation, and allowlisted reason
(`ocr_garbage` or `recognition_fragment`). No replace/insert/reorder, corrected
transcription, arbitrary executable instructions, block IDs or occurrence counting.
Empty item list is a legitimate no-op. Freeze numeric schema/quote/item limits and
fixture-tested aggregate deletion safety limits in CLEAN-01/03 before enabling writes;
real-book calibration/false-deletion review remains mandatory in CLEAN-10;
do not silently truncate a plan or treat model confidence as proof.

Resolve against the SAME immutable accepted source/readable mapping as splitting.
Backend creates canonical half-open Unicode code-point spans, validates exact quote,
adjacency, uniqueness and source hash, then maps spans to the in-memory candidate hierarchy.
No fuzzy repair or guessing repeated occurrences. Page indexes are optional evidence,
not unique text locators; cross-page spans retain both page mappings. Cross-section
and ambiguous normalized-to-canonical spans are skipped in v1. Duplicates coalesce;
overlapping nonidentical deletions reject the affected items rather than merging them.
Protect structural heading anchors, meaningful multilingual quotes/names, poetry,
formulas, lists and numbers. Unknown or disputed semantic meaning is retained.
Exact matching proves location, not semantic uselessness; real-book review must
measure false deletions, and uncertain candidates remain visible for manual review.

### Persistence, Projection And Consistency

Stage validated canonical deletion intervals as a versioned reading overlay, with
operation/plan digest, source/structure/revision preconditions, per-item reason,
original quote and page evidence where available; persist it together with the
candidate hierarchy in one atomic commit. Original source, raw_text, hashes,
section source slices and page offsets remain immutable. No alternate hierarchy.
Backend-generated block identities must not be renumbered after cleanup. Overlay
projection must also work before task units exist: preserve canonical-to-effective
span mapping through unit construction, rather than guessing new block IDs from
the pre-reparse document. This extends, not merely calls, current block-only overlays.

Individually invalid instructions are skipped with reasons. Validate a complete
accepted subset and stage it on a separate candidate copy; preparation/application
failure discards that copy's cleanup, retaining the validated uncleaned split candidate.
Atomically persist chosen hierarchy, overlay, audit/receipt, reading revision and
dependent artifact invalidation once. Final storage failure rolls back the entire
transaction, preserving the old database document; it must never claim split success.
No partial per-item writes or independently committed cleanup transaction in this flow.
Replaying the same operation/plan is idempotent. Source/structure/reading revision
and consent are rechecked at final commit; concurrent changes reject the stale write.
Transactional version checks are permitted, but not reloading the saved candidate
as input to a second cleanup pass. All preconditions refer to the initial accepted
snapshot, while cleanup targeting uses its derived split candidate. New IDs and
the committed structure version are assigned by storage, not by the LLM.
Previously edited/deleted user content wins; skip intersecting automatic deletions.
Reject a subset that would erase all usable document text. Do not delete empty
hierarchy containers or reconstruct deleted text through fallback aggregation.

Cleanup-only changes leave structure_version unchanged, advance reading revision
and invalidate generated insight/quiz/thinking/summary and ancestor contexts through
the existing hierarchy-aware repository boundary. Notes remain with stale-source
indication for automatic span cleanup; manual block deletion and edit-clear follow
the separate targeted Note policy above. Structural reparse still
clears ALL interactions at the final successful commit under existing consent,
independently of cleanup success.
Protect against late generation writes at the old reading revision. All five target
levels use effective content; none may resurrect raw deleted spans. Existing profile
stays reused. A review region may be automatically resolved only when trusted
source-to-region mapping proves that accepted cleanup fully addresses its review
content, and the final atomic commit succeeds. Remove that region from the active
queue without requiring another Ignore Region action; retain evidence and audit.
Do not infer resolution from the model's claim, a successful request, matching page
number or disappearance of a block. Missing OCR text/mapping, partial coverage,
rejected items and failed cleanup do not prove resolution. Routine visibility,
pending counts and generation guards follow the planned Source Review Mapping
policy above: only proven never-in-body evidence can be excluded without resolution.
Unknown mapping and manually deleted linked blocks are not that exclusion case.
The server owns review_required; the UI must not hide warnings independently.

Provide batch audit and reversible cleanup using the existing single-level Undo
model, extended for span overlays. Undo restores pre-cleanup reading state only,
never the previous hierarchy or deleted generated artifacts; it advances reading
revision and invalidates derived output again. Restore review states resolved by
that cleanup batch too; never revert unrelated manual region decisions. A newer manual edit supersedes that
undo slot, with an explicit conflict/unavailable result rather than lost user work.

### Outcomes, Consent And Observability

Keep structural `change_kind` unchanged/titles_only/structure_changed separate from
`cleanup.status`: not_applicable, no_changes, completed, partial, skipped, failed.
Proposed additive outcome includes proposed/applied/skipped item counts, safe reason,
reading_content_changed and authoritative reading_revision. Zero applied items must
not claim content changed. Success of cleanup does not certify perfect OCR accuracy.
A structural no-op can still have a cleanup change; prior no-op statements describe
structure only once this optional feature is enabled. Do not bypass cleanup just
because change_kind=unchanged, or clear notes under structural cleanup semantics.

Preflight confirmation explains automatic OCR noise removal, preserved original,
restoration, generated-output invalidation and the existing separate all-interaction
structural replacement condition. Bind policy/prompt and current reading revision
to consent; old consent cannot silently authorize this new mutation. No cost quota.
Log operation-correlated stage durations, eligibility, counts, reason codes,
versions, capacity outcome, commit/rollback/replay and layout outcome. Never log raw
source/quotes, full model responses, keys or confirmation tokens.

### Ordered Implementation And Acceptance

Canonical task IDs are counted once in their owner's checklist:
CLEAN-01 contract (structure) -> CLEAN-02 prompt (structure) -> CLEAN-03 validator
(structure) -> CLEAN-04 overlay/atomic mutation (structure) -> CLEAN-05 source and
layout handoff (preparation) -> CLEAN-06 API/orchestration (app) -> CLEAN-07 flow
(UI) -> CLEAN-08 audit/undo (UI) -> CLEAN-09 fault matrix (structure) ->
CLEAN-10 real-book acceptance (structure, supporting scripts/UI).

Use verified backups under ignored data/test-books; preserve original files and
manifests. OCR positives: 中國文化的深層結構_Sample and Sample2, including known
garbage and meaningful vertical/horizontal controls. Include 暗水幽灵_sample when
available to reproduce the reported garbage analysis. Non-OCR regressions: Madame
Bovary.txt and 红色赌盘_261014_145614.pdf Fast. Missing corpus is Needs Confirmation,
never synthetic success. Live LLM tests plus actual file/PostgreSQL/API/layout/UI
acceptance are required; scoped success is not a universal OCR quality claim.
No production book deletion, OCR rerun or paid calls occur in contract delivery.

### CLEAN-09 Cross-Stage Acceptance (2026-10-23)

**[Code-Confirmed]** `scripts/test_cleanup_consistency.py` adds six test methods
covering 18 isolated API/SQL scenarios. No production implementation changes were
needed in this task. Tests compose the existing disposable PostgreSQL/API fixture,
without importing its test class into discovery or counting fixture cases again.

The end-to-end planner test removes the fixture's mocked candidate builder. Both
production planners, exact semantic quote resolution, candidate validation, overlay
staging, repository commit and automatic task layout execute. Only model transport
returns deterministic schema-valid responses. It asserts the persisted document is
still the original at BOTH model calls, then observes exactly:
`split_model -> cleanup_model -> stage -> hierarchy_write -> commit_returned -> layout`.
There is one hierarchy write inside the final transaction, one commit call, and no
save/reload/clean/save pass. The later explicit layout-cache save is permitted and
must follow commit; it is not counted as another hierarchy/cleanup commit.
Source bytes, canonical text, extraction identity, page spans and review evidence
remain unchanged. Hidden preparation/profile and subprocess extraction are forbidden.

| Acceptance branch | Evidence |
| --- | --- |
| Missing/empty/malformed cleanup and valid empty plan | New matrix across unchanged, titles_only and structure_changed; unchanged reading revision/review queue, single commit, usable candidate retained. |
| Optional overlay application failure | Inject projection conflict at staging for all three structural outcomes; discard staged deletion/audit, retain valid split/title update and automatically build layout. |
| Final write failure | Inject failure after SQL receipt writes for all three outcomes; original snapshot/fingerprint remains identical, no layout call, reconciliation reports unknown. Existing file tests inject atomic-save failure. |
| Lost acknowledgement plus process restart plus failed layout | Combined fault chain followed by successful POST retry and replay; one model cleanup, one candidate build, one commit, one revision/audit/receipt. Result lookup does not resume work. |
| Legacy non-OCR opt-out | Native accepted source with omitted cleanup flag retains legacy response semantics and never invokes cleanup or writes its receipts. |
| Timeout/retry, over-capacity, partial subset, concurrent edits | Existing `test_enhancement_cleanup_api.CleanupApiTests` exercises real routes and SQL, including zero dispatch on overflow and stale-commit rejection. |
| Review resolution, Undo, no resurrection/late writers | Existing file/SQL overlay suites cover fully resolved versus partial/absent/unmapped regions, all five effective target levels, revision CAS, notes preservation, generated invalidation, Undo and stale artifact writes. Preparation suites cover canonical units and no hidden extraction/profile work. |

Focused reproduction from the backend root, with a test-only PostgreSQL DSN:
```sh
PYTHONPATH=.:scripts DEEP_READER_LOG_DIR=/tmp/deep-reader-clean09-logs \
  .venv/bin/python -m unittest scripts.test_cleanup_consistency.CleanupConsistencyTests
```
Set `DEEP_READER_TEST_POSTGRES_DSN` externally; the harness creates/drops isolated
schemas. Never substitute a production book fixture. The focused run passes six
methods. The combined run selects contract/planner/validator modules and explicit
file/SQL overlay, preparation, API, consistency, reading-content, legacy outcome,
operation and candidate classes: 279 tests run, 278 pass, one inherited opt-in source
readback skipped because `ENHANCEMENT_SOURCE_DOCUMENT` is unset. These totals are
not additive with earlier delivery runs. Logs: `/tmp/clean09-consistency.log` and
`/tmp/clean09-regression.log`. An initial new planner fixture used a repeated opening
quote and was correctly rejected; a unique fixture opening fixes the test, with no
relaxation of production quote validation.

Limits: deterministic responses do not establish semantic cleanup quality. No live
model, real-book deletion review, browser run, OCR engine, production data mutation,
Docker restart or deployment is claimed by CLEAN-09. CLEAN-10 remains unchecked.

Governance receipt: inspected AGENTS.md, proposal.md, high-level-design.md,
docs/modules/index.md, progress.md, structure and scripts design/checklist pairs,
app design/checklist, preparation checklist and UI checklist. Modified this design
to record the matrix and limits; modified the structure checklist to close only
CLEAN-09 with dated evidence; synchronized backend progress from 6/8 to 7/8.
The shared queue is 9/10 including the two already completed UI tasks; UI is not
counted in backend progress. Production source files are unchanged in this task;
one test file was added. Validation checks scoped diffs, checklist arithmetic and
timestamps, test results and architecture boundaries; validation itself writes no
files. Existing unrelated working-tree edits remain untouched.

### CLEAN-01 Contract Delivery

**[Code-Confirmed]** `ocr_cleanup_contract.py` provides strict frozen Pydantic
models and pure helpers, with no API/provider/repository wiring. The model receives
only `schema_version=ocr_cleanup_plan_v1` and deletion items. A quote is 1..512
Unicode code points and must not be all whitespace; optional adjacent contexts
are null or 1..128 code points, with exact whitespace preserved. At most 128 items;
zero is a valid no-op. Item reasons are `ocr_garbage` / `recognition_fragment`.
Unknown fields, replacement actions, malformed Unicode and duplicate JSON keys
are rejected. Provider schema requires explicit null for unused contexts; the local
parser also accepts omitted contexts. Parsing errors expose fixed codes, not input.

The raw JSON response ceiling is 1 MiB UTF-8, an independent parsing resource
guard, not a token allowance or OCR quota. These v1 limits bound proposal handling
without truncation; they do not authorize deleting that much text. Actual prompt
output reserves/dynamic model capacity are CLEAN-02; excessive-deletion safeguards
and exact quote resolution are delivered separately in CLEAN-03 below. No semantic cleanup accuracy is claimed.

`CleanupBinding` contains operation/document/namespace, existing ExtractionIdentity,
canonical text digest/length, positive structure_version and structural fingerprint,
nonnegative expected_reading_revision, explicit canonical/readable representation,
prompt_version, policy_version=ocr_cleanup_policy_v1 and
consent_scope=ocr_cleanup_reading_overlay_v1. `BoundCleanupProposal.bind` checks
identity/text digest/length against the accepted ExtractionResult; no source is
modified. It is not a consent verifier, locator validator or commit authorization.
The binding never accepts model-generated technical identity or source positions.

`cleanup_eligibility` is format-neutral: explicit ocr is eligible, fast/native is
not_applicable, unknown modes skip. Auto requires `ActualOcrEvidence`, normalized
by a future trusted extraction handoff, with matching full identity and text hash,
ordered nonoverlapping canonical OCR spans within text bounds and optional physical
page indexes. Empty verified spans mean native-only; missing, stale or invalid
evidence skips. This contract does not yet read/normalize persisted Auto diagnostics;
CLEAN-05 owns that handoff. A caller cannot infer eligibility from a filename,
profile, garbled words or a requested Auto mode alone.

`CleanupOutcome` fixes terminal status/count/reason consistency. Null proposed_count
means no valid plan count is known; it is not reported as zero proposals. For known
counts, proposed=applied+skipped, with duplicates/rejections counted as skipped by
future validation. Only completed/partial may report content changed, and require
positive applied count and reading revision. Partial requires skipped items; no_changes
requires an empty plan. Failure/skip reasons are allowlisted. Structure change_kind
is absent from this independent result. The executor must later verify revision
advancement against the binding and supply authoritative storage state.

Evidence: `scripts/test_ocr_cleanup_contract.py` has 19 passing tests, including
schema ownership, byte/item/Unicode boundaries, exact text, duplicate keys,
evidence/source mismatch, frozen binding/JSON round-trip and all terminal states.
With semantic structure, generic extraction and change-comparator regressions:
`PYTHONPATH=.:scripts .venv/bin/python -m unittest scripts.test_ocr_cleanup_contract
scripts.test_semantic_structure_plan scripts.test_generic_extraction
scripts.test_enhancement_change` passes 80 tests. Initial run without scripts in
PYTHONPATH had a test import error; corrected invocation passes all. No live LLM,
production writes, source extraction, API changes or UI behavior changes.

### CLEAN-02 Prompt And Planner Delivery

**[Code-Confirmed]** Structural prompt v17 adds conditional OCR-noise guidance:
do not use unmistakable corruption for titles/themes/boundaries or interpret it as
symbolism, but preserve ALL canonical text and semantic splitting autonomy. This
general conditional warning does not classify a document as OCR and never emits
cleanup items in the structural schema. Common parser behavior is unchanged.

Historical CLEAN-02 delivered `ocr_cleanup_prompt_v1` and `plan_ocr_cleanup` with
`structure_ready=True`; CLEAN-06 subsequently wired the enhance route and CLEAN-10
updated the prompt to v2. This boolean/candidate coupling is not the new service
contract: SCLEAN-01/02 replace it with validated cleanup input from either existing
Common hierarchy or an in-memory split candidate. Until implemented, the runtime
helper retains the old precondition; standalone API support must not be claimed.

Cleanup v1 sends canonical text, not the whitespace-normalized structure view.
The binding must explicitly use canonical_text_v1 and the current cleanup prompt
version; mismatches skip instead of silently changing interpretation. Both plans
still refer to the same immutable extraction. JSON-framed source parts concatenate
to the complete source, without truncation: explicit OCR has one eligible part;
Auto uses proven OCR intervals and marks native gaps as context-only. No model IDs,
offsets or hashes are introduced. Exact coordinates/deletion permission still require
CLEAN-03 validation; prompt instructions alone are not a source-quality guarantee.

Instructions protect meaningful multilingual names/quotes, ancient text, formulas,
poetry, lists, page labels and headings; uncertain fragments stay. Embedded source
commands are untrusted data. Only minimal exact garbage quotes and adjacent context
are requested, with no rewriting/repair/search and an explicit empty-plan option.
A fixed fixture SHA-256 snapshots the complete v1 prompt/schema serialization.

The existing SemanticPlanProvider now accepts an optional schema factory; default
structural schema/output behavior is unchanged. Cleanup uses its own provider schema
and complete-request capacity check, including prompt, schema transport, safety
margin and output allowance. Output-override providers reserve their model maximum;
legacy adapters retain their configured reserve. Missing tokenizer uses the existing
labelled UTF-8 estimate. Overflow or unknown capacity skips with zero generation
calls, without changing a structural result or shared provider defaults. No source
truncation, cost quota or automatic retry loop is added.

Provider/timeout failures and invalid cleanup JSON yield independent failed outcomes
with zero applied items. Valid output, including an empty plan, is returned as a
BoundCleanupProposal, never reported as applied/completed cleanup. Source/prompt
changes, ineligible evidence and pre/post-call cancellation produce safe reasons.
Cancellation is checked before dispatch and after return; in-flight transport timeout
is provider-owned, not an interruptible job guarantee. No persistence/LLM workflow
rollback, layout work, profile calls, extraction or OCR are performed by this helper.

Logs correlate a hashed operation identifier with assembly/provider timing, total
elapsed time, capacity measurement/reserve, proposal count and fixed reason codes.
They omit source, response bodies, quotes, caller operation text and exception text.
New skip reasons: structure_not_ready, cleanup_policy_changed, capacity_unavailable.
Evidence: 19 planner tests plus existing contracts/semantic/budget/metrics/extraction/
change/full-source/candidate/autonomy/preflight/repair tests pass as one 152-test run
with PYTHONPATH=.:scripts. Actual model generation and real-book semantic acceptance
remain CLEAN-10; only fake providers were used. No deployment or runtime API wiring.


### CLEAN-03 Exact Resolution And Safety Delivery

**[Code-Confirmed]** `ocr_cleanup_validation.py` is a pure pre-persistence helper.
It consumes the bound proposal, accepted ExtractionResult, independently supplied
expected binding and derived in-memory StructuredDocument. All binding fields must
match the accepted snapshot, including namespace, operation, source identity/hash,
existing structure fingerprint/version, reading revision and prompt/consent policy.
The derived hierarchy is source-coverage validated; its ID and boundary-origin
evidence must be coherent. The binding refers to the initial snapshot, not to an
invented persisted version of the newly split candidate. No DB, provider, OCR,
profile, artifact mutation or automatic review resolution occurs here.

Exact quote plus adjacent prefix/suffix must have one occurrence, including
overlapping occurrences. No fuzzy, case, whitespace or Unicode normalization.
Canonical code-point intervals are mapped to one candidate section and chapter.
Readable-view inputs must reproduce the trusted source-mapping builder exactly;
if a deletion would include normalized-away characters it is skipped, not widened.
Optional physical page mappings must be ordered/in-bounds and cover nonblank text;
cross-page deletions retain clipped evidence for each page. Page metadata is not
a quote disambiguator. Auto deletions require full actual-OCR span coverage, with
no deletion across native gaps. Explicit OCR remains format-neutral.

Source-origin heading lines, caller-supplied user-edit ranges and protected retention
ranges cannot be deleted. Generated navigation titles do not create fake source
anchors. Cross-section deletions and unsafe alphanumeric/CRLF/combining/joiner
boundaries are skipped. Exact duplicate intervals coalesce, counting later items
as rejected duplicates. Every participant in a nonidentical overlap is rejected,
independently of item order; adjacent nonoverlapping spans remain independent.

Policy v1 permits at most 4096 canonical code points per automatic batch, and at
most 10% of BOTH all characters and non-whitespace characters. Equality is allowed;
over-limit subsets are rejected together rather than truncated in proposal order.
Whitespace/front-matter padding cannot authorize deletion of all document text or
all main-body text. These are reversible-content safety guards, not OCR usage/cost
quotas. Boundaries are calibrated against deterministic fixtures only; representative
real-book suitability and meaningful-text false-deletion acceptance remain CLEAN-10.

Result: immutable canonical deletions with backend section/chapter IDs, quote,
reason and page evidence, plus one safe rejection code per rejected item. Counts
sum to the proposal count; empty proposals are valid. Diagnostics expose counts
and allowlisted reasons, not quotes/IDs. This is not a terminal CleanupOutcome,
commit permission or evidence that any review region was fixed. CLEAN-04 must
revalidate current candidate/source/revision at application and final CAS commit.
Exact location does not establish semantic uselessness: meaningful-retention tests
exercise explicit protected ranges, not an automatic multilingual semantic detector.
An incorrect model proposal outside protected ranges remains a semantic risk;
do not advertise perfect OCR repair or remove review entries based on this result.

Evidence: 31 offline tests in `scripts/test_ocr_cleanup_validation.py`, covering
immutable source/candidate, repeated/contextual quotes, Unicode/CRLF, readable
mapping, pages, headings, manual edits, multilingual retention, overlaps, aggregate
limits, stale scope, Auto evidence and privacy-safe diagnostics. Combined CLEAN-01/02,
semantic/budget/metrics/extraction/change/full-source/candidate/autonomy/preflight/
repair regression run passes 183 tests. No paid calls, production mutations or
runtime integration. Next is CLEAN-04's reversible overlay and atomic commit.

### CLEAN-03 Governance Receipt (2026-10-22)

Inspected: AGENTS.md, proposal.md, high-level-design.md, docs/modules/index.md,
progress.md; design/checklist pairs for document_structure, document_preparation,
app, scripts and Deep_Reader_UI. Existing unrelated changes were preserved.
Modified markdown in this task: structure design/checklist (verified CLEAN-03
delivery and corrected per-region auto-resolution/Undo requirements); app design/
checklist (future authoritative review-state handoff); UI design/checklist (future
resolved-queue removal without redundant Ignore); progress.md (backend-only 3/8).
No global architecture, preparation or scripts markdown edits in this task.
Evidence: validator/tests above; auto-resolution is a confirmed design requirement,
not implemented behavior. Validation: 183 tests pass, changed checklist timestamps
are 2026-10-22; no hierarchy authority, hidden writes or UI truth introduced.

### CLEAN-08 Review Projection Support (2026-10-23)

Supporting the UI-owned task: ReadingContentService.read optionally includes a
read-only cleanup_audit projection of applied batches, newest first, ten per page.
It exposes counts, revision/time, exact retained quotes, allowlisted reason/page
references, remaining span count and latest-action undoability. Internal operation
IDs/bindings/canonical offsets are excluded. Audit remains after Undo, while current
cleanup_spans determine whether deletions are still active. Non-applied proposals
have no applied-batch history; skipped counts describe rejected proposals in an
applied batch. Default reads omit this history; no LLM/OCR/profile or storage writes.

Snapshot responses add undo_action from the existing single-level slot. Explicit
edit/delete/discard/Undo transactions advance an existing task_layout.reading_revision
only if it matched the prior reading revision. Other cache keys and all canonical
units/IDs remain unchanged; an absent or already stale layout is never made ready.
Projection validation still occurs before commit. This fixes post-Undo page reload
without implicit read-time preparation or extra hierarchy saves. Current-state
revision CAS/idempotency and generated-output invalidation/notes retention are unchanged.
Evidence: UI test-cleanup-audit.py (five new plus 24 imported cases), 237 overlapping
backend regression cases and desktop/mobile actual API/disposable SQL tests. Completion
is UI CLEAN-08 only; CLEAN-09/10 remain open and no backend progress total is advanced.

### CLEAN-04 Overlay And Atomic Storage Delivery (2026-10-23)

**[Code-Confirmed]** `ocr_cleanup_overlay.py` implements CleanupApplication and
pure staging against the locked authoritative baseline plus the already-built
in-memory split candidate. It rechecks binding/source/version/revision/fingerprint,
reruns exact validation with existing edits/deletions protected, and verifies the
effective projection before writes. No LLM/extraction/profile work occurs under
the transaction. Validation or new-overlay mapping failure yields a skipped cleanup
and retains the uncleaned candidate; stale preconditions/manual-edit mapping conflicts
reject the operation to preserve user work. Final storage errors roll back all writes.

Reading state remains under parse_provenance.reading_content, extended with
cleanup_spans (canonical half-open positions, exact quote, reason, item index and
page evidence), cleanup_source_sha256, cleanup_receipts and source_edits. Temporary
candidate chapter/section IDs are deliberately excluded from persisted span records;
storage-assigned IDs cannot invalidate canonical targeting. Receipts bind the full
proposal/candidate/evidence digest and store the historical outcome/change kind.
Same-operation replay does not advance revision, duplicate audit or apply deletion
again; changed commands conflict. Receipt history is bounded at 1000 operations,
a correctness/storage guard rather than an OCR quota.

File storage exposes internal commit_cleanup_enhancement: exclusive lock, one
baseline read and one atomic file replacement. PostgreSQL commit_enhancement gains
an optional internal cleanup argument, using its existing locks and transaction.
Unchanged/title-only cleanup updates reading metadata and invalidates generated
artifacts with stable hierarchy/version; structural replacement includes staged
reading state in the single hierarchy save. The private preserve-reading-state flag
is honored only with an existing transaction. Normal hard-reparse behavior remains
unchanged. Core profile is reused; the existing structural branch still invalidates
only its derived structure metadata. No public endpoint initiates cleanup yet.

Reading projection now handles pre-unit canonical spans as well as explicit block
edits. Original raw_text, canonical section content and page maps remain persisted
unchanged. Exact ordered unit content maps to section coordinates; render block
spans map within those units before display normalization. Surviving IDs are not
renumbered. Fully deleted units/sections stay empty, without raw-text fallback.
Unmappable existing units make automatic cleanup skip, never guess offsets or
silently expose deleted text. CLEAN-05 still owns production preparation/cache
handoff and canonical-to-effective unit construction, rather than assuming that
the existing preparation pipeline already consumes this new state.

Manual block edits win. Structural replacement carries them into canonical source
edits because old block IDs will no longer exist; ambiguous/partially overlapping
replacement mappings fail safely. Cleanup-only preserves notes and advances reading
revision, clearing generated outputs/ancestor caches. Structural changes still
clear all interactions under existing consent. File metadata writers reject old
structure versions; existing PostgreSQL guards and generation-revision guards
reject late writes. Cumulative effective document/main-body emptiness is checked,
not just each new proposal's deletion ratio.

Source review auto-resolution requires matching source-file SHA, canonical text
SHA, an explicit trusted region canonical_span, exact nonempty raw_candidate, and
full interval coverage by accepted cleanup. Missing/partial/stale mappings stay
unresolved. Current loader regions generally do not carry this mapping yet; CLEAN-05
must provide evidence, never manufacture it from a page number or model assertion.
Resolved regions leave the active queue after commit but retain original evidence
and audit. Existing Ignore remains available for unresolved regions. Single-level
Undo restores the previous spans, manual reading state and region decisions,
advances reading revision and invalidates generated data again; it does not restore
old hierarchy or generated artifacts. A subsequent manual action supersedes the slot.

Operation-hashed logs cover staging, application skip, commit preparation/completion,
replay, lock conflict and rollback, with timing/counts and no quotes or source bodies.
PostgreSQL unchanged/title-only logs commit_ready before context-manager commit,
not fabricated committed success; final execution reconciliation is now delivered
by app CLEAN-06 using the atomic operation receipt.

Evidence: 24 file-backed and 27 disposable-PostgreSQL cases in
test_ocr_cleanup_overlay.py / test_ocr_cleanup_overlay_postgres.py cover real storage
reload, one-save boundaries, partial cleanup, all five target projections, empty
sections/units, manual edits, note retention/generated invalidation, stale writes,
replay/concurrency, fully/partly resolved review, Undo, source changes and injected
post-write rollback. Combined regression: 304 cases, 303 passed, one existing
private-source opt-in test skipped. This is storage acceptance on controlled text,
not live-book/LLM semantic acceptance; CLEAN-10 remains open. No production books,
Docker deployment or paid model requests were touched.

Governance receipt: inspected global AGENTS/proposal/high-level-design/module index/
progress and the design/checklist pairs for structure, DB, preparation, section_tasks
and scripts. Modified structure design (implementation/limits), structure checklist
(CLEAN-04 evidence/date), DB design (internal transaction contract) and progress
(backend 4/8 only). Existing unrelated edits were retained. Validation checks
architecture boundaries, checklist dates and diff whitespace; no UI documentation
or code changes in this task.

## Reading Content Overlays (2026-10-21)

OCR-REVIEW-03..06 implementation: reading_content.py owns explicit edit/delete,
discard-region and single-level undo commands. No add/insert command exists.
The existing hierarchy and immutable source remain truth; parse_provenance stores
source_review evidence and reading_content overlays/audit/receipts. Pure reading
projection preserves block IDs and tombstones; never persist the projected hierarchy.
Commands require source token, expected revision and idempotent mutation ID. Edits
accept 1..10000 Unicode code points; reject missing/deleted IDs, stale sources and
conflicting receipts. Manual edit/delete has no product operation-count limit;
the separate automatic-cleanup receipt guard is not a manual correction quota.
File writes use the artifact lock and atomic save; required artifact clearing follows
the explicit Keep/Clear/Delete policy in the same commit. Late ordinary saves preserve overlays and cannot restore pre-edit
artifacts. Hard structural reparse clears overlays; no-op/title-only changes preserve
them. Edit with Keep retains the block Note; edit with explicit interaction clearing
and block deletion remove that block's Note. Other notes remain. Undo restores text,
not purged interactions; profile/raw text are not rewritten. These targeted policies
are implemented in reading_content_artifacts.py and the repositories. Durable review
mapping and removal of manual-delete auto-Ignore remain pending in SRMAP-01..08.
See doc_loaders Reviewable OCR Recovery for coverage policy, guards and acceptance.

## Common Multilingual Validation Mismatch (2026-10-21)

**[Code-Confirmed]** High-priority correctness issue, recorded only; no fix has
been implemented. CommonSectionSplitter first detects candidates using
DocumentStructureLanguageRegistry, then calls filter_common_sections. The final
heading_decision uses its own predominantly Chinese/English _NUMBERED/_REGION
patterns and has no language/rule-evidence input. The supplied normalizer does not
transfer the registry's reasons for accepting a candidate. Consequently a heading
recognized by a language-specific rule can lose its strong-heading status and be
merged into preceding content when generic whitespace/prose tests are not met.

This does not mean all non-Chinese/English headings fail: some pass generic layout
conditions. Affected-language coverage and real-book incidence remain unverified.
Do not describe this as merely a threshold that needs global relaxation.

Future work COMMON-LANG-01 must first reproduce candidate-versus-final disagreement
with exact source spans and rejection reasons, then design shared language-aware
evidence/validation ownership. Candidate recognition is evidence, not unconditional
acceptance: prose, lists, TOC entries and false headings still need rejection.
Retain exact canonical text, hierarchy ownership and the valid single-body outcome
for genuinely untitled articles. Do not add LLM calls, profile regeneration or
metadata classification as parser authority. Audit shared heading_decision callers
(including projection and LLM plan application) before changing its interface.

Acceptance must cover the current language registry, including French/German and
other non-Chinese/English headings, controlled single/double-newline variations,
numbering/region markers and false-positive prose/list cases. Verify both candidate
classification and final hierarchy, not only regex matching; conserve all source
characters and retain Chinese/English and shared-caller regressions. Language
entries without trustworthy fixtures remain explicitly unverified.

Governance: inspected AGENTS.md, proposal.md, high-level-design.md, module index,
progress.md and this module's design/checklist. Modified only this design (cause,
scope and acceptance), checklist (one pending task) and progress (pending-track
link). Evidence: section_splitter.py split/filter handoff, heading_acceptance.py
heading_decision/filter_common_sections and document_structure_language_registry.py.
Validation: documentation-only scope, pending status and timestamp checked;
runtime reproduction/tests deferred as requested. No source code or parser changes.

## Generic Extraction Acceptance Support (2026-10-21)

Supporting scripts EXTRACT-10: direct live Madame Bovary API acceptance exposed
an unlocated generated_boundary_inside_prose failure after a source-origin title
was downgraded to generated by validate_candidate. Semantic quote validation must
classify that effective origin before checking sentence/paragraph placement, so
repair receives the offending boundary index and source evidence. Do not weaken
exact quote matching or body preservation. Add a deterministic regression and
repeat real API/model/commit/layout acceptance before closing EXTRACT-10.

## Explicit Annotation Mutation Boundary (2026-10-21)

Supporting SN-07/14/15: DocumentArtifactRepository exposes an atomic artifact
callback and explicit rename callback. File adapter uses a per-document POSIX
lock with atomic replacement; initial writes/reparse carry interaction_epoch.
Ordinary full metadata saves preserve note/public_summary from the locked current
document. Only explicit annotation mutations or reparse cleanup can replace them.
The PostgreSQL adapter delegates to its owning transactional store. This adds no
hierarchy level, root sections, task-layout mutation or parser authority.

## Semantic Plan Ownership (2026-10-21)

Confirmed replacement requirement: the model plans semantic divisions; the backend
owns locations, slices and entity IDs. The v11 reference-copying mitigation below
is historical, not the target architecture. Do not keep improving ID-repair prompts.

The model-facing contract contains only an ordered list of boundaries with title,
level, origin, role and exact `start_quote`. Real-model experiments showed that
`preceding_quote` was repeatedly misread as the previous section title; that field
has been removed, not silently ignored. Copy a short opening phrase (at most 32
characters); extend its exact continuation within that bound to disambiguate a
repeated opening. Unresolved repeated quotations remain invalid. No boundary_id,
segment_id, offsets, entity IDs or occurrence fields are
accepted. Source is supplied once in full as readable text, without locator tables.
Model instructions are not source evidence; copied text remains untrusted data.

Backend matching uses the deterministic readable-to-canonical map. Only exact
matches are permitted. Missing or ambiguous matches reject the proposal rather than choosing a
nearest/first occurrence. Reject duplicate positions and invalid semantic cuts.
Paragraph indentation includes Unicode whitespace (notably U+3000 in the reported
book), not only ASCII spaces. Skipping indentation for the boundary check must not
change quoted characters, slice offsets or source text; mid-sentence whitespace is
still not a valid semantic start. Batch these failures with quote errors so local
repair identifies every rejected division, rather than reporting an unlocated error.
Convert uniquely resolved quotes into the existing internal executable-plan form;
that adapter may generate internal locators, never expose them to the model. Local
application slices original canonical text and proves complete ordered coverage.
Entity IDs are generated by existing backend hierarchy/persistence code.
An explicitly single implicit body is structurally anchored at source zero and
must quote the exact beginning; this special case does not authorize first-match
selection for repeated interior quotations.

Keep at most two validator-feedback corrections after an initial proposal. Feedback
identifies quote-field errors without an ID list. Every corrected request is still
budget-checked and bound to the original source/model configuration. Invalid or
exhausted proposals write nothing. No-op/title-only/structural commit semantics and
automatic layout preparation remain unchanged; no API/UI redesign is required.
The full readable source is plain text, once, with length-framed metadata; it is not
an escaped JSON string. Provider Structured Outputs enforce the typed response
shape; local validation still checks meaning, exact quotes and coverage. A failed
quote may receive exact-prefix source excerpts for model correction (all candidates
up to eight, otherwise none). Excerpts never authorize a boundary, select a nearest
match or mutate a quote. The model must submit a replacement that passes the same
unique exact-match validator. No excerpt/quote values enter application logs.

Corrections are local repair requests, not another full-book planning pass. The
initial request contains the entire source; repairs contain the previous proposal,
safe validation errors and relevant exact-prefix source excerpts. Each error is
also identified by its rejected quote and title, avoiding ambiguous array numbering.
Evidence includes exact short opening options copied from each candidate excerpt;
these are textual evidence, not model-facing location IDs. Generated-title/region
violations are reported in the same batch as quote and boundary failures, and ask
the model to use main_body rather than fabricate an original source region.
The model must repair a supported quote or merge an unsupported division into an
adjacent valid section. The backend never silently drops boundaries or accepts a
partial invalid proposal. Removing a model-proposed cut cannot remove body text:
the final validated intervals must still cover the full canonical source exactly.
Repair prompts have their own resource/token checks, while the initial full-source
budget/consent identity is revalidated before each call. Logs record prompt size,
evidence count and attempt outcome without private excerpts.

Planner-only provider adapter reserves and requests the configured model's maximum
supported output (32768 for this model), with a per-call 300-second timeout for the
larger output. Shared insight/quiz defaults are unchanged. Preflight and every call
count the additional API response schema as input and reserve output exactly once.
The initial 3276-token experiment ended mid-JSON; later attempts exposed repeated
and altered quotations, so neither successful JSON nor mock tests constitute final
acceptance. Final v15 acceptance is recorded below, separately from those failures.

SEM-01..04 own implementation and acceptance. Real-book generation must use the
configured model with original storage read-only, require effective structural
change, and validate exact body preservation. Replay accepted live output through
isolated API/PostgreSQL/real layout. Synthetic tests supplement, never replace,
this evidence. Old V2 locator tests remain compatibility/internal-contract tests,
not proof that the new model-facing contract works.

Final v15 evidence: real case `085747b65094d52f` passes independent attempts
`20261021T044839792942Z` and `20261021T044933593384Z`, respectively 3 chapters/33
sections after two calls and 4 chapters/27 sections after three calls. Both produce
actual structural change and preserve all 147401 canonical characters (SHA-256
`ff9b7b1ce6b1684db85e4cf4ba8711f782f071a435ffd64c14642ebb44670ffb`). Original
storage/interactions remain unchanged. First result passes real API/disposable
PostgreSQL conditional commit, full seeded interaction deletion, paid embedding
and actual layout resolver, prepare/read HTTP 200 with 107 task units. API replay
is persistence/layout evidence, not a third independent planner generation.
Backend regression acceptance runs 180 tests: 179 pass, one private-source opt-in
skipped; the explicit real-book harnesses above run separately and are not skipped.

Observed failures led to typed ID-free output, plain source framing, native schema,
request-local output allowance, Unicode-indentation recognition, batched role/quote
diagnostics and local exact-evidence correction. Deduplicate repeated rejected
quotes before constructing evidence to prevent quadratic repair-prompt growth.
No fuzzy location matching, unchecked slicing, model-generated IDs or body rewriting
was introduced to obtain a pass. Grouping/title quality still varies (one accepted
plan groups 31 subsections under one chapter); this is not human editorial approval
of every label or a universal reliability claim. Old failed attempts stay private.

Final lifecycle runner also passes UI build and browser suites at 1280px/390px;
screenshots inspected, diff checks clean. Local Compose API was rebuilt after
acceptance: health endpoint succeeds and container imports v15_semantic. Deployment
does not reparse the original book or clear its interactions.

## Real-Book Locator Recovery (2026-10-21)

Historical v11 mitigation, superseded by Semantic Plan Ownership above.

This supersedes v8's numeric position-map wire format and single-attempt invalid-plan
handling, not the strict source/coverage/commit guards. The reported private book
has 147401 canonical characters (SHA-256
`ff9b7b1ce6b1684db85e4cf4ba8711f782f071a435ffd64c14642ebb44670ffb`).
Live reproduction found invented boundary IDs and repeated chapter/first-section
positions. Token capacity and OCR were not the cause. Historical error logs alone
could not distinguish missing IDs from occurrence errors; current safe diagnostics
include boundary index, reference membership, occurrence validity and locator form.
No raw ID, title, source excerpt or provider response is written to application logs.

Prompt v11 uses `located_source_passages_v2`: each supplied ID is adjacent to its
actual passage text. Concatenating passages reconstructs the same readable segment;
canonical source offsets remain server-owned. No duplicate source/preview prose is
sent, no source range is omitted, and exact segment-anchor fallback remains valid.
Instructions after the long source repeat locator uniqueness and generated-title
rules. For this book required input is 169964 locally counted tokens including
framing, versus 992085 available. These are estimates, not provider usage metrics.

Prompt changes alone were insufficient: v11 passed once and failed on a repeat.
The planner now allows at most three proposals per request: initial proposal plus
two validator-feedback corrections. Both planner entrypoints share this policy.
Each correction includes the rejected proposal as untrusted data, safe diagnostic,
explicit invalid/duplicate IDs and the actual allowed reference list. Only the model
chooses replacements. The server does not guess offsets, fuzzy-match IDs, remove
duplicates or silently accept an invalid plan. JSON/schema/reference failures and
specified source-title/semantic-boundary errors may be corrected. Internal failures,
oversized responses, capacity/configuration changes and provider transport errors
are not blindly retried here; existing transport recovery remains separate.

Before every call, validate the original preflight-bound source/configuration budget
identity. For correction calls, also enforce actual expanded prompt resource/token
limits before dispatch. A repair is not permission to change model/source or exceed
capacity. Log attempt number, validation reason, accepted outcome and exhaustion.
All proposals stay in memory until full validation/application succeeds; exhausted
failures preserve the original structure and interactions. Valid no-ops do not retry.

Final implementation real-model evidence: private case `085747b65094d52f`, independent
attempts `20261021T035533111403Z` and `20261021T035655611230Z`, both accepted with
gpt-4.1-mini after one correction (two calls each). Results respectively 12 chapters /
25 sections and 28 chapters / 28 sections, with exact canonical body preservation.
The first accepted response was replayed through real preflight/reparse APIs and
disposable PostgreSQL: structural commit, seeded interaction cleanup, real layout
resolver/embedding service, prepare/read HTTP 200 and 106 task units. Original
production snapshot was verified unchanged. Replay tests persistence separately;
they are not counted as additional live planner successes. This evidence fixes the
reported case, not a statistical guarantee or human approval of every chapter label.

Future feature acceptance must include representative real-book data and, for
model-dependent behavior, explicitly authorized real-provider runs. Mocks remain
fast regression/fault-injection tools, never sufficient evidence of live quality.
Private files/requests/responses stay under gitignored `data/test-books/` in the backend;
versioned evidence contains hashes, counts, safe failure shapes and test code only.
See scripts design for reproduction and privacy requirements.

## Enhancement Autonomy And Change Semantics (2026-10-20)

Status (2026-10-21): ENH-REV-01 comparator and ENH-REV-04 autonomous prompt implemented;
db ENH-REV-02 and app ENH-REV-03 are also complete in their owner checklists.
UI ENH-REV-05 is complete; integrated ENH-REV-06 engineering acceptance is complete.
That historical engineering acceptance did not evaluate a live model. The later
Real-Book Locator Recovery section records the explicitly scoped live acceptance.
This remains the structural enhancement policy. The planned Optional OCR Cleanup
Plan section adds a separate pre-commit reading-overlay phase after split validation; it does not
relax canonical conservation. Earlier FULL/ENH/NATIVE completion records
remain historical engineering evidence, not acceptance of these new semantics.

### Confirmed Requirements

- Common conservatively recognizes source structure. Its "when uncertain, do not
  split" rule must NOT govern LLM semantic planning.
- LLM enhance autonomously organizes useful chapters/sections and creates navigation
  titles in the document language. Existing headings/TOCs are not prerequisites;
  Common output is advisory, not the shape to preserve or the authority to override LLM.
  No fixed chapter quota, equal-length splitting or obligation to retain one body.
- Hard guards preserve exact canonical body text, order and coverage, valid locators
  and coherent parent/child structure. Generated titles describe navigation, not
  fabricated source headings. Strict source-heading evidence applies only when a title
  claims source origin. Semantic planning is not Common heading detection.
- An unchanged structure plan does not itself commit a new version, clear interactions,
  rebuild units or trigger an automatic retry. Report "no structure change", not
  enhanced quality. Planned cleanup is separately evaluated; actual reading-content
  changes have their own revision/invalidation/layout consequences, not a fake
  structural change.
- Navigation-title-only changes are effective changes, but preserve interactions.
- Boundary, hierarchy level or parent/grouping changes are body-structure changes:
  clear ALL document interactions on successful atomic replacement. Partial reparse
  and selective interaction retention are explicitly out of scope.
- Model failure/invalid plans retain original data. A valid single body is not
  automatically invalid merely because the document is long; compare the actual
  result, rather than forcing arbitrary splits or treating validity as improvement.
- Recovery clarification: transient LLM HTTP/provider failures may retry; cost
  controls are deferred. Token overflow blocks before generation. Unknown outcomes
  reconcile to avoid duplicate commit/cleanup, while confirmed downstream failures
  resume preparation/read without regenerating completed work. The automatic
  no-op retry prohibition is not a blanket ban on LLM failure recovery.

### Deterministic Change Classification

Proposed internal result: `unchanged | titles_only | structure_changed`.
Compare the validated, applied candidate against the authoritative persisted
hierarchy on the SAME accepted canonical source. Do not compare raw LLM JSON.
First verify identical canonical text/hash and complete coverage; changed body text
is invalid, not another allowed change category.

Build an ID-independent ordered structural signature from canonical intervals,
hierarchy levels and parent/group relationships (including target-affecting roles).
Generated IDs, timestamps, request/provenance fields and task-layout-derived metadata
are not structural improvements. Compare navigation labels separately using a
documented deterministic representation; do not use another LLM to judge equivalence,
collapse distinct labels as synonyms, or normalize body text.
Same structural signature + same labels => unchanged; same signature + changed
labels => titles_only; changed signature => structure_changed. Label updates map to
existing nodes by structural correspondence, never newly generated candidate IDs.
Use the same comparator for reporting and commit-time revalidation.

Implemented comparator (ENH-REV-01, 2026-10-21):
`enhancement_change.compare_enhancement_documents` takes the existing hierarchy,
applied candidate and caller-supplied accepted canonical text. It validates both
documents' exact ordered section coverage, document identity, unique node IDs and
explicit parent links. PostgreSQL's omitted `raw_text` is allowed only because
section slices still prove exact coverage; nonempty `raw_text` must match exactly.
Body Unicode is not normalized. Invalid inputs raise safe reason codes, never a
permitted change kind.

The nested signature includes chapter/section levels, chapter/section roles and
section intervals; nested grouping establishes parent correspondence independently
of IDs. Labels include chapter title, section title and section container title;
only surrounding whitespace is stripped and None/empty labels are equivalent.
Case, internal whitespace and Unicode differences remain label changes.
Parser-origin fields (`section_kind`, `is_implicit_section`), ordinal metadata,
provenance, source path, library title, derived units and artifacts are excluded.
They cannot become evidence of body-structure improvement.

Equal signatures return immutable chapter/section correspondence with the existing
IDs and candidate positions/IDs. Structural changes return no reusable mapping.
The safe diagnostics projection contains change kind and existing/accepted candidate
chapter/section counts only. Raw provider proposal counts remain the planner's
diagnostics; accepted counts must not be misreported as raw proposal counts.
No source, labels, IDs or provider responses enter this diagnostic projection.
This helper performs no LLM calls, persistence or logging side effects; wiring
commit-time revalidation and emitting diagnostics belongs to ENH-REV-02/03.
Evidence: 20 offline comparator tests plus 42 existing plan contract/application
regressions pass. No live-model quality claim or production mutation is implied.

### Incident Evidence And Limits

Read-only investigation of 2026-10-20 UTC 14:42:32 and 14:44:23 executions:
request IDs `18a1d231da0145df8a6d460426f4b523` and
`e922c6f134b14529ad8b0a7ec7dd9bd9`. Both reparse requests returned HTTP 200;
database parse events advanced structure versions 1 -> 2 -> 3. The following
task-layout reads returned 409/missing_task_units. These were committed parses with
failed reader recovery, not token rejection or failed provider calls.

Both requests used v7, complete 147401-character source and no omitted ranges.
Required input was 149890 local-tokenizer tokens including framing; available input
was 992085. Both response lengths were 186 characters. Latest stored provenance
records proposed/resolved/accepted boundaries=1, rejected=0, implicit title and
coverage [0,147401). Thus the latest plan was already single-body, not a useful
multi-chapter plan subsequently erased by local heading validation.
Original provider JSON/finish reason/usage were not retained; identical response
lengths do not prove identical JSON or reveal the model's reasoning.

The deployed prompt file matched the inspected workspace SHA-256
`720e7c50aba9ee6c68533caea49769ec905de8b62d0d5c10eb09fb27f8149dda`.
`plan_prompt` does ask for semantic chapter/section improvements. However, it also
prefers the smallest set, repeats anti-splitting cautions, includes "When uncertain,
do not split", and permits single_body without a decision explanation. This
misapplies the desired Common/LLM responsibility split. The misleading instruction
balance is a confirmed design mismatch and a plausible output bias, NOT proven
knowledge of why the provider chose that answer.

Read-only source/mapping inspection: 5003 lines, 4897 short nonempty lines, no matches
for the inspected conventional Chinese chapter-marker or standalone numeric-line
patterns. These narrow heuristics do not prove an absence of semantic structure.
Brief samples span different narrative/business episodes and end matter. The prompt
uses 74 arbitrary 2000-character windows and 4736 references; every window hits the
64-reference cap. Compact text still contains all 147401 characters and exact-anchor
fallback remains available. This is a navigation burden/coverage-risk hypothesis to
measure, not evidence of lost body text or proof that more references fixes planning.
Output allowance was 3276 tokens; a 186-character answer gives no evidence of output
truncation. Do not raise limits, change models or spend paid retries based on this alone.

At the time of this incident, `validate_candidate` checked schema, source visibility
and locators; `apply_candidate` checked boundaries and exact coverage. Neither
compared the accepted plan to current structure for meaningful change. The then-current
commit treated accepted candidates as hard replacement. That explains how an unchanged shape could become a new version
and clear interactions; it does not establish semantic improvement.

### Autonomous Prompt V8 (ENH-REV-04, 2026-10-21)

`structure_plan_prompt_v8` removes the smallest-division objective and the blanket
uncertain-do-not-split instruction. It asks the model to read the whole source,
identify its progression, organize chapters/subsections, then select locators.
Common is advisory, not a template or authority over semantic planning. Sustained
topic/argument/scene/function transitions may become generated navigation titles
without printed headings or a physical TOC. Source-heading uncertainty affects only
the claim of source origin; it does not veto a meaningful generated-title division.

No chapter quota, equal-length chunks or required minimum number of sections is
introduced. Coherent short articles may remain one implicit body; length alone is
not a split rule. Keep claims/evidence, quotations/attribution and lists/introduction
together. Generated titles remain neutral, in the document language, and not invented
source headings, events or analytical conclusions. Exact body preservation, coverage,
ordering, safe semantic positions and strict source-origin validation are unchanged.
Common parser rules, candidate schema and persistence behavior are unchanged.

Locator/output audit: the capped reference map is explicitly described as navigation
assistance, not the full set of allowed boundaries. A useful unlisted transition may
use an exact segment anchor. Synthetic 147401-character full-source assembly still
uses 74 windows, each at most 64 references; it preserves the full source and performs
no provider calls. A representative 12-boundary JSON fits below 3276 even with a
UTF-8-byte upper estimate; this proves format feasibility only, not a recommended
chapter count or sufficiency for every book. Existing configured output allowance,
model/capacity policy and reference cap are unchanged. The v8 short-input serialization
regression ceiling is 11400 characters (observed maximum 11349 in those fixtures),
not a model token limit. Preflight continues measuring the actual assembled prompt.

Safe `enhancement_plan_result` logs add validated outcome, proposed boundary count,
accepted section count, level-1 boundary count, generated-title count and prompt
version. These are application facts, not semantic-quality scores. Existing response
length/duration/failure logs remain. The provider abstraction returns only text, so
finish reason and provider token usage are unavailable at this boundary; do not infer
them or log full responses/chain-of-thought. No provider API was expanded for this task.

Evidence: `test_enhancement_autonomy.py` verifies generated chapters/subsections for
synthetic multi-topic prose without headings, unchanged Common behavior, a legitimate
single body without regeneration, exact-anchor fallback past the reference cap, safe
logs and full-source/output-format audit. Prompt regressions assert v8 autonomy while
retaining source/language/locator constraints. Ten-suite acceptance runs 145 tests:
144 pass, one private-source opt-in skipped, including isolated PostgreSQL/HTTP
conditional-commit and recovery regressions. Deterministic providers validate wiring
and guards, NOT live-model segmentation quality. No paid LLM, private-book processing,
production mutation or deployment; live usefulness remains explicitly unverified.

### Execution Order And Evidence

Eight tasks; 8/8 accepted (100%) as of 2026-10-21: UI ENH-UX-01/02 and
ENH-REV-01..06. UI completion evidence remains in its owned checklist.
1. UI ENH-UX-01: automatic layout-only recovery after confirmed structural commit.
2. UI ENH-UX-02: readiness-only UI, remove model/token diagnostic panel.
3. ENH-REV-01 (here): pure three-way comparison and safe outcome diagnostics.
4. ENH-REV-02 (db): atomic no-op/title-update/structural-replacement branches.
5. ENH-REV-03 (app): consent, result exposure and preparation/recovery handoff.
6. ENH-REV-04 (here): autonomous semantic prompt, distinct from Common policy.
7. ENH-REV-05 (UI): conditional warning, typed outcomes and cache/draft preservation.
8. ENH-REV-06 (here): cross-module acceptance and separate semantic-quality evidence.

No raw book text/prompt/provider JSON is committed to fixtures or ordinary logs.
Proposed diagnostics include prompt/model identity, candidate/accepted counts,
change kind, commit kind, versions, interaction effect and recovery stage. Token
counts are estimates unless provider usage is explicitly available. Capture safe
finish/status information where available, not model chain-of-thought. Test meaningful
multi-topic cases and legitimate single-body cases, not only schema-shaped stub success.
Live paid/private-book evaluation remains explicit opt-in and separately reported.

### ENH-REV-06 Engineering Acceptance (2026-10-21)

`scripts/accept_enhancement_lifecycle.py` is the single acceptance entry. It requires
an explicit disposable PostgreSQL DSN, a running UI, and Playwright/Chromium; it fails
rather than silently skipping database/browser checks. It refuses the private-source
environment option and never loads development credentials or calls a paid model.
The existing PostgreSQL harness creates and removes isolated schemas and temporary
synthetic source files; production documents and services are not modified.

The backend suite covers comparison, autonomous prompt, HTTP outcome/replay,
operation concurrency/expiry, capacity, full-source preservation, plan application,
and task-layout readiness. Inherited conditional-commit tests assert title snapshot
consistency, updated consent, delayed metadata writers, row-lock conflicts and
transaction rollback. There are 155 test executions: 154 pass; the deliberately
opt-in private-source case is skipped. No other skips are accepted.

`accept_enhancement_browser.py` now serves the result lookup route and runs both
the capacity suite and `test-enhancement-lifecycle-live.mjs`. At both 1280px and
390px the actual UI calls real FastAPI/coordinator/PostgreSQL for all three outcomes,
provider timeout -> same-consent retry, lost title/structural response -> result
lookup, and preparation failure -> preparation/read recovery without regeneration.
Provider output and task-unit resolution remain deterministic fixtures. The failed
preparation response and lost HTTP response are injected by browser transport;
successful preparation and all enhancement/result requests use the real backend.

For unchanged results every stored row plus PostgreSQL xmin stays equal; for title
updates document version, node IDs, task units, content blocks, artifact payloads
and parse events stay equal, and persisted snapshot titles match authoritative rows.
Fixtures contain nonempty note/quiz/thinking/analysis payloads, not count-only checks.
Structural results advance exactly once and remove all interactions before explicit
layout-only preparation. Every task-layout read returns 200 and is checked for zero
row writes. Full canonical text is identical after every accepted operation.
Across the integrated run, 14 deterministic provider attempts produce exactly three
structural version increments; transient retries and reconciliation do not duplicate
cleanup. Rejected capacity/consent checks dispatch no generation.

The same entry runs UI confirmation/outcome tests for saved insight and unsent draft
retention, missing/contradictory fields, source selection, unknown/restart handling,
late responses and recovery checkpoints, plus 42 reader identity guard cases.
TypeScript/production build and desktop/mobile screenshot inspection pass. The
existing bundle-size warning remains; no production deployment is claimed.

Reproduction from `Deep_Reflective_Reader`: set `DEEP_READER_TEST_POSTGRES_DSN` to a
local test-capable database, `PLAYWRIGHT_MODULE` to an installed module, optional
`CHROMIUM_PATH` and `UI_URL`, then run
`.venv/bin/python scripts/accept_enhancement_lifecycle.py`. Its final JSON explicitly
separates `engineering_acceptance=passed` from `live_model_usefulness=not_evaluated`.

### Separate Opt-In Semantic Review

Engineering success does not establish that real model boundaries improve reading.
No paid/private-source evaluation was run. A future explicitly authorized evaluation
must use a disposable copy, the configured model and full-source capacity gate; retain
the original source/hash and compare Common and enhanced results without forcing a
minimum chapter count. Include both coherent short prose and multi-topic prose without
printed headings. Review transition coherence, preservation of arguments/quotations,
useful neutral titles in document language, honest generated/source origins and exact
body coverage. Record case-level observations and model/prompt identity separately
from schema validity, chapter counts and lifecycle tests. A legitimate single body is
not a quality failure by itself; fake-provider divisions are never quality evidence.

## Full-Source Enhancement Superseding Plan (2026-10-20)

FULL-01 through FULL-05 implemented and synthetic integrated acceptance completed.
This supersedes enhancement's earlier head/tail preview
and partial-visibility acceptance strategy; do not delete historical evidence or
count the previous six completed tasks as completion of this follow-up. Latest live
v6 requests b75c8cad.../c963c8e6... passed schema and provider execution but failed
single_body_requires_full_source_visibility. Source=147401, visible=120000,
prompt=193828 chars, local input count=122009 tokens; budget accepted. The fixed
120000-character source_preview clamp discarded 27401 source characters before
the model-aware gate. The model returned an implicit single-body plan; exact raw
responses were not retained. Full input may remove this rejection condition but
does not guarantee useful segmentation or improve live model reliability by itself.

New invariant: enhancement sees the entire accepted canonical source through the
compact, locally source-mapped representation or makes zero LLM calls. Remove the
fixed preview clamp and capability-derived excerpt truncation from this execution
path; do not simply raise 120000 to another magic number. No head/tail substitution,
summary substitution, automatic model switch, paid retry or multi-call decomposition.
Keep exact source maps, locator validation, compact references and success-only
atomic replacement/interaction cleanup for body-structure changes. Never relax single-body validation to
accept a partial-view conclusion. Reusable preview helpers outside enhancement may
retain explicit partial-preview semantics; audit callers before changing them.

FULL-02 depends on FULL-01. Assemble full source/context/schema/instructions and
check the complete serialized request against the shared capacity policy. Require
visible_source_chars=source_chars, no omitted ranges, and full_source=true before
dispatch. Check actual selected references still map to canonical spans. Existing
64-reference-per-window ceiling must not silently remove valid fallback anchor
capability. Streaming/count-first assembly may avoid a large duplicate allocation.

Set a separate configurable local assembly byte/work ceiling before expensive
normalization/allocation. Choose and document its numeric default from memory/work
measurement in FULL-02; it is not a model context limit. Exceeding it stops assembly
with a distinct local_resource_limit reason, never truncates the source. Unknown
model capacity stops with capacity_unavailable. Tokenizer failure uses labelled
conservative accounting, not an unlabeled chars/4 guess. Full-prompt over-limit
returns model_capacity_exceeded with counts and zero generation/persistence calls.

Logs/API planned measurement includes model/policy/tokenizer identity, full source
coverage, character/byte/token count with count_method, capacity limits, effective
output reserve, safety margin, available input and rejection reason. No raw source,
anchors, prompt, exception text or secrets. If resource ceiling prevents measurement,
unknown counts remain null, not fabricated. Preflight must reuse the same assembly/
budget helper and execution must revalidate it against current authoritative state.

Five-task order: FULL-01 llm capability contract -> FULL-02 full-source planner ->
FULL-03 app/API preflight -> FULL-04 UI explanation -> FULL-05 integrated acceptance.
Current follow-up progress 5/5 (100%); backend 4/4, UI 1/1.
Budget policy v3 consumes explicit context/input/output capacities and configurable
margin on the limiting input budget; metrics expose context_window_tokens separately.
Full-source assembly now bypasses the fixed preview clamp in both production entries.
Acceptance includes this
147401-character shape, multilingual full coverage, exact/over limit, oversized
schema, unknown model, unavailable tokenizer, local resource protection, changed
source/model/policy between preflight and execution, zero-call failure and unchanged
hierarchy/interactions. Full-source eligibility is not semantic-quality acceptance;
paid model evaluation requires separate authorization and honest reporting.

### FULL-05 Integrated Acceptance (2026-10-20)

The existing acceptance runners now cover the complete full-source path without
changing production behavior. `scripts/accept_enhancement_prompt.py` includes capacity,
full-source and consent suites: 139 tests ran, 138 passed, and the private-document
opt-in case was deliberately skipped. Boundaries, tokenizer fallback, prompt parity,
local ceilings and privacy-safe diagnostics are covered alongside storage rollback.

`scripts/accept_enhancement_browser.py` serves the real FastAPI routes against a
new disposable PostgreSQL schema. Its loopback-only `/fixture` control exists solely
in the test harness, never in the product application. Synthetic accepted text is
147401 characters with English, Chinese and French; dispatch asserts contiguous
exact reconstruction, full_source=true and omitted_ranges=[] inside a deterministic
provider. The accepted source file is deliberately not a parseable PDF and full
preparation is forbidden during the browser run.

At both 1280px and 390px, real preflight rejects exceeded, unavailable and local
resource cases without execution. After successful preflight, capacity shrinking,
configuration loss, resource limits, model changes, policy changes and source-file
changes produce the expected 413/503/409 response and disabled confirmation.
Every rejected execution and every preflight compares before/after stored snapshots
and provider call counts: unchanged hierarchy/interactions and zero generation.
Each viewport then commits once, clears stale interactions atomically, preserves
the exact accepted text and source bytes, and explicitly prepares the reading layout
after its read-only 409. Total deterministic provider calls: two.

Reproduce with `DEEP_READER_TEST_POSTGRES_DSN` set to an explicitly selected local
test-capable database; the runner creates/drops its unique schema. Run from backend:
`.venv/bin/python scripts/accept_enhancement_prompt.py` and
`.venv/bin/python scripts/accept_enhancement_browser.py`.
Browser execution additionally needs the existing `UI_URL`, installed Playwright
via `PLAYWRIGHT_MODULE`, and optionally `CHROMIUM_PATH`. No real API traffic is
forwarded to production documents. All temporary schemas were removed after runs.

UI mock regressions also passed on both viewports, including legacy fields,
transport loss, timeout reconciliation and stale responses. `npm run check` passed
with the existing bundle-size warning; desktop/mobile screenshots were inspected.
This is engineering acceptance only: no paid LLM call, private-source evaluation,
Docker restart or deployment occurred. Live semantic quality/reliability remains
Needs Confirmation and is not implied by the 100% task completion figure.

### FULL-02 Delivery And Resource Evidence

`LLMSectionSplitter.assemble_candidate_request` is read-only and shared by both
candidate entrypoints. It guards local resources before maps/Common parsing, builds
source_preview(raw, None), verifies contiguous full coverage with no omissions, then
assembles instructions/schema/Common advice/full compact source and budgets the exact
prompt string. No provider normalizer/head-tail excerpt is used on this path.
Numeric source_preview budgets and legacy v1 helpers retain their explicit preview
semantics for existing callers/tests. Prompt v7 declares full-source semantics.
The 64-ID/window cap remains only an optimization: anchor+segment fallback beyond
the cap is regression-tested. Canonical maps and application validation are unchanged.

`EnhancementResourceLimits` is constructor-injectable: default source 2000000 chars
(existing source mapper ceiling), UTF-8 source 8000000 bytes, 100000 line separators,
serialized prompt 16000000 bytes. Character/line guards precede map/list allocation;
source encoding is bounded by the character guard. Prompt size is a secondary guard
after serialization; bounded source/windows/references constrain prior assembly work.
These are local process protections, not model capacity. No truncation on rejection.
Changing the source ceiling above mapper support is refused explicitly.

Local tracemalloc measurement on 2026-10-20 used repeated English source sentences,
the real Common context/compact assembly and a synthetic 20M-token-capacity provider
with zero generation calls. At 147401 chars: about 226KB serialized prompt, 8.95MiB
peak Python allocations, 1.813s with tracing. At 2000000 chars: about 3.04MB prompt,
121.46MiB peak, 24.193s with tracing. This supports retaining the existing 2M mapper
ceiling instead of unbounded assembly. Measurements are not RSS, worst-case guarantees,
production latency or model quality claims; line-heavy input has a separate guard.
The final v7 explanatory sentence adds under 200 prompt bytes to this measured draft.

Budget v3 counts complete prompt locally or uses the labelled UTF-8 byte estimate,
adds configurable 256-token transport-framing allowance, and reserves output and
safety margin separately. Framing is conservative policy for the current simple text
provider request, not billed-token precision; future tools/system additions require
reassessment. Logs expose full coverage/bytes, model, required tokens including framing,
count method, reserve and limits without source. Existing budget-exceeded/unavailable
codes are retained for compatibility until FULL-03; local_resource_limit is distinct
and maps through the builder to HTTP 413. Coverage invariant failure never dispatches.

Validation: 80 offline tests passed, including nine new full-source/resource cases,
147401-character both-entrypoint dispatch, EN/ZH/FR full coverage, reference-cap
fallback, exact/one-over budget including framing, unknown capacity, resource guards
before parsing, safe logs, and zero generation/save on failure. Existing API rejection
mapping tests cover 413 for local resource rejection. git diff --check passed.
No paid LLM, OCR, production mutation or Docker restart. FULL-03 now delivers public
capacity preflight, consent identity binding and revalidation; FULL-04 owns UI copy.

FULL-03 supporting changes: builder shares TOC observation/resource checks between
preflight and execution. Budget errors carry safe local decision metadata. A private
prompt/configuration digest and request-local dispatch guard prevent late changes
from bypassing consent. The guard compares the actual prompt and recomputed budget
immediately before complete_text; unknown/over-limit inputs still stop first.
Preparation/app own public projection and authorization. Evidence is counted only
in app FULL-03, not another structure task. Persistent hierarchy authority is unchanged.

## Enhancement Prompt Reliability Plan (2026-10-19)

Status: ENH-PLAN-01..06 engineering delivery and isolated acceptance complete;
live model reliability and production deployment remain Needs Confirmation.
Owner: document_structure; supporting changes to
provider/config/app/API boundaries require their module memory to be loaded before
implementation. Execution order and completion authority: ENH-PLAN-01..06 in this
module's checklist. This follow-up does not reopen historical delivered work or
claim that previous mocked acceptance established live model reliability.

### Evidence And Correction

The latest four live requests (log times 2026-10-19 15:29:31 through 15:30:46,
container clock) returned provider HTTP 200 and then failed with
`schema:exactly_one_boundary_reference_required`. Each recorded source_chars=147401,
response_chars=167 and prompt version v4. Container stdout records prompt length
647232, approximately 4.39 times source length. These are character counts, NOT
token counts. Request IDs: a996acf9dabf4473b09725772cae79b3,
343c6921ec9c45d58935e8be3d6dc8f4, 212a0ffae41e4836b578ba57f6586f7c,
62e97cbe36854c6c9293508f9d586442.

Source: `data/logs/deep-reader.log`, Compose API stdout, `structure_plan_contract.py`,
`llm_section_splitter.py` and `llm/openai_llm_provider.py`. These requests reached
candidate validation without a new extraction/OCR stage and failed before commit.
The older extraction-path incident below is historical, not the latest cause.
Raw responses were not retained: missing both references versus supplying both
cannot be distinguished from this error alone. Equal response lengths do not prove
identical responses. Prompt inflation is confirmed; its causal contribution to the
invalid response and its exact component/token breakdown remain unmeasured.

### Input Representation And Budget

Measure the final serialized prompt, not only raw source length. Record component
character costs (instructions, schema, source text, optional analysis text,
reference records, Common context and JSON overhead), total size, selected source
coverage, candidate count and input/output budget. Use a tokenizer matching the
configured provider when available; explicitly label conservative estimates and
fallbacks. Character ratios must never be reported as token ratios. Logs contain
counts, versions and safe identifiers, not raw source, anchors, prompt or response.

Proposed compact representation: serialize one readable source representation per
visible span, with stable short references and bounded navigation metadata. Retain
canonical text, offsets and analysis-to-source mapping locally. Do not routinely
send both raw text and analysis_text or duplicate every candidate's 64-character
preview; reference labels should point into the already supplied readable text.
Any exceptional supplementary source excerpt needs an explicit bound and evidence
of necessity. Common parser context remains advisory and bounded, never authority.
Do not remove the ability to select a valid boundary to achieve a smaller prompt.

ENH-PLAN-01 establishes the reproducible baseline and numeric acceptance caps before
compaction. ENH-PLAN-03 must demonstrate materially reduced non-source overhead on
the reported long-source case and unchanged short-document behavior; no success
claim based only on shrinking visible source coverage. Compare like-for-like
coverage and report both characters and tokens/labelled estimates.

The full assembled request must fit the configured input/context budget with an
explicit output reserve and safety margin, including schema and references. Use a
bounded deterministic selection/rebudget pass; never an unbounded retry loop. If a
complete plan cannot be supported within budget, return an explicit recoverable
reason without calling the model. When partial visibility is supported, mark omitted
ranges, prohibit guessed boundaries there, retain exact source coverage locally and
keep the existing prohibition on single_body from a truncated preview. Do not silently
claim full-book analysis. Hierarchical multi-call planning is out of this task group.

### Output Contract

Current Pydantic model validators require exactly one boundary reference, but the
generated JSON Schema marks boundary_id and anchor optional without the exclusive
constraint. Make the model-visible contract and executable validation agree:
either a boundary_id branch, or a segment_id plus exact anchor branch. Represent
mutual exclusion and required fields explicitly; include small valid single-body
and multi-boundary examples and invalid both/neither examples. Avoid conflicting
optional/default-null examples. Version the changed prompt/contract as appropriate.
Provider-native structured output is optional and requires a verified capability
path; do not claim the existing complete_text call already enforces a schema.

Local validation remains authoritative: known references, exact source positions,
distinct ordered boundaries, source/generated title evidence, hierarchy levels and
complete coverage are mandatory. No fabricated reference, silent invalid-plan repair,
fallback success or relaxed validation. Existing stored hierarchy remains usable
when enhancement fails; successful replacement retains the atomic destructive
consent/interaction-cleanup contract.

### Failure Reporting And Acceptance

Record safe validation stage, field path/boundary index and reference presence
booleans to distinguish missing/conflicting references without logging their values.
Expose a bounded recoverable rejection reason through existing result/error paths;
do not misclassify expected invalid model output as an unexplained server failure.
Distinguish budget rejection, provider failure, malformed JSON, schema rejection,
unresolvable reference and commit failure. App/API supporting work must preserve
their response contracts and must not expose Pydantic input/context or raw content.
No automatic paid retry in this track; evaluate a separately budgeted repair attempt
only after measurement shows it is useful and its behavior is explicitly planned.

Acceptance covers missing/both references, valid branches, single-body, long/mixed
language input, fragmented extraction, omitted ranges, titles, duplicate positions,
budget exhaustion and schema overhead. Run offline planner/application tests and
isolated orchestration tests proving failure preserves version/hierarchy/interactions,
success commits once, and enhancement invokes zero OCR/extraction. Record live LLM
evaluation separately with sample count, versions, rejection categories, input/output
tokens, latency and cost; do not run paid/private-document experiments implicitly.
Offline success alone cannot close the live reliability question.

### ENH-PLAN-01 Measurement Delivery

`plan_prompt_metrics.py` accounts for the actual serialized prompt with disjoint
component character costs that sum exactly to total length. Reference preview
characters are a diagnostic subset of references, not an extra additive component.
Source/analysis string costs include escaping; serialization_metadata includes
wrappers, field names, separators and source/segment identifiers. Metrics do not
contain identifiers from source, text, anchors, titles or response bodies.
Both candidate planning entry points emit `enhancement_prompt_metrics` before the
provider call, followed by `enhancement_model_result` with duration, response size
or provider failure. Existing request_id log correlation remains in effect.
Capability input/output limits and actual configured output allowance are recorded
when available; unknown values remain null. This task does not enforce budgets.

Runtime currently has no configured tokenizer counter for this planner. It reports
UTF-8 byte length as a conservative upper estimate, explicitly NOT provider token
usage, and leaves input_token_count null. A supplied local counter is supported by
the measurement helper and tested; exact provider-matched integration remains part
of ENH-PLAN-04, not a claimed live measurement. No tokenizer download or extra
model call is performed. Accounting/capability errors never leak exception text or
block generation; an accounting failure emits a safe unavailable event.

Reproduce without LLM/storage using
`PYTHONPATH=.:scripts .venv/bin/python scripts/test_plan_prompt_metrics.py`.
Frozen synthetic v4 baselines (not a reconstruction of the private live response):

| Fixture | Source chars | Visible chars | Prompt chars | Reference chars | Reference preview subset |
|---|---:|---:|---:|---:|---:|
| Short prose | 34 | 34 | 6861 | 171 | 51 |
| Long sentence-dense prose | 147401 | 120000 | 640932 | 501711 | 249600 |
| Fragmented CJK | 800 | 800 | 17925 | 10066 | 6144 |

Long fixture: 60 segments, 3840 references, 81.41% source visibility; escaped
source text costs 122033 chars, schema 1643, instructions 4502, Common context 2,
serialization metadata 11041, analysis_text 0. This reproduces a similar expansion
without duplicated analysis_text: references alone account for about 78.3%.
The exact private-case breakdown remains unknown until measured with this code.

Frozen ENH-PLAN-03 acceptance targets at unchanged visible ranges: no duplicated
full raw/analysis representation; long prompt <= 1.75 * escaped source chars + 10000
(223557 chars for this fixture); references <= 0.5 * escaped source chars + 2048
(63064 chars); short prompt <= 10000 chars; fragmented prompt <= 10000 chars.
These are future compaction targets, NOT current passing assertions or runtime
limits. ENH-PLAN-04 must additionally fit the configured model's full request budget
with output reserve and >=5% input-limit safety margin; unknown limits require an
explicit safe policy. Model limits, not these fixture caps, govern real requests.

Validation: metrics + contract suites pass 24 tests. Including application tests:
41/42 pass; the legacy API test expects unconfirmed reparse HTTP 200 but now gets
409 confirmation_required. Bypassing the new metrics wrapper reproduces that same
failure, so it is an existing consent-test mismatch, not fixed in this scoped task.
No live LLM, private-document writes, deployment, prompt compaction or schema change.

### ENH-PLAN-02 Boundary Contract Delivery

Prompt v5 exports PlanBoundary's exclusive reference constraint through JSON Schema
oneOf: a non-null boundary_id with absent/null anchor, or non-null segment_id and
non-blank anchor with absent/null boundary_id. Reference defaults are omitted from
the displayed schema so null defaults do not suggest a valid locator-free boundary.
Other property constraints and runtime exact-source validation remain unchanged.
Legacy redundant segment_id with boundary_id is still accepted and resolved from
the authoritative ID; the prompt recommends omitting redundant fields. This is
compatibility, not a second location authority. No schema relaxation or ID repair.

Single-body and multi-boundary fictional examples demonstrate both locator forms;
tests execute them against their declared fictional source. Explicit both/neither
and missing-segment examples are labelled invalid. Examples are formatting guidance,
not permission to copy IDs or split a short article. Implicit null-title bodies must
still provide a source locator. Prompt identity advances to structure_plan_prompt_v5;
wire schema remains structure_plan_v2 because accepted runtime payloads are unchanged.
This is model-visible JSON Schema, not provider-native structured-output enforcement.
Python validators still enforce source-dependent and title/hierarchy rules.

Validation: 27 contract/metrics tests plus 17 application tests pass. The known
legacy no-consent API test is excluded from the application run, not marked fixed.
No live LLM, production mutation or deployment. Full schema-engine conformance was
not run; tests inspect exported exclusive branches and exercise runtime null/missing
reference matrices, examples and source mapping.

Measured v5 fixture totals: short 8466, long 642537, fragmented 19530 characters.
The bounded examples/explicit schema add 1605 characters versus v4; source coverage,
reference counts and payloads are unchanged. This task is a correctness correction,
not compaction. ENH-PLAN-01 frozen compaction targets remain unchanged for task 03.

### ENH-PLAN-03 Compact Prompt Delivery (2026-10-20)

Current prompt: structure_plan_prompt_v6. `compact_prompt_source` is a wire-only
projection: one common_analysis_view text per visible segment, no duplicate
analysis_text, and boundary_references mapping each existing ID to a zero-based
Unicode character position in that displayed text. No repeated 64-character
candidate previews. The canonical preview, raw text, original references and
source offsets remain local and unchanged for validation/application. Every
projected reference must map exactly; unmappable references fail rather than being
dropped. Segment IDs, source ranges, omitted ranges and all candidate IDs persist.
Anchor fallback still resolves against the canonical/analysis views locally.

Context and schema use compact JSON serialization; instructions explain the new
map and preserve v5 exclusive locator rules. Common context is capped at 128 entries,
200 characters per advisory title, and whitelisted range/level/implicit fields;
no body content can be injected through Common context. This truncates advisory
labels only, never source text or authoritative structure. Metrics v2 counts actual
compact reference encoding and reports zero repeated preview characters; it retains
compatibility with the older reference-list accounting format.

| Frozen fixture | v5 chars | v6 chars | Visible source chars | References |
|---|---:|---:|---:|---:|
| Short prose | 8466 | 8215 | 34 | 2 |
| Long sentence-dense prose | 642537 | 192629 | 120000 | 3840 |
| Fragmented CJK | 19530 | 9962 | 800 | 64 |

Long reference serialization falls from 501711 to 52491 characters. Overall long
prompt drops about 70.0% with identical 81.41% source coverage, not by hiding more
source. All frozen ENH-PLAN-01 character caps pass; token values remain labelled
UTF-8 byte estimates, not measured provider usage. The old private 647232-character
request was not regenerated, and these synthetic ratios are not a live token/cost
or model-success-rate claim. Full provider budget enforcement remains ENH-PLAN-04.

Validation: 30 metrics/contract tests and 17 application tests pass. Tests cover
frozen caps, unchanged ranges/reference counts, exact readable-to-canonical mapping
on fragmented sources, no preview mutation, escaped text, bounded Common context,
source preservation and invalid-plan rejection. The previously documented no-consent
API regression remains excluded/unfixed. No LLM calls, user-document mutations,
OCR, or deployment were performed.

### ENH-PLAN-04 Full Request Budget (2026-10-20)

`plan_prompt_budget.py` checks the exact serialized prompt before either candidate
entry calls the provider. Policy enhancement_prompt_budget_v1 rejects rather than
rebudgeting/truncating source or making extra planner calls. Available input is
max_input_tokens minus output reserve minus max(256, ceil(5% of input limit)).
This conservatively reserves output even when provider input/output limits are
separate. Effective configured output allowance is used when available; otherwise
reserve the entire declared output maximum. Missing/nonpositive/noninteger limits
and invalid output allowances fail closed. Equality with available input is accepted.

Optional LLMProvider.count_input_tokens returns a local count or None. OpenAI's
implementation uses tiktoken's encoding_for_model for the configured model and
encodes source special-token spellings as ordinary text. Unsupported/unavailable
tokenization falls back to UTF-8 byte length, explicitly labelled a conservative
estimate, not actual provider usage. The safety reserve covers transport framing;
the local count is for the complete prompt string, not an SDK wire-usage claim.
Tiktoken may initialize/download its encoding cache on first use; no model/API
generation is involved. Other providers can supply a matching local counter.

The budget decision log `enhancement_prompt_budget` records policy, measurement
method, count/estimate, limit, output reserve, margin, available input and decision,
without raw text/errors. Earlier component metrics continue to report their labelled
byte estimate; the budget event is authoritative for the selected tokenizer count
and acceptance decision. Stable errors: enhancement_budget_unavailable,
enhancement_output_budget_invalid and enhancement_prompt_budget_exceeded. The
splitter preserves these exceptions rather than reporting provider failure; the
builder preserves their codes in rejected candidate documents. Public HTTP error
classification is still ENH-PLAN-05; no new API status behavior is claimed here.

Tests: 45 budget/metrics/contract/candidate tests plus 17 application tests pass.
Coverage includes exact/one-over limit, multilingual byte fallback, local tokenizer
selection/failure, unknown limits, output reserve, schema overhead on tiny source,
both entrypoints with zero provider calls on rejection, sanitized logs and reason
preservation. Tokenizer adapter tests mock encoding initialization, with no API or
download. Existing legacy no-consent API test remains excluded. No production data
mutation, live model calls or deployment. Source coverage/omitted-range and
single-body checks remain unchanged; task 05/06 still own API diagnostics and final
integrated acceptance.

### ENH-PLAN-05 Safe Rejection Diagnostics (2026-10-20)

`plan_rejection.py` produces bounded diagnostics from rejected plans. Both planner
entries emit enhancement_plan_diagnostic with json/schema/reference/application
stage, allowlisted reasons and up to 16 schema issues. Paths are limited to eight
components and known field names/indices; unknown field names are redacted. Boundary
schema errors include non-null-presence flags for boundary_id, anchor and segment_id,
not their values. Missing and conflicting locators can now be distinguished.
No raw Pydantic input/context/message, source text, response or provider exception
is logged. Existing prompt budget/provider-response events cover those stages.

EnhancementPlanRejected carries the sanitized reason through the builder and existing
candidate rejection result. Reparse API retains success=false/error response fields;
the enhancement rejection prefix maps invalid JSON/schema/reference/application
plans to 422, prompt-budget exhaustion to 413, provider failure to 502, unavailable
provider/budget or invalid output configuration to 503. Unexpected validation/builder
failures remain 500; existing consent/commit conflicts retain their separate mapping.
Unexpected enhanced-execution exceptions return only enhancement_execution_failed,
not exception text. Common/manual paths are unchanged. No automatic repair/retry.

Validation: 50 diagnostic/budget/metrics/contract/candidate tests and 17 application
tests pass. Checks include unknown model-supplied field names, malformed JSON,
missing/both locators, safe reason propagation and direct route response mapping
with mocked coordinator results. Existing failure-preservation tests still pass.
This is not live HTTP/DB acceptance; ENH-PLAN-06 owns that final evidence. The known
legacy no-consent API test remains excluded/unfixed. No model calls, user-data
mutation or deployment.

### ENH-PLAN-06 Integrated Acceptance (2026-10-20)

Reproducible entry: `scripts/accept_enhancement_prompt.py`, run from backend root
with DEEP_READER_TEST_POSTGRES_DSN explicitly set to a local test-capable database.
The runner requires the DSN (cannot silently skip integration), rejects an enabled
private-document opt-in, runs seven suites and prints counts plus explicit false
live_llm_evaluated/production_deployment flags. PostgreSQL tests create a unique
schema, apply migrations there and drop it in class cleanup. All documents/sources
are synthetic and temporary; existing user namespaces are not written.

Final result: 101 discovered, 100 passed, zero failures/errors, one intentional skip
for the opt-in private-document copy test. PostgreSQL tests were actually executed,
not skipped. The previous API test expecting unconfirmed reparse success is corrected:
Prepare succeeds, unconfirmed enhanced reparse returns 409 confirmation_required,
and provider/save counts do not increase. Successful confirmed reparse is tested
separately through real FastAPI TestClient routes and disposable PostgreSQL storage.

New HTTP/storage assertions: missing/conflicting/unknown locators and malformed JSON
return 422 with safe codes; budget exhaustion returns 413 with zero provider calls;
unknown capability returns 503 with zero calls; provider failure returns 502. Every
failure leaves the database snapshot (including version and interactions) unchanged;
source bytes remain unchanged. Existing commit tests verify rollback after cleanup,
conflicting/stale consent, success-only cleanup/version increment, no repeated commit
on replay, and no extraction/OCR/full Prepare during structure-only enhancement.

Long English/Chinese/French application fixtures retain 120000 visible characters
from 130000 canonical characters and apply only supplied boundary IDs. Their full
canonical source remains byte-for-byte/text-equal in the accepted intervals,
including omitted source; one deterministic model call each. This proves source
preservation, not semantic coverage of omitted text or live title-language quality.
Earlier fixed prompt-size regressions remain passing: v5 long 642537 -> v6 192629
characters, unchanged 120000 visible chars/3840 references, approximately 70% fewer
characters. Token estimates/local counts must not be described as billed usage.

Scope limit: no paid/live model calls, no production deployment/restart and no new
browser acceptance in this task. Real model success rate, private-book semantic
quality, token charges and production latency remain Needs Confirmation. Future
authorized evaluation must record sample count, prompt/model versions, rejection
categories, actual usage, latency and cost; synthetic passes do not satisfy that
claim. The six-task 100% figure means engineering plus isolated integration only.

## Planned Entry Correction: Reuse Accepted Source (2026-10-19)

Historical incident: three enhanced requests failed in extraction/OCR before the planner;
prompt v4 cannot repair that entry-path failure. Forcing all preparation during a
structure-only reparse is not the intended design. The planned accepted-source and
consent workflow is owned by document_preparation/app/db (see preparation's dated
Source-Preserving Enhancement section). This module consumes the verified immutable
canonical text and bounded Common context, proposes a plan and validates exact
coverage; it does not reload PDFs, choose OCR fallbacks or clear interaction data.
Prompt/plan success alone does not prove safe public enhancement or consent support.

## Prompt V4 Refinement (2026-10-19)

Historical prompt refinement: `structure_plan_prompt_v4`; deployed prompt at the
2026-10-20 investigation is v7, with ENH-REV-04 pending. The old conservative
semantic-planning preference is superseded by Enhancement Autonomy above.
The v3 live verification below is
historical evidence, not validation of v4 generation quality. This refinement is
limited to prompt wording/version and regression coverage; schema, application,
API contracts, reference mapping and persistence gates are unchanged.

The current target is autonomous semantic organization around topic, argument,
scene/time or document-function relationships, without importing Common's
uncertain-heading rejection rule. Retain coherent claims/evidence and quotations/
attribution. Common single-body output must not anchor the semantic plan.
Titles are concise neutral navigation in document language, not summaries, analysis
or outside-knowledge conclusions. Clarify top-level versus subordinate transitions
and require a containing level-1 boundary before level 2. Output self-check covers
valid unique references, source order, meaningful transitions, title origin/language,
hierarchy and outcome; it never asks for reasoning prose or extra response fields.

Historical evidence: `test_prompt_v4_quality_and_self_check_contract` verified the
former v4 instructions, not the new autonomy requirement. At that delivery,
`accept_native_recovery.py --regressions` passed 175
cases with no skips/expected failures. No paid LLM evaluation, private-document
reparse or persistence mutation was performed for v4. Semantic-quality improvement
remains unmeasured; previous v3 live success is not counted again.

## Unified Enhancement Correction (2026-10-19)

Maintainer correction supersedes the two-action/two-purpose design below. LLM
enhance is one optional explicit optimization of Common parsing, producing a useful
TOC/navigation and chapter/section segmentation plan. It can retain source headings
or propose semantic headings in the document language. Generated headings remain
labelled generated, never evidence of an original physical TOC. Common still prefers
not splitting when source-heading evidence is insufficient; that constraint does
not prohibit meaningful generated navigation during explicit LLM enhancement.

Prompt `structure_plan_prompt_v3` includes bounded Common section ranges/titles
(up to 128), declared/observed TOC evidence and source-mapped text. Common boundaries
are advisory, not merged back into the accepted plan. Existing `enhancement_intent`
wire values remain accepted and bound from the request for compatibility, but both use this unified
policy. No second user action is required. Existing API incompatible-mode guards
remain; supplying an intent cannot activate enhancement on Common/read paths.

Failure evidence: live logs contain 8,200 extracted characters, zero OCR attempts
and a successful provider HTTP response before candidate rejection. The old catch
discarded the validation detail, so the original response cannot be reconstructed.
A read-only live reproduction against the same PDF reported
`single_body_requires_one_boundary`: multiple boundaries with outcome=single_body.
This is a redundant model-label contradiction, not missing source text. Outcome is
now derived from typed boundaries. A single enclosing JSON fence is accepted; prose
around JSON, malformed JSON, unknown source IDs, unresolved anchors and invalid source
boundaries still fail. Local exact coverage and generated-boundary guards are unchanged.
Subsequent live verification also exposed source-title mismatches: model-cleaned
titles were labelled source. Complete title copying is now checked at the resolved
source position, not constrained to the length of the identifying anchor. Nonliteral
main-body titles are conservatively labelled generated and still pass the semantic
boundary gate; nonliteral TOC/other source-region titles remain rejected. Actual
literal source headings still pass the strict Common heading-evidence gate. This
does not rewrite prose or upgrade generated navigation into original structure.
Safe rejection logs contain allowlisted reasons/counts/version, not response text,
source excerpts or Pydantic input/context. Rejection preserves accepted hierarchy,
readiness and artifacts; only explicit validated replacement commits.

The live provider also produced an unresolvable copied anchor. Rather than fuzzy
matching or accepting guessed positions, source previews now supply deterministic
boundary references: at most 64 per 2,000-character segment, each with a request-local
ID, exact local source position and a 64-character preview. Sentence/line candidates
come from the mapped analysis view and pass the same whole-document semantic-boundary
gate as local application. This removes fragment-internal positions exposed by
arbitrary preview windows; segments are explicitly not chapters. A candidate is
not proof of a heading. Level 1 starts a chapter and its first section together,
so the prompt forbids duplicate chapter/section entries at one position. The model
preferentially returns only boundary_id, never raw offsets or duplicated segment
identity. The server derives the unique containing segment from its own catalog;
an erroneous redundant segment_id cannot move the selected boundary. Arbitrary
preview chunk edges are not inserted as boundary candidates. Unknown IDs and
occurrence other than one fail. Legacy exact anchors with segment_id remain
supported as an alternative, never simultaneously with boundary_id. The same local
heading/semantic-boundary gate and full coverage checks apply to both paths. No
automatic provider retry or fuzzy match was added.

Structured candidates may omit a leading unheaded region: the existing applier
retains that exact prefix as an implicit container rather than rejecting the entire
plan for not supplying offset zero. Single-body plans still require source start;
duplicate positions remain invalid. Valid distinct boundaries are sorted by exact
source position locally; model array order cannot reorder the document. The legacy
intent echo is also bound from the request rather than trusted as model authority.
This makes source conservation a
local responsibility instead of requiring the model to describe a redundant prefix.

Verification: 174 regression cases pass with no skips/expected failures, UI build
and seven mocked desktop/mobile browser suites pass. A read-only live gpt-4.1-mini
planner/application run against the reported PDF succeeded with 8,200 source
characters, five generated-origin sections and exact full-source coverage. Earlier
live attempts exposed the failures documented above; this is one successful final
sample, not a measured reliability rate or a claim of semantic perfection. No
production document/profile/artifact was saved by diagnostics. API image is rebuilt;
explicit user enhancement is still needed to replace the existing document. Earlier
acceptance counts remain historical; PostgreSQL migration acceptance was not rerun.

## Current Track Status (2026-10-19)

NATIVE-01..06, INTERACT-01..03 and READER-01 are complete: 10/10 (100%). This is
isolated engineering acceptance, not a production deployment or live LLM quality claim.
UI checklist/design record frontend recovery and bounded provenance
delivery with seven mocked browser suites and four backend contract tests.
The section_tasks detailed design records guard, language and policy
recovery, including 38 passing current unittest cases and 12 standalone regressions
plus one documented legacy task-layout fixture failure.
The dated delivery sections below retain their historical validation snapshots.

## NATIVE-06 Acceptance (2026-10-19)

`scripts/accept_native_recovery.py` provides an opt-in, repeatable acceptance entry.
It uses real fast PDF extraction and canonical hierarchy builders, temporary JSON
stores, deterministic planner/interaction providers and a disposable PostgreSQL
schema. Source text is never sent to a remote model or committed as a fixture.
PDF bytes are checked before/after; subprocess calls are forbidden during private
fast extraction and socket connections are forbidden during private parsing and
interaction checks. Source text is not included in the JSON report.

Observed results:

- 160 regression cases pass, with zero skips and zero expected failures: heading
  safeguards, source mapping, hints/bookmarks, both intents, language/policy recovery,
  layout provenance/readiness, three-mode PDF routing and artifact cleanup.
- The local 4-page ritual-material PDF yields 8,190 characters; the 2-page family
  article yields 3,557. Auto/Present/Absent all yield one chapter and one section,
  preserving every source character without false short-marker headings.
- Each PDF passes six successful and six rejected explicit enhancement attempts
  (three hints x two intents x accepted/rejected). Rejected candidates leave the
  isolated structured file byte-identical. Fake valid plans deliberately exercise
  valid single-body execution; generated multi-boundary origins are covered by the
  synthetic planner contracts and browser recovery fixtures.
- Each PDF passes all five target levels x three interactions (30 combinations
  total). Real target resolution and artifact stores are used; analysis/quiz reuse
  and thinking readback do not generate or mutate storage. Units are deterministic
  per-section fixtures, not a claim about live embedding segmentation.
- Live local PostgreSQL, in a unique temporary schema with current migrations,
  passes accepted version 1 -> 2 replacement and forced transaction rollback.
  Analysis/quiz/thinking artifact rows survive failed replacement and are cleared
  only after accepted replacement. The schema is dropped in finally; no production
  hierarchy is rebuilt or deleted.
- Seven frontend browser suites pass at desktop/mobile sizes using intercepted
  APIs, including explicit action gating, stale-policy recovery, drafts, accepted
  ID replacement and rejected-enhancement preservation.

Reproduction from repository root (provide a test DSN through the environment,
never commit credentials):

```bash
Deep_Reflective_Reader/.venv/bin/python Deep_Reflective_Reader/scripts/accept_native_recovery.py --regressions --postgres --pdf <first-local.pdf> --pdf <second-local.pdf>
```

`--postgres` requires `DEEP_READER_TEST_POSTGRES_DSN`; no production-DSN fallback.
Omitted groups are explicitly reported as not run, not counted as passed. The local
venv was missing the repository-pinned psycopg[binary]==3.3.5; that existing dependency
was installed without changing requirements. Runtime parser/API code was unchanged.

Remaining evidence limits: no live LLM semantic-quality/cost evaluation, no complete
live browser/current API/provider flow, no new production latency measurement and
no Docker deployment. Earlier documented unrelated legacy fixture failures are not
silently counted in this selected acceptance matrix or repaired by this task.

## NATIVE-05 Runtime Delivery (2026-10-19)

Production enhanced selection now uses v2 planning and local application, including
the direct splitter entry. The v1 merger is retained only behind the explicitly
named compatibility entry `split_legacy_with_provenance`; production does not use
its augmentation, occurrence clamping or regex anchors. Both intents and a valid
single-body plan execute through the explicit preparation/reparse path. An accepted
LLM plan is never overwritten by native Outline/TOC projection or merged with
Common boundaries. Enhanced planning uses bounded text-only TOC observations;
unvalidated candidates remain candidate_unusable, not validated physical TOCs.

Preview segments optionally expose a mapped CJK analysis view alongside exact
canonical text. Original source hash/ranges remain unchanged. Exact/raw and mapped
matches are deduplicated by original position; occurrence is mandatory, positive
and in range, never clamped. Unknown segments, changed anchors, omitted-region
anchors and unordered/missing-prefix plans fail. No regex executes in v2. All
accepted section contents are exact source slices and concatenate to the full raw
text. Source-title candidates must pass the shared heading gate; unsupported
boundaries are omitted without adding Common boundaries. If none survive, the
enhancement fails. Generated boundaries cannot split a reconstructed fragmented
word. Implicit regions cannot invent a TOC; generated headings remain main-body
navigation and cannot validate TOC evidence.

`heading_acceptance.py` requires source-line isolation plus structural/contextual
corroboration. Numbered headings, supported region markers and isolated unnumbered
headings with following prose have deterministic rules; uncertain short markers,
quotes and inline references remain body text. Existing Common OCR heading
normalization plugins run before comparison, without rewriting source. The same
gate checks Outline/TOC destination titles; rejected Outline heading evidence does
not trigger extra layout/OCR work. Accepted projections retain pre-heading source
as an implicit front-matter interval instead of dropping it. This is conservative
heuristic evidence, not a guarantee of perfect heading recall in every language.

Persisted provenance records schema/prompt/heading policy versions, effective
intent/hint, proposed/resolved/rejected/accepted counts, complete coverage and
heading origins keyed by canonical intervals. Chapter metadata records title_origin
where a chapter adopts a candidate heading, otherwise implicit. Public task-layout
response fields are unchanged; future UI consumption needs explicit API mapping.
Common policy is now common-source-preserving-v2. Explicit cache checks include
hint, parser mode and intent; enhanced requests intentionally rebuild, while normal
reads retain no-write behavior. Missing/old identity is not silently accepted by
explicit policy-aware preparation.

Generation/validation failure returns an error, never a successful Common fallback
replacement. Profile rebuilding/enrichment and free-QA artifact work are deferred
until enhanced structure has passed validation and saved; rejected attempts perform
no structure/profile/artifact writes. Existing atomic JSON replace/PostgreSQL save
boundaries are reused, with readback before success; no storage migration is added.
Provider and candidate errors expose stable codes rather than provider payloads.
The last persisted hierarchy/readiness/version/artifacts remain available after a
rejected attempt; the failed request itself reports failure, not fake success.
Post-commit storage/readback failures retain existing store semantics and are not
claimed to roll back an already accepted commit.

Validation: eight unittest suites, 92 cases (91 passing, one expected failure owned
by INTERACT-01), including isolated real route execution for Prepare/Reparse,
failed-attempt preservation and generated-origin serialization. Six standalone
checks pass: heading normalization, legacy region-plan compatibility, TOC projection,
hard-reparse artifact cleanup, structured reuse and Common profile ordering.
The profile-order test double was updated for the existing extraction argument.
No live LLM call, production DB mutation, private-document reparse or deployment.
Live PostgreSQL/end-to-end private-document acceptance remains NATIVE-06.
Combined progress: 5/10 (50%); the next ordered task is INTERACT-01.

## NATIVE-04 Planner Delivery (2026-10-19)

Historical delivery snapshot; its staged runtime guard is superseded by NATIVE-05 above.

`structure_plan_contract.py` defines strict `structure_plan_v2` candidates and
server-owned `structure_plan_prompt_v2`. `LLMSectionSplitter.build_candidate_plan`
is a planning-only entry: it receives request-local enhancement intent, TOC
declaration, server-observed TOC state (default not_checked) and document language.
Source is untrusted evidence, not instructions. Regions are optional; insufficient
heading evidence must not produce a boundary. The existing runtime prompt also
receives the conservative-heading instruction without changing its legacy schema.

The v2 contract distinguishes valid single_body from structured output, malformed
JSON/schema, empty source, missing provider and provider exceptions. Both intents
permit one implicit body; generated titles are allowed only in an explicit
generate_reading_outline main-body plan. Source/implicit/generated title origins
are typed; no placeholder title is synthesized for an implicit body. Two-level
navigation uses boundary titles, not independently invented container names.

Preview segments retain exact canonical source slices and SHA-256/range-based IDs,
with at most 120,000 source characters, 2,000 characters per segment and explicit
omitted ranges. This is an identity source map, not destructive CJK normalization.
JSON/schema overhead is additional to the source-character budget. Candidates have
at most 128 boundaries, 256-character copied anchors and a 200,000-character
response limit. Regex and unknown fields are rejected. Local candidate checks
verify supplied IDs, exact within-segment occurrence, monotonic order, retained
prefix and source-title copying; these checks do not prove semantic heading
validity. Single-body claims require full source visibility. Cross-line mapped
resolution, the full conservative acceptance gate and source slicing remain 05.

Prepare/task-layout/reparse requests now accept enhancement_intent with default
restore_source. Generated intent requires llm_enhanced; manual/common/layout-only
incompatible requests are rejected. Adapters forward intent, but runtime generated
outline application deliberately returns `reading_outline_application_not_available`
(400) before source loading, paid generation or persistence. Existing restore
requests retain their legacy execution path. The v2 planner is NOT connected to
the old applier, which still merges Common boundaries and rejects single-boundary
results. NATIVE-05 owns activation, source mapping/gating, generated-origin
persistence and atomic replacement; this delivery does not claim those behaviors.

Evidence: 80 unittest cases, 78 passing and two existing expected failures;
legacy LLM region-plan smoke passes. Tests use fake providers/isolated stores,
cover strict plans/prompts, declaration versus observation, source references,
intent/default/forwarding and no-load/no-save staged rejection. No live LLM,
private-document change or deployment. Combined progress: 4/10 (40%).

## Conservative Heading Acceptance (2026-10-19)

Status: maintainer-approved design; NATIVE-04 delivers prompt/plan constraints,
and NATIVE-05 delivers the local runtime gate described above.
NATIVE-03 fixes known fragment cases; it does not establish a general heading
acceptance gate. In particular, `_line_matches_marker` still accepts exact marker
matches before contextual checks. A keyword or extracted line is a candidate,
not sufficient evidence of a source heading.

For Common and LLM source-origin headings, prefer an omitted boundary over an unsupported
boundary. Evaluate candidates as accepted, uncertain or rejected. Both uncertain
and rejected candidates remain in the original body, with no text loss, fabricated
title or preparation error. A completely unheaded article is valid single-body
output. These states are internal decisions, not new public API fields.

Acceptance policy:
- Require an independently delimited source position plus a complete structural
  signal (e.g. a full chapter-number heading), or corroborating source evidence
  such as a locally validated bookmark/TOC destination or available typography.
  A heading-like token embedded in a sentence or quotation is not a boundary.
- Newlines, short length, keywords, capitalization or an LLM confidence assertion
  alone never suffice. PDF physical lines may be extraction artifacts. Inspect
  surrounding text and continuation before accepting a line as a heading.
- Short markers such as `序` and `前言` require contextual corroboration, such as
  paragraph isolation and a plausible transition into prose; word continuation,
  fragmented glyph runs and contradictory context keep them in the body.
- Missing font/layout evidence is unknown, not affirmative or negative evidence.
  Use existing source evidence only; do not render, OCR or request an LLM solely
  to resolve a doubtful Common heading. Sequence/format consistency corroborates
  but is not mandatory, since an excerpt may contain only one genuine heading.
- Define deterministic decision/reason rules with synthetic controls before
  choosing thresholds. Avoid an uncalibrated score that adds several correlated
  weak cues into artificial certainty. Metadata/profile/user TOC declarations
  remain advisory and cannot override conflicting source evidence.

Execution ownership stays within the existing ten-task track: NATIVE-04 encodes
this policy in source-restoration prompts/plans; NATIVE-05 implements the local
acceptance gate for Common and restored LLM boundaries before candidate selection,
including validation of bookmark/TOC-derived boundaries. It must not merge rejected
Common boundaries back into an accepted plan. Generated reading outlines are part
of the same explicitly requested enhancement, use source-verifiable anchors and mark generated
titles rather than claiming original headings. Any changed acceptance policy must
version its explicit-preparation cache identity; reads never rebuild old documents.
NATIVE-06 validates both paths. No new task is counted and NATIVE-03 is not reopened.

Required controls include inline chapter references, quoted headings, standalone
short words, missing/ambiguous whitespace, character-per-line extraction, real
prefaces, numbered headings, genuine unnumbered headings with corroboration,
single-chapter excerpts, poetry/lists, misleading TOC declarations and conflicting
bookmark destinations. Assert exact non-overlapping source coverage, no unsupported
split, valid single-body readiness and no additional OCR/LLM calls. Verify generated
outline headings cannot leak into source-restoration mode.

## NATIVE-03 Delivery (2026-10-19)

Common parsing now uses the NATIVE-02 mapped analysis view. Fragment evidence
requires at least eight consecutive short CJK lines, twelve CJK characters and
sentence punctuation; ordinary prose, verse/list controls, blank lines, explicit
page separators and complete chapter headings are retained. A run-initial genuine
preface remains a heading. The heuristic is deliberately conservative, not a
general PDF layout reconstruction engine. Physical page identity remains unknown
without page evidence; canonical source text, hashes and offsets are never rewritten.

Common boundaries map back to exact original slices. Pre-heading text receives
an unheaded body interval rather than disappearing; adjoining slices cover the
source once. Overlap or mapping-budget failure retains a single full-source
container. Fragment runs are excluded from text-only TOC title-list evidence.
This coverage change applies to Common splitting/fallback, not a redesign of
validated native-outline or printed-TOC projection.

Explicit prepare, prepare-task-layout and reparse accept typed `toc_hint`:
`auto` (default), `present`, `absent`. Absent bypasses Common's dedicated TOC
search and builder TOC/layout scheduling, but not body headings or validated
native PDF bookmarks. Present does not fabricate a TOC when evidence is absent.
Persisted `common_parse_policy` records version `common-source-preserving-v1`,
hint, TOC observation, implicit-single-body and unconfirmed-declaration flags.
A valid unheaded article is successful Common output, not a structure error.

Public explicit preparation requires matching policy version/hint before reusing
structure. Internal callers that omit the hint retain legacy read/reuse behavior;
reads never reparse. Request-local assets carry the declaration without shared
state. `layout_only` rejects non-auto declarations; manual reparse rejects them.
LLM prompt/intent consumption remains NATIVE-04: this slice only records the
declaration and controls deterministic builder TOC work, not LLM planning.
The new provenance is persisted internally; expanded task-layout response mapping
and UI consumption remain future work, not an implemented API guarantee.

Validation: 68 unittest cases, 66 passing and two existing expected failures
(INTERACT-01 and NATIVE-05). Heading normalization, LLM region-plan, TOC projection
and structured-reuse standalone regressions pass. No paid LLM call, private-document
mutation or deployment. Combined track: 3/10 (30%); next is NATIVE-04.

## Investigation: Fragmented Native PDF / LLM Reparse (2026-10-17)

Status: investigation and future implementation design only. No runtime change,
user-document reparse, artifact deletion or LLM request was made in this investigation.
This is a new track, not a reopening/count extension of the completed 17-item PDF
mode track. Execution index below owns ten unique tasks across three modules.

### Evidence And Confidence

- **Confirmed, live read-only evidence:** document 33, structure version 2
  (`《歸來》白事／喪葬／鄉村儀式素材整理稿`) has 13 chapters and 28 sections.
  All section titles are `序`; the first section starts at raw offset 263,
  leaving the opening source range outside stored section coverage. This is a
  slicing/coverage defect, not evidence that an LLM wrote new prose.
- API request `4e5859bd848a4fd0afeeb6c2cbc30cea` logged
  `LLMSectionSplitter#apply_fallback_common: reason=no_resolved_boundaries` and
  a successful 13-chapter/28-section build. Persisted provenance confirms
  requested `llm_enhanced`, effective `common`, fallback true. Document 32
  (`家醜不可外揚`, version 2) has the same fallback reason, one main-body
  chapter, a 3590-character section and three task units totaling 3576 characters.
- **Confirmed reproduction:** read-only native extraction in the running container
  with pypdf 6.17.0 and the common splitter yields 28 `序` sections for document
  33. Individual `序` lines originate inside words such as `秩序` and `程序`:
  PDF text extraction exposes many single-character lines, not semantic headings.
  The joined diagnostic text is 8203 characters (page joining differs from stored
  preparation's 8200); do not compare offsets across these extraction contracts.
- Both documents have `toc_detection.detected=true` but `usable=false`, with
  missing page anchors and failed global validation. These are rejected TOC
  candidates, not proof of a physical TOC or accepted TOC projection. The bad
  `序` hierarchy is reproduced in the common splitter independently of TOC projection.
- **Confirmed code defects/risks:** `section_splitter.py` treats each extracted line
  as a heading candidate. `llm_section_splitter.py` sends raw fragmented preview
  text, resolves anchors one physical line at a time, clamps an out-of-range
  occurrence to the last match, rejects fewer than two boundaries, and merges
  common boundaries back into otherwise resolved plans. New LLM main-body
  boundaries not present in the common baseline can be discarded.
- **Needs Confirmation:** the exact historical LLM response/anchors are unavailable.
  Cross-line anchors are a demonstrated mechanism risk, not a proven account of
  every failed anchor in those historical requests. No new paid LLM request was
  made merely to reconstruct the missing response.

### Proposed Implementation Boundaries

1. Introduce a bounded source-mapped analysis view for extraction-induced CJK
   line fragmentation. Preserve canonical raw text, page spans and hashes; every
   normalized interval must map monotonically to an original source interval.
   Keep page/paragraph evidence and original line identities. Normalization is
   evidence processing, not permission to concatenate all text indiscriminately.
   Reuse existing normalization primitives only where their behavior is suitable;
   `normalize_ocr_text` alone has no offset map and must not replace canonical text.
   Fast remains native-only: no automatic OCR, renderer or LLM escalation.
2. Distinguish a physical TOC, validated source headings, and internal reading
   containers. A short article may have headings but no TOC; a genuinely unheaded
   article remains one implicit main-body container. Do not invent `序`, chapters
   or a physical TOC to fill the hierarchy schema. Existing hierarchy contracts
   remain intact; synthetic containers need explicit provenance, not title guessing.
3. Heading checks must consider reconstructed source context: a character inside
   `秩序` is not a preface. Preserve real standalone `序`/chapter headings with
   positive controls. TOC candidates remain advisory until validated. Require
   full non-overlapping source coverage, including pre-heading text; a failed
   candidate must fall back to a source-preserving flat container, never omit text.
4. Extend the LLM plan contract and prompt to express a valid unheaded/single-body
   plan separately from empty/invalid output and service failure. Regions are
   optional, not required slots. In source-restoration mode disallow invented headings;
   explicit reading-outline generation may label semantic segments as generated.
   Both modes disallow paraphrased anchors and rewritten source content.
   Supply stable source segment identities with bounded context and explicit omitted
   ranges; copied anchors must be locally verifiable. Schema/prompt versions are
   server-owned. LLM remains a candidate planner, never a persistence authority.
5. Resolve bounded cross-line anchors through the source map. Reject ambiguous,
   missing and out-of-range occurrences rather than choosing a convenient match;
   prefer segment identity plus copied-text verification. Audit unrestricted regex
   support and impose safe bounds or remove it from the new generated-plan contract.
   Accept valid single-body plans. Do not automatically restore rejected common
   boundaries: choose a validated candidate atomically, retaining common only as
   an explicit fallback. Head/tail preview gaps must not justify unobserved splits.
6. Record requested/effective parser, fallback reason, proposed/resolved/rejected
   boundary counts, coverage result and policy versions without full text, prompts,
   secrets or LLM payload logging. Reuse existing provenance fields where possible;
   additional public fields require explicit API mapping and contract tests, not
   assumptions in UI docs. No read-path reparse or profile/diagnostic write-back.
7. Old incorrect hierarchy changes only through explicit reprepare/reparse, with
   existing atomic version advance and artifact cleanup. Document 33 needs
   source-prefix preservation verified before any user-data repair. Persisted
   empty-raw-text task-layout hashes were also observed; their validity semantics
   require checking in acceptance, not declaring a separate confirmed root cause.

### TOC Declaration And Enhancement Intent (2026-10-19)

Maintainer-requested addition; the delivery note above identifies the implemented
NATIVE-03 toc_hint subset and NATIVE-04 planning/request intent subset. Runtime
v2 application and UI controls below remain proposed until their owning tasks.
Delivered NATIVE-01 remains a baseline only. No additional task is
counted: NATIVE-03/04/05, READER-01 and NATIVE-06 own the following acceptance work.

**Readiness:** successful common parsing means readable source is preserved in
order with valid hierarchy/target identities and required preparation complete.
Neither a TOC nor multiple chapters is required. An unheaded short article may
have one implicit `Main Body` container and remains openable/interactable, subject
to each action's content checks. Lack of headings alone is not insufficient content.
LLM enhancement is optional; no hidden paid enhancement on load/open. A failed or
invalid enhancement preserves the last usable common hierarchy, readiness, version
and artifacts. Only validated, explicitly requested replacement advances structure
version and runs existing atomic invalidation. Readability is not a guarantee that
every tiny block supports every action.

**Independent settings (corrected):**

| Field | Values / default | Meaning |
|---|---|---|
| `pdf_extraction_mode` | Existing fast/ocr/auto | How source text is extracted; unchanged |
| `toc_hint` | auto (default), present, absent | User declaration about a physical TOC |
| `enhancement_intent` | restore_source (default), generate_reading_outline | Legacy wire values for the same explicit enhancement policy; not UI choices |

`toc_hint` applies to common parsing and is forwarded to an explicit LLM plan;
it cannot select OCR or activate LLM by itself. `enhancement_intent` is effective
only with explicit enhanced parsing. Common requests must not silently run generation
because the field was supplied; reject incompatible combinations in the new contract.
Omitted fields preserve existing-client compatibility with conservative defaults.

- auto: bounded deterministic TOC discovery using available source evidence.
- present: prioritize evidence-backed TOC discovery; pass the declaration and any
  unresolved evidence to the unified planner when the user explicitly requests enhancement.
  Missing/ambiguous anchors leave TOC unusable, not fabricated. Warn that the declared
  TOC could not be verified and retain source-preserving common output.
- absent: skip dedicated TOC discovery, but still recognize genuine body headings.
  Do not remove source pages or imply that the document has no chapters. Contradictory
  evidence already encountered may produce a warning without silently reversing the
  setting or performing extra expensive discovery. User can change the hint explicitly.

Keep declared hint separate from observed TOC evidence (`not_checked`, `not_found`,
`candidate_unusable`, `validated` as proposed states); absent is not verified absence.
PDF Outline bookmarks are distinct from a printed TOC. Existing validated native
Outline handling may remain available even when dedicated printed-TOC search is skipped.
Profile/LLM classification and user declarations never replace local anchor validation.

**LLM plan/prompt:** one enhancement optimizes Common output, retaining supported
source structure or proposing semantic segments and concise document-language
headings with generated origin explicitly marked. It may retain one segment when
splitting adds no value. A mistaken present declaration cannot fabricate an original
physical TOC. The unified prompt receives Common boundaries, the legacy intent
value for compatibility, declared hint, observed evidence and
source-mapped segments; source text is evidence, not instructions. All boundaries
must resolve locally, preserve order and cover the source without rewriting it.
Generated navigation is not a claim that a physical TOC existed, and must not set
TOC detection to validated. Record origin per heading/container (source, implicit,
generated) plus effective parser/intent; preserve explicit user edits as user origin.

**Handoff, storage and reuse:** NATIVE-03 owns validation/propagation of toc_hint
through explicit preparation entry points; NATIVE-04 owns enhancement intent and
plan schema; NATIVE-05 owns application, provenance and cache integration. Supporting
main/api_schemas/document_preparation/app adapters are part of those slices and
must have their owned docs loaded before implementation. Carry settings through
upload-to-prepare, prepare-task-layout and explicit reparse paths using request
snapshots. Store effective settings/policy versions in parse provenance, not in
document profile as parser authority. Include effective parser-affecting settings
in reuse identity; missing historical identity is unverified. Reading existing
data never triggers a rebuild. Changing a UI draft is not a write; explicit
prepare/reparse validates the new identity and preserves prior usable data on failure.
Enrichment failure must not turn a previously readable document into an unreadable one.

**Acceptance additions:** cover the three hints with headed/unheaded articles,
real/missing/malformed TOCs, contradictory declarations, bookmarks without printed
TOC, both enhancement intents, single-body success, no useful semantic split,
generated-label provenance, cache identity changes and failed replacement rollback.
Assert no OCR escalation, implicit LLM call, fabricated source heading or background
mutation. NATIVE-06 verifies all handoffs and source conservation end to end.

### Ordered Implementation Index

All ten items are complete as of 2026-10-19 under the NATIVE-06 evidence limits above. Supporting modules/tests
are part of the named slice, not additional counted tasks. Execute one per turn.

| Order | ID | Owner | Deliverable / Dependency |
|---|---|---|---|
| 1 | NATIVE-01 | document_structure | Synthetic failure fixtures, private-fixture diagnostic, source coverage baseline |
| 2 | NATIVE-02 | document_structure | Source-mapped analysis view; depends on 01 |
| 3 | NATIVE-03 | document_structure | Heading/TOC safeguards and source-preserving flat fallback; depends on 02 |
| 4 | NATIVE-04 | document_structure | Versioned LLM plan/prompt with valid single-body state; depends on 02 |
| 5 | NATIVE-05 | document_structure | Validated anchors, candidate selection and fallback diagnostics; depends on 03/04 |
| 6 | INTERACT-01 | section_tasks | Fragmentation-aware semantic guard; depends on 01/02 |
| 7 | INTERACT-02 | section_tasks | Consistent explicit output-language policy |
| 8 | INTERACT-03 | section_tasks | Server-owned policy/version reuse and recovery; depends on 06/07 |
| 9 | READER-01 | Deep_Reader_UI | Honest hierarchy/fallback/interaction recovery display; depends on 05/08 |
| 10 | NATIVE-06 | document_structure | Isolated end-to-end and repair-path acceptance; depends on all above |

Priority confirmed 2026-10-19: source preservation and reproducible failures come
first, followed by mapped evidence, parser correctness, interaction policy,
consumer recovery and final acceptance. No task is combined with another delivery.

### NATIVE-02 Source Mapping Delivery (2026-10-19)

`source_mapped_text.py` provides an opt-in pure `build_source_mapped_text` primitive.
It does not classify fragmentation or alter existing parser/interaction behavior.
Without confirmed ranges the analysis view is identical to source, including
ambiguous single-character headings, poetry and lists. NATIVE-03 must establish
source-based evidence before passing confirmed ranges; a short line alone is not
sufficient. This separates evidence classification from reversible text handling.

`SourceRange(start,end)` and `PageRange(page_index,start,end)` use canonical Python
character offsets and half-open intervals. The immutable result retains source,
analysis text, one canonical offset per retained character, original line intervals
(including line endings), page intervals, confirmed/removed ranges and a versioned
policy ID. `source_text_sha256` hashes UTF-8 canonical text, not original PDF bytes;
existing loader source hashes/page evidence remain unchanged and caller-owned.
Nonempty analysis intervals map to minimal enclosing original intervals, including
removed inner whitespace. Page intersection lookup returns original coordinates;
empty source pages remain in evidence. Empty/invalid analysis intervals are rejected
instead of inventing an ambiguous insertion offset.

Within explicitly confirmed intervals, only spaces/ideographic spaces or one LF,
CR or CRLF between two CJK characters can be removed. No Unicode folding, trimming,
punctuation removal, Latin/numeric concatenation or source rewriting occurs. Blank
lines, tabs, form feeds and Unicode line/paragraph separators remain hard barriers.
Page evidence must be ordered, nonoverlapping and cover all non-whitespace source;
confirmed ranges cannot cross a page. Without supplied pages, physical-page identity
is unknown; explicit form-feed boundaries still survive, but callers requiring
physical-page guarantees must provide page evidence.

The primitive uses linear source scanning/storage plus bounded evidence validation:
at most 2,000,000 source characters and 4096 ranges in each evidence collection.
Exceeding limits raises `ValueError`; it never produces truncated success. Ordered
source-offset mapping is tested exhaustively on a synthetic fragmented sentence.
Consumers must handle budget/invalid-evidence errors conservatively, not silently
replace canonical text or start OCR. No network, LLM, renderer or persistence
dependency is introduced.

Validation: 11 new tests pass; combined with fast extraction and NATIVE-01, 30
tests run (25 passing, 5 expected failures for unresolved consumer defects).
LLM region-plan and TOC projection smoke tests also pass. No consumer integration,
real-document repair or deployment is claimed. Combined progress: 2/10 (20%);
next is NATIVE-03.

### NATIVE-01 Baseline Delivery (2026-10-19)

`scripts/test_native_fragmentation_baseline.py` contains authored synthetic prose,
not copied private material. Five expected failures capture desired behavior for
word-internal headings, prefix coverage, false TOC detection, fragmented-prose
guard rejection and cross-line anchors. Five passing controls verify the coverage
oracle (gap/overlap/invalid ranges), ordinary prose, unusable TOC rejection,
genuine standalone preface detection and repeated-junk rejection. Expected failures
are unresolved defects, not successful repair tests; unexpected success fails the
suite until the owning follow-up removes the decorator and validates the fix.

`scripts/diagnose_native_fragmentation.py` requires explicit `--pdf` and `--language`.
Run with `PYTHONPATH=Deep_Reflective_Reader` and the backend Python environment.
It uses `PdfDocumentLoader.extract(mode=fast, source_bytes=snapshot)`, blocks
subprocess execution, emits metadata/counts only and checks the file hash again.
There are no DB, artifact, renderer, OCR or LLM calls. Reports use this exact fast
extraction contract, not the earlier diagnostic's manual page concatenation.

Local Python 3.12.2 / pypdf 6.9.2 results: the four-page fixture has 8190 characters,
2688 nonempty lines (2362 single-character), 23 sections, unusable TOC candidate
and repetitive-content rejection. The two-page fixture has 3557 characters, 1241
lines (1214 single-character), one section, unusable TOC candidate and fragmented-
heading rejection. Both current fast reports have zero uncovered/overlapping
nonspace characters and unchanged source bytes. This does not negate the historical
persisted 28-section/prefix-loss evidence: runtime/version/extraction contracts
differ. The synthetic prefix-loss test reproduces the defect independently.

Validation: 10 baseline tests (5 pass, 5 expected failures), both opt-in diagnostics,
existing LLM region-plan and TOC projection smoke scripts. No live reparse, deployment
or product logic change. Combined follow-up: 1/10 (10%); next is NATIVE-02.

Acceptance includes both reported PDFs (private fixtures, not committed source text),
genuine TOCs/prefaces, normal Chinese/English paragraphs, scans, corrupt text,
repeated junk, no-heading articles, ambiguous anchors, stale cache and read-only
entry points. No thresholds are simply disabled to turn failures green. Existing
PDF mode timing acceptance did not validate hierarchy semantics or LLM language;
its 21/21 result must not be presented as coverage of these new defects.

## 1. Module Purpose

`document_structure/` 是結構化閱讀核心，負責把原文切分並收斂為 hierarchy-first 的 `StructuredDocument`，並提供結構一致性與結構相關持久化邊界。 **[Code-Confirmed]**

## 2. Position in Overall Architecture

- Document Structure Core

## 3. Key Files

| File | Responsibility | Notes |
|---|---|---|
| `document_structure/structured_document.py` | 定義 `StructuredDocument/StructuredChapter/StructuredSection` 契約與序列化 | normal `from_dict/from_json` 已 strict 要求 chapters；legacy loader 僅 migration-only **[Code-Confirmed]** |
| `document_structure/structured_document_builder.py` | 以 splitter 建立 structured document，錯誤時 fallback doc | 支援 parser mode 選擇入口 **[Code-Confirmed]** |
| `document_structure/structured_hierarchy_builder.py` | 將 flat sections 收斂成 chapter->section hierarchy | 主流程不再生成 structure_nodes mirror **[Code-Confirmed]** |
| `document_structure/document_hierarchy_index.py` | hierarchy-first 查找/flatten/一致性校驗 helper | `get_effective_sections` 已 hierarchy-only **[Code-Confirmed]** |
| `document_structure/section_splitter.py` | common parser split 實作 | 受 language registry 支持 **[Code-Confirmed]** |
| `document_structure/llm_section_splitter.py` | llm enhanced parser split 實作 | 作為 selector 另一條路徑 **[Code-Confirmed]** |
| `document_structure/section_splitter_selector.py` | common/llm split mode 選擇 | parser mode contract 中樞 **[Code-Confirmed]** |
| `document_structure/structured_document_store.py` | structured JSON load/save | normal load 經 strict hierarchy contract 校驗 **[Code-Confirmed]** |
| `document_structure/document_artifact_repository.py` | artifact repository 抽象介面 | lightweight document discovery plus section/chapter/task-unit/document-level/parser-replacement methods **[Code-Confirmed]** |
| `document_structure/structured_document_artifact_repository.py` | 具體 artifact repository（hierarchy-aware） | normal read/write 路徑要求 chapters；legacy 讀取僅 explicit migration helper；parser-level replacement clears derived artifacts **[Code-Confirmed]** |
| `document_structure/enhanced_parse_trigger_evaluator.py` | enhanced parse recommendation 評估器 | recommendation only，非 auto-switch **[Code-Confirmed] + [From HLD]** |
| `document_structure/document_structure_language_registry.py` | parser/regional/profile-evidence 用語言標記 registry | multi-consumer registry **[Code-Confirmed]** |

## 4. Main Responsibilities

1. 定義 hierarchy 主契約（Document -> Chapter -> Section）。 **[Code-Confirmed]**
2. 實現 common/llm enhanced 結構切分入口與 fallback 產物。 **[Code-Confirmed]**
3. 提供 hierarchy-first 查找與一致性檢查 helper。 **[Code-Confirmed]**
4. 管理 structured artifact 讀寫與 hierarchy-aware artifact 更新。 **[Code-Confirmed]**
5. 提供 enhanced parse recommendation 訊號。 **[Code-Confirmed]**
6. 提供 lightweight structured document discovery contract，供 API list/search 使用；不返回 hierarchy/content heavy payload。 **[Code-Confirmed]**

## 5. Non-Responsibilities

1. 不應直接承擔 API request/response mapping。 **[From HLD]**
2. 不應讓 profile metadata 直接硬控制 parser 切分規則。 **[From HLD]**
3. 不應在 task-layout read path 做隱性寫回。 **[From HLD]**
4. 不承擔 task-layout projection contract 本身（該責任屬 `section_tasks/` + `app/`；此 module 僅提供 hierarchy helper 與 persistence primitives）。 **[Code-Confirmed] + [From HLD]**
5. 不承擔 profile diagnostics 組裝與 response 投影責任。 **[Code-Confirmed] + [From HLD]**

## 6. Important Data Structures / Contracts

- `StructuredDocument`
- `StructuredChapter`
- `StructuredSection`
- `SectionSplitterMode`
- `DocumentArtifactRepository` (interface)
- `DocumentListItem`
- `EnhancedParseTriggerDecision`

以上是本 module 的高層契約代表。 **[Code-Confirmed]**

### Explicit Document Deletion

`DocumentArtifactRepository.delete_document(doc_name) -> bool` is the explicit structured
storage deletion boundary. It returns false only for an absent target; unsupported
implementations and storage errors propagate instead of pretending deletion succeeded.
The file implementation unlinks the normalized structured JSON path, including malformed
artifacts, without loading hierarchy. The PostgreSQL adapter overrides this operation
with namespace-scoped store deletion; the route never substitutes file cleanup for a DB
write. Raw source and retrieval-directory cleanup remain outside this repository.
No list/read/task-layout operation invokes deletion. **[Code-Confirmed]**

## 7. Architecture Constraints

1. persisted source of truth 是 `chapters[].sections[].task_units[]`。 **[Code-Confirmed] + [From HLD]**
2. 不重新引入 root `sections` mirror 作新寫入來源。 **[Code-Confirmed] + [From HLD]**
3. `structure_nodes` 不作主流程結構來源。 **[Code-Confirmed] + [From HLD]**
4. parser metadata 屬 advisory，不是 parser authority。 **[From HLD]**
5. `StructuredDocument.to_dict()` 預設不輸出 root `sections[]`/`structure_nodes[]`，僅在 legacy include 旗標顯式開啟時輸出。 **[Code-Confirmed]**
6. artifact 是 interaction output/support material，不是 hierarchy truth source。 **[Code-Confirmed] + [From HLD]**
7. artifact write path 必須以 hierarchy-aware targeting 為前提，不得反向改寫 chapter/section/task-unit identity。 **[Code-Confirmed]**
8. artifact availability projection ownership 不在本 module（屬 task-layout/coordinator DTO 層）。 **[Code-Confirmed] + [From HLD]**

## 8. Module Relationships

- depends on:
  - `language/`（language code + structure language registry）
  - `shared/`（task artifacts / task unit model）
- used by:
  - `document_preparation/`
  - `app/section_task_coordinator.py`
  - `section_tasks/`
- reads from:
  - structured JSON artifact
- writes to:
  - structured JSON artifact（store/repository）
- advisory relationship:
  - 與 `profile/` 是弱耦合（metadata advisory，不是 parser authority） **[From HLD]**

## 9. Main Flows Involving This Module

1. prepare flow：`structured_document_builder` build + `structured_document_store` save。 **[Code-Confirmed]**
2. hierarchy build flow：flat sections -> hierarchy chapters/sections。 **[Code-Confirmed]**
3. artifact write flow：repository 更新 section/chapter/task-unit artifacts。 **[Code-Confirmed]**
4. hierarchy helper flow（被 task-layout/runtime 使用）：提供 `get_effective_sections` 與 find helpers；不等同於擁有 task-layout projection contract。 **[Code-Confirmed] + [From HLD]**
5. enhanced recommendation flow：evaluator 輸出 should_recommend/score/reasons。 **[Code-Confirmed]**
6. hard reparse replacement flow：`save_reparsed_document(...)` sanitizes the accepted replacement document by clearing document/chapter/section/task-unit derived task artifacts before persistence. **[Code-Confirmed]**

## 10. Failure Semantics Matrix

| Scenario | Current Behavior | Notes |
|---|---|---|
| `StructuredDocument` normal 載入僅含舊 `sections`/`structure_nodes` | fail-fast | legacy payload 需走 explicit migration-only loader **[Code-Confirmed] + [Test-Confirmed]** |
| 無 chapters 的 runtime 結構查找 | 上層多為 fail-fast | helper 層逐步收斂 **[Code-Confirmed]** |
| severe hierarchy inconsistency | 上層 coordinator fail-fast | 不以 legacy fallback 掩蓋 **[Code-Confirmed]** |
| parser mode invalid | selector/上層拋錯 | 依 route/coordinator 轉 HTTP **[Code-Confirmed]** |

## 11. Persistence / Side Effects

- read persistence：是（structured store/repository）
- write persistence：是（structured store/repository）
- mutate structured document：是（builder/repository update）
- generate runtime projection：否（主要由 coordinator/task_layout DTO 層完成）
- call LLM：部分（`llm_section_splitter`）
- diagnostics only：否（同時有寫入/建模責任）

## 12. Known Legacy / Compatibility Behavior

| Legacy Item | Can Read | New Write | Runtime Primary Path |
|---|---:|---:|---:|
| root `sections[]` | Migration-only helper | No (default) | No |
| `structure_nodes[]` | Migration-only helper | No (default) | No |
| sections-only payload migration | Explicit only | Not automatic in normal repository write path | No |
| root artifact mirror | N/A | No | No |

說明：`find_*_effective(...allow_legacy_fallback=...)` 已完成退場；普通 helper API surface 不再暴露該參數，effective lookup 現為 hierarchy-only，sections-only legacy 文檔不再經由 effective helper 解析。 **[Code-Confirmed] + [Test-Confirmed] + [Maintainer-Confirmed]**
補充：legacy payload 若需處理，需走 explicit migration-only loader（如 `StructuredDocument.from_legacy_dict_for_migration` / `from_legacy_json_for_migration` 與 repository internal migration helper），不經 normal `from_dict/from_json` 與 ordinary repository write/read path。 **[Code-Confirmed]**

## 13. Terminology Governance Audit

| Term | Current Meaning in This Module | Classification | Governance Decision |
|---|---|---|---|
| root `sections[]` | legacy compatibility input field；normal `from_dict/from_json` 不消費，僅 migration-only loader 可讀 | **[Code-Confirmed]** | 不得描述為 primary persistence source |
| top-level sections | 若指 root `sections[]`，僅 compatibility 語境可用；正式契約應改稱 `chapters[].sections[]` hierarchy | **[Doc-Confirmed] + [Code-Confirmed]** | 在 module 文檔中避免作 current contract 用語 |
| `structure_nodes[]` | legacy experimental hierarchy field；normal `from_dict/from_json` 不消費，僅 migration-only loader 可讀 | **[Code-Confirmed]** | 僅保留 old JSON / round-trip compatibility 語義 |
| `StructuredDocumentNode` | compatibility type，非主流程 hierarchy source | **[Code-Confirmed]** | 不得在架構圖描述為 active main flow |
| flat `task_units` | 非 persistence truth；task units 應掛載於 section (`chapters[].sections[].task_units[]`) | **[Code-Confirmed] + [From HLD]** | 禁止作新 write contract |
| mirror（deprecated terminology） | 僅 historical/compatibility 搜尋語境；正式術語改為 `legacy compatibility fields` / `compatibility-only fields` | **[Maintainer-Confirmed] + [Doc-Confirmed]** | 不得作為現行 architecture contract/persistence authority/runtime primary path 術語 |
| legacy | 指 migration-only compatibility，不代表 runtime primary path | **[Doc-Confirmed] + [Code-Confirmed]** | 文檔必須顯式區分 compatibility vs primary contract |

### Terminology Validation Notes

1. 本 module 文檔現已將 `root sections[]` 與 `structure_nodes[]` 固定為 compatibility 語義，不再暗示主流程來源。  
2. task-layout 與 diagnostics ownership 已明確放在 module boundary 外，避免責任漂移。  
3. metadata / LLM classification 被明確標記為 advisory，非 parser authority。  
4. `mirror` 已退出正式 contract wording；文檔統一以 `legacy compatibility fields` / `compatibility-only fields` 表述。 **[Maintainer-Confirmed]**  
5. `allow_legacy_fallback` 已從普通 helper API surface 移除；runtime primary lookup 已不依賴 legacy fallback。 **[Code-Confirmed] + [Test-Confirmed]**  

## 14. Artifact Governance Boundary

### 14.1 Hierarchy Truth vs Artifact Output

1. hierarchy truth source 固定為 `chapters[].sections[].task_units[]`；artifact payload 不得成為 hierarchy truth source。 **[Code-Confirmed] + [From HLD]**  
2. section/task-unit artifact 僅作 interaction output，依既有 hierarchy target 更新，不可反向創建/重命名 hierarchy identity。 **[Code-Confirmed]**  
3. chapter summary/quiz artifact 目前持久化於 `document_task_artifacts.chapter_artifacts`（id-first key + legacy key candidate），屬內容輸出索引，不是 hierarchy identity source。 **[Code-Confirmed]**

### 14.2 Persistence Ownership

1. `document_structure` 擁有 artifact persistence primitives（repository contract + atomic save + hierarchy consistency guard）。 **[Code-Confirmed]**  
2. `document_structure` 不擁有 artifact availability projection contract；availability/validity 展示屬 coordinator + task-layout DTO。 **[Code-Confirmed] + [From HLD]**  
3. `document_structure` 不擁有 profile diagnostics projection contract。 **[Code-Confirmed] + [From HLD]**
4. Parser-level hard reparse replacement clears derived artifact payloads and referenced-artifact metadata from the replacement structured document before save; ordinary artifact/task-layout updates do not use this cleanup path. **[Code-Confirmed]**

### 14.3 Runtime Projection Boundary

1. availability（`has_summary/has_quiz/cache_valid`）是 runtime projection concern，不是 persisted hierarchy truth。 **[Code-Confirmed]**  
2. task-layout 讀路徑應消費 repository 已持久化 artifact 與 hierarchy，並做當次可用性判斷；該投影邏輯不在本 module。 **[Code-Confirmed] + [Inferred]**

### 14.4 Compatibility Boundary

1. legacy 可讀/migration 可保留，但不得重新引入 root sections mirror 或 flat task_units 作新 artifact write contract。 **[Code-Confirmed] + [From HLD]**  
2. `structure_nodes` compatibility 不延伸到 artifact truth contract。 **[Code-Confirmed]**

## 15. Storage Ownership Boundary

### 15.1 Owned By `document_structure`

1. `document_structure` owns the hierarchy persistence contract: `chapters[].sections[].task_units[]` remains the structured document truth source. **[Code-Confirmed] + [From HLD]**
2. `document_structure` owns the structured document storage contract through `StructuredDocumentStore`, `DocumentArtifactRepository`, and the hierarchy-aware concrete repository. **[Code-Confirmed]**
3. `document_structure` owns hierarchy-aware artifact write primitives only insofar as artifacts are persisted against existing hierarchy targets. **[Code-Confirmed]**

### 15.2 Not Owned By `document_structure`

1. `document_structure` does not own profile persistence; profile artifacts and advisory metadata snapshots belong to `profile/`. **[Code-Confirmed] + [From HLD]**
2. `document_structure` does not own retrieval persistence; FAISS index, node records, and fingerprint metadata belong to `retrieval/`, `fingerprint_handler.py`, and bundle orchestration. **[Code-Confirmed] + [Inferred]**
3. `document_structure` does not own runtime cache persistence; runtime bundle cache belongs to `bundle_factory.py` / `bundle_provider.py` and is not authoritative. **[Code-Confirmed] + [Inferred]**
4. `document_structure` does not own raw uploaded file storage; canonical raw document loading belongs to `doc_loaders/`. **[Code-Confirmed] + [Inferred]**

### 15.3 Storage Abstraction Boundary

1. `document_structure` owns the `StructuredDocument` contract and hierarchy truth. **[Code-Confirmed]**
2. `document_structure` owns the domain semantics of structured persistence across file and future DB-backed representations. **[Maintainer-Provided] + [Future Direction]**
3. `document_structure` does not own storage backend selection; that belongs to future `config/` backend policy. **[Maintainer-Provided] + [Future Direction]**
4. `document_structure` does not own DB rollout strategy, migration rollout switches, or backend enablement policy. **[Maintainer-Provided] + [Future Direction]**
5. A future storage abstraction must preserve hierarchy-first `StructuredDocument` semantics and must not make the backend, schema, or file path a new hierarchy authority. **[Maintainer-Provided] + [Future Direction]**

## 16. Current Risks

1. risk：新 helper/feature 誤把 legacy sections 重新引入 runtime lookup
- why：會破壞 hierarchy-only 路徑一致性
- guardrail：維持 effective lookup hierarchy-only regression + coordinator fail-fast regression

2. risk：common/llm parser mode decision 無統一契約
- why：可能出現 recommendation 與實際行為落差
- guardrail：補 recommendation decision contract 文檔

3. risk：artifact 寫入若偏離 hierarchy source
- why：可能再度形成 dual representation drift
- guardrail：維持 repository hierarchy-aware write tests

4. risk：Part->Chapter 非目標若未清晰文件化
- why：後續開發者可能重引入多層 persistence
- guardrail：在 module docs/non-goals 明確固定 current scope **[From Proposal] + [From HLD]**

## 17. Open Questions for Maintainer

1. `llm_section_splitter` 的輸出契約是否要加入更明確 schema guard（僅文檔層）？
2. enhanced recommendation 的 score threshold 調整責任層級在哪（config 或 evaluator 固化）？
3. chapter summary/quiz artifact 長期是否固定以 `document_task_artifacts.chapter_artifacts` 為唯一寫入權威（chapter node `task_artifacts` 僅作投影輔助）？
4. `legacy compatibility fields`（原 historical mirror wording）是否需在全專案文檔補一份 alias 對照表以利遷移搜尋？

## 18. Suggested Next Documentation Improvements

1. 增加「common vs llm enhanced parser lifecycle」sequence diagram。
2. 增加 artifact repository write boundary 狀態圖（section/chapter/task-unit/document-level）。
3. 補 `document_structure_language_registry` 的 consumer matrix。

## 19. Future Direction Note: Rich Task-Unit Content Governance Preparation

> 本節屬 future-direction governance preparation，非當前 implementation。 **[Inferred]**

1. hierarchy source 固定為 `chapters[].sections[].task_units[]`；不得把 `Document -> Chapter -> Section -> TaskUnit -> ContentBlock` 描述成 persisted hierarchy contract。 **[Code-Confirmed] + [Inferred]**
2. rich content content-block/segment model 的定位是 `task_unit` 內部 render segmentation / interaction segmentation，不是 hierarchy node、不是 persisted structure authority、也不是 runtime navigation hierarchy。 **[Inferred]**
3. content block 不得替代 `task_unit`、`section`、`chapter` ownership；`TaskUnit` 仍是主要 interaction container。 **[Code-Confirmed] + [Inferred]**
4. `content_block_id` / `content_segment_id` 的定位是 interaction target id，不是 hierarchy node id。 **[From Proposal] + [Inferred]**
5. future content-block artifacts 可作 interaction/annotation/evidence target，但仍不是 hierarchy truth source，且不得反向決定 chapter/section ownership 或 task-unit identity。 **[Inferred]**
6. rich content governance 需避免 dual hierarchy representation（例如把 content block 漂移為 structure authority 或 retrieval authority）。 **[Inferred]**
7. 目前 `task_unit.content` 為 string；兼容方向可規劃 `string -> single content block` adapter，但該 adapter 屬 future compatibility strategy，不是 legacy fallback runtime path。 **[Code-Confirmed] + [Inferred]**
8. 本節僅做 governance/邊界對齊；不引入 persistence schema、migration algorithm、runtime API 或 execution model。 **[Doc-Confirmed]**

## 20. Future Direction Note: Artifact Target Validation Boundary Preparation

> 本節屬 validation-boundary governance preparation，非當前 repository/runtime implementation。 **[Future Direction] + [Maintainer-Confirmed]**

### 20.1 Current Boundary (What Is Already True)

1. `ArtifactTargetRef` 目前語義是 metadata + target intent，不是 validated persistence truth。 **[Code-Confirmed] + [Doc-Confirmed]**
2. hierarchy truth source 仍固定為 `chapters[].sections[].task_units[]`；content block 不是 hierarchy node。 **[Code-Confirmed]**
3. content endpoint 現階段僅 pass through target metadata，不查 artifact repository、不驗證 artifact existence、不創建 persistence target。 **[Doc-Confirmed]**

### 20.2 Future Validation Lifecycle (Design Boundary)

`ArtifactTargetRef` (request intent)
-> hierarchy-aware validation
-> resolved target identity
-> repository trust boundary

治理要求：repository 層不得直接 trust target ref；必須先完成 hierarchy-aware resolution。 **[Future Direction] + [Maintainer-Confirmed]**

### 20.3 Reparse / Restructure Stale-Ref Semantics

以下情況可使 target ref 失效：

1. task-unit split policy 改變
2. content-block id regeneration
3. section/chapter restructure
4. source text edit
5. `source_hash` mismatch

治理語義：stale ref != malformed payload；future validation layer 必須可區分 stale/unresolved/malformed。 **[Future Direction] + [Maintainer-Confirmed]**

### 20.4 Failure Boundary (Fail-Fast Direction)

future validation boundary 應 fail-fast 於以下類型：

1. invalid `target_level`
2. required id 缺失
3. hierarchy mismatch
4. `content_block_id` not found under selected `task_unit_id`
5. `source_hash` mismatch
6. stale ref after reparse/restructure

並區分 error class：malformed / unresolved / stale / hierarchy-mismatched。 **[Future Direction] + [Maintainer-Confirmed]**

### 20.5 Allowed Target Combinations (Endpoint Context Policy)

1. `target_level=content_block`：必須至少包含 `task_unit_id` + `content_block_id`。
2. `target_level=task_unit`：必須至少包含 `task_unit_id`。
3. `target_level=document/chapter/section`：可保留 enum vocabulary，但本階段不宣稱完整 artifact persistence semantics。

本政策用於避免 enum 漂移被誤解為完整 persistence 支援。 **[Doc-Confirmed] + [Maintainer-Confirmed]**

### 20.6 Minimum Metadata Glossary (Cross-Module Anti-Drift)

允許的最小 glossary key：

1. `source_hash`
2. `content_block_id`
3. `quote_span_start`
4. `quote_span_end`
5. `schema_version`

治理要求：metadata glossary 用於 interaction/reference pass-through，非 parser authority、非 hierarchy truth、非 artifact persistence truth。 **[Doc-Confirmed] + [Maintainer-Confirmed]**

### 20.7 Non-Goals In This Pass

1. 不實作 repository validation engine。
2. 不實作 artifact read/write flow。
3. 不實作 persistence schema migration。
4. 不接入 retrieval/question/evaluated_answer/LLM integration。

以上僅為 future design boundary preparation。 **[Doc-Confirmed]**

## 21. Future Direction Note: Content Block Segmentation Boundary Design Preparation

> 本節僅做 segmentation governance/boundary 設計準備，不代表 segmentation algorithm 已實作。 **[Doc-Confirmed]**

### 21.1 Segmentation vs Hierarchy Parser Boundary

1. hierarchy parser responsibility 仍是建立/維護 `chapters[].sections[].task_units[]` 主契約。 **[Code-Confirmed] + [From HLD]**
2. segmentation responsibility 定位於 task-unit 內部內容切分（render/interaction segmentation），不重新定義 hierarchy。 **[Maintainer-Confirmed] + [Future Direction]**
3. `task_unit` boundary 與 `content_block` boundary 必須分離：`task_unit` 是 hierarchy-interaction container；`content_block` 是其內部 interaction target。 **[Maintainer-Confirmed] + [Future Direction]**

### 21.2 Content-Block Persistence Semantics (Non-Authority Rule)

1. 即使 future `content_blocks` 有 persisted 表示，content block 仍不是 hierarchy truth、不是 structure authority、不是 runtime navigation hierarchy。 **[Maintainer-Confirmed] + [Future Direction]**
2. 禁止形成 `Document -> Chapter -> Section -> TaskUnit -> ContentBlock` persisted hierarchy model。 **[Maintainer-Confirmed] + [Future Direction]**
3. content block 不得覆蓋 chapter/section ownership，也不得替代 task-unit identity。 **[Maintainer-Confirmed] + [Future Direction]**

### 21.3 Stale-Target Semantics for Reparse/Resegmentation

1. reparse/resegmentation 後，`content_block_id` 可能 stale。 **[Maintainer-Confirmed] + [Future Direction]**
2. stale target != malformed payload；後續邊界需區分 malformed / unresolved / stale / source-mismatched。 **[Maintainer-Confirmed] + [Future Direction]**
3. stale detection 方向可依賴 `source_hash`、`quote_span`、`segmentation_version`、`content fingerprint`，但本輪不定義 executable detection engine。 **[Maintainer-Confirmed] + [Future Direction]**

### 21.4 content_block_id Dependency Semantics

1. `content_block_id` 必須依附 `task_unit_id` 語境，不可脫離 task unit 成為獨立 hierarchy node。 **[Maintainer-Confirmed] + [Future Direction]**
2. `content_block_id` 是 interaction target id，不是 hierarchy node id。 **[Code-Confirmed] + [Maintainer-Confirmed]**
3. segmentation 不能改變 `task_unit_id` identity，不得反向重寫 hierarchy ownership。 **[Maintainer-Confirmed] + [Future Direction]**

### 21.5 Legacy String Compatibility Direction

1. 現況 `task_unit.content`（string）仍是 compatibility field，`string -> block` 屬 adapter strategy。 **[Code-Confirmed]**
2. future multi-block strategy 不得變成 legacy fallback runtime path，也不得回退 root-sections/structure_nodes compatibility 主流程。 **[Maintainer-Confirmed] + [Future Direction]**
3. compatibility 演進需避免 dual hierarchy representation。 **[Maintainer-Confirmed] + [Future Direction]**

### 21.6 Segmentation Non-Authority and Artifact Boundary Guardrails

1. segmentation 不得改變 hierarchy truth，不得成為 parser authority 或 retrieval authority。 **[Maintainer-Confirmed] + [Future Direction]**
2. `ArtifactTargetRef` 與 content-block target metadata 不得漂移為 persistence truth 或 hierarchy authority。 **[Maintainer-Confirmed] + [Future Direction]**
3. 允許 resegmentation policy change 與 content block regeneration；不得假設 `content_block_id` 永久穩定。 **[Maintainer-Confirmed] + [Future Direction]**
4. 本節不引入 segmentation algorithm、persistence schema、repository logic、runtime API、stale-detection engine、retrieval integration。 **[Doc-Confirmed]**

## 22. Future Direction Note: DB-Centric Structured Persistence Migration

> 本節屬 DB-centric structured persistence migration 的 documentation/governance preparation，非目前 implementation。 **[Maintainer-Provided] + [Future Direction]**

### 22.1 Current Active Implementation Boundary

1. 目前 active structured persistence implementation 仍是 structured JSON file storage。 **[Code-Confirmed]**
2. 現有 `data/structured/*.structured.json` file path 在 DB migration 期間仍是有效 storage path，不得在本階段移除或破壞。 **[Maintainer-Provided]**
3. DB storage 是 future representation target，不是目前 runtime read/write behavior。 **[Maintainer-Provided] + [Future Direction]**

### 22.2 StructuredDocument Contract Preservation

1. DB storage 必須保留 `StructuredDocument` hierarchy contract。 **[Maintainer-Provided] + [Future Direction]**
2. hierarchy truth 仍固定為 `chapters[].sections[].task_units[]`。 **[Code-Confirmed] + [Maintainer-Provided]**
3. DB rows、JSONB documents、relational tables 都只是 persistence representations，不是新的 parser authority、structure authority、或 hierarchy identity authority。 **[Maintainer-Provided] + [Future Direction]**
4. DB schema 不得改變 chapter/section/task-unit identity semantics。 **[Maintainer-Provided] + [Future Direction]**

### 22.3 Legacy and Repository Boundary

1. DB migration 必須保留 explicit legacy migration-only boundary。 **[Maintainer-Provided] + [Future Direction]**
2. DB-backed repository 必須仍是 hierarchy-aware repository，不得信任 root `sections[]`、`structure_nodes[]`、或 flat `task_units` 作 primary flow。 **[Maintainer-Provided] + [Future Direction]**
3. DB read-path switch 前必須先定義 validation rules，確認 hierarchy parity、target resolution、artifact target consistency、以及 fail-fast error behavior。 **[Maintainer-Provided] + [Future Direction]**
4. file-backed structured JSON 在 DB readiness 被驗證前仍可作 compatibility / fallback / migration source。 **[Maintainer-Provided] + [Future Direction]**

### 22.4 Separate Persistence Concerns

1. content blocks 是 task-unit internal interaction/render concern，不是 structured hierarchy persistence authority。 **[Maintainer-Provided] + [Future Direction]**
2. artifacts 是 interaction output；artifact persistence migration 需要獨立 track，不應混入 hierarchy truth contract。 **[Code-Confirmed] + [Maintainer-Provided]**
3. profile metadata 是 advisory；profile DB migration 需要獨立 track，不得成為 parser authority。 **[Code-Confirmed] + [Maintainer-Provided]**
4. retrieval records / FAISS artifacts 是 retrieval persistence concern，需要獨立 DB migration track。 **[Maintainer-Provided] + [Future Direction]**
5. raw files / user-uploaded documents 是 user-scoped ownership/copyright concern；DB migration 不代表跨使用者共享文件內容。 **[Maintainer-Provided]**
6. 本節不引入 DB schema、DB dependency、repository implementation、runtime/API behavior change、data migration tooling、或 `data/` retirement execution。 **[Doc-Confirmed]**

## 23. Future Direction Note: Source-Agnostic Manual Structure Override

> 本節記錄「任何文檔解析失敗時」的人工結構修正方向，不限於 OCR/scanned PDF。這是 future-direction design，不代表目前已有 API 或 parser implementation。 **[Maintainer-Provided] + [Future Direction]**

### 23.1 Problem Scope

1. 任何 raw source 都可能產生錯誤 hierarchy：born-digital PDF、掃描 OCR PDF、TXT、EPUB-like text dump、LLM split plan、common parser、native outline、TOC projection 都可能失敗。 **[Maintainer-Provided] + [Inferred]**
2. manual structure override 的目的，是在自動 parser / outline / TOC / LLM fallback 失敗後，由使用者提供顯式目錄或結構邊界，觸發一次明確 reparse。 **[Maintainer-Provided] + [Future Direction]**
3. 這不是 OCR 專用補丁，也不是只為 `暗水幽靈` 設計；OCR 只是其中一個會暴露自動目錄失敗的來源。 **[Maintainer-Provided]**

### 23.2 Authority Boundary

1. 使用者提供的 manual TOC / manual structure plan 是 explicit parser input，不是第二套 persisted hierarchy。 **[Future Direction]**
2. reparse 成功後，唯一 runtime hierarchy truth 仍必須是 `chapters[].sections[].task_units[]`。 **[Code-Confirmed] + [Future Direction]**
3. manual plan 不得作為 task-layout read path 的 hidden mutation，也不得直接編輯 task-layout projection。 **[From HLD] + [Future Direction]**
4. manual plan 不得重新引入 root `sections[]`、`structure_nodes[]`、或 flat `task_units` 作 primary flow。 **[From HLD] + [Future Direction]**
5. manual plan provenance 可保存在 `parse_provenance`，但 provenance 是 observability，不是 parser authority 或 fallback hierarchy source。 **[Future Direction]**

### 23.3 Input Model Direction

最小 manual structure entry 應描述使用者意圖與可驗證定位：

```text
title
level
start anchor
optional end anchor
optional external/user id
optional notes
```

anchor 可分階段支持：

1. `char_start` / `char_end`：適用於任何已抽取 raw text 的文檔，最 source-agnostic。 **[Future Direction]**
2. `page_index` / `page_range`：適用於 PDF/OCR/native page-boundary 可用的文檔。 **[Future Direction]**
3. `printed_page_number + offset_hypothesis`：可作後續擴展，不應作 MVP 唯一定位方式。 **[Future Direction]**
4. `title_match_hint`：只能作 validation/evidence 輔助，不得單獨授權 projection。 **[Future Direction]**

### 23.4 Projection Rules

1. manual structure projection 必須收斂到現有 two-layer normalized hierarchy。 **[From HLD] + [Future Direction]**
2. `level=1` 可形成 chapter；`level=2` 可形成 section。 **[Future Direction]**
3. 只有 chapter entries 時，每個 chapter 應生成一個同名 section，以維持 `chapters[].sections[]` contract。 **[Future Direction]**
4. manual structure MVP 僅允許 `level=1` chapter 與 `level=2` section；`level>2` 必須 fail-fast，不得自動折疊或隱式合併。 **[Maintainer-Provided] + [Future Direction]**
5. task-unit generation 仍是 downstream concern，不由 manual TOC 直接持久化 flat task units。 **[Future Direction]**

### 23.5 Validation and Failure Semantics

manual plan 必須先 validation，再 commit reparse：

1. entry title 不得為空。 **[Future Direction]**
2. anchors 必須在 raw text/page boundary 範圍內。 **[Future Direction]**
3. ranges 必須單調、非重疊、非空。 **[Future Direction]**
4. hierarchy levels 必須只包含 chapter/section 兩層；任何 `level>2` 或跳層結構都應 fail-fast。 **[Maintainer-Provided] + [Future Direction]**
5. partial projection 不得落盤；validation 失敗時應保留現有 structured document。 **[Future Direction]**
6. validation failure 不應 silent fallback 到 common parser 並宣稱 manual reparse 成功。 **[Future Direction]**

### 23.6 Reparse Lifecycle Direction

建議 lifecycle：

```text
manual structure request
  -> validate / preview
  -> explicit commit reparse
  -> build StructuredDocument
  -> save single active hierarchy source
  -> downstream task-layout reads normal hierarchy
```

成功 reparse 的 `parse_provenance` 建議包含：

```text
requested_parser_mode=manual_structure
effective_parser_mode=manual_structure_projection
source=user_supplied_structure
fallback_used=false
manual_entry_count
anchor_type
validation_summary
```

本方向不實作 API、schema、builder、repository、或 UI；僅固定 future design boundary。 **[Doc-Confirmed]**

## 24. Page-Backed Manual Structure Anchors

> 本節支援 UI TOC editor 的 page-first anchor UX。Manual `page_range` draft building is implemented when preparation-provided page-boundary evidence is available; task-layout prefill projection remains a separate checkpoint. **[Maintainer-Provided] + [Code-Confirmed] + [Future Direction]**

1. `document_structure/` owns deterministic validation and hierarchy projection semantics for manual structure anchors. **[Code-Confirmed] + [From HLD]**
2. Manual projection can represent `page_range`, and manual draft building can resolve page-backed anchors into raw-text spans when validated page boundaries are supplied. **[Code-Confirmed]**
3. Page-backed manual commit resolves validated `page_range` anchors using preparation-provided page boundaries before building a hierarchy-only `StructuredDocument`. **[Code-Confirmed]**
4. Page-backed anchors are validated for page-boundary availability, unique page-index evidence, page existence/resolvability, sibling overlap, empty projected ranges, stale source evidence, and source hash consistency through the projection/source-evidence/draft-build gates. **[Code-Confirmed]**
5. Accepted manual page anchors record advisory parse provenance including anchor type, source hash, validation summary, entry count, chapter count, and section count. Provenance is observability only and must not become a second hierarchy source. **[Code-Confirmed]**
6. `project_structure_anchor_evidence(...)` can expose lightweight chapter/section anchor evidence from the existing parsed hierarchy for task-layout/UI prefill. This evidence is read-only projection metadata and does not rewrite hierarchy identity or artifacts. **[Code-Confirmed]**
7. If page evidence is missing or ambiguous, including duplicate `page_index` boundaries, validation fails for `page_range` and leaves `char_range` as the explicit fallback. **[Maintainer-Provided] + [Code-Confirmed]**
8. Existing-structure anchor evidence prefers `page_range` when validated page boundaries cover the current hierarchy spans, falls back to `char_range` when page evidence is absent, and marks missing/invalid spans as unavailable instead of inventing anchors from titles, OCR guesses, profile metadata, or task-layout state. **[Code-Confirmed]**

## 25. TOC Shape Classification and Two-Layer Projection

> 本節記錄 deterministic TOC projection 的 shape handling。 **[Code-Confirmed]**

1. `TableOfContentsDetector` classifies TOC evidence into `flat_chapter`, `chapter_section`, `deep_hierarchy`, or `unknown` without using metadata or LLM output as parser authority. **[Code-Confirmed]**
2. Validated TOC projection maps flat chapter entries to chapter-level sections; `structured_hierarchy_builder` then materializes one chapter with one same-name section for each entry. **[Code-Confirmed]**
3. Validated chapter/section TOCs map level-1 entries to `toc_chapter` and level-2 entries to `toc_subsection`, producing `chapters[].sections[]` without root `sections[]` or `structure_nodes` as persisted truth. **[Code-Confirmed]**
4. Deep TOC levels (`level > 2`) are collapsed into the nearest projected section content range rather than persisted as a third hierarchy level. **[Code-Confirmed]**
5. TOC projection provenance records original level, projected level, printed page number, resolved source page index, and merge reason for collapsed entries. Provenance is observability only and does not become a second hierarchy source. **[Code-Confirmed]**
6. Task-unit generation remains downstream of the resulting two-layer sections; TOC projection does not write flat task units or artifact targets. **[From HLD] + [Code-Confirmed]**

## 26. TOC Page/Character Boundary Validation and Atomic Fallback

> 本節記錄 validated TOC projection 的 page/char gate。 **[Code-Confirmed]**

1. TOC projection requires body title matches after the detected TOC region; insufficient body-title recall rejects projection without partially applying TOC sections. **[Code-Confirmed]**
2. Printed page numbers are validated against body title offsets through one explicit page-offset hypothesis. Mixed or incompatible offsets reject projection atomically. **[Code-Confirmed]**
3. Printed page anchors must resolve to available source page boundaries after applying the offset; missing page anchors reject projection. **[Code-Confirmed]**
4. Projected section ranges must be non-empty, monotonic, and non-overlapping before `StructuredSection` objects are materialized. **[Code-Confirmed]**
5. When any TOC projection gate fails, `StructuredDocumentBuilder` preserves the current parser result and does not write `toc_projection` provenance or claim `validated_toc_projection` as the effective parser. **[Code-Confirmed]**
6. This path does not introduce root `sections[]` persistence, `structure_nodes` main flow, hidden task-layout mutation, profile write-back, or LLM/parser-metadata authority. **[Code-Confirmed] + [From HLD]**

## 27. Universal Page-Level TOC Candidate Scoring

> 本節記錄 page-level TOC candidate scoring。Candidate detection identifies pages worth reconstructing or validating later; it does not by itself authorize structure splitting. **[Code-Confirmed]**

1. `TableOfContentsDetector.detect_page_candidates(...)` assigns deterministic scores from page-local evidence and returns candidate pages once the score reaches the conservative threshold. **[Code-Confirmed]**
2. Scoring covers document-relative early position, horizontal/vertical writing mode, left-to-right/right-to-left reading order, multiple text columns, coordinate OCR availability, short-title density, dotted leader lines, page-number evidence, separated page-number columns, circled/boxed page numbers, OCR-fragmented title density, and explicit TOC markers. **[Code-Confirmed]**
3. Candidate scoring can detect TOC-like pages even when TOC markers are missing or page numbers are visually separated from titles. Missing or unreliable anchors still require later entry reconstruction and global validation before projection. **[Code-Confirmed]**
4. `detected` and splitting usability remain separate decisions: page candidates may raise `detected=true`, while `validate_for_projection(...)` and `StructuredDocumentBuilder` still enforce entry count, page-number coverage, body-title matches, monotonic page mapping, and non-overlapping ranges before any hierarchy replacement. **[Code-Confirmed]**
5. This path uses OCR/layout text evidence only; it does not use metadata, diagnostics profiles, or LLM classification as parser authority and does not persist an alternate hierarchy source. **[Code-Confirmed] + [From HLD]**

## 28. Geometry-First TOC Entry Reconstruction

> 本節記錄 page-level TOC entry reconstruction。Reconstruction creates auditable title/page-number pairs from OCR geometry for later validation; it does not by itself authorize hierarchy replacement. **[Code-Confirmed]**

1. `PdfPageLayoutEvidence` can retain OCR word-level geometry from TSV input, including text, confidence, bounding box, and block/paragraph/line/word indices. This evidence is page-local parser evidence, not hierarchy truth. **[Code-Confirmed]**
2. `TableOfContentsDetector.reconstruct_page_entries(...)` reconstructs TOC candidate entries from geometry when word boxes are available. The output records title, page number, source page index, title region box, page-number region box, confidence, and evidence reasons. **[Code-Confirmed]**
3. Horizontal TOC reconstruction groups OCR words by row, orders title regions left-to-right, excludes leader tokens from titles, and pairs the title with the right-side page-number region by row alignment and leader-line endpoint evidence. **[Code-Confirmed]**
4. Vertical TOC reconstruction clusters OCR words into columns, sorts right-to-left pages by descending x-coordinate, orders title characters top-to-bottom within each column, and pairs the title with the lower page-number region. **[Code-Confirmed]**
5. Circled page numbers are recognized during geometry reconstruction for bounded page-number regions. The current implementation reconstructs candidate pairs only; later checkpoints still own OCR quality gates, global page-number validation, and projection authorization. **[Code-Confirmed]**
6. Geometry reconstruction is detection-only and must not replace raw text, page-boundary evidence, `chapters[].sections[]`, or parser validation. It does not write task-layout state, profile diagnostics, artifacts, root `sections[]`, or `structure_nodes`. **[Code-Confirmed] + [From HLD]**

## 29. Logical Reading-Order Normalization With Coordinate Preservation

> 本節記錄 TOC OCR geometry 的 logical reading-order normalization。The normalized stream exists only for deterministic detection and audit; it does not replace raw text or become a hierarchy source. **[Code-Confirmed]**

1. `TableOfContentsDetector.normalize_page_reading_order(...)` converts page OCR word boxes into `TocNormalizedToken` records in logical reading order. Each token retains page index, logical index, group index, normalized text, raw OCR text, original coordinate box, confidence, writing mode, reading order, rotation/orientation, and order hypothesis. **[Code-Confirmed]**
2. Horizontal pages use row grouping with left-to-right or right-to-left ordering according to the page reading order. Vertical pages use column grouping with top-to-bottom ordering within each column and right-to-left column sequencing when page evidence indicates RTL. **[Code-Confirmed]**
3. `TableOfContentsDetector.reconstruct_normalized_page_pairs(...)` exposes normalized title/page pairs with page index, title/page-number boxes, raw OCR sequence, writing mode, reading order, rotation, order hypothesis, evidence reasons, confidence, and confidence breakdown. **[Code-Confirmed]**
4. The normalized stream preserves raw OCR and coordinates for audit while giving detection code a stable logical order. It must not overwrite `PdfPageLayoutEvidence.ocr_text`, page-boundary evidence, source hashes, or extracted raw text. **[Code-Confirmed]**
5. Normalized pairs are detection-only and do not authorize projection. Later quality gates and global validation still decide whether entries are usable for splitting. **[Code-Confirmed] + [From HLD]**
6. This path does not introduce root `sections[]`, `structure_nodes`, hidden task-layout mutation, diagnostics profile write-back, metadata authority, or LLM parser authority. **[Code-Confirmed] + [From HLD]**

## 30. TOC-Specific OCR Quality Gates

> 本節記錄 TOC OCR quality gates。A page can be detected as TOC-like while still rejected as unusable for splitting when OCR entries are incomplete or unreliable. **[Code-Confirmed]**

1. `TableOfContentsDetector.evaluate_toc_ocr_quality(...)` evaluates page-local TOC OCR quality without mutating parser output or authorizing projection. It returns `detected` separately from `usable_for_splitting`. **[Code-Confirmed]**
2. The quality gate measures reconstructed entry count, title completeness, page-number coverage, geometric title/page pairing coverage, page-number ordering consistency, body-title recall, confidence, and explicit rejection reasons. **[Code-Confirmed]**
3. A high-scoring TOC-like page with missing or corrupt page numbers can remain `detected=true` while `usable_for_splitting=false`. The gate does not infer unreadable page numbers from sequence, row count, or title order. **[Code-Confirmed]**
4. Split usability requires enough reconstructed entries, complete titles, sufficient page-number coverage, paired title/page-number geometry, monotonic page numbers, and body-title recall against supplied body text. Missing body text rejects usability rather than silently trusting OCR layout. **[Code-Confirmed]**
5. Quality gate rejection preserves reasons such as missing reconstructed entries, incomplete titles, missing page numbers, missing geometric pairs, page-order failure, low body-title recall, and missing body text. **[Code-Confirmed]**
6. This path is detection/validation evidence only. It does not persist hierarchy, replace current parser output, write task-layout/profile/artifacts, or introduce root `sections[]`, `structure_nodes`, metadata authority, or LLM parser authority. **[Code-Confirmed] + [From HLD]**

## 31. Page-Number Region Recognition and Global Offset Validation

> 本節記錄 TOC page-number region recognition and explicit printed-page offset validation。Region recognition normalizes bounded page-number tokens; global validation decides whether one page-number-to-source-page mapping is reliable enough for later projection. **[Code-Confirmed]**

1. `TableOfContentsDetector.recognize_page_number_regions(...)` emits page-local `TocPageNumberRegion` records from reconstructed TOC title/page pairs. Each record keeps the source page index, raw page-number token, normalized text, parsed value, numeral system, region box, confidence, and evidence reasons. **[Code-Confirmed]**
2. Page-number recognition is region-scoped: only page-number regions paired by TOC geometry are normalized as anchors. The detector does not scan arbitrary body text or profile metadata for parser authority. **[Code-Confirmed] + [From HLD]**
3. Recognition supports Arabic numerals, multi-digit numbers, Chinese numerals, Roman numerals, circled numbers, boxed/bracketed numbers, and fragmented OCR digits such as spaced digit tokens. Raw OCR is preserved so symbols normalized into digits still retain their original evidence. **[Code-Confirmed]**
4. `TableOfContentsDetector.validate_page_number_anchor_offsets(...)` validates explicit printed-page-to-source-page offset hypotheses against supplied page text. A valid result requires monotonic page numbers, plausible source-page anchors, and at least two body-title matches under one unambiguous offset. **[Code-Confirmed]**
5. Ambiguous offsets are rejected atomically. If multiple offsets produce the same sufficient title-match evidence, or if matches are insufficient, page numbers are non-monotonic, page text is missing, or anchors fall out of range, validation returns rejection reasons and no selected offset. **[Code-Confirmed]**
6. This path produces validation evidence only. It does not persist hierarchy, replace current parser output, mutate task-layout/profile/artifacts, or introduce root `sections[]`, `structure_nodes`, metadata authority, or LLM parser authority. **[Code-Confirmed] + [From HLD]**

## 32. Outline-First Lazy Evidence Handoff (OCR-06)

`StructuredDocumentBuilder.build` accepts an optional `page_evidence_provider`
alongside the existing eager `page_evidence`. Native Outline projection runs first;
only an accepted projection can skip optional layout. Mere `outline.usable` is
insufficient: source hashes, ordered page identities, exact raw-text slices,
canonical separator gaps and destinations must validate. Actual spans replace the
old separator guess, so internal double-newline paragraphs do not reject an
otherwise valid single-newline page join. **[Code-Confirmed]**

When Outline is missing/rejected, the existing splitter/selector runs once and
layout is requested once if eager evidence was not provided. Provider exceptions
are advisory, with exception type (never message/body) recorded in existing parse
provenance. Layout evidence with a different or missing hash is rejected when the
canonical boundaries carry known source identity. Text-only callers with no hash
retain compatibility. `layout_evidence.status` and `pdf_layout_schedule` explain
skipping/loading/unavailability/failure/source mismatch without granting parser
authority. TOC projection gates and hierarchy persistence remain unchanged.
**[Code-Confirmed]**

This is supporting work for preparation-owned OCR-06, not an additional task.
Evidence: 15 tests in `scripts/test_prepare_lazy_pdf_layout.py` pass locally and
in isolated Compose; existing Outline, TOC projection and page-evidence scripts
also pass on 2026-10-14. Required raw recovery remains the loader's responsibility
and completes before this builder is invoked. **[Test-Confirmed]**
