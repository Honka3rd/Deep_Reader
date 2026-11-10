# reading_preferences Detailed Design

## Language Learning Reading Mode Plan (2026-10-28)

**[Code-Confirmed; LLRN-02/09 complete]** 本目錄已交付 Python domain contracts、mode catalog、repository port 與 POSIX file adapter。依使用者確認的單人 workspace 過渡方案建立唯一偏好 owner；db 已另交付 LLRN-07 PostgreSQL adapter/顯式 migration；LLRN-09 已交付有效設定服務；LLRN-15 已提供 runtime DI，LLRN-16 已提供 API，尚無用戶系統。

## Generation snapshot validation support (LLRN-13)

LLRN-16 新增穩定 eligibility reason `learning_mode_not_active`（field=`active_mode`），
供 app 拒絕尚未選用 learning 的生成命令。現有正常閱讀設定本身仍 eligible，不因此增加
讀取副作用或把 normal mode 當壞設定；active learning 因母語／原文／能力而失效時，沿用
原有精確 reasons。API schema 同步此有限 reason union；不新增語言或模式清單。

`ReadingPreferencesService.scope` 公開 owner scope；`resolve_snapshot(workspace, book)` 消費 repository 已鎖住的同 scope 設定，沿用 override／existing profile 及 registry eligibility projection。它不重新取得 preference locks、不寫資料或生成 profile；注入的 profile reader 不得取得 document/preference locks 或另開 write transaction。App 在 claim/dispatch/commit 驗證有效 mode、語言與 hints proficiency；profile 仍是獨立 advisory storage，不新增 profile revision／原子跨資源交易保證。File／真 SQL 的設定與 profile race 驗證列於 [app design](../app/module-detailed-design.md#llrn-13-implementation-contract-2026-10-28)。

## Document lifecycle staging (LLRN-12)

`lifecycle.py` 擁有純 `stage_document_lifecycle`：rename 保留模式設定，destination revision 推進，source 留下 normal/empty settings tombstone；delete 只留下 source tombstone，workspace 母語不變。既有活躍 destination 拒絕。由 authoritative document lifecycle 呼叫時允許來源已有 tombstone（同名文件可能重新建立），但一般偏好 CAS／retention 驗證不放寬。

File owner 公開 `lifecycle_lock`、owned data/journal paths、`stage_document_lifecycle`，回傳經驗證的 `PreferenceFileWrite`，不在 staging 時提交。`document_structure.FileLearningDocumentLifecycle` 協調三個 owners，journal 存在時偏好 reads/writes 拒絕，只有顯式 recovery 可向前完成；讀取不修復。SQL owner 以 `apply_document_lifecycle(..., connection=...)` 在來源 store 的同一交易鎖定／保存，無第二筆交易。此層不寫正文或決定 source truth；runtime composition 已由 LLRN-15 交付。

## Responsibility and scope

集中 workspace 母語、book active mode ID、共用可空 source-language override，以及按 mode ID 分開的 typed/versioned settings（語言學習含 proficiency scheme/version/level）；每份設定有 revision。以 server-resolved workspace scope 加 authoritative document identity 查找，不收 client 任意 namespace/user_id。普通模式是無資料時的唯讀預設，不在讀取時補寫。

本模組持有 domain/validation、repository port 和 file adapter；db 實作 SQL/migration；app 協調 existing profile 與 document lifecycle；config 組裝。偏好不放 env、散落 browser storage、parser profile 或臨時對話 session。此 scope 是單人共享工作區，不保證目前不存在的使用者隔離。未來以顯式 migration 將 workspace settings/book records/生成 scope 綁定真 user identity，不能僅替換 UI key 即宣稱完成。

## Mode catalog and activation contract

**[Code-Confirmed catalog/contracts/eligibility]** `models.list_modes()` 純讀返回 immutable definitions，初期 ID 為 `normal`、`language_learning`，各含 label、description、settings_version。這是閱讀模式定義，不是另一份 language registry。LLRN-09 後端服務已依 book/workspace 與既有 profile 提供 eligibility 和原因，前端依 ID 對應本地 feature；catalog 不承載 UI component、可執行程式或任意動態表單。

Book 只有一個 active mode，但可保留各模式已保存的設定。普通模式不受語言學習必填欄位約束；母語/原文 override 保持共用真值，proficiency 由語言學習型別擁有。保存時嚴格校驗 mode/settings/version 相符，未知提交拒絕；已存未知或暫不可用 mode 不在 read 時回寫預設。Mode change 與該模式設定用同一 expected revision 原子保存，拒絕時保留原 mode/settings。

UI 選取候選不等於啟用。只有明確套用命令寫入；保存不啟動模型，不活躍模式也不排程生成。保留其它模式有效 cache 與設定以便返回；重新啟用仍須以最新 workspace/source/version 驗證。未來新增模式需擴充受驗證的設定型別與 mode definition，不開放任意 JSON 或本期不存在的用戶自訂模式。

## Effective settings and writes

LLRN-09 已交付設定讀寫與 eligibility projection；profile loader 由 server composition 注入。LLRN-12 已交付 document lifecycle staging/協調；LLRN-13 已接入生成 admission/dispatch/commit 的有效 profile 重新驗證；runtime DI 已由 LLRN-15 交付。

- 學習語言 = 有效 book source override，否則使用已存 profile language。Profile 讀失敗與未設定要區分；本模組不呼叫語言偵測/LLM 或重建 profile。
- 母語及原文必須屬現有 language registry 且不同；scheme/version/level 必須符合該學習語言。缺少必要值時 learning eligibility=false，保留原文可讀，不猜母語/能力。
- 顯式 save 使用 expected settings revision 做 CAS，回傳 authoritative settings/eligibility；同值 save 不令有效生成 key 漂移。讀取/Library 保存都不生成譯文。
- 原文 override 變更後，不把舊語言 level 帶到新語言；需提交相符分級或顯示待設定。母語改變後，各書按最新設定計算 eligibility；同語書暫停學習投影，不批量覆寫 saved mode。
- Mode/level/mother tongue/source override 變動由 app 切換當頁 cache key，來源有效的舊 cache 可保留；setting revision 用於拒絕在途舊命令覆蓋。普通 Reading display 仍是獨立 session 顯示偏好。
- Book rename 依既有 document identity/namespace lifecycle 保留設定；book delete 清除對應偏好，workspace 母語不受影響。需明確 file crash recovery 與 SQL 交易邊界，不能宣稱檔案/DB 跨資源原子性。

## Implemented preference contracts (LLRN-02)

- `models.py`：immutable `WorkspaceScope(namespace)` 的 kind 固定為 `workspace`，由 server composition 提供，非 client scope DTO；沒有虛構 user ID。Workspace 設定包含 `native_language`、`revision`。Book identity 使用該 scope 下的 authoritative document ID，port 不自行去副檔名或改寫名稱；未來 app 必須先依既有 namespace/document identity 解析後呼叫。
- `BookSettings`：`active_mode`、可空 `source_language_override`、typed `modes` tuple。`LanguageLearningSettings` 包含 `language_code`、`scheme_id`、`scheme_version`、`level_id`、設定 `version="1"`；language_code 綁定能力選項，不能取代原文真值。`BookPreferences` 另帶 revision、deleted tombstone。
- 未有紀錄時 workspace 母語為空、book 為普通模式且 revision=0；constructor、catalog、read 不建立目錄、lock、預設檔案或修改既有資料。未知 persisted mode/settings version 以 immutable `PreservedModeSettings` 保留；可在切回普通模式時保留原值，不能提交新的未知設定或啟用未知模式。壞 JSON、重複 key、錯誤 scope/儲存版本、IO 失敗都不冒充「沒有資料」。
- `repository.py` 提供 workspace/book read/save、book rename/delete。Save 使用 expected revision；book save 同時要求 expected workspace revision，將 mode/settings 一次保存。普通模式不要求學習欄位；切換不得遺失其它 mode 已保存的設定。啟用 learning 時驗證 registry、分級及原文/母語不同。內部 caller 傳入 `resolved_source_language`；此層不讀 profile；LLRN-09 服務已解析原文，LLRN-13 已在生成時重驗有效 profile/source，獨立 profile 儲存仍無跨資源原子保證。設定保存不宣稱與 profile 檔案跨資源原子提交。
- 同值 save 不增加設定 revision。每筆紀錄保存最後一次 mutation ID 與 command signature；相同命令可重放，不同命令重用最後 ID 拒絕；已被後續版本取代的舊命令需重新讀取及處理 conflict。這不是永久 operation history，也沒有 HTTP operation API。母語修改不改寫每書 saved mode/settings；有效 eligibility 由 LLRN-09 計算。

## File persistence and lifecycle boundary

`file_repository.py` 由 caller 注入 base directory，採 scope kind/namespace SHA-256 作為檔名，一個 workspace 保存一份 schema_version=1 metadata snapshot（workspace、book map、revision、receipt）。不存正文或生成 cache。`serialization.py` 負責嚴格版本化 codec；未知資料不在 read 時遷移。

所有寫入共享穩定 POSIX `flock`，在同一鎖內重新讀取、CAS、驗證與提交，避免不同書籍更新互相覆蓋。提交採同目錄 temporary file → file fsync → atomic replace → directory fsync；無鎖讀取只見完整舊版或新版。replace 前已知失敗不改舊資料；replace 開始後失敗回報 `PreferenceCommitUnknown`，由 read 與同命令重放確認結果。程序被終止時遺留的 temp 不會被讀取或自動升為正式資料；讀取也不順便清理它。測試涵蓋程序中斷，不宣稱所有檔案系統、斷電或新建父目錄均已驗證 durability。

LLRN-06 新增公開的 file-only `locked_snapshot(document_id)` 協調 context：在原有 workspace flock 內，以一次讀取返回 immutable workspace/book preferences 並維持鎖到 context 結束。用於 learning 寫入的 source/settings revalidation，固定鎖序 preferences → document → learning sidecar；不得在 context 內重入 preference writes。Context 不保存偏好，普通 `read_*` 仍不建 lock/file。它不是跨檔案寫入交易、effective settings/profile service 或 SQL port 方法；LLRN-07 SQL 偏好寫入已用同一 DB transaction；LLRN-08 SQL learning coordination 已用 `PostgresReadingPreferencesRepository.lock_snapshot(document_id, connection=...)` 在 caller-owned active transaction 中取得 workspace/book snapshot 與 workspace lock，不另開連線或 commit；固定先 preferences 再 document。此 SQL helper 不新增 domain port 方法，不能套用 file-only API 或在 learning commit 中呼叫另一個會自行 commit 的 preference mutation。`test_learning_file_repository` 已驗證 preference writer 不能穿過 learning commit，以及 revision/native/mode/level 改變會擋住過時生成；原有 28 項 preference tests 全數通過。

Rename 在同一份偏好 snapshot 移動 source→destination，兩端 revision 做 CAS；拒絕已存在的 live destination，source 留空設定 tombstone，防止舊請求復活原身份。Delete 同樣清除 book 設定並保留遞增 revision tombstone；workspace 母語不變。重建舊身份必須使用 tombstone 的最新 revision。LLRN-12/15 已將這些 primitives 接入 authoritative document lifecycle；file 使用顯式 journal recovery，SQL 使用來源交易，不能宣稱 file 跨資源原子性。

**R3 explicit reapply clarification (2026-11-10):** 同名重新匯入不在 read 時解除 `deleted`；book read 仍投影 `book_deleted`。正式 book apply 先驗 canonical document 存在，再以目前 workspace/book revisions 顯式保存，清除 tombstone；即使 Normal settings 相同也推進 book revision。Learning 仍重驗 registry/來源/母語/級別。HTTP 舊 expected revision 必須拒絕（包括舊 mutation ID 的重送）；unknown acknowledgement 先 read 對帳，不放寬舊 source/worker fences。對應 file/PostgreSQL HTTP 回歸在 `scripts/test_learning_routes.py`，UI 提供明確 Save and apply 提示與入口。

此 file adapter 適用單人 workspace metadata，每次讀寫會處理該 workspace 全部偏好與 tombstones，成本隨 book 數量增加；不是已完成的 production 大型資料集方案。LLRN-07 SQL adapter 已以有界單筆查詢、workspace row lock/交易/CAS 實作相同語義，不沿用全表 snapshot；見 [SQL 設計與升降版流程](../db/module-detailed-design.md#implemented-preference-sql-persistence-llrn-07)。未來真 user scope 需顯式 migration 同時處理 workspace/book records 及生成資源 ownership，不能改一個 UI key 或在 read 時偷偷轉移。

## Implemented effective settings service (LLRN-09)

**[Code-Confirmed, 2026-10-28]** `service.py` 的 `ReadingPreferencesService` 消費 scope-bound repository port 與 `ExistingProfileLanguageReader`，提供 catalog、workspace read/save、book read/save。`effective.py` 持有 immutable `EffectiveBookPreferences`、`ModeEligibility`、`PreferenceIssue` 及純 projection；含原始 workspace/book/revisions、有效來源與 provenance、registry proficiency scheme/selection、各 catalog mode 與已保存 active mode 的 eligibility。原因附 stable code、field 和可理解文字，transport/localization 留給 API/UI。

- Source override 優先且完全略過不需要的 profile lookup；否則透過 caller 注入的既有唯讀 loader 取得 `document_language_code`。不自行建立 storage config、不讀整本正文、不偵測語言或建立 profile。Loader 收 server scope + exact document ID；app/config 後續負責權威 document identity 與既有 store 的接線。
- Profile 不存在、格式損壞、語言未定/不支援、IO 不可用分別回報 `profile_missing`、`profile_invalid`、`source_language_required`/`source_language_invalid`、`profile_unavailable`，不把失敗冒充缺省。既有 profile decoder 的錯誤型別/形狀屬 invalid；未預期 runtime/database exceptions 原樣上拋。已明確 override 時不讀該失敗來源。Log 僅記 operation/outcome/duration，不記 document identity、profile/private data。
- Learning eligibility 要求有效且不同的母語/原文、目前原文的有效 scheme/version/level 與支援 settings version。全 19 registry languages 使用自己的 proficiency definitions。能力只在與目前原文相符時回傳 resolved selection；舊設定保留於 saved book，不能自動轉用至另一語言。
- 缺少條件、未知 saved mode/version、deleted preference tombstone 會使相應 mode unavailable；`effective_mode=None` 表示暫不可用，不是回寫或偷偷啟用普通模式。普通 source text 仍由既有閱讀路徑提供。母語更改/清空或 profile 語言更改後按新值重算，不批量修改書籍或刪 cache；恢復有效條件可再次得到 eligible projection。
- 普通模式獨立驗證，不要求 learning 設定；已知 profile 缺失/損壞/IO 失敗不阻止正常模式保存。模式往返保留 inactive settings；未知 persisted mode/version 可以保留並切回普通模式，不能作為新設定或被啟用。未實作 eligibility policy 的新 catalog ID 不會自動取得資格。
- Book read 先讀 workspace/book、解析來源，再讀兩筆偏好核對；變動回報 `preference_snapshot_changed`，不無限重試、不建立鎖/檔案/預設資料。最多四次有界偏好讀取、一次必要 profile read；SQL 不掃描所有書籍。這是偏好的一致 snapshot，不是 profile/document 的跨資源交易。
- Book save 在任何寫入前完成來源解析、shape/eligibility projection；repository 在原有鎖/交易內再次驗證 workspace/book revisions、mode/settings 與保留規則，一次 CAS 保存。回傳 committed book record 與本次解析的 eligibility；commit 後不再 IO，避免成功後的讀取故障冒充保存失敗。Workspace/book 並行修改拒絕過時套用；同命令重放沿用 repository receipt。同值保存不增 revision；`PreferenceCommitUnknown` 原樣傳遞，先讀取對帳，服務不自動重試。
- **整合邊界：** profile 是另外保存的 advisory snapshot，這次只保存偏好，不建立 profile revision/新鎖或聲稱與 reparse 原子。若 profile 同時改變，下次 read 必須重新計算 eligibility；LLRN-13 的生成 admission/commit 必須以目前 source/profile 與偏好重新驗證，不可把此 response 當作生成授權。LLRN-12 lifecycle ports 已交付；LLRN-13 application orchestration 已完成；LLRN-15 runtime composition 已完成，production API/lifespan 綁定已由 LLRN-16 完成。

Verification: `scripts/test_reading_preferences_service.py` 64 tests（33 file、31 real PostgreSQL）通過，含已有 `DocumentProfileStore` 真 JSON 讀取/損壞形狀、profile bytes 不變、全部 registry 語言、同語禁用、明確校正、舊版本保留、母語清空/切換、模式往返、並行套用單一勝出者、重複命令、setting races、unknown commit 對帳、提交後零 IO 及 log privacy。另 82 項既有 preference file/SQL/proficiency tests 通過；合計 146 項不同測試。僅 disposable PostgreSQL/random schemas 與 temp files；無 provider 呼叫、真書 mutation、runtime migration、API/UI 接線。

## Verification evidence (LLRN-02, 2026-10-28)

`scripts/test_reading_preferences.py` 共 28 項測試：file reload、唯讀預設、typed settings/registry 防線、scope/identity 隔離、模式保留、CAS/idempotency、workspace revision、未知舊設定、corrupt/IO failure、rename/delete tombstones、實際多程序競爭與 replace 前後程序終止。使用 temporary directories，不改真書資料、不呼叫 LLM。命令：`.venv/bin/python -m unittest scripts.test_reading_preferences scripts.test_language_proficiency_registry -v`（46 項），另以 `.venv/bin/python -m scripts.test_language_script_registry` 及 `.venv/bin/python -m scripts.test_language_discourse_registry` 各執行 5 項既有函式測試；這兩份腳本不是 unittest discovery cases。SQL/API/UI/實際 document lifecycle 不在本次驗收範圍。

## Dependencies and implementation gates

`language` 提供 codes/proficiency；`shared` 僅共享純資料契約；`app` 將消費本模組服務。禁止依賴 UI、routes、provider 或其他模組私有 SQL。File port 已有專測；LLRN-07 SQL adapter 另以 36 項真 DB tests 驗證相同語義、bounded reads、migration/recovery、rollback/concurrency。File/SQL 共用 `commands.py` 的 receipt signature/replay/CAS helpers，無 SQL 私有方法跨模組呼叫。SQL schema 與有效設定服務已交付但尚無 API/runtime wiring，既有 file 資料不自動遷移；完整 document lifecycle 仍待後續任務。

Canonical requirements/data lifecycle: [learning design](../section_tasks/module-detailed-design.md#language-learning-reading-mode-plan-2026-10-28)。唯一 tasks：LLRN-02 契約/file adapter、LLRN-09 驗證服務；SQL 由 db 的 LLRN-07 負責，不重複勾選。
