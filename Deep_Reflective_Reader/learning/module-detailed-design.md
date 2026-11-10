# learning Detailed Design

## Module extraction (2026-11-10)

Status: Implemented and locally verified; maintainer-approved option A. The existing LLRN-13
and completion-notification contracts below migrate from app without changing
behavior. Historical acceptance is preserved as historical evidence; the package
move has the separate local regression receipt below.

`learning/` is a top-level application-service module, a sibling of `app/`.
It owns learning request orchestration, effective-setting/source guards, read
projections, accepted background execution and process-local completion observation.
It does not own prompts, segmentation, cache/admission policy, repositories, SQL,
transport routes, configuration or UI. Those retain their existing module owners.

| File | Responsibility |
| --- | --- |
| `__init__.py` | Public service, response models and errors; import creates no runtime resources |
| `service.py` | LearningService use cases, command/reconcile and completion wait |
| `context.py` | Current-page projections, effective-setting/source guards and response models |
| `worker.py` | Backend-owned execution slots and durable dispatch/checkpoint/commit coordination |
| `completion.py` | Process-local observers; durable state remains authoritative |

External consumers import `LearningService`, `LearningBlockView`, `LearningPage`,
`LearningStatus`, `LearningSubmission`, `LearningModeUnavailable` and `LearningBusy`
from `learning`. Internal files use relative imports; no `app.learning*` shim or
reverse dependency on `app` is introduced. Config owns runtime creation/drain;
API schemas/main retain pure projections, transport and lifespan responsibility.
The worker logger follows its module name (`learning.worker`); safe event fields
and operation behavior remain unchanged.

Ordinary `app/shared_translation.py` remains the ordinary translation adapter and
uses the same injected LearningService/repository/worker. Existing runtime access
through service.context/service.worker remains for this adapter; this extraction
does not redesign the shared translation pipeline. A narrow public
`LearningService.record_start_failure(document_id, snapshot)` records a known
pre-dispatch executor rejection, so app no longer calls the worker's private
failure method across the new boundary. It does not retry or grant dispatch.

No schema, cache-key, source identity, language/proficiency, API or persisted-result
migration is required. Generation and storage stay in section_tasks,
document_structure and db. The cross-module product specification and task DAG
remain in section_tasks. LLRN-13 and completion-observation checkboxes have one
owner here; old app anchors are forwarding links only.

Validation must cover file/SQL orchestration, transport/DI, completion/unknown and
restart behavior, shared ordinary/learning reuse, executor rejection and backed-up
real-book excerpts with an offline provider. Import checks alone are insufficient.
External commit review and Docker deployment acceptance remain later workflow gates.

## Learning completion observation (2026-11-09)

`LearningService.wait_for_completion` subscribes to the backend worker's
process-local `LearningCompletion` before reading durable status, preventing a
lost completion between subscription and snapshot. Terminal, missing, obsolete or
noncurrent operations return immediately. Otherwise it waits for notification or
120 seconds, then reads status once more: at most two status reads, no periodic
storage polling, model call, lease update or implicit reconcile. The timeout bounds
HTTP observation, not model capacity or the accepted job's lifetime.

Worker finalization releases its execution slot and notifies all local observers.
Disconnect/cancellation removes only the observer; accepted jobs remain owned by
the executor and shutdown drains them. Durable state remains authoritative even
when a worker ends with uncertain storage outcome. Notifications do not cross
processes and are not persisted: other-process or pre-restart jobs may only be
observed at the timeout snapshot. Multi-worker deployment would need a shared
notification mechanism; this change does not claim cross-process push. Explicit
Check progress retains existing read/reconcile recovery rules without automatic
paid retries. File/PostgreSQL orchestration tests cover concurrent observers,
completion races, bounded read count, timeout/cancellation and provider-call count.

## Language Learning Reading Mode Plan (2026-10-28)

**[Code-Confirmed: LLRN-13 implemented and verified]** [Canonical design](../section_tasks/module-detailed-design.md#language-learning-reading-mode-plan-2026-10-28)；[task dependency queue](../section_tasks/module-checklist.md#language-learning-reading-mode-queue-2026-10-28)。本節不改寫既有功能完成狀態。

新增 cohesive learning use-case service，協調 reading_preferences、existing profile read、hierarchy-resolved sources、section_tasks 生成和 repository。不得把 prompt/切句/SQL 塞入 coordinator。Saved read 為純讀；生成先驗證模式/設定/source、reuse 再 claim，集中 generation policy 於 provider 前套用，commit 時再驗證。每項翻譯/提示獨立 operation；lost response 先回查、舊來源不得提交。設定保存本身不觸發模型；已接受工作不以 browser abort 當取消完成。

### LLRN-13 implementation contract (2026-10-28)

LLRN-16 supporting ports：`LearningService.read_book_preferences` 先經 repository 驗 canonical
document 存在，再交 preference service；`source_binding` 只返回 server 解出的目前 block
identity，供 transport 比對 opaque token，沒有 client source authority。`LearningModeUnavailable`
保留原 `LearningContractError` 類型／code 並帶明確 eligibility reasons；normal mode 的生成
拒絕使用 preference owner 的 `learning_mode_not_active`，不誤稱模型故障或原文同語。

`read_partial_snapshot` 在 bounded checkpoint read 前後各讀一次 status，比較 source/settings
current flags 與完整 operation snapshot；任何並行變動拒絕 `learning_operation_conflict`。
返回同一觀測區間一致的 status/batches，既有 `read_partial` 委派並保留 batch-only 返回型別。
這是有界樂觀檢查，不鎖住獨立 profile、不做 read repair／retry，也不把完整 batches 當作
final commit。LLRN-16 file/真 SQL HTTP 測試覆蓋 partial failure 與讀取中設定變動；原
81 orchestration 回歸通過。

`learning/service.py` 的 `LearningService`、`learning/context.py` 的 current-page projection／有效設定驗證、`learning/worker.py` 的 backend worker 與 `learning/completion.py` 的通知共同組成獨立 learning application module。Reads 純計算 deterministic sentence/key projection 並讀 saved/status，不初始化 provider/tokenizer、不存切句或自動 expire。每頁最多 50 blocks（兩 components 共 100 keys），屬查詢界限，不是生成文字／用量上限。Client 僅提交 block/component、source 與設定版本期待值、command intent；語言、級別、版本由 server 重解。LLRN-14 public source expectation 是 opaque token；後續 route 必須先驗 actual source/token，再將 server-resolved binding 傳入本服務，不能接受 client hierarchy/scope。

每個明確 request 先 reuse／receipt replay，再在 repository 短交易 claim。Backend 執行器持有已接受工作，HTTP 取消不取消工作；以可配置 worker concurrency/lease 控制執行，不引入用戶配額。每 batch 新 dispatch CAS commit acknowledgement 後才呼叫 provider，原集中 admission 在 claim 及每次 dispatch 重新評估；payload checkpoint、完成組裝與 commit 分開，失敗保留舊 saved／其它 component。失去 claim/dispatch/checkpoint/commit acknowledgement 先停下回查，不盲重送。

Repository 接收明確 internal validator，使用 owner 已鎖住的 workspace/book snapshot 及 existing-profile reader 重新驗 effective mode/source/proficiency，避免重入 preference locks；profile reader 不得回寫或另開 write transaction。Profile 存在於獨立儲存時仍不宣稱跨資源原子 profile snapshot／鎖；canonical source/settings CAS 與提交前重讀共同拒絕已觀測變更。LLRN-15 已以 source-only profile reader 與 readonly path resolver 提供正式 loader/DI。

Status/command lookup/read_partial 純讀；只有明確 reconcile 才更新 expired/stale 或恢復工作。Running/unknown 不重新派模型；unexpired claimed/partial 經新 dispatch CAS 可接續；完整 durable checkpoints 經 section_tasks `assemble_checkpoints` 驗證原 provenance 後零模型 assemble/commit，不依賴目前 provider/tokenizer 可用性。未完成 partial 僅在重建 key/plan 一致時接續，否則拒絕 `learning_recovery_plan_changed`；不得用新模型配置猜測舊分批。Unknown 的人工 retry 必須新 command、指向原 operation 並確認可能重複費用，沿用 domain policy。不承諾程序退出後自動執行完畢或 exactly-once billing。LLRN-14 已交付 public schema/pure projections，LLRN-15 DI 已交付，正式 API/lifespan 綁定已由 LLRN-16 完成，UI 交付狀態由 UI owner 與共同任務 DAG 維護。沒有自動掃描／接管失聯工作的 scheduler；程序重啟後由明確 reconcile 恢復安全工作。

舊 operation 因設定變更成為 stale，與 canonical source 已過時是兩種證據：前者可在目前有效來源／設定下依 domain policy 明確 retry，費用未知仍要求確認；後者拒絕生成。Page read 不將 operation stale 誤報成 source conflict。查詢回傳至多 50 blocks；目前 source resolver 仍載入一次完整 hierarchy snapshot，並非 SQL source row pagination。

驗證：`scripts/test_learning_orchestration.py` 81 項 file／真 PostgreSQL tests，以及 policy、生成、repository、偏好服務與 lifecycle 290 項回歸通過。涵蓋雙客戶端 claim／reconcile 去重、每 batch admission、provider 在鎖外、設定／profile／來源競爭、SQL 寫入後 rollback、四階段 lost acknowledgement、partial／unknown 恢復、已接受工作不受 request owner 離開影響、純讀零寫入／零 provider 與安全日誌。兩本 hash-verified 備份短段落以固定 provider 回應驗證讀取、分項保存、重建 repositories/service 後重用（各 component 一次呼叫）；證據在 ignored `data/test-books/learning-orchestration/20261028T081811Z-dcea4dbf/report.json`。未呼叫真模型、未改備份或 live Library；LLRN-10/11 語義品質未通過紀錄維持。

## Extraction validation (2026-11-10)

299 tests passed across orchestration, API schemas, container wiring, production
routes, backend integration, shared translation and ordinary translation actions.
Both file and real PostgreSQL repositories were exercised in an isolated temporary
database, removed afterward. The new ordinary-translation executor-rejection case
checks known failure persistence, zero provider calls, slot release across five
rejections and a subsequent successful retry on both adapters.

Backed-up English and Chinese excerpts retained 2/4 aligned sentences respectively;
each used one offline provider call, then learning reused the ordinary result.
Source backup hashes were unchanged. This does not claim paid-provider semantic
quality or deployment acceptance. Evidence: `/tmp/learning-module-refactor/regression.log`
and `/tmp/learning-module-refactor/real-books.log`.

The maintainer authorized one combined local checkpoint commit on 2026-11-10,
including the pre-existing workspace changes and this extraction, resolving the
baseline-scope question. External exact-SHA review, Docker rebuild/acceptance and
push remain pending; current local tests do not replace those gates.
