# learning Checklist

## Learning audit (2026-11-15)

- [x] LEARN-01: Guard partial recovery provenance and provide explicit recovery for unassemblable complete checkpoints.
  Status: Locally verified; external learning review pending.
  Acceptance: Changed generation identity performs zero recovery dispatch; valid complete checkpoints finish offline; explicit retry preserves history and old saved; real file/PostgreSQL controls.
  Evidence: 22/22 new recovery tests, 370/370 adjacent executions, 56/56 routes, 8/8 external guards; corrected external matrix 4/4, two immutable-book excerpts/8 flows. `/tmp/learn01-fix-471db7b/report.txt`.
  Timestamp: 2026-11-15

- [x] LEARN-02: Reconcile late successful replies after expiry observation without replaying provider calls.
  Status: Locally verified; exact-SHA external review pending.
  Acceptance: Same operation/fence reply uses guarded CAS; source/settings/new fence/unknown-ack protections retained.
  Evidence: Expiry regression 26/26; cumulative 474/474; external expiry 4/4 and guards 8/8; two hash-verified book samples/8 file-SQL flows. `/tmp/learn02-fix-471db7b/report.txt`.
  Timestamp: 2026-11-16

## Module extraction (2026-11-10)

- [x] LEARN-MOD-01: Extract the learning application module from app.
  Owner: learning. Status: Implemented and locally verified; delivery review remains pending.
  Acceptance: Four implementation files move to a sibling package with a public import surface; consumers and single-owner documentation migrate; no stale imports or duplicated ownership; ordinary/learning reuse, completion, errors and file/SQL behavior remain compatible.
  Evidence: 299 file/real PostgreSQL/orchestration/DI/API/shared-translation tests passed; two hash-verified English/Chinese book excerpts saved and reused aligned results with one offline call each. Import/ownership/link checks and task diff are recorded in `/tmp/learning-module-refactor/`. External commit review, Docker acceptance and push remain pending; no deployment claim.
  Timestamp: 2026-11-10

Historical completed items below were transferred from app on 2026-11-10; their
original test evidence is retained, not represented as new extraction validation.

## Vocabulary filtering and completion observation (2026-11-09)

- [x] Observe learning completion without periodic storage polling.
  Owner: learning. Status: Implemented and verified.
  Acceptance: Subscribe-before-read and one bounded 120-second notification wait, at most two status reads. Cancellation detaches observer only; no implicit recovery or provider dispatch. Process-local limitation documented.
  Evidence: File/SQL orchestration: concurrent observers, lost-wakeup race, timeout/cancel, listener cleanup, terminal/missing state and one provider call.
  Timestamp: 2026-11-10

## Language Learning Reading Mode Queue (2026-10-28)

**Implemented and verified: 1/1 owned task (LLRN-13).** [Shared dependency queue](../section_tasks/module-checklist.md#language-learning-reading-mode-queue-2026-10-28)；[canonical design](../section_tasks/module-detailed-design.md#language-learning-reading-mode-plan-2026-10-28)。本檔持有 learning 的唯一 checklist；LLRN-14 transport 與 LLRN-15 DI 已交付，LLRN-16 production API/lifespan 已交付，UI 狀態由 UI owner 與共同任務 DAG 維護。

### LLRN-13

- [x] LLRN-13: Learning orchestration and admission integration.
  Owner: learning. Status: Implemented and verified (application services; DI delivered by LLRN-15, API/lifespan delivered by LLRN-16).
  Depends on: LLRN-08, LLRN-09, LLRN-10, LLRN-11, LLRN-12.
  Deliverable: 組合設定解析、當前 block batch read、get-or-generate、分項 refresh/status；每次付費呼叫前經集中 policy 與 durable claim。
  Acceptance: 讀取零生成/補寫；先 reuse，再 claim，provider 在交易外；讀失敗非 missing。已接受工作由 backend 擁有，設定/source 衝突拒絕晚到提交，unknown 先 reconcile。
  Evidence: `service.py`、`context.py`、`worker.py` 及 supporting repository/profile/domain ports；`scripts/test_learning_orchestration.py` 81 項 file/真 PostgreSQL tests + 290 項既有回歸通過。雙客戶端 claim/reconcile、每 batch admission、source/settings/profile race、partial/unknown、claim/dispatch/checkpoint/commit ack loss、SQL rollback、offline 完成組裝、明確 stale retry 與零生成/補寫 reads 均驗證。兩本 immutable 備份短段落以固定 provider 驗證保存及重建後重用，備份不變；此項不替代 live-provider 語義品質或 API/UI acceptance。
  Timestamp: 2026-11-10
