# reading_preferences Checklist

## Language Learning Reading Mode Queue (2026-10-28)

**Foundation and service implemented: LLRN-02/09 complete; runtime DI delivered by LLRN-15, API/lifespan delivered by LLRN-16, UI pending.** [Shared dependency queue](../section_tasks/module-checklist.md#language-learning-reading-mode-queue-2026-10-28)；[canonical design](../section_tasks/module-detailed-design.md#language-learning-reading-mode-plan-2026-10-28)。本檔只持有 reading_preferences 的唯一 checklist；基礎契約完成不代表 API/UI 或完整模式已可用。

### LLRN-02

- [x] LLRN-02: Workspace and book preference contracts.
  Owner: reading_preferences. Status: Implemented and verified (domain/port/file foundation only).
  Depends on: LLRN-01.
  Deliverable: 定義集中 workspace 母語、book mode/source override/proficiency 與 revision、repository port、server-resolved scope；交付 file adapter 及明確未來 user migration 接點。 同時定義穩定 mode catalog、單一 active mode 與各模式 typed/versioned settings，交付 file round-trip。
  Acceptance: 缺省為普通閱讀；未設母語、未知原文或缺分級時不得偷偷啟用。無假 user_id、瀏覽器持久化或讀取時補寫設定。Book identity 跟隨既有 namespace/document identity。 不以布林開關定義模式；保留非 active 模式設定，普通模式不要求語言學習欄位。
  Evidence: `models.py`、`repository.py`、`serialization.py`、`file_repository.py`；`scripts/test_reading_preferences.py` 28 tests passed，涵蓋 reload、唯讀預設、typed/registry validation、未知設定保留、原子失敗、CAS/idempotency、scope 隔離、rename/delete tombstone、實際多程序寫入及 replace 前後程序中斷。另通過 language proficiency/script/discourse 回歸。Rename/delete 僅 preference primitive，實際文件 lifecycle、profile eligibility、SQL/API/DI 均未接線。
  Timestamp: 2026-10-28

### LLRN-09

- [x] LLRN-09: Validated reading preferences service.
  Owner: reading_preferences. Status: Implemented and verified (service foundation; runtime DI delivered by LLRN-15, API/lifespan delivered by LLRN-16, UI pending).
  Depends on: LLRN-01, LLRN-02, LLRN-07.
  Deliverable: 實作 settings read/save 與有效設定解析：book override 優先，否則讀既有 profile；母語來自 workspace；返回 eligibility 與可理解原因。
  Acceptance: 學習語言必須為有效原文且異於母語；非法 registry/scheme/level 拒絕。改原文後不沿用其它語言 level；母語變更使同語書暫停學習投影，不批量改書或刪除 cache。 Backend 回傳 mode eligibility；active mode 與該模式設定一次 CAS 保存，未知 mode/錯配型別拒絕；讀未知 persisted mode 不補寫預設。
  Evidence: `service.py`、`effective.py`、`profile_source.py`；`scripts/test_reading_preferences_service.py` 64 tests passed（33 file、31 real PostgreSQL），另 82 項既有 file/SQL/proficiency 回歸通過。涵蓋中文母語/中文書禁用、英文書可用、19 registry 語言、缺/錯/失敗 profile 的明確校正、未知舊設定保留、模式往返、普通模式獨立驗證、read-only catalog、設定並行與原子拒絕、重複命令、unknown commit 對帳。既有 profile JSON bytes 不變、提交後無 IO、無 LLM/prepare/profile mutation。Profile/document 與生成提交協調仍待 LLRN-12/13/15；未接 API/UI，未更新使用者 DB。
  Timestamp: 2026-10-28
