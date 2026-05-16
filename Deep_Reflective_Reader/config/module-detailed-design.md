# config Detailed Design

## 1. Module Purpose

`config/` 負責應用層設定契約與依賴注入組裝，並提供 storage namespace normalization 與 token budget 解析策略。 **[Code-Confirmed]**

## 2. Position in Overall Architecture

- Configuration / DI Layer

## 3. Key Files

| File | Responsibility | Notes |
|---|---|---|
| `config/app_DI_config.py` | `AppDIConfig` 與多組 policy dataclass | 集中 app-level policy 值 **[Code-Confirmed]** |
| `config/container.py` | `ApplicationLookupContainer` 依賴注入組裝 | services/repositories/providers wiring **[Code-Confirmed]** |
| `config/token_budget_resolver.py` | capability-aware token budget 計算 | 有 fallback_used 訊號 **[Code-Confirmed]** |
| `config/faiss_storage_config.py` | FAISS artifact path + namespace migration | 含 legacy folder migration **[Code-Confirmed]** |
| `config/structured_document_storage_config.py` | structured JSON path + namespace migration | 含 legacy filename migration **[Code-Confirmed]** |
| `config/storage_namespace_helper.py` | namespace normalization 共用 helper | 去除 `.pdf/.txt` extension **[Code-Confirmed]** |

## 4. Main Responsibilities

1. 提供統一 app policy（retrieval/profile/scope/task-unit/recommendation）。 **[Code-Confirmed]**
2. 組裝跨模組依賴圖。 **[Code-Confirmed]**
3. 提供 token budget capability clamp/fallback。 **[Code-Confirmed]**
4. 提供 artifact namespace/path normalization 與 legacy 檔名遷移。 **[Code-Confirmed]**

## 5. Non-Responsibilities

1. 不實作業務流程（prepare、task-layout、artifact write）。 **[Code-Confirmed]**
2. 不定義 API request/response schema。 **[Code-Confirmed]**
3. 不應承擔 parser 決策 authority。 **[From HLD]**

## 6. Important Data Structures / Contracts

- `AppDIConfig`
- `RetrievalTokenBudgetConfig`
- `PromptTextNormalizationConfig`
- `ProfilePromptPolicyConfig`
- `QuestionScopePolicyConfig`
- `TaskUnitSplitPolicyConfig`
- `EnhancedParsePolicyConfig`
- `EffectiveTokenBudgets`
- `FaissStorageConfig`
- `StructuredDocumentStorageConfig`

## 7. Module Relationships

- used by: almost all runtime modules via `container.py`
- depends on: `llm/`, `document_structure/`, `profile/`, `section_tasks/`, `retrieval/`, `question/`, `context/`, `auth/`
- writes/reads: 僅設定與 path 解析，非業務資料

## 8. Main Flows Involving This Module

1. 啟動時 DI assembly flow。 **[Code-Confirmed]**
2. prepare/ask/task path 注入 policy。 **[Code-Confirmed]**
3. storage namespace normalization 與 legacy migration。 **[Code-Confirmed]**

## 9. Persistence / Side Effects

- read persistence：間接（storage path 檢查）
- write persistence：否
- mutate artifact：否
- side effect：legacy namespace rename（檔案系統）。 **[Code-Confirmed]**

## 10. Known Legacy / Compatibility Behavior

1. `FaissStorageConfig` 會把舊 namespace 目錄遷移到 normalized namespace。 **[Code-Confirmed]**
2. `StructuredDocumentStorageConfig` 會把舊 structured filename 遷移到 normalized namespace。 **[Code-Confirmed]**

## 11. Current Risks

1. risk：`AppDIConfig` 欄位數量持續膨脹
- why：維護理解成本高
- guardrail：分組 dataclass 與文件化 policy owner

2. risk：DI container 過度集中
- why：改一處可能波及多模組
- guardrail：維持 module-level docs + smoke tests

3. risk：storage migration side effect 未被觀測
- why：部署時可能與既有檔案衝突
- guardrail：保留 migration log 並補文件

## 12. Open Questions for Maintainer

1. `AppDIConfig` 是否要進一步分檔（僅文件與組織，不改行為）？ **[Needs Confirmation]**
2. storage migration 是否需要版本標記或一次性工具？ **[Needs Confirmation]**

## 13. Suggested Next Documentation Improvements

1. 補 container dependency graph（高層）。
2. 補 policy groups 對應的 owner/調參策略。
