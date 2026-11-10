# config Detailed Design

## Shared sentence translation (2026-11-09)

Production TranslationAction receives a lazy factory for the same learning_service
resource used by learning routes. It uses the existing single-attempt provider,
model capabilities, repositories, admission and worker pool. The factory avoids an
import-time runtime initialization and resolves a fresh worker after resource
shutdown/reinitialization. TranslationService remains an explicit legacy adapter
for historical callers/tests; production has no fallback to it when shared storage
or generation is unavailable. No additional provider, language config or migration.

Validation: 470 backend tests and actual API/React mode-switch acceptance pass; detailed evidence and limits are recorded in the shared section_tasks design.


## Language Learning Reading Mode Plan (2026-10-28)

**[Code-Confirmed: LLRN-15 implemented]** [Canonical design](../section_tasks/module-detailed-design.md#language-learning-reading-mode-plan-2026-10-28)；[task dependency queue](../section_tasks/module-checklist.md#language-learning-reading-mode-queue-2026-10-28)。Production routes/lifespan 綁定已由 LLRN-16 完成，UI 與完整模式驗收尚未交付。

只組裝 language definitions、偏好/學習 repositories、services、learning application module 和既有 provider/capability。母語/每書能力屬持久化偏好，不放 env、AppDIConfig 常量或另一份語言白名單；source/prompt policy 不在 DI 重新實作。支援既有 file/SQL 選擇但缺契約明確 unavailable，無 silent fallback、import-time DDL/LLM 或讀取時初始化。

### LLRN-15 composition and lifecycle

`learning_storage.py` 由既有 `structured_storage_backend` 選 file/PostgreSQL。Container 的 raw document selector 只供組裝；公開 `structured_document_artifact_repository`、`reading_preferences_repository`、`learning_repository` 都來自同一 singleton aggregate。File 以實際 document repository namespace 建立 `WorkspaceScope`，在公開給一般讀寫 consumers 前安裝 `FileLearningDocumentLifecycle`，rename/delete 因此使用同一偏好及 learning sidecar；不改原有 structured file 路徑。SQL 以實際 store namespace/DSN 校驗組裝，將同一 store/preferences 注入 `PostgresLearningRepository`，不依賴 SQL artifact adapter 繼承的 file namespace。Constructor 不查 DB、不建表；004/005 等 migrations 必須由部署顯式執行，缺表 read 明確失敗，不轉 missing 或 fallback file。

`learning_runtime.py` 從 `learning` 公開入口匯入 `LearningService`，以 `providers.Resource` 擁有 `ReadingPreferencesService`、兩個生成 service 與 `LearningService`。`learning_service` / `reading_preferences_service` 為同一 resource 的公開入口；公開 document repository 若被 override 成另一實例，learning 組裝拒絕不一致。`learning_admission` 是可注入的單一 policy，預設 `AllowValidGeneration`，claim 與每次 dispatch 共用；不冒充已存在的用戶／配額系統。

`LearningRuntimeConfig` 只集中 operational resources：preferences `data/reading-preferences`、results `data/learning`、4 workers、600,000 ms lease，可由 AppDIConfig 注入；不新增母語、能力、語言白名單或模型 token 常量。啟動／一般 reads 不初始化 provider、tokenizer、sidecar 或 recovery。呼叫 `container.shutdown_resources()` 會等待已接受工作完成；reinitialize 會建新 worker、保留同一 storage/lifecycle。LLRN-16 已在 application lifespan 啟動時發布唯一 runtime、關閉時接入該 resource shutdown，不能跟 browser request cancellation 綁定。程序崩潰後仍需明確 reconcile；無隱式接管 scheduler。File journal 恢復入口為 `container.learning_storage().lifecycle.recover()`，只可由明確維護操作使用，pending journal 仍 fail closed。

Profile reader 綁定 server scope/exact document ID，透過 `FaissStorageConfig(..., read_only=True)` 與 `VersionedDocumentProfileStore.load_source` 讀既有 source language；不重入 document/preferences locks、不查 structure invalidation 或生成／回寫 profile。缺檔、壞檔、不可用保持不同原因；拒絕文件 alias。Profile 仍是原有 `data/faiss/<doc>/profile.json` 獨立檔案，未新增 workspace 分區或跨資源原子性保證。

`LearningProvider` 延後取得現有 `llm_provider` singleton；能力及 offline tokenizer 沿用同一 provider，輸出 allowance 來自既有 model capability。只接受公開 `complete_text_once_with_output_budget` 契約，沒有該契約／key／能力時 preflight 回報 domain capacity_unavailable，不寫 claim。OpenAI 實作在既有 provider 內使用 request-local SDK client、`max_retries=0`，沿用其 model/API base/key/temperature，完成或失敗皆 close；不改共享 ordinary interaction adapter 的 retry/default allowance。429/5xx/timeout 不偷偷重送付費請求，unknown 仍經既有 durable recovery。沒有新 provider 設定來源，也沒有改 prompt 或語義品質判定。

Supporting fix：file rename/delete 已取得 preference lock 並留下 tombstone 時，遲到 learning writer 先拒絕 `learning_document_missing`，避免再開啟已移除原文件而漏出 FileNotFoundError；SQL 原有拒絕行為保持一致。

驗證入口：`scripts/test_learning_container_wiring.py`、`scripts/test_learning_provider_once.py`。涵蓋 file/真 SQL 組裝、同一 instance/scope、缺 migration/key/capability/一次請求契約、零 import/startup network/files、純讀、profile 無交易重入、journal fail closed、admission、shutdown drain/reinit、生成保存與 offline reuse、rename/delete late writer。真 SDK 使用 HTTP mock transport 驗證兩類 endpoint 的成功、429/500/503/timeout 都只送一次；不當作真模型語義驗收。既有 storage/lifecycle/orchestration、preferences、schemas、ordinary DI/capacity 回歸證據見 owner checklist；未部署或修改使用者 DB/書籍。

## Block Translation Wiring (2026-10-26)

TRANS-API-02 injects the existing llm_provider into TranslationService and wires it
with the selected structured repository and profile store into TranslationAction.
Main exposes the action through app.state; no model settings/new provider/private
key handling or import-time completion. FaissStorageConfig adds opt-in read_only
path resolution for this action, suppressing mkdir and legacy directory migration;
existing callers retain their defaults. No automatic migration on translation reads.

## Smart Cleanup Wiring (2026-10-24)

SCLEAN-07 supporting composition: the configured llm_provider is injected directly
as DocumentPreparationPipeline.cleanup_provider and into the singleton standalone
service factory. Coordinator supplies coherent accepted-input, shared enhancement
lease and explicit readiness ports; main installs the resulting service in app.state.
No splitter-private provider discovery, new model settings or import-time request.
Backends lacking coherent enhancement snapshot/lease ports remain unavailable; no
unlocked fallback. Container/default-route and isolated SQL tests are counted once
in preparation SCLEAN-07, not a second config task.

## Summary Service Wiring (2026-10-21)

SN-10 supporting composition: SummaryInteractionService receives the configured
structured repository and current LLM provider, and is injected into the task
coordinator. Notes require only that existing repository and no provider. No
private key, model selection or capacity configuration changes were introduced.

## FULL-01 Capacity Consumer Migration (2026-10-20)

Token budget resolver uses combined_token_limit because QA's TokenBudgetManager
subtracts output downstream. Existing target caps/utilization and no-capability
fallback remain unchanged. EffectiveTokenBudgets now reports context_window_tokens
separately from optional input-only capacity; no model/DI setting changed.
Completion/testing evidence belongs to llm FULL-01, not a new config task.

## Enhancement Profile Wiring (2026-10-19)

Supporting PREP-03: the shared profile-store provider uses VersionedDocumentProfileStore
with the selected structured repository injected. All existing runtime consumers get
this adapter; PostgreSQL invalidates file-derived structure advice through committed
provenance, while file storage keeps existing behavior. Config only wires dependencies,
not invalidation semantics. Container wiring and real API/DB acceptance pass.

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

## 14. Future Direction Note: Storage Backend Governance

> 本節是 future DB migration governance preparation，非目前 implementation。 **[Future Direction]**

1. Future DB migration should be configured through the `config/` layer. **[Future Direction]**
2. Storage backend selection belongs in `config/`, including eventual file / DB coexistence policy and DI wiring choice. **[Future Direction]**
3. Persistence semantics do not belong in `config/`; hierarchy semantics remain owned by `document_structure/`, profile semantics by `profile/`, retrieval semantics by `retrieval/`, and raw loading semantics by `doc_loaders/`. **[Code-Confirmed] + [Inferred]**
4. `config/` may own backend selection policy, namespace normalization policy, and provider assembly, but must not define schema authority, parser authority, artifact authority, or runtime projection semantics. **[Future Direction]**
5. Current file path behavior remains active; this note does not introduce storage abstractions, DB dependencies, schema design, dual write, or runtime switch behavior. **[Doc-Confirmed]**

### 14.1 Future Storage Backend Governance Boundary

`config/` should eventually own the configuration policy for:
- file backend enablement
- DB backend enablement
- migration rollout switches

`config/` does not own:
- hierarchy contracts
- document semantics
- parser semantics

Backend governance may select and route storage implementations, but it must not redefine the domain meaning of structured documents, profiles, retrieval artifacts, or raw document inputs. **[Maintainer-Provided] + [Future Direction]**

## 15. Future Direction Note: Reading Interaction Configuration

> 本節記錄 reading interaction policy configuration；不代表目前 implementation。 **[Maintainer-Provided] + [Future Direction]**

1. Config should own target-level quiz default maximums: `task_unit=3`, `section=5`, `chapter=10`, `document/book=25`. **[Maintainer-Provided] + [Future Direction]**
2. Config should expose valid quiz type vocabulary only as policy input to services/prompts; schema/service validation remains the enforcement boundary. **[Future Direction]**
3. Config should expose interaction token/cost policy hooks for generation, refresh, critical-thinking evaluation, and retry. **[Maintainer-Provided] + [Future Direction]**
4. Config should not define prompt wording, artifact payload schema, hierarchy semantics, or route behavior. **[From HLD] + [Future Direction]**
5. Config should support future permission/quota integration without gating ordinary persisted artifact reads. **[Maintainer-Provided] + [Future Direction]**
6. Current code wires the analysis/insight vertical slice through DI: `AnalysisInteractionLLMGenerator -> AnalysisInteractionService -> DocumentReadingInteractionArtifactStore -> AnalysisInteractionOrchestrator -> SectionTaskCoordinator`. This is dependency assembly only; `config/` does not own the analysis prompt contract, target resolution semantics, route mapping, or artifact storage semantics. **[Code-Confirmed]**
7. Current code wires the generic quiz drawer API through DI: `QuizInteractionLLMGenerator -> QuizInteractionService -> DocumentReadingInteractionArtifactStore -> QuizInteractionOrchestrator -> SectionTaskCoordinator`. This makes the exposed `/documents/reading-interactions/quiz/*` route family runtime-usable without using the legacy section/chapter quiz routes. **[Code-Confirmed]**
8. Current code wires the critical-thinking session API through DI: `CriticalThinkingLLMGenerator -> CriticalThinkingSessionService -> DocumentReadingInteractionArtifactStore -> SectionTaskCoordinator`. The store persists session ids and session-shaped artifacts through the shared reading-interaction metadata path. **[Code-Confirmed]**
