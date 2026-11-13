# auth Detailed Design

## 1. Module Purpose

`auth/` 提供 API 金鑰取得的抽象與 OpenAI 環境變數實作，讓上層模組不直接綁定金鑰來源。 **[Code-Confirmed]**

## 2. Position in Overall Architecture

- Peripheral / Support Layer

## 3. Key Files

| File | Responsibility | Notes |
|---|---|---|
| `auth/api_key_provider.py` | 定義 `APIKeyProvider` 抽象介面 | `get()` contract **[Code-Confirmed]** |
| `auth/openai_api_key_provider.py` | 從 `OPENAI_API_KEY` 或指定 `env_var` 載入金鑰 | provider 初始化時驗證；不是容器啟動檢查 **[Code-Confirmed]** |

## 4. Main Responsibilities

1. 提供統一 API key provider contract。 **[Code-Confirmed]**
2. 提供 OpenAI key 的環境變數讀取實作。 **[Code-Confirmed]**
3. 讓 `llm/` 與 `embeddings/` 透過 DI 注入金鑰，不直接讀 env。 **[Code-Confirmed]**

## 5. Non-Responsibilities

1. 不負責模型選擇與 token policy。 **[Code-Confirmed]**
2. 不負責 key rotation/secret manager 整合。 **[Needs Confirmation]**
3. 不負責 API rate-limit/retry。 **[Code-Confirmed]**

## 6. Important Data Structures / Contracts

- `APIKeyProvider`
- `OpenAIAPIKeyProvider`

### AUTH-01：環境值與初始化邊界

**[Code-Confirmed]** 先對選定環境變數的值執行 `strip()`，再拒絕缺失、空字串及全空白值，拋出 `RuntimeError`。預設與自訂 `env_var` 適用相同規則；自訂來源不得回退到 `OPENAI_API_KEY`。合法值只修剪前後空白，不新增金鑰格式或遠端有效性檢查。

成功初始化後保存當時的金鑰快照，`get()` 不重新讀取環境；之後修改或移除環境變數不影響既有 instance。錯誤只指出環境變數名稱，不包含金鑰值。測試一律使用合成金鑰。

「provider 初始化」與「應用程式啟動」不是同一事件：`ApplicationLookupContainer.build()` 與 lazy learning resource 的 `init_resources()` 不應解析 key provider；無金鑰時仍可啟動這些資源並讀取既存資料。只有實際解析依賴金鑰的 provider 時才驗證。此保證不等同完整 FastAPI 入口無金鑰啟動：目前 `main.py` 建立的 `QACoordinator` 會立即解析 LLM provider。AUTH-01 不改動該既有 eager 接線，也不新增啟動期金鑰檢查。

驗證：`scripts/test_auth_api_key_provider.py` 的 7 項測試涵蓋預設／自訂來源、ASCII／Unicode 空白、來源隔離、錯誤內容及 instance 快照；獨立程序禁止網路，驗證缺失／空白金鑰時的 lazy 初始化、真實暫存檔案正文／偏好／學習狀態讀取及讀前後檔案不變（profile reader 使用固定語言 stub）。合併 `test_learning_container_wiring.WiringBoundaryTests`、`FileCompositionTests` 與 `test_learning_provider_once` 共 31 項通過、零跳過；沒有呼叫真模型或驗證完整 HTTP 入口無金鑰啟動。

## 7. Module Relationships

- used by: `llm/openai_llm_provider.py`, `embeddings/openai_embedder.py`
- assembled by: `config/container.py`
- advisory relationship: 無

## 8. Main Flows Involving This Module

1. DI container 註冊 lazy `api_key_provider` singleton，尚不讀取金鑰。 **[Code-Confirmed]**
2. 首次解析該 provider 時讀取並驗證環境值；LLM/Embedder 使用其 `get()` 快照。 **[Code-Confirmed]**

## 9. Persistence / Side Effects

- read persistence：否
- write persistence：否
- call external service：否
- side effect：provider 初始化時讀取環境變數，無可用值時 fail-fast；不寫入資料或連線遠端。

## 10. Known Legacy / Compatibility Behavior

No known legacy compatibility responsibility.

## 11. Current Risks

1. risk：缺少可用 key 時，實際解析依賴它的 provider 失敗
- why：該生成／嵌入操作不可用；若應用入口 eager 解析 provider，也會影響該入口啟動
- guardrail：保留 lazy DI 的無金鑰初始化與讀取能力；於 provider 初始化邊界驗證，不將驗證移到容器建構或純讀取路徑

2. risk：provider 單一來源
- why：未來多供應商擴展會受限
- guardrail：維持 `APIKeyProvider` 抽象

## 12. Open Questions for Maintainer

1. 是否需要支援多 key/source（例如 vault）？ **[Needs Confirmation]**
2. 是否需要 key refresh lifecycle（目前是每個 provider instance 初始化時讀一次）？ **[Needs Confirmation]**

## 13. Suggested Next Documentation Improvements

1. 補 deployment secrets management 邊界（若未來導入）。
