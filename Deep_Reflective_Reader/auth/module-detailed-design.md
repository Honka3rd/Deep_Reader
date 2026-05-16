# auth Detailed Design

## 1. Module Purpose

`auth/` 提供 API 金鑰取得的抽象與 OpenAI 環境變數實作，讓上層模組不直接綁定金鑰來源。 **[Code-Confirmed]**

## 2. Position in Overall Architecture

- Peripheral / Support Layer

## 3. Key Files

| File | Responsibility | Notes |
|---|---|---|
| `auth/api_key_provider.py` | 定義 `APIKeyProvider` 抽象介面 | `get()` contract **[Code-Confirmed]** |
| `auth/openai_api_key_provider.py` | 從 `OPENAI_API_KEY` 載入金鑰 | 啟動時即驗證缺失會 raise **[Code-Confirmed]** |

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

## 7. Module Relationships

- used by: `llm/openai_llm_provider.py`, `embeddings/openai_embedder.py`
- assembled by: `config/container.py`
- advisory relationship: 無

## 8. Main Flows Involving This Module

1. DI container 建立 `api_key_provider` singleton。 **[Code-Confirmed]**
2. LLM/Embedder 初始化時讀取金鑰。 **[Code-Confirmed]**

## 9. Persistence / Side Effects

- read persistence：否
- write persistence：否
- call external service：否
- side effect：讀取環境變數，缺失時 fail-fast。 **[Code-Confirmed]**

## 10. Known Legacy / Compatibility Behavior

No known legacy compatibility responsibility.

## 11. Current Risks

1. risk：env key 缺失導致啟動失敗
- why：整體服務不可用
- guardrail：部署前健康檢查與環境設定檢查

2. risk：provider 單一來源
- why：未來多供應商擴展會受限
- guardrail：維持 `APIKeyProvider` 抽象

## 12. Open Questions for Maintainer

1. 是否需要支援多 key/source（例如 vault）？ **[Needs Confirmation]**
2. 是否需要 key refresh lifecycle（目前是啟動時讀一次）？ **[Needs Confirmation]**

## 13. Suggested Next Documentation Improvements

1. 補 deployment secrets management 邊界（若未來導入）。
