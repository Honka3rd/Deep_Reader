# bundle_provider.py Detailed Design

## 1. Module Purpose

`bundle_provider.py` 提供較高層 facade：載入 raw document、建立 runtime objects、呼叫 `BundleFactory` 取得可查詢 bundle。 **[Code-Confirmed]**

## 2. Position in Overall Architecture

- Peripheral / Support Layer（application glue）

## 3. Key Files

| File | Responsibility | Notes |
|---|---|---|
| `bundle_provider.py` | `BundleProvider` facade | get_bundle / get_bundle_from_raw_text **[Code-Confirmed]** |

## 4. Main Responsibilities

1. 封裝 bundle 取得流程給上層 coordinator 使用。 **[Code-Confirmed]**
2. 透過 loader factory 載入 raw text。 **[Code-Confirmed]**
3. 在需要時觸發 force_rebuild invalidate。 **[Code-Confirmed]**

## 5. Non-Responsibilities

1. 不負責索引實際建置細節。 **[Code-Confirmed]**
2. 不負責 profile schema。 **[Code-Confirmed]**

## 6. Important Data Structures / Contracts

- `BundleProvider`
- `FaissIndexBundle`
- `FaissStorageConfig`

## 7. Module Relationships

- depends on: `bundle_factory.py`, `doc_loaders/`, `config/faiss_storage_config.py`
- used by: `app/qa_coordinator.py`

## 8. Main Flows Involving This Module

1. get_bundle(doc_name) flow（含 raw load）。 **[Code-Confirmed]**
2. get_bundle_from_raw_text flow（prepare 或上層已提供 raw）。 **[Code-Confirmed]**

## 9. Persistence / Side Effects

- persistence：透過 bundle factory 間接讀寫
- side effect：force_rebuild 時 invalidate cache

## 10. Known Legacy / Compatibility Behavior

No explicit legacy compatibility responsibility（實際兼容在 bundle factory/retrieval 層）。 **[Code-Confirmed]**

## 11. Current Risks

1. risk：facade 層錯誤處理不足
- why：上層只看到 RuntimeError
- guardrail：保留明確錯誤訊息與 logs

## 12. Open Questions for Maintainer

1. 是否需要在 provider 層暴露更細緻 diagnostics？ **[Needs Confirmation]**

## 13. Suggested Next Documentation Improvements

1. 補 provider/factory boundary 圖。
