# bundle_factory.py Detailed Design

## Reuse-Only Profile Policy (2026-10-21)

CONTENT-03 supersedes the former missing/corrupt-profile rebuild fallback.
ensure_profile_ready now only loads the persisted profile or raises actionable
profile_not_ready/profile_unreadable. Index clearing/rebuilding cannot clear,
generate, save or enrich profiles, even with supplied raw text. Initial preparation
and explicit full reparse belong to document_preparation, not runtime bundles.

## 1. Module Purpose

`bundle_factory.py` 負責 FAISS runtime bundle 的建立、快取、重建與 profile/index artifact readiness 保證。 **[Code-Confirmed]**

## 2. Position in Overall Architecture

- Peripheral / Support Layer（retrieval bundle orchestration）

## 3. Key Files

| File | Responsibility | Notes |
|---|---|---|
| `bundle_factory.py` | `BundleFactory` | cache + ensure_profile_ready + ensure_index_ready **[Code-Confirmed]** |

## 4. Main Responsibilities

1. 管理 doc-scoped bundle cache（LRU-like OrderedDict）。 **[Code-Confirmed]**
2. 依 fingerprint + records schema 判斷重用/重建。 **[Code-Confirmed]**
3. Load the existing profile or report readiness failure; never rebuild it. **[Code-Confirmed]**
4. Persist index and fingerprint metadata only. **[Code-Confirmed]**

## 5. Non-Responsibilities

1. 不負責 API route 協調。 **[Code-Confirmed]**
2. 不負責 structured hierarchy parser。 **[Code-Confirmed]**

## 6. Important Data Structures / Contracts

- `BundleFactory`
- `FaissIndexBundle`
- `FaissStorageConfig`
- `DocumentProfile`

## 7. Module Relationships

- depends on: `retrieval/`, `profile/`, `fingerprint_handler.py`, `config/`
- used by: `bundle_provider.py`, `app/qa_coordinator.py`

## 8. Main Flows Involving This Module

1. ensure index/profile ready flow。 **[Code-Confirmed]**
2. cache load/reuse flow。 **[Code-Confirmed]**
3. legacy records schema detection -> rebuild flow。 **[Code-Confirmed]**

## 9. Persistence / Side Effects

- read/write persistence: index/records/meta; profile is read-only.
- side effects：cache mutation、clear/remove artifacts

## 10. Known Legacy / Compatibility Behavior

1. 發現 legacy records schema（缺 position metadata）時自動 rebuild。 **[Code-Confirmed]**

## 11. Current Risks

1. risk：cache 命中與 artifact 狀態漂移
- why：可能讀到 stale bundle
- guardrail：fingerprint + ensure_index_ready gate

2. risk: missing/corrupt profile prevents a complete runtime bundle.
- why: downstream prompts require stable document metadata.
- guardrail: explicit readiness error; only initial preparation/full reparse owns recovery.

## 12. Open Questions for Maintainer

1. bundle cache eviction 是否需要更多可觀測 metrics？ **[Needs Confirmation]**

## 13. Suggested Next Documentation Improvements

1. 補 bundle readiness sequence diagram。
