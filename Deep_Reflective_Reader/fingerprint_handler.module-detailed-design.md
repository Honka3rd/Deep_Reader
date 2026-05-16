# fingerprint_handler.py Detailed Design

## 1. Module Purpose

`fingerprint_handler.py` 提供內容與索引配置（embedding model/chunk params）的 fingerprint 計算與比對，用於 artifact reuse 決策。 **[Code-Confirmed]**

## 2. Position in Overall Architecture

- Peripheral / Support Layer（cache/reuse guard）

## 3. Key Files

| File | Responsibility | Notes |
|---|---|---|
| `fingerprint_handler.py` | `FingerprintHandler` | build/save/load/matches/clear **[Code-Confirmed]** |

## 4. Main Responsibilities

1. 生成 text+config fingerprint payload。 **[Code-Confirmed]**
2. 儲存/載入 fingerprint metadata JSON。 **[Code-Confirmed]**
3. 比對是否可重用既有索引。 **[Code-Confirmed]**

## 5. Non-Responsibilities

1. 不負責索引建置。 **[Code-Confirmed]**
2. 不負責 profile build。 **[Code-Confirmed]**

## 6. Important Data Structures / Contracts

- `FingerprintHandler`
- fingerprint payload: `content_hash`, `embedding_model`, `chunk_size`, `chunk_overlap`

## 7. Module Relationships

- used by: `bundle_factory.py`
- depends on: filesystem + hashlib/json

## 8. Main Flows Involving This Module

1. ensure_index_ready 前比對 fingerprint。 **[Code-Confirmed]**
2. rebuild 完成後保存 fingerprint。 **[Code-Confirmed]**

## 9. Persistence / Side Effects

- read/write persistence：是（meta.json）
- side effect：clear 時刪除 meta file

## 10. Known Legacy / Compatibility Behavior

No known legacy compatibility responsibility.

## 11. Current Risks

1. risk：fingerprint 維度不足
- why：若新增關鍵索引參數未納入，可能誤重用
- guardrail：變更索引行為時同步更新 payload contract

## 12. Open Questions for Maintainer

1. fingerprint payload 是否要加入 version 字段？ **[Needs Confirmation]**

## 13. Suggested Next Documentation Improvements

1. 補 fingerprint invalidation contract（命名與 reason code）對應文檔。 **[From Proposal] + [From HLD]**
