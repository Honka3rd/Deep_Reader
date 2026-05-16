# retrieval Detailed Design

## 1. Module Purpose

`retrieval/` 管理節點解析後的向量索引建置、持久化、重載與檢索，形成 `FaissIndexBundle` 供 QA 與 context 模組使用。 **[Code-Confirmed]**

## 2. Position in Overall Architecture

- Document Preparation Layer + Runtime Retrieval Core

## 3. Key Files

| File | Responsibility | Notes |
|---|---|---|
| `retrieval/node_provider.py` | 原文切 chunk + 位置 metadata 附加 | 產出 `ParsedDocument` **[Code-Confirmed]** |
| `retrieval/parsed_document.py` | parsed document DTO | nodes + language **[Code-Confirmed]** |
| `retrieval/faiss_index_builder.py` | 從 parsed nodes 建 FAISS bundle | embedding batch + token budget resolve **[Code-Confirmed]** |
| `retrieval/faiss_index_store.py` | FAISS index/records save/load/clear | records schema 檢查 **[Code-Confirmed]** |
| `retrieval/faiss_index_bundle.py` | runtime query bundle | search + profile attachment **[Code-Confirmed]** |
| `retrieval/node_record.py` | persisted node record contract | chunk/char/neighbor metadata **[Code-Confirmed]** |
| `retrieval/search_metadata.py` | retrieval hit DTO | score/source/chapter/position **[Code-Confirmed]** |

## 4. Main Responsibilities

1. 把 raw text 轉換為可檢索節點與位置 metadata。 **[Code-Confirmed]**
2. 建立與保存 FAISS index + records。 **[Code-Confirmed]**
3. 重載 index 並提供 query search contract。 **[Code-Confirmed]**
4. 將 profile 附掛在 runtime bundle。 **[Code-Confirmed]**

## 5. Non-Responsibilities

1. 不負責 structured hierarchy parser。 **[Code-Confirmed]**
2. 不負責 summary/quiz artifact write。 **[Code-Confirmed]**
3. 不負責 API mapping。 **[Code-Confirmed]**

## 6. Important Data Structures / Contracts

- `ParsedDocument`
- `NodeRecord`
- `FaissIndexBundle`
- `FaissIndexBuilder`
- `FaissIndexStore`
- `SearchMetadata`

## 7. Module Relationships

- depends on: `doc_loaders/`, `language/`, `embeddings/`, `llm/`, `config/`
- used by: `bundle_factory.py`, `bundle_provider.py`, `context/`, `app/qa_coordinator.py`

## 8. Main Flows Involving This Module

1. prepare/index flow：node parse -> build -> save。 **[Code-Confirmed]**
2. runtime load flow：load index/records -> bundle search。 **[Code-Confirmed]**
3. query flow：embed query -> FAISS search -> SearchMetadata。 **[Code-Confirmed]**

## 9. Persistence / Side Effects

- read persistence：是（index.faiss + records.json）
- write persistence：是
- mutate structured document：否
- call LLM：間接（透過 language detector capability path）

## 10. Known Legacy / Compatibility Behavior

1. `FaissIndexStore.has_position_metadata` 用於偵測 legacy records schema 並觸發 rebuild。 **[Code-Confirmed]**
2. `NodeRecord.position()` 保留 legacy metadata accessor。 **[Code-Confirmed]**

## 11. Current Risks

1. risk：records schema drift
- why：load path 可能失敗或 silently 缺欄
- guardrail：`has_position_metadata` + rebuild

2. risk：embedding/token budgets 與模型能力不一致
- why：context/quality 成本波動
- guardrail：token budget resolver + capability logs

3. risk：node text fallback 定位偏移
- why：char offsets 可能不精準
- guardrail：保留 metadata 欄位與 downstream容錯

## 12. Open Questions for Maintainer

1. records schema 是否需要版本欄位（目前靠欄位存在判斷）？ **[Needs Confirmation]**

## 13. Suggested Next Documentation Improvements

1. 補 index artifact schema 文件（index + records + meta + profile）。
