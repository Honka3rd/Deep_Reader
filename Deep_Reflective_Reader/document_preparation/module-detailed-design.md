# document_preparation Detailed Design

## 1. Module Purpose

`document_preparation/` 負責準備流程編排（prepare orchestration）：把 raw document 轉成可用於 QA/task-layout 的 artifacts readiness snapshot。 **[Code-Confirmed]**

## 2. Position in Overall Architecture

- Document Preparation Layer

## 3. Key Files

| File | Responsibility | Notes |
|---|---|---|
| `document_preparation/document_preparation_pipeline.py` | 主準備流程（raw/language/profile/structured/faiss/bundle） | 包含 Step 4.5 post-structure enrichment **[Code-Confirmed]** |
| `document_preparation/prepared_document_assets.py` | readiness DTO（ready flags/path/errors） | API prepare response 主要來源之一 **[Code-Confirmed]** |
| `document_preparation/prepared_document_result.py` | prepare + loaded artifacts 組合 DTO | 給 coordinator consume **[Code-Confirmed]** |
| `document_preparation/preparation_mode.py` | `base` / `free_qa` mode contract | mode resolve + validation **[Code-Confirmed]** |

## 4. Main Responsibilities

1. pipeline step ordering（載入、語言、profile、structured、faiss、bundle）。 **[Code-Confirmed]**
2. base/free_qa mode 差異化準備。 **[Code-Confirmed]**
3. profile build 與 profile store reuse/force rebuild 控制。 **[Code-Confirmed]**
4. structured build + atomic save。 **[Code-Confirmed]**
5. post-structure metadata enrichment 並儲存更新 profile。 **[Code-Confirmed]**
6. 失敗不阻塞的錯誤收集（尤其 profile/enrichment）。 **[Code-Confirmed]**

## 5. Non-Responsibilities

1. 不負責 API request/response mapping。 **[From HLD]**
2. 不負責 task-layout 組裝與 diagnostics projection。 **[From HLD]**
3. 不應把 parser metadata 當 parser hard rule。 **[From Proposal] + [From HLD]**

## 6. Important Data Structures / Contracts

- `PreparedDocumentAssets`
- `PreparedDocumentResult`
- `PreparationMode`
- `SectionSplitterMode`（作為 structured parser mode 輸入）

## 7. Preparation Lifecycle

```text
Step 1  Load canonical raw text
Step 2  Detect document language
Step 3  Prepare document profile
Step 4  Prepare structured document
Step 4.5 Enrich profile with post-structure metadata
Step 5  Prepare FAISS artifacts
Step 6  Prepare runtime bundle
```

說明：`BASE` mode 也會走 profile（Step 3）與 enrichment（Step 4.5）。 **[Code-Confirmed]**

## 8. Step I/O Artifact Matrix

| Step | Inputs | Outputs | Persistence |
|---|---|---|---|
| 1 Raw load | `doc_name`, loader | `raw_text` | read `data/raw` |
| 2 Language detect | `raw_text`, optional existing artifacts | `document_language` | read profile/records (if exists) |
| 3 Profile prepare | `raw_text`, `document_language` | `DocumentProfile`, `profile_ready` | write `profile.json` |
| 4 Structured prepare | `raw_text`, parser mode | `StructuredDocument`, `structured_document_ready` | write `*.structured.json` |
| 4.5 Post enrich | profile + structured doc | enriched profile snapshot | write `profile.json` |
| 5 FAISS prepare | `raw_text` (+ language via parsed document) | index/records/meta | write `index.faiss`/`records.json`/`meta.json` |
| 6 Runtime bundle | index + profile | `FaissIndexBundle` | runtime cache |

## 9. Non-Blocking Policy Matrix

| Failure Point | Blocks Prepare? | `profile_ready` | `structured_document_ready` | Error Collection |
|---|---:|---:|---:|---|
| profile load/build fail | No | `false` | unaffected | add error reason |
| post-structure enrichment fail | No | keep prior profile result | unaffected | add error reason |
| structured build/save fail | Yes for structured-ready | unaffected | `false` | add error reason |
| faiss/bundle fail (free_qa path) | Yes for index/bundle readiness | unaffected | unaffected | add error reason |

**[Code-Confirmed]**

## 10. Module Relationships

- depends on:
  - `doc_loaders/`, `language/`, `document_structure/`, `retrieval/`, `profile/`, `bundle_provider.py`
- used by:
  - `app/qa_coordinator.py`
  - `app/section_task_coordinator.py`（via coordinator chain）
  - `main.py` `/documents/prepare`
- reads from/writes to:
  - structured storage
  - faiss storage
  - profile storage

## 11. Main Flows Involving This Module

1. prepare flow（核心）。 **[Code-Confirmed]**
2. structured hierarchy build flow（透過 structured builder）。 **[Code-Confirmed]**
3. profile metadata flow（pre-structure builder + post-structure enricher）。 **[Code-Confirmed]**
4. bundle/index readiness flow（free_qa mode）。 **[Code-Confirmed]**

## 12. Persistence / Side Effects

- read persistence：是（structured/profile/faiss）
- write persistence：是（structured/profile/faiss）
- mutate structured document：間接（透過 structured builder/repository 寫出）
- generate runtime projection：否
- call LLM：間接（language detector/profile builder/llm parser path）
- diagnostics only：否

## 13. Known Legacy / Compatibility Behavior

1. profile load path 支援 cache hit/reload failure rebuild。 **[Code-Confirmed]**
2. structured load/save 仍允許舊 JSON 讀取（由 model/store 層承接）。 **[Code-Confirmed]**
3. 本模組不直接管理 root sections/structure_nodes compatibility 細節（委派到 structure layer）。 **[Code-Confirmed]**

## 14. Current Risks

1. risk：prepare step 增長造成錯誤來源難追蹤
- why：多 artifact pipeline 容易有 partial success
- guardrail：維持 assets.errors 分層前綴與 step log

2. risk：profile/cache policy 與 parser 策略邊界混淆
- why：可能把 advisory metadata 變成 parse authority
- guardrail：文件化 cache-only contract + 測試約束

3. risk：base/free_qa 分支差異被新功能破壞
- why：可能出現 base mode 不完整或重複構建
- guardrail：mode-specific regression tests

4. risk：post-structure enrichment 失敗處理語義漂移
- why：若變成阻塞會改變 API 行為
- guardrail：固定 non-blocking 行為測試

## 15. Open Questions for Maintainer

1. cache-first 命名標準化與實作命名遷移先標註、後落地的時程是否固定？ **[From HLD] + [Needs Confirmation]**
2. post-structure enrichment 是否需要獨立 refresh 機制（非 prepare 路徑）？
3. prepare logs 是否需要標準化為可機器解析的 event code？

## 16. Suggested Next Documentation Improvements

1. 增加 preparation pipeline lifecycle diagram（base vs free_qa）。
2. 增加錯誤分類矩陣（blocking/non-blocking）。
3. 增加 profile cache policy appendix（hash/version/rebuild triggers）。
