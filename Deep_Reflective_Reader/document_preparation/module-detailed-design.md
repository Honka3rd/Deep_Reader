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

5. risk：cache hit 被誤解為完整 prepare reuse
- why：`task_layout_cache_hit` 或 OCR/page text cache hit 只證明該層資料可重用，不等同於 language/profile/structured preparation 已全部短路。若 route 仍進入 prepare flow，可能再次觸發 raw/OCR 讀取、language detection、profile build，甚至間接 LLM call。 **[Code-Observed] + [Inferred]**
- guardrail：將 cache hit log 分層命名；明確區分 raw/OCR cache、structured artifact reuse、profile/language cache、task-layout projection cache。read-like caller 不應把任何單一 cache hit 當作整條 prepare pipeline 完成。 **[Future Direction]**

6. risk：base preparation 在既有文檔讀取場景未完全避免昂貴 path
- why：UI 選擇既有文檔時，若只是要讀取當前 task-layout，進入 `base` prepare 可能因 structured/profile/language reuse 條件不成立而重新載入 PDF/OCR，並間接觸發 LLM-backed language/profile 分析。 **[Code-Observed] + [Inferred]**
- guardrail：`force_rebuild=false`、common parser、既有 structured artifact 可用時，應優先 short-circuit 到既有 hierarchy；若 task-layout cache 可用但 structured reuse 失敗，應記錄為 cache-boundary inconsistency 供排查，而不是安靜走昂貴 prepare。 **[Future Direction]**

7. risk：language detection cache 無法被有效使用
- why：language detection 若未接收可用的 storage/config context，可能無法從 profile 或 retrieval records reuse 既有語言結果，只能 fallback 到 LLM detector。 **[Code-Observed] + [Inferred]**
- guardrail：language detection 應有明確 cache source priority（profile/records/source metadata -> deterministic fallback -> LLM fallback），並以測試保證既有文檔的 read-like path 不重複呼叫 LLM。 **[Future Direction]**

## 15. Open Questions for Maintainer

1. cache-first 命名標準化與實作命名遷移先標註、後落地的時程是否固定？ **[From HLD] + [Needs Confirmation]**
2. post-structure enrichment 是否需要獨立 refresh 機制（非 prepare 路徑）？
3. prepare logs 是否需要標準化為可機器解析的 event code？

## 16. Suggested Next Documentation Improvements

1. 增加 preparation pipeline lifecycle diagram（base vs free_qa）。
2. 增加錯誤分類矩陣（blocking/non-blocking）。
3. 增加 profile cache policy appendix（hash/version/rebuild triggers）。

## 17. Future Storage Independence Note

> 本節是 storage abstraction boundary preparation，非目前 implementation。 **[Maintainer-Provided] + [Future Direction]**

1. The preparation pipeline should prepare artifacts and readiness snapshots. **[Code-Confirmed] + [Future Direction]**
2. The preparation pipeline should not permanently assume file-path persistence as the only possible destination. **[Maintainer-Provided] + [Future Direction]**
3. Future storage destination should be configurable through a storage/backend policy boundary, while preparation lifecycle semantics remain owned by `document_preparation/`. **[Maintainer-Provided] + [Future Direction]**
4. This note does not introduce storage abstractions, repositories, DB schemas, dual-write behavior, read-path switches, API changes, parser changes, or runtime behavior changes. **[Doc-Confirmed]**

## 18. Future Direction Note: Manual Structure Reparse Preparation Boundary

> 本節記錄 source-agnostic manual structure override 的 preparation-layer boundary；不代表目前 implementation。 **[Maintainer-Provided] + [Future Direction]**

1. manual structure reparse may apply to any raw document source, not only OCR/scanned PDFs. **[Maintainer-Provided]**
2. `document_preparation/` should own preparation-time handoff of raw text, language, source identity, page boundaries when available, and existing structured/profile artifacts needed by validation/reparse orchestration. **[Future Direction]**
3. The preparation layer should not decide manual hierarchy semantics; projection and validation semantics belong to `document_structure/`. **[Future Direction]**
4. For source-agnostic support, manual anchors should be able to target raw-text character spans even when page evidence is unavailable. **[Future Direction]**
5. For PDF/OCR/native-page sources, preparation may provide page boundaries and OCR provenance as validation evidence, but page evidence remains advisory/supporting data rather than parser authority by itself. **[From HLD] + [Future Direction]**
6. manual validation failure must be reported as structured readiness failure for that explicit reparse attempt; it must not overwrite existing structured artifacts and must not silently fallback to common parser as a successful manual reparse. **[Future Direction]**
7. Existing `force_rebuild` semantics should remain explicit: manual structure commit is a mutation/reparse operation, while validation/preview is read/analysis-oriented and should not persist a hierarchy. **[Future Direction]**
8. This note does not introduce new API fields, parser modes, persistence behavior, or runtime route behavior. **[Doc-Confirmed]**

## 19. Page Evidence Handoff For TOC Editing

> 本節支援 UI TOC editor 的 page-first anchor UX。Manual-structure source evidence can now carry validated page boundaries; downstream `page_range` hierarchy materialization and task-layout prefill projection remain separate checkpoints. **[Maintainer-Provided] + [Code-Confirmed] + [Future Direction]**

1. `load_manual_structure_source_evidence(...)` provides a reusable source-evidence handoff containing canonical raw text, raw-text source hash, best-effort language, and validated page boundaries when available. **[Code-Confirmed]**
2. For pageable sources such as PDFs, page evidence includes stable document page indices, optional display labels/page labels, and raw-text offset ranges per page. **[Code-Confirmed]**
3. Preparation validates manual page evidence before handoff: duplicate or non-contiguous page indices, invalid ranges, out-of-bounds ranges, non-monotonic ranges, and page-text mismatches are rejected. **[Code-Confirmed]**
4. Invalid or unavailable page evidence is reported in source-evidence `errors` and dropped while preserving canonical raw text and `char_range` fallback. **[Code-Confirmed]**
5. This evidence is needed by:
   - manual-structure validation/commit for `page_range` anchors
   - task-layout anchor evidence projection for UI edit-existing prefill
6. Preparation must not decide user-defined hierarchy semantics; it only supplies evidence. Projection and hierarchy draft building remain owned by `document_structure/`. **[From HLD] + [Code-Confirmed]**
7. Page evidence remains supporting evidence, not parser authority by itself. Ambiguous, incomplete, or stale page evidence must be reported explicitly and must not authorize partial hierarchy persistence. **[From HLD] + [Future Direction]**
8. `char_range` remains available when page evidence is unavailable or rejected. **[Maintainer-Provided] + [Code-Confirmed]**

## 20. Future Direction Note: Prepare Reuse And LLM Cost Boundary

> 本節記錄本輪運維觀察後的 cache/prepare governance；不代表目前 implementation 已完成。 **[Code-Observed] + [Future Direction]**

1. A task-layout cache hit is a projection-layer reuse signal, not proof that document preparation has been fully skipped. **[Code-Observed]**
2. OCR/page-text cache hits are raw-source reuse signals. They do not by themselves guarantee language/profile/structured reuse. **[Code-Observed] + [Inferred]**
3. For read-like flows that only need the current active hierarchy or task-layout, preparation should avoid re-entering expensive OCR/LLM-backed stages when a valid structured artifact and compatible task-layout projection already exist. **[Future Direction]**
4. Language detection should receive enough storage/config context to reuse existing language evidence from profile or retrieval records before calling LLM-backed detection. **[Future Direction]**
5. Profile building/classification may use LLM as fallback, but repeated UI document selection should not implicitly rebuild profile metadata when source hash, parser mode, and structured artifact are unchanged. **[Future Direction]**
6. If `task_layout_cache_hit` is observed before a later prepare-stage OCR or LLM call for the same document request, logs should expose whether the cause was forced rebuild, structured artifact miss, source hash mismatch, parser/schema version mismatch, profile cache miss, language cache miss, or storage-backend mismatch. **[Future Direction]**
7. This note does not change parser authority: metadata and LLM classification remain advisory, and hierarchy truth remains `chapters[].sections[].task_units[]`. **[From HLD]**
