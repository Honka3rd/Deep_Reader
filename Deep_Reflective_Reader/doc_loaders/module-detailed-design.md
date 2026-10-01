# doc_loaders Detailed Design

## 1. Module Purpose

`doc_loaders/` 提供 raw document 載入抽象，依副檔名選擇 txt/pdf loader，輸出 canonical raw text 供 prepare/retrieval 流程使用。 **[Code-Confirmed]**

## 2. Position in Overall Architecture

- Document Preparation Layer（input loader sub-layer）

## 3. Key Files

| File | Responsibility | Notes |
|---|---|---|
| `doc_loaders/abstract_document_loader.py` | loader 抽象介面 | `load(doc_name)->str` **[Code-Confirmed]** |
| `doc_loaders/text_document_loader.py` | 讀取 `data/raw/*.txt` | UTF-8 text **[Code-Confirmed]** |
| `doc_loaders/pdf_document_loader.py` | 讀取 `data/raw/*.pdf` 並抽取全文 | `pypdf` pages join **[Code-Confirmed]** |
| `doc_loaders/document_loader_factory.py` | loader 選擇器 | extension/path existence 判斷 **[Code-Confirmed]** |
| `doc_loaders/pdf_document_loader.py` OCR path | 掃描圖片 PDF optional OCR fallback | 透過 explicit config gate 啟用本地 Tesseract OCR；預設關閉 **[Code-Confirmed]** |
| `doc_loaders/pdf_document_loader.py` OCR provenance | OCR provenance + page text handoff | OCR output is retained in memory during one prepare pass and handed to structured-store OCR run persistence; `data/ocr_text` file cache is not an active persistence path. **[Code-Confirmed]** |

## 4. Main Responsibilities

1. 讀取 raw source 並回傳文字內容。 **[Code-Confirmed]**
2. 根據檔名/副檔名選擇 loader。 **[Code-Confirmed]**

## 5. Non-Responsibilities

1. 不做 parser split。 **[Code-Confirmed]**
2. 不做 profile/metadata。 **[Code-Confirmed]**
3. 不做 artifact persistence。 **[Code-Confirmed]**

## 6. Important Data Structures / Contracts

- `AbstractDocumentLoader`
- `TextDocumentLoader`
- `PdfDocumentLoader`
- `DocumentLoaderFactory`

## 7. Module Relationships

- used by: `document_preparation/`, `bundle_provider.py`
- depends on: filesystem + `pypdf`

## 8. Main Flows Involving This Module

1. prepare flow 讀取 raw doc。 **[Code-Confirmed]**
2. bundle provider ensure index 時讀取 raw doc。 **[Code-Confirmed]**

## 9. Persistence / Side Effects

- read persistence：是（`data/raw`）
- write persistence：否
- call external service：否

## 10. Known Legacy / Compatibility Behavior

1. `DocumentLoaderFactory` 在無法明確判斷時維持歷史預設回傳 txt loader。 **[Code-Confirmed]**

## 11. Scanned PDF / OCR Boundary

目前 `PdfDocumentLoader` 僅支援 born-digital PDF，即 PDF 內已有可抽取文字層的檔案。其實作透過 `pypdf.PdfReader(...).pages[*].extract_text()` 取得文字並 join 成 canonical raw text。對掃描圖片 PDF，頁面內容主要是 image XObject，沒有 embedded text layer，`extract_text()` 會回傳空字串。 **[Code-Confirmed]**

已觀察失敗案例：

- `Deep_Reflective_Reader/data/raw/暗水幽灵.pdf`
- PDF 共 261 頁。 **[Code-Confirmed]**
- 每頁包含 image XObject。 **[Code-Confirmed]**
- 未觀察到 page font resources。 **[Code-Confirmed]**
- `pypdf.extract_text()` 全文抽取結果為 0 字。 **[Code-Confirmed]**
- `/documents/prepare` 在 raw-load 階段得到空 raw text，回傳 `load_raw_text_empty:<doc_name>`，後續 language/profile/structured build 皆以 `missing_raw_text` 跳過。 **[Code-Confirmed]**

此類問題不是 `document_structure` parser 問題；parser 沒有收到 raw text，因此 common / llm parser 都不會被真正執行。 **[Inferred]**

Current OCR behavior：

1. `PdfDocumentLoader.inspect(doc_name)` 可區分「PDF 無文字層但疑似掃描圖片」與「真正空文件」，並輸出 page count、native text chars、image-page ratio、font-page ratio。 **[Code-Confirmed]**
2. 對掃描圖片 PDF 且 OCR 未啟用時，loader 會 raise `RawTextRequiresOcrError`，prepare raw-load 階段映射為 `load_raw_text_requires_ocr:<doc_name>`。 **[Code-Confirmed]**
3. OCR fallback 透過 explicit configuration gate 啟用，不會隱式改變所有 PDF 的成本與 latency。 **[Code-Confirmed]**
4. 目前 gate 可透過 `DEEP_READER_PDF_OCR_ENABLED=1` 啟用；OCR language 預設讀取 `DEEP_READER_PDF_OCR_LANGUAGE`，未配置時為 `eng+chi_sim+chi_tra`；Tesseract binary 可由 `DEEP_READER_TESSERACT_CMD` 指定，未配置時為 `tesseract`。 **[Code-Confirmed]**
5. OCR 目前走本地 Tesseract CLI：loader 逐頁抽取 embedded image，寫入暫存檔，呼叫 `tesseract <image> stdout -l <language>`，將非空 stdout join 成 raw text。 **[Code-Confirmed]**
6. OCR 成功產出的文字可作為 raw text input；OCR failure 會映射為 `load_raw_text_ocr_failed:<doc_name>:<reason>`。 **[Code-Confirmed]**
7. OCR file cache has been removed as an active persistence path; loader must not read or write `data/ocr_text`. **[Code-Confirmed]**
8. OCR output is retained in memory only within the active loader instance so `load()` and page-boundary loading in the same prepare pass do not run OCR twice. Durable OCR output/provenance belongs to the structured-store `save_ocr_run` path after document creation. **[Code-Confirmed]**
9. OCR provenance includes `doc_name`, source filename, raw PDF SHA-256, OCR engine, OCR engine version, OCR language, OCR page limit, page count, renderer, and render DPI. **[Code-Confirmed]**
10. OCR language 應由配置或顯式 request 決定；後續可加入 heuristic，但不應依賴 LLM 分類作硬控制。 **[From HLD] + [Future Direction]**

Observed OCR quality failure（`國富論lite.pdf`）：

1. `國富論lite.pdf` 是無 native text layer 的 scanned-image PDF；runtime log 顯示 `pages=3 native_text_chars=0 image_pages=3 font_pages=0 scanned=True`。 **[Code-Confirmed]**
2. OCR enabled 時，`PdfDocumentLoader` 成功產生約 `1727` characters，prepare 因 raw text 非空而繼續 language/profile/structured build。 **[Code-Confirmed]**
3. API task-unit content 返回的 `chapter_title` / `section_title` / `title` 出現 `HE mm姐1]`，表示 corrupted OCR text 已進入 structured hierarchy 並被 task-layout/content read path 投影。 **[Maintainer-Provided] + [Inferred]**
4. Container spot check 顯示同一頁在不同 Tesseract PSM 下品質差異明顯：default PSM 產生大量 Latin-like garbage，`--psm 6` 產生較多中文但仍存在順序與噪聲問題，`--psm 11` 產生碎字。 **[Code-Confirmed]**
5. `_ocr_image_text()` 目前是 default PSM 先行，只要 default PSM 有非空文字就停止；因此直排/混排頁面可能被低品質 default PSM 輸出截斷後續候選。 **[Code-Confirmed] + [Inferred]**

Implemented OCR quality remediation：

1. OCR raw-text handoff 不再只以「非空」作成功標準；low-quality OCR gate 會阻止 corrupted OCR 成為 canonical raw text。 **[Code-Confirmed]**
2. `_ocr_image_text()` 會比較多個候選，而不是 default PSM 非空即接受；目前候選包含 default、`--psm 5`、`--psm 6`、`--psm 11`。 **[Code-Confirmed]**
3. 候選選擇使用 deterministic quality scoring：CJK ratio、Latin-like garbage ratio、符號/括號/重複碎片密度、字數下限與 OCR candidate score。 **[Code-Confirmed]**
4. OCR quality scoring 是 loader-level raw-text quality decision，不是 parser authority；不決定 chapter/section split，也不把 OCR/profile/LLM metadata 升格為 hierarchy truth。 **[Code-Confirmed] + [From HLD]**
5. 對直排中文與 mixed-layout pages，default PSM 不再能 silent win；另一候選若有更強 deterministic evidence 可勝出。Full region-first ordering 與 competing layout hypothesis retention 仍屬後續 deeper layout work。 **[Code-Confirmed] + [Future Direction]**
6. 當 OCR 產生文字但低於 quality gate 時，prepare 會收到明確 raw-load failure：`load_raw_text_ocr_low_quality:<doc_name>:<reason>`，並跳過 language/profile/structured build。 **[Code-Confirmed]**
7. 已被 corrupted OCR 產物污染的文件（例如 `國富論lite`）需要 explicit `force_rebuild=true` 重新產生 structured artifacts；task-layout read path 不承擔修復持久化資料的責任。 **[Code-Confirmed]**

Suggested implementation shape（future）：

```text
PdfDocumentLoader
  -> native pypdf text extraction
  -> pdf inspection metrics
  -> if native text exists: return text
  -> if scanned-image signature and OCR enabled: render pages and run bounded OCR candidates
  -> score candidates and reject low-quality OCR before raw-text handoff
  -> if OCR quality passes: expose text/pages/provenance for structured-store OCR run persistence
  -> if OCR is low quality: fail with explicit low-quality OCR reason
  -> if scanned-image signature and OCR disabled: fail with requires_ocr reason
```

OCR dependency candidates:

- document rendering: Poppler (`pdftoppm`) or PyMuPDF
- OCR engine: Tesseract / OCRmyPDF / cloud OCR service
- Python integration: `pytesseract` + `Pillow`, or wrapper around `ocrmypdf`

Container/runtime implication：目前 `requirements.txt` 只包含 `pypdf`，Docker image 未安裝 Tesseract/OCRmyPDF，也未安裝中文 OCR language data；因此 OCR fallback 只有在 runtime image 或 host 已提供 Tesseract binary 與所需語言包時才可用。 **[Code-Confirmed]**

## 12. Current Risks

1. risk：pdf 文本抽取品質不穩
- why：會影響後續 parser/profile
- guardrail：保留 enhanced parse recommendation 與人工 reparse

2. risk：副檔名判斷歧義（同名 txt/pdf）
- why：可能載入非預期來源
- guardrail：要求 API 明確 doc_name + extension（若需） **[Needs Confirmation]**

3. risk：掃描圖片 PDF 被誤報成一般空 raw text
- why：使用者無法區分「文件真的空」與「需要 OCR」，UI/API 也無法提示下一步
- guardrail：新增 scanned PDF detection 與 requires-OCR reason code **[Future Direction]**

4. risk：OCR fallback 若隱式啟用會造成 prepare latency / dependency / cost 不可預期
- why：OCR 對大 PDF 成本高，且依賴本地二進位或外部服務
- guardrail：以 explicit option/config gate 啟用，並加入 OCR cache **[Future Direction]**

5. risk：`DocumentLoaderFactory` 使用相對 `data/raw`，在不同工作目錄下可能誤判 loader
- why：API container 工作目錄與 repo-root script 工作目錄不同
- guardrail：將 raw base dir 收斂為配置或 package-relative/project-root-aware path **[Future Direction]**

6. risk：OCR non-empty output 被誤當作 trusted raw text
- why：直排/混排掃描頁可能在 default Tesseract PSM 下產生大量亂碼；非空亂碼會污染 language/profile/structured hierarchy，並在 task-layout/content read path 中被投影
- guardrail：比較多 PSM OCR candidates，加入 deterministic quality scoring 與 low-quality raw-load failure，低品質 OCR 不得進入 structured persistence **[Code-Confirmed]**

7. risk：直排中文閱讀順序未被 raw OCR path 正確處理
- why：頁面可能需要 vertical/right-to-left column ordering 或 region-first OCR；單一線性 OCR output 無法可靠代表 canonical body text
- guardrail：目前已加入 candidate PSM scoring 與 rejection reason evidence；更完整的 page-level orientation、writing mode、reading order、region-first OCR 仍屬 future deeper layout work；layout evidence 保持 advisory/provenance，不作 parser authority **[Code-Confirmed] + [Future Direction]**

## 13. Open Questions for Maintainer

1. 同名 `.txt` 與 `.pdf` 並存時是否要 fail-fast 而非預設 txt？ **[Needs Confirmation]**
2. OCR fallback 是否應作為 prepare request option、environment config，或獨立 `/documents/ocr` 類明確 mutation endpoint？ **[Needs Confirmation]**
3. 中文 OCR 預設語言包應包含 `chi_sim`、`chi_tra`，還是由使用者在文件層指定？ **[Needs Confirmation]**
4. OCR text cache 長期是否要升級為 preparation artifact cache，而不是 `doc_loaders/` local raw-text cache？ **[Needs Confirmation]**
5. Low-quality OCR 應新增專用 exception type，還是重用 `RawTextOcrFailedError` 並以 stable detail code 表示？ **[Needs Confirmation]**
6. OCR quality threshold 應固定於 `doc_loaders/`，還是由 runtime config 控制？ **[Needs Confirmation]**
7. Low-quality OCR diagnostics 應只經 prepare response 暴露，還是後續需要獨立 diagnostics endpoint？ **[Needs Confirmation]**

## 14. Suggested Next Documentation Improvements

1. 補 raw source naming convention 文件。
2. 補 raw-load failure reason taxonomy（missing file / empty native text / scanned requires OCR / OCR failed / unsupported encrypted PDF）。
3. 補 OCR dependency and deployment matrix（local dev / Docker / production）。
4. 補 OCR quality gate taxonomy（empty OCR / low-confidence OCR / symbol-heavy OCR / implausible-language OCR / ambiguous vertical reading order）。

## 15. Page Boundary Evidence For TOC Anchors

> 本節支援 UI TOC editor 的 page-first anchor UX。Loader-level compact page boundary evidence is implemented; downstream `page_range` validation/commit wiring remains owned by later modules. **[Maintainer-Provided] + [Code-Confirmed] + [Future Direction]**

1. `PdfDocumentLoader.load_page_boundary_evidence(doc_name)` exposes stable page-boundary evidence without changing the canonical `load(doc_name) -> str` raw-text contract. **[Code-Confirmed]**
2. Page evidence includes document page index, optional PDF page label when available, raw-text start/end offsets for that page, and source PDF SHA-256. **[Code-Confirmed]**
3. `load_page_text_boundaries(doc_name)` remains as the compatibility handoff for existing preparation/outline callers and now returns the same enriched boundary entries. **[Code-Confirmed]**
4. Native text PDFs and OCR PDFs share the compact boundary contract; deeper mixed-layout and coordinate evidence remains future work. **[Code-Confirmed] + [Future Direction]**
5. OCR/layout region details should not be pushed into task-layout; detailed geometry remains diagnostic/supporting evidence and must stay bounded. **[From HLD] + [Future Direction]**
6. Page evidence may support manual `page_range` validation and UI prefill, but it cannot independently become parser authority or hierarchy truth. **[From HLD] + [Maintainer-Provided]**
