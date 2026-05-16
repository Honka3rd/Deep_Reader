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

## 11. Current Risks

1. risk：pdf 文本抽取品質不穩
- why：會影響後續 parser/profile
- guardrail：保留 enhanced parse recommendation 與人工 reparse

2. risk：副檔名判斷歧義（同名 txt/pdf）
- why：可能載入非預期來源
- guardrail：要求 API 明確 doc_name + extension（若需） **[Needs Confirmation]**

## 12. Open Questions for Maintainer

1. 同名 `.txt` 與 `.pdf` 並存時是否要 fail-fast 而非預設 txt？ **[Needs Confirmation]**

## 13. Suggested Next Documentation Improvements

1. 補 raw source naming convention 文件。
