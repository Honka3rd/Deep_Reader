# doc_loaders Checklist

## Purpose

This checklist records completed, code-confirmed or design-confirmed tasks for the `doc_loaders` module.

It is used to:
- preserve module-level implementation memory
- reduce hallucination in future Codex tasks
- prevent context-window compression from losing completed work
- track future task completion explicitly

## Source Documents

- `Deep_Reflective_Reader/doc_loaders/module-detailed-design.md`
- `Deep_Reflective_Reader/proposal.md`
- `Deep_Reflective_Reader/high-level-design.md`
- `Deep_Reflective_Reader/doc_loaders/`

## Rules

- Only completed work is listed as checked.
- Future work must not be added unless explicitly requested.
- If a new task is added later, it must first be added unchecked.
- Once completed, it must be checked in this file.
- Uncertain items must go to `Needs Confirmation`, not the completed checklist.

## Completed Checklist

- [x] Defines loader abstraction via `AbstractDocumentLoader.load(doc_name) -> str`.
  Evidence: `Deep_Reflective_Reader/doc_loaders/abstract_document_loader.py; Deep_Reflective_Reader/doc_loaders/module-detailed-design.md (Key Files)`
  Notes: Raw-text loading contract is explicit and reusable.

- [x] Implements TXT loader and PDF loader for canonical raw text extraction.
  Evidence: `Deep_Reflective_Reader/doc_loaders/text_document_loader.py; Deep_Reflective_Reader/doc_loaders/pdf_document_loader.py; Deep_Reflective_Reader/doc_loaders/module-detailed-design.md (Main Responsibilities)`
  Notes: Both loaders normalize `doc_name` with extension handling.

- [x] Implements loader selection through `DocumentLoaderFactory` with extension/path heuristics and historical TXT default.
  Evidence: `Deep_Reflective_Reader/doc_loaders/document_loader_factory.py; Deep_Reflective_Reader/doc_loaders/module-detailed-design.md (Known Legacy / Compatibility Behavior)`
  Notes: Factory keeps compatibility default when ambiguity remains.

- [x] Detect scanned-image PDFs before prepare treats them as generic empty raw text
  Evidence: `Deep_Reflective_Reader/doc_loaders/pdf_document_loader.py`; `Deep_Reflective_Reader/scripts/test_pdf_document_loader_inspection.py`
  Notes: `PdfDocumentLoader.inspect(doc_name)` reports native text chars, image-page ratio, font-resource presence, and stable scanned-image classification; `暗水幽灵.pdf` is detected as 261 pages, 0 native text chars, 261 image pages, 0 font pages.

- [x] Add explicit requires-OCR raw-load failure reason
  Evidence: `Deep_Reflective_Reader/doc_loaders/document_load_errors.py`; `Deep_Reflective_Reader/doc_loaders/pdf_document_loader.py`; `Deep_Reflective_Reader/document_preparation/document_preparation_pipeline.py`; `Deep_Reflective_Reader/scripts/test_document_preparation_raw_load_errors.py`
  Notes: scanned-image PDF load raises `RawTextRequiresOcrError`; preparation raw-load maps it to `load_raw_text_requires_ocr:<doc_name>` without invoking parser fallback or OCR.

- [x] Design and implement optional OCR fallback behind explicit configuration or request option
  Evidence: `Deep_Reflective_Reader/doc_loaders/pdf_document_loader.py`; `Deep_Reflective_Reader/doc_loaders/document_load_errors.py`; `Deep_Reflective_Reader/document_preparation/document_preparation_pipeline.py`; `Deep_Reflective_Reader/scripts/test_pdf_document_loader_inspection.py`; `Deep_Reflective_Reader/scripts/test_document_preparation_raw_load_errors.py`
  Notes: OCR fallback is disabled by default and gated by `DEEP_READER_PDF_OCR_ENABLED=1` or constructor injection; it uses local Tesseract CLI with configurable language/binary and maps OCR runtime failure to `load_raw_text_ocr_failed:<doc_name>:<reason>`.

- [x] Deploy multilingual OCR language packages and align OCR language selection with project language-code strategy
  Evidence: `Dockerfile`; `docker-compose.yml`; `.env.example`; `Deep_Reflective_Reader/doc_loaders/pdf_ocr_language_policy.py`; `Deep_Reflective_Reader/doc_loaders/pdf_document_loader.py`; `Deep_Reflective_Reader/scripts/test_pdf_document_loader_inspection.py`
  Notes: Docker runtime installs Tesseract English, simplified Chinese, and traditional Chinese language data; PDF OCR defaults to `eng+chi_sim+chi_tra` for unknown-language raw loading while preserving explicit env/constructor override.

- [x] Retire OCR text file-cache persistence after OCR run storage
  Evidence: `Deep_Reflective_Reader/doc_loaders/pdf_document_loader.py`; `Deep_Reflective_Reader/document_preparation/document_preparation_pipeline.py`; `Deep_Reflective_Reader/db/postgres_structured_document_store.py`; `Deep_Reflective_Reader/scripts/test_pdf_document_loader_inspection.py`; `docker-compose.yml`; `.env.example`
  Notes: `PdfDocumentLoader` no longer reads or writes `data/ocr_text`. OCR output is retained only in memory during a single prepare pass and is durably persisted through the structured store `save_ocr_run` path after document creation.

- [x] Use renderer-first PDF page normalization for OCR when embedded image decoding is unreliable
  Evidence: `Dockerfile`; `Deep_Reflective_Reader/doc_loaders/pdf_document_loader.py`; `docker-compose.yml`; container verification on `國富論.pdf` page 5.
  Notes: Poppler `pdftoppm` renders complete pages before OCR, covering `JBIG2Decode` and broken Pillow image-stream cases. Renderer command and DPI are configurable and included in OCR cache provenance; the source PDF remains unchanged.

## Needs Confirmation

No unresolved confirmation items identified in this pass.

## Future Task Policy

New future tasks for this module must be added here first as unchecked items:

- [ ] Stabilize raw data directory resolution across API container and repo-root scripts
  Evidence needed: `DocumentLoaderFactory` resolves `data/raw` consistently regardless of current working directory.
  Notes: Current relative `Path("data/raw")` can choose different loader behavior when run outside `Deep_Reflective_Reader` working directory.
- [x] Remove OCR file-cache persistence from PDF loading
  Evidence: `Deep_Reflective_Reader/doc_loaders/pdf_document_loader.py`; `Deep_Reflective_Reader/scripts/test_pdf_document_loader_inspection.py`; `docker-compose.yml`; `.env.example`; container verification that `DEEP_READER_PDF_OCR_CACHE_ENABLED` is absent and OCR memory reuse does not create a cache directory.
  Notes: OCR cache was a temporary bootstrap mechanism. It is removed as a second persistence surface now that OCR run storage exists.
- [ ] Preserve page-aware OCR provenance for future table-of-contents detection
  Evidence needed: OCR output can expose stable page boundaries, page numbers, source hashes, and page-level text without changing the public raw-text loading contract.
  Notes: Planning-only. Page boundaries are required to use TOC page numbers as anchors for noisy/scanned documents.

- [x] Detect and expose native PDF Outline/bookmark structure before OCR
  Evidence: `Deep_Reflective_Reader/doc_loaders/pdf_outline.py`; `Deep_Reflective_Reader/doc_loaders/pdf_document_loader.py`; `Deep_Reflective_Reader/scripts/test_pdf_outline.py`; container verification on `Deep_Reflective_Reader/data/raw/许三观卖血记.pdf`.
  Notes: Loader inspects `/Outlines`, bookmark nesting, destination page indices, page labels, destination type, and source PDF SHA-256. Document Info metadata is not treated as hierarchy evidence. Invalid or incomplete destinations remain rejected evidence rather than silently repaired.

- [ ] Define universal PDF page-layout evidence metadata
  Evidence needed: every PDF page supports compact source identity, dimensions, native-text/image metrics, orientation hypotheses, writing mode, reading order, OCR confidence, and evidence schema version.
  Notes: Applies to born-digital, scanned, mixed, horizontal, vertical, rotated, and mixed-layout PDFs. Full OCR bounding boxes and connected-component evidence are retained only for candidate pages or explicit diagnostics.

- [ ] Define deterministic orientation and reading-order evidence
  Evidence needed: page-level rules distinguish horizontal/vertical text, `left_to_right`/`right_to_left` column order, rotation (`0/90/180/270`), and mixed regions; ambiguous pages retain competing hypotheses and confidence instead of silent selection.
  Notes: Vertical Chinese columns must support right-to-left ordering. OCR/OSD is evidence only and cannot independently authorize hierarchy.

- [ ] Define tiered PDF inspection and OCR cost policy
  Evidence needed: all pages receive cheap inspection; only candidate pages receive coordinate OCR; only high-scoring candidates receive high-resolution OCR or image-geometry analysis.
  Notes: Record analysis stage, OCR engine/language/version, resolution, elapsed time, cache key, cache hit/miss, and failure reason.

- [ ] Define region-first OCR for mixed and vertical layouts
  Evidence needed: vertical columns, horizontal blocks, rotated regions, headers, footers, watermarks, and artwork can be separated before OCR, with source coordinates and local reading order retained.
  Notes: Detect connected components/line bands, cluster by x/y overlap and stroke orientation, rotate vertical regions to OCR-friendly orientation, OCR each region with bounded PSM/language candidates, then merge by deterministic geometry. Keep raw region OCR and normalized logical text separately.

- [ ] Define page-level orientation and reading-order evidence contract
  Evidence needed: each page preserves dimensions, rotation hypotheses, horizontal/vertical mode, left-to-right/right-to-left order, region count, OCR confidence, and schema version.
  Notes: Compute hypotheses from word/line bounding boxes and regions, not OCR text alone. Preserve competing hypotheses when confidence is close. Orientation/order metadata is evidence only and cannot authorize hierarchy splitting by itself.

- [ ] Define OCR quality and character provenance contract
  Evidence needed: normalized spans trace to page index, region id, OCR output, confidence, bounding box, and normalization version.
  Notes: Flag low-confidence, symbol-heavy, repeated-garbage, and implausible-language spans. Whitespace normalization must not erase punctuation, page identity, or source traceability. Low-quality OCR cannot become trusted structure evidence.

- [ ] Prepare OCR layout and renderer-failure fixtures
  Evidence needed: fixtures cover `暗水幽靈.pdf` artistic vertical TOC, `國富論.pdf` vertical body text, horizontal scans, mixed orientation, circular page numbers, leader lines, and renderer decode failure.
  Notes: Each fixture defines expected orientation/order, candidate regions, known limitations, and fallback behavior. Artistic scans need not achieve full-text equality; page/region evidence and conservative rejection are required.

After implementation, the task owner must update this checklist and mark the task as completed:

- [x] <completed task>

No coding task should be considered complete unless the corresponding module checklist is updated.

## Maintenance Notes

- This checklist is module memory for completed work.
- It does not replace the module detailed design document.
- It does not replace tests and test evidence.
- It does not replace proposal/HLD decisions and governance context.
