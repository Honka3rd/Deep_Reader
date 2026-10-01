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

- [x] Stabilize raw data directory resolution across API container and repo-root scripts
  Evidence: `Deep_Reflective_Reader/doc_loaders/raw_data_paths.py`; `Deep_Reflective_Reader/doc_loaders/document_loader_factory.py`; `Deep_Reflective_Reader/doc_loaders/text_document_loader.py`; `Deep_Reflective_Reader/doc_loaders/pdf_document_loader.py`; `Deep_Reflective_Reader/scripts/test_pdf_document_loader_inspection.py`; `Deep_Reflective_Reader/.venv/bin/python Deep_Reflective_Reader/scripts/test_pdf_document_loader_inspection.py`.
  Notes: Default raw document loading now resolves `data/raw` to the package/project raw directory independent of process cwd. Explicit absolute/custom `base_dir` injection remains supported for tests and specialized callers, while cwd-relative shadow `data/raw` directories no longer influence `DocumentLoaderFactory` selection.
- [x] Remove OCR file-cache persistence from PDF loading
  Evidence: `Deep_Reflective_Reader/doc_loaders/pdf_document_loader.py`; `Deep_Reflective_Reader/scripts/test_pdf_document_loader_inspection.py`; `docker-compose.yml`; `.env.example`; container verification that `DEEP_READER_PDF_OCR_CACHE_ENABLED` is absent and OCR memory reuse does not create a cache directory.
  Notes: OCR cache was a temporary bootstrap mechanism. It is removed as a second persistence surface now that OCR run storage exists.
- [x] Preserve page-aware OCR provenance for future table-of-contents detection
  Evidence: `Deep_Reflective_Reader/doc_loaders/pdf_page_evidence.py`; `Deep_Reflective_Reader/doc_loaders/pdf_document_loader.py`; `Deep_Reflective_Reader/scripts/test_pdf_document_loader_inspection.py`; `Deep_Reflective_Reader/scripts/test_manual_structure_preparation_handoff.py`; `Deep_Reflective_Reader/scripts/test_manual_structure_commit_source_evidence.py`.
  Notes: PDF page-boundary evidence preserves stable page indices, optional page labels/page numbers, source PDF SHA-256, raw-text offsets, and page text while keeping `load(doc_name) -> str` unchanged. This supports TOC page anchors and manual page-range validation as evidence only; full OCR geometry, competing layout hypotheses, and parser authority remain separate future work.

- [x] Expose compact PDF page-boundary evidence for manual TOC anchors
  Evidence: `Deep_Reflective_Reader/doc_loaders/pdf_page_evidence.py`; `Deep_Reflective_Reader/doc_loaders/pdf_document_loader.py`; `Deep_Reflective_Reader/scripts/test_pdf_document_loader_inspection.py`; `Deep_Reflective_Reader/scripts/test_pdf_outline.py`.
  Notes: `PdfDocumentLoader.load_page_boundary_evidence(doc_name)` exposes source SHA-256, page count, page index, optional page label, and raw-text offset range per page while preserving `load(doc_name) -> str`; existing page-boundary callers continue through `load_page_text_boundaries()`. Detailed OCR geometry remains diagnostic and must not enter task-layout payload.

- [x] Detect and expose native PDF Outline/bookmark structure before OCR
  Evidence: `Deep_Reflective_Reader/doc_loaders/pdf_outline.py`; `Deep_Reflective_Reader/doc_loaders/pdf_document_loader.py`; `Deep_Reflective_Reader/scripts/test_pdf_outline.py`; container verification on `Deep_Reflective_Reader/data/raw/许三观卖血记.pdf`.
  Notes: Loader inspects `/Outlines`, bookmark nesting, destination page indices, page labels, destination type, and source PDF SHA-256. Document Info metadata is not treated as hierarchy evidence. Invalid or incomplete destinations remain rejected evidence rather than silently repaired.

- [x] Define universal PDF page-layout evidence metadata
  Evidence needed: every PDF page supports compact source identity, dimensions, native-text/image metrics, orientation hypotheses, writing mode, reading order, OCR confidence, and evidence schema version.
  Evidence: `Deep_Reflective_Reader/doc_loaders/pdf_page_evidence.py`; `Deep_Reflective_Reader/doc_loaders/pdf_document_loader.py`; `Deep_Reflective_Reader/scripts/test_pdf_document_loader_inspection.py`; `Deep_Reflective_Reader/scripts/test_pdf_page_evidence.py`; `Deep_Reflective_Reader/scripts/test_toc_projection.py`.
  Notes: `PdfPageLayoutEvidence` now carries compact source identity (`source_file_name`, `source_sha256`), dimensions, native text/image metrics, orientation, writing mode, reading order, OCR confidence/text, analysis stage, evidence reasons, and `evidence_schema_version`. `PdfDocumentLoader.load_page_layout_evidence(...)` populates source metadata for every page while preserving `load(doc_name) -> str`; OCR word boxes remain bounded to candidate/coordinate-OCR evidence and parser authority stays in `document_structure`.

- [x] Define deterministic orientation and reading-order evidence
  Evidence needed: page-level rules distinguish horizontal/vertical text, `left_to_right`/`right_to_left` column order, rotation (`0/90/180/270`), and mixed regions; ambiguous pages retain competing hypotheses and confidence instead of silent selection.
  Evidence: `Deep_Reflective_Reader/doc_loaders/pdf_page_evidence.py`; `Deep_Reflective_Reader/scripts/test_pdf_page_evidence.py`.
  Notes: `PdfPageLayoutEvidence` now preserves page orientation hypotheses with rotation degrees, selected rotation when deterministic, writing-mode and reading-order confidence, and explicit `ambiguous_orientation` evidence when dimensions are too close to choose safely. Vertical Chinese columns retain `vertical` + `right_to_left` evidence; ambiguous pages keep competing portrait/landscape hypotheses instead of silently authorizing hierarchy.

- [x] Define tiered PDF inspection and OCR cost policy
  Evidence needed: all pages receive cheap inspection; only candidate pages receive coordinate OCR; only high-scoring candidates receive high-resolution OCR or image-geometry analysis.
  Evidence: `Deep_Reflective_Reader/doc_loaders/pdf_page_evidence.py`; `Deep_Reflective_Reader/doc_loaders/pdf_document_loader.py`; `Deep_Reflective_Reader/scripts/test_pdf_page_evidence.py`; `Deep_Reflective_Reader/scripts/test_pdf_document_loader_inspection.py`.
  Notes: Page-layout evidence now records explicit `analysis_cost_tier`, stage, OCR engine/language, render DPI, OCR pass count, cache key, cache hit/miss, failure reason, and whether high-cost analysis is permitted. `PdfDocumentLoader.load_page_layout_evidence(...)` records every page as cheap inventory when OCR is disabled or outside the candidate limit, and only candidate pages enter coordinate OCR. High-cost analysis remains disabled by default and represented as policy metadata, not hidden work.

- [x] Define region-first OCR for mixed and vertical layouts
  Evidence needed: vertical columns, horizontal blocks, rotated regions, headers, footers, watermarks, and artwork can be separated before OCR, with source coordinates and local reading order retained.
  Evidence: `Deep_Reflective_Reader/doc_loaders/pdf_page_evidence.py`; `Deep_Reflective_Reader/scripts/test_pdf_page_evidence.py`.
  Notes: Detect connected components/line bands, cluster by x/y overlap and stroke orientation, rotate vertical regions to OCR-friendly orientation, OCR each region with bounded PSM/language candidates, then merge by deterministic geometry. Keep raw region OCR and normalized logical text separately.
  Implemented Notes: `PdfOcrRegionEvidence` now captures page-local OCR regions with source bounding boxes, raw OCR, normalized logical text, local writing mode, local reading order, rotation metadata, confidence, token count, quality flags, and deterministic evidence reasons. OCR words are grouped into vertical/right-to-left column regions, horizontal text regions, and bounded header/footer regions before serialization. Region evidence remains provenance only and does not authorize hierarchy splitting or replace canonical raw text.

- [x] Define page-level orientation and reading-order evidence contract
  Evidence needed: each page preserves dimensions, rotation hypotheses, horizontal/vertical mode, left-to-right/right-to-left order, region count, OCR confidence, and schema version.
  Evidence: `Deep_Reflective_Reader/doc_loaders/pdf_page_evidence.py`; `Deep_Reflective_Reader/scripts/test_pdf_page_evidence.py`.
  Notes: `PdfPageLayoutEvidence` now serializes dimensions, rotation degrees, competing orientation hypotheses, writing mode, reading order, writing/order confidence, `region_count`, OCR confidence, OCR word boxes, and schema version. Empty OCR pages keep `region_count=0`; OCR pages expose at least one bounded page-level region, with multi-column pages retaining a higher count. Orientation/order metadata remains evidence only and cannot authorize hierarchy splitting by itself.

- [x] Define OCR quality and character provenance contract
  Evidence needed: normalized spans trace to page index, region id, OCR output, confidence, bounding box, and normalization version.
  Evidence: `Deep_Reflective_Reader/doc_loaders/pdf_page_evidence.py`; `Deep_Reflective_Reader/scripts/test_pdf_page_evidence.py`.
  Notes: `PdfOcrWordEvidence` now carries page index, region id, raw OCR token, normalized token, normalization version, confidence, bounding box, TSV hierarchy numbers, and bounded quality flags for low-confidence, symbol-heavy, and repeated-garbage tokens. Token provenance remains OCR/layout evidence only; low-quality OCR still cannot become trusted structure evidence by itself.

- [x] Define multi-PSM OCR candidate selection for scanned PDF raw loading
  Evidence needed: `_ocr_image_text()` compares bounded Tesseract candidates instead of accepting the first non-empty default output.
  Evidence: `Deep_Reflective_Reader/doc_loaders/pdf_document_loader.py`; `Deep_Reflective_Reader/scripts/test_pdf_document_loader_inspection.py`; `Deep_Reflective_Reader/.venv/bin/python Deep_Reflective_Reader/scripts/test_pdf_document_loader_inspection.py`.
  Notes: `_ocr_image_text()` now compares default, `--psm 5`, `--psm 6`, and `--psm 11` instead of accepting the first non-empty default output. Selection is deterministic and records candidate score/rejection reason as OCR provenance only.

- [x] Define low-quality OCR raw-load failure gate
  Evidence needed: non-empty but low-quality OCR output maps to an explicit raw-load failure before language/profile/structured build.
  Evidence: `Deep_Reflective_Reader/doc_loaders/document_load_errors.py`; `Deep_Reflective_Reader/doc_loaders/pdf_document_loader.py`; `Deep_Reflective_Reader/document_preparation/document_preparation_pipeline.py`; `Deep_Reflective_Reader/scripts/test_document_preparation_raw_load_errors.py`; container prepare API verification on `國富論lite`.
  Notes: Non-empty but low-quality OCR now raises `RawTextOcrLowQualityError` and maps to `load_raw_text_ocr_low_quality:<doc_name>:<reason>` before language/profile/structured build.

- [x] Define vertical Chinese OCR reading-order remediation for raw text handoff
  Evidence needed: vertical/mixed scanned pages preserve writing-mode and reading-order hypotheses, and default PSM cannot silently win when another candidate has stronger deterministic evidence.
  Evidence: `Deep_Reflective_Reader/doc_loaders/pdf_document_loader.py`; `Deep_Reflective_Reader/scripts/test_pdf_document_loader_inspection.py`; container OCR verification on `國富論lite`.
  Notes: Default PSM can no longer silently win for vertical/mixed scanned Chinese raw text when another candidate has stronger deterministic evidence. This remains loader-level quality control and does not decide chapter/section hierarchy.

- [x] Add OCR quality regression coverage for `國富論lite`
  Evidence needed: tests or container verification prove `國富論lite` no longer produces corrupted hierarchy titles such as `HE mm姐1]` after force rebuild.
  Evidence: `Deep_Reflective_Reader/scripts/test_pdf_document_loader_inspection.py`; `Deep_Reflective_Reader/scripts/test_document_preparation_raw_load_errors.py`; container prepare API verification returned `structured_document_ready=false` and `load_raw_text_ocr_low_quality` for `國富論lite` with `force_rebuild=true`.
  Notes: Regression covers candidate selection, low-quality rejection, explicit prepare error mapping, OCR memory reuse, and non-empty garbage rejection. `國富論lite` low-quality OCR is rejected before structured hierarchy persistence.

- [x] Prepare OCR layout and renderer-failure fixtures
  Evidence needed: fixtures cover `暗水幽靈.pdf` artistic vertical TOC, `國富論.pdf` vertical body text, horizontal scans, mixed orientation, circular page numbers, leader lines, and renderer decode failure.
  Evidence: `Deep_Reflective_Reader/doc_loaders/pdf_ocr_layout_fixtures.py`; `Deep_Reflective_Reader/scripts/test_pdf_ocr_layout_fixtures.py`.
  Notes: Each fixture defines expected orientation/order, candidate regions, known limitations, and fallback behavior. Artistic scans need not achieve full-text equality; page/region evidence and conservative rejection are required.
  Implemented Notes: The OCR layout fixture catalog now covers `暗水幽灵` artistic vertical TOC, `國富論` vertical body text, horizontal scan leader lines, mixed header/body/footer orientation regions, circular page numbers, and renderer decode failure fallback. Fixture tests validate page-local layout evidence, region typing, TOC candidate reconstruction where appropriate, conservative quality rejection for artistic/body-text cases, and renderer failure fallback expectations without making OCR evidence parser authority.

After implementation, the task owner must update this checklist and mark the task as completed:

- [x] <completed task>

No coding task should be considered complete unless the corresponding module checklist is updated.

## Maintenance Notes

- This checklist is module memory for completed work.
- It does not replace the module detailed design document.
- It does not replace tests and test evidence.
- It does not replace proposal/HLD decisions and governance context.
