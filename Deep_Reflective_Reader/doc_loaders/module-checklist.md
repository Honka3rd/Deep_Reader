# doc_loaders Checklist

## Reviewable OCR Recovery

Ordered cross-module queue; design: module-detailed-design.md Reviewable OCR Recovery.
No duplicate completion counts in supporting modules. Explicit OCR partial imports
are enabled with evidence exposure and interaction guards; Fast/Auto are unchanged.

- [x] OCR-REVIEW-01: Recover Chinese margin title/leader/page-number components.
  Acceptance: backed-up Sample2 all-page OCR, original Sample regression, exact affected title/digits, geometry preserved and garbage controls retained.
  Evidence: chinese-columns-v2; scripts/accept_chinese_margin.py passes all three physical pages (450/0/644 characters, confirmed blank page 2), exact page-3 margin and source offsets. Original six-page Sample and composite mixed-layout acceptance pass unchanged. Private backup hash 05650a15d6a0428d543e4ffee273e5423ecdbcecf14c7ae0f4a02b16d13574de; report data/test-books/ocr-review-20261021/acceptance.json. No general partial import or production reparse.
  Timestamp: 2026-10-21
- [x] OCR-REVIEW-02: Expose bounded page/region failure reasons without source content.
  Acceptance: distinguish quality rejection from engine/render/budget failures; no generic masking of known quality reasons; tests for both paths.
  Evidence: full OCR maps allowlisted Chinese quality reasons to quality_failed with physical page and region; unknown execution errors remain extraction_failed without source/stderr. Nonzero regional engine exit is no longer empty text. 28 focused tests pass; PDF discovery runs 182 (180 pass, two opt-in skips). Existing render/budget semantics preserved; Auto diagnostic expansion is not claimed.
  Timestamp: 2026-10-21
- [x] OCR-REVIEW-03: Persist generic unresolved-region evidence and gated partial policy.
  Acceptance: immutable source-linked evidence, all-page accounting, no fabricated text and bounded readable-body acceptance; explicit OCR only.
  Evidence: source-review-v1 records hash/page/bbox/DPI/reason/raw candidate, with no synthetic block for empty output. Coverage requires >=40 readable characters and accepted rectangle area >= uncertain area. Sample2 real OCR and fault-injected margin recovery pass; all-unreadable input fails. This fixture-tested heuristic is not universal semantic calibration. Strict rollback flag remains available.
  Timestamp: 2026-10-22
- [x] OCR-REVIEW-04: Expose source review and implement Library/reader warning UI.
  Acceptance: existing namespace-scoped read-only original-region preview, source replacement conflict, page/region selection and persistent review-needed state without hidden OCR/profile work.
  Evidence: GET reading-content and source-review/{region_id}/image, Library review_required and reader warning. Real backup crop and wrong-source hash rejection pass; image preview performs one render and zero OCR. Desktop/mobile editor/API verification passes. No new authentication system is claimed.
  Timestamp: 2026-10-22
- [x] OCR-REVIEW-05: Implement versioned block edit/delete/undo and atomic invalidation.
  Acceptance: document_structure/section_tasks/main/api_schemas/UI owners implement immutable source overlays, bounded edits, revision conflicts, deletion confirmation, undo, ancestor hash/cache invalidation, note preservation and dirty-draft guards.
  Evidence: 12 file-store plus 12 disposable PostgreSQL tests pass, including concurrent writers, failed-transaction rollback, stale saves/generation and notes. Edit/Delete live in the existing block interactive menu; no manual add/insert action. Undo is single-level; receipts/audit limit 1000 mutations. Generated invalidation is deliberately document-wide, not selective. Source/profile/hierarchy IDs stay unchanged.
  Timestamp: 2026-10-22
- [x] OCR-REVIEW-06: Guard interactions and accept the complete reviewable workflow.
  Acceptance: unresolved sources cannot produce invented analysis/quiz/thinking/summary; backup-only real-source tests, correction/delete/undo/conflict/failure cases and desktop/mobile UI verification before enabling partial imports.
  Evidence: accept_reading_review.py passes real Sample2 normal (450/0/644) and fault-injected partial (450/0/634, one unresolved region), all four generator/context poison guards and immutable-source mutations. Original Sample/mixed-image regression passes unchanged (12/17 columns, scoped CER 4.82%/1.85%). PDF suite 184 runs, 182 pass/two skips. Production-editor browser/API harness passes 1280/390px original preview, failed Save retry, dirty-close rejection, concurrent conflict, deletion confirmation and undo. npm run check passes. No production Library reparse or Docker deployment claimed.
  Timestamp: 2026-10-22

## Direction-Aware OCR Remediation

Ordered implementation queue; design: module-detailed-design.md#direction-aware-ocr-remediation-2026-10-21.
Status: 7/7 complete for the documented Chinese-local scope. OCR sample/mixed-image acceptance passes; Auto on this sample remains a verified protective rejection, not extraction success. Decorative-title transcription errors remain. No live Library reparse/deployment is claimed.

- [x] OCR-DIR-01: Preserve the real mixed-direction source and establish reviewed regression expectations.
  Acceptance: private backup of 中國文化的深層結構_Sample.pdf under data/test-books, manifest/hash verification and tests targeting the backup; reproduce pages 5/6 garbage acceptance and page 3 horizontal errors; reviewed transcription and column/title/page-number associations with explicit accuracy/coverage/order thresholds. No Library deletion.
  Evidence: private ocr-direction-20261021/raw backup and manifest; SHA-256 3a0c9e6acb1b5f4f63e770b724a3b1615303139cfbee0417ed2092a6b1d73571 verified before/after acceptance. Reference title CER ceiling 8%; report explicitly excludes decorative headings/punctuation from that metric.
  Timestamp: 2026-10-21
- [x] OCR-DIR-02: Define page/region direction hypotheses and the model capability contract.
  Acceptance: separate rotation from horizontal/vertical/mixed/uncertain writing mode; preserve source geometry, alternatives and reasons; define explicit configuration compatibility, missing-model failures, pinned vertical model packaging/checksums and extraction policy identity without changing Fast identity. No request-time model downloads.
  Chinese scope: only horizontal left-to-right lines ordered top-to-bottom, or vertical top-to-bottom columns ordered right-to-left. Own these rules in the PDF loader's Chinese strategy, not shared language-independent logic or upper pipeline branches; other languages remain independent.
  Evidence: chinese_ocr_strategy.py Column/ChineseOcrResult, pinned traditional/simplified model hashes and Docker build checks; missing-model typed failure, non-Chinese bypass and Fast identity tests pass. No automatic script identification or universal rotation detection claimed.
  Timestamp: 2026-10-21
- [x] OCR-DIR-03: Implement bounded direction-aware recognition routing inside the PDF loader.
  Acceptance: visual page/region evidence selects suitable model/PSM candidates; uncertain cases compare bounded alternatives, not one hard guess; test horizontal-to-vertical transitions, same-page mixed layout and missing models. Reuse rendered pages and existing execution budgets; do not rebuild/use profile as authority.
  Evidence: width-stable projection routing, bounded numeric/short-label alternatives and mixed-band routing in chinese_ocr_strategy.py. Original six-page OCR and composite real-image header/body pass; all subprocesses remain under loader execution budgets.
  Timestamp: 2026-10-21
- [x] OCR-DIR-04: Assemble source-faithful region text in reading order.
  Acceptance: within the Chinese strategy only, vertical characters top-to-bottom with right-to-left columns, or horizontal characters left-to-right with top-to-bottom lines; mixed regions combine these two orders. Associate TOC labels/leaders/page numbers without merging columns or dropping text; retain source mapping and detect missing/duplicate regions. Unsupported/uncertain arrangements must not be forced into an accepted order. Do not rotate upright glyphs indiscriminately or invent text with LLM.
  Evidence: all 12/17 reference TOC column associations/order pass; region boxes and logical offsets retained, mixed-band offsets translated. Unsupported interleaving is not universally recognized; unexplained coverage rejects. Small decorative labels remain imperfect OCR, not LLM-corrected text.
  Timestamp: 2026-10-21
- [x] OCR-DIR-05: Correct candidate selection and quality gates for mixed-direction documents.
  Acceptance: known page 5/6 Latin-like garbage cannot pass solely on letter count; readable TOC candidates are not rejected solely for leaders; combine coverage/layout/confidence checks with text checks and genuine bilingual negatives. No acceptable candidate means explicit failure, not silently partial persistence; thresholds fixed against reviewed references.
  Evidence: CJK/coverage/confidence guards, numeric-label association and leader separation; garbage, low-confidence body and excessive/missing coverage fixtures reject. Low-confidence short labels are explicitly flagged and bounded to 5% of page text; not a perfect-transcription claim. Reference CER is 4.82%/1.85% against the unchanged 8% formal acceptance ceiling.
  Timestamp: 2026-10-21
- [x] OCR-DIR-06: Integrate OCR/Auto routes, identity invalidation and decision logging.
  Acceptance: both OCR-required paths share the router; Fast invokes no direction analysis/render/OCR, Auto native pages stay native; fingerprint routing/model/quality versions; log per-page/region hypothesis, model/PSM, timings, metrics and selection reasons without text/images. Reads do not rebuild artifacts/profile; explicit preparation handles stale extraction via existing lifecycle rules.
  Evidence: shared _ocr_image_text_with_quality entry, model/policy extraction fingerprint and existing SourceArtifact provenance handoff; region diagnostics/metrics logged without text. Fast actual-source check has zero subprocesses. No upper pipeline, UI or read-side mutation added.
  Timestamp: 2026-10-21
- [x] OCR-DIR-07: Complete real-engine acceptance and regression validation.
  Acceptance: backed-up sample meets predeclared transcription/coverage/order thresholds, including pages 3/5/6; additional blank/rotated/mixed/bilingual/unavailable-model cases pass; verify failure preserves accepted data, warm reuse does not rerun OCR/profile, and Fast/OCR/Auto isolation. Publish reproducible commands, model hashes, timings and remaining limitations; nonempty output or model installation alone cannot close this task.
  Scope validation: both supported Chinese orders and their mixed-region combination; unusual arrangements require safe uncertainty/rejection, not successful recognition. Assert non-Chinese strategy behavior is unchanged and no Chinese reading-order rules leak into shared orchestration or upper preparation layers.
  Evidence: private acceptance-ocr.json and acceptance-auto-boundary.json; mixed real-image fixture passes. 15 Chinese unit tests, 101 overlapping focused tests, 180 PDF tests (178 pass/two skips), and 18/18 real-engine synthetic mode cases pass. Sideways coverage fixture checks rejection of implausible output, not general rotated-text OCR. Auto native_coverage_unresolved remains intentional on this book; decorative titles and arbitrary layouts are not certified.
  Timestamp: 2026-10-21

## Loader Boundary Governance

- [x] LOADER-01: Define source-session and extracted-artifact contracts, including lazy evidence and source verification.
  Acceptance: identity before extraction, one extraction per session, generic source-format schema and compatibility boundary.
  Evidence: source_session.py, abstract loader contract and generic identity validation; seven boundary tests including synthetic third format, lazy capture, cache and source-race checks pass.
  Timestamp: 2026-10-21
- [x] LOADER-02: Supply sessions from PDF/TXT loaders without changing extraction strategies or legacy callers.
  Acceptance: format validation stays in adapters; immutable captured bytes, optional evidence and legacy status/provenance parity.
  Evidence: PDF/TXT open_source adapters; loader-owned PDF identity with compatibility import; 180 PDF Docker tests pass. New session default is Fast, legacy load unchanged. Both real backup sources pass exact-text/cold-warm acceptance.
  Timestamp: 2026-10-21

## Format-Neutral Extraction Backlog (2026-10-21)

Implemented; evidence and scope are recorded per item. Ordered design: [shared plan](../document_preparation/module-detailed-design.md#format-neutral-extraction-plan-2026-10-21).

- [x] EXTRACT-01: Define immutable format-neutral extraction identity/result contracts.
  Depends on: none.
  Acceptance: Strict identity/schema/hash validation, stable settings serialization, text digest/length/completion, optional mappings and load() compatibility tests; no PDF dependency in generic contract.
  Evidence: Generic extraction_identity.py contracts; strict digest/schema/completion/settings tests in test_generic_extraction.py.
  Timestamp: 2026-10-21

- [x] EXTRACT-02: Adapt PDF extraction to the shared result without changing its three strategies.
  Depends on: EXTRACT-01.
  Acceptance: Fast/OCR/Auto retain source/settings/version identity, canonical text and page evidence; single extraction, failure parity and no new fallback verified.
  Evidence: PdfExtractionResult.as_extraction_result adapts same-pass text; 180 PDF regression tests pass in Docker.
  Timestamp: 2026-10-21

- [x] EXTRACT-03: Add deterministic TXT extraction identity and result.
  Depends on: EXTRACT-01.
  Acceptance: Hash/decode the same bytes; pin UTF-8, BOM and universal-newline semantics; test invalid bytes, empty text, rename and changed bytes; no OCR/LLM/profile calls.
  Evidence: TextDocumentLoader.extract preserves UTF-8/BOM/universal newlines and same-byte hashes; invalid/empty-input and rename tests pass.
  Timestamp: 2026-10-21

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

- [x] Add per-page PDF render and OCR candidate timing logs.
  Evidence: `pdf_document_loader.py`; passing `scripts/test_pdf_document_loader_inspection.py`; shared timing helper failure/correlation checks in `scripts/test_upload_preparation_logging.py`.
  Notes: Logs page/DPI/PSM/language, process outcome, recognized character counts and elapsed time, excluding OCR content. Rendering, candidate selection and quality gates remain unchanged.
  Timestamp: 2026-10-14

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

- [x] Define region evidence for mixed and vertical layouts (not canonical region-first OCR)
  Evidence needed: vertical columns, horizontal blocks, rotated regions, headers, footers, watermarks, and artwork can be separated before OCR, with source coordinates and local reading order retained.
  Evidence: `Deep_Reflective_Reader/doc_loaders/pdf_page_evidence.py`; `Deep_Reflective_Reader/scripts/test_pdf_page_evidence.py`.
  Scope correction: completed work is post-recognition region evidence only. Pre-recognition segmentation/model routing and canonical reading-order assembly remain OCR-DIR-02..04. Whole-page rotation is not a replacement for upright vertical-glyph recognition.
  Implemented Notes: `PdfOcrRegionEvidence` now captures page-local OCR regions with source bounding boxes, raw OCR, normalized logical text, local writing mode, local reading order, rotation metadata, confidence, token count, quality flags, and deterministic evidence reasons. OCR words are grouped into vertical/right-to-left column regions, horizontal text regions, and bounded header/footer regions before serialization. Region evidence remains provenance only and does not authorize hierarchy splitting or replace canonical raw text.
  Timestamp: 2026-10-21

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

- [x] Implement deterministic multi-PSM selection for raw text handoff (not vertical-order acceptance)
  Scope correction: competing text candidates are scored; direction-aware model routing and correct vertical reading order are not established.
  Evidence: `Deep_Reflective_Reader/doc_loaders/pdf_document_loader.py`; `Deep_Reflective_Reader/scripts/test_pdf_document_loader_inspection.py`; container OCR verification on `國富論lite`.
  Notes: The 2026-10-21 real sample demonstrates default PSM garbage can still win and pass. OCR-DIR-03..07 track the missing behavior and acceptance; previous 國富論lite rejection proved a negative guard only, not readable vertical OCR.
  Timestamp: 2026-10-21

- [x] Add OCR quality regression coverage for `國富論lite`
  Evidence needed: tests or container verification prove `國富論lite` no longer produces corrupted hierarchy titles such as `HE mm姐1]` after force rebuild.
  Evidence: `Deep_Reflective_Reader/scripts/test_pdf_document_loader_inspection.py`; `Deep_Reflective_Reader/scripts/test_document_preparation_raw_load_errors.py`; container prepare API verification returned `structured_document_ready=false` and `load_raw_text_ocr_low_quality` for `國富論lite` with `force_rebuild=true`.
  Notes: Regression covers candidate selection, low-quality rejection, explicit prepare error mapping, OCR memory reuse, and non-empty garbage rejection. `國富論lite` low-quality OCR is rejected before structured hierarchy persistence.

- [x] Avoid repeated native-text extraction for PDF page-boundary evidence
  Evidence: `Deep_Reflective_Reader/doc_loaders/pdf_document_loader.py`; `Deep_Reflective_Reader/scripts/test_pdf_document_loader_inspection.py`; direct timing check for `APPLE.pdf` page-boundary evidence returned 121 boundaries in 3.338s with `ocr_started=False`.
  Notes: Native PDF loading now extracts page text and metrics in one pass for `load()`, `load_pages()`, and `load_page_boundary_evidence(...)`. The `APPLE.pdf` regression proves a simple born-digital PDF can produce compact page-boundary evidence without invoking OCR or repeating expensive native extraction loops.
  Timestamp: 2026-10-05

- [x] Prepare OCR layout and renderer-failure fixtures
  Evidence needed: fixtures cover `暗水幽靈.pdf` artistic vertical TOC, `國富論.pdf` vertical body text, horizontal scans, mixed orientation, circular page numbers, leader lines, and renderer decode failure.
  Evidence: `Deep_Reflective_Reader/doc_loaders/pdf_ocr_layout_fixtures.py`; `Deep_Reflective_Reader/scripts/test_pdf_ocr_layout_fixtures.py`.
  Notes: Each fixture defines expected orientation/order, candidate regions, known limitations, and fallback behavior. Artistic scans need not achieve full-text equality; page/region evidence and conservative rejection are required.
  Implemented Notes: The OCR layout fixture catalog now covers `暗水幽灵` artistic vertical TOC, `國富論` vertical body text, horizontal scan leader lines, mixed header/body/footer orientation regions, circular page numbers, and renderer decode failure fallback. Fixture tests validate page-local layout evidence, region typing, TOC candidate reconstruction where appropriate, conservative quality rejection for artistic/body-text cases, and renderer failure fallback expectations without making OCR evidence parser authority.

After implementation, the task owner must update this checklist and mark the task as completed:

- [x] <completed task>

No coding task should be considered complete unless the corresponding module checklist is updated.

## Selective Per-Page OCR Track

Design: `module-detailed-design.md` section 16. Eight unique tasks across modules;
OCR-06 is owned by `document_preparation/module-checklist.md`, OCR-08 by
`scripts/module-checklist.md`. Execute in numeric order, one task per turn.

- [x] OCR-01: Define typed per-page OCR decision contract.
  Acceptance: Immutable source/page/purpose identity; skip/required/uncertain decisions, stable reasons, versioned metadata-only serialization; invalid identities/states rejected; independent text/layout decisions; contract tests without OCR, LLM or persistence.
  Evidence: `pdf_ocr_decision.py`; `scripts/test_pdf_ocr_decision.py` (7 tests passed); existing `scripts/test_pdf_document_loader_inspection.py` and `scripts/test_pdf_page_evidence.py` passed on 2026-10-14.
  Scope: No loader runtime routing change or performance improvement claimed.
  Timestamp: 2026-10-14

- [x] OCR-02: Collect cheap per-page evidence without OCR.
  Acceptance: Native extraction reused inside inspection; text quality, resource/geometry availability and clipped image coverage captured; unknown evidence differs from zero; nested forms/inline images and failures tested without image decoding/OCR/LLM.
  Evidence: `pdf_page_inspection.py`; `PdfDocumentLoader.inspect_pages`; `scripts/test_pdf_page_inspection.py` (18 tests), `test_pdf_ocr_decision.py` (7 tests), existing PDF loader inspection, page-evidence and Outline regressions passed on 2026-10-14.
  Runtime compatibility: The 18 inspection tests also passed in a disposable Compose container with pinned pypdf 6.17.0 and read-only source mount; local tests use pypdf 6.9.2. No API restart or user-document mutation.
  Scope: New internal collector only; one native extraction per page, no automatic decisions/routing. Nested Form geometry and custom clipping remain explicitly unknown; retained geometry is bounded. Design section 16.6 records limitations. No deployed performance gain claimed.
  Depends on: OCR-01.
  Timestamp: 2026-10-14

- [x] OCR-03: Implement conservative three-way decision policy.
  Acceptance: Named injectable thresholds; native, scanned, blank, illustration, header-only, corrupted/duplicate hidden text and mixed-language fixtures; missing evidence stays uncertain; bounded uncertain-page escalation is explicit; text and layout decisions remain separate.
  Evidence: `pdf_ocr_policy.py`; additional stable reasons in `pdf_ocr_decision.py`; `scripts/test_pdf_ocr_policy.py` (22 tests, local and isolated Docker), OCR-01's 7 tests, OCR-02's 18 tests and existing PDF loader inspection passed on 2026-10-14.
  Scope: Pure rules and bounded probe planning only. Visual-check input is source/page matched; no detector/executor implemented. Large image alone remains uncertain; exhausted/disabled probing never becomes skip. Design section 16.7 records thresholds, limitations and future integration boundaries.
  Depends on: OCR-02.
  Timestamp: 2026-10-14

- [x] OCR-04: Gate coordinate OCR with native-first layout evidence.
  Acceptance: Usable native pages do not render or call Tesseract simply because they are inside the candidate limit; candidate zero means zero OCR; optional layout failure retains advisory fallback; scanned/vertical candidate behavior and cheap inventory remain covered.
  Evidence: `PdfDocumentLoader.load_page_layout_evidence`; `scripts/test_pdf_layout_routing.py` (14 tests passed locally and in isolated Compose); earlier decision/policy/inspection, loader, page-evidence, Outline and OCR fixture regressions passed on 2026-10-14.
  Scope: Layout-only gates; native-only uncertain inventory remains uncertain, visual/rotated candidates retain optional explicitly enabled OCR. Single-page errors preserve native inventory and later pages. No raw recovery, scheduling or durable cache change. Design section 16.8 records limitations and fallback semantics.
  Smoke: Read-only local four-page `《歸來》` layout call retained 8,187 native characters with zero render/OCR calls in 346.1 ms. Not full preparation acceptance or deployed speedup; live API and user artifacts unchanged.
  Depends on: OCR-03.
  Timestamp: 2026-10-14

- [x] OCR-05: Route raw recovery per page and preserve mixed-PDF boundaries.
  Acceptance: Native pages retained, suspect regions/pages OCRed only when authorized, duplicate text avoided; page order/raw offsets consistent across load/load_pages/boundaries; OCR-disabled and low-quality failures explicit; no silent partial raw text.
  Evidence: `PdfDocumentLoader._load_canonical_pages` and selective `_load_pages_with_ocr`; focused native/background inspection changes in `pdf_page_inspection.py`; `scripts/test_pdf_raw_recovery.py` (25 tests locally and isolated Compose); PDF policy/inspection/layout/decision, loader, page-evidence, OCR fixtures, updated Outline, manual-structure and upload/preparation logging regressions passed on 2026-10-14.
  Scope: Exact native preservation and shared offsets; full-page fallback where safe region merging is unavailable, normalized native coverage and per-page quality gates, preflight prefix limits, source/state invalidation and uniform-render blank proof. No general text detector, region merging, DB/API change or live deployment. Design section 16.9 records conservative rejection cases and OCR-only provenance slots.
  Smoke: Read-only `《歸來》` retained 8,187 stripped native characters / 8,200 canonical characters across four exact page spans with OCR disabled. Outline cover/trailing recovery is mocked, not real OCR acceptance. No user artifacts changed.
  Depends on: OCR-04.
  Timestamp: 2026-10-14

OCR-06 supporting evidence (counted only in the preparation checklist):
`load_with_page_boundary_evidence` shares one canonical pass between preparation
raw text and page spans; Outline uses one byte snapshot for reading/hashing.
`scripts/test_prepare_lazy_pdf_layout.py` passes 15 tests locally/in isolated
Compose on 2026-10-14; existing PDF regressions pass. Design section 16.10 records
the request-local handoff. Broader layout inspection reuse remains OCR-07.

- [x] OCR-07: Add bounded reuse, execution budgets and decision diagnostics.
  Acceptance: Per-pass evidence keyed by source/page/purpose/policy/schema and relevant engine/language/DPI/region settings; actual cache hit/miss verified; render/OCR attempt and time limits enforced; required-text exhaustion fails explicitly, optional layout exhaustion falls back; content-free request-correlated logs; no new file cache.
  Evidence: `pdf_execution_pass.py`; `pdf_document_loader.py`; factory/preparation scope handoff; `scripts/test_pdf_execution_pass.py` (25 new tests); all 111 PDF unittest cases passed locally and in isolated Compose on 2026-10-14. The 15 lazy-layout tests and 11 standalone PDF/TOC/preparation/manual/logging scripts also passed locally; compilation and diff checks passed.
  Scope: Context-local bounded LRU and OCR handoff; same-pass inspection/decision/raw/layout reuse; source/settings/binary build invalidation; purpose-separated render/OCR/time limits and actual-attempt logs. Unknown executable identity misses; required failures do not publish partial text/hierarchy, optional failures retain inventory. Design section 16.11 records defaults and limitations. No durable cache, API/schema change, live deployment or full performance acceptance.
  Depends on: OCR-06 in the preparation checklist.
  Timestamp: 2026-10-14

## PDF Modes / Multilingual Follow-up (2026-10-15)

Design sections 17/18 record evidence and the superseding maintainer decision.
Loader track: 5/5 complete; PDF-FIX-03 final matrix acceptance passed. Not a retroactive extension of OCR-01..08.
PDF-FIX-01 is superseded, not completed: its automatic-routing work is now
PDF-AUTO-01 below, deferred until fast/ocr delivery. Historical checked OCR tasks
remain evidence of the old implementation, not delivery of the three-mode contract.

- [x] PDF-MODE-02: Implement the default native-only fast path.
  Acceptance: Extract native page text/boundaries without OCR-decision graphics inspection, rendering or OCR, including optional layout fallback. Preserve native Outline/parser fallback and cheap empty/text validity diagnostics. Empty extraction fails actionably without automatic escalation; report empty page indexes without claiming full visual coverage. Poison-call tests prove zero renderer/OCR/complex-routing calls on native, scanned and watermarked inputs.
  Boundary: One validated entry dispatch and independent fast/ocr/auto executors as design 18.1; fast must not wrap auto with ocr_enabled=false. Tests poison auto inspection/render/OCR dependencies. Shared primitives cannot select modes; optional layout binds native-only capability.
  Depends on: PDF-MODE-01 in main checklist. Fast is not the old ocr_enabled=false path if that path still performs conservative routing/rejects unresolved clipping.
  Evidence: pdf_document_loader.py explicit extract dispatch; pdf_fast_extraction.py; pdf_extraction_result.py; scripts/test_pdf_fast_extraction.py (9 passed), existing raw recovery (25), layout routing (14), lazy preparation (15), API mode contract (11) and native Outline regressions passed. Source snapshot providers preserve exact spans/native bookmarks and bind native-only optional layout; poison-call assertions verify zero forbidden calls.
  Scope: Loader-only delivery. Legacy entry points unchanged; public PDF prepare remains gated until PDF-MODE-04/05. OCR/auto explicit executors are not claimed complete. Native-only visual completeness is unverified, with empty-page/encoding diagnostics; no live deployment or persistent mode cache.
  Timestamp: 2026-10-15

- [x] PDF-MODE-03: Implement explicit full-page OCR mode.
  Acceptance: Render and recognize every physical page without native-vs-OCR routing; canonical text comes from accepted OCR, never silent native fallback. Preserve page order, legitimate blank-page handling, bounded attempts/timeouts, provenance and no truncated-success publication. Missing engine/language pack or quality/budget failure is explicit. Existing budgets must reject over-budget input honestly, not silently omit pages.
  Boundary: Independent all-page executor shares OCR/render/quality primitives, not auto routing. Tests poison automatic decision-policy calls; no nested mode flags or silent cross-mode delegation. See design 18.1.
  Depends on: PDF-MODE-01 and PDF-FIX-02 for quality acceptance; independent of PDF-AUTO-01.
  Evidence: pdf_full_ocr_extraction.py; explicit dispatch in pdf_document_loader.py; PdfExtractionResult provenance/quality fields; test_pdf_full_ocr_extraction.py (7 passed), fast (9), quality (5), raw recovery (25), execution pass (25) and API contract (11) regressions passed.
  Scope: Independent all-page executor with snapshot rendering and existing bounded OCR/quality primitives. Poison tests exclude native/auto fallback. No real-engine whole-document acceptance, persistent cache, preparation integration or deployment claimed; API 501 gate remains. See design 18.3.
  Timestamp: 2026-10-15

- [x] PDF-FIX-02: Remove OCR quality's language-pack-as-page-language assumption.
  Acceptance: Valid English pages pass under eng+chi_sim+chi_tra without weakening Chinese/mixed/numeric quality or corrupted-output rejection. Calibrate page-evidence/script-neutral scoring and candidate selection; no profile/LLM parser authority or one-off global threshold reduction. Include the legitimate English page-10 case and native-coverage/empty-text negative controls.
  Evidence: Request a0a2d1a52f214e8d8a27be5b5fff35ce; `_score_ocr_candidate`; page-10 render/native diagnostic. Required for ocr and auto; independent of deferred PDF-AUTO-01.
  Implementation evidence: observed-script-v2 scoring independent of installed packs; pass-first candidate selection and summary; quality version in cache/provenance. scripts/test_pdf_ocr_quality_scripts.py (5 passed, local page-10 diagnostic included), existing loader inspection, raw recovery (25), execution pass (25) and fast extraction (9) regressions passed.
  Scope: Existing score thresholds unchanged; explicit fragmentation/encoding rejection retains garbage controls. Local page-10 evidence is native-text scorer input, not a real-engine whole-document rerun. Native coverage and atomic failure remain enforced; no API gate removal, deployment or automatic-routing completion claimed. See design 17.1 for heuristic limitations.
  Timestamp: 2026-10-15

- [x] PDF-FIX-03: Validate fast/ocr/auto contracts end to end in disposable storage.
  Acceptance: Cover native, scanned, mixed, watermarked bilingual, blank, corrupt and over-budget PDFs; assert fast zero-render/OCR, ocr all-page recognition, auto selective recovery, ordered spans, English retention, mode-specific cache reuse and explicit failures. Use the actual 13-page fixture with runtime versions recorded. Measure extraction separately from language/profile/layout time; no user-artifact writes. Validate fast/ocr first, auto separately with documented limitations; final completion requires all three mode contracts, not all deferred auto-quality enhancements.
  Depends on: PDF-MODE-01..05, PDF-FIX-02, LIB-06 and the delivered PDF-AUTO-01 routing correction. Supporting scripts are not separately counted deliveries.
  Acceptance attempt (2026-10-16): `scripts/accept_pdf_modes.py` ran 21 real-engine mode cases in a read-only, network-disabled Docker container with temporary JSON storage: 19 passed, 2 failed. All 18 synthetic/budget cases passed; the actual 13-page fixture passed fast (163.5 ms extraction, zero render/OCR, warm reuse). Full OCR rejected page 1 after one render/four OCR candidates; auto also returned quality_failed after one render/four candidates. Both failures preserved the previously saved fast hierarchy. Local PDF regression suite: 162 tests passed.
  Historical blocker, now resolved by PDF-AUTO-01: actual-file Auto native-coverage rejection prevented acceptance. Native completeness guards and acceptance expectations were not weakened. See designs 18.5 and 18.7.
  Narrowed diagnostic: Page-1 default/PSM-6/PSM-11 exceed the CJK score threshold but fail fragmented_tokens (short-token ratios 0.9943/0.9957/0.9759); PSM-5 fails fragmented_latin. Validate Chinese OCR word-spacing versus genuine fragmentation before changing the guard.
  Follow-up evidence (2026-10-16): observed-script-v3 excludes CJK/punctuation-only tokens from fragmentation penalties and checks; Latin fragmentation cannot be masked by CJK. Thresholds, encoding/symbol rejection and native coverage remain. Eight quality tests and all 165 PDF regressions pass, including OCR/auto cache invalidation with fast identity unchanged. Real-engine matrix improves to 20/21: fast and OCR retain 13 ordered pages, page-10 English word retention 1.0 and warm reuse. OCR makes 13 renders/52 calls in 258.7 seconds extraction. Auto completes recognition but rejects replacement; a single-page real-engine reproduction confirms page=1:native_coverage_unresolved. No partial hierarchy write or deployment.
  Fail-fast slice (2026-10-16): Coverage validation now runs after each recovered page, including blank/cached results; public failure exposes page/reason and explicit fast/OCR retry options without source text. Actual 13-page Auto stops after page 1 (1 render/4 OCR calls, 21.2 seconds), preserving prior fast persistence instead of processing 13 pages before rejection. Four new tests and all 169 PDF regressions pass. This is failure-path cost/diagnostic improvement, not successful Auto acceptance; no additional completed task. See design 18.6.
  Final acceptance (recorded 2026-10-17): 21/21 real-engine cases passed, including all three modes on the actual 13-page source, ordered spans, English retention 1.0, warm reuse and atomic failure controls. Fast/Auto made zero render/OCR calls; full OCR rendered 13 pages and made 52 recognition calls. All 177 PDF regressions and 18 preparation/acceptance support tests passed. Read-only, network-disabled Docker with temporary JSON and language/profile stubs; no deployment. Runtime and timings: design 18.8.
  Timestamp: 2026-10-17

- [x] PDF-AUTO-01: Improve automatic routing for supported clipping/translucent text.
  Acceptance: In auto only, model supported scoped graphics/clip evidence for the section-17 fixture. Unsupported hidden/masked/Form content stays uncertain; no unknown-equals-zero shortcut. Preserve exact native spans and regression coverage. Existing conservative auto limitations are documented until this task is complete.
  Priority adjustment: Delivered after fast/ocr acceptance, before the final PDF-FIX-03 rerun because its actual-file Auto case exposed this routing dependency. Replaces PDF-FIX-01; no double counting.
  Evidence: `pdf_native_visibility.py` proves supported axis-aligned scoped clips and TrueType glyph containment with bounded fontTools 4.67.0 fallback; unsupported fonts, hidden/masked/Form/image content remain uncertain. Inspection/policy and Auto cache identity are versioned; fast/full-OCR identity and executors are unchanged. Eight new visibility tests plus 177 PDF regressions pass. Isolated Docker Auto preparation of the actual 13-page source succeeds with subprocess calls poisoned: zero render/OCR, exact ordered native spans, English retention 1.0, persisted warm reuse; extraction 692.7 ms, total 1376.1 ms with language/profile stubs. See design 18.7.
  Scope: Only this routing task is checked. PDF-FIX-03 still needs the complete three-mode matrix rerun. No production rebuild/deployment or user-artifact changes; unsupported graphics remain conservative.
  Timestamp: 2026-10-16

## Maintenance Notes

- This checklist is module memory for completed work.
- It does not replace the module detailed design document.
- It does not replace tests and test evidence.
- It does not replace proposal/HLD decisions and governance context.
