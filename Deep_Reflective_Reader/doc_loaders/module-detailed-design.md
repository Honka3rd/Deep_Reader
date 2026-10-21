# doc_loaders Detailed Design

## Reviewable OCR Recovery (2026-10-21)

**[Code-Confirmed]** Partial recognition defects must not
automatically invalidate an otherwise readable document. Target outcomes are
`completed`, `needs_review`, and extraction failure. The reading-content projection
exposes the first two; extraction errors retain their existing contract. Engine/render/budget errors remain failures, not
quality warnings. All-unreadable input remains a failure. Fast stays native-only;
Auto native-preservation rules are not relaxed. Chinese geometry stays loader-local.

Sample2 evidence: two requests fail at physical page 3, region 18, bbox
(32,116,71,970), after 18 accepted regions. The source image contains a running
title, a long dotted leader and vertically stacked page digits 003. Treating that
mixed marginal region as one body column yields empty OCR; this is not proof of
poor source quality. Recover separate text/number components before relaxing
quality. Never discard an entire margin simply because of its location.

Delivery order (single tracking queue in this module; cross-module ownership is
explicit and requires those modules' memory before implementation):

1. OCR-REVIEW-01: component-aware Chinese marginal recognition and real Sample2
   regression; preserve all recognized words/digits and source geometry.
2. OCR-REVIEW-02: actionable page/region failure diagnostics; distinguish engine
   execution failure from recognition rejection without returning source text.
3. OCR-REVIEW-03: generic persisted review evidence and partial-document policy.
   Owners: doc_loaders/document_preparation/document_structure. Store source hash,
   physical page, pixel bbox/render DPI, failed-region reason, raw candidate text,
   trustworthy char spans where available, and policy version. Empty recognition
   has no fabricated text/span; retain an image-backed unresolved region instead.
   Aggregate evidence across every page; no early quality abort hiding later pages.
   Calibrate usable-body coverage on real fixtures, not a global confidence reduction.
   Enable only with the review exposure and interaction guards described below.
4. OCR-REVIEW-04: read-only review projection and source-region image endpoint.
   Owners: main/api_schemas/document_structure and UI. Use the existing repository namespace,
   source hash/page/bbox, bound rendering and return conflicts after replacement.
   Show review-needed state in Library/reader, page navigation and original crop;
   never call OCR/profile/LLM from a read route. Heavy evidence is not task-layout.
5. OCR-REVIEW-05: versioned block correction/delete/undo service and editor.
   Owners: document_structure/section_tasks/main/api_schemas and UI. Preserve the
   immutable original extraction; persist a reading-content overlay and audit trail.
   Use stable server block IDs, expected revision/hash and idempotent mutation ID;
   reject concurrent edits, confirm deletion, bound text to 10000 Unicode code points,
   preserve failed drafts and support explicit undo. Update affected unit/ancestor
   effective content hashes atomically with interaction/ancestor-summary invalidation.
   Preserve user notes separately with stale-source notice, never silently delete them.
   Existing document profile is reused, not regenerated. Reparse explicitly handles
   overlays through the existing destructive-change confirmation, never orphaning them.
6. OCR-REVIEW-06: enforce unresolved-source interaction guards and end-to-end
   acceptance, then enable reviewable imports. Owners: section_tasks/scripts/UI.
   Unresolved regions must not yield invented insight/quiz/thinking/summary; parent
   context must retain incomplete-source status. Verify correction, deletion, undo,
   stale requests and failed writes using backed-up Sample and Sample2 plus all-bad,
   blank, engine-failure, mixed-layout and non-Chinese controls. No production-book
   deletion or silent reparse is authorized by these tests.

This delivery supersedes the old unconditional any-region-rejects-document product
requirement for explicit OCR only. Never ship silent missing text as completed reading.
Historical OCR-DIR completion covers its original fixtures only, not this new track.

### Delivered Review Contract

`ocr_review.py` owns format-neutral region records; Chinese orientation/model rules
remain in the Chinese loader strategy. Explicit OCR defaults to review policy v1;
`DEEP_READER_OCR_REVIEW_ENABLED=0` restores strict rejection. The flag participates
in OCR extraction identity, not Fast identity. Auto retains its strict recovery policy.
Every physical page is accounted for, including confirmed blank pages. Failed
regions retain source SHA-256, zero-based page index, pixel bbox, DPI, reason and
raw candidate when available; empty output gets no fabricated span or content block.
Evidence flows through SourceArtifact.storage_metadata into parse_provenance.source_review.

Partial acceptance requires at least 40 readable characters and accepted region
rectangle area no smaller than uncertain area. A page without region evidence uses
its full rendered rectangle. This is a conservative fixture-tested coverage heuristic,
not calibrated semantic confidence or a guarantee for arbitrary scans. Entirely
unreadable input fails. Execution/render/model/budget failures are never downgraded.

Reading mutations are explicit, versioned overlays in parse_provenance.reading_content.
Edit and Delete are content-block interactive actions in the existing three-dot menu.
There is NO manual add/insert action, API or synthesized block for an empty region.
Source review permits inspecting the original and explicitly ignoring a region;
it does not claim to recover its missing text. An ignored region remains auditable.
Undo restores the immediately preceding edit/delete/review decision, not a multilevel
history. Receipts/audit are bounded to 1000 mutations per structural source epoch;
the limit rejects further writes rather than losing idempotency history silently.

All generated interactions are conservatively invalidated document-wide in the
same mutation transaction because ancestors can affect descendants. Notes survive
with a stale-source notice. Original source, hierarchy IDs and profile are unchanged.
Any unresolved region blocks analysis, quiz, thinking and summary generation for
the document and descendants before context/LLM calls. UI warns before opening
generation; backend guards are authoritative. Review reads never prepare or OCR.

The image endpoint resolves server-owned regions through a loader capability,
verifies current source bytes against the recorded hash, validates bbox/DPI/area,
renders one page and returns a no-store PNG. It accepts no client file path or bbox.
This uses the existing local-app repository namespace, not a newly added auth system.

Acceptance: scripts/accept_reading_review.py uses only the immutable Sample2 backup.
Unmodified real OCR yields 450/0/644 characters. Fault-injecting disabled margin
recovery (not mocked OCR text) yields 450/0/634 plus one unresolved region; the real
crop, four zero-generator guards and edit/delete/discard/undo contracts pass.
Original Sample/mixed-image regression retains 12/17 columns and scoped CER 4.82%/1.85%.
Reports: data/test-books/ocr-review-20261021/reading-review-acceptance.json and
ocr-direction-20261021/review-regression.json. No production Library reparse/deletion.

### First Delivery Evidence

Historical OCR-REVIEW-01/02 baseline: Chinese policy v2 recognizes
margin components only with outer-page location AND a long aligned run of at least
eight small dots separating title from 1-4 digit components. It does not drop whole
margins or treat all edge ink as decoration. The title uses the vertical model;
each stacked digit uses English single-character PSM 10 with bounded PSM 13 fallback.
Existing confidence threshold 40 is retained. Title/digits remain in canonical
text, dotted-rule geometry is recorded, and text/component offsets retain original
page coordinates. The strict-mode baseline rejects unresolved recognition; the
review policy above now handles eligible partial output. Engine nonzero exits cannot masquerade as empty low-quality regions.
Full OCR exposes allowlisted reason/page/region, never source or engine stderr.

Sample2 backed-up source SHA-256 is
05650a15d6a0428d543e4ffee273e5423ecdbcecf14c7ae0f4a02b16d13574de.
All 3 physical pages extracted in 7.50 seconds in isolated Docker; character counts
450/0/644, page 2 rendered blank, page 3 margin exactly `新千年版序 003`.
Acceptance checks representative body text and exact spans, not whole-book CER.
Original Sample retains 12/17 TOC columns and 4.82%/1.85% scoped CER; mixed real-image
header/body still passes. Unit suites: 28 focused; PDF 182 run, 180 pass/two skips
(overlapping suites). Source tree mounted read-only, no LLM/profile/Library mutation,
and no running API rebuild. Immutable private backups and reports live in
data/test-books/ocr-review-20261021 and ocr-direction-20261021.

## Direction-Aware OCR Remediation (2026-10-21)

### Delivered Implementation and Acceptance

**[Code-Confirmed]** Chinese-local routing is implemented in
`chinese_ocr_strategy.py`, called only at the PDF raw OCR boundary. No upper
pipeline language branches were added. Both explicit OCR and Auto recovery use
this entry; Fast does not invoke it or fingerprint its models. Explicit non-Chinese
OCR language configuration bypasses it. Simplified-only configuration selects
chi_sim_vert; configurations including traditional Chinese use chi_tra_vert.
This is a capability/configuration policy, not automatic script identification.

Visual projections locate narrow columns; character horizontal extent, not ink
count alone, retains sparse strokes such as 人. Width-relative thresholds make
column detection stable across header crops/vertical padding. Vertical columns
are read right-to-left; horizontal header/body bands retain top-to-bottom order.
Ambiguous interleaved layouts are not universally supported. The generic horizontal
candidate path remains available; this change is not a universal orientation or
language detector. Rotation is not inferred solely from glyph boxes.

TOC leader gaps are excluded from recognition crops without deleting source pixels;
isolated bottom numeric labels are recognized separately with bounded PSM 7/13
candidates and kept on the same logical line. Vertical bodies use PSM 5; short
labels may use bounded per-glyph PSM 13. Region boxes, logical text offsets, model,
uncalibrated direction evidence/confidence, OCR confidence and quality flags are
retained in page-quality provenance, including mixed-band offset translation.
OCR-run metadata receives these diagnostics through existing SourceArtifact storage;
they do not authorize hierarchy splits or enter task-layout payloads.

Guard thresholds are heuristic: unexplained ink above 5% rejects or enters bounded
mixed-band analysis; at most 64 columns/16 bands; body OCR confidence at least 40;
transcription-size/glyph-height proxy between 0.65 and 2.5. CJK evidence is required
for selected Chinese vertical regions, preventing the reproduced Latin garbage
from winning. Short labels of at most four characters may retain low-confidence
transcription only when their aggregate share is at most 5% of page text, with
explicit low_confidence_short_label flags. This does not certify those labels.
No claim of perfect recognition, arbitrary artistic layout or reliable 180-degree
orientation detection is made. Existing subprocess/time/page budgets still apply.

Docker pins both vertical models to tessdata_fast commit
87416418657359cb625c412a48b6e1d6d41c29bd and validates SHA-256 at build/runtime.
Traditional SHA-256: 1df02a4b210e5c217b783819538b63e9dfe6904e2b5e53b62664f1b9f7a989d0.
Simplified SHA-256: 20590de84725bab69cde93bd6e8ed360a13cc5421a7e7364ddeb93e9af53d6da.
No request downloads are performed. Missing/unverified required models surface
existing engine_unavailable semantics. Extraction identity includes Chinese policy
and model hashes, including ordinary Chinese/English auxiliary models. Region and
selection logs contain geometry, timings and metrics, not source text/images.

Private backup/manifest: data/test-books/ocr-direction-20261021. Source SHA-256:
3a0c9e6acb1b5f4f63e770b724a3b1615303139cfbee0417ed2092a6b1d73571.
`scripts/accept_chinese_direction.py` verifies backup bytes, zero Fast subprocesses,
the horizontal reference phrase, all 12/17 vertical TOC columns and their ordered
numeric associations. The fixed 8% CJK title-error ceiling passes: page 5 4.82%,
page 6 1.85%. This metric excludes decorative headings and enumeration punctuation;
the decorative 目錄 still has errors. A composite real-image horizontal-header /
vertical-body fixture also passes, including region mapping. No source correction
by LLM, production reparse, profile generation or Library mutation was performed.

Auto on this same file intentionally retains native_coverage_unresolved rejection
at page 5: OCR still differs from the native layer. The explicit negative acceptance
checks this boundary and unchanged source bytes; it is NOT an Auto success claim.
Use explicit OCR for a new transcription. Independent 18-case real-engine mode
regressions verify native preservation, warm reuse and failure-preserved storage.
See scripts design for commands, test counts and acceptance limitations. Docker
image build is verified; running API containers require an explicit rebuild/restart
to adopt these changes. Existing accepted data is not silently regenerated.

### Confirmed Failure and Scope

**[Code-Confirmed / Investigation]** `中國文化的深層結構_Sample.pdf` has six pages:
page 3 contains horizontal author information; pages 5/6 contain vertical TOC
columns. Production persisted a completed OCR run at 200 DPI using
`eng+chi_sim+chi_tra`. Page 5 selected default PSM (score 0.7501); page 6 selected
PSM 5 (0.7192). Both passed despite reproducing the reported Latin-like garbage.
These scores are heuristics, not recognition accuracy or calibrated confidence.
Horizontal output also contains transcription errors; it is not a perfect baseline.

The runtime has chi_sim/chi_tra/eng/osd, not chi_tra_vert. Isolated experiments
used an official tessdata_fast chi_tra_vert model in /tmp, without changing service
configuration or Library data. On the same page images, vertical recognition
recovered readable titles but still missed columns/page numbers. A cropped column
with PSM 5 was substantially more readable with chi_tra_vert than chi_tra. Full-page
vertical candidates were rejected as symbol_heavy because of TOC leaders, while
Latin-like garbage passed the non-CJK scoring branch. This demonstrates model and
quality-gate defects, not successful end-to-end acceptance. Temporary experiments
are not a durable fixture or a measured character-error-rate benchmark.

Evidence: pdf_document_loader.py::_ocr_image_text_with_quality/_score_ocr_candidate,
pdf_full_ocr_extraction.py, pdf_page_evidence.py, runtime OCR log request
81984998a3334f3082ea01b2b2554b44 and the matching ocr_runs row; official model:
https://github.com/tesseract-ocr/tessdata_fast/blob/main/chi_tra_vert.traineddata.

At investigation time the flow was fixed language models plus multiple PSM candidates, followed by
text scoring. Existing TSV geometry infers writing mode after recognition and
provides advisory layout evidence; it is not a pre-recognition model router.
Explicit full OCR exposes a text inventory and does not re-enter Auto coordinate
OCR. Earlier completed evidence/PSM tasks do not establish mixed-direction OCR
correctness. The backlog below supersedes any broader historical completion claim.

### Confirmed Target Flow

**[Maintainer-Confirmed / Implemented within the delivered scope above]**

```text
OCR-required page -> rendered-page layout hypotheses
 -> horizontal / vertical / mixed / uncertain, with evidence
 -> suitable region and model/PSM candidates
 -> recognition -> within-column and between-column ordering
 -> transcription, coverage and order checks -> accepted source artifact
```

Writing direction and page rotation are separate properties: rotating a whole
page is not a substitute for recognizing upright vertical Chinese glyphs.
Direction detection is not an irreversible hard classification. Use visual layout
evidence first; when uncertain, compare bounded candidates, retaining their reasons.
Recognition may refine the hypothesis, but corrupted recognized character ratios
must not be the sole evidence for selecting script/direction. A mixed page needs
region-local routing. Do not impose the first page's direction on the whole book.

PDF loader owns direction analysis, models, region recognition and ordering.
Downstream preparation consumes SourceArtifact/ExtractionResult, not PDF-specific
branches. Neither OCR direction nor profile/LLM metadata may authorize hierarchy
splitting. No LLM rewriting or invented recovery of missing source text is allowed.

Mode boundaries remain explicit:
- Fast: native extraction only; no new rendering, direction analysis or OCR cost.
- OCR: rendered-page extraction; apply direction-aware routing to nonblank pages.
- Auto: retain native/OCR page selection; apply the same router only where OCR is
  required. Do not replace native text merely because direction support exists.

### Implementation Contracts and Guardrails

#### Chinese-Local Reading Order Scope

**[Maintainer-Confirmed]** Chinese OCR supports only:
- Horizontal: glyphs left-to-right within a line, lines top-to-bottom.
- Vertical: glyphs top-to-bottom within a column, columns right-to-left.

Mixed pages combine these two region-local orders; they do not introduce a third
Chinese reading mode. Historical right-to-left horizontal text, left-to-right
vertical columns, circular/artistic arrangements and other unusual layouts are
outside the supported scope. They need not be successfully transcribed: when
evidence cannot establish a supported order, report uncertainty/failure using
existing extraction semantics instead of forcing one of the two orders or silently
dropping unsupported regions. Rotation normalization remains separate and does not
expand this typography scope. Unsupported-layout fixtures assert safe rejection,
not universal recognition.

Keep this policy inside the PDF loader's Chinese OCR strategy. Shared orchestration
and geometry/evidence contracts may dispatch strategies, but must not hard-code
Chinese column order, treat every vertical script as right-to-left, or force
non-Chinese regions through the Chinese policy. Other languages retain independent
behavior; do not introduce a general multilingual framework merely for this change.
Selecting the Chinese strategy must not trigger profile generation or make profile
parser authority. Upper preparation/pipeline layers consume the same format-neutral
artifact and must not acquire Chinese-specific branches or configuration semantics.
Fast remains unaffected. Regression coverage must demonstrate this locality as well
as both supported Chinese orders and bilingual region handling.

**[Planned]** Keep page-local source identity, dimensions and region bounding boxes;
record rotation separately from writing mode, hypothesis confidence/reasons,
model/PSM candidates, selected candidate and rejection reasons. For vertical Chinese,
preserve top-to-bottom glyph order and evidence-backed right-to-left column order.
Horizontal regions retain their own order. Treat TOC leaders, labels and page
numbers separately while retaining their source association; do not simply remove
all punctuation or assume every page is a TOC. Missing columns/duplicate spans must
not be hidden by joining plausible fragments.

Pin model versions/checksums and verify availability before recognition. Deploy
models in the runtime image, not by hidden request-time downloads. Preserve explicit
language configuration compatibility and report unsupported/missing required models.
Fingerprint model, routing, normalization and quality-policy versions in extraction
identity so incompatible OCR results are not reused on explicit preparation.
Fast identity/behavior must not be invalidated by an OCR-only policy change.

Quality decisions must combine region coverage, OCR confidence, layout consistency
and bounded text checks. Do not prefer garbage solely for having more letters or
reject readable TOC text solely for leader punctuation. Language evidence remains
advisory: genuine bilingual text must pass, and profile must not be created/rebuilt
to choose an OCR model. Thresholds and uncertainty handling require fixture evidence;
they are not claimed as calibrated here. If no candidate is acceptable, use existing
explicit extraction failure semantics rather than persisting silently partial text.
No new API error names or per-page partial-success contract is approved here.

Reuse rendering within the extraction pass and keep candidate execution within
existing timeout/page/process budgets. Log page/region, direction hypotheses,
model/version, PSM, timings, candidate metrics, selection and rejection reasons.
Do not log source text, images or secrets. Keep diagnostic evidence separate from
task-layout payloads. Existing accepted Library data must not be silently rebuilt
on reads; repaired extraction requires an explicit preparation/reparse lifecycle,
with existing profile/interaction consistency rules, not a loader-side mutation.

### Ordered Acceptance Plan

OCR-DIR-01..07 in the module checklist are the authoritative ordered work queue.
First back up this real source under data/test-books with a manifest/hash and private
ignore rules, then test against that backup. Do not delete the uploaded Library book
for this track. Pin a manually reviewed expected transcription and column/title/page
association for pages 5/6 and representative horizontal text on page 3. Include blank,
rotated, same-page mixed, bilingual and unavailable-model cases as additional fixtures;
the six-page sample alone cannot prove those cases.

Acceptance must test readable text, omission/duplication and reading order, not just
exit code, nonempty output, score or CJK ratio. Record errors against the reviewed
reference, set thresholds before evaluating candidates, and prohibit relaxing them
merely to pass. Real-engine tests must also verify Fast has zero OCR calls, Auto's
native pages stay native, failed extraction leaves accepted data intact, and warm
reuse invokes no OCR/profile rebuild. Model-only experiments are not completion.

## Loader Boundary Governance (2026-10-21)

**[Code-Confirmed]** LOADER-01/02 delivered. AbstractDocumentLoader now requires
open_source/source_path as well as legacy load. SourceSession owns a captured
identity, memoized successful extraction and bounded hash/stat verification;
SourceArtifact wraps ExtractionResult and optional evidence/diagnostics. New
session default is Fast, including omitted internal mode; Auto/OCR must be explicit.
Legacy low-level load() semantics are unchanged. PDF settings fingerprints moved
to doc_loaders/pdf_extraction_identity.py; the former preparation module reexports
its public names only. Generic identities validate label syntax, not a PDF/TXT
allowlist; actual format/mode support is validated by the selected adapter.
PDF evidence DTO names and legacy API fields remain compatibility surfaces, not
format dispatch. Adding a real format still requires factory/API registration;
the synthetic third-format test is not an EPUB implementation.

Maintainer requirement: downstream preparation consumes a digital source artifact,
not a PDF/TXT implementation. Prior EXTRACT completion unified provenance, not
loader behavior. LOADER-01/02 introduce a request-local source session: capture
source bytes and identity without extraction, then extract at most once on cache
miss. The artifact carries canonical text, optional same-source page/outline/layout
evidence and diagnostics; it contains no generated hierarchy/profile/interactions.
Loader adapters own format options, dependency fingerprints, source paths and
legacy PDF provenance/API compatibility. Pipeline owns ordering and persistence.
No mutable session state may be stored on a shared loader. Optional evidence must
remain lazy; Fast must never enter OCR. Source changes before save must reject.
Keep legacy load() and low-level PDF extract() callers compatible; production
preparation uses open_source() for both loaders. Legacy load-only injected callers
remain an explicitly bounded compatibility path, not a model for new adapters.
PDF-named evidence DTOs remain wire-compatible; capability presence, not format,
selects their use. EPUB/DOCX production adapters are out of scope, but a synthetic
third-format session must pass the pipeline without adding format branches.

Evidence: test_loader_boundary.py (seven cases), generic extraction tests, 180 PDF
tests in Docker, and isolated real-book report loader-boundary-20261021T133550.json.
See scripts module for full acceptance and the live acceptance-script incident.

## Format-Neutral Extraction (2026-10-21)

**[Code-Confirmed]** EXTRACT-01..03 are implemented. Shared contracts and ordered
acceptance are defined in [the preparation plan](../document_preparation/module-detailed-design.md#format-neutral-extraction-plan-2026-10-21).
This module owns immutable ExtractionIdentity/ExtractionResult and PDF/TXT adapters.
Keep load()->str compatibility for existing callers while new preparation consumes
the typed result; text must come from one captured extraction, not two loader calls.
PDF retains Fast/OCR/Auto settings and evidence; TXT uses versioned UTF-8/native
semantics with no OCR dependency. Page mappings are optional format capabilities.
Future EPUB/DOCX adapters must satisfy the same contract; implementation is excluded.

## 1. Module Purpose

PDF raw OCR adds request-correlated `pdf_page_render` and `ocr_candidate` stage timings.
Metadata includes one-based page number, render page total/DPI, OCR language/PSM,
process return code and recognized character count. Starts are logged before work;
failures record exception type and elapsed time without OCR text or stderr content.
Existing progress, memory reuse and deterministic quality gates remain unchanged.
**[Code-Confirmed]**

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
8. OCR output reuse is now limited to an explicit preparation scope, not the loader instance lifetime (OCR-07, section 16.11). Context-local last-result properties support the existing structured-store `save_ocr_run` handoff after document creation; they do not authorize reuse in the next scope. **[Code-Confirmed]**
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
5. 已實作多 PSM 分數比較，但不保證直排閱讀順序或正確選出候選；2026-10-21 真實樣本證實 default PSM 亂碼仍會勝出。方向感知模型路由、region-first canonical ordering 與品質修正由 OCR-DIR track 處理。 **[Code-Confirmed] + [Future Direction]**
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

Container/runtime implication：2026-10-21 runtime 已有 Tesseract、Poppler 與 eng/chi_sim/chi_tra/osd；缺少 chi_tra_vert。舊有「未安裝 Tesseract 與中文模型」描述已過時；安裝一般中文模型也不代表直排 OCR 已完成。 **[Code-Confirmed]**

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

## 16. Selective Per-Page OCR (2026-10-14)

### 16.1 Evidence And Scope

The four-page `《歸來》白事／喪葬／鄉村儀式素材整理稿` upload took 68.8 ms;
the subsequent task-layout request took 63,140.3 ms, including 51,352.1 ms in
`pdf_layout_evidence`. Inspection reported 8,187 native characters, zero image
pages and four font-bearing pages. Evidence: `data/logs/deep-reader.log`,
request ids `cf8002287a8f4fda8cfb01e8f3dd9ef2` and
`ee99839b50204673a79bf602c32c90cf` on 2026-10-14. These are one observed run,
not a performance guarantee. **[Code-Observed]**

Before OCR-04, `load_page_layout_evidence` rendered/OCRed pages inside its candidate
limit whenever OCR was enabled, even with native text. Existing cost-tier and
cache-key metadata does not implement semantic candidate selection or durable
cache reuse by itself. OCR-04 routing is now recorded in section 16.8; durable
reuse remains pending. Earlier completed cost-policy items describe metadata only.
Text recovery and coordinate/layout analysis must be decided separately.
**[Code-Confirmed] + [Future Direction]**

### 16.2 Decision Contract

`pdf_ocr_decision.py` implements `PdfPageOcrDecision` as an immutable, internal
metadata contract. OCR-01 is complete; OCR-04 wires layout-only assessment.
**[Code-Confirmed]**

| Field | Meaning |
|---|---|
| `source_sha256` | SHA-256 of PDF bytes, not extracted text; required for source identity |
| `page_index` | Zero-based non-negative PDF page index |
| `purpose` | `text_recovery` or `layout_analysis`; one record per purpose/page |
| `decision` | `skip`, `required`, or `uncertain`; no default success decision |
| `reason_codes` | Non-empty stable codes, not content or exception messages |
| `policy_version` | Required version of the decision rules; independent of DTO version |
| `evidence_schema_version` | Contract version, initially `1` |

`skip` means OCR is unnecessary for that purpose, not that the whole page is
fully understood. `required` describes a need, not authorization to launch a
subprocess. `uncertain` preserves incomplete evidence; it is neither implicit
skip nor unconditional full-page OCR. Config gates, limits and failures remain
execution-layer concerns. Two records may legitimately disagree by purpose.
The contract does not contain an invented confidence percentage, source text,
image bytes, hierarchy fields, a new API response, or persistence behavior.
Enum values and source/page/version validity are enforced; reason/state evidence
consistency belongs to the later decision policy, not this DTO. Hashes normalize
to lowercase; reasons are defensively copied and deduplicated in input order.
Evidence: `scripts/test_pdf_ocr_decision.py` (7 passing tests).
**[Code-Confirmed]**

### 16.3 Cheap Evidence And Conservative Policy

Collect native text once per page; retain it internally for raw-text/boundary
handoff, not logs. Inspect printable/replacement/control-character ratios,
suspicious repetition, native text positions when reliable, font resources,
page dimensions/rotation and image drawing coverage without decoding every
image or invoking OCR/LLM. Coverage is the clipped union of drawn rectangles,
not resource count or summed overlapping image area. Missing/unsupported
geometry stays unknown, never zero; nested forms/inline images and extraction
errors need explicit flags. Character count and font presence alone cannot
certify text quality or completeness. No language-specific ratio alone may
reject legitimate Chinese, Latin, numeric or mixed text. **[Future Direction]**

| Evidence | Text recovery decision |
|---|---|
| Usable native text, adequate evidence of coverage, no unresolved content regions | `skip` |
| Confirmed blank or confirmed non-text page | `skip` with explicit reason |
| Missing/unusable text layer and positive text-bearing visual evidence | `required` |
| Large image without reliable text-bearing evidence | `uncertain`, not automatically a scan |
| Header-only text, mixed native/image content, suspect hidden OCR layer, extraction failure or missing coverage | `uncertain` |

Uncertain pages may receive a bounded low-resolution/text-detection pass.
Only suspect regions should proceed to OCR when geometry permits; otherwise
use bounded page fallback. Unresolved text recovery must surface an explicit
failure rather than publish incomplete canonical text. Optional layout failure
may retain current parser fallback with diagnostic provenance. Thresholds and
budgets must be named, injectable and fixture-calibrated in OCR-03/07; no claim
of perfect classification or fixed millisecond latency is made. **[Future Direction]**

### 16.4 Runtime, Reuse And Safety

- Native text coordinates, rotations and Outline evidence precede coordinate
  OCR. No OCR on an ordinary native-text page merely because it is in the
  first 30 pages. Preserve useful cheap evidence for every page.
- Raw extraction remains `load(doc_name) -> str`; mixed PDFs preserve native
  pages and merge accepted OCR regions without duplicated text, changed page
  order or broken raw-text offsets. Existing low-quality OCR rejection stays.
- OCR-disabled behavior remains explicit; do not turn `required` into enabled.
  Existing force rebuild is not a new force-OCR switch.
- Reuse is bounded to one preparation pass initially. Match PDF hash, page,
  purpose, policy/schema versions and execution settings (engine/renderer
  versions, language, DPI, stage/region/rotation). Unknown identity means miss.
  No resurrection of `data/ocr_text` or new durable cache/schema in this track.
- Logs record request id, page, purpose, decision/reasons, stage elapsed time,
  render/OCR attempts, budget exhaustion and actual cache outcomes; exclude
  page text and binary data. Separate detection need from attempted execution.
- OCR evidence never authorizes chapter/section splitting. The structure
  module retains validation/acceptance; no profile write-back, API change,
  task-layout mutation, LLM detection or UI feature is added by this track.

OCR-04's layout routing is implemented in section 16.8. Remaining runtime changes
above remain **[Future Direction]** until their respective tasks pass; OCR-05's
canonical raw-page routing is now implemented in section 16.9.

### 16.5 Ordered Delivery And Ownership

| ID | Deliverable | Owner | Depends on |
|---|---|---|---|
| OCR-01 | Typed decisions, reasons, serialization and contract tests | doc_loaders | none |
| OCR-02 | Cheap per-page evidence collection and extraction reuse | doc_loaders | OCR-01 |
| OCR-03 | Deterministic three-way policy, thresholds and uncertain-page rules | doc_loaders | OCR-02 |
| OCR-04 | Native-first layout path and selective coordinate OCR | doc_loaders | OCR-03 |
| OCR-05 | Per-page/region raw recovery for mixed PDFs and boundary parity | doc_loaders | OCR-04 |
| OCR-06 | Outline-first preparation scheduling and lazy layout handoff | document_preparation | OCR-05 |
| OCR-07 | Bounded per-pass reuse, execution budgets and decision logs | doc_loaders | OCR-06 |
| OCR-08 | Fixture regressions and real-document latency/quality acceptance | scripts | OCR-07 |

Progress counts these eight IDs once, excluding this governance pass and
supporting tests from extra task counts. Canonical status lives in each owner's
checklist; delivery is one task per user turn. Broader existing layout/TOC
roadmaps remain open: this track does not complete full region detection,
durable caches or every historical layout task automatically.

### 16.6 OCR-02 Implemented Evidence Collection

`PdfDocumentLoader.inspect_pages(doc_name)` now returns immutable
`PdfPageInspection` records from `pdf_page_inspection.py`. It reads one PDF byte
snapshot, hashes those same bytes, and retains each page's exact extracted text
in memory. Native extraction is invoked once per page; all text metrics use
that result, without a second extraction. This is an additive internal entry
point: `load`, `load_pages`, aggregate `inspect`, page-boundary and layout OCR
routing remain unchanged. Cross-call/preparation reuse still belongs to OCR-07.
**[Code-Confirmed]**

Evidence includes zero-based page/source identity, crop box in unrotated PDF
user space, page rotation, font-resource presence, drawn-image count, clipped
image rectangles/union coverage, native text origins, vector-paint presence and
explicit incompleteness flags. Printable/control/replacement ratios, longest
repeated-character run and duplicate-line ratios are script-neutral signals,
not a quality verdict or OCR decision. Empty extraction is distinct from failed
extraction (`native_text=None`, `text_quality=None`). Text-show operators with
empty extraction are flagged, not silently treated as a blank page.
**[Code-Confirmed]**

Geometry uses pypdf's parsed content operations and `Transformation`, without
calling `page.images`, decoding image streams, rendering, OCR or LLM. Axis-aligned
image draws, translation/scale/reflection, quarter-turn transforms, graphics
state save/restore and inline images are supported. The union of crop-clipped
rectangles prevents overlap/repeated draws from inflating coverage. Unused image
resources do not count as drawn images. Coverage describes geometric footprint,
not verified text-bearing area or opaque visible pixels. **[Code-Confirmed]**

Conservative limitations are explicit: Form/unknown XObjects (including nested
Forms), non-axis-aligned image transforms, custom clipping and malformed streams
leave image coverage unknown, never zero. Incomplete inventory also makes draw
count/vector absence unknown. Form-local text coordinates are not reported as
complete page coordinates. Masks/optional image visibility, external graphics
state, annotations, missing font mapping and invisible text modes are flagged
for later policy. Native text origins are not glyph boxes and cannot prove
full native-text coverage. No image/blank/text-bearing classification is added.
**[Code-Confirmed]**

Geometry collection retains at most 512 image rectangles and 2,048 text origins;
exceeding either limit is flagged and cannot produce a complete coverage/position
claim. These bound retained geometry, not total PDF parsing time or source size;
execution budgets remain OCR-07. `to_metadata()` and object repr omit native text;
the returned text is available only for internal handoff. There is no disk cache,
schema/API change, artifact write or source modification. **[Code-Confirmed]**

Validation: `scripts/test_pdf_page_inspection.py` (18 tests),
`test_pdf_ocr_decision.py` (7 tests), existing PDF loader inspection,
page-evidence and Outline regressions passed on 2026-10-14. Fixtures verify
single extraction, unavailable-versus-zero semantics, overlap/clipping, nested
Forms, inline images, extraction failure, bounded geometry, immutable/content-free
metadata and source replacement without stale reuse. These are OCR-02 tests,
not OCR-08 real-document performance acceptance. **[Code-Confirmed]**

The 18 new tests also pass in a disposable Docker Compose container using the
runtime's pinned pypdf 6.17.0 (local venv uses 6.9.2). Source was mounted read-only;
the running API was not rebuilt or restarted and no user document was prepared.
**[Code-Confirmed]**

### 16.7 OCR-03 Implemented Decision Policy

`pdf_ocr_policy.py` implements the pure `PdfOcrPolicy.assess(page, purpose,
visual_check=None)` function surface, returning the existing `PdfPageOcrDecision`.
It consumes OCR-02 evidence without re-extracting text, rendering, invoking a
detector/OCR/LLM, writing artifacts or changing loader/preparation routing.
Need and execution permission remain separate. **[Code-Confirmed]**

Decision precedence:

1. Mismatched visual-check source/page is rejected. Unsupported inspection schema,
   missing/invalid text metrics and recorded extraction/resource/stream failure
   remain `uncertain`, even when a visual check is supplied.
2. Without a visual check, incomplete geometry and unknown/unresolved flags remain
   `uncertain`. Zero native characters, no image draws and no vector paint with
   complete clean evidence permit `skip:blank_page`. An image, however large, or
   vector drawing without text does not prove either a scan or an illustration.
3. Corrupt/control/replacement/repetitive text remains `uncertain`. Large amounts
   of native text do not excuse images or vector content: mixed pages remain
   uncertain. Small text and header/footer-only origins are separately reasoned.
4. Clean native-only evidence with known font resources, sufficient quality/count
   and an interior text origin permits text-recovery `skip`. Layout skip additionally
   requires unrotated, complete native origins with sufficient vertical span.
   A text skip is not automatically a layout skip; origins are not glyph boxes
   and do not authorize hierarchy or reading-order decisions.

All of the above is **[Code-Confirmed]**. Thresholds are an initial synthetic-fixture
baseline, not universal accuracy claims. Valid-looking Unicode with an incorrect
font mapping may escape cheap quality signals; real-document calibration remains
OCR-08. Sparse legitimate pages are uncertain, not rejected as corrupted.

`PdfOcrPolicyConfig` injects named validated thresholds. Defaults:

| Setting | Default / Rule |
|---|---|
| Native non-whitespace count | At least 40 for uncorroborated skip |
| Printable ratio | At least 0.98 |
| Replacement / control ratio | At most 0.01 / 0.0 |
| Longest repeated-character / duplicate-line ratio | At most 0.35 / 0.5; repeat gate applies only at the native count threshold |
| Header/footer margins | Outer 12% each; native origin must be strictly inside |
| Layout origin vertical span | At least 15% of crop height; unrotated, native-only page |
| Uncertain-page probe | At most 1 attempt by default, 96 DPI, 1,000,000 pixels, 5 seconds |

Configuration rejects non-finite/out-of-range ratios and invalid integer budgets.
Probe configuration hard caps are 2 attempts, 120 DPI, 2,000,000 pixels and 10
seconds. Zero attempts disables probing. `policy_version` contains the rule version
and deterministic configuration fingerprint, so threshold changes cannot silently
reuse a differently configured decision. **[Code-Confirmed]**

`PdfPageVisualCheck` is an internal input contract for a future bounded detector,
not an implemented detector, LLM label or user-facing override. It requires PDF
hash/page identity, detector version, full-page coverage and tri-state text presence;
optional `native_text_complete` requires positive text evidence. Partial/unknown
checks stay uncertain. Full-page no-text evidence permits non-text skip only when
native text is empty; disagreement is `conflicting_evidence`. Positive text evidence
with missing/unreliable native text or explicitly incomplete native coverage permits
`required`. Confirmed complete usable native text permits text skip, but cannot
certify missing layout geometry. Contradictory native-complete/empty-or-corrupt data
stays uncertain. The existing OCR quality gate is not bypassed. **[Code-Confirmed]**

`plan_probe` returns limits and one of `probe`, `not_needed`, `disabled`,
`unavailable`, or `budget_exhausted`; it performs no work and never changes the
decision. It requires explicit OCR enablement and detector availability. The
caller must share the source/page attempt count across both purposes, enforce
pixel/time/attempt limits, and preserve uncertainty on failed/exhausted probes.
Actual detector execution/integration remains future work in the routing/budget
tasks; absent a detector, callers use `detector_available=false`, not fabricated
visual evidence. Future cached assessments must include visual-check contents and
detector version as well as source/page/config identity. **[Code-Confirmed] + [Future Direction]**

Evidence: `scripts/test_pdf_ocr_policy.py` (22 tests) covers the decision matrix,
source/flag/failure conflicts, multilingual and numeric pages, exact threshold
boundaries, configuration identity, bounded probe planning and synthetic PDF
inspection handoff. Local and isolated Docker runs pass; OCR-01/OCR-02 tests and
existing loader inspection also pass. No API restart or performance claim.
**[Code-Confirmed]**

### 16.8 OCR-04 Implemented Native-First Layout Routing

`load_page_layout_evidence` now takes one byte snapshot for native inspection and
source hashing, collects each page once, and assesses `layout_analysis` through
OCR-03. It always emits page-local inventory, including native text, source hash,
rotation, drawn-image count and content-free decision/reason/policy tokens in the
existing `evidence` list. No word boxes or reading order are invented from native
origins. An unknown image count is represented by the existing integer field's
zero placeholder plus mandatory `image_count_unknown` evidence; it is not proof
of image absence. No API/schema/persistence contract is added. **[Code-Confirmed]**

Execution gates are distinct from the need assessment:

- OCR must be explicitly enabled and the page must be inside the candidate prefix.
  `None` retains the historical all-pages prefix; zero means no OCR, and negative,
  boolean or non-integer limits are rejected. The preparation caller still uses
  30. A skipped native page does not move the prefix forward to later pages.
- Policy `skip` returns inventory without image decoding, rendering or Tesseract.
- Without a detector, `uncertain` is not promoted to `required` or `skip`.
  Only drawn-image, unresolved Form/XObject or rotated-page candidates retain
  the optional legacy coordinate pass under the explicit gates above. This is
  bounded candidate eligibility, not confirmation that an image contains text.
  Native-only/vector-only unresolved pages retain inventory and
  `layout_uncertain_inventory_fallback`; raw-text recovery is unchanged.
- Missing Tesseract is detected before rendering. Selected pages render first;
  embedded images are decoded only as a renderer-failure/empty-render fallback.
  Rendered-image header dimensions supply the pixel coordinate frame for OCR;
  inventory stays in PDF page units. Embedded fallback retains image dimensions.
  Existing PSM 11/5 selection and vertical layout analysis remain in place.
- Renderer/image decoding/optional OCR errors or empty OCR keep the page's native
  inventory with a failure reason, without discarding later pages. Successful
  image fallback retains renderer-failure provenance. These are optional layout
  fallbacks, never successful or complete raw-text recovery claims.

All above is **[Code-Confirmed]**. Actual visual probing, cross-call reuse, stronger
total execution budgets and actual attempt accounting remain OCR-07; Outline-first
scheduling is implemented by OCR-06 (section 16.10). Existing cache keys are
descriptive, not cache hits.
Native-only uncertain pages can lose optional coordinate enrichment, but the
policy uncertainty and existing native/parser fallback remain explicit. This task
does not certify detection accuracy or fix mixed raw extraction (OCR-05).

Evidence: `scripts/test_pdf_layout_routing.py` passes 14 tests locally and in an
isolated Compose container on 2026-10-14. Coverage includes zero/disabled/prefix
limits, single native extraction, poisoned render/image/OCR paths, native-only
uncertainty, horizontal/vertical/rotated/mixed candidates, missing engine, timeout,
empty OCR, renderer/decode failure, image fallback, pixel dimensions and unchanged
raw/boundaries.
Earlier decision/policy/inspection, loader, page-evidence, Outline and OCR-layout
fixture regressions also pass. **[Code-Confirmed]**

Read-only local smoke of the four-page `《歸來》` PDF (SHA-256
`1d0f1b559d73bb1a533924d7fc3a3aa2c3748f8aeb55886adde2b36794ed0862`) retained 8,187
native characters, returned four inventory records and made zero render/OCR calls
with OCR enabled and candidate limit 30. Elapsed layout-method time was 346.1 ms
on this run. All pages remain `uncertain` due to `graphics_state_uninspected`,
not falsely `skip`. Subprocess/render paths were poisoned to verify zero calls.
This is not a comparable end-to-end benchmark against the earlier container's
51-second layout stage; OCR-08 still owns cold/warm acceptance. No user artifact
was rebuilt, and the running API image was not rebuilt/restarted. **[Code-Observed]**

### 16.9 OCR-05 Implemented Canonical Page Recovery

`load`, `load_pages` and `load_page_boundary_evidence` now share
`_load_canonical_pages`. One byte snapshot supplies the reader/source hash, and
one native inspection per page supplies the raw candidates. Aggregate scanned-PDF
ratios remain available through `inspect`, but no longer decide whether individual
pages disappear from a mixed PDF. **[Code-Confirmed]**

Routing and integrity rules:

- Text-recovery `skip` preserves the exact native string. For otherwise uncertain
  native-only pages, a complete image-free stream inventory, usable text quality,
  font resources and no unresolved visual content also permit native preservation.
  Short/title-only text is not missing text on such a page. Missing/bounded text
  coordinates alone do not force raw OCR; raw extraction is not layout analysis.
  Other inspection failures/unknown content remain unresolved.
- A verified whole-crop rectangle fill before any text is separately marked
  `has_only_page_background`; it does not turn all vector artwork into native-only
  content. Partial rectangles, arbitrary paths and overlays after text cannot use
  this exception. `has_vector_paint` remains true for diagnostic fidelity.
- ExtGState inspection recognizes only default opaque alpha (`ca`/`CA` = 1) with
  normal blending and optional type; unknown keys/transparency remain flagged.
  Duplicate-line quality ignores single-glyph lines, which pypdf can emit for CJK.
  Long repeated-character and repeated multi-character-line gates remain active.
- Other suspect pages enter a recovery trial only with explicit OCR enablement
  and inside the configured physical-page prefix. This does not relabel uncertain
  images as confirmed text. Disabled OCR raises `RawTextRequiresOcrError`, even
  when other pages contain native text. Zero or an insufficient page limit fails
  before rendering; partial-document success is no longer allowed.
- Current origins/image footprints cannot establish safe glyph coverage/order for
  region merging. OCR-05 therefore uses the permitted full-page fallback, requiring
  exactly one complete rendered image per selected page. Accepted OCR replaces the
  suspect page, never concatenates with its native layer. Other native pages remain
  byte-for-byte identical as Python text strings.
- Usable native alphanumeric content must remain a contiguous subsequence after
  whitespace/punctuation removal and case folding in the recovered page; otherwise
  recovery fails with `native_coverage_unresolved`. Image-bearing pages that merely
  repeat the native header also fail. This conservative guard can reject legitimate
  interleaved or reordered text; it does not guess missing region order.
- Each selected nonblank page must pass the existing OCR candidate quality gate.
  A successful page cannot hide an empty/low-quality later page under the old 35%
  aggregate threshold. A full rendered page with identical opaque RGB pixels may
  be confirmed blank without Tesseract; empty OCR alone is never blank evidence.
  Alpha/palette/transparency-bearing images do not use the uniform-render shortcut.
- Failures clear published OCR state; no partial pages/provenance are exposed.
  Timeouts/process errors have typed raw-load failures, and source changes during
  recovery invalidate the result. Pure-native loads clear prior document OCR state.

All above is **[Code-Confirmed]**. These are loader-level safety decisions, not
chapter/section authority. No LLM, UI/API change, profile write or DB migration is
introduced. Illustrations and other nonuniform pages without reliable text-bearing
evidence can still produce explicit unresolved failures; a general visual detector,
region merging, fuller budgets and calibrated accuracy are not claimed complete.

Canonical offsets:

- `load_pages` now returns exact native whitespace, rather than independently
  stripping pages. `load` joins nonempty canonical pages with `\n` if the original
  PDF has native text, otherwise `\n\n` for recovered scans.
- Every physical page retains one boundary, in source order with its original
  label/hash. Empty pages have zero-length spans and add no separator. Every span
  must slice the exact text returned by `load`, including recovered pages after
  leading/interleaved/trailing blanks. Native pages are not substituted back into
  recovered page slots during boundary construction.

OCR handoff keeps original page indexes: `last_ocr_pages` contains OCR text only,
with empty slots for native/unprocessed/confirmed-blank pages. It must not label
preserved native strings as Tesseract output in the existing OCR-run store.
Canonical merged text exists separately in the returned page list. Provenance
schema 3 adds `recovery_page_indexes`, `raw_recovery_version` and policy identity;
`last_ocr_quality` contains selected-page evidence, including uniform blank proof.
Existing one-loader memory reuse matches this provenance; no durable cache or
broader inspection cache is added. OCR-07 still owns expanded reuse/budgets/logs.
**[Code-Confirmed]**

Validation on 2026-10-14: `scripts/test_pdf_raw_recovery.py` passes 25 tests locally
and in isolated Compose, with renderer/OCR seams mocked for deterministic coverage.
Earlier policy (22), inspection (18), layout routing (14), decision (7), loader,
page-evidence, OCR-layout fixtures, upload/preparation logging, manual structure
handoff/source evidence and Outline regressions pass. **[Code-Confirmed]**

The Outline fixture `许三观卖血记` has a raster cover (page index 0) and unresolved
trailing page (123). Its regression now asserts disabled recovery fails, then mocks
only those recovery slots; it does not claim real OCR quality. Recovering the cover
removes the old empty-page offset collision, preserving the front-matter chapter
before chapters one and two under the unchanged builder. The historical one-page
OCR-budget test now asserts rejection rather than success with a truncated book.
**[Code-Confirmed]**

Read-only `《歸來》` smoke preserves 8,187 native characters (8,200 including original
whitespace and joins), all four page strings and exact raw-span parity with OCR
disabled and poisoned subprocess calls. No user artifacts or live API image were
changed. This is correctness evidence, not OCR-08 real-scan or full preparation
performance acceptance. **[Code-Observed]**

### 16.10 OCR-06 Preparation Source Handoff

`load_with_page_boundary_evidence(doc_name)` returns canonical raw text and
`PdfPageBoundaryEvidenceResult` from one `_load_canonical_pages` call. Preparation
passes these spans onward instead of extracting/recovering the PDF again for
boundaries. Existing `load`, `load_pages` and manual boundary entrypoints preserve
their contracts; `load_page_boundary_evidence` delegates to the combined method.
Outline reading also uses one byte snapshot for the reader and source hash.
**[Code-Confirmed]**

No retained inspection cache is introduced. A preparation call owns its raw/spans;
fallback layout can still inspect pages again, and reuse/budget work remains
OCR-07. Structure-owned validation decides whether the lazy layout provider is
needed and rejects mismatched evidence hashes. Required raw recovery still precedes
Outline acceptance, so bookmarks never bypass incomplete scanned body text.
Evidence: `scripts/test_prepare_lazy_pdf_layout.py` passes 15 tests locally and in
isolated Compose; all 86 PDF unittest cases and existing loader/page/Outline/TOC
script regressions pass. This supports OCR-06 in the preparation checklist only,
without claiming OCR-07 or OCR-08 completion. **[Test-Confirmed]**

### 16.11 OCR-07 Bounded Execution And Evidence Reuse

`pdf_execution_pass.py` owns `PdfExecutionLimits`, `PdfExecutionPass`, the internal
`PdfBudgetExceeded` signal and the scoped loader decorator. `DocumentLoaderFactory`
opens a loader scope around preparation and explicit manual-source reads. Nested
loader methods share that scope; standalone public calls open/close their own.
`ContextVar` state isolates concurrent worker requests, including OCR provenance,
page text and quality handoff. Exit always clears the evidence cache, even on
failure; next-pass results cannot reuse the previous `last_ocr_*` snapshot.
No disk cache, DB table, API field or parser authority is introduced.
**[Code-Confirmed]**

The LRU cache retains at most 512 entries and 16 MiB of recursively estimated
entry storage. Oversize/disabled entries are not retained; values are defensively
copied. This bounds retained reusable evidence, not total PDF parsing/rendering
memory or caller-owned canonical text. Native inspection is shared across raw,
boundary and lazy-layout consumers using source SHA-256/page/inspection-version
identity. Policy decisions additionally include purpose, policy version and
evidence schema. Only successful full-page layout and complete raw recovery
outputs are reusable. Required recovery failure clears evidence, but does not
reset consumed execution budgets. **[Code-Confirmed]**

Execution keys include source/page(s), purpose, policy/schema and implementation
version, full-page stage/rotation, OCR enablement, language, render DPI and command
timeout. Raw keys also include recovery indexes, page limit, doc identity and raw
recovery schema; layout keys preserve source filename identity.
Successful coordinate evidence exposes a `pdf-layout:v2` digest of the full
execution key (or no key for unknown identity), not the older descriptive alias.
Engine/renderer build identity uses resolved executable path plus binary SHA-256, memoized only
against matching size/mtime/ctime during the scope. Unknown executable/rotation
identity is a miss. This is binary build identity, not a new version subprocess;
existing Tesseract version provenance is still queried with a bounded metadata
command. Language names are keyed, but language-pack files/shared libraries are
not fingerprinted: runtime installation is assumed stable during this short
synchronous scope. Region merging remains unsupported; embedded-image fallback
is not cached as a successful full-page render. **[Code-Confirmed]**

Constructor-injected defaults (`execution_limits=PdfExecutionLimits(...)`):

| Purpose | Render attempts | OCR attempts | Elapsed execution window |
|---|---:|---:|---:|
| Required text recovery | 1024 | 4096 | 1800 seconds |
| Optional layout analysis | 30 | 120 | 60 seconds |

Metadata commands have a separate four-attempt limit per purpose and share that
purpose's time window. Zero means no attempts/time, never unlimited. Existing
physical-page prefix gates still apply before reuse; force rebuild is not force
OCR. Windows begin at the first external command for that purpose. Each real
subprocess attempt, including failures, is counted; its timeout is capped by the
remaining window. Late results are rejected. Text/layout budgets are independent
so optional work cannot consume the required-text allowance. Native extraction,
file hashing, Python geometry work and LLM/profile calls are not hard-deadline
bounded by these subprocess limits. No full preparation latency SLA is claimed.
**[Code-Confirmed]**

Required exhaustion maps to existing `RawTextOcrFailedError` with stable detail
`text_recovery_{render|ocr|metadata|time}_budget_exhausted`, and no partial text or
hierarchy is saved. Optional exhaustion preserves native inventory for every page
and records `layout_analysis_{render|ocr|time}_budget_exhausted`, allowing existing
structure-owned fallback. Counts reflect executed commands, not planned needs.
**[Code-Confirmed]**

Request-correlated logs now include `pdf_page_decision`, `pdf_page_gate`, native
inspection duration, actual `pdf_evidence_cache` hit/miss/unknown-identity outcomes,
`pdf_process` purpose/stage/page/attempt/duration, `pdf_budget_exhausted` and a
`pdf_pass_finished` count/cache-size summary. They omit page/OCR text, image bytes,
command output and exception messages. Existing request-id filtering remains in
use; execution/cache records also carry request id explicitly. **[Code-Confirmed]**

Evidence: `scripts/test_pdf_execution_pass.py` adds 25 tests. All 111 PDF unittest
cases pass locally and in isolated Compose on 2026-10-14, plus the 15 lazy-layout
tests and 11 standalone PDF/TOC/preparation/manual/logging scripts locally. Tests
cover cache eviction/disabled/unknown identity, source/binary/policy/settings
invalidation, actual call counts, zero/failed/time budgets, late-result rejection,
required failure atomicity, optional fallback, thread isolation, cleanup, factory
integration and private-content exclusion. Existing memory tests now explicitly
open a preparation scope; timeout fixtures expect budget-specific reasons when
the window is exhausted. Synthetic OCR outputs are fixtures, not scan-quality
acceptance. No live API restart or user artifact mutation; OCR-08 remains open.
**[Test-Confirmed]**

## 17. Native Watermark And Mixed-Language Regression Investigation (2026-10-15)

Investigation only; the original correction order below is superseded by section 18.
PDF-FIX-01 is replaced by deferred PDF-AUTO-01; OCR-01..08 stays historical. Source:
`20250515-为什么中式思维既不产生科学，也不产生民主.pdf`, 13 pages, SHA-256
`87d7685e1b95d8842fc749e15b908b631d4cc5def5afcf7a9a02d7011e9ecc83`.
Read-only inspection with the running Docker pypdf extracts 19,157 native
characters, including 2,553 on physical page 10. Local and container pypdf versions
produce slightly different totals; use runtime/version-specific baselines rather
than treating a cross-version count difference as new data loss.

All pages have native text and zero observed image draws, but
`custom_clipping_unresolved` and `graphics_state_uninspected` make coverage unknown.
The inspected first/tenth pages have no XObject resources; content contains `W`
clipping and `gs` state, with `/Gs1` fill alpha `/ca=0.4684`. Rendering page 10
confirms readable English text and pale text watermarks, not a scanned/garbled page.
Inspection deliberately does not yet resolve these graphics states; the policy
returns uncertain and `_native_only_text_complete` rejects these flags. The raw
execution path consequently schedules all 13 pages for full-page recovery.
This is an overly conservative routing limitation, not proof OCR was necessary.
Do not simply equate unknown image coverage with zero or globally ignore clipping.
**[Code-Observed] + [Code-Confirmed]**

Log request `a0a2d1a52f214e8d8a27be5b5fff35ce` took 213,905.9 ms and returned
HTTP 400. It executed ten renders and forty OCR calls before rejecting page 10;
pages 11..13 were not reached. Raw load consumed 213,847.0 ms. Page 10 OCR engines
exited zero and produced substantial text, but the selected PSM 5 candidate had
score -0.0889, `low_cjk_ratio,latin_heavy,quality_rejected`. This is quality
classification failure, not engine crash, timeout or absent text. No structure,
language or profile readiness was published. UTC-like container log timestamps
span 2026-10-14 16:48:49..16:52:23 (local investigation date 2026-10-15).

Second root cause: `_score_ocr_candidate` switches to a Chinese-weighted formula
whenever the configured engine language string contains `chi`. `eng+chi_sim+chi_tra`
is therefore treated as an expectation of Chinese on every page. Even feeding the
legible native page-10 text into that scorer diagnostically yields -0.1954 / failed.
Engine language availability is not page-language truth; a bilingual book can
legitimately contain English-only pages. No document profile or LLM should become
parser authority to solve this. **[Code-Confirmed] + [Code-Observed]**

Original investigation proposals (priority/scope superseded by section 18):

- PDF-FIX-01: Track supported clip/graphics state with q/Q scoping, distinguish
  ordinary translucent text watermark/background from hidden or masked content,
  and prove native-text completeness before skipping recovery. Keep unsupported
  Forms, masks, invisible text and unresolved regions uncertain. If necessary use
  a bounded diagnostic fallback, not an unconditional all-page OCR policy. Preserve
  native bytes/whitespace and exact page spans when accepted.
- PDF-FIX-02: Make OCR quality script-neutral or based on defensible page evidence,
  not the presence of a language pack. Cover Chinese, Latin, bilingual and numeric
  pages; retain rejection of gibberish, empty/fragmented output and missing native
  coverage. Prefer sound native text over degraded recovery only when coverage is
  established. Do not lower one global threshold to make this fixture pass.
- PDF-FIX-03: Add synthetic transparency/clipping/multilingual regressions plus a
  disposable actual-source pipeline smoke. Assert no unnecessary OCR on supported
  native pages, all 13 ordered page spans and English pages preserved; include
  negative scanned/hidden/masked cases, budget exhaustion and no partial writes.
  Record call counts and timings separately from text accuracy. The existing
  111-test matrix did not establish real watermarked bilingual PDF correctness.

No full OCR/prepare was rerun in this investigation. Only read-only native
inspection and one diagnostic page rendering were executed; no source, user
artifacts, runtime code or thresholds were changed.

### 17.1 PDF-FIX-02 Quality Correction (2026-10-15)

Implemented after the investigation: candidate scoring now chooses the existing
CJK-weighted formula only when observed candidate CJK/signal ratio is at least
0.35. Otherwise it uses the existing Latin/numeric formula. Installed OCR language
packs no longer affect scores or generate low_cjk_ratio/latin_heavy rejection.
The existing score thresholds (0.28/0.22) are unchanged; no profile/LLM language
classification is consulted. This is a deterministic text heuristic, not proof
of recognition accuracy or semantic correctness.

To avoid admitting the existing Latin-looking OCR garbage, non-CJK candidates
with at least 10 Latin tokens and over 60% tokens of length 1-2 are rejected.
Symbol-heavy (>0.35), fragmented (>0.82 short tokens with at least 20 tokens),
replacement/control-character output and fewer than four signal characters are
also rejected. The existing too_short (<20 non-space characters) diagnostic stays
advisory so useful short labels such as Memory OCR text remain valid. Unusual
abbreviation-heavy tables may need future calibration; arbitrary long-word
gibberish and other script families are not claimed solved by this heuristic.

Candidate selection and page-quality summaries both rank passing candidates
before score. The quality policy version observed-script-v2 is included in
execution cache identity, page-quality summaries and OCR provenance. Native
coverage checks, required-page failure atomicity, routing and budgets are unchanged.
No public API gate is removed and no service restart/deployment was performed.

Evidence: `scripts/test_pdf_ocr_quality_scripts.py` has 5 passing tests covering
language-pack invariance, English/Chinese/mixed/numeric positives, short labels,
empty/fragmented/symbol/encoding negatives, candidate-selection consistency and
the local source's English physical page 10. That last test scores extracted
native text as a diagnostic, not actual OCR transcription; it skips if the private
fixture is unavailable. Existing loader inspection, raw recovery (25), execution
pass (25) and fast extraction (9) regressions pass. Whole-document real-engine
acceptance remains PDF-FIX-03; automatic graphics routing remains PDF-AUTO-01.

## 18. Explicit PDF Extraction Modes (2026-10-15)

Current PDF-MODE-05 status: fast/ocr/auto now dispatch from `extract` and are wired
to explicit preparation. The temporary 501 gate described in earlier delivery
notes is superseded. Auto uses `_extract_auto`, a snapshot wrapper around the
existing selective executor; it does not use fast or full-OCR dispatch. Its legacy
OCR configuration/policy limits remain. `source_bytes` lets preparation bind
identity and extraction to the same immutable input. Per-result optional providers
preserve that snapshot. Real-engine/source acceptance remains PDF-FIX-03.

PDF-MODE-02 implements the independent fast executor at the explicit loader entry;
PDF-MODE-03 adds the full-page OCR executor. Auto at that entry and preparation
integration remain pending.
PDF-MODE-01 implements the shared
IO-free enums/errors in `pdf_extraction_contract.py` and the API contract, with a
preparation gate that rejects PDF requests before legacy extraction/cache reuse.
This supersedes automatic classification as a prerequisite for every PDF. Contract `pdf_extraction_mode`
values are `fast` (default), `ocr`, `auto`, independent of base/free_qa and
`structured_parser_mode`. Existing selective OCR remains the auto implementation
baseline, not the default; historic section-16 completion does not prove this API.

| Mode | Required behavior | Failure/coverage semantics |
|---|---|---|
| fast | Native extraction and page spans only; no complex OCR-decision inspection, rendering or OCR anywhere in the preparation path | Cheap empty/corrupt checks; actionable no-text failure, empty-page warnings; no claim of visual completeness, no automatic escalation |
| ocr | Render/recognize every page; accepted OCR supplies canonical text | Quality checks, legitimate blank-page handling and resource limits remain; no partial/truncated success or silent native fallback |
| auto | Existing conservative page inspection, native reuse and selective recovery | Unknown evidence remains uncertain; mixed PDFs supported subject to documented routing limits |

Fast may still read all pages to extract text; it avoids extra classification and
external commands, not all per-page work. Native Outline and optional native layout
may be used without introducing the complex OCR decision path. A lazy optional
layout provider must respect fast too; rejecting OCR in raw loading alone is insufficient.
Language/profile generation remains separate and may dominate overall preparation.
Mixed content can be omitted in fast: empty-page warnings are useful, not proof that
all image text was detected. No silent fast-to-auto/ocr fallback is permitted.

Pure OCR simplifies routing, not recognition quality. PDF-FIX-02 remains mandatory
for legitimate English under Chinese-inclusive language packs. Engine failure,
budget exhaustion and ambiguous nonblank output stay explicit. Optional TOC/layout
failure may fall back to the parser, but required canonical text failure may not.

Delivery ownership: main PDF-MODE-01 (public contract), loader PDF-MODE-02/03
(fast/ocr), preparation PDF-MODE-04/05 (mode-aware reuse and propagation), UI
LIB-06 (selector/retry), loader PDF-FIX-02/03 (quality/acceptance). PDF-AUTO-01
replaces the old PDF-FIX-01 and is last priority. Auto initially retains known
watermark/graphics limitations; do not label those fixed by adding a selector.
Source fingerprint, mode and extraction settings/version must isolate canonical
text provenance/cache reuse; preparation owns downstream invalidation. No profile
or LLM becomes parser authority; no read-side hidden rebuild is authorized.

### 18.1 Execution Boundary Requirement (2026-10-15)

The explicit `extract` entry now implements fast/OCR dispatch; auto fails with
mode_unavailable until their executor integration is delivered. The legacy
`_load_canonical_pages` still inspects/assesses pages and checks `ocr_enabled`;
setting that flag false is NOT the new fast path.

Resolve/validate the extraction enum once at the loader entry, then select exactly
one implementation through a small dispatch table or exhaustive switch:

```text
validated mode -> fast native extractor
               -> full-page OCR extractor
               -> automatic selective extractor
```

Use three named functions/modules with the same canonical-page result contract
(ordered text, page spans, provenance, diagnostics). No inheritance/plugin framework
is required. Fast does not call the auto executor with OCR disabled; full OCR does
not call auto with forced decisions. Only auto owns native-versus-OCR decisions.
Share source snapshots, native extraction primitives, rendering, OCR commands,
quality scoring, budgets and page-result assembly; those helpers must not select
a mode, perform implicit fallback or depend on global mutable mode flags.

Optional layout/evidence work receives the selected strategy's explicit provider
or capability at the same boundary. Fast binds native-only evidence, never an
OCR-capable fallback. Preparation forwards immutable mode and consumes results;
it must not duplicate per-page routing through nested mode/ocr_enabled checks.
Mode belongs in cache/provenance identity, not scattered flow-control predicates.

PDF-MODE-02/03 acceptance must include dependency-spy/poison-call tests: fast cannot
call inspection/render/OCR; full OCR cannot call the auto decision policy; auto
alone invokes classification. Each path is independently testable, and cross-mode
calls are limited to shared primitives, not one executor invoking another. Normal
validation/blank-page/quality branches remain allowed inside their owning path.

### 18.2 Native Fast Executor Delivered (2026-10-15)

`PdfDocumentLoader.extract(doc_name, mode=fast)` validates once and dispatches to
`pdf_fast_extraction.extract_fast`. OCR/auto selections fail before source I/O.
The legacy load/load_pages/boundary/layout APIs retain their prior behavior; this
slice does not switch preparation or bypass its API 501 gate. PDF-MODE-04/05 own
mode-aware cache identity and end-to-end propagation. No live deployment occurred.

Fast reads a source snapshot, hashes it and uses pypdf native extraction once per
physical page. It does not inspect graphics, classify OCR need, decode images,
render, run subprocesses or consult OCR quality/language policy. Text is stripped
per page, joined with a newline between nonempty pages, with exact ordered spans
including zero-width empty-page spans. Cheap encoding checks report replacement/
control characters, reject entirely unusable encoding, and do not judge language
or semantic quality. Entirely empty output returns no_text with an explicit retry
suggestion; partial page extraction failure/corrupt/encrypted input returns
extraction_failed, never partial success or automatic escalation.

`PdfExtractionResult` carries text/pages/boundaries, effective mode, extraction
version, zero-based empty_page_indexes and warnings (page messages are one-based).
Source SHA-256 is carried in boundary evidence. No persistent cache is added.
Native-only extraction cannot establish visual completeness: scanned pages in a
mixed book may be empty, and a watermark-only layer can still be returned. Every
success records native_only_visual_coverage_unverified, not proof of full coverage.

The result binds lazy layout/Outline providers to the same reader snapshot. Native
layout returns text/basic page dimensions with unknown image inventory, no OCR
words/regions, zero OCR passes and no high-cost permission. The existing numeric
image_count field uses 0 with image_count_unknown, not a confirmed absence of images.
Geometry failure records unavailable evidence. Native Outline reuses the existing
bookmark validator; missing/unusable bookmarks allow the existing parser fallback.
Neither provider can enter the auto layout API or re-read a changed source file.

Evidence: `scripts/test_pdf_fast_extraction.py` (9 cases) poisons inspection,
classification, auto extraction/layout, image decoding, rendering and subprocess
dependencies. Fixtures cover native, watermarked, mixed, scan-only, blank, corrupt,
encrypted, page failures, encoding, span parity and deferred snapshot consistency.
Existing raw recovery (25), layout routing (14), lazy preparation (15), mode
contract (11) and native Outline regressions also pass. No whole-book latency or
production OCR integration claim is made.

### 18.3 Full-Page OCR Executor Delivered (2026-10-15)

`extract(mode=ocr)` dispatches to `pdf_full_ocr_extraction.extract_full_ocr`,
independently of the legacy ocr_enabled flag. It never calls native text extraction,
inspection, auto decisions, selective recovery or automatic coordinate layout.
The source bytes captured at entry are written to a temporary PDF snapshot for
rendering; returned text, spans, source hash and native bookmarks refer to that
snapshot even if the original file changes. No user artifacts are written.

Every physical page is rendered. Uniform rendered blank pages retain empty spans;
other pages run the existing bounded four-PSM recognition and observed-script-v2
quality scoring. Only accepted OCR supplies canonical text, never native fallback.
Missing executable/language packs fail as engine_unavailable before rendering;
page limits or execution budgets fail as budget_exceeded. Invalid render/process
output fails as extraction_failed, rejected nonblank recognition as quality_failed,
and all-blank output as no_text. Errors never return a partial result. Required
render/OCR/metadata subprocesses use existing request-local attempt/time budgets.

Result provenance records mode, source hash, page count, OCR language, DPI and
quality version; per-page quality is returned on the result, not published to
legacy last_ocr fields or persistence. The lazy layout provider is text inventory
only with unknown geometry/image inventory, no coordinate OCR or extra rendering.
No persistent mode cache is introduced. Snapshot size/native parsing limits and
real transcription accuracy remain subject to later integration/acceptance.

Seven new isolated tests poison native/auto dependencies and cover all-page
recognition with legacy OCR disabled, blanks, unavailable engines/languages,
page limits, mid-document rejection, budget failure and source replacement.
Fast (9), quality (5), raw recovery (25), execution pass (25) and API contract (11)
regressions pass. These use synthetic OCR outputs, not real-engine accuracy
acceptance. PDF-MODE-04/05 still own preparation/cache integration; public PDF
prepare remains 501. No live deployment or automatic-mode completion is claimed.

### 18.4 Three-Mode Acceptance Attempt (2026-10-16)

PDF-FIX-03 remains incomplete. `scripts/accept_pdf_modes.py` exercises real
extraction, structure building and temporary JSON persistence, with deterministic
language/profile collaborators instead of LLM requests. The source repository is
mounted read-only, the container has no network, and all generated artifacts are
temporary. This is not live API, PostgreSQL or LLM latency acceptance.

Runtime: Python 3.11.17, pypdf 6.17.0, Tesseract 5.5.0, pdftoppm 25.03.0,
eng+chi_sim+chi_tra, default 200 DPI. The 13-page input is
`20250515-为什么中式思维既不产生科学，也不产生民主.pdf`, SHA-256
`87d7685e1b95d8842fc749e15b908b631d4cc5def5afcf7a9a02d7011e9ecc83`.

All 18 synthetic/budget cases pass: native, scanned, mixed with blank and Outline,
all-blank, corrupt, and zero OCR budget, each in fast/ocr/auto. Assertions cover
ordered exact spans, known scan transcript, explicit errors, unchanged persistence
on failure, mode-switch invalidation and warm reuse without extraction/profile
generation. Fast native parity includes the documented per-page whitespace trim;
fast mixed files intentionally do not recover image-only pages.

Actual-file results from the final run:

| Mode | Extraction ms | Total ms | Extraction render/OCR calls | Result |
|---|---:|---:|---|---|
| fast | 163.5 | 431.9 | 0 / 0 | 13 pages, 19,168 chars, warm reuse passes |
| ocr | 20,966.7 | 20,972.4 | 1 / 4 | quality_failed: page 1 |
| auto | 21,372.0 | 21,399.2 | 1 / 4 | quality_failed |

Both failed attempts preserve the prior fast hierarchy byte-for-byte. Overall
19/21 pass; the harness exits nonzero. Fast page-10 native English retention is
1.0, but OCR/auto never reach that assertion, so English OCR retention remains
unverified. This is a distinct real-engine gap from the earlier native-text
scoring test. Candidate rejection causes and renderer fidelity require inspection;
neither the root cause nor a threshold change is justified by this result alone.

A separate read-only page-1 candidate diagnostic narrows the immediate rejection:
default/PSM-6/PSM-11 score 0.5738/0.5747/0.5273 (above the CJK 0.28 threshold),
but all carry `fragmented_tokens`, with short-token ratios
0.9943/0.9957/0.9759. PSM-5 instead carries `fragmented_latin` and is rejected.
The whitespace-token heuristic therefore blocks these Chinese-dominant candidates;
whether their actual transcription is acceptable still requires render/text
comparison. A script-aware fragmentation correction needs focused Chinese word-
spacing positives and genuine fragmented-output negatives, not global gate removal.

Extraction counters are separate from optional layout work: the synthetic scan's
auto path performs one extraction render/four OCR calls, then another render/two
OCR calls for structure layout. Stage timers overlap and must not be summed as
independent durations. Language/profile are stubs, not production timing evidence.
The English retention check uses native words as a proxy, not human transcription
accuracy. Conservative auto clipping/translucency routing remains PDF-AUTO-01;
that refinement is not implemented or required to relax this acceptance gate.

Next: inspect actual page-1 render and candidate diagnostics, retain garbage and
language-independent negative controls, then rerun all three modes. No runtime
quality thresholds, routing policy, user artifacts or deployed services were
changed by this acceptance attempt. The local `test_pdf_*.py` suite passes 162
tests; those tests do not override the failed real-engine acceptance.

### 18.5 CJK Fragmentation Correction And Remaining Auto Gate (2026-10-16)

This follow-up supersedes the unresolved fragmentation diagnosis in 18.4, not its
historical measurements. A read-only real-engine page-1 comparison found CJK-only
character-sequence similarity to native text of 0.9265/0.9043/0.9235 for default/
PSM-6/PSM-11, versus 0.0072 for incorrect-orientation PSM-5. Excerpts confirm
legitimate one/two-character Chinese word spacing. These are diagnostic sequence
comparisons, not human-annotated recognition accuracy; simplified/traditional
substitution and other transcription differences remain visible.

`observed-script-v3` uses lexical tokens containing at least one non-CJK alphanumeric
character for the short-token score penalty and fragmentation gate. Pure CJK
(U+3400..U+9FFF) and punctuation-only tokens are excluded, but the original raw
`single_token_ratio` remains diagnostic. Latin fragmentation is checked even when
the overall candidate is CJK-dominant, preventing Chinese text from hiding Latin
garbage. Numerical fragments, symbol-heavy output, invalid encoding, minimum
signal and existing score thresholds remain guarded. No language/profile/LLM
authority or native fallback is introduced. This heuristic does not establish
semantic accuracy of arbitrary Chinese glyph sequences or cover all scripts.

The quality version changes execution/provenance and OCR/auto preparation identity;
fast identity is unchanged. Eight quality tests cover Chinese character/word
spacing and punctuation, bilingual positives, mixed Latin/numeric/symbol/encoding
negatives, language-pack independence and version invalidation. All 165 PDF tests
pass locally. No API service was restarted or rebuilt.

Full isolated matrix using the same runtime, 200 DPI and fixture hash as 18.4:

| Mode | Extraction ms | Total ms | Extraction render/OCR calls | Result |
|---|---:|---:|---|---|
| fast | 174.9 | 452.2 | 0 / 0 | 13 pages, 19,168 chars, warm reuse |
| ocr | 258,692.1 | 258,845.5 | 13 / 52 | 13 pages, 24,991 chars, warm reuse |
| auto | 268,609.7 | 268,615.1 | 13 / 52 | extraction_failed, prior OCR hierarchy preserved |

Fast and OCR both reach page-10 English retention 1.0 (native-word proxy, not
character accuracy). All 18 synthetic/budget cases pass: overall 20/21, nonzero
exit remains correct. Auto reaches the native-coverage gate after recognition;
a separate temporary page-1 PDF with real renderer/OCR confirms the exception
chain `PdfExtractionError -> RawTextOcrFailedError: page=1:native_coverage_unresolved`.
This differs from the earlier page-quality failure. Canonical native text is not
an exact normalized substring of the OCR replacement, so completeness is not
proven. The test expectation and coverage guard have not been relaxed.

PDF-FIX-03 remains unchecked. Its next acceptance work must address/adjudicate this
native-coverage limitation without silently dropping usable source text. Auto's
graphics-routing refinement is still separately owned by PDF-AUTO-01; no scoped
clipping/translucency optimization was implemented here. Storage is disposable
JSON, language/profile are stubs, and measured OCR latency is not full production
preparation latency. The user's original PDF and artifacts remain untouched.

### 18.6 Auto Coverage Fail-Fast And Actionable Failure (2026-10-16)

The coverage rule remains unchanged: accepted OCR must contain the usable native
normalized text, and repeating only native text is insufficient when image content
may be missing. Neither approximate similarity nor simplified/traditional conversion
is a completeness proof. This change reduces wasted recovery work; it does not
make the actual fixture's Auto extraction successful.

The canonical Auto caller supplies a page validator to `_load_pages_with_ocr`.
It runs immediately after each page passes OCR quality, on uniform blank renders,
and before cached OCR results are published. Canonical assembly retains its final
validation boundary. Rejection stops subsequent renders/OCR; the existing outer
failure cleanup clears request-local OCR result/cache state. A retry cannot reuse
partial results. Explicit full OCR and fast do not receive this Auto validator.

The public extraction error remains `extraction_failed`; for the narrowly matched
`page=N:native_coverage_unresolved` internal reason its message now includes the
one-based page, reason, and explicit alternatives: fast for native text or OCR for
a new transcription. It does not include source text, paths or raw engine errors,
does not silently switch modes, and does not authorize partial writes.

Evidence: four new raw-recovery tests cover early termination/retry, uniform blank
conflict, actionable public errors and cached-result validation. All 169 PDF tests
pass. A read-only/no-network Docker run uses the same actual 13-page fixture and
engine image as 18.5, first saving fast into temporary JSON, then requesting Auto.
Auto rejects page 1 after 1 render/4 OCR calls, extraction 21,189.6 ms (total
21,195.0 ms), preserving the fast hierarchy byte-for-byte. The earlier run used
13 renders/52 calls and 268,609.7 ms before rejecting. This is avoided work on a
failure path, not successful preparation throughput or a general speedup claim.

PDF-FIX-03 remains unresolved; the full matrix was not rerun in this change and
its previous 20/21 result is historical. Fast/explicit OCR success is established
in 18.5. The remaining successful Auto acceptance depends on resolving conservative
graphics routing (PDF-AUTO-01) or an explicit acceptance-policy decision, not another
quality-threshold adjustment. No source/user artifacts, UI, API schema or deployed
service were changed. This supporting slice does not add a completed top-level task.

### 18.7 Supported Native Visibility Routing (2026-10-16)

PDF-AUTO-01 is delivered before the final PDF-FIX-03 rerun: the actual source's
successful fast/full-OCR acceptance exposed Auto routing as a prerequisite. This
supersedes the earlier deferred-last order, not the acceptance requirements.

`pdf_native_visibility.py` is a fail-closed proof for a limited text-only subset.
It tracks CTM, rectangular clip intersection, nonstroking alpha, font/size/spacing
and text position through balanced q/Q scopes. Glyph bounds, not baseline origins,
must fit the active clip. Supported paint is fill-mode TrueType text, normal blend
with alpha 0.1..1, and axis-aligned page-covering background rectangles painted
before text. Nested clips restore on Q. No per-document filename exception exists.

FontDescriptor bounds are used conservatively first. When those whole-font bounds
cross a clip, embedded TrueType glyph bounds may refine them using fontTools
4.67.0, pinned in requirements. Only unambiguous MacRoman/WinAnsi mapping, consistent
PDF/font advance widths and non-composite glyphs qualify. Unicode cmap is preferred;
a unique Macintosh Roman cmap supports subset fonts lacking Unicode cmap. Empty
glyph contours are allowed only with a validated mapping/advance. Missing library,
unknown encoding, missing metrics or corrupt font means unknown, never an empty
glyph assumption. Reference: [fontTools TTFont documentation](https://fonttools.readthedocs.io/en/latest/ttLib/ttFont.html).

Limits: 100,000 operations/glyphs, 128 graphics-stack levels, 8 MB decoded font and
4,096 embedded glyphs. Fonts remain page-local; no new persistent cache. Existing
pypdf source/decompression limits still apply before these checks. This is not a
complete PDF renderer or proof of visual contrast/semantic accuracy. Rotated/skewed
glyphs outside the supported axis-aligned transforms, arbitrary paths, text clipping,
stroke/invisible modes, very faint alpha, unknown operators, optional content, masks,
Forms, images and post-text vector painting stay uncertain. Graphics saves inside
text objects are unsupported. No OCR quality/native coverage guard was weakened.

Only when the proof succeeds and the existing inventory confirms zero image draws
does inspection clear its clipping/graphics-state unknown flags and establish zero
image coverage. Other flags remain. Verified page-background paint is also accepted
by the native layout policy, so optional structure layout cannot reintroduce OCR
for this supported case. Exact native text/whitespace and page spans are unchanged.
Inspection cache v2 and OCR policy v2 identify the new evidence. Auto preparation
identity includes visibility policy, OCR policy and fontTools availability/version;
fast and explicit full-OCR settings/identity are unchanged.

Validation: eight new tests cover nested/restored clips, glyph-edge containment,
translucency, hidden/masked/unknown cases, images, spacing, embedded glyph fallback,
Auto-only identity invalidation and exact actual-file native parity. All 177 PDF
tests pass. Docker uses the existing Python/pypdf image plus read-only mounted
fontTools 4.67.0, no network, temporary JSON storage and stub language/profile.
The same 13-page source from section 17 succeeds in Auto while every subprocess
call is poisoned: 692.7 ms extraction, 1376.1 ms total, 19,169 canonical characters,
13 ordered pages, English word retention 1.0 and warm reuse without extraction.
Native source bytes are unchanged; no renderer/OCR or production API was invoked.

These measurements are isolated preparation, not production LLM latency. The
full fast/OCR/Auto matrix remains PDF-FIX-03, the only pending follow-up item.
Deployment requires rebuilding/installing updated requirements; merely restarting
the old Docker image does not install fontTools. No production deployment occurred.

### 18.8 Final Three-Mode Acceptance (2026-10-17)

PDF-FIX-03 closes after the full `scripts/accept_pdf_modes.py` rerun: 21/21
real-engine cases pass. This supersedes the pending matrix status in 18.7 and
the intermediate failures above, without changing acceptance expectations.
Native, scan, mixed/outline, blank, corrupt and over-budget fixtures cover mode
isolation, explicit failures, unchanged persistence on failure, ordered spans and
mode-specific warm reuse. The actual 13-page bilingual watermark source has
SHA256 `87d7685e1b95d8842fc749e15b908b631d4cc5def5afcf7a9a02d7011e9ecc83`;
its bytes remain unchanged.

| Actual-source mode | Extraction ms | Total prepare ms | Render/OCR calls | Characters |
|---|---:|---:|---:|---:|
| fast | 166.8 | 446.0 | 0/0 | 19168 |
| ocr | 269371.7 | 269536.8 | 13/52 | 24991 |
| auto | 765.1 | 1506.5 | 0/0 | 19169 |

All three retain 13 ordered pages, page-10 English word retention 1.0 and warm
reuse without extraction. Retention is a native-word proxy, not manual OCR
accuracy certification. OCR/native character counts need not match. Synthetic
Auto scan/mixed extraction recognizes only the required page; optional layout
may make additional render/OCR calls, reported separately by the harness.

Runtime: Python 3.11.17, pypdf 6.17.0, fontTools 4.67.0, Tesseract 5.5.0,
pdftoppm 25.03.0, eng+chi_sim+chi_tra, default 200 DPI. The container is read-only
and network-disabled, with temporary JSON storage and language/profile stubs.
These are isolated preparation timings, not live API/LLM latency. All 177 PDF
regressions and 18 lazy-preparation/acceptance support tests also pass.

No production deployment or user-artifact writes occurred. Updated requirements
must be installed or the image rebuilt to supply fontTools. Unsupported Auto
graphics remain conservative as specified in 18.7; passing this matrix does not
claim universal PDF coverage. Loader follow-up is 5/5 complete.
